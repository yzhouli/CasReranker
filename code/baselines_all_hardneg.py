# -*- coding: utf-8 -*-
"""所有小模型在 test_hardneg 上训练+推理 N=20/50/100/500"""
import sys, os, random, pickle, time, traceback, json
sys.path.insert(0, "c:/Prcharm_Code/DiffAgent/Baseline")
import numpy as np
import torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)

DP = "../Casbench"  # 数据集相对路径
SAVE = "../results/baselines_all_results.json"  # 结果保存路径
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {dev}", flush=True)

N_VALUES = [20, 50, 100, 500]
TEST_PKL = "../Casbench/test_hardneg.pkl"
CASCADES = f"{DP}/cascades.txt"
USERS_ALL = f"{DP}/users_all.pkl"
NEWS_ALL = f"{DP}/news_all.pkl"

def L(p):
    with open(p, "rb") as f: return pickle.load(f)

def save_json(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=2))

def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

def metrics(targets, preds, ks=(1, 2, 5)):
    R = {f"H@{k}": [] for k in ks}
    R.update({f"N@{k}": [] for k in ks})
    R.update({f"M@{k}": [] for k in ks})
    for t, p in zip(targets, preds):
        try: r = p.index(t) + 1
        except ValueError: r = float("inf")
        for k in ks:
            R[f"H@{k}"].append(1 if r <= k else 0)
            R[f"M@{k}"].append(1.0 / r if r <= k else 0)
            R[f"N@{k}"].append(1.0 / np.log2(r + 1) if r <= k else 0)
    return {m: round(float(np.mean(v)), 4) for m, v in R.items()}

def bg(s):
    s = str(s)
    return set(s[i:i+2] for i in range(len(s)-1))

# ── 加载数据 ──
print("Loading data...", flush=True); t0 = time.time()
test_full = L(TEST_PKL)
test_news_set = {str(it["news_id"]).strip() for it in test_full}
users_all = L(USERS_ALL)
news_all = L(NEWS_ALL)
print(f"Loaded {len(test_full)} test items, {len(users_all)} users, {time.time()-t0:.0f}s", flush=True)

# ── 构建候选 (每个N一组) ──
print("Building candidate sets...", flush=True)
candidates_by_n = {n: [] for n in N_VALUES}
targets = []
for idx, it in enumerate(test_full):
    true = str(it["next_user"]).strip()
    targets.append(true)
    for n in N_VALUES:
        c = [str(u).strip() for u in it["neg_users"][:n-1]] + [true]
        random.Random(1000 + n * 10000 + idx).shuffle(c)
        candidates_by_n[n].append(c)

# ── Null baselines (不依赖训练) ──
def eval_null(name, rank_fn):
    results = {}
    for n in N_VALUES:
        preds = [rank_fn(c, true, it) for c, true, it in
                 zip(candidates_by_n[n], targets, test_full)]
        results[f"N={n}"] = metrics(targets, preds)
    return results

print("\n=== Null Baselines ===", flush=True)
t0 = time.time()
all_results = {}

# 加载已有结果
if os.path.exists(SAVE):
    all_results = load_json(SAVE)

# Random
if "Random" not in all_results:
    all_results["Random"] = eval_null("Random", lambda c, t, it: random.sample(c, len(c)))
    save_json(SAVE, all_results)
    print(f"Random done", flush=True)

# Degree
if "Degree" not in all_results:
    all_results["Degree"] = eval_null("Degree",
        lambda c, t, it: sorted(c, key=lambda u: -len(users_all.get(u, {}).get("social", []) or [])))
    save_json(SAVE, all_results)
    print(f"Degree done", flush=True)

# Keyword
if "Keyword" not in all_results:
    def kw_rank(c, true, it):
        tb = bg(str(news_all.get(it["news_id"], {}).get("text", "")))
        scores = {}
        for u in c:
            uh = users_all.get(u, {}).get("history", []) or []
            ub = bg(" ".join(str(h) for h in uh[:8]))
            scores[u] = (len(tb & ub) / len(tb | ub)) if (tb and ub) else 0.0
        return sorted(c, key=lambda u: -scores.get(u, 0))
    all_results["Keyword"] = eval_null("Keyword", kw_rank)
    save_json(SAVE, all_results)
    print(f"Keyword done ({time.time()-t0:.0f}s)", flush=True)

# ── torch.fx monkey-patch for PyTorch 2.5+ compatibility ──
import torch.fx._symbolic_trace
if not hasattr(torch.fx._symbolic_trace, 'List'):
    torch.fx._symbolic_trace.List = list
# HGAT/HGCN monkey-patch: PyG 2.6 需要 HeteroData.numel 且其返回值支持 len()
from torch_geometric.data import HeteroData
import torch as _torch
if not hasattr(HeteroData, 'numel'):
    def _hetero_numel(self):
        total = sum(v.numel() for v in self.values() if hasattr(v, 'numel'))
        return _torch.tensor([total])  # tensor 支持 len()，也支持 int 转换
    HeteroData.numel = _hetero_numel

# ── 训练类模型 ──
# 三类异构图模型 API 不同
HETERO_STD   = ["GraphSAGE", "GCN", "GAT", "GIN"]     # metadata in ctor, simple train
HETERO_HGT   = ["HGT"]                                  # metadata in ctor, extended train
HETERO_SIMP  = ["HGAT", "HGCN"]                        # no metadata, extended train

for model_name in ["SASRec", "PMRCA"] + HETERO_STD + HETERO_HGT + HETERO_SIMP + ["DIN"]:
    if model_name in all_results:
        print(f"\n=== {model_name} SKIP (already done) ===", flush=True)
        continue
    print(f"\n=== {model_name} ===", flush=True)
    t0 = time.time()
    try:
        mod = __import__(model_name)

        # ── SASRec ──
        if model_name == "SASRec":
            seqs, umap = mod.build_sequence_data(CASCADES, exclude_news_set=test_news_set, max_len=50)
            nU = len(umap)
            m = mod.DiffusionSASRec(nU, 64, max_seq_len=50).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            mod.train_model(m, seqs, nU, opt, max_len=50, batch_size=256, epochs=20)
            m.eval()
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                with torch.no_grad():
                    for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                        hs = [umap.get(str(u), 0) for u in test_full[idx]["history_users"] if str(u) in umap] or [0]
                        x = torch.tensor([hs[-50:]], dtype=torch.long, device=dev)
                        intent = m(x)[0, -1, :]
                        sc = []
                        for u in c:
                            u_str = str(u)
                            if u_str in umap:
                                emb = m.get_user_embedding(torch.tensor(umap[u_str], device=dev))
                                sc.append((u, torch.dot(emb, intent).item()))
                            else:
                                sc.append((u, float("-inf")))
                        preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                        if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        # ── PMRCA ──
        elif model_name == "PMRCA":
            ei, seqs, umap, nmap = mod.build_graph_and_sequence_data(CASCADES, test_news_set, max_len=50)
            nU, nN = len(umap), len(nmap)
            adj, adjt = mod.build_normalized_sparse_adj(ei, nU, nN, dev)
            m = mod.FullPMRCAReranker(nU, nN, 64, max_seq_len=50).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            mod.train_model(m, seqs, adj, adjt, nU, opt, max_len=50, batch_size=256, epochs=80)
            m.eval()
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                with torch.no_grad():
                    for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                        hist = [str(u) for u in test_full[idx]["history_users"]]
                        hs = [umap.get(u, 0) for u in hist if u in umap] or [0]
                        hs = hs[-50:]; hs = hs + [0] * (50 - len(hs))
                        x = torch.tensor([hs], dtype=torch.long, device=dev)
                        so, _ = m(x, adj, adjt)
                        vl = int((x != 0).sum().item())
                        intent = so[0, max(vl - 1, 0), :]
                        sc = []
                        for u in c:
                            u_str = str(u)
                            if u_str in umap:
                                emb = m.get_user_embedding(torch.tensor(umap[u_str], device=dev))
                                sc.append((u, torch.dot(emb, intent).item()))
                            else:
                                sc.append((u, float("-inf")))
                        preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                        if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        # ── 异构图标准型 (GraphSAGE, GCN, GAT, GIN) ──
        elif model_name in HETERO_STD:
            data, umap, nmap = mod.build_hetero_graph(CASCADES, exclude_news_set=test_news_set)
            data = data.to(dev)
            m = mod.DiffusionReRanker(data['user'].num_nodes, data['news'].num_nodes, 64, data.metadata()).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-2)
            mod.train_model(m, data, opt, epochs=1500)
            m.eval()
            with torch.no_grad():
                out = m(data); UE = out['user']
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                    hist = [str(u) for u in test_full[idx]["history_users"]]
                    he = [UE[umap[u]] for u in hist if u in umap]
                    ne = torch.stack(he).mean(0) if he else torch.zeros(UE.size(1), device=dev)
                    sc = []
                    for u in c:
                        u_str = str(u)
                        if u_str in umap:
                            sc.append((u, torch.dot(UE[umap[u_str]], ne).item()))
                        else:
                            sc.append((u, float("-inf")))
                    preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                    if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        # ── HGT (metadata + extended train) ──
        elif model_name in HETERO_HGT:
            data, umap, nmap = mod.build_hetero_graph(CASCADES, exclude_news_set=test_news_set)
            data = data.to(dev)
            m = mod.DiffusionReRanker(data['user'].num_nodes, data['news'].num_nodes, 64, data.metadata()).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-2)
            mod.train_model(m, data, opt, dataset_path=DP, user_map=umap, news_map=nmap, epochs=1500)
            m.eval()
            with torch.no_grad():
                out = m(data); UE = out['user']
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                    hist = [str(u) for u in test_full[idx]["history_users"]]
                    he = [UE[umap[u]] for u in hist if u in umap]
                    ne = torch.stack(he).mean(0) if he else torch.zeros(UE.size(1), device=dev)
                    sc = []
                    for u in c:
                        u_str = str(u)
                        if u_str in umap:
                            sc.append((u, torch.dot(UE[umap[u_str]], ne).item()))
                        else:
                            sc.append((u, float("-inf")))
                    preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                    if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        # ── HGAT, HGCN (no metadata + extended train, reduced epochs) ──
        elif model_name in HETERO_SIMP:
            data, umap, nmap = mod.build_hetero_graph(CASCADES, exclude_news_set=test_news_set)
            data = data.to(dev)
            m = mod.DiffusionReRanker(data['user'].num_nodes, data['news'].num_nodes, 64).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-2)
            mod.train_model(m, data, opt, dataset_path=DP, user_map=umap, news_map=nmap, epochs=300)
            m.eval()
            with torch.no_grad():
                out = m(data); UE = out['user']
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                    hist = [str(u) for u in test_full[idx]["history_users"]]
                    he = [UE[umap[u]] for u in hist if u in umap]
                    ne = torch.stack(he).mean(0) if he else torch.zeros(UE.size(1), device=dev)
                    sc = []
                    for u in c:
                        u_str = str(u)
                        if u_str in umap:
                            sc.append((u, torch.dot(UE[umap[u_str]], ne).item()))
                        else:
                            sc.append((u, float("-inf")))
                    preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                    if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        # ── DIN (序列模型, 不同forward API) ──
        elif model_name == "DIN":
            seqs, umap = mod.build_sequence_data(CASCADES, exclude_news_set=test_news_set, max_len=50)
            nU = len(umap)
            m = mod.DiffusionDIN(nU, 64).to(dev)
            opt = torch.optim.Adam(m.parameters(), 1e-3)
            mod.train_model(m, seqs, nU, opt, max_len=50, batch_size=256, epochs=80, dataset_path=DP, user_map=umap)
            m.eval()
            model_results = {}
            for n in N_VALUES:
                preds = []; oov = 0
                with torch.no_grad():
                    for idx, (c, true) in enumerate(zip(candidates_by_n[n], targets)):
                        hist = [str(u) for u in test_full[idx]["history_users"]]
                        hs = [umap.get(u, 0) for u in hist if u in umap] or [0]
                        hs_tensor = torch.tensor([hs[-50:]], dtype=torch.long, device=dev)
                        sc = []
                        for u in c:
                            u_str = str(u)
                            if u_str in umap:
                                cu = torch.tensor([umap[u_str]], dtype=torch.long, device=dev)
                                score = m(hs_tensor, cu).item()
                            else:
                                score = float("-inf")
                            sc.append((u, score))
                        preds.append([u for u, _ in sorted(sc, key=lambda z: z[1], reverse=True)])
                        if str(true) not in umap: oov += 1
                model_results[f"N={n}"] = metrics(targets, preds)
                model_results[f"N={n}"]["_oov"] = oov
                print(f"  N={n}: H@1={model_results[f'N={n}']['H@1']:.4f} H@5={model_results[f'N={n}']['H@5']:.4f} OOV={oov}", flush=True)
            all_results[model_name] = model_results
            save_json(SAVE, all_results)

        print(f"  Total: {time.time()-t0:.0f}s", flush=True)

    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        traceback.print_exc()
        all_results[model_name] = {"error": str(e)}
        save_json(SAVE, all_results)

# ── 汇总 ──
print("\n=== 汇总 ===", flush=True)
for name in sorted(all_results.keys()):
    M = all_results[name]
    if isinstance(M, dict) and "N=20" in M:
        n20 = M["N=20"]
        print(f"{name:15s}: H@1={n20.get('H@1',0):.4f} H@5={n20.get('H@5',0):.4f}")

print(f"\nSaved to {SAVE}", flush=True)
