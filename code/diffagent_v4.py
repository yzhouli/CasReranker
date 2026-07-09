# -*- coding: utf-8 -*-
"""
DiffAgent V4 — 最终完整版
- 多卡并行: 自动探测GPU, 每卡一个vLLM, 线程池分发
- 全特性: RAG持久记忆 + 三专家辩论 + 协调器裁决 + 拓扑感知
- 自适应token: N值缩放, 不完整翻倍重试
"""
import sys, pickle, random, json, time, numpy as np, os, subprocess, threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# ═══════════════════════════
# 配置
# ═══════════════════════════
DP = "../Casbench"  # 数据集相对路径
MODEL_PATH = "YOUR_MODEL_PATH"  # 替换为本地模型路径，如 /data1/yz/Qwen3.5_4B
MODEL_NAME = "Qwen3.5_4B"  # 替换为实际的 vLLM served-model-name
N_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MEMORY_PATH = "./diffagent_v4_memory.json"  # RAG 记忆文件路径
SAVE_PATH = f"/home/yz/diffagent_v4_N{N_CAND}.json"
LOG_PATH = f"/home/yz/diffagent_v4_N{N_CAND}.log"

# ═══════════════════════════
# GPU/vLLM 管理
# ═══════════════════════════
def get_free_gpus(min_mem_mb=30000):
    out = subprocess.check_output(
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader", shell=True
    ).decode().strip().split('\n')
    free = []
    for line in out:
        parts = line.split(',')
        if int(parts[1].strip().split()[0]) > min_mem_mb:
            free.append(int(parts[0].strip()))
    return free

def start_vllm(gpu_id, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = (f"source /opt/anaconda3/etc/profile.d/conda.sh && "
           f"conda activate yz_vllm && "
           f"vllm serve {MODEL_PATH} --served-model-name {MODEL_NAME} "
           f"--port {port} --max-model-len 32768 --max-num-seqs 4 "
           f"--reasoning-parser qwen3 --gpu-memory-utilization 0.90")
    return subprocess.Popen(["bash", "-c", cmd], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def wait_vllm(port, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1", timeout=5).models.list()
            return True
        except:
            time.sleep(5)
    return False

# ═══════════════════════════
# 持久记忆 + RAG
# ═══════════════════════════
class PersistentMemory:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {"users": {}}
        if os.path.exists(MEMORY_PATH):
            self.data = json.load(open(MEMORY_PATH))
            if "users" not in self.data:
                self.data["users"] = {}

    def save(self):
        with self.lock:
            json.dump(self.data, open(MEMORY_PATH, "w"), ensure_ascii=False)

    def get_rag(self, uid, cur_bigrams, topk=3, min_sim=0.04):
        uid = str(uid)
        interests = self.data["users"].get(uid, {}).get("interests", [])
        if not interests or not cur_bigrams:
            return []
        scored = []
        for e in interests:
            past = set(e.get("bg", []))
            if not past:
                continue
            sim = len(cur_bigrams & past) / max(len(cur_bigrams | past), 1)
            if sim >= min_sim:
                scored.append((e.get("summary", ""), sim, e.get("rank"), e.get("nd"), e.get("th")))
        scored.sort(key=lambda x: -x[1])
        return scored[:topk]

    def add_interest(self, uid, topic_bigrams, topic_summary, rank_val, nd_val, th_val):
        uid = str(uid)
        with self.lock:
            m = self.data["users"].setdefault(uid, {"n": 0, "interests": []})
            m["n"] += 1
            n = m["n"]
            if "avg_rank" in m:
                m["avg_rank"] = m["avg_rank"] + (rank_val - m["avg_rank"]) / n
            else:
                m["avg_rank"] = float(rank_val)
            m["interests"].append({
                "bg": list(topic_bigrams)[:40],
                "summary": topic_summary[:60],
                "rank": rank_val, "nd": nd_val, "th": th_val
            })
            if len(m["interests"]) > 20:
                m["interests"] = m["interests"][-20:]

# ═══════════════════════════
# 三专家辩论 (单样本)
# ═══════════════════════════
def debate_one(port, topic_text, topic_bigrams, candidates, topo_scores,
               cascade_ctx, users, news, memory):
    client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1", timeout=300, max_retries=1)

    cand_lines = []
    for u in candidates:
        ts = topo_scores.get(u, {})
        nd = ts.get('nd', 0); th = ts.get('th', 0); deg = ts.get('deg', 0)
        nd_b = "none" if nd == 0 else ("low" if nd <= 2 else "high")
        th_b = "none" if th == 0 else ("low" if th <= 5 else ("mid" if th <= 15 else "high"))
        deg_b = "normal" if deg <= 10 else ("active" if deg <= 50 else "influencer")
        bio = str(users.get(u, {}).get("description", ""))[:60].replace("\n", " ")
        uh = [str(h)[:24].replace("\n", " ") for h in (users.get(u, {}).get("history", []) or [])[:3]]
        uh_str = " / ".join(uh) if uh else "none"

        # RAG
        rag = memory.get_rag(u, topic_bigrams)
        rag_str = " | ".join([f"{r[0][:25]}(sim={r[1]:.2f},r#{r[2]})" for r in rag[:2]]) if rag else "none"

        cand_lines.append(
            f"{u}: nd={nd_b} th={th_b} deg={deg_b} kw={ts.get('kw', 0):.2f} act={ts.get('act', 0)} "
            f"bio:{bio} recent:{uh_str} RAG:{rag_str}"
        )

    prompt = f"""Topic: {topic_text[:400]}

Cascade Background: {cascade_ctx}

Candidates (nd=direct connections, th=two-hop neighbors, deg=influence, kw=keyword match, act=activity, bio=profile, recent=recent topics, RAG=similar past interests):
{chr(10).join(cand_lines)}

Three experts debate, then coordinator decides:

Semantic Expert: Analyze the topic content (theme/entity/sentiment). Compare each candidate's kw, bio, and recent topics against the topic. Assess semantic relevance. Perceive topology as context: nd>0 means user is in the cascade path; th>0 means their social neighborhood is activated. Score 0-100 with reasoning for top-5.

Profile Expert: Analyze user profile (bio + activity) and RAG history (similar past topic interests). High RAG similarity + high activity = genuinely interested. Low RAG + high nd/th = in the path but may not care. Score 0-100 with reasoning for top-5, explicitly addressing disagreements with Semantic expert.

Topology Expert: Analyze structural position. nd>0: in direct cascade chain. th high: social neighborhood heavily activated. deg high but nd=th=0: isolated influencer, structurally irrelevant to this cascade. nd/th both high: core structural position. Score 0-100 with reasoning for top-5.

Coordinator: Compare the three expert assessments. Weight evidence: structural proximity (nd/th) is hard evidence; semantic relevance (kw/bio) and profile interest (RAG) are soft evidence. Resolve conflicts. Output JSON: {{"ranked":[...all {len(candidates)} ids in descending order], "explanations":{{top-10:"<=20 words"}}, "debate_log":"key disagreements and resolution <=100 chars"}}"""

    sysmsg = "Multi-agent diffusion ranking. 3 experts debate: Semantic(content), Profile(interest/RAG), Topology(structure). Coordinator resolves. Output JSON only."

    # 自适应 token
    for base_tok in [max(2000, len(candidates) * 8), len(candidates) * 24]:
        try:
            r = client.chat.completions.create(
                model=MODEL_NAME, temperature=0, max_tokens=base_tok,
                messages=[{"role": "system", "content": sysmsg},
                          {"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            txt = r.choices[0].message.content or ""
        except:
            txt = ""
        ranked = []
        try:
            s = txt.find("{"); e = txt.rfind("}") + 1
            if s >= 0 and e > s:
                js = json.loads(txt[s:e])
                ranked = [u for u in js.get("ranked", []) if u in set(candidates)]
        except:
            pass
        if len(ranked) == len(candidates):
            break

    seen = set(ranked)
    ranked += [u for u in candidates if u not in seen]
    return ranked

# ═══════════════════════════
# 主流程
# ═══════════════════════════
def main():
    # ── 启动 vLLM ──
    free_gpus = get_free_gpus(min_mem_mb=30000)
    print(f"Free GPUs: {free_gpus}", flush=True)
    if not free_gpus:
        print("No free GPU!"); return

    ports = []
    procs = []
    for i, gpu in enumerate(free_gpus):
        port = 8100 + i
        print(f"GPU{gpu} -> :{port}", flush=True)
        procs.append(start_vllm(gpu, port))
        ports.append(port)
        time.sleep(5)

    def _wait(p):
        return p if wait_vllm(p) else None
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        ports = [r for r in ex.map(_wait, ports) if r is not None]
    print(f"Ready ports: {ports}", flush=True)
    if not ports:
        print("No vLLM ready!"); return

    # ── 加载数据 ──
    print("Loading data...", flush=True); t0 = time.time()
    users = pickle.load(open(f"{DP}/users_all.pkl", "rb"))
    news = pickle.load(open(f"{DP}/news_all.pkl", "rb"))
    tf = f"{DP}/test_hardneg.pkl" if N_CAND <= 100 else f"{DP}/test_hardneg1000.pkl"
    test = pickle.load(open(tf, "rb"))
    print(f"Loaded {len(test)} items, {time.time() - t0:.0f}s", flush=True)

    def soc(u):
        return users.get(u, {}).get("social", []) or []

    def hist(u):
        return users.get(u, {}).get("history", []) or []

    def bg(s):
        s = str(s); return set(s[i:i + 2] for i in range(len(s) - 1))

    # ── 预计算 ──
    memory = PersistentMemory()
    print("Precomputing topology...", flush=True)
    tasks = []
    for idx in range(len(test)):
        it = test[idx]; true = str(it["next_user"])
        cset = set(str(x) for x in it["history_users"])
        nb = set()
        for c in cset:
            nb.update(soc(c))
        topic_text = str(news.get(it["news_id"], {}).get("text", ""))
        topic_bigrams = bg(topic_text)
        cascade_ctx_parts = []
        for c in list(cset)[:3]:
            cd = str(users.get(c, {}).get("description", ""))[:50].replace("\n", " ")
            if cd.strip():
                cascade_ctx_parts.append(f"{c}:{cd}")
        cascade_ctx = " | ".join(cascade_ctx_parts) if cascade_ctx_parts else "no context"

        negs = [str(u) for u in it["neg_users"][:N_CAND - 1]]
        candidates = negs + [true]
        random.Random(2026 + idx).shuffle(candidates)

        topo_scores = {}
        for u in candidates:
            su = set(soc(u))
            ub = bg(" ".join(str(h) for h in hist(u)[:8]))
            kw = (len(topic_bigrams & ub) / len(topic_bigrams | ub)) if (topic_bigrams and ub) else 0.0
            topo_scores[u] = {"nd": len(su & cset), "th": len(su & nb), "deg": len(su),
                              "kw": round(kw, 4), "act": len(hist(u))}

        tasks.append({
            "idx": idx, "true": true, "topic_text": topic_text,
            "topic_bigrams": topic_bigrams, "cascade_ctx": cascade_ctx,
            "candidates": candidates, "topo_scores": topo_scores
        })

    print(f"Precomputed {len(tasks)} tasks", flush=True)

    # ── 并行推理 ──
    ranks = [0] * len(tasks)
    lock = threading.Lock()
    completed = 0

    def process(task, port):
        nonlocal completed
        ranked = debate_one(port, task["topic_text"], task["topic_bigrams"],
                           task["candidates"], task["topo_scores"],
                           task["cascade_ctx"], users, news, memory)
        rank = ranked.index(task["true"]) + 1
        topic_summary = task["topic_text"][:80].replace("\n", " ")
        for i, u in enumerate(ranked[:20]):
            memory.add_interest(u, task["topic_bigrams"], topic_summary,
                               i + 1, task["topo_scores"][u]["nd"],
                               task["topo_scores"][u]["th"])

        with lock:
            ranks[task["idx"]] = rank
            nonlocal completed; completed += 1
            if completed % 10 == 0:
                json.dump({"ranks": ranks, "completed": completed}, open(SAVE_PATH, "w"))
                memory.save()
            if completed % 50 == 0:
                M = metrics(ranks)
                print(f"  [{completed}/{len(test)}] H@1={M['H@1']:.4f} H@5={M['H@5']:.4f} | {time.time() - t0:.0f}s", flush=True)
        return rank

    print(f"Starting parallel inference with {len(ports)} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        futures = []
        for i, task in enumerate(tasks):
            port = ports[i % len(ports)]
            futures.append(pool.submit(process, task, port))
        for f in as_completed(futures):
            f.result()

    # ── 最终 ──
    memory.save()
    M = metrics(ranks)
    print(f"\n=== DiffAgent V4 N={N_CAND} Final ===", flush=True)
    for k in [1, 2, 5]:
        print(f"  H@{k}={M['H@%d' % k]:.4f}  M@{k}={M['M@%d' % k]:.4f}  N@{k}={M['N@%d' % k]:.4f}")
    print(f"Total: {time.time() - t0:.0f}s", flush=True)
    json.dump({"N": N_CAND, "metrics": M, "total_s": time.time() - t0}, open(SAVE_PATH, "w"))

    for proc in procs:
        proc.kill()

KS = [1, 2, 5]
def metrics(r):
    M = {}
    for k in KS:
        M[f"H@{k}"] = float(np.mean([1.0 if v <= k else 0.0 for v in r]))
        M[f"M@{k}"] = float(np.mean([1.0 / v if v <= k else 0.0 for v in r]))
        M[f"N@{k}"] = float(np.mean([1.0 / np.log2(v + 1) if v <= k else 0.0 for v in r]))
    return M

if __name__ == "__main__":
    main()
