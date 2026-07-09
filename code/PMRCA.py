import os
import json
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 基础数据读取工具 & 评价指标
# ==========================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_metrics(target_ids, pred_lists, ks=[1, 2, 5]):
    results = {f'HITS@{k}': [] for k in ks}
    results.update({f'MAP@{k}': [] for k in ks})
    results.update({f'NDCG@{k}': [] for k in ks})

    for target, pred in zip(target_ids, pred_lists):
        try:
            rank = pred.index(target) + 1
        except ValueError:
            rank = float('inf')

        for k in ks:
            if rank <= k:
                results[f'HITS@{k}'].append(1)
                results[f'MAP@{k}'].append(1.0 / rank)
                results[f'NDCG@{k}'].append(1.0 / np.log2(rank + 1))
            else:
                results[f'HITS@{k}'].append(0)
                results[f'MAP@{k}'].append(0)
                results[f'NDCG@{k}'].append(0)

    return {metric: np.mean(values) for metric, values in results.items()}


# ==========================================
# 2. 联合数据构建 (全局图 + 局部序列)
# ==========================================
def build_graph_and_sequence_data(file_path, exclude_news_set=None, max_len=50):
    if exclude_news_set is None: exclude_news_set = set()

    user_to_idx = {'<PAD>': 0}
    news_to_idx = {}
    edges_user, edges_news = [], []
    sequences = []  # 保存 (cascade_seq, news_idx)

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(' ', 1)
            if len(parts) < 2: continue

            news_raw = parts[0].strip()
            if news_raw in exclude_news_set: continue

            if news_raw not in news_to_idx:
                news_to_idx[news_raw] = len(news_to_idx)
            news_idx = news_to_idx[news_raw]

            user_time_pairs = parts[1].split(',')
            cascade_seq = []

            for pair in user_time_pairs:
                u_raw, _ = pair.strip().split(' ')
                if u_raw not in user_to_idx:
                    user_to_idx[u_raw] = len(user_to_idx)
                u_idx = user_to_idx[u_raw]

                cascade_seq.append(u_idx)
                edges_user.append(u_idx)
                edges_news.append(news_idx)

            if len(cascade_seq) > 1:
                sequences.append((cascade_seq, news_idx))

    edge_index = torch.tensor([edges_user, edges_news], dtype=torch.long)
    return edge_index, sequences, user_to_idx, news_to_idx


def build_normalized_sparse_adj(edge_index, num_users, num_news, device):
    user_idx, news_idx = edge_index[0], edge_index[1]

    user_deg = torch.bincount(user_idx, minlength=num_users).float()
    news_deg = torch.bincount(news_idx, minlength=num_news).float()

    user_deg_inv = 1.0 / user_deg.clamp(min=1e-6)
    news_deg_inv = 1.0 / news_deg.clamp(min=1e-6)

    e2n_vals = user_deg_inv[user_idx]
    adj_sparse = torch.sparse_coo_tensor(indices=edge_index, values=e2n_vals, size=(num_users, num_news)).to(
        device).coalesce()

    n2e_vals = news_deg_inv[news_idx]
    adj_t_sparse = torch.sparse_coo_tensor(indices=torch.stack([news_idx, user_idx]), values=n2e_vals,
                                           size=(num_news, num_users)).to(device).coalesce()

    return adj_sparse, adj_t_sparse


# ==========================================
# 3. 完全版 PMRCA 重排序模型
# ==========================================
class FullPMRCAReranker(nn.Module):
    def __init__(self, num_users, num_news, hidden_dim, max_seq_len=50, gcn_layers=2, n_heads=4):
        super().__init__()
        self.num_users = num_users
        self.cas_num = num_news
        self.hidden_dim = hidden_dim
        self.gcn_layers = gcn_layers

        # --- 初始权重粗调 ---
        self.ssl_temp = 0.1
        self.ssl_reg = 1e-0  # 降低图对比初始权重，防止原始值过大
        self.alpha = 1.0
        self.proto_reg = 1e-0  # 拔高聚类对比初始权重，防止0梯度

        self.num_clusters = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.user_centroids = None
        self.user_2cluster = None
        self.cas_centroids = None
        self.cas_2cluster = None

        self.user_emb = nn.Embedding(num_users, hidden_dim, padding_idx=0)
        self.cas_emb = nn.Embedding(num_news, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        self.attn_size = hidden_dim // 2
        self.W1 = nn.Linear(hidden_dim, self.attn_size)
        self.W2 = nn.Linear(self.attn_size, 1)

        self.align_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.cas_emb.weight)

    def before_epoch(self):
        self.e_step()

    def e_step(self):
        user_embeddings = self.user_emb.weight.detach().cpu().numpy()
        cas_embeddings = self.cas_emb.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings)
        self.cas_centroids, self.cas_2cluster = self.run_kmeans(cas_embeddings)

    def run_kmeans(self, x):
        import faiss
        kmeans = faiss.Kmeans(d=self.hidden_dim, k=self.num_clusters, gpu=True if torch.cuda.is_available() else False)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)

        centroids = torch.Tensor(cluster_cents).to(self.device)
        centroids = F.normalize(centroids, p=2, dim=1)
        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def ssl_layer_loss(self, current_embedding, previous_embedding, cas_idx, user_idx):
        current_cas_embeddings, current_user_embeddings = torch.split(current_embedding, [self.cas_num, self.num_users])
        previous_cas_embeddings_all, previous_user_embeddings_all = torch.split(previous_embedding,
                                                                                [self.cas_num, self.num_users])

        # User SSL
        current_user_embeddings = current_user_embeddings[user_idx]
        previous_user_embeddings = previous_user_embeddings_all[user_idx]
        norm_user_emb1 = F.normalize(current_user_embeddings)
        norm_user_emb2 = F.normalize(previous_user_embeddings)
        norm_all_user_emb = F.normalize(previous_user_embeddings_all)

        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # Item (Cascade) SSL
        current_item_embeddings = current_cas_embeddings[cas_idx]
        previous_item_embeddings = previous_cas_embeddings_all[cas_idx]
        norm_item_emb1 = F.normalize(current_item_embeddings)
        norm_item_emb2 = F.normalize(previous_item_embeddings)
        norm_all_item_emb = F.normalize(previous_cas_embeddings_all)

        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss

    def aware_loss(self, node_embedding, cas_idx, user_idx):
        if self.user_centroids is None: return torch.tensor(0.0).to(self.device)

        cas_embeddings_all, tuser_embeddings_all = torch.split(node_embedding, [self.cas_num, self.num_users])

        # User Prototype Loss
        user_embeddings = tuser_embeddings_all[user_idx]
        norm_user_embeddings = F.normalize(user_embeddings)
        user2cluster = self.user_2cluster[user_idx]
        user2centroids = self.user_centroids[user2cluster]

        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # Cascade Prototype Loss
        item_embeddings = cas_embeddings_all[cas_idx]
        norm_item_embeddings = F.normalize(item_embeddings)
        item2cluster = self.cas_2cluster[cas_idx]
        item2centroids = self.cas_centroids[item2cluster]

        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.cas_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss

    def forward(self, input_seq, adj_sparse, adj_t_sparse):
        ego_u = self.user_emb.weight
        ego_n = self.cas_emb.weight

        embedding_list = [torch.cat([ego_n, ego_u], dim=0)]

        for _ in range(self.gcn_layers):
            next_u = torch.sparse.mm(adj_sparse, ego_n)
            next_n = torch.sparse.mm(adj_t_sparse, ego_u)
            ego_u, ego_n = next_u, next_n
            embedding_list.append(torch.cat([ego_n, ego_u], dim=0))

        gcn_u_all = torch.mean(torch.stack([e[self.cas_num:] for e in embedding_list], dim=1), dim=1)

        batch_size, seq_len = input_seq.size()
        positions = torch.arange(seq_len, device=input_seq.device).unsqueeze(0).expand(batch_size, seq_len)
        padding_mask = (input_seq == 0)

        original_seq_emb = self.user_emb(input_seq)
        dyemb = gcn_u_all[input_seq] + self.pos_emb(positions)

        attn_score = self.W2(torch.tanh(self.W1(original_seq_emb)))
        attn_score = attn_score.masked_fill(padding_mask.unsqueeze(-1), float('-inf'))
        attn_score = F.softmax(attn_score, dim=1)

        intention_vectors = torch.sum(original_seq_emb * attn_score, dim=1, keepdim=True)
        intention_expanded = intention_vectors.expand(-1, seq_len, -1)

        att_out, _ = self.align_attention(query=dyemb, key=intention_expanded, value=intention_expanded,
                                          key_padding_mask=padding_mask)

        final_seq_repr = att_out + dyemb

        return final_seq_repr, embedding_list

    def get_user_embedding(self, user_indices):
        return self.user_emb(user_indices)


# ==========================================
# 4. 训练逻辑 (带 Warm-up & Clamp)
# ==========================================
def get_train_batches(sequences, batch_size, max_len):
    random.shuffle(sequences)
    x_batch, target_batch, news_batch = [], [], []

    for seq, news_idx in sequences:
        seq = seq[-max_len - 1:]
        hist = seq[:-1]
        target = seq[-1]

        pad_len = max_len - len(hist)
        hist_pad = hist + [0] * pad_len

        x_batch.append(hist_pad)
        target_batch.append(target)
        news_batch.append(news_idx)

        if len(x_batch) == batch_size:
            yield (torch.tensor(x_batch, dtype=torch.long),
                   torch.tensor(target_batch, dtype=torch.long),
                   torch.tensor(news_batch, dtype=torch.long))
            x_batch, target_batch, news_batch = [], [], []

    if len(x_batch) > 0:
        yield (torch.tensor(x_batch, dtype=torch.long),
               torch.tensor(target_batch, dtype=torch.long),
               torch.tensor(news_batch, dtype=torch.long))


def train_model(model, sequences, adj_sparse, adj_t_sparse, num_users, optimizer, max_len=50, batch_size=256,
                epochs=500, dataset_path=None, user_map=None):
    device = next(model.parameters()).device

    # ================= 动态截断策略配置 =================
    warmup_epochs = 50  # 预热期: 前 50 Epoch 纯净 BPR
    max_aux_ratio = 0.20  # 辅助损失上限: 绝对不超过 BPR 的 20%
    # ====================================================

    pqbr = tqdm(total=epochs)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        # 预热期内不运行 K-Means 聚类以节省时间并防止错乱
        if epoch >= warmup_epochs:
            model.before_epoch()

        model.train()

        total_bpr, total_ssl, total_proto = 0.0, 0.0, 0.0
        batches = get_train_batches(sequences, batch_size, max_len)

        for x, pos_y, news_idx in batches:
            x, pos_y, news_idx = x.to(device), pos_y.to(device), news_idx.to(device)
            optimizer.zero_grad()

            neg_y = torch.randint(1, num_users, pos_y.size(), device=device)

            seq_out, embedding_list = model(x, adj_sparse, adj_t_sparse)

            valid_lens = (x != 0).sum(dim=1) - 1
            final_intent = seq_out[torch.arange(x.size(0)), valid_lens]

            pos_emb = model.get_user_embedding(pos_y)
            neg_emb = model.get_user_embedding(neg_y)

            # --- 1. 主任务 Loss ---
            pos_scores = (final_intent * pos_emb).sum(dim=-1)
            neg_scores = (final_intent * neg_emb).sum(dim=-1)
            loss_bpr = -F.logsigmoid(pos_scores - neg_scores).mean()

            loss = loss_bpr
            loss_ssl_val, loss_proto_val = 0.0, 0.0

            # --- 2. 辅助任务 Loss (带截断) ---
            if epoch >= warmup_epochs:
                center_embedding = embedding_list[0]
                context_embedding = embedding_list[1]

                raw_loss_ssl = model.ssl_layer_loss(context_embedding, center_embedding, news_idx, pos_y)
                raw_loss_proto = model.aware_loss(center_embedding, news_idx, pos_y)

                # 动态锚定当前 Batch 的 BPR 绝对值
                bpr_magnitude = loss_bpr.detach()
                max_allowed_aux = bpr_magnitude * max_aux_ratio

                # 核心：使用 torch.clamp 强制截断天花板
                loss_ssl = torch.clamp(raw_loss_ssl, max=max_allowed_aux)
                loss_proto = torch.clamp(raw_loss_proto, max=max_allowed_aux)

                loss = loss_bpr + loss_ssl + loss_proto

                loss_ssl_val = loss_ssl.item()
                loss_proto_val = loss_proto.item()

            loss.backward()
            optimizer.step()

            total_bpr += loss_bpr.item()
            total_ssl += loss_ssl_val
            total_proto += loss_proto_val

        # 进度条展示区分预热期和正式期
        if epoch < warmup_epochs:
            pqbr.desc = f"[Warm-up] BPR: {total_bpr:.2f}"
        else:
            pqbr.desc = f"BPR: {total_bpr:.2f} | SSL(Clamp): {total_ssl:.2f} | Proto(Clamp): {total_proto:.2f}"
        pqbr.update(1)

        # if (epoch + 1) % 50 == 0 and dataset_path is not None:
        #     print(f"\n--- Epoch {epoch + 1} 性能评估 ---")
        #     metrics = evaluate_reranking_with_pkl(model, adj_sparse, adj_t_sparse, dataset_path, user_map, max_len)
        #     line = ''
        #     for metric, value in metrics.items():
        #         print(f"{metric}: {value:.4f}")
        #         line += f" & {value:.4f}"
        #     print("LaTeX 行格式: " + line)
    time_span = time.perf_counter() - start_time
    return time_span


# ==========================================
# 5. 基于 test.pkl 的推理评估
# ==========================================
def evaluate_reranking_with_pkl(model, adj_sparse, adj_t_sparse, dataset_path, user_map, base_time, max_len=50):
    test_db = load_pkl(f'{dataset_path}/test.pkl')
    model.eval()
    device = next(model.parameters()).device
    target_ids, pred_lists = [], []

    with torch.no_grad():
        start_time = time.perf_counter()
        for item in test_db:
            next_uid_raw = str(item["next_user"]).strip()
            history_users_raw = [str(u).strip() for u in item.get("history_users", [])]
            candidate_users_raw = [str(u).strip() for u in item["neg_users"][:999]] + [next_uid_raw]
            random.shuffle(candidate_users_raw)

            hist_seq = [user_map[u] for u in history_users_raw if u in user_map]
            if len(hist_seq) == 0: hist_seq = [0]
            hist_seq = hist_seq[-max_len:]
            hist_seq = hist_seq + [0] * (max_len - len(hist_seq))
            x_tensor = torch.tensor([hist_seq], dtype=torch.long, device=device)

            seq_out, _ = model(x_tensor, adj_sparse, adj_t_sparse)
            valid_len = (x_tensor != 0).sum(dim=1).item()
            final_intent_emb = seq_out[0, valid_len - 1, :]

            scores = []
            for u_raw in candidate_users_raw:
                if u_raw in user_map:
                    u_emb = model.get_user_embedding(torch.tensor(user_map[u_raw], device=device))
                    score = torch.dot(u_emb, final_intent_emb).item()
                else:
                    score = float('-inf')
                scores.append((u_raw, score))

            ranked_candidates = [u_raw for u_raw, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
            target_ids.append(next_uid_raw)
            pred_lists.append(ranked_candidates)
        time_span = time.perf_counter() - start_time
        print('time_span:', round((time_span + base_time) / len(test_db), 4))

    return evaluate_metrics(target_ids, pred_lists, ks=[1, 2, 5])


# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == '__main__':
    dataset_path = '../Casbench'
    cascades_file = os.path.join(dataset_path, 'cascades.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("1. 解析测试集隔离...")
    test_db = load_pkl(os.path.join(dataset_path, 'test.pkl'))
    test_news_set = {str(item["news_id"]).strip() for item in test_db}

    print("\n2. 构建全局图与局部序列联合数据...")
    max_sequence_length = 100
    edge_index, sequences, user_map, news_map = build_graph_and_sequence_data(cascades_file, test_news_set,
                                                                              max_len=max_sequence_length)
    num_users, num_news = len(user_map), len(news_map)
    print(f"   Info: {num_users} users, {num_news} news, {len(sequences)} valid cascades.")

    print("\n3. 预计算归一化稀疏矩阵 (极速 GCN 传播)...")
    adj_sparse, adj_t_sparse = build_normalized_sparse_adj(edge_index, num_users, num_news, device)

    print("\n4. 初始化动态截断版 PMRCA 重排序模型...")
    hidden_dim = 64
    model = FullPMRCAReranker(num_users, num_news, hidden_dim, max_seq_len=max_sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    time_span = train_model(model, sequences, adj_sparse, adj_t_sparse, num_users, optimizer,
                max_len=max_sequence_length, batch_size=256, epochs=1,
                dataset_path=dataset_path, user_map=user_map)

    print("\n========== PMRCA Final Evaluation ==========")
    evaluate_reranking_with_pkl(model, adj_sparse, adj_t_sparse, dataset_path, user_map, max_len=max_sequence_length, base_time=time_span)