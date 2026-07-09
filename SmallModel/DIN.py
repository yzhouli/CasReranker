import os
import json
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 基础数据读取工具
# ==========================================
def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


def save_json(path, content):
    with open(path, "w", encoding='utf-8') as f:
        f.write(json.dumps(content, ensure_ascii=False))


# ==========================================
# 2. 评价指标
# ==========================================
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

    final_metrics = {metric: np.mean(values) for metric, values in results.items()}
    return final_metrics


# ==========================================
# 3. 序列数据构建 (替代静态图)
# ==========================================
def build_sequence_data(file_path, exclude_news_set=None, max_len=50):
    if exclude_news_set is None:
        exclude_news_set = set()

    # 索引 0 保留为 Padding (填充字符)
    user_to_idx = {'<PAD>': 0}
    sequences = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            parts = line.split(' ', 1)
            if len(parts) < 2: continue

            news_raw = parts[0].strip()
            if news_raw in exclude_news_set:
                continue

            user_time_pairs = parts[1].split(',')
            cascade_seq = []

            # 严格按照时间先后提取序列
            for pair in user_time_pairs:
                u_raw, _ = pair.strip().split(' ')
                if u_raw not in user_to_idx:
                    user_to_idx[u_raw] = len(user_to_idx)
                cascade_seq.append(user_to_idx[u_raw])

            if len(cascade_seq) > 1:
                sequences.append(cascade_seq)

    return sequences, user_to_idx


# ==========================================
# 4. DIN (深度兴趣网络) 模型构建
# ==========================================
class LocalActivationUnit(nn.Module):
    """目标级注意力机制 (Target Attention)"""

    def __init__(self, hidden_dim):
        super().__init__()
        # DIN 经典的外积拼接形式
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query, keys):
        # query: 候选人 [B, 1, D]
        # keys: 历史序列 [B, T, D]
        T = keys.size(1)
        query_expanded = query.expand(-1, T, -1)

        # 拼接：候选、历史、差异、乘积
        concat_features = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys
        ], dim=-1)

        # 计算注意力权重 [B, T, 1]
        attention_weights = self.fc(concat_features)
        return attention_weights


class DiffusionDIN(nn.Module):
    def __init__(self, num_users, hidden_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_dim, padding_idx=0)
        self.attention = LocalActivationUnit(hidden_dim)

        # 顶层多层感知机 (输出标量得分)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, history_seqs, candidate_users):
        # history_seqs: [B, T], candidate_users: [B]

        # 提取 Padding 掩码
        mask = (history_seqs != 0).float().unsqueeze(-1)  # [B, T, 1]

        hist_emb = self.user_emb(history_seqs)  # [B, T, D]
        cand_emb = self.user_emb(candidate_users).unsqueeze(1)  # [B, 1, D]

        # 计算注意力并 Mask 掉 padding 位置
        attn_weights = self.attention(cand_emb, hist_emb)  # [B, T, 1]
        attn_weights = attn_weights * mask

        # 核心：DIN 不使用 Softmax，以保留兴趣的绝对强度
        user_interest = torch.sum(attn_weights * hist_emb, dim=1)  # [B, D]

        cand_emb_squeeze = cand_emb.squeeze(1)  # [B, D]

        # 最终特征组合
        concat_features = torch.cat([
            user_interest,
            cand_emb_squeeze,
            user_interest * cand_emb_squeeze
        ], dim=-1)

        score = self.mlp(concat_features).squeeze(-1)  # [B]
        return score


def get_train_batches(sequences, batch_size, max_len):
    """提取每条序列的最后时刻作为 Target，其余作为 History"""
    random.shuffle(sequences)
    hist_batch, target_batch = [], []

    for seq in sequences:
        # 截取最后 (max_len + 1) 长度
        seq = seq[-max_len - 1:]
        hist = seq[:-1]
        target = seq[-1]

        # 补 0 (Padding) 到同一长度
        pad_len = max_len - len(hist)
        hist_pad = hist + [0] * pad_len

        hist_batch.append(hist_pad)
        target_batch.append(target)

        if len(hist_batch) == batch_size:
            yield torch.tensor(hist_batch, dtype=torch.long), torch.tensor(target_batch, dtype=torch.long)
            hist_batch, target_batch = [], []

    if len(hist_batch) > 0:
        yield torch.tensor(hist_batch, dtype=torch.long), torch.tensor(target_batch, dtype=torch.long)


def train_model(model, sequences, num_users, optimizer, max_len=50, batch_size=256, epochs=50, dataset_path=None,
                user_map=None):
    model.train()
    device = next(model.parameters()).device

    pqbr = tqdm(total=epochs)
    for epoch in range(epochs):
        total_loss = 0.0
        batches = get_train_batches(sequences, batch_size, max_len)

        for hist_seqs, pos_users in batches:
            hist_seqs, pos_users = hist_seqs.to(device), pos_users.to(device)
            optimizer.zero_grad()

            # 随机采样负样本
            neg_users = torch.randint(1, num_users, pos_users.size(), device=device)

            # DIN 分别对正负候选打分
            pos_scores = model(hist_seqs, pos_users)
            neg_scores = model(hist_seqs, neg_users)

            # BPR Loss 优化
            loss = -F.logsigmoid(pos_scores - neg_scores).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pqbr.desc = f"BPR Loss: {total_loss:.4f}"
        pqbr.update(1)

        if epoch > 0 and epoch % 50 == 0:
            show_metrics(model, dataset_path=dataset_path, user_map=user_map, max_sequence_length=max_len)
            model.train()  # 评估完切回训练模式


# ==========================================
# 5. 基于 test.pkl 的重排推理
# ==========================================
def evaluate_reranking_with_pkl(model, dataset_path, user_map, max_len=50):
    test_db = load_pkl(f'{dataset_path}/test.pkl')

    model.eval()
    device = next(model.parameters()).device
    target_ids = []
    pred_lists = []

    with torch.no_grad():
        for item in test_db:
            next_uid_raw = str(item["next_user"]).strip()
            history_users_raw = [str(u).strip() for u in item.get("history_users", [])]

            candidate_users_raw = [str(u).strip() for u in item["neg_users"][:19]]
            candidate_users_raw.append(next_uid_raw)
            random.shuffle(candidate_users_raw)

            # 1. 转换历史序列
            hist_seq = []
            for u in history_users_raw:
                if u in user_map:
                    hist_seq.append(user_map[u])

            # 处理极端冷启动情况
            if len(hist_seq) == 0:
                hist_seq = [0]

            # 截断与 Padding
            hist_seq = hist_seq[-max_len:]
            pad_len = max_len - len(hist_seq)
            hist_seq = hist_seq + [0] * pad_len

            # [1, max_len]
            hist_tensor = torch.tensor([hist_seq], dtype=torch.long, device=device)
            # 扩展出 20 份历史，以匹配 20 个候选人，利用 GPU 并行推理
            hist_tensor_expanded = hist_tensor.expand(len(candidate_users_raw), -1)

            # 2. 准备候选人序列
            cand_seq = []
            for u_raw in candidate_users_raw:
                if u_raw in user_map:
                    cand_seq.append(user_map[u_raw])
                else:
                    cand_seq.append(0)  # OOV 用户
            cand_tensor = torch.tensor(cand_seq, dtype=torch.long, device=device)

            # 3. DIN 并行打分 (一次通过 20 个样本)
            scores = model(hist_tensor_expanded, cand_tensor).cpu().numpy()

            # 4. 排序
            score_tuples = [(u_raw, score) for u_raw, score in zip(candidate_users_raw, scores)]
            ranked_candidates = [u_raw for u_raw, _ in sorted(score_tuples, key=lambda x: x[1], reverse=True)]

            target_ids.append(next_uid_raw)
            pred_lists.append(ranked_candidates)

    return evaluate_metrics(target_ids, pred_lists, ks=[1, 2, 5])


def show_metrics(model, dataset_path, user_map, max_sequence_length):
    metrics = evaluate_reranking_with_pkl(model, dataset_path, user_map, max_len=max_sequence_length)
    print("\n========== Evaluation Results ==========")
    line = ''
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        line += f" & {value:.4f}"

    print()
    print(line)


# ==========================================
# 6. 主程序入口
# ==========================================
if __name__ == '__main__':
    dataset_path = '../Casbench'
    cascades_file = os.path.join(dataset_path, 'cascades.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("1. 正在解析需要隔离的测试集新闻节点...")
    test_db = load_pkl(os.path.join(dataset_path, 'test.pkl'))
    test_news_set = {str(item["news_id"]).strip() for item in test_db}

    print("\n2. 加载级联序列并构建有序数据 (Sequence Data)...")
    max_sequence_length = 1024
    sequences, user_map = build_sequence_data(cascades_file, exclude_news_set=test_news_set,
                                              max_len=max_sequence_length)
    num_users = len(user_map)
    print(f"   Seq Info: {num_users} users (incl. PAD), {len(sequences)} valid cascades.")

    print("\n3. 初始化并训练 DIN 模型 (Local Target Attention + BPR)...")
    hidden_dim = 64
    model = DiffusionDIN(num_users=num_users, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, sequences, num_users, optimizer, max_len=max_sequence_length, batch_size=256, epochs=2000,
                dataset_path=dataset_path, user_map=user_map)

    show_metrics(model, dataset_path, user_map, max_sequence_length)