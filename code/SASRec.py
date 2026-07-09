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
            # 同样防数据泄露：过滤测试集中的级联
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

            # 如果序列长度超过 max_len，进行截断；保留最近的 max_len 个用户
            if len(cascade_seq) > 1:
                sequences.append(cascade_seq)

    return sequences, user_to_idx


# ==========================================
# 4. SASRec 模型构建
# ==========================================
class DiffusionSASRec(nn.Module):
    def __init__(self, num_users, hidden_dim, max_seq_len=50, num_heads=2, num_blocks=2, dropout_rate=0.2):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        # 词嵌入: num_users 包含了 Padding(0) 索引
        self.user_emb = nn.Embedding(num_users, hidden_dim, padding_idx=0)
        # 位置嵌入
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)

    def forward(self, user_seqs):
        # user_seqs: [batch_size, seq_len]
        batch_size, seq_len = user_seqs.size()

        # 生成位置索引
        positions = torch.arange(seq_len, device=user_seqs.device).unsqueeze(0).expand(batch_size, seq_len)

        # Padding 掩码 (告诉注意力机制忽略 padding 为 0 的位置)
        padding_mask = (user_seqs == 0)

        # 因果掩码 (严格保证 t 时刻只能看到 1 到 t 的信息)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=user_seqs.device)

        # 融合 ID 与 位置 Embedding
        x = self.user_emb(user_seqs) + self.pos_emb(positions)

        # 序列表示抽取
        # output: [batch_size, seq_len, hidden_dim]
        output = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        return output

    def get_user_embedding(self, user_indices):
        return self.user_emb(user_indices)


def get_train_batches(sequences, batch_size, max_len):
    """序列切片与 Pad 打包生成器"""
    random.shuffle(sequences)
    x_batch, pos_batch = [], []

    for seq in sequences:
        # 输入序列为 [0: N-1], 目标序列为 [1: N]
        seq = seq[-max_len - 1:]  # 截取尾部保证长度
        x = seq[:-1]
        pos_y = seq[1:]

        # 补 0 (Padding) 到同一长度
        pad_len = max_len - len(x)
        x_pad = x + [0] * pad_len
        pos_pad = pos_y + [0] * pad_len

        x_batch.append(x_pad)
        pos_batch.append(pos_pad)

        if len(x_batch) == batch_size:
            yield torch.tensor(x_batch, dtype=torch.long), torch.tensor(pos_batch, dtype=torch.long)
            x_batch, pos_batch = [], []

    if len(x_batch) > 0:
        yield torch.tensor(x_batch, dtype=torch.long), torch.tensor(pos_batch, dtype=torch.long)


def train_model(model, sequences, num_users, optimizer, max_len=50, batch_size=256, epochs=50, dataset_path=None, user_map=None):
    model.train()
    device = next(model.parameters()).device

    pqbr = tqdm(total=epochs)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        total_loss = 0.0
        batches = get_train_batches(sequences, batch_size, max_len)

        for x, pos_y in batches:
            x, pos_y = x.to(device), pos_y.to(device)
            optimizer.zero_grad()

            # 随机采样负样本
            neg_y = torch.randint(1, num_users, pos_y.size(), device=device)

            # 获取 Transformer 序列输出
            # seq_out: [batch_size, seq_len, hidden_dim]
            seq_out = model(x)

            # 获取目标候选人向量
            pos_emb = model.get_user_embedding(pos_y)
            neg_emb = model.get_user_embedding(neg_y)

            # 点积打分
            pos_scores = (seq_out * pos_emb).sum(dim=-1)
            neg_scores = (seq_out * neg_emb).sum(dim=-1)

            # BPR Loss (只在有效位上计算，忽略 padding 位)
            mask = (pos_y != 0).float()
            loss = -F.logsigmoid(pos_scores - neg_scores) * mask
            loss = loss.sum() / mask.sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        pqbr.desc = f"BPR Loss: {total_loss:.4f}"
        pqbr.update(1)
        # if epoch % 50 == 0:
        #     show_metrics(model, dataset_path=dataset_path, user_map=user_map, max_sequence_length=max_len)
    time_span = time.perf_counter() - start_time
    return time_span


# ==========================================
# 5. 基于 test.pkl 的重排推理
# ==========================================
def evaluate_reranking_with_pkl(model, dataset_path, user_map, base_time, max_len=50):
    test_db = load_pkl(f'{dataset_path}/test_999.pkl')

    model.eval()
    device = next(model.parameters()).device
    target_ids = []
    pred_lists = []

    with torch.no_grad():
        start_time = time.perf_counter()
        for item in test_db:
            next_uid_raw = str(item["next_user"]).strip()
            history_users_raw = [str(u).strip() for u in item.get("history_users", [])]

            candidate_users_raw = [str(u).strip() for u in item["neg_users"][:9]]
            candidate_users_raw.append(next_uid_raw)
            random.shuffle(candidate_users_raw)

            # 1. 转换历史序列
            hist_seq = []
            for u in history_users_raw:
                if u in user_map:
                    hist_seq.append(user_map[u])

            # 处理极端冷启动情况（如果没有有效历史，给一个[PAD]）
            if len(hist_seq) == 0:
                hist_seq = [0]

            # 截断与转 Tensor
            hist_seq = hist_seq[-max_len:]
            x_tensor = torch.tensor([hist_seq], dtype=torch.long, device=device)

            # 2. 通过 Transformer 拿到序列向量
            # seq_out: [1, seq_len, hidden_dim]
            seq_out = model(x_tensor)

            # 取最后一步的输出作为整个历史上下文的最终意图表征
            final_intent_emb = seq_out[0, -1, :]

            # 3. 对 20 个候选人打分
            scores = []
            for u_raw in candidate_users_raw:
                if u_raw in user_map:
                    u_idx = user_map[u_raw]
                    u_emb = model.get_user_embedding(torch.tensor(u_idx, device=device))
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

def show_metrics(model, dataset_path, user_map, max_sequence_length, base_time):
    metrics = evaluate_reranking_with_pkl(model, dataset_path, user_map, max_len=max_sequence_length, base_time=base_time)
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

    print("\n3. 初始化并训练 SASRec 模型 (Causal Transformer + BPR)...")
    hidden_dim = 64
    model = DiffusionSASRec(num_users=num_users, hidden_dim=hidden_dim, max_seq_len=max_sequence_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 序列模型训练收敛快，通常不需要像 GNN 跑上千 Epoch。
    time_span = train_model(model, sequences, num_users, optimizer, max_len=max_sequence_length, batch_size=256, epochs=1, dataset_path=dataset_path, user_map=user_map)

    show_metrics(model, dataset_path, user_map, max_sequence_length, base_time=time_span)
