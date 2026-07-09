import os
import json
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
# 核心修改：引入 GATConv
from torch_geometric.nn import GATConv, to_hetero
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
# 3. 静态图构建 (防止数据泄露)
# ==========================================
def build_hetero_graph(file_path, exclude_news_set=None):
    if exclude_news_set is None:
        exclude_news_set = set()

    user_to_idx, news_to_idx = {}, {}
    edges_user, edges_news = [], []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            parts = line.split(' ', 1)
            if len(parts) < 2: continue

            news_raw = parts[0].strip()
            if news_raw in exclude_news_set:
                continue

            if news_raw not in news_to_idx:
                news_to_idx[news_raw] = len(news_to_idx)
            news_idx = news_to_idx[news_raw]

            user_time_pairs = parts[1].split(',')
            for pair in user_time_pairs:
                u_raw, _ = pair.strip().split(' ')
                if u_raw not in user_to_idx:
                    user_to_idx[u_raw] = len(user_to_idx)
                user_idx = user_to_idx[u_raw]

                edges_user.append(user_idx)
                edges_news.append(news_idx)

    data = HeteroData()
    data['user'].num_nodes = len(user_to_idx)
    data['news'].num_nodes = len(news_to_idx)

    edge_index = torch.tensor([edges_user, edges_news], dtype=torch.long)
    data['user', 'interacts', 'news'].edge_index = edge_index
    data['news', 'interacts_by', 'user'].edge_index = edge_index.flip([0])

    return data, user_to_idx, news_to_idx


# ==========================================
# 4. GAT 基线模型与训练
# ==========================================
class BaseGNN(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # 修复：使用 GATConv。设置 heads=4（多头注意力），concat=False（保持输出维度为 hidden_channels 不变），
        # 并严格设置 add_self_loops=False 以适配二分图。
        self.conv1 = GATConv(-1, hidden_channels, heads=4, concat=False, add_self_loops=False)
        self.conv2 = GATConv(-1, hidden_channels, heads=4, concat=False, add_self_loops=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        # GAT 论文标准做法：使用 ELU 激活函数替代 ReLU
        x = F.elu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class DiffusionReRanker(nn.Module):
    def __init__(self, num_users, num_news, hidden_channels, metadata):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, hidden_channels)
        self.news_emb = nn.Embedding(num_news, hidden_channels)
        self.gnn = to_hetero(BaseGNN(hidden_channels), metadata=metadata, aggr='mean')

    def forward(self, data):
        x_dict = {'user': self.user_emb.weight, 'news': self.news_emb.weight}
        return self.gnn(x_dict, data.edge_index_dict)


def train_model(model, data, optimizer, epochs=50):
    model.train()
    edge_index = data['user', 'interacts', 'news'].edge_index
    num_news = data['news'].num_nodes

    pqbr = tqdm(total=epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_dict = model(data)
        user_embs = out_dict['user']
        news_embs = out_dict['news']

        users_pos = edge_index[0]
        news_pos = edge_index[1]

        # 随机采样负样本边进行 BPR 训练
        news_neg = torch.randint(0, num_news, (users_pos.size(0),), device=users_pos.device)

        pos_scores = (user_embs[users_pos] * news_embs[news_pos]).sum(dim=1)
        neg_scores = (user_embs[users_pos] * news_embs[news_neg]).sum(dim=1)

        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        loss.backward()
        optimizer.step()

        pqbr.desc = f"BPR Loss: {loss.item():.4f}"
        pqbr.update(1)


# ==========================================
# 5. 基于 test.pkl 的重排推理 (归纳式冷启动)
# ==========================================
def evaluate_reranking_with_pkl(model, data, dataset_path, user_map, news_map):
    test_db = load_pkl(f'{dataset_path}/test.pkl')

    model.eval()
    target_ids = []
    pred_lists = []

    with torch.no_grad():
        out_dict = model(data)
        user_embs = out_dict['user']
        news_embs = out_dict['news']

        pdbr = tqdm(test_db, desc='GAT Eval')
        for item in pdbr:
            news_id = str(item["news_id"]).strip()
            next_uid = str(item["next_user"]).strip()
            history_users = [str(u).strip() for u in item.get("history_users", [])]

            candidate_users = [str(u).strip() for u in item["neg_users"][:19]]
            candidate_users.append(next_uid)
            random.shuffle(candidate_users)

            # 通过聚合该 news 的历史传播者 (history_users) 的特征，实时生成未知 news 的 Embedding
            valid_hist_embs = []
            for u in history_users:
                if u in user_map:
                    valid_hist_embs.append(user_embs[user_map[u]])

            if len(valid_hist_embs) > 0:
                target_news_emb = torch.stack(valid_hist_embs).mean(dim=0)
            else:
                target_news_emb = torch.zeros(user_embs.size(1), device=user_embs.device)

            scores = []
            for u_raw in candidate_users:
                if u_raw in user_map:
                    u_idx = user_map[u_raw]
                    score = torch.dot(user_embs[u_idx], target_news_emb).item()
                else:
                    score = float('-inf')
                scores.append((u_raw, score))

            ranked_candidates = [u_raw for u_raw, _ in sorted(scores, key=lambda x: x[1], reverse=True)]

            target_ids.append(next_uid)
            pred_lists.append(ranked_candidates)

    return evaluate_metrics(target_ids, pred_lists, ks=[1, 2, 5])


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
    print(f"   共锁定 {len(test_news_set)} 个测试新闻，将在训练图中强制剔除以防泄露。")

    print("\n2. 加载级联数据并构建全局图结构...")
    data, user_map, news_map = build_hetero_graph(cascades_file, exclude_news_set=test_news_set)
    data = data.to(device)
    print(f"   Graph Info: {data['user'].num_nodes} users, {data['news'].num_nodes} news.")

    print("\n3. 初始化并训练 GAT 模型 (BPR Loss)...")
    hidden_dim = 64
    model = DiffusionReRanker(data['user'].num_nodes, data['news'].num_nodes, hidden_dim, data.metadata()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(model, data, optimizer, epochs=3000)

    print("\n4. 启动基于 test.pkl 的重排推理评测 (归纳式验证)...")
    metrics = evaluate_reranking_with_pkl(model, data, dataset_path, user_map, news_map)

    print("\n========== Evaluation Results ==========")
    line = ''
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        line += f" & {value:.4f}"

    print()
    print(line)