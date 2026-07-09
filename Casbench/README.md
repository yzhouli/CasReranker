# CasBench

多模态信息扩散重排序基准数据集。

## 下载

数据集托管于 Kaggle：

👉 **[https://www.kaggle.com/datasets/yangzhou32/casbench](https://www.kaggle.com/datasets/yangzhou32/casbench)**

## 文件说明

| 文件 | 大小 | 说明 |
|---|---|---|
| `users_all.pkl` | 7.8 GB | 835,845 用户画像（简介、历史行为、社交关系） |
| `cascades.txt` | 33 MB | 全部信息级联序列 |
| `train_cascades.txt` | 29 MB | 训练集级联 |
| `val_cascades.txt` | 4.0 MB | 验证集级联 |
| `edges.txt` | 30 MB | 社交关系边 |
| `news_all.pkl` | 1.5 MB | 话题内容（文本/多模态路径） |
| `test.pkl` | 12 MB | 原始测试集 |
| `test_hardneg.pkl` | 1.9 MB | 难负样本测试集（N=20/50/100） |
| `test_hardneg1000.pkl` | 13 MB | 大候选集测试（N=500/1000） |
| `user2id.pkl` | 15 MB | 用户 ID 映射 |
| `news2id.pkl` | 148 KB | 话题 ID 映射 |

## 数据格式

### cascades.txt / train_cascades.txt / val_cascades.txt
```
news_id user_1,user_2,user_3,...
```

### users_all.pkl
Python dict: `{user_id: {"description": str, "social": list, "history": list}}`

### test.pkl / test_hardneg.pkl
Python list of dicts: `[{"news_id": str, "history_users": list, "next_user": str, "neg_users": list}]`
```
