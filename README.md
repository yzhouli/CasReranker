# CasReranker — 面向社交信息级联重排序的多智能体协同架构

> 以大语言模型为认知核心的多智能体系统正逐渐成为下一代智能信息系统的核心范式。然而在社交信息级联重排序应用中现有研究面临两大局限。首先，当前基准主要基于纯ID构建，阻碍了多模态语义推理并引发严重的冷启动问题。其次，级联扩散上下文长且复杂，单一智能体极易陷入信息过载与灾难性遗忘的认知瓶颈。针对首个局限，本文构建了多模态扩散基准 **CasBench**，通过整合源端内容与高维用户表征打破了纯ID范式。为突破认知瓶颈，进一步提出多智能体协同重排序架构 **CasReranker**，该架构采用工作流范式，依次完成传播源深度理解以及动态兴趣与拓扑吸引力的双轨感知，最终由决策智能体汇聚多维异质信号进行综合推理与重排序。同时，结合检索增强的持久记忆机制实现了跨样本的知识复用。实验表明CasReranker显著超越了传统基于ID的模型与单体大模型基线，充分证明了多智能体工作流应对复杂信息系统的卓越效能。

## 运行环境

本仓库代码已在以下环境中验证：

| 组件 | 环境 | Python | 关键依赖 |
|---|---|---|---|
| CasReranker (LLM推理) | `yz_vllm` (conda) | 3.10 | vLLM, openai, numpy, tqdm |
| 单智能体LLM评估 | `yz_vllm` (conda) | 3.10 | openai, numpy, tqdm, pillow, opencv-python |
| GNN基线训练 | `idp` (conda) | 3.12 | PyTorch 2.7.0+cu128, PyTorch Geometric |

**硬件**: 4×NVIDIA A800 80GB GPU

**模型**: Qwen3.5-4B (`/data1/yz/Qwen3.5_4B`)

## 目录结构

```
release/
├── CasReranker/                       # 多智能体协同重排序
│   ├── CasReranker.py                 # 主流水线（多GPU并行、RAG记忆、自适应token）
│   ├── SourcePerception_Agent.py      # 传播源感知智能体
│   ├── DynamicInterest_Agent.py       # 动态兴趣感知智能体
│   ├── TopologyAttraction_Agent.py    # 拓扑吸引感知智能体
│   └── Decision_Agent.py              # 决策智能体
├── SmallModel/                        # GNN 基线模型
│   ├── GraphSAGE.py / GCN.py / GAT.py / GIN.py   # 异构图 GNN
│   ├── HGT.py / HGAT.py / HGCN.py                 # 异构图 Transformer
│   ├── DIN.py / SASRec.py / PMRCA.py              # 序列推荐基线
│   ├── baselines_all_hardneg.py       # 全基线批量训练+评估
│   └── evaluation.py                  # 评估指标计算
├── LLM/                               # 单智能体 LLM 评估
│   ├── mllms.py                       # 通用 LLM 排序
│   └── mllms_qwen3.7_hardneg.py       # 闭源 API 模型评估
├── Casbench/                          # 数据集（从 Kaggle 下载）
├── results/                           # 实验结果输出
└── README.md
```

## 快速开始

### 1. 启动 vLLM 推理服务

```bash
# 激活环境
conda activate yz_vllm

# 每张GPU启动一个vLLM实例（CasReranker自动探测可用GPU）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /data1/yz/Qwen3.5_4B \
    --served-model-name Qwen3.5_4B \
    --port 8300 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 &

# 单GPU测试模式（端口8400）
python -m vllm.entrypoints.openai.api_server \
    --model /data1/yz/Qwen3.5_4B \
    --served-model-name Qwen3.5_4B \
    --port 8400 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 &
```

### 2. 运行 CasReranker 推理

```bash
cd CasReranker
conda activate yz_vllm

# N=20（消融实验默认）
python CasReranker.py 20

# N=50（主实验）
python CasReranker.py 50

# N=100 / N=500 / N=1000
python CasReranker.py 100
```

运行前确认 `CasReranker.py` 中的配置：
- `DP = "../Casbench"` — 数据集路径
- `MODEL_NAME = "Qwen3.5_4B"` — vLLM served-model-name
- `MEMORY_PATH = "./CasReranker_memory.json"` — RAG持久记忆文件
- 自动探测端口 8300-8303 上已启动的 vLLM 实例

### 3. 运行单智能体 LLM 评估

```bash
cd LLM
conda activate yz_vllm

# 开放API模型（需先配置 api_key 和 base_url）
python mllms.py

# 闭源API模型（Qwen3.7-plus / GLM-5.2 / GPT-5.4）
python mllms_qwen3.7_hardneg.py
```

### 4. 运行 GNN 基线训练+评估

```bash
cd SmallModel
conda activate idp

# 训练并评估全部10个GNN模型
python baselines_all_hardneg.py

# 单独运行
python GCN.py
python GraphSAGE.py
python PMRCA.py
```

## CasReranker 架构

```
级联话题 + 候选用户
        │
        ▼
┌───────────────────┐
│  传播源感知智能体   │ ← 多模态内容语义匹配
│  (SourcePerception)│
└───────┬───────────┘
        │
        ▼
┌───────────────────┐    ┌───────────────────┐
│ 动态兴趣感知智能体  │    │ 拓扑吸引感知智能体   │
│ (DynamicInterest)  │    │ (TopologyAttraction)│
│ 画像 + RAG记忆     │    │ 社交图 nd/th/deg    │
└───────┬───────────┘    └───────┬───────────┘
        │                        │
        └────────┬───────────────┘
                 ▼
        ┌───────────────┐
        │   决策智能体    │ ← 分层加权 + 冲突消解
        │   (Decision)   │
        └───────┬───────┘
                ▼
        排序结果 + 辩论日志
```

## 消融实验

在 `CasReranker/CasReranker.py` 中修改配置：

| 变体 | 方法 |
|---|---|
| 完整模型 | 默认配置 |
| no_sem | 跳过 SourcePerceptionAgent 调用 |
| no_prof | 跳过 DynamicInterestAgent 调用 |
| no_topo | 跳过 TopologyAttractionAgent 特征注入 |
| no_rag | 设置 RAG 开关为 False |
| no_filter | 设置 RAG 相似度阈值为 0 |
| amnesia | 每次推理清空跨样本记忆 |

## 候选集规模

| N | 命令 | 测试文件 |
|---|------|---------|
| 20 | `CasReranker/CasReranker.py 20` | test_hardneg.pkl |
| 50 | `CasReranker/CasReranker.py 50` | test_hardneg1000.pkl |
| 100 | `CasReranker/CasReranker.py 100` | test_hardneg1000.pkl |
| 500 | `CasReranker/CasReranker.py 500` | test_hardneg1000.pkl |
| 1000 | `CasReranker/CasReranker.py 1000` | test_hardneg1000.pkl |

## 评估指标

- **Hits@K**: 真实目标用户出现在排序前K位的比例
- **MAP@K**: 前K位的平均精度
- **NDCG@K**: 归一化折损累计增益

均在 `SmallModel/evaluation.py` 中实现，支持 K=1,2,5,10,20,50。

## 验证状态

| 组件 | 状态 | 说明 |
|---|---|---|
| CasReranker 2样本推理 | ✅ | 语义+拓扑+协调器全链路通过 |
| 单智能体 4B 2样本推理 | ✅ | vLLM API 调用正常 |
| GNN ×10 1-epoch训练 | ✅ | GCN/GraphSAGE/GAT/GIN/HGT/HGAT/HGCN/SASRec/DIN/PMRCA 全部通过 |
| 数据加载 | ✅ | 8文件 7.8GB 全部校验 |
| API Key 清除 | ✅ | 无硬编码密钥残留 |

## 论文引用

本仓库对应论文《CasReranker：面向智能体信息系统的多专家协同级联重排序方法》（软件学报专刊投稿）。

## 数据集

**CasBench** 已上传至 Kaggle：[https://www.kaggle.com/datasets/yangzhou32/casbench](https://www.kaggle.com/datasets/yangzhou32/casbench)

若本地 `Casbench/` 目录为空，请从 Kaggle 下载数据文件并放入该目录。详细说明见 `Casbench/README.md`。

## License

学术研究用途。数据集版权归原始作者所有。
