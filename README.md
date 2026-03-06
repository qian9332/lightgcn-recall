# LightGCN Graph Recall Model

基于用户-物品交互构建二部图的LightGCN图召回模型实现。

## 📌 项目概述

本项目完整实现了 LightGCN (Light Graph Convolution Network) 用于推荐系统的图召回任务：

- **图结构**: 用户-物品二部图 (Bipartite Graph)
- **模型核心**: 移除传统GCN的特征变换与非线性激活，仅保留邻居归一化聚合与层组合
- **图卷积**: 3层堆叠建模三阶连通性 (3-hop connectivity)
- **层组合**: 各层Embedding加权求和得到最终表示
- **损失函数**: BPR (Bayesian Personalized Ranking) 优化成对排序目标
- **负采样**: 流行度加权采样 (Popularity-weighted) + In-Batch Negative

## 📂 项目结构

```
lightgcn-recall/
├── README.md              # 项目说明
├── requirements.txt       # 依赖
├── preprocess.py          # 数据预处理脚本
├── dataset.py             # 数据集加载 & BPR采样
├── model.py               # LightGCN模型 + BPR Loss
├── evaluate.py            # 评估指标 (Recall@K, NDCG@K)
├── train.py               # 训练主流程
├── data/                  # 数据目录
│   ├── gowalla_checkins.txt  # 原始Gowalla数据
│   └── processed/         # 预处理后的数据
├── logs/                  # 训练日志
├── checkpoints/           # 模型检查点
└── configs/               # 配置文件
```

## 📊 数据集: Gowalla

| 指标 | 值 |
|---|---|
| 数据来源 | [SNAP Stanford](https://snap.stanford.edu/data/loc-Gowalla.html) |
| 原始签到数 | 6,442,892 |
| 过滤条件 | 用户/物品最少10次交互 |
| 过滤后用户数 | ~29,000+ |
| 过滤后物品数 | ~40,000+ |
| 训练/验证/测试 | 8:1:1 |

## 🏗️ 模型架构

### LightGCN 核心设计

```
E^(0) = [E_user ; E_item]            # 初始Embedding拼接
E^(k+1) = D^{-1/2} A D^{-1/2} E^(k) # 图卷积：无特征变换，无非线性激活
E_final = Σ_{k=0}^{K} α_k * E^(k)    # 层组合：加权求和
score(u,i) = E_u^T * E_i             # 预测得分：内积
```

### 关键特性

1. **简化GCN**: 去除权重矩阵W和激活函数σ，避免过拟合
2. **多阶连通性**: 3层卷积捕获1-hop, 2-hop, 3-hop邻居信息
3. **层组合**: 融合不同阶数的结构信息
4. **BPR + In-Batch Negative**: 高效的成对排序优化

## 🚀 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 1. 数据预处理

```bash
python preprocess.py
```

### 2. 模型训练

```bash
# 默认配置 (CPU)
python train.py

# GPU训练
python train.py --batch_size 4096

# 自定义配置
python train.py \
    --embedding_dim 64 \
    --n_layers 3 \
    --epochs 80 \
    --batch_size 2048 \
    --lr 1e-3 \
    --reg_weight 1e-4 \
    --use_in_batch_neg \
    --in_batch_neg_weight 0.5
```

### 3. 查看训练日志

```bash
cat logs/train_*.log
```

## ⚙️ 超参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `embedding_dim` | 64 | Embedding维度 |
| `n_layers` | 3 | GCN层数 (建模k阶连通性) |
| `layer_combination` | mean | 层组合: mean / learnable |
| `batch_size` | 2048 | 训练batch大小 |
| `lr` | 1e-3 | 学习率 |
| `reg_weight` | 1e-4 | L2正则系数 |
| `use_in_batch_neg` | True | 启用In-Batch Negative |
| `in_batch_neg_weight` | 0.5 | In-Batch Neg损失权重 |
| `epochs` | 80 | 训练轮次 |
| `patience` | 15 | Early stopping耐心值 |

## 📈 评估指标

- **Recall@K**: 召回率，衡量推荐列表覆盖用户真实兴趣的能力
- **NDCG@K**: 归一化折扣累积增益，衡量推荐列表的排序质量

## 📝 参考文献

- [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126) (SIGIR 2020)
- [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/0905.2515) (UAI 2009)

## License

MIT License
