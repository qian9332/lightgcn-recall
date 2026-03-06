# LightGCN Graph Recall Model

基于用户-物品交互构建二部图的 **LightGCN** 图召回模型完整实现。

## 📌 项目概述

本项目完整实现了 [LightGCN](https://arxiv.org/abs/2002.02126) (SIGIR 2020) 用于推荐系统的图召回任务：

| 特性 | 实现 |
|---|---|
| **图结构** | 用户-物品二部图 (Bipartite Graph) |
| **核心设计** | 移除传统GCN的特征变换与非线性激活，仅保留邻居归一化聚合 |
| **图卷积** | 3层堆叠建模三阶连通性 (3-hop connectivity) |
| **层组合** | 各层Embedding加权求和: E_final = Σ α_k · E^(k) |
| **损失函数** | BPR (Bayesian Personalized Ranking) 成对排序优化 |
| **负采样** | 流行度加权采样 P(i) ∝ pop(i)^0.75 + In-Batch Negative |
| **数据集** | Gowalla (30K users, 41K items, 1M+ interactions) |

## 📊 训练结果

在 Gowalla 数据集上的表现：

| Epoch | Loss | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|-------|------|-----------|---------|-----------|---------|
| 5     | 5.189 | 0.0717   | 0.0395  | 0.1184    | 0.0508  |
| 10    | 5.101 | 0.0662   | 0.0362  | 0.1098    | 0.0468  |
| 20    | 4.796 | 0.0725   | 0.0404  | 0.1200    | 0.0519  |
| 30    | 4.554 | 0.0846   | 0.0488  | 0.1369    | 0.0616  |
| 40    | 4.357 | 0.0896   | 0.0516  | 0.1444    | 0.0651  |
| **50** | **4.201** | **0.0933** | **0.0526** | **0.1493** | **0.0665** |

> 训练完整日志见 `logs/training.log`，指标历史见 `logs/training_history.json`

## 📂 项目结构

```
lightgcn-recall/
├── README.md               # 项目说明文档
├── requirements.txt         # Python依赖
├── .gitignore
│
├── preprocess.py            # 数据预处理 (Gowalla原始数据→训练格式)
├── dataset.py               # 数据集加载、二部图邻接矩阵构建、BPR采样
├── model.py                 # LightGCN模型 + BPR Loss + In-Batch Negative
├── evaluate.py              # 评估指标 (Recall@K, NDCG@K)
├── train.py                 # 完整训练脚本 (适合GPU环境)
├── step_train.py            # 单步训练脚本 (适合资源受限环境)
├── train_resume.py          # 断点续训脚本
├── run_train.py             # 快速训练脚本 (pickle缓存加速)
├── run_epoch.py             # 单epoch训练脚本
├── auto_train.sh            # 自动多轮训练Shell脚本
│
├── data/
│   ├── gowalla_checkins.txt    # 原始数据 (gitignore, 需自行下载)
│   └── processed/              # 预处理后数据
│       ├── train.txt           # 训练集 (847,671条)
│       ├── val.txt             # 验证集 (91,320条)
│       ├── test.txt            # 测试集 (91,320条)
│       ├── stats.txt           # 数据集统计
│       └── item_popularity.npy # 物品流行度
│
├── logs/
│   ├── training.log            # 完整训练日志
│   └── training_history.json   # 指标历史 (JSON)
│
└── checkpoints/                # 模型检查点 (训练时生成)
```

## 🏗️ 模型架构

### LightGCN vs 传统 GCN

```
传统GCN:  E^(k+1) = σ(D^{-1/2} A D^{-1/2} · E^(k) · W^(k))
                     ↑ 非线性激活              ↑ 特征变换权重

LightGCN: E^(k+1) = D^{-1/2} A D^{-1/2} · E^(k)
                     仅保留归一化邻居聚合 ✓
```

### 完整计算流程

```python
# 第0层：初始Embedding
E^(0) = [E_user ; E_item]              # shape: (N_users + N_items, dim)

# 第1-3层：图卷积 (无特征变换，无非线性激活！)
E^(1) = Ã · E^(0)                      # 1-hop: 直接邻居
E^(2) = Ã · E^(1)                      # 2-hop: 2阶邻居
E^(3) = Ã · E^(2)                      # 3-hop: 3阶邻居

# 层组合：加权求和
E_final = (1/4)(E^(0) + E^(1) + E^(2) + E^(3))

# 预测
score(u, i) = E_u^T · E_i              # 内积评分

# 其中 Ã = D^{-1/2} · A · D^{-1/2} 为对称归一化邻接矩阵
```

### BPR + In-Batch Negative

```
L = L_BPR + 0.5 * L_InBatch + 1e-4 * L_Reg

L_BPR = -Σ log σ(s(u,i⁺) - s(u,i⁻))   # 成对排序损失
L_InBatch = CrossEntropy(S, labels)       # batch内负样本增强
L_Reg = (1/2)||Θ₀||²                     # 初始Embedding L2正则
```

### 负采样策略

- **流行度加权**: P(i) ∝ popularity(i)^0.75 (平滑长尾)
- **In-Batch Negative**: batch内其他用户的正样本作为额外负样本

## 📊 数据集

### Gowalla

| 属性 | 值 |
|---|---|
| 来源 | [SNAP Stanford](https://snap.stanford.edu/data/loc-Gowalla.html) |
| 类型 | 用户位置签到数据 |
| 原始记录数 | 6,442,892 |
| 去重后 | 3,981,334 |
| 过滤条件 | 用户/物品 ≥ 10次交互 |
| **最终用户数** | **30,038** |
| **最终物品数** | **41,137** |
| **最终交互数** | **1,030,311** |
| 密度 | 0.0834% |
| 训练集 | 847,671 (82.3%) |
| 验证集 | 91,320 (8.9%) |
| 测试集 | 91,320 (8.9%) |

## 🚀 快速开始

### 环境要求

```bash
pip install -r requirements.txt
# PyTorch >= 2.0, NumPy < 2.0, SciPy, Pandas, tqdm
```

### 1. 下载原始数据 & 预处理

```bash
# 下载 Gowalla 数据集
cd data
wget https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
gunzip loc-gowalla_totalCheckins.txt.gz
mv loc-gowalla_totalCheckins.txt gowalla_checkins.txt
cd ..

# 预处理
python preprocess.py
```

### 2. 训练模型

**GPU 训练 (推荐)**:
```bash
python train.py \
    --embedding_dim 64 \
    --n_layers 3 \
    --epochs 80 \
    --batch_size 4096 \
    --lr 1e-3 \
    --reg_weight 1e-4 \
    --use_in_batch_neg \
    --in_batch_neg_weight 0.5 \
    --eval_interval 5 \
    --topk "20,50"
```

**CPU / 资源受限环境**:
```bash
# 先生成缓存加速
python -c "
import pickle
from dataset import GowallaDataset
ds = GowallaDataset('data/processed')
with open('data/processed/dataset_cache.pkl', 'wb') as f: pickle.dump(ds, f)
"

# 分步训练 (每次1个epoch)
for i in $(seq 1 50); do python step_train.py 5; done
```

### 3. 查看结果

```bash
cat logs/training.log
cat logs/training_history.json
```

## ⚙️ 超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `embedding_dim` | 64 | Embedding维度 |
| `n_layers` | 3 | GCN层数 (建模3阶连通性) |
| `layer_combination` | mean | 层组合: mean / learnable |
| `batch_size` | 2048-8192 | 训练batch大小 |
| `lr` | 1e-3 | 学习率 (Adam) |
| `reg_weight` | 1e-4 | L2正则系数 |
| `use_in_batch_neg` | True | 启用In-Batch Negative |
| `in_batch_neg_weight` | 0.5 | In-Batch Neg损失权重 |
| `epochs` | 50-80 | 训练轮次 |
| `eval_interval` | 5 | 每N个epoch评估一次 |

## 📈 评估指标

- **Recall@K**: 推荐列表中命中的正样本占全部正样本的比例
- **NDCG@K**: 考虑位置信息的推荐质量评估，排在前面的命中权重更高

## 📝 参考文献

1. He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126). SIGIR 2020.
2. Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/0905.2515). UAI 2009.

## License

MIT License
