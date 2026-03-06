"""
数据集加载 & BPR训练采样器
- 构建用户-物品二部图的稀疏邻接矩阵
- 流行度加权负采样 + In-Batch Negative
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


class GowallaDataset:
    """Gowalla数据集管理器"""

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir

        # 读取统计信息
        stats = {}
        with open(os.path.join(data_dir, "stats.txt")) as f:
            for line in f:
                k, v = line.strip().split("\t")
                stats[k] = int(v) if k != "density" else float(v)

        self.n_users = stats["n_users"]
        self.n_items = stats["n_items"]
        self.n_nodes = self.n_users + self.n_items

        # 读取交互数据
        self.train_df = pd.read_csv(os.path.join(data_dir, "train.txt"),
                                     sep="\t", header=None, names=["user", "item"])
        self.val_df = pd.read_csv(os.path.join(data_dir, "val.txt"),
                                   sep="\t", header=None, names=["user", "item"])
        self.test_df = pd.read_csv(os.path.join(data_dir, "test.txt"),
                                    sep="\t", header=None, names=["user", "item"])

        # 构建训练集正样本集合(用于负采样过滤)
        self.train_user_items = {}
        for _, row in self.train_df.iterrows():
            uid, iid = int(row["user"]), int(row["item"])
            self.train_user_items.setdefault(uid, set()).add(iid)

        # 构建测试/验证ground truth
        self.val_user_items = {}
        for _, row in self.val_df.iterrows():
            uid, iid = int(row["user"]), int(row["item"])
            self.val_user_items.setdefault(uid, set()).add(iid)

        self.test_user_items = {}
        for _, row in self.test_df.iterrows():
            uid, iid = int(row["user"]), int(row["item"])
            self.test_user_items.setdefault(uid, set()).add(iid)

        # 物品流行度(用于负采样)
        self.item_popularity = np.load(os.path.join(data_dir, "item_popularity.npy"))
        self._build_neg_sample_prob()

        # 构建稀疏邻接矩阵
        self.adj_matrix = self._build_adj_matrix()

        print(f"📊 Dataset loaded: {self.n_users:,} users, {self.n_items:,} items, "
              f"{len(self.train_df):,} train interactions")

    def _build_neg_sample_prob(self):
        """构建流行度加权的负采样概率分布 P(i) ∝ popularity(i)^0.75"""
        pop = self.item_popularity.astype(np.float64)
        pop = np.power(pop + 1, 0.75)  # +1 smoothing
        self.neg_sample_prob = pop / pop.sum()

    def _build_adj_matrix(self) -> torch.sparse.FloatTensor:
        """
        构建归一化的用户-物品二部图邻接矩阵 (LightGCN)
        A = D^{-1/2} * R * D^{-1/2}
        其中 R 是对称的 (n_users+n_items) x (n_users+n_items) 邻接矩阵
        """
        users = self.train_df["user"].values
        items = self.train_df["item"].values

        # 构建对称邻接矩阵: user->item 和 item->user
        row = np.concatenate([users, items + self.n_users])
        col = np.concatenate([items + self.n_users, users])
        data = np.ones(len(row), dtype=np.float32)
        adj = sp.coo_matrix((data, (row, col)),
                            shape=(self.n_nodes, self.n_nodes))

        # D^{-1/2} A D^{-1/2} 归一化
        degree = np.array(adj.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(degree + 1e-12, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = D_inv_sqrt @ adj @ D_inv_sqrt
        norm_adj = norm_adj.tocoo()

        # 转为 PyTorch sparse tensor
        indices = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))
        values = torch.FloatTensor(norm_adj.data)
        shape = torch.Size(norm_adj.shape)
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    def get_adj_matrix(self, device: torch.device) -> torch.sparse.FloatTensor:
        return self.adj_matrix.to(device)


class BPRTrainDataset(Dataset):
    """BPR训练数据集: (user, pos_item, neg_item)"""

    def __init__(self, dataset: GowallaDataset):
        self.dataset = dataset
        self.users = dataset.train_df["user"].values
        self.pos_items = dataset.train_df["item"].values
        self.n_items = dataset.n_items
        self.train_user_items = dataset.train_user_items
        self.neg_sample_prob = dataset.neg_sample_prob

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.pos_items[idx]

        # 流行度加权负采样
        while True:
            neg_item = np.random.choice(self.n_items, p=self.neg_sample_prob)
            if neg_item not in self.train_user_items.get(user, set()):
                break

        return (torch.LongTensor([user]),
                torch.LongTensor([pos_item]),
                torch.LongTensor([neg_item]))


def get_train_loader(dataset: GowallaDataset, batch_size: int = 2048,
                     num_workers: int = 4) -> DataLoader:
    train_ds = BPRTrainDataset(dataset)
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
