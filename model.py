"""
LightGCN 模型实现
- 移除特征变换权重矩阵
- 移除非线性激活函数
- 仅保留邻居归一化聚合
- 层组合: 各层Embedding加权求和
- 支持 In-Batch Negative 增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    核心设计:
    1. 图卷积: E^(k+1) = A_norm * E^(k)  (无特征变换，无非线性激活)
    2. 层组合: E_final = Σ α_k * E^(k)   (各层表示加权求和)
    3. 预测:   score(u,i) = E_u^T * E_i
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers: int = 3, layer_combination: str = "mean",
                 dropout: float = 0.0):
        """
        Args:
            n_users: 用户数量
            n_items: 物品数量
            embedding_dim: Embedding维度
            n_layers: 图卷积层数 (建模k阶连通性)
            layer_combination: 层组合方式 "mean" | "learnable"
            dropout: 消息传播dropout
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        # 初始化Embedding (第0层)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # 可学习的层组合权重
        if layer_combination == "learnable":
            self.layer_weights = nn.Parameter(torch.ones(n_layers + 1) / (n_layers + 1))
        else:
            # 均匀加权: α_k = 1/(K+1)
            self.register_buffer("layer_weights",
                                 torch.ones(n_layers + 1) / (n_layers + 1))

    def graph_convolution(self, adj: torch.sparse.FloatTensor) -> tuple:
        """
        执行多层图卷积, 返回最终用户/物品Embedding
        
        E^(0) = [E_user ; E_item]  (初始Embedding拼接)
        E^(k+1) = A_norm * E^(k)   (逐层传播, 无变换无激活)
        E_final = Σ α_k * E^(k)    (加权层组合)
        """
        # 第0层: 拼接user和item的初始embedding
        ego_embed = torch.cat([self.user_embedding.weight,
                               self.item_embedding.weight], dim=0)  # (n_users+n_items, dim)

        all_layer_embeds = [ego_embed]  # 收集每层输出

        current_embed = ego_embed
        for layer in range(self.n_layers):
            # 图卷积: 邻居归一化聚合 (核心: 无W变换, 无σ激活)
            if self.training and self.dropout > 0:
                # 训练时对邻接矩阵做dropout (Message Dropout)
                current_embed = torch.sparse.mm(adj, current_embed)
                current_embed = F.dropout(current_embed, p=self.dropout,
                                         training=self.training)
            else:
                current_embed = torch.sparse.mm(adj, current_embed)

            all_layer_embeds.append(current_embed)

        # 层组合: 加权求和
        all_layer_embeds = torch.stack(all_layer_embeds, dim=0)  # (n_layers+1, n_nodes, dim)
        weights = F.softmax(self.layer_weights, dim=0).view(-1, 1, 1)
        final_embed = (weights * all_layer_embeds).sum(dim=0)  # (n_nodes, dim)

        user_embeds = final_embed[:self.n_users]
        item_embeds = final_embed[self.n_users:]
        return user_embeds, item_embeds

    def forward(self, adj: torch.sparse.FloatTensor,
                users: torch.LongTensor,
                pos_items: torch.LongTensor,
                neg_items: torch.LongTensor):
        """
        前向传播, 返回BPR loss所需的embedding
        
        Args:
            adj: 归一化邻接矩阵
            users: 用户ID (batch,)
            pos_items: 正样本物品ID (batch,)
            neg_items: 负样本物品ID (batch,)
        """
        all_user_embeds, all_item_embeds = self.graph_convolution(adj)

        user_embeds = all_user_embeds[users]       # (batch, dim)
        pos_embeds = all_item_embeds[pos_items]     # (batch, dim)
        neg_embeds = all_item_embeds[neg_items]     # (batch, dim)

        # 初始Embedding (用于L2正则)
        user_ego = self.user_embedding(users)
        pos_ego = self.item_embedding(pos_items)
        neg_ego = self.item_embedding(neg_items)

        return user_embeds, pos_embeds, neg_embeds, user_ego, pos_ego, neg_ego

    def get_all_embeddings(self, adj: torch.sparse.FloatTensor):
        """获取所有用户和物品的最终Embedding (用于推理)"""
        return self.graph_convolution(adj)

    @torch.no_grad()
    def predict(self, adj: torch.sparse.FloatTensor,
                users: torch.LongTensor) -> torch.FloatTensor:
        """
        预测指定用户对所有物品的得分
        score(u, i) = E_u^T * E_i
        """
        user_embeds, item_embeds = self.get_all_embeddings(adj)
        user_e = user_embeds[users]  # (batch, dim)
        scores = user_e @ item_embeds.T  # (batch, n_items)
        return scores


class BPRLoss(nn.Module):
    """
    BPR (Bayesian Personalized Ranking) 损失函数
    L_BPR = -Σ log σ(score(u,i+) - score(u,i-)) + λ||Θ||^2
    
    支持 In-Batch Negative: 将batch内其他用户的正样本作为额外负样本
    """

    def __init__(self, reg_weight: float = 1e-4, use_in_batch_neg: bool = True,
                 in_batch_neg_weight: float = 0.5):
        super().__init__()
        self.reg_weight = reg_weight
        self.use_in_batch_neg = use_in_batch_neg
        self.in_batch_neg_weight = in_batch_neg_weight

    def forward(self, user_embeds, pos_embeds, neg_embeds,
                user_ego, pos_ego, neg_ego):
        """
        Args:
            user_embeds: 用户GCN输出embedding (batch, dim)
            pos_embeds: 正样本GCN输出embedding (batch, dim)
            neg_embeds: 负样本GCN输出embedding (batch, dim)
            user_ego/pos_ego/neg_ego: 初始embedding (用于L2正则)
        """
        batch_size = user_embeds.shape[0]

        # ── 标准BPR损失 ──
        pos_scores = (user_embeds * pos_embeds).sum(dim=1)  # (batch,)
        neg_scores = (user_embeds * neg_embeds).sum(dim=1)  # (batch,)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # ── In-Batch Negative损失 ──
        in_batch_loss = torch.tensor(0.0, device=user_embeds.device)
        if self.use_in_batch_neg and batch_size > 1:
            # 计算用户与batch内所有正样本的得分矩阵
            all_scores = user_embeds @ pos_embeds.T  # (batch, batch)
            # 对角线是正样本得分
            pos_diag = torch.diag(all_scores)  # (batch,)
            # In-batch negative: 使用softmax + cross entropy
            labels = torch.arange(batch_size, device=user_embeds.device)
            in_batch_loss = F.cross_entropy(all_scores, labels)

        # ── L2正则 (仅对初始Embedding) ──
        reg_loss = (1/2) * (user_ego.norm(2).pow(2) +
                            pos_ego.norm(2).pow(2) +
                            neg_ego.norm(2).pow(2)) / batch_size

        total_loss = (bpr_loss +
                      self.in_batch_neg_weight * in_batch_loss +
                      self.reg_weight * reg_loss)

        return total_loss, bpr_loss.item(), in_batch_loss.item(), reg_loss.item()
