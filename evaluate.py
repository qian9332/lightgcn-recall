"""
评估指标: Recall@K, NDCG@K
全量用户评估, 排除训练集已交互物品
"""

import math
import torch
import numpy as np
from tqdm import tqdm


def evaluate(model, dataset, adj, device, topk_list=(20, 50),
             batch_size=256, verbose=True):
    """
    评估模型在验证/测试集上的 Recall@K 和 NDCG@K
    
    Args:
        model: LightGCN模型
        dataset: GowallaDataset
        adj: 归一化邻接矩阵 (on device)
        device: 计算设备
        topk_list: 评估的K值列表
        batch_size: 评估batch大小
        verbose: 是否显示进度条
    
    Returns:
        dict: {metric_name: value}
    """
    model.eval()
    max_k = max(topk_list)

    # 获取所有Embedding
    with torch.no_grad():
        user_embeds, item_embeds = model.get_all_embeddings(adj)

    # 收集评估指标
    recalls = {k: [] for k in topk_list}
    ndcgs = {k: [] for k in topk_list}

    # 获取需要评估的用户 (测试集中有交互的用户)
    test_users = list(dataset.test_user_items.keys())
    n_test_users = len(test_users)

    iterator = range(0, n_test_users, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="Evaluating", ncols=80)

    for start in iterator:
        end = min(start + batch_size, n_test_users)
        batch_users = test_users[start:end]
        users_tensor = torch.LongTensor(batch_users).to(device)

        with torch.no_grad():
            # 计算得分: (batch, n_items)
            batch_user_embeds = user_embeds[users_tensor]
            scores = batch_user_embeds @ item_embeds.T

        # 排除训练集已交互物品 (设为-inf)
        for i, uid in enumerate(batch_users):
            train_items = dataset.train_user_items.get(uid, set())
            if train_items:
                scores[i, list(train_items)] = -float("inf")

        # Top-K
        _, topk_indices = torch.topk(scores, max_k, dim=1)
        topk_indices = topk_indices.cpu().numpy()

        # 计算指标
        for i, uid in enumerate(batch_users):
            gt_items = dataset.test_user_items.get(uid, set())
            if not gt_items:
                continue

            pred_items = topk_indices[i]

            for k in topk_list:
                pred_k = pred_items[:k]
                hits = len(set(pred_k) & gt_items)

                # Recall@K
                recalls[k].append(hits / min(len(gt_items), k))

                # NDCG@K
                dcg = 0.0
                for pos, item_id in enumerate(pred_k):
                    if item_id in gt_items:
                        dcg += 1.0 / math.log2(pos + 2)
                idcg = sum(1.0 / math.log2(pos + 2)
                          for pos in range(min(len(gt_items), k)))
                ndcgs[k].append(dcg / idcg if idcg > 0 else 0.0)

    results = {}
    for k in topk_list:
        results[f"Recall@{k}"] = np.mean(recalls[k]) if recalls[k] else 0.0
        results[f"NDCG@{k}"] = np.mean(ndcgs[k]) if ndcgs[k] else 0.0

    return results
