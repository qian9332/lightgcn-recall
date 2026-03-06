"""
LightGCN 训练主流程
- 3层图卷积建模三阶连通性
- BPR损失 + In-Batch Negative
- 流行度加权负采样
- 完整训练日志记录
"""

import os
import sys
import time
import json
import logging
import argparse
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime

from dataset import GowallaDataset, get_train_loader
from model import LightGCN, BPRLoss
from evaluate import evaluate

# ── 参数配置 ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LightGCN Graph Recall Model")
    # 模型参数
    parser.add_argument("--embedding_dim", type=int, default=64,
                        help="Embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of GCN layers (models k-hop connectivity)")
    parser.add_argument("--layer_combination", type=str, default="mean",
                        choices=["mean", "learnable"],
                        help="Layer combination method")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Message dropout rate")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--reg_weight", type=float, default=1e-4,
                        help="L2 regularization weight")
    parser.add_argument("--use_in_batch_neg", action="store_true", default=True,
                        help="Use In-Batch Negative sampling")
    parser.add_argument("--in_batch_neg_weight", type=float, default=0.5,
                        help="Weight for In-Batch Negative loss")

    # 评估参数
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate every N epochs")
    parser.add_argument("--topk", type=str, default="20,50",
                        help="Top-K values for evaluation, comma separated")
    parser.add_argument("--eval_batch_size", type=int, default=256,
                        help="Evaluation batch size")

    # 其他
    parser.add_argument("--data_dir", type=str, default="data/processed",
                        help="Processed data directory")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader num_workers")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")

    return parser.parse_args()


def setup_logger(log_dir: str) -> logging.Logger:
    """配置训练日志"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("LightGCN")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # 文件handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # 控制台handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger, log_file


def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    set_seed(args.seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 日志
    logger, log_file = setup_logger(args.log_dir)
    logger.info("=" * 70)
    logger.info("LightGCN Graph Recall Model - Training Start")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {json.dumps(vars(args), indent=2)}")
    logger.info(f"Log file: {log_file}")

    # ── 数据加载 ──────────────────────────────────────────
    logger.info("\n📦 Loading dataset ...")
    dataset = GowallaDataset(args.data_dir)
    adj = dataset.get_adj_matrix(device)
    train_loader = get_train_loader(dataset, args.batch_size, args.num_workers)
    logger.info(f"   Users: {dataset.n_users:,}  Items: {dataset.n_items:,}")
    logger.info(f"   Train: {len(dataset.train_df):,}  Val: {len(dataset.val_df):,}  Test: {len(dataset.test_df):,}")
    logger.info(f"   Adjacency matrix: {dataset.n_nodes:,} x {dataset.n_nodes:,}")

    # ── 模型构建 ──────────────────────────────────────────
    topk_list = [int(k) for k in args.topk.split(",")]
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        layer_combination=args.layer_combination,
        dropout=args.dropout
    ).to(device)

    criterion = BPRLoss(
        reg_weight=args.reg_weight,
        use_in_batch_neg=args.use_in_batch_neg,
        in_batch_neg_weight=args.in_batch_neg_weight
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"\n🏗️  Model Architecture:")
    logger.info(f"   LightGCN with {args.n_layers} layers (3-hop connectivity)")
    logger.info(f"   Embedding dim: {args.embedding_dim}")
    logger.info(f"   Layer combination: {args.layer_combination}")
    logger.info(f"   Total params: {total_params:,}  Trainable: {trainable_params:,}")
    logger.info(f"   BPR + In-Batch Negative: {args.use_in_batch_neg}")
    logger.info(f"   Popularity-weighted negative sampling: enabled")

    # ── 训练循环 ──────────────────────────────────────────
    best_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    training_history = []

    logger.info(f"\n🚀 Starting training for {args.epochs} epochs ...")
    logger.info("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_inbatch = 0.0
        epoch_reg = 0.0
        n_batches = 0
        start_time = time.time()

        for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
            users = users.squeeze(1).to(device)
            pos_items = pos_items.squeeze(1).to(device)
            neg_items = neg_items.squeeze(1).to(device)

            # Forward
            user_e, pos_e, neg_e, user_ego, pos_ego, neg_ego = model(
                adj, users, pos_items, neg_items
            )

            # Loss
            loss, bpr_l, inbatch_l, reg_l = criterion(
                user_e, pos_e, neg_e, user_ego, pos_ego, neg_ego
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bpr += bpr_l
            epoch_inbatch += inbatch_l
            epoch_reg += reg_l
            n_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / n_batches
        avg_bpr = epoch_bpr / n_batches
        avg_inbatch = epoch_inbatch / n_batches
        avg_reg = epoch_reg / n_batches

        log_msg = (f"Epoch {epoch:3d}/{args.epochs} | "
                   f"Loss: {avg_loss:.4f} (BPR: {avg_bpr:.4f}, "
                   f"InBatch: {avg_inbatch:.4f}, Reg: {avg_reg:.4f}) | "
                   f"Time: {epoch_time:.1f}s")
        logger.info(log_msg)

        # ── 评估 ──────────────────────────────────────────
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            logger.info(f"  📊 Evaluating at epoch {epoch} ...")
            eval_start = time.time()
            metrics = evaluate(model, dataset, adj, device,
                             topk_list=topk_list,
                             batch_size=args.eval_batch_size,
                             verbose=False)
            eval_time = time.time() - eval_start

            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"  ✅ [{eval_time:.1f}s] {metrics_str}")

            # 记录历史
            record = {"epoch": epoch, "loss": avg_loss, **metrics}
            training_history.append(record)

            # 检查最佳模型
            main_metric = f"Recall@{topk_list[0]}"
            current_recall = metrics.get(main_metric, 0.0)
            if current_recall > best_recall:
                best_recall = current_recall
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                os.makedirs(args.ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "args": vars(args)
                }, ckpt_path)
                logger.info(f"  🏆 New best! {main_metric}={current_recall:.4f} "
                           f"(saved to {ckpt_path})")
            else:
                patience_counter += args.eval_interval
                logger.info(f"  ⏳ No improvement for {patience_counter} epochs "
                           f"(best: {best_recall:.4f} @ epoch {best_epoch})")

            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"\n⛔ Early stopping triggered at epoch {epoch}")
                break

    # ── 最终测试 ──────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("🎯 Final Evaluation on Test Set")
    logger.info("=" * 70)

    # 加载最佳模型
    ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics = evaluate(model, dataset, adj, device,
                           topk_list=topk_list,
                           batch_size=args.eval_batch_size,
                           verbose=True)

    logger.info("\n📋 Test Results:")
    for k, v in test_metrics.items():
        logger.info(f"   {k}: {v:.4f}")

    # ── 保存训练历史 ──────────────────────────────────────
    history_path = os.path.join(args.log_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"\n📝 Training history saved to {history_path}")

    # 保存最终模型
    final_ckpt = os.path.join(args.ckpt_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": test_metrics,
        "args": vars(args)
    }, final_ckpt)
    logger.info(f"💾 Final model saved to {final_ckpt}")
    logger.info("\n✅ Training completed!")


if __name__ == "__main__":
    main()
