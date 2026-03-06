"""
分段训练脚本 - 支持checkpoint恢复
每次运行训练指定的epoch数，通过checkpoint断点续训
"""

import os
import sys
import time
import json
import logging
import pickle
import torch
import torch.optim as optim
import numpy as np
from datetime import datetime

from dataset import GowallaDataset, get_train_loader
from model import LightGCN, BPRLoss
from evaluate import evaluate

# ── 配置 ──
EMBEDDING_DIM = 64
N_LAYERS = 3
BATCH_SIZE = 2048
LR = 1e-3
REG_WEIGHT = 1e-4
USE_IN_BATCH_NEG = True
IN_BATCH_NEG_WEIGHT = 0.5
TOTAL_EPOCHS = 80
EVAL_INTERVAL = 5
TOPK = [20, 50]
SEED = 2024
DATA_DIR = "data/processed"
LOG_DIR = "logs"
CKPT_DIR = "checkpoints"
RESUME_CKPT = os.path.join(CKPT_DIR, "resume_checkpoint.pt")
STATE_FILE = os.path.join(CKPT_DIR, "training_state.json")

# 每次运行训练的epoch数
EPOCHS_PER_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, "training.log")
    
    logger = logging.getLogger("LightGCN")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()
    
    # 加载数据
    logger.info("Loading dataset ...")
    dataset = GowallaDataset(DATA_DIR)
    adj = dataset.get_adj_matrix(device)
    train_loader = get_train_loader(dataset, BATCH_SIZE, num_workers=0)
    
    # 构建模型
    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=EMBEDDING_DIM,
        n_layers=N_LAYERS,
    ).to(device)
    
    criterion = BPRLoss(
        reg_weight=REG_WEIGHT,
        use_in_batch_neg=USE_IN_BATCH_NEG,
        in_batch_neg_weight=IN_BATCH_NEG_WEIGHT
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 恢复checkpoint
    start_epoch = 1
    best_recall = 0.0
    best_epoch = 0
    training_history = []
    
    if os.path.exists(RESUME_CKPT):
        logger.info(f"Resuming from checkpoint: {RESUME_CKPT}")
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt.get("best_recall", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        training_history = ckpt.get("history", [])
        logger.info(f"Resumed at epoch {start_epoch}, best Recall@20={best_recall:.4f}")
    else:
        logger.info("=" * 70)
        logger.info("LightGCN Graph Recall - Training Start")
        logger.info(f"Device: {device}")
        logger.info(f"Users: {dataset.n_users:,}  Items: {dataset.n_items:,}")
        logger.info(f"Train: {len(dataset.train_df):,}  Val: {len(dataset.val_df):,}  Test: {len(dataset.test_df):,}")
        logger.info(f"Model: {N_LAYERS}-layer LightGCN, dim={EMBEDDING_DIM}")
        logger.info(f"BPR + In-Batch Negative + Popularity-weighted sampling")
        logger.info("=" * 70)
    
    end_epoch = min(start_epoch + EPOCHS_PER_RUN - 1, TOTAL_EPOCHS)
    logger.info(f"\nTraining epochs {start_epoch} -> {end_epoch}")
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_loss = 0.0
        epoch_bpr = 0.0
        epoch_inbatch = 0.0
        epoch_reg = 0.0
        n_batches = 0
        t0 = time.time()
        
        for users, pos_items, neg_items in train_loader:
            users = users.squeeze(1).to(device)
            pos_items = pos_items.squeeze(1).to(device)
            neg_items = neg_items.squeeze(1).to(device)
            
            user_e, pos_e, neg_e, u_ego, p_ego, n_ego = model(adj, users, pos_items, neg_items)
            loss, bpr_l, inb_l, reg_l = criterion(user_e, pos_e, neg_e, u_ego, p_ego, n_ego)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_bpr += bpr_l
            epoch_inbatch += inb_l
            epoch_reg += reg_l
            n_batches += 1
        
        dt = time.time() - t0
        avg_loss = epoch_loss / n_batches
        logger.info(f"Epoch {epoch:3d}/{TOTAL_EPOCHS} | Loss: {avg_loss:.4f} "
                    f"(BPR: {epoch_bpr/n_batches:.4f}, InBatch: {epoch_inbatch/n_batches:.4f}, "
                    f"Reg: {epoch_reg/n_batches:.4f}) | {dt:.1f}s")
        
        # 评估
        do_eval = (epoch % EVAL_INTERVAL == 0) or (epoch == end_epoch) or (epoch == TOTAL_EPOCHS)
        if do_eval:
            logger.info(f"  Evaluating ...")
            metrics = evaluate(model, dataset, adj, device, topk_list=TOPK,
                             batch_size=256, verbose=False)
            mstr = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"  Results: {mstr}")
            
            training_history.append({"epoch": epoch, "loss": avg_loss, **metrics})
            
            r20 = metrics.get("Recall@20", 0.0)
            if r20 > best_recall:
                best_recall = r20
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                }, os.path.join(CKPT_DIR, "best_model.pt"))
                logger.info(f"  ★ New best! Recall@20={r20:.4f}")
        
        # 保存resume checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_recall": best_recall,
            "best_epoch": best_epoch,
            "history": training_history,
        }, RESUME_CKPT)
    
    # 保存训练历史
    with open(os.path.join(LOG_DIR, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)
    
    if end_epoch >= TOTAL_EPOCHS:
        logger.info("\n" + "=" * 70)
        logger.info("Training COMPLETE! Final evaluation on test set ...")
        best_ckpt = os.path.join(CKPT_DIR, "best_model.pt")
        if os.path.exists(best_ckpt):
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded best model from epoch {ckpt['epoch']}")
        
        test_metrics = evaluate(model, dataset, adj, device, topk_list=TOPK,
                               batch_size=256, verbose=False)
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        logger.info("✅ All training completed!")
    else:
        logger.info(f"\nRun completed (epoch {start_epoch}-{end_epoch}). "
                    f"Next: python3 train_resume.py {EPOCHS_PER_RUN}")
    
    logger.info(f"Best so far: Recall@20={best_recall:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
