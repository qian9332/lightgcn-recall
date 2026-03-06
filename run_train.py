"""
快速分段训练脚本 - 使用pickle缓存加速数据加载
用法: python3 run_train.py [epochs_this_run]
"""
import os, sys, time, json, logging, pickle
import torch, torch.optim as optim, numpy as np
from model import LightGCN, BPRLoss
from evaluate import evaluate
from dataset import get_train_loader

# ── Config ──
CFG = dict(
    embed_dim=64, n_layers=3, batch_size=2048, lr=1e-3,
    reg_weight=1e-4, in_batch_neg=True, in_batch_w=0.5,
    total_epochs=80, eval_interval=5, topk=[20, 50], seed=2024,
)
CACHE = "data/processed/dataset_cache.pkl"
CKPT = "checkpoints/resume.pt"
BEST = "checkpoints/best_model.pt"
LOG_FILE = "logs/training.log"
HIST_FILE = "logs/training_history.json"
EPOCHS_THIS_RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 2

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logger
logger = logging.getLogger("LGN")
logger.setLevel(logging.INFO)
logger.handlers = []
for h in [logging.FileHandler(LOG_FILE, "a", "utf-8"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)

np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data (from cache)
t0 = time.time()
with open(CACHE, "rb") as f:
    dataset = pickle.load(f)
adj = dataset.get_adj_matrix(device)
loader = get_train_loader(dataset, CFG["batch_size"], num_workers=0)
logger.info(f"Data loaded in {time.time()-t0:.1f}s | {dataset.n_users} users, {dataset.n_items} items")

# Model
model = LightGCN(dataset.n_users, dataset.n_items, CFG["embed_dim"], CFG["n_layers"]).to(device)
criterion = BPRLoss(CFG["reg_weight"], CFG["in_batch_neg"], CFG["in_batch_w"])
optimizer = optim.Adam(model.parameters(), lr=CFG["lr"])

# Resume
start_epoch = 1; best_recall = 0.0; best_ep = 0; history = []
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=device)
    model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optim"])
    start_epoch = ckpt["epoch"] + 1; best_recall = ckpt["best_r"]; best_ep = ckpt["best_ep"]
    history = ckpt.get("hist", [])
    logger.info(f"Resumed @ epoch {start_epoch}, best R@20={best_recall:.4f}")
else:
    logger.info("=" * 60)
    logger.info(f"LightGCN Training | device={device} | {CFG['n_layers']}-layer, dim={CFG['embed_dim']}")
    logger.info(f"BPR + InBatchNeg + PopWeighted Sampling | {dataset.n_users} users, {dataset.n_items} items, {len(dataset.train_df)} train")
    logger.info("=" * 60)

end_epoch = min(start_epoch + EPOCHS_THIS_RUN - 1, CFG["total_epochs"])
logger.info(f"--- Running epochs {start_epoch} -> {end_epoch} ---")

for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    losses = [0.0]*4; nb = 0; t1 = time.time()
    for users, pos_items, neg_items in loader:
        u = users.squeeze(1).to(device); p = pos_items.squeeze(1).to(device); n = neg_items.squeeze(1).to(device)
        ue, pe, ne, u0, p0, n0 = model(adj, u, p, n)
        loss, bpr, inb, reg = criterion(ue, pe, ne, u0, p0, n0)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses[0] += loss.item(); losses[1] += bpr; losses[2] += inb; losses[3] += reg; nb += 1
    
    dt = time.time() - t1
    l = [x/nb for x in losses]
    logger.info(f"Epoch {epoch:3d}/{CFG['total_epochs']} | Loss={l[0]:.4f} (BPR={l[1]:.4f} InB={l[2]:.4f} Reg={l[3]:.4f}) | {dt:.1f}s")
    
    # Evaluate
    do_eval = (epoch % CFG["eval_interval"] == 0) or (epoch == end_epoch)
    if do_eval:
        metrics = evaluate(model, dataset, adj, device, topk_list=CFG["topk"], batch_size=256, verbose=False)
        mstr = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(f"  >> Eval: {mstr}")
        history.append({"epoch": epoch, "loss": l[0], **metrics})
        
        r20 = metrics.get("Recall@20", 0.0)
        if r20 > best_recall:
            best_recall = r20; best_ep = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": metrics}, BEST)
            logger.info(f"  ★ New best! R@20={r20:.4f}")
        else:
            logger.info(f"  (best: R@20={best_recall:.4f} @ ep{best_ep})")
    
    # Save checkpoint
    torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
                "best_r": best_recall, "best_ep": best_ep, "hist": history}, CKPT)

# Save history
with open(HIST_FILE, "w") as f:
    json.dump(history, f, indent=2)

# Final eval if done
if end_epoch >= CFG["total_epochs"]:
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    if os.path.exists(BEST):
        bst = torch.load(BEST, map_location=device)
        model.load_state_dict(bst["model"])
        logger.info(f"Loaded best model from epoch {bst['epoch']}")
    tm = evaluate(model, dataset, adj, device, topk_list=CFG["topk"], batch_size=256, verbose=False)
    for k, v in tm.items():
        logger.info(f"  {k} = {v:.4f}")
    logger.info("✅ Training completed!")
    torch.save({"epoch": end_epoch, "model": model.state_dict(), "test_metrics": tm}, "checkpoints/final_model.pt")

logger.info(f"Best: R@20={best_recall:.4f} @ epoch {best_ep}")
print(f"\nDone. Next run: python3 run_train.py {EPOCHS_THIS_RUN}")
