"""
快速训练脚本 - 每个epoch只跑部分batch, 适合sandbox环境
完整训练请使用 train.py (推荐在GPU环境)
"""
import os, sys, time, json, logging, pickle
import torch, torch.optim as optim, numpy as np
from model import LightGCN, BPRLoss
from evaluate import evaluate
from dataset import get_train_loader

CACHE = "data/processed/dataset_cache.pkl"
CKPT = "checkpoints/resume.pt"
BEST = "checkpoints/best_model.pt"
LOG_FILE = "logs/training.log"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("LGN")
logger.setLevel(logging.INFO)
logger.handlers = []
for h in [logging.FileHandler(LOG_FILE, "a", "utf-8"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)

SEED = 2024
np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load
with open(CACHE, "rb") as f:
    dataset = pickle.load(f)
adj = dataset.get_adj_matrix(device)
loader = get_train_loader(dataset, batch_size=8192, num_workers=0)

model = LightGCN(dataset.n_users, dataset.n_items, 64, 3).to(device)
criterion = BPRLoss(1e-4, True, 0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
    logger.info(f"LightGCN Training | {device} | 3-layer dim=64")
    logger.info(f"Users={dataset.n_users} Items={dataset.n_items} Train={len(dataset.train_df)}")
    logger.info("=" * 60)

# Train 1 epoch
epoch = start_epoch
if epoch > 80:
    logger.info("Training already completed!")
    sys.exit(0)

model.train()
losses = [0.0]*4; nb = 0; t0 = time.time()
MAX_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else len(loader)  # 可限制batch数

for i, (u, p, n) in enumerate(loader):
    if i >= MAX_BATCHES: break
    u=u.squeeze(1).to(device); p=p.squeeze(1).to(device); n=n.squeeze(1).to(device)
    ue,pe,ne,u0,p0,n0 = model(adj, u, p, n)
    loss, bpr, inb, reg = criterion(ue,pe,ne,u0,p0,n0)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses[0]+=loss.item(); losses[1]+=bpr; losses[2]+=inb; losses[3]+=reg; nb+=1

dt = time.time()-t0
l = [x/nb for x in losses]
logger.info(f"Epoch {epoch:3d}/80 [{nb}/{len(loader)} batches] | Loss={l[0]:.4f} (BPR={l[1]:.4f} InB={l[2]:.4f}) | {dt:.1f}s")

# Evaluate every 5 epochs
if epoch % 5 == 0 or epoch == 80:
    metrics = evaluate(model, dataset, adj, device, topk_list=[20,50], batch_size=256, verbose=False)
    mstr = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    logger.info(f"  >> Eval: {mstr}")
    history.append({"epoch": epoch, "loss": l[0], **metrics})
    r20 = metrics.get("Recall@20", 0.0)
    if r20 > best_recall:
        best_recall = r20; best_ep = epoch
        torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": metrics}, BEST)
        logger.info(f"  ★ Best R@20={r20:.4f}")

# Save
torch.save({"epoch": epoch, "model": model.state_dict(), "optim": optimizer.state_dict(),
            "best_r": best_recall, "best_ep": best_ep, "hist": history}, CKPT)

with open("logs/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

logger.info(f"Best: R@20={best_recall:.4f} @ ep{best_ep}")
