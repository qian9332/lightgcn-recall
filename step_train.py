"""
单步训练脚本 - 每次调用训练1个epoch
适配sandbox 90秒超时限制
用法: python3 step_train.py [max_batches]
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
HIST_FILE = "logs/training_history.json"
TOTAL_EPOCHS = 50

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("LGN")
logger.setLevel(logging.INFO)
logger.handlers = []
for h in [logging.FileHandler(LOG_FILE, "a", "utf-8"), logging.StreamHandler(sys.stdout)]:
    h.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(h)

np.random.seed(2024); torch.manual_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CACHE, "rb") as f:
    dataset = pickle.load(f)
adj = dataset.get_adj_matrix(device)
loader = get_train_loader(dataset, batch_size=8192, num_workers=0)

model = LightGCN(dataset.n_users, dataset.n_items, 64, 3).to(device)
criterion = BPRLoss(1e-4, True, 0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

start_epoch = 1; best_recall = 0.0; best_ep = 0; history = []
if os.path.exists(CKPT):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optim"])
    start_epoch = ckpt["epoch"] + 1; best_recall = ckpt["best_r"]; best_ep = ckpt["best_ep"]
    history = ckpt.get("hist", [])
else:
    logger.info("=" * 60)
    logger.info(f"LightGCN Training | {device} | 3-layer dim=64")
    logger.info(f"Users={dataset.n_users} Items={dataset.n_items} Train={len(dataset.train_df)}")
    logger.info(f"BPR + InBatchNeg + PopWeighted | BS=8192 | batches/ep={len(loader)}")
    logger.info("=" * 60)

epoch = start_epoch
if epoch > TOTAL_EPOCHS:
    logger.info("Training completed!"); sys.exit(0)

MAX_B = int(sys.argv[1]) if len(sys.argv) > 1 else len(loader)

model.train()
L = [0.0]*4; nb = 0; t0 = time.time()
for i, (u, p, n) in enumerate(loader):
    if i >= MAX_B: break
    u=u.squeeze(1).to(device); p=p.squeeze(1).to(device); n=n.squeeze(1).to(device)
    ue,pe,ne,u0,p0,n0 = model(adj, u, p, n)
    loss,b,ib,r = criterion(ue,pe,ne,u0,p0,n0)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    L[0]+=loss.item(); L[1]+=b; L[2]+=ib; L[3]+=r; nb+=1
dt = time.time()-t0
l = [x/nb for x in L]
logger.info(f"Ep {epoch:2d}/{TOTAL_EPOCHS} [{nb}/{len(loader)}b] Loss={l[0]:.4f} BPR={l[1]:.4f} InB={l[2]:.4f} Reg={l[3]:.6f} | {dt:.0f}s")

# Eval every 5 epochs
if epoch % 5 == 0:
    m = evaluate(model, dataset, adj, device, topk_list=[20,50], batch_size=512, verbose=False)
    ms = " ".join(f"{k}={v:.4f}" for k,v in m.items())
    logger.info(f"   Eval: {ms}")
    history.append({"epoch":epoch, "loss":l[0], **m})
    r20 = m.get("Recall@20",0)
    if r20 > best_recall:
        best_recall=r20; best_ep=epoch
        torch.save({"epoch":epoch,"model":model.state_dict(),"metrics":m}, BEST)
        logger.info(f"   ★ Best R@20={r20:.4f}")
    with open(HIST_FILE, "w") as f: json.dump(history, f, indent=2)

torch.save({"epoch":epoch,"model":model.state_dict(),"optim":optimizer.state_dict(),
            "best_r":best_recall,"best_ep":best_ep,"hist":history}, CKPT)
