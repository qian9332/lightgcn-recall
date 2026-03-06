"""
Gowalla数据集预处理脚本
- 读取原始签到数据(user, timestamp, lat, lon, item)
- 过滤低频用户和物品(至少10次交互)
- 重新编码ID
- 按8:1:1划分训练/验证/测试集
- 输出稀疏交互矩阵与划分文件
"""

import os
import random
import numpy as np
import pandas as pd
from collections import Counter

random.seed(2024)
np.random.seed(2024)

RAW_PATH = "data/gowalla_checkins.txt"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_USER_INTER = 10  # 用户最少交互数
MIN_ITEM_INTER = 10  # 物品最少交互数

# ── 1. 读取原始数据 ─────────────────────────────────────
print("📖 Reading raw data ...")
df = pd.read_csv(RAW_PATH, sep="\t", header=None,
                 names=["user", "timestamp", "lat", "lon", "item"],
                 dtype={"user": int, "item": int})
print(f"   Raw interactions: {len(df):,}")

# 去重 (user, item) 保留唯一交互
df = df[["user", "item"]].drop_duplicates()
print(f"   After dedup: {len(df):,}")

# ── 2. 核心过滤 ─────────────────────────────────────────
print("🔍 Filtering low-frequency users/items ...")
for round_i in range(5):  # 迭代过滤
    before = len(df)
    user_cnt = df["user"].value_counts()
    df = df[df["user"].isin(user_cnt[user_cnt >= MIN_USER_INTER].index)]
    item_cnt = df["item"].value_counts()
    df = df[df["item"].isin(item_cnt[item_cnt >= MIN_ITEM_INTER].index)]
    after = len(df)
    print(f"   Round {round_i+1}: {before:,} -> {after:,}")
    if before == after:
        break

# ── 3. 重新编码ID ───────────────────────────────────────
print("🔢 Re-indexing IDs ...")
user_map = {uid: idx for idx, uid in enumerate(sorted(df["user"].unique()))}
item_map = {iid: idx for idx, iid in enumerate(sorted(df["item"].unique()))}
df["user"] = df["user"].map(user_map)
df["item"] = df["item"].map(item_map)

n_users = len(user_map)
n_items = len(item_map)
n_interactions = len(df)
print(f"   Users: {n_users:,}  Items: {n_items:,}  Interactions: {n_interactions:,}")
print(f"   Density: {n_interactions / (n_users * n_items) * 100:.4f}%")

# ── 4. 划分数据集 ────────────────────────────────────────
print("✂️  Splitting train/val/test (8:1:1) ...")
train_list, val_list, test_list = [], [], []

user_items = df.groupby("user")["item"].apply(list).to_dict()
for uid, items in user_items.items():
    random.shuffle(items)
    n = len(items)
    n_val = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    test_list.extend([(uid, iid) for iid in items[:n_test]])
    val_list.extend([(uid, iid) for iid in items[n_test:n_test+n_val]])
    train_list.extend([(uid, iid) for iid in items[n_test+n_val:]])

train_df = pd.DataFrame(train_list, columns=["user", "item"])
val_df = pd.DataFrame(val_list, columns=["user", "item"])
test_df = pd.DataFrame(test_list, columns=["user", "item"])

print(f"   Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

# ── 5. 保存 ─────────────────────────────────────────────
train_df.to_csv(os.path.join(OUT_DIR, "train.txt"), sep="\t", index=False, header=False)
val_df.to_csv(os.path.join(OUT_DIR, "val.txt"), sep="\t", index=False, header=False)
test_df.to_csv(os.path.join(OUT_DIR, "test.txt"), sep="\t", index=False, header=False)

# 保存统计信息
with open(os.path.join(OUT_DIR, "stats.txt"), "w") as f:
    f.write(f"n_users\t{n_users}\n")
    f.write(f"n_items\t{n_items}\n")
    f.write(f"n_train\t{len(train_df)}\n")
    f.write(f"n_val\t{len(val_df)}\n")
    f.write(f"n_test\t{len(test_df)}\n")
    f.write(f"density\t{n_interactions / (n_users * n_items) * 100:.6f}\n")

# 保存物品流行度(用于负采样)
item_pop = train_df["item"].value_counts().sort_index()
pop_array = np.zeros(n_items, dtype=np.int64)
pop_array[item_pop.index.values] = item_pop.values
np.save(os.path.join(OUT_DIR, "item_popularity.npy"), pop_array)

print("✅ Preprocessing done! Files saved to:", OUT_DIR)
