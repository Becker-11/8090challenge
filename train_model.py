#!/usr/bin/env python3
"""
train_model.py  –  GBM + bias‑table trainer for the ACME challenge.

Usage
-----
    python train_model.py public_cases.json [N_TREES]

    • N_TREES defaults to 600.
      The printed block goes into calculate_reimbursement.py.

Runtime dependencies
--------------------
Only scikit‑learn + NumPy *during training*.
The exported block uses pure Python at inference.
"""
import json, math, sys, numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.ensemble import GradientBoostingRegressor

# ─── CLI -------------------------------------------------------------------
if len(sys.argv) not in (2, 3):
    sys.exit("Usage: train_model.py public_cases.json [N_TREES]")
cases_path = Path(sys.argv[1])
N_TREES    = int(sys.argv[2]) if len(sys.argv) == 3 else 600
LR         = 0.02                       # learning rate used later

# ─── Feature engineering (must match runtime!) ----------------------------
def make_features(days, miles, receipts):
    mpd = miles / days
    spd = receipts / days
    return [
        days, miles, receipts,
        mpd, spd,
        max(0, miles-100), max(0, miles-500),
        max(0, spd-50), max(0, spd-120),
        1 if days==5  else 0,
        1 if days<=3  else 0,
        1 if days>=7  else 0,
        1 if int(round(receipts*100))%10==9 else 0,
        max(0, spd-150),
        max(0, spd-250),
        max(0, mpd-300),
        math.log1p(receipts),
        1 if days>=7 else 0,
        (1 if days>=7 else 0)*max(0, spd-150),
    ]

raw = json.load(open(cases_path))
X  = np.array([make_features(c["input"]["trip_duration_days"],
                             c["input"]["miles_traveled"],
                             c["input"]["total_receipts_amount"])
               for c in raw], float)
y  = np.array([c["expected_output"] for c in raw], float)

# ─── Train tuned GBM -------------------------------------------------------
model = GradientBoostingRegressor(
            n_estimators=N_TREES,
            learning_rate=LR,
            max_depth=4,
            min_samples_leaf=10,
            loss="huber",
            alpha=0.9,                   # Huber threshold
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=0).fit(X, y)

pred = model.predict(X)
mae  = np.mean(np.abs(pred - y))
print(f"Public‑set MAE after GBM: {mae:.3f}")

# ─── Build 3‑D bias table --------------------------------------------------
def cell(feat_row):
    days = int(feat_row[0])
    mpd  = feat_row[3]
    spd  = feat_row[4]
    return (min(days,9), int(mpd//50), int(spd//50))

bucket = defaultdict(list)
for f, res in zip(X, y - pred):
    bucket[cell(f)].append(res)

BIAS_TABLE = {k: float(sum(v)/len(v)) for k,v in bucket.items()}

# ─── Global tweaks ---------------------------------------------------------
arte_adj = float(np.mean([r for f,r in zip(X, y-pred)
                          if int(round(f[2]*100))%50==49]))
five_adj = float(np.mean([r for f,r in zip(X, y-pred) if int(f[0])==5]))
long_adj = float(np.mean([r for f,r in zip(X, y-pred) if int(f[0])>=12]))

print("MAE after bias table  :", 
      np.mean(np.abs((pred +
                      np.array([BIAS_TABLE[cell(f)] for f in X]))
                     - y)).round(3))

# ─── Export GBM trees ------------------------------------------------------
trees_out = []
for est in model.estimators_.ravel():
    t = est.tree_
    nodes=[]
    for i in range(t.node_count):
        n={"v": float(t.value[i][0][0])}
        if t.children_left[i] != -1:
            n.update(f=int(t.feature[i]),
                     t=float(t.threshold[i]),
                     l=int(t.children_left[i]),
                     r=int(t.children_right[i]))
        nodes.append(n)
    trees_out.append(nodes)

bias = float(model.init_.constant_[0][0])

# ─── Print paste‑ready block ----------------------------------------------
print("\n### Paste the block below into calculate_reimbursement.py ###\n")
print(f"LEARNING_RATE = {LR}")
print(f"GBM_BIAS = {bias:.6f}")
print("GBM_TREES = [")
for tr in trees_out:
    print(" [")
    for n in tr:
        print("  ", n, ",")
    print(" ],")
print("]\n")

print("BIAS_TABLE = {")
for k,v in BIAS_TABLE.items():
    print(f" {k}: {round(v,4)},")
print("}\n")

print(f"ARTE_ADJ = {round(arte_adj,4)}")
print(f"FIVE_ADJ = {round(five_adj,4)}")
print(f"LONG_ADJ = {round(long_adj,4)}")
