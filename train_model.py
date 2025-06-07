#!/usr/bin/env python3
"""
train_model.py  –  learn a Gradient‑Boosted Tree ensemble that replicates
the ACME legacy reimbursement logic.

Usage
-----
    python train_model.py public_cases.json [N_TREES]

    • N_TREES defaults to 120 (good balance of accuracy / size).
      Increase for a bit more accuracy or decrease for smaller model.

Outputs
-------
    • MAE on the 1 000 public cases.
    • A literal Python block (GBM_BIAS, GBM_TREES) to paste into
      calculate_reimbursement.py.

Note
----
This script *requires* scikit‑learn + NumPy, but those libraries **are
NOT** needed at runtime after you paste the printed block.
"""

import json, math, sys, numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor

# ─── CLI -------------------------------------------------------------------
if len(sys.argv) not in (2, 3):
    sys.exit("Usage: train_model.py public_cases.json [N_TREES]")
cases_path = Path(sys.argv[1])
N_TREES    = int(sys.argv[2]) if len(sys.argv) == 3 else 120

# ─── Load & feature‑engineer ----------------------------------------------
def load_cases(fname):
    raw = json.load(open(fname))
    X, y = [], []
    for c in raw:
        d = c["input"]["trip_duration_days"]
        m = c["input"]["miles_traveled"]
        r = c["input"]["total_receipts_amount"]

        mpd = m / d
        spd = r / d

        X.append([
            # base features
            d, m, r,
            mpd, spd,
            max(0, m-100), max(0, m-500),
            max(0, spd-50), max(0, spd-120),
            1 if d == 5 else 0,
            1 if d <= 3 else 0,
            1 if d >= 7 else 0,
            1 if int(round(r*100)) % 10 == 9 else 0,
            # new targeted signals
            max(0, spd-150),                        # spd_hi
            max(0, spd-250),                        # spd_vhi
            max(0, mpd-300),                        # mpd_hi
            math.log1p(r),                          # receipts_log
            1 if d >= 7 else 0,                     # long trip flag
            (1 if d >= 7 else 0) * max(0, spd-150), # long & hi‑spend
        ])
        y.append(c["expected_output"])
    return np.array(X, dtype=float), np.array(y, dtype=float)

X, y = load_cases(cases_path)

# ─── Train GBM --------------------------------------------------------------
model = GradientBoostingRegressor(
            n_estimators=N_TREES,
            max_depth=4,
            learning_rate=0.05,
            loss="absolute_error",      # modern alias for LAD
            random_state=0).fit(X, y)

mae = np.mean(np.abs(model.predict(X) - y))
print(f"Public‑set MAE: {mae:.4f}\n")

# ─── Export to pure‑Python lists -------------------------------------------
trees_out = []
for est in model.estimators_.ravel():
    t = est.tree_
    nodes = []
    for i in range(t.node_count):
        # Build a dict per node; omit unused keys for leaves
        n = {
            "v": float(t.value[i][0][0])
        }
        if t.children_left[i] != -1:   # not a leaf
            n.update(
                f=int(t.feature[i]),
                t=float(t.threshold[i]),
                l=int(t.children_left[i]),
                r=int(t.children_right[i]),
            )
        nodes.append(n)
    trees_out.append(nodes)

bias = float(np.asarray(model.init_.constant_).ravel()[0])

print("### Paste the block below into calculate_reimbursement.py ###\n")
print(f"GBM_BIAS = {bias:.6f}")
print("GBM_TREES = [")
for tree in trees_out:
    print(" [")
    for node in tree:
        print("  ", node, ",")
    print(" ],")
print("]")
