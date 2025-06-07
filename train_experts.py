#!/usr/bin/env python3
"""
train_experts.py  –  learner for a 3‑expert mixture model.

Usage
-----
    python train_experts.py public_cases.json [TREES_PER_EXPERT=200]

Prints a model block to paste into calculate_reimbursement.py.
"""

import json, math, sys, numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

N_TREES = int(sys.argv[2]) if len(sys.argv) == 3 else 200
raw     = json.load(open(sys.argv[1]))

# ─── feature function (20 dims) ───────────────────────────────────────────
def feats(d, m, r):
    mpd, spd = m/d, r/d
    return [
        d, m, r,
        mpd, spd,
        max(0, m-100), max(0, m-500),
        max(0, spd-50), max(0, spd-120),
        int(d==5), int(d<=3), int(d>=7),
        int(round(r*100)%10==9),
        max(0, spd-150), max(0, spd-250),
        max(0, mpd-300),
        math.log1p(r),
        int(d>=7), int(d>=7)*max(0, spd-150),
        int(mpd>350),
    ]

X, y = [], []
for c in raw:
    d = c["input"]["trip_duration_days"]
    m = c["input"]["miles_traveled"]
    r = c["input"]["total_receipts_amount"]
    X.append(feats(d, m, r))
    y.append(c["expected_output"])
X, y = np.array(X,float), np.array(y,float)

# ─── stage‑1: single GBM to get residual patterns -------------------------
base_gbm = GradientBoostingRegressor(
              n_estimators=120,max_depth=3,learning_rate=0.05,
              loss="absolute_error",random_state=0).fit(X,y)
pred0    = base_gbm.predict(X)

# ─── k‑means cluster on (actual,pred) to label 3 regimes ------------------
cl = KMeans(n_clusters=3, random_state=0).fit(np.column_stack([y, pred0]))
seg = cl.labels_

# ensure deterministic class order: sort by median spend/day
order = np.argsort([
    np.median([ raw[i]["input"]["total_receipts_amount"] /
                raw[i]["input"]["trip_duration_days"]
                for i in np.where(seg == k)[0] ])
    for k in range(3)
])
mapper = {old:new for new,old in enumerate(order)}
seg = np.vectorize(mapper.get)(seg)       # 0 = normal, 1 = high‑$ damp, 2 = jackpot

# ─── router (depth‑3, sees all 20 features) -------------------------------
router = DecisionTreeClassifier(max_depth=3, random_state=1).fit(X, seg)

# ─── train 3 experts (stronger) -------------------------------------------
experts=[]
for k in range(3):
    mask = seg==k
    mdl  = GradientBoostingRegressor(
               n_estimators=N_TREES,max_depth=4,learning_rate=0.05,
               loss="absolute_error",random_state=k).fit(X[mask], y[mask])
    experts.append(mdl)

mae = np.mean(np.abs(np.choose(seg, [e.predict(X) for e in experts]) - y))
print("Public MAE:", round(mae,3))

# ─── export helpers --------------------------------------------------------
def dump_tree(sklearn_tree):
    t = sklearn_tree.tree_
    out=[]
    for i in range(t.node_count):
        node={'v':float(t.value[i][0][0])}
        if t.children_left[i]!=-1:
            node.update(f=int(t.feature[i]),
                        t=float(t.threshold[i]),
                        l=int(t.children_left[i]),
                        r=int(t.children_right[i]))
        out.append(node)
    return out

def export_gbm(mdl):
    bias=float(np.asarray(mdl.init_.constant_).ravel()[0])
    forest=[dump_tree(est) for est in mdl.estimators_.ravel()]
    return bias, forest

# router nodes
rt = router.tree_
ROUTER_NODES = dump_tree(router)

# ─── emit python block -----------------------------------------------------
print("\n### paste below ###\n")
print("LEARNING_RATE = 0.05")
print("ROUTER_NODES = [")
for n in ROUTER_NODES: print(" ", n, ",")
print("]")

for i,mdl in enumerate(experts):
    b,forest = export_gbm(mdl)
    print(f"\nBIAS_{i} = {b:.6f}")
    print(f"TREES_{i} = [")
    for tr in forest:
        print(" [")
        for nd in tr: print("  ", nd, ",")
        print(" ],")
    print("]")
