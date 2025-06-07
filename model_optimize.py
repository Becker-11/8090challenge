#!/usr/bin/env python3
"""
train_model_optuna.py  –  GBM + bias‑table trainer **with Optuna HPO** for the ACME challenge.

Usage
-----
    python train_model_optuna.py public_cases.json \
        [--n_trials 100] [--timeout 600] [--seed 0]

    • By default 50 Optuna trials are run; set --n_trials 0 to skip HPO.
    • After the search, the script retrains on the full set with the
      best hyper‑parameters **and** prints a paste‑ready block that can be
      dropped into `calculate_reimbursement.py` (same format as the
      original script).

Runtime dependencies
--------------------
* scikit‑learn ≥ 1.3
* numpy
* optuna ≥ 3.6 (only during training; not needed at inference)

The exported block uses **pure Python** at runtime – no external deps.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:
    import optuna  # only needed when --n_trials > 0
except ImportError:
    optuna = None

# ────────────────────────── Feature engineering (keep in sync !) ───────────
LR_DEFAULT = 0.02  # used when HPO is off


def make_features(days: float, miles: float, receipts: float) -> list[float]:
    """Return feature row (must match inference)."""
    mpd = miles / days
    spd = receipts / days
    return [
        days,
        miles,
        receipts,
        mpd,
        spd,
        max(0, miles - 100),
        max(0, miles - 500),
        max(0, spd - 50),
        max(0, spd - 120),
        1 if days == 5 else 0,
        1 if days <= 3 else 0,
        1 if days >= 7 else 0,
        1 if int(round(receipts * 100)) % 10 == 9 else 0,
        max(0, spd - 150),
        max(0, spd - 250),
        max(0, mpd - 300),
        math.log1p(receipts),
        1 if days >= 7 else 0,
        (1 if days >= 7 else 0) * max(0, spd - 150),
    ]


# ────────────────────────────── Helpers ─────────────────────────────────────

def load_dataset(json_path: Path):
    raw = json.load(open(json_path))
    X = np.array([
        make_features(
            c["input"]["trip_duration_days"],
            c["input"]["miles_traveled"],
            c["input"]["total_receipts_amount"],
        )
        for c in raw
    ], float)
    y = np.array([c["expected_output"] for c in raw], float)
    return X, y


def build_gbm(params: dict, seed: int) -> GradientBoostingRegressor:
    params = params.copy()
    params.setdefault("n_estimators", 600)
    params.setdefault("learning_rate", LR_DEFAULT)
    params.setdefault("max_depth", 4)
    params.setdefault("min_samples_leaf", 10)
    params.setdefault("loss", "huber")
    params.setdefault("alpha", 0.9)
    params.setdefault("subsample", 1.0)

    return GradientBoostingRegressor(
        **params,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )


# ─────────────────────── Hyper‑parameter optimisation ──────────────────────

def run_optuna(X, y, n_trials: int, timeout: int | None, seed: int):
    if optuna is None:
        raise RuntimeError("Optuna is not installed – run `pip install optuna` or set --n_trials 0")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 2000, step=100),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100),
            loss=trial.suggest_categorical(
                "loss", ["squared_error", "absolute_error", "huber"]
            ),
            alpha=trial.suggest_float("alpha", 0.7, 0.95),
        )

        model = build_gbm(params, seed)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return mean_absolute_error(y_val, pred)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    print("\nBest Optuna params (val MAE {:.4f}):".format(study.best_value))
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


# ───────────────────────────── Bias table ──────────────────────────────────

def build_bias_table(X, residuals):
    def cell(feat_row):
        days = int(feat_row[0])
        mpd = feat_row[3]
        spd = feat_row[4]
        return (min(days, 9), int(mpd // 50), int(spd // 50))

    bucket: defaultdict[tuple[int, int, int], list[float]] = defaultdict(list)
    for f, r in zip(X, residuals):
        bucket[cell(f)].append(r)
    return {k: float(sum(v) / len(v)) for k, v in bucket.items()}


# ───────────────────────────── Main entry‑point ────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path, help="public_cases.json path")
    p.add_argument("--n_trials", type=int, default=50, help="Optuna trials (0 = skip)")
    p.add_argument("--timeout", type=int, default=None, help="Optuna timeout seconds")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    args = p.parse_args()

    X, y = load_dataset(args.json_path)

    # — Hyper‑parameter search —
    params = {}
    if args.n_trials > 0:
        params = run_optuna(X, y, args.n_trials, args.timeout, args.seed)
    else:
        print("Skipping hyper‑parameter optimisation (use --n_trials > 0 to enable).")

    model = build_gbm(params, args.seed).fit(X, y)
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    print(f"Public‑set MAE after GBM: {mae:.3f}")

    # — Bias corrections —
    residuals = y - pred
    bias_table = build_bias_table(X, residuals)

    arte_adj = float(np.mean([r for f, r in zip(X, residuals) if int(round(f[2] * 100)) % 50 == 49]))
    five_adj = float(np.mean([r for f, r in zip(X, residuals) if int(f[0]) == 5]))
    long_adj = float(np.mean([r for f, r in zip(X, residuals) if int(f[0]) >= 12]))

    mae_bias = float(
        np.mean(
            np.abs(
                pred + np.array([bias_table[(min(int(f[0]), 9), int(f[3] // 50), int(f[4] // 50))] for f in X]) - y
            )
        )
    )
    print(f"MAE after bias table  : {mae_bias:.3f}")

    # — Export GBM trees —
    trees_out = []
    for est in model.estimators_.ravel():
        t = est.tree_
        nodes = []
        for i in range(t.node_count):
            n = {"v": float(t.value[i][0][0])}
            if t.children_left[i] != -1:
                n.update(
                    f=int(t.feature[i]),
                    t=float(t.threshold[i]),
                    l=int(t.children_left[i]),
                    r=int(t.children_right[i]),
                )
            nodes.append(n)
        trees_out.append(nodes)

    gbm_bias = float(model.init_.constant_[0][0])
    lr_used = model.learning_rate

    # — Print paste‑ready block —
    print("\n### Paste the block below into calculate_reimbursement.py ###\n")
    print(f"LEARNING_RATE = {lr_used}")
    print(f"GBM_BIAS = {gbm_bias:.6f}")
    print("GBM_TREES = [")
    for tr in trees_out:
        print(" [")
        for n in tr:
            print("  ", n, ",")
        print(" ],")
    print("]\n")

    print("BIAS_TABLE = {")
    for k, v in bias_table.items():
        print(f" {k}: {round(v, 4)},")
    print("}\n")

    print(f"ARTE_ADJ = {round(arte_adj, 4)}")
    print(f"FIVE_ADJ = {round(five_adj, 4)}")
    print(f"LONG_ADJ = {round(long_adj, 4)}")


if __name__ == "__main__":
    main()
