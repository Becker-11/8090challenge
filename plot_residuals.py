#!/usr/bin/env python3
"""
plot_residuals.py – Visual diagnostics for ACME reimbursement model.

This script **imports your populated `calculate_reimbursement.py`** (the one
with the pasted export block) to guarantee the same feature logic and model
parameters.  It then plots:

1. Scatter plots of residuals vs. key raw features.
2. 2‑D binned heatmaps (days×receipts and days×miles).

The figures are saved to a multi‑page PDF (default `residual_diagnostics.pdf`).

Usage
-----
    python plot_residuals.py public_cases.json [-o out.pdf]
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import calculate_reimbursement as calc  # MUST have export block pasted!

# ────────────────────────────── Helpers ────────────────────────────────────

def predict(days, miles, receipts):
    feats = calc.make_features(days, miles, receipts)
    base  = calc.gbm_predict(feats)
    bias_adj = calc.BIAS_TABLE.get(calc.cell(feats), 0.0)
    if int(round(receipts * 100)) % 50 == 49:
        bias_adj += calc.ARTE_ADJ
    if int(days) == 5:
        bias_adj += calc.FIVE_ADJ
    if int(days) >= 12:
        bias_adj += calc.LONG_ADJ
    return base + bias_adj

# ───────────────────────────────── Main ────────────────────────────────────

def default_cell(feat_row):
    """Fallback cell implementation (days≤9, mpd & spd buckets of 50)."""
    days = int(feat_row[0])
    mpd  = feat_row[3]
    spd  = feat_row[4]
    return (min(days, 9), int(mpd // 50), int(spd // 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("residual_diagnostics.pdf"))
    args = ap.parse_args()

    # Load test cases
    raw = json.load(open(args.json_path))
    days   = np.array([c["input"]["trip_duration_days"] for c in raw], float)
    miles  = np.array([c["input"]["miles_traveled"]     for c in raw], float)
    rec    = np.array([c["input"]["total_receipts_amount"] for c in raw], float)
    y_true = np.array([c["expected_output"]              for c in raw], float)

    # Use calc.cell if present, else fallback
    cell_fn = getattr(calc, "cell", default_cell)

    def predict_vectorized():
        preds = []
        for d, m, r in zip(days, miles, rec):
            feats = calc.make_features(d, m, r)
            base  = calc.gbm_predict(feats)
            bias_adj = calc.BIAS_TABLE.get(cell_fn(feats), 0.0)
            if int(round(r * 100)) % 50 == 49:
                bias_adj += calc.ARTE_ADJ
            if int(d) == 5:
                bias_adj += calc.FIVE_ADJ
            if int(d) >= 12:
                bias_adj += calc.LONG_ADJ
            preds.append(base + bias_adj)
        return np.array(preds)

    y_pred = predict_vectorized()
    resid  = y_true - y_pred

    with PdfPages(args.out) as pdf:
        # Scatter plots
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs = axs.ravel()
        for ax, x, lbl in zip(
            axs,
            [days, miles, rec, rec / days],
            ["Trip duration (days)", "Miles traveled", "Total receipts ($)", "Receipts per day ($)"]
        ):
            ax.scatter(x, resid, alpha=0.3, s=8)
            ax.axhline(0, color="k", lw=0.7)
            ax.set_xlabel(lbl)
            ax.set_ylabel("Residual (true – pred)")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # 2‑D heatmaps
        for x, y, xlabel, ylabel in [
            (days, rec, "Trip duration (days)", "Total receipts ($)"),
            (days, miles, "Trip duration (days)", "Miles traveled"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 5))
            vmax = np.max(np.abs(resid))
            hb = ax.hexbin(x, y, C=resid, gridsize=40, cmap="RdBu", vmin=-vmax, vmax=vmax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("Residual (true – pred)")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved residual diagnostics → {args.out}")

if __name__ == "__main__":
    main()
