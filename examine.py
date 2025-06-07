#!/usr/bin/env python3
"""
Visualize reimbursement patterns from a JSON file that stores records like
{
  "input": {
    "trip_duration_days": 3,
    "miles_traveled": 399,
    "total_receipts_amount": 141.39
  },
  "expected_output": 546.04
}

Usage
-----
    python visualize_expense_rules.py path/to/expense_rules.json
"""

import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3‑D)
from sklearn.linear_model import LinearRegression


def main(path):
    # ── 1  Load JSON ────────────────────────────────────────────────────────
    with open(path, "r", encoding="utf‑8") as f:
        records = json.load(f)

    # Flatten “input.*” into columns & rename the target
    df = (pd.json_normalize(records)
            .rename(columns={"expected_output": "reimbursement"}))

    # Keep tidy column names (drop “input.” prefix)
    df.columns = [c.split(".")[-1] for c in df.columns]

    # ── 2  2‑D scatter plots ───────────────────────────────────────────────
    x_cols = ["trip_duration_days", "miles_traveled", "total_receipts_amount"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, col in zip(axes, x_cols):
        ax.scatter(df[col], df["reimbursement"])
        ax.set_xlabel(col.replace("_", " "))
    axes[0].set_ylabel("reimbursement")
    fig.suptitle("Reimbursement vs. each individual input")
    fig.tight_layout()

    # ── 3  3‑D scatter with colour = reimbursement ─────────────────────────
    fig3d = plt.figure(figsize=(6, 5))
    ax3d = fig3d.add_subplot(projection="3d")
    sc   = ax3d.scatter(df["trip_duration_days"],
                        df["miles_traveled"],
                        df["total_receipts_amount"],
                        c=df["reimbursement"],
                        cmap="viridis")
    ax3d.set_xlabel("trip duration (days)")
    ax3d.set_ylabel("miles traveled")
    ax3d.set_zlabel("total receipts ($)")
    fig3d.colorbar(sc, label="reimbursement")
    fig3d.suptitle("3‑D view of inputs (colour = reimbursement)")

    # ── 4  Correlations & simple linear model ──────────────────────────────
    print("\nCorrelation with reimbursement:")
    print(df[x_cols + ["reimbursement"]].corr(numeric_only=True)["reimbursement"])

    X, y = df[x_cols], df["reimbursement"]
    model = LinearRegression().fit(X, y)
    print("\nSimple linear‑regression fit")
    print("  R² =", model.score(X, y))
    for name, coef in zip(x_cols, model.coef_):
        print(f"  {name} coefficient = {coef:.3f}")
    print("  intercept          =", round(model.intercept_, 3))

    # ── 5  Show all figures ────────────────────────────────────────────────
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_expense_rules.py <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
