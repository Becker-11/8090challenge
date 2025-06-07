#!/usr/bin/env python3
"""
tune_cfg.py  –  lightweight, dependency‑free fitter for ACME heuristic
"""

import json, random, math, sys
from decimal import Decimal, ROUND_HALF_UP as RHU
random.seed(0)

# ────────────── CLI & data ────────────────────────────────────────────────
if len(sys.argv) not in (2, 3):
    sys.exit("Usage: tune_cfg.py public_cases.json [N_ITER]")
cases_file = sys.argv[1]
N_ITER = int(sys.argv[2]) if len(sys.argv) == 3 else 20_000

raw = json.load(open(cases_file))
CASES = [(
    int(c["input"]["trip_duration_days"]),
    Decimal(c["input"]["miles_traveled"]),
    Decimal(str(c["input"]["total_receipts_amount"])),
    Decimal(str(c["expected_output"])),
) for c in raw]

# ─────────── parameter helpers ────────────────────────────────────────────
NAME_ORDER = [
    # mileage rates
    "rate_first", "rate_mid", "rate_long",
    # efficiency
    "mpd_low", "mpd_mid", "mpd_peak",
    "bonus_peak", "bonus_mid", "pen_slow", "pen_fast",
    # spend thresholds
    "short_lo", "short_hi", "medium_hi", "long_hi",
    # spend multipliers
    "bonus_short", "bonus_med", "bonus_long",
    "pen_small", "pen_med_hi", "pen_long_hi",
    # artefact
    "arte_pct",
]

DEFAULTS = {
    "rate_first": 0.58, "rate_mid": 0.401, "rate_long": 0.299,
    "mpd_low": 100, "mpd_mid": 150, "mpd_peak": 220,
    "bonus_peak": 0.10, "bonus_mid": 0.05,
    "pen_slow": -0.05, "pen_fast": -0.03,
    "short_lo": 25, "short_hi": 75, "medium_hi": 120, "long_hi": 90,
    "bonus_short": 0.30, "bonus_med": 0.20, "bonus_long": 0.15,
    "pen_small": -0.20, "pen_med_hi": -0.10, "pen_long_hi": -0.20,
    "arte_pct": 0.015,
}

def vec_to_cfg(vec):
    return {k: Decimal(str(v)) for k, v in zip(NAME_ORDER, vec)}

def cfg_to_vec(cfg):
    return [float(cfg[k]) for k in NAME_ORDER]

# ─────────── reimbursement calc (Decimal all the way) ────────────────────
def mileage_pay(miles, cfg):
    if miles <= 100:
        return miles * cfg["rate_first"]
    if miles <= 500:
        return Decimal(100) * cfg["rate_first"] + (miles - 100) * cfg["rate_mid"]
    return (Decimal(100) * cfg["rate_first"]
            + Decimal(400) * cfg["rate_mid"]
            + (miles - 500) * cfg["rate_long"])

def compute(case, vec, cache):
    key = tuple(vec)
    cfg = cache.setdefault(key, vec_to_cfg(vec))

    days, miles, receipts = case
    base     = Decimal(days) * Decimal(100)
    mileage  = mileage_pay(miles, cfg)
    gross    = base + mileage
    mpd      = miles / days

    # efficiency
    if cfg["mpd_mid"] <= mpd <= cfg["mpd_peak"]:
        eff = gross * cfg["bonus_peak"]
    elif cfg["mpd_low"] <= mpd < cfg["mpd_mid"]:
        eff = gross * cfg["bonus_mid"]
    elif mpd < cfg["mpd_low"]:
        eff = gross * cfg["pen_slow"]
    else:
        eff = gross * cfg["pen_fast"]

    # spend
    rpd = receipts / days
    if days <= 3:
        if rpd <= cfg["short_lo"]:
            sp = receipts * cfg["pen_small"]
        elif rpd <= cfg["short_hi"]:
            sp = receipts * cfg["bonus_short"]
        else:
            sp = receipts * Decimal("0.05")
    elif 4 <= days <= 6:
        sp = receipts * (cfg["bonus_med"]
                         if rpd <= cfg["medium_hi"] else cfg["pen_med_hi"])
    else:
        sp = receipts * (cfg["bonus_long"]
                         if rpd <= cfg["long_hi"] else cfg["pen_long_hi"])

    arte = receipts * cfg["arte_pct"] if int((receipts * 100) % 10) == 9 else Decimal(0)
    total = (base + mileage + eff + sp + arte).quantize(Decimal("0.01"), rounding=RHU)
    return total

def mae(vec, cache):
    err = Decimal(0)
    for case in CASES:
        err += abs(compute(case[:3], vec, cache) - case[3])
    return float(err / len(CASES))

# ─────────── optimiser: random local search ──────────────────────────────
vec   = cfg_to_vec(DEFAULTS)
cache = {tuple(vec): vec_to_cfg(vec)}
best  = mae(vec, cache)
print(f"Initial MAE: {best:.4f}")

step = 0.02
for _ in range(N_ITER):
    i = random.randrange(len(vec))
    new_vec = vec.copy()
    new_vec[i] *= math.exp(random.uniform(-step, step))
    new_err = mae(new_vec, cache)
    if new_err < best:
        vec, best = new_vec, new_err
        step *= 0.999          # slow annealing

print(f"Final MAE:   {best:.4f}\n")

# ─────────── emit tuned CFG block ────────────────────────────────────────
cfg = vec_to_cfg(vec)
print("### Paste into calculate_reimbursement.py ###\nCFG = {")
for k in NAME_ORDER:
    print(f'    "{k}": Decimal("{cfg[k]:.6f}"),')
print("}")
