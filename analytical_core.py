# constants solved once (fit on low‑noise rows)
R0,R1,R2   = 0.58, 0.401, 0.299     # mileage $
BETA5      = 0.09                   # +9 %
BETA_LONG  = -0.12                  # –12 %
GAMMA49    = 0.015                  # +1.5 % of receipts

def eff_ratio(mpd):
    if mpd < 100:   return -0.05
    if mpd < 150:   return +0.05
    if mpd < 220:   return +0.10
    if mpd < 300:   return -0.03
    return -0.05

def spend_ratio(spd, days):
    if days <= 3:
        return 0.30 if spd <= 75 else 0.05
    if 4 <= days <= 6:
        return 0.20 if spd <= 120 else -0.10
    # long
    return 0.15 if spd <= 90 else -0.20

def analytic_pred(days, miles, receipts):
    base = 100 * days
    mileage = (R0*min(miles,100) +
               R1*max(0, min(miles,500)-100) +
               R2*max(0, miles-500))
    subtotal = base + mileage

    subtotal *= 1 + eff_ratio(miles/days)
    subtotal += receipts * spend_ratio(receipts/days, days)

    if days == 5:
        subtotal *= 1 + BETA5
    if days >= 12:
        subtotal *= 1 + BETA_LONG
    if int(round(receipts*100))%50 == 49:
        subtotal += GAMMA49 * receipts

    return subtotal
