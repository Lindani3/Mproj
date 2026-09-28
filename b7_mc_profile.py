"""
b7_mc_profile.py
================
Monte Carlo cashflow profile for Model 1b — one curve, M=200 paths per date.
Prints t_k, V_fixed, V_float, diff for comparison against b6 analytical values.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    T_MONITOR, T_PAY, TAU,
    hw_bond_price,
    load_all_svensson_params,
    market_discount, par_swap_rate, svensson_forward,
)

_code_dir = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = (
    os.path.join(_code_dir, "data", "feds_1000.csv")
    if os.path.exists(os.path.join(_code_dir, "data", "feds_1000.csv"))
    else os.path.join(_code_dir, "feds200628.csv")
)
A         = 0.10
SIGMA     = 0.015
M         = 200
SEED      = 42


def main():
    rng        = np.random.default_rng(SEED)
    all_params = load_all_svensson_params(FEDS_FILE)
    p          = all_params[0]
    K          = par_swap_rate(p)

    print(f"Curve : {p.get('date', 'unknown')}   K={K*100:.4f}%")
    print(f"HW    : a={A},  sigma={SIGMA},  M={M} paths\n")

    rows = []
    for t_k in T_MONITOR:
        rem = T_PAY[T_PAY > t_k + 1e-9]
        if len(rem) == 0 or t_k < 1e-8:
            rows.append({"time": t_k, "V_fixed": 0.0, "V_float": 0.0, "diff": 0.0})
            continue

        f0t = float(svensson_forward(t_k, p))
        mu  = f0t + (SIGMA**2 / (2*A**2)) * (1 - np.exp(-A*t_k))**2
        s   = np.sqrt((SIGMA**2 / (2*A)) * (1 - np.exp(-2*A*t_k)))

        r_samples = rng.normal(mu, s, size=M)
        bonds     = hw_bond_price(t_k, rem, r_samples, p, A, SIGMA)  # (M, n_rem)

        V_float = float((1.0 - bonds[:, -1]).mean())
        V_fixed = float((K * TAU * bonds.sum(axis=1)).mean())
        diff    = V_float - V_fixed

        rows.append({"time": t_k, "V_fixed": V_fixed, "V_float": V_float, "diff": diff})

    df  = pd.DataFrame(rows)
    hdr = f"{'time':>6}  {'V_fixed':>14}  {'V_float':>14}  {'diff':>14}"
    print(hdr)
    print("-" * len(hdr))
    for _, row in df.iterrows():
        print(f"{row['time']:>6.1f}  {row['V_fixed']:>14.8f}  "
              f"{row['V_float']:>14.8f}  {row['diff']:>14.8f}")

    df.to_csv("data/mc_profile.csv", index=False, float_format="%.8f")
    print("\nSaved: data/mc_profile.csv")


if __name__ == "__main__":
    main()
