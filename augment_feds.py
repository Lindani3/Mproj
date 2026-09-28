"""
augment_feds.py
===============
Generate feds_3000.csv from feds_1000.csv by adding two parallel-shifted
copies of each curve: +1 basis point and -1 basis point applied to BETA0.

A parallel shift of the Svensson yield curve by delta is achieved by adding
delta to BETA0, since BETA0 enters the yield formula as a constant term
independent of maturity:

    y(t) = BETA0 + BETA1*f(t/tau1) + BETA2*g(t/tau1) + BETA3*g(t/tau2)

Adding delta to BETA0 shifts y(t) by exactly delta for all t.

The resulting curves are arbitrage-free: they are valid initial term
structures for the Hull-White model, which accommodates any smooth positive
curve through the theta(t) calibration function.

Output
------
    data/feds_3000.csv   — 3,000 rows (1,000 original + 1,000 up + 1,000 down)
    Compatible with load_all_svensson_params() in hw_utils.py.
"""

import argparse
import os

import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augment FEDS curves by ±1bp parallel shift")
    p.add_argument("--feds", type=str,
                   default=os.path.join(CODE_DIR, "data", "feds_1000.csv"))
    p.add_argument("--out",  type=str,
                   default=os.path.join(CODE_DIR, "data", "feds_3000.csv"))
    p.add_argument("--shift_bps", type=float, default=1.0,
                   help="Shift size in basis points (default: 1.0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shift_pct = args.shift_bps / 100.0   # convert bps to percent (CSV units)

    df = pd.read_csv(args.feds, skiprows=9)
    df = df.dropna(subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
    df = df[["Date", "BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]].copy()

    n = len(df)
    print(f"Loaded {n:,} curves from {args.feds}")

    df_up   = df.copy()
    df_down = df.copy()

    df_up["BETA0"]   = df["BETA0"] + shift_pct
    df_down["BETA0"] = df["BETA0"] - shift_pct

    df_up["Date"]   = df["Date"].astype(str) + "_up"
    df_down["Date"] = df["Date"].astype(str) + "_dn"

    df_all = pd.concat([df, df_up, df_down], ignore_index=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Write 9 placeholder header lines so load_all_svensson_params() (skiprows=9) works
    with open(args.out, "w") as f:
        for _ in range(9):
            f.write("# augmented FEDS curves\n")
    df_all.to_csv(args.out, mode="a", index=False)

    print(f"Saved {len(df_all):,} curves to {args.out}")
    print(f"  Original : {n:,}")
    print(f"  +{args.shift_bps:.0f}bp     : {n:,}")
    print(f"  -{args.shift_bps:.0f}bp     : {n:,}")


if __name__ == "__main__":
    main()
