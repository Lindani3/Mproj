"""
c1_datagen_1a.py
================
Generate training data for Model 1a (analytical expected IRS value).

For each combination of (FEDS date, a, sigma, t_k):
    label = E[V_affine(t_k)]  -- analytical via MGF of r_{t_k} ~ N(mu, s^2)

No simulation. Boundary values (t_k=0, t_k=10) are exactly zero.

Grid
----
  16,160 FEDS curves  x  8 a-values  x  8 sigma-values  x  21 t_k
  = ~21.7 M samples

Inputs
------
  X_disc   (N, 20)  P(0, T_j) at T_PAY = [0.5, 1.0, ..., 10.0]
  X_scalar (N,  3)  [a, sigma, t_k]

Output
------
  data/train_1a.h5  keys: X_disc, X_scalar, y_price
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    T_MONITOR, T_PAY,
    irs_expected_affine, load_all_svensson_params,
    market_discount, par_swap_rate, svensson_forward,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "feds200628.csv")

A_GRID     = np.linspace(0.01, 0.30, 8)
SIGMA_GRID = np.linspace(0.005, 0.030, 8)
N_A        = len(A_GRID)
N_SIG      = len(SIGMA_GRID)
N_MON      = len(T_MONITOR)                  # 21
BATCH      = N_A * N_SIG * N_MON             # 1344 per FEDS date
CHUNK      = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model 1a data generation — analytical labels")
    p.add_argument("--feds", type=str, default=FEDS_FILE)
    p.add_argument("--out",  type=str,
                   default=os.path.join(CODE_DIR, "data", "train_1a.h5"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    all_params = load_all_svensson_params(args.feds)
    n_feds     = len(all_params)
    total      = n_feds * BATCH

    print(f"FEDS curves   : {n_feds:,}")
    print(f"a  values     : {N_A}  ({A_GRID[0]:.3f} to {A_GRID[-1]:.3f})")
    print(f"sig values    : {N_SIG}  ({SIGMA_GRID[0]:.4f} to {SIGMA_GRID[-1]:.4f})")
    print(f"t_k dates     : {N_MON}  (0.0 to 10.0 semi-annual)")
    print(f"Total samples : {total:,}")
    print(f"Output        : {args.out}")

    # Pre-build (a, sigma, t_k) scalar combinations — same for every FEDS date
    a_rep   = np.repeat(np.repeat(A_GRID,     N_SIG), N_MON)
    sig_rep = np.tile(np.repeat(SIGMA_GRID, N_MON), N_A)
    tk_rep  = np.tile(T_MONITOR, N_A * N_SIG)
    scalar_combos = np.column_stack([a_rep, sig_rep, tk_rep]).astype(np.float32)  # (1344, 3)

    with h5py.File(args.out, "w") as hf:
        ds_disc   = hf.create_dataset("X_disc",   shape=(0, 20),
                                       maxshape=(None, 20), dtype="float32",
                                       chunks=(CHUNK, 20), compression="gzip")
        ds_scalar = hf.create_dataset("X_scalar", shape=(0, 3),
                                       maxshape=(None, 3),  dtype="float32",
                                       chunks=(CHUNK, 3),  compression="gzip")
        ds_price  = hf.create_dataset("y_price",  shape=(0,),
                                       maxshape=(None,),    dtype="float32",
                                       chunks=(CHUNK,),    compression="gzip")

        offset = 0
        t0     = time.time()

        for fi, p in enumerate(all_params):
            K       = par_swap_rate(p)
            d_knots = market_discount(T_PAY, p).astype(np.float32)   # (20,)

            # Analytical labels for all 1344 (a, sigma, t_k) combinations
            y_batch = np.empty(BATCH, dtype=np.float32)
            idx = 0
            for a in A_GRID:
                for sigma in SIGMA_GRID:
                    for t_k in T_MONITOR:
                        y_batch[idx] = irs_expected_affine(t_k, K, p, a, sigma)
                        idx += 1

            X_disc_batch = np.tile(d_knots, (BATCH, 1))   # (1344, 20)

            # Resize and write
            ds_disc.resize(offset + BATCH, axis=0)
            ds_scalar.resize(offset + BATCH, axis=0)
            ds_price.resize(offset + BATCH, axis=0)

            ds_disc[offset:offset + BATCH]   = X_disc_batch
            ds_scalar[offset:offset + BATCH] = scalar_combos
            ds_price[offset:offset + BATCH]  = y_batch

            offset += BATCH

            if (fi + 1) % 1000 == 0 or fi == 0 or fi == n_feds - 1:
                elapsed = time.time() - t0
                rate    = offset / max(elapsed, 1)
                eta     = (total - offset) / max(rate, 1)
                print(f"  [{fi+1:>5}/{n_feds}]  samples={offset:>12,}  "
                      f"elapsed={elapsed:>6.0f}s  eta={eta:>6.0f}s")

    print(f"\nSaved {offset:,} samples to {args.out}")


if __name__ == "__main__":
    main()
