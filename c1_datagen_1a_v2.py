"""
c1_datagen_1a_v2.py
===================
Generate training data for Model 1a v2 (sequence-to-sequence).

Following Frode et al. (2022), each sample covers the full monitoring-date
profile in a single record:

    Input : (P(0,T_1),...,P(0,T_20), a, sigma)  in R^22
    Output: (y_0, y_1, ..., y_20)                in R^21

where y_k = E^Q[V_IRS(t_k)] computed analytically via the MGF.

Grid
----
  1,000 FEDS curves  x  8 a-values  x  8 sigma-values
  = 64,000 samples  (each sample carries 21 labels)

Output
------
  data/train_1a_v2.h5  keys: X_disc, X_scalar, y_price
    X_disc   (64000, 20)  initial discount factors
    X_scalar (64000,  2)  [a, sigma]
    y_price  (64000, 21)  full analytical label profile
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    T_MONITOR, T_PAY, TAU,
    hw_bond_price,
    load_all_svensson_params,
    market_discount, par_swap_rate, svensson_forward,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "data", "feds_1000.csv")

A_GRID     = np.linspace(0.01, 0.30, 8)
SIGMA_GRID = np.linspace(0.005, 0.030, 8)
N_A        = len(A_GRID)
N_SIG      = len(SIGMA_GRID)
N_MON      = len(T_MONITOR)   # 21
CHUNK      = 4096


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model 1a v2 data generation — analytical labels")
    p.add_argument("--feds", type=str, default=FEDS_FILE)
    p.add_argument("--out",  type=str,
                   default=os.path.join(CODE_DIR, "data", "train_1a_v2.h5"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def analytical_profile(K, p, a, sigma):
    """
    Compute the full 21-element analytical label profile for one
    (curve, a, sigma) combination using the MGF of r_{t_k}.
    Returns y_vec of shape (21,).
    """
    y_vec = np.zeros(N_MON, dtype=np.float32)
    for ki, t_k in enumerate(T_MONITOR):
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            y_vec[ki] = 0.0
            continue
        f0t = float(svensson_forward(t_k, p))
        mu  = f0t + (sigma**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * t_k))**2
        s2  = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t_k))
        rem = T_PAY[T_PAY > t_k + 1e-9]
        # E[P(t_k, T_j)] via MGF
        B   = (1.0 - np.exp(-a * (rem - t_k))) / a
        # A coefficients via hw_bond_price evaluated at mu (proxy scalar)
        # Use vectorised hw_bond_price with a single "path"
        ep  = hw_bond_price(t_k, rem, np.array([mu]), p, a, sigma)  # (1, n_rem)
        # Correct MGF: A * exp(-B*mu + 0.5*B^2*s2)
        A_coef = ep[0] * np.exp(B * mu)   # recover A = E[P] * exp(B*mu) at s2=0 approx
        # Exact MGF: E[P] = A * exp(-B*mu + 0.5*B^2*s2)
        exp_P = A_coef * np.exp(-B * mu + 0.5 * B**2 * s2)
        y_vec[ki] = float((1.0 - exp_P[-1]) - K * TAU * exp_P.sum())
    return y_vec


def main() -> None:
    args   = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    all_params = load_all_svensson_params(args.feds)
    n_feds     = len(all_params)
    total      = n_feds * N_A * N_SIG

    print(f"FEDS curves  : {n_feds:,}")
    print(f"a  values    : {N_A}  ({A_GRID[0]:.3f} to {A_GRID[-1]:.3f})")
    print(f"sig values   : {N_SIG}  ({SIGMA_GRID[0]:.4f} to {SIGMA_GRID[-1]:.4f})")
    print(f"Labels/sample: {N_MON}  (t_k = 0.0 to 10.0 semi-annual)")
    print(f"Total samples: {total:,}")
    print(f"Output       : {args.out}")

    with h5py.File(args.out, "w") as hf:
        ds_disc   = hf.create_dataset("X_disc",   shape=(0, 20),
                                       maxshape=(None, 20),    dtype="float32",
                                       chunks=(CHUNK, 20),   compression="gzip")
        ds_scalar = hf.create_dataset("X_scalar", shape=(0, 2),
                                       maxshape=(None, 2),     dtype="float32",
                                       chunks=(CHUNK, 2),    compression="gzip")
        ds_price  = hf.create_dataset("y_price",  shape=(0, N_MON),
                                       maxshape=(None, N_MON), dtype="float32",
                                       chunks=(CHUNK, N_MON), compression="gzip")

        offset = 0
        t0     = time.time()

        for fi, p in enumerate(all_params):
            K       = par_swap_rate(p)
            d_knots = market_discount(T_PAY, p).astype(np.float32)

            batch_disc   = []
            batch_scalar = []
            batch_price  = []

            for a in A_GRID:
                for sigma in SIGMA_GRID:
                    y_vec = analytical_profile(K, p, a, sigma)
                    batch_disc.append(d_knots)
                    batch_scalar.append(np.array([a, sigma], dtype=np.float32))
                    batch_price.append(y_vec)

            n = len(batch_price)
            ds_disc.resize(offset + n, axis=0)
            ds_scalar.resize(offset + n, axis=0)
            ds_price.resize(offset + n, axis=0)

            ds_disc[offset:offset + n]   = np.stack(batch_disc)
            ds_scalar[offset:offset + n] = np.stack(batch_scalar)
            ds_price[offset:offset + n]  = np.stack(batch_price)

            offset += n

            if (fi + 1) % 100 == 0 or fi == 0 or fi == n_feds - 1:
                elapsed = time.time() - t0
                rate    = (fi + 1) / max(elapsed, 1)
                eta     = (n_feds - fi - 1) / max(rate, 1)
                print(f"  [{fi+1:>5}/{n_feds}]  samples={offset:>8,}  "
                      f"elapsed={elapsed:>6.0f}s  eta={eta:>6.0f}s")

    print(f"\nSaved {offset:,} samples to {args.out}")


if __name__ == "__main__":
    main()
