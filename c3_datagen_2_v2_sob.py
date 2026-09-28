"""
c3_datagen_2_v2_sob.py
======================
Sobolev-augmented data generation for Model 2 (EPE labels, MC).

EPE(t_k) = E^Q[max(V_IRS(t_k), 0)] computed via MC short-rate draws.
Scalar Sobolev labels (dJ_a, dJ_sig) via central finite differences.
Yield-curve Jacobian omitted (too expensive with MC).

Output HDF5 keys
----------------
  X_disc    (N, 20)      initial discount factors
  X_scalar  (N,  2)      [a, sigma]
  y_price   (N, 21)      MC EPE profile labels
  dJ_scalar (N, 21,  2)  [d(EPE_k)/da, d(EPE_k)/dsig] via central diff
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
N_MON      = len(T_MONITOR)
N_PAY      = len(T_PAY)
CHUNK      = 256

EPS_A   = 1e-4
EPS_SIG = 1e-5


def _epe_profile(p, a_val, sig_val, K, M, rng):
    """21-element EPE profile for one (curve, a, sigma)."""
    y_vec = np.zeros(N_MON, dtype=np.float32)
    for ki, t_k in enumerate(T_MONITOR):
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            continue
        f0t       = float(svensson_forward(t_k, p))
        mu        = f0t + (sig_val**2 / (2.0 * a_val**2)) * (1.0 - np.exp(-a_val * t_k))**2
        s         = np.sqrt((sig_val**2 / (2.0 * a_val)) * (1.0 - np.exp(-2.0 * a_val * t_k)))
        r_samples = rng.normal(mu, s, size=M)
        rem       = T_PAY[T_PAY > t_k + 1e-9]
        bonds     = hw_bond_price(t_k, rem, r_samples, p, a_val, sig_val)
        V         = (1.0 - bonds[:, -1]) - K * TAU * bonds.sum(axis=1)
        y_vec[ki] = float(np.maximum(V, 0.0).mean())
    return y_vec


def parse_args():
    p = argparse.ArgumentParser(description="Model 2 v2 Sobolev data generation")
    p.add_argument("--feds",    type=str, default=FEDS_FILE)
    p.add_argument("--out",     type=str,
                   default=os.path.join(CODE_DIR, "data", "train_2_v2_sob.h5"))
    p.add_argument("--m_paths", type=int, default=5000)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    all_params = load_all_svensson_params(args.feds)
    N_CURVES   = len(all_params)
    N_TOTAL    = N_CURVES * len(A_GRID) * len(SIGMA_GRID)
    M          = args.m_paths

    print(f"FEDS curves   : {N_CURVES:,}")
    print(f"MC paths      : {M:,}")
    print(f"Total samples : {N_TOTAL:,}")
    print(f"Output        : {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with h5py.File(args.out, "w") as hf:
        ds_xd = hf.create_dataset("X_disc",    (N_TOTAL, N_PAY),    dtype="f4", chunks=(CHUNK, N_PAY))
        ds_xs = hf.create_dataset("X_scalar",  (N_TOTAL, 2),        dtype="f4", chunks=(CHUNK, 2))
        ds_y  = hf.create_dataset("y_price",   (N_TOTAL, N_MON),    dtype="f4", chunks=(CHUNK, N_MON))
        ds_js = hf.create_dataset("dJ_scalar", (N_TOTAL, N_MON, 2), dtype="f4", chunks=(CHUNK, N_MON, 2))

        idx = 0
        t0  = time.time()

        for ci, p in enumerate(all_params):
            disc_np = market_discount(T_PAY, p).astype(np.float32)
            K       = par_swap_rate(p)

            for a_val in A_GRID:
                for sig_val in SIGMA_GRID:
                    y_base = _epe_profile(p, a_val,         sig_val,           K, M, rng)
                    y_ap   = _epe_profile(p, a_val + EPS_A, sig_val,           K, M, rng)
                    y_am   = _epe_profile(p, a_val - EPS_A, sig_val,           K, M, rng)
                    y_sp   = _epe_profile(p, a_val,         sig_val + EPS_SIG, K, M, rng)
                    y_sm   = _epe_profile(p, a_val,         sig_val - EPS_SIG, K, M, rng)

                    dJ_a   = (y_ap - y_am) / (2.0 * EPS_A)
                    dJ_sig = (y_sp - y_sm) / (2.0 * EPS_SIG)
                    dJ_s   = np.stack([dJ_a, dJ_sig], axis=-1).astype(np.float32)

                    ds_xd[idx] = disc_np
                    ds_xs[idx] = np.array([a_val, sig_val], dtype=np.float32)
                    ds_y[idx]  = y_base
                    ds_js[idx] = dJ_s
                    idx += 1

            if (ci + 1) % 10 == 0 or ci == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (ci + 1) * (N_CURVES - ci - 1)
                print(f"  [{ci+1:5d}/{N_CURVES}]  samples={idx:>10,}  elapsed={elapsed:6.0f}s  eta={eta:6.0f}s")

    print(f"\nSaved {N_TOTAL:,} samples to {args.out}")


if __name__ == "__main__":
    main()
