"""
c1_datagen_1a_v2_sob.py
=======================
Sobolev-augmented data generation for Model 1a (analytical MGF labels).

Extends c1_datagen_1a_v2.py by computing the full input-output Jacobian
for each sample via PyTorch autograd, enabling Sobolev (differential) training.

Grid
----
  N_curves FEDS curves  x  8 a-values  x  8 sigma-values

Output HDF5 keys
----------------
  X_disc    (N, 20)       initial discount factors  P(0,T_j)
  X_scalar  (N,  2)       [a, sigma]
  y_price   (N, 21)       analytical profile  E[V_IRS(t_k)]
  dJ_disc   (N, 21, 20)   d(y_k)/d(P(0,T_j))  via autograd
  dJ_scalar (N, 21,  2)   [d(y_k)/da, d(y_k)/dsigma]  via autograd
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    T_MONITOR, T_PAY, TAU,
    load_all_svensson_params,
    market_discount, par_swap_rate, svensson_forward,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "data", "feds_1000.csv")

A_GRID     = np.linspace(0.01, 0.30, 8)
SIGMA_GRID = np.linspace(0.005, 0.030, 8)
N_MON      = len(T_MONITOR)   # 21
N_PAY      = len(T_PAY)       # 20
TAU_T      = float(TAU)
CHUNK      = 256

# Precompute monitoring-date index into T_PAY
# T_MONITOR[ki] == T_PAY[ki-1] for ki = 1..19  (0 and 20 are boundary)
_TK_TO_DISC_IDX = {}
for ki, tk in enumerate(T_MONITOR):
    for ji, tj in enumerate(T_PAY):
        if abs(tk - tj) < 1e-9:
            _TK_TO_DISC_IDX[ki] = ji
            break


def _profile_and_jacobian(disc_np, a_val, sig_val, f0_tks):
    """
    Compute the 21-output price profile and its full Jacobian via autograd.

    Parameters
    ----------
    disc_np  : (20,) float64 array  P(0,T_j) for T_j in T_PAY
    a_val    : float  mean-reversion speed
    sig_val  : float  volatility
    f0_tks   : (21,) float64 array  f(0,t_k) for t_k in T_MONITOR

    Returns
    -------
    y_vec    : (21,) float32
    dJ_disc  : (21, 20) float32   d(y_k) / d(P(0,T_j))
    dJ_scal  : (21,  2) float32   [d(y_k)/da, d(y_k)/dsig]
    """
    x_disc = torch.tensor(disc_np, dtype=torch.float64, requires_grad=True)
    x_a    = torch.tensor(a_val,   dtype=torch.float64, requires_grad=True)
    x_sig  = torch.tensor(sig_val, dtype=torch.float64, requires_grad=True)

    # Par rate as function of disc factors (differentiable)
    K = (1.0 - x_disc[-1]) / (TAU_T * x_disc.sum())

    T_PAY_t = torch.tensor(T_PAY, dtype=torch.float64)

    y_list = []
    for ki, t_k in enumerate(T_MONITOR):
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            y_list.append(torch.zeros((), dtype=torch.float64))
            continue

        f0t = float(f0_tks[ki])
        mu  = f0t + (x_sig**2 / (2.0 * x_a**2)) * (1.0 - torch.exp(-x_a * t_k))**2
        s2  = (x_sig**2 / (2.0 * x_a)) * (1.0 - torch.exp(-2.0 * x_a * t_k))

        mask    = T_PAY > t_k + 1e-9
        rem_idx = [j for j, m in enumerate(mask) if m]
        rem_T   = torch.tensor(T_PAY[mask], dtype=torch.float64)

        B   = (1.0 - torch.exp(-x_a * (rem_T - t_k))) / x_a          # (n_rem,)
        P0T = x_disc[rem_idx]                                          # (n_rem,) differentiable
        P0t = x_disc[_TK_TO_DISC_IDX[ki]]                             # scalar differentiable

        lnA = (torch.log(P0T / P0t)
               + B * f0t
               - (x_sig**2 / (4.0 * x_a)) * B**2 * (1.0 - torch.exp(-2.0 * x_a * t_k)))
        E_P = torch.exp(lnA - B * mu + 0.5 * B**2 * s2)

        y_list.append((1.0 - E_P[-1]) - K * TAU_T * E_P.sum())

    y_vec = torch.stack(y_list)                         # (21,)

    dJ_disc = torch.zeros(N_MON, N_PAY, dtype=torch.float64)
    dJ_scal = torch.zeros(N_MON, 2,    dtype=torch.float64)

    for ki in range(N_MON):
        t_k = T_MONITOR[ki]
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            continue
        grads = torch.autograd.grad(
            y_vec[ki], [x_disc, x_a, x_sig], retain_graph=True
        )
        dJ_disc[ki]    = grads[0]
        dJ_scal[ki, 0] = grads[1]
        dJ_scal[ki, 1] = grads[2]

    return (
        y_vec.detach().numpy().astype(np.float32),
        dJ_disc.numpy().astype(np.float32),
        dJ_scal.numpy().astype(np.float32),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Model 1a v2 Sobolev data generation")
    p.add_argument("--feds", type=str, default=FEDS_FILE)
    p.add_argument("--out",  type=str,
                   default=os.path.join(CODE_DIR, "data", "train_1a_v2_sob.h5"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    all_params = load_all_svensson_params(args.feds)
    N_CURVES   = len(all_params)
    N_TOTAL    = N_CURVES * len(A_GRID) * len(SIGMA_GRID)

    print(f"FEDS curves   : {N_CURVES:,}")
    print(f"a  values     : {len(A_GRID)}  ({A_GRID[0]:.3f} to {A_GRID[-1]:.3f})")
    print(f"sig values    : {len(SIGMA_GRID)}  ({SIGMA_GRID[0]:.4f} to {SIGMA_GRID[-1]:.4f})")
    print(f"Total samples : {N_TOTAL:,}")
    print(f"Output        : {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with h5py.File(args.out, "w") as hf:
        ds_xd  = hf.create_dataset("X_disc",    (N_TOTAL, N_PAY),       dtype="f4", chunks=(CHUNK, N_PAY))
        ds_xs  = hf.create_dataset("X_scalar",  (N_TOTAL, 2),           dtype="f4", chunks=(CHUNK, 2))
        ds_y   = hf.create_dataset("y_price",   (N_TOTAL, N_MON),       dtype="f4", chunks=(CHUNK, N_MON))
        ds_jd  = hf.create_dataset("dJ_disc",   (N_TOTAL, N_MON, N_PAY), dtype="f4", chunks=(CHUNK, N_MON, N_PAY))
        ds_js  = hf.create_dataset("dJ_scalar", (N_TOTAL, N_MON, 2),    dtype="f4", chunks=(CHUNK, N_MON, 2))

        idx   = 0
        t0    = time.time()

        for ci, p in enumerate(all_params):
            disc_np = market_discount(T_PAY, p).astype(np.float64)
            f0_tks  = svensson_forward(T_MONITOR, p).astype(np.float64)

            for a_val in A_GRID:
                for sig_val in SIGMA_GRID:
                    y_vec, dJ_d, dJ_s = _profile_and_jacobian(disc_np, a_val, sig_val, f0_tks)

                    ds_xd[idx]  = disc_np.astype(np.float32)
                    ds_xs[idx]  = np.array([a_val, sig_val], dtype=np.float32)
                    ds_y[idx]   = y_vec
                    ds_jd[idx]  = dJ_d
                    ds_js[idx]  = dJ_s
                    idx += 1

            if (ci + 1) % 50 == 0 or ci == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (ci + 1) * (N_CURVES - ci - 1)
                print(f"  [{ci+1:5d}/{N_CURVES}]  samples={idx:>10,}  elapsed={elapsed:6.0f}s  eta={eta:6.0f}s")

    print(f"\nSaved {N_TOTAL:,} samples to {args.out}")


if __name__ == "__main__":
    main()
