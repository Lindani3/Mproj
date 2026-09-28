"""
c3_datagen_2_v2_sob_pathwise.py
================================
Sobolev-augmented data generation for Model 2 (EPE labels, MC).

Uses pathwise (reparameterisation-trick) derivatives via PyTorch autograd
instead of central finite differences.  The short rate is expressed as

    r(t_k) = mu(t_k; a, sigma) + s(t_k; a, sigma) * Z_k

where Z_k ~ N(0,1) is fixed.  Because a and sigma appear explicitly in mu
and s, autograd traces

    d/da  EPE(t_k) = (1/M) * sum_m  1[V^m > 0] * dV^m/da
    d/dsig EPE(t_k) = (1/M) * sum_m  1[V^m > 0] * dV^m/dsig

through the bond-price / IRS-value formula.  This eliminates the O(1/eps^2)
variance amplification of bump-and-reprice, giving much cleaner Jacobian
labels at lower computational cost (one forward + 19 backward passes per
sample instead of five full MC re-runs).

Reference
---------
Broadie & Glasserman (1996), Estimating security price derivatives using
simulation, Management Science 42(2): 269-285.

Output HDF5 keys
----------------
  X_disc    (N, 20)      initial discount factors  P(0, T_j), T_j in T_PAY
  X_scalar  (N,  2)      [a, sigma]
  y_price   (N, 21)      MC EPE profile
  dJ_scalar (N, 21,  2)  pathwise [dEPE_k/da, dEPE_k/dsig]
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
N_MON      = len(T_MONITOR)        # 21
N_PAY      = len(T_PAY)            # 20
CHUNK      = 256

# Precompute the index mapping: T_MONITOR[ki] = T_PAY[ki-1] for ki = 1..20
# So P(0, T_MONITOR[ki]) = disc_np[ki-1]  (for ki >= 1)
_T_MON_F64 = np.array(T_MONITOR, dtype=np.float64)
_T_PAY_F64 = np.array(T_PAY,     dtype=np.float64)


def _epe_pathwise(disc_np, f0_mon, a_val, sig_val, K, Z_t):
    """
    Compute EPE profile and scalar Jacobian for one (curve, a, sigma) sample.

    Parameters
    ----------
    disc_np : (N_PAY,) ndarray  — P(0, T_j) for T_j in T_PAY
    f0_mon  : (N_MON,) ndarray  — f(0, t_k) for t_k in T_MONITOR
    a_val, sig_val : float
    K       : float             — par swap rate at t=0
    Z_t     : (M, N_MON) Tensor — fixed standard normals (no grad)

    Returns
    -------
    y_price   : (N_MON,) float32 ndarray
    dJ_scalar : (N_MON, 2) float32 ndarray  — columns: [d/da, d/dsig]
    """
    dtype = torch.float64
    M     = Z_t.shape[0]

    a_t   = torch.tensor(a_val,   dtype=dtype, requires_grad=True)
    sig_t = torch.tensor(sig_val, dtype=dtype, requires_grad=True)

    disc_t = torch.from_numpy(disc_np.astype(np.float64))   # (N_PAY,) — constant
    K_t    = torch.tensor(K, dtype=dtype)

    y_price   = np.zeros(N_MON, dtype=np.float32)
    dJ_scalar = np.zeros((N_MON, 2), dtype=np.float32)

    for ki in range(N_MON):
        t_k = float(_T_MON_F64[ki])
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            continue

        t_k_t = torch.tensor(t_k, dtype=dtype)

        # --- Marginal distribution of r(t_k) | r(0) = f(0,0) ---
        # mu(t_k) = f(0,t_k) + (sig^2 / 2a^2) * (1 - exp(-a*t_k))^2
        # s(t_k)  = sig * sqrt((1 - exp(-2a*t_k)) / (2a))
        f0t_k = float(f0_mon[ki])
        mu_k  = (f0t_k
                 + (sig_t ** 2 / (2.0 * a_t ** 2))
                 * (1.0 - torch.exp(-a_t * t_k_t)) ** 2)
        s_k   = (sig_t
                 * torch.sqrt((1.0 - torch.exp(-2.0 * a_t * t_k_t))
                              / (2.0 * a_t)))

        # Reparameterised short rate — gradient flows through mu and s
        r_k = mu_k + s_k * Z_t[:, ki]                        # (M,)

        # --- Remaining payment dates T_j > t_k ---
        # T_PAY[ki:] are the dates strictly after t_k
        # (because T_PAY[ki-1] = T_MONITOR[ki] = t_k for ki >= 1)
        T_rem    = torch.tensor(_T_PAY_F64[ki:], dtype=dtype)  # (n_rem,)
        disc_rem = disc_t[ki:]                                  # (n_rem,)
        n_rem    = T_rem.shape[0]

        # P(0, t_k) — constant from market
        P0t_k = float(disc_np[ki - 1])                         # ki >= 1 always here

        # B(t_k, T_j) = (1 - exp(-a*(T_j - t_k))) / a          (n_rem,)
        B_kj = (1.0 - torch.exp(-a_t * (T_rem - t_k_t))) / a_t

        # ln A(t_k, T_j) — affine HW bond price coefficient
        # = ln(P(0,T_j)/P(0,t_k)) + B_kj*f(0,t_k) - (sig^2/4a)*B_kj^2*(1-exp(-2a*t_k))
        lnA_kj = (torch.log(disc_rem / P0t_k)
                  + B_kj * f0t_k
                  - (sig_t ** 2 / (4.0 * a_t))
                  * B_kj ** 2
                  * (1.0 - torch.exp(-2.0 * a_t * t_k_t)))     # (n_rem,)

        # Bond prices P(t_k, T_j | r_k)                         (M, n_rem)
        P_kj = torch.exp(lnA_kj.unsqueeze(0)
                         - r_k.unsqueeze(1) * B_kj.unsqueeze(0))

        # IRS value: float leg − fixed leg
        V_k = (1.0 - P_kj[:, -1]) - K_t * TAU * P_kj.sum(dim=1)  # (M,)

        # EPE
        EPE_k = torch.clamp(V_k, min=0.0).mean()               # scalar

        y_price[ki] = float(EPE_k.detach())

        # Pathwise Jacobian — one backward per date (graph freed after each)
        ga, gs = torch.autograd.grad(EPE_k, [a_t, sig_t])
        dJ_scalar[ki, 0] = float(ga)
        dJ_scalar[ki, 1] = float(gs)

    return y_price, dJ_scalar


def parse_args():
    p = argparse.ArgumentParser(
        description="Model 2 v2 Sobolev data generation — pathwise derivatives")
    p.add_argument("--feds",    type=str, default=FEDS_FILE)
    p.add_argument("--out",     type=str,
                   default=os.path.join(CODE_DIR, "data", "train_2_v2_sob.h5"))
    p.add_argument("--m_paths", type=int, default=5000)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    torch.set_num_threads(8)   # use all available CPUs for matrix ops

    all_params = load_all_svensson_params(args.feds)
    N_CURVES   = len(all_params)
    N_TOTAL    = N_CURVES * len(A_GRID) * len(SIGMA_GRID)
    M          = args.m_paths

    print(f"FEDS curves   : {N_CURVES:,}")
    print(f"MC paths      : {M:,}")
    print(f"Total samples : {N_TOTAL:,}")
    print(f"Jacobian mode : pathwise (reparameterisation + autograd)")
    print(f"Output        : {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with h5py.File(args.out, "w") as hf:
        ds_xd = hf.create_dataset("X_disc",    (N_TOTAL, N_PAY),    dtype="f4",
                                  chunks=(CHUNK, N_PAY))
        ds_xs = hf.create_dataset("X_scalar",  (N_TOTAL, 2),        dtype="f4",
                                  chunks=(CHUNK, 2))
        ds_y  = hf.create_dataset("y_price",   (N_TOTAL, N_MON),    dtype="f4",
                                  chunks=(CHUNK, N_MON))
        ds_js = hf.create_dataset("dJ_scalar", (N_TOTAL, N_MON, 2), dtype="f4",
                                  chunks=(CHUNK, N_MON, 2))

        idx = 0
        t0  = time.time()

        for ci, p in enumerate(all_params):
            disc_np = market_discount(_T_PAY_F64, p).astype(np.float64)
            f0_mon  = np.array([svensson_forward(t, p) for t in _T_MON_F64],
                               dtype=np.float64)
            K       = par_swap_rate(p)

            for a_val in A_GRID:
                for sig_val in SIGMA_GRID:
                    # Generate fixed Z for this sample — M paths × N_MON dates
                    Z_np = rng.standard_normal(size=(M, N_MON))
                    Z_t  = torch.from_numpy(Z_np)               # no requires_grad

                    y_price, dJ_s = _epe_pathwise(
                        disc_np, f0_mon, a_val, sig_val, K, Z_t)

                    ds_xd[idx] = disc_np.astype(np.float32)
                    ds_xs[idx] = np.array([a_val, sig_val], dtype=np.float32)
                    ds_y[idx]  = y_price
                    ds_js[idx] = dJ_s
                    idx += 1

            if (ci + 1) % 10 == 0 or ci == 0:
                elapsed = time.time() - t0
                eta     = elapsed / (ci + 1) * (N_CURVES - ci - 1)
                print(f"  [{ci+1:5d}/{N_CURVES}]  samples={idx:>10,}"
                      f"  elapsed={elapsed:6.0f}s  eta={eta:6.0f}s")

    print(f"\nSaved {N_TOTAL:,} samples to {args.out}")


if __name__ == "__main__":
    main()
