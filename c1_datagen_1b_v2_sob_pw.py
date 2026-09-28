"""
c1_datagen_1b_v2_sob_pw.py
===========================
Sobolev-augmented data generation for Model 1b (Monte Carlo price labels,
analytical gradient labels).

Price labels y_price are MC estimates of V_IRS(t_k, r_0), identical in
construction to c1_datagen_1b_v2_sob.py.  Scalar Sobolev labels dJ_scalar
are computed analytically via PyTorch autograd through the Hull-White affine
bond price formula, replacing the finite-difference estimator.

The finite-difference approach in c1_datagen_1b_v2_sob.py computes gradients
as (V_MC(a+eps) - V_MC(a-eps)) / (2*eps) using independent Monte Carlo runs.
Dividing two independent noisy quantities by a small step amplifies variance
by O(1/eps^2), making the gradient labels orders of magnitude noisier than the
price labels.  The analytical gradient eliminates this amplification entirely:

    V(t_k, r_0; a, sigma) = (1 - P(t_k, T_n | r_0))
                             - K * tau * sum_j P(t_k, T_j | r_0)

    P(t_k, T_j | r_0) = exp(lnA(t_k, T_j; a, sigma) - B(t_k, T_j; a) * r_0)

Since V_analytical(t_k, r_0; a, sigma) = lim_{M->inf} V_MC(t_k, r_0; a, sigma),
the analytical gradient is the zero-variance limit of the pathwise Monte Carlo
gradient.  Using it as the Sobolev label isolates the effect of price label
noise (intrinsic to Model 1b) from gradient label noise (an artefact of the
finite-difference estimator).

Reference
---------
Broadie & Glasserman (1996), Estimating security price derivatives using
simulation, Management Science 42(2): 269-285.

Output HDF5 keys
----------------
  X_disc    (N, 20)      initial discount factors  P(0, T_j), T_j in T_PAY
  X_scalar  (N,  2)      [a, sigma]
  y_price   (N, 21)      MC IRS value profile labels
  dJ_scalar (N, 21,  2)  analytical [dV_k/da, dV_k/dsig]
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
    simulate_hw_paths, compute_irs_pv,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "data", "feds_1000.csv")

A_GRID     = np.linspace(0.01, 0.30, 8)
SIGMA_GRID = np.linspace(0.005, 0.030, 8)
N_MON      = len(T_MONITOR)   # 21
N_PAY      = len(T_PAY)       # 20
CHUNK      = 256

_T_MON_F64 = np.array(T_MONITOR, dtype=np.float64)
_T_PAY_F64 = np.array(T_PAY,     dtype=np.float64)


def _mc_profile(p, a_val, sig_val, K, M, rng):
    """MC IRS profile for one (curve, a, sigma) — identical to 1b_v2_sob."""
    disc_np = market_discount(T_PAY, p).astype(np.float32)
    r0      = float(-np.log(disc_np[0]) / T_PAY[0])
    r_batch = np.full(M, r0, dtype=float)

    y_vec = np.zeros(N_MON, dtype=np.float32)
    for ki, t_k in enumerate(T_MONITOR):
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            continue
        r_at_pay, logD_at_pay, rem = simulate_hw_paths(
            t_k, r_batch, p, a_val, sig_val, rng=rng
        )
        if len(rem) == 0:
            continue
        V_paths = compute_irs_pv(
            t_k, r_batch, r_at_pay, logD_at_pay, rem, K, p, a_val, sig_val
        )
        y_vec[ki] = float(V_paths.mean())
    return y_vec, disc_np


def _analytical_grad(disc_np, f0_mon, r0, a_val, sig_val, K):
    """
    Compute dV(t_k, r_0)/d(a, sigma) analytically for all monitoring dates.

    Parameters
    ----------
    disc_np : (N_PAY,) float64 ndarray  — P(0, T_j) for T_j in T_PAY
    f0_mon  : (N_MON,) float64 ndarray  — f(0, t_k) for t_k in T_MONITOR
    r0      : float                     — initial short rate
    a_val, sig_val, K : float

    Returns
    -------
    dJ_scalar : (N_MON, 2) float32 ndarray — columns [dV_k/da, dV_k/dsig]
    """
    dtype = torch.float64

    a_t   = torch.tensor(a_val,   dtype=dtype, requires_grad=True)
    sig_t = torch.tensor(sig_val, dtype=dtype, requires_grad=True)

    disc_t = torch.from_numpy(disc_np)   # (N_PAY,) — no grad
    K_t    = torch.tensor(K,  dtype=dtype)
    r0_t   = torch.tensor(r0, dtype=dtype)

    dJ_scalar = np.zeros((N_MON, 2), dtype=np.float32)

    for ki in range(N_MON):
        t_k = float(_T_MON_F64[ki])
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            continue

        t_k_t = torch.tensor(t_k, dtype=dtype)
        f0t_k = float(f0_mon[ki])

        # Remaining payment dates T_j > t_k
        T_rem    = torch.tensor(_T_PAY_F64[ki:], dtype=dtype)   # (n_rem,)
        disc_rem = disc_t[ki:]                                    # (n_rem,)
        P0t_k    = float(disc_np[ki - 1])                        # P(0, t_k); ki >= 1

        # B(t_k, T_j) = (1 - exp(-a*(T_j - t_k))) / a            (n_rem,)
        B_kj = (1.0 - torch.exp(-a_t * (T_rem - t_k_t))) / a_t

        # ln A(t_k, T_j)                                          (n_rem,)
        lnA_kj = (torch.log(disc_rem / P0t_k)
                  + B_kj * f0t_k
                  - (sig_t ** 2 / (4.0 * a_t))
                  * B_kj ** 2
                  * (1.0 - torch.exp(-2.0 * a_t * t_k_t)))

        # Bond prices P(t_k, T_j | r_0)                          (n_rem,)
        P_kj = torch.exp(lnA_kj - B_kj * r0_t)

        # IRS value: no clamp — V not EPE
        V_k = (1.0 - P_kj[-1]) - K_t * TAU * P_kj.sum()

        ga, gs = torch.autograd.grad(V_k, [a_t, sig_t])
        dJ_scalar[ki, 0] = float(ga)
        dJ_scalar[ki, 1] = float(gs)

    return dJ_scalar


def parse_args():
    p = argparse.ArgumentParser(
        description="Model 1b v2 Sobolev data generation — analytical gradient labels")
    p.add_argument("--feds",    type=str, default=FEDS_FILE)
    p.add_argument("--out",     type=str,
                   default=os.path.join(CODE_DIR, "data", "train_1b_v2_sob_pw.h5"))
    p.add_argument("--m_paths", type=int, default=5000)
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng  = np.random.default_rng(args.seed)

    torch.set_num_threads(8)

    all_params = load_all_svensson_params(args.feds)
    N_CURVES   = len(all_params)
    N_TOTAL    = N_CURVES * len(A_GRID) * len(SIGMA_GRID)
    M          = args.m_paths

    print(f"FEDS curves   : {N_CURVES:,}")
    print(f"MC paths      : {M:,}")
    print(f"Total samples : {N_TOTAL:,}")
    print(f"Jacobian mode : analytical (autograd through affine formula)")
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
            disc_f64 = market_discount(T_PAY, p).astype(np.float64)
            f0_mon   = np.array([svensson_forward(t, p) for t in _T_MON_F64],
                                dtype=np.float64)
            K        = par_swap_rate(p)

            for a_val in A_GRID:
                for sig_val in SIGMA_GRID:
                    y_price, disc_f32 = _mc_profile(p, a_val, sig_val, K, M, rng)

                    r0 = float(-np.log(disc_f64[0]) / _T_PAY_F64[0])
                    dJ_s = _analytical_grad(disc_f64, f0_mon, r0, a_val, sig_val, K)

                    ds_xd[idx] = disc_f32
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
