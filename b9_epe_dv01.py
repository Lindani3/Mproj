"""
b9_epe_dv01.py
==============
Per-date DV01 R² evaluation for Model 2 EPE surrogate models.

Model 2 is trained directly on EPE labels:
    EPE(t_k) = E^Q[max(V(t_k), 0)]

The reference DV01 is therefore the EPE DV01, not the swap-value DV01:
    DV01_EPE_ref(t_k)  = EPE_ref(t_k; P+) - EPE_ref(t_k; P)
    DV01_EPE_surr(t_k) = ŷ_k(P+, a, σ)   - ŷ_k(P, a, σ)

EPE_ref is computed by Monte Carlo:
    1. Draw r_{t_k} ~ N(μ_k, s_k²)  (closed-form HW distribution)
    2. Evaluate V(t_k, r) = (1 - P(t_k,T_N;r)) - K_par · τ · Σ P(t_k,T_j;r)
    3. EPE = mean(max(V, 0))

K_par is ALWAYS fixed from the base (unbumped) curve so the DV01 measures
sensitivity of the EPE of an existing at-the-money swap, not a freshly
re-struck one.

Common random numbers are used: the same z-sample is shared between the base
and bumped evaluations of r_{t_k}, so the DV01 estimate is low-variance.

Usage
-----
  python b9_epe_dv01.py \\
      --checkpoint data/best_model_2_v22.pt \\
      --data       data/train_2_v22.h5 \\
      --label      2_v22 \\
      --n_mc       500 \\
      --out_dir    data/r2
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import T_MONITOR, T_PAY, TAU

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BUMP_BPS = 0.0001   # 1 basis point parallel yield shift
N_MON    = len(T_MONITOR)   # 21


# ---------------------------------------------------------------------------
# MC EPE reference
# ---------------------------------------------------------------------------

def _epe_ref_profile(
    X_disc:   np.ndarray,   # (N, 20)  base discount factors P(0, T_j)
    X_scalar: np.ndarray,   # (N,  2)  [a, sigma]
    bump:     float = 0.0,  # 0.0 for base, BUMP_BPS for shifted curve
    n_mc:     int   = 500,
    seed:     int   = 42,
) -> np.ndarray:
    """
    Compute EPE_ref(t_k) = E^Q[max(V(t_k), 0)] for all samples and dates.

    K_par is always derived from the BASE (unbumped) X_disc, ensuring the
    DV01 measures sensitivity of an existing par swap.

    Parameters
    ----------
    X_disc   : (N, 20) base discount factors
    X_scalar : (N,  2) [a, sigma]
    bump     : parallel zero-yield shift (0 = base, 0.0001 = +1bp)
    n_mc     : number of short-rate draws per sample per monitoring date
    seed     : RNG seed for common random numbers (same seed → same z)

    Returns
    -------
    EPE_ref  : (N, 21) float64 — zero at boundary dates t_0=0 and t_20=10
    """
    N       = X_disc.shape[0]
    a_arr   = X_scalar[:, 0].astype(np.float64)   # (N,)
    sig_arr = X_scalar[:, 1].astype(np.float64)   # (N,)

    # K_par ALWAYS from base disc
    base_disc = X_disc.astype(np.float64)
    K_par = (1.0 - base_disc[:, -1]) / (TAU * base_disc.sum(axis=1))   # (N,)

    # Bumped disc factors (used only for bond pricing inside EPE)
    if bump == 0.0:
        disc = base_disc
    else:
        disc = base_disc * np.exp(-bump * T_PAY[np.newaxis, :])

    # Forward rates from BASE log-discounts (for r_{t_k} distribution)
    log_base = np.log(base_disc)   # (N, 20)

    EPE_ref = np.zeros((N, N_MON), dtype=np.float64)

    rng = np.random.default_rng(seed)
    # Pre-draw all z once so base/bumped share the same normals
    z = rng.standard_normal((n_mc,))   # (M,)

    for k in range(1, 20):
        t_k = float(T_MONITOR[k])

        # --- f(0, t_k) from base log-discounts ---
        if k == 1:
            f0t = -(log_base[:, 1] - log_base[:, 0]) / TAU
        else:
            f0t = -(log_base[:, k] - log_base[:, k - 2]) / (2.0 * TAU)

        # Bumped forward (only shifts mean of r_{t_k})
        f0t_bump = f0t + bump

        # --- r_{t_k} ~ N(mu, s^2) distribution parameters ---
        ea  = np.exp(-a_arr * t_k)
        e2a = np.exp(-2.0 * a_arr * t_k)
        s2  = (sig_arr**2 / (2.0 * a_arr)) * (1.0 - e2a)          # (N,)
        mu  = f0t_bump + (sig_arr**2 / (2.0 * a_arr**2)) * (1.0 - ea)**2   # (N,)
        s   = np.sqrt(np.maximum(s2, 0.0))                         # (N,)

        # --- short-rate draws: r ~ mu + s * z ---
        # r_mc : (N, M)
        r_mc = mu[:, None] + s[:, None] * z[None, :]

        # --- remaining payment dates ---
        rem_mask = T_PAY > t_k + 1e-9
        T_rem    = T_PAY[rem_mask]          # (n_rem,)
        disc_rem = disc[:, rem_mask]        # (N, n_rem)
        disc_t   = disc[:, k - 1]          # (N,) P+(0, t_k)

        # B(t_k, T_j): (N, n_rem)
        B = ((1.0 - np.exp(-a_arr[:, None] * (T_rem[None, :] - t_k)))
             / a_arr[:, None])

        # ln A(t_k, T_j): (N, n_rem)
        lnA = (np.log(disc_rem / disc_t[:, None])
               + B * f0t_bump[:, None]
               - (sig_arr[:, None]**2 / (4.0 * a_arr[:, None]))
               * B**2 * (1.0 - e2a[:, None]))

        # P(t_k, T_j | r): (N, M, n_rem)
        #   lnA: (N, 1, n_rem),  B: (N, 1, n_rem),  r_mc: (N, M, 1)
        log_bond = lnA[:, None, :] - B[:, None, :] * r_mc[:, :, None]
        P_bond   = np.exp(log_bond)   # (N, M, n_rem)

        # Swap value: V = (1 - P_N) - K_par * tau * sum_j P_j
        V = (1.0 - P_bond[:, :, -1]) - K_par[:, None] * TAU * P_bond.sum(axis=2)

        # EPE = mean(max(V, 0))
        EPE_ref[:, k] = np.mean(np.maximum(V, 0.0), axis=1)

    return EPE_ref


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Per-date EPE DV01 R² for Model 2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--label",      type=str, default="model")
    p.add_argument("--n_mc",       type=int, default=500)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device",     type=str, default="cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--out_dir",    type=str,
                   default=os.path.join(CODE_DIR, "data"))
    return p.parse_args()


def main():
    import torch
    from c4_model_v2 import IRSSurrogateV2

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------ model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location="cpu")
    model = IRSSurrogateV2(
        n_yields   = ckpt.get("n_yields",   20),
        n_scalar   = ckpt.get("n_scalar",    2),
        n_out      = ckpt.get("n_out",      21),
        hidden_dim = ckpt.get("hidden_dim", 128),
        n_layers   = ckpt.get("n_layers",    2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device     = torch.device(args.device)
    model.to(device)
    norm_stats = ckpt.get("norm_stats", None)
    print(f"  Epoch {ckpt['epoch']},  val MSE {ckpt['val_loss']:.6f}")
    if norm_stats is not None:
        print("  Normalisation stats found (Sobolev model).")

    # ------------------------------------------------------------------ data
    print(f"Loading data: {args.data}")
    with h5py.File(args.data, "r") as hf:
        X_disc   = hf["X_disc"][:].astype(np.float32)    # (N, 20)
        X_scalar = hf["X_scalar"][:].astype(np.float32)  # (N,  2)
    N = X_disc.shape[0]
    print(f"  {N:,} samples,  n_mc={args.n_mc},  seed={args.seed}")

    # ------------------------------------------------------------------ bump
    T_PAY_f32 = T_PAY.astype(np.float32)
    X_disc_up = X_disc * np.exp(-BUMP_BPS * T_PAY_f32[np.newaxis, :])

    # ------------------------------------------------------------------ surrogate forward passes
    def _forward(disc_np: np.ndarray) -> np.ndarray:
        disc_t = torch.from_numpy(disc_np)

        if norm_stats is not None:
            xd_mean = norm_stats['xd_mean']
            xd_std  = norm_stats['xd_std']
            xs_mean = norm_stats['xs_mean']
            xs_std  = norm_stats['xs_std']
            y_mean  = norm_stats['y_mean'].numpy()
            y_std   = norm_stats['y_std'].numpy()
            disc_n  = (disc_t - xd_mean) / xd_std
            xs_n    = (torch.from_numpy(X_scalar) - xs_mean) / xs_std
        else:
            disc_n = disc_t
            xs_n   = torch.from_numpy(X_scalar)
            y_mean, y_std = 0.0, 1.0

        preds = np.empty((N, 21), dtype=np.float64)
        bs    = args.batch_size
        with torch.no_grad():
            for i in range(0, N, bs):
                xd  = disc_n[i:i+bs].unsqueeze(-1).to(device)
                xs  = xs_n[i:i+bs].to(device)
                out = model(xd, xs).cpu().numpy().astype(np.float64)
                preds[i:i+bs] = out

        if norm_stats is not None:
            preds = preds * y_std + y_mean
        return preds

    print("Forward pass — base curve ...")
    y_base = _forward(X_disc)

    print("Forward pass — bumped curve (+1bp) ...")
    y_up   = _forward(X_disc_up)

    DV01_surr = y_up - y_base   # (N, 21)

    # ------------------------------------------------------------------ MC EPE reference DV01
    print("Computing MC EPE reference (base) ...")
    EPE_base  = _epe_ref_profile(X_disc, X_scalar, bump=0.0,
                                 n_mc=args.n_mc, seed=args.seed)

    print("Computing MC EPE reference (bumped) ...")
    EPE_up    = _epe_ref_profile(X_disc, X_scalar, bump=BUMP_BPS,
                                 n_mc=args.n_mc, seed=args.seed)

    DV01_ref  = EPE_up - EPE_base   # (N, 21)

    # Diagnostic: print DV01 scale to detect normalisation issues
    interior = list(range(1, 20))
    ref_rms  = float(np.sqrt(np.mean(DV01_ref[:, interior]**2)))
    surr_rms = float(np.sqrt(np.mean(DV01_surr[:, interior]**2)))
    print(f"\n  DV01_ref  RMS (interior dates): {ref_rms:.6e}")
    print(f"  DV01_surr RMS (interior dates): {surr_rms:.6e}")
    print(f"  Scale ratio (surr/ref):         {surr_rms / (ref_rms + 1e-30):.4f}")

    # ------------------------------------------------------------------ R² at each date
    rows = []
    for k, t_k in enumerate(T_MONITOR):
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            rows.append({"time": t_k, "n": N, "R2_DV01": np.nan})
            continue
        yt = DV01_ref[:, k]
        yp = DV01_surr[:, k]
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else np.nan
        rows.append({"time": t_k, "n": N, "R2_DV01": float(r2)})

    df      = pd.DataFrame(rows)
    mean_r2 = float(np.nanmean(df["R2_DV01"]))

    hdr = f"{'time':>6}  {'n':>10}  {'R2_DV01':>12}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for _, row in df.iterrows():
        r2_str = f"{row['R2_DV01']:.6f}" if not np.isnan(row["R2_DV01"]) else "         NaN"
        print(f"{row['time']:>6.1f}  {int(row['n']):>10,}  {r2_str:>12}")
    print("-" * len(hdr))
    print(f"{'Mean EPE DV01 R²':>22}  {mean_r2:.6f}")

    out = os.path.join(args.out_dir, f"epe_dv01_profile_{args.label}.csv")
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
