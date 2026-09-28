"""
test_cva_bump_recovery.py
===========================
CVA-level DV01 recovery test, on genuinely held-out curves.

Two tests per held-out curve, per model:

  Test 1 (base CVA accuracy): CVA_surr(base) vs CVA_ref(base), on curves
    the model never saw during training or validation.
  Test 2 (bump recovery): apply a 1bp parallel yield bump to the same
    curve, recompute CVA, and compare Delta_CVA_surr = CVA_surr(bumped)
    - CVA_surr(base) against Delta_CVA_ref = CVA_ref(bumped) - CVA_ref(base).

CVA formula (matches ResultsAndAnalysis.tex eq:cva_formula exactly):
    CVA = (1 - R) * sum_n EPE(t_n) * [S(t_{n-1}) - S(t_n)]
    S(t) = exp(-lambda * t),  R = 0.40,  lambda = 0.01,  n = 1..19

EPE(t_n) is obtained two ways per model family:
  - Model 2 (EPE-direct): the surrogate's raw output IS the EPE profile.
  - Models 1a/1b (price-direct): EPE_surr(t_n) = max(V_surr(t_n), 0).
Reference EPE is always the Monte Carlo HW distribution (b9_epe_dv01.py's
_epe_ref_profile, imported directly, not reimplemented), fixed-K from the
base curve, common random numbers between base and bumped legs.

Held-out curves: drawn from feds200628.csv (the full 16,912-curve FEDS
pool), excluding every date present in feds_1000.csv or feds_3000.csv
(the pools used for all training data). Genuinely unseen by every model
tested here.

Usage
-----
  python test_cva_bump_recovery.py --checkpoint data/best_model_1a_v22.pt --model_type 1a
  python test_cva_bump_recovery.py --checkpoint data/best_model_2_v22.pt  --model_type 2
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from hw_utils import T_MONITOR, T_PAY, TAU, load_all_svensson_params, market_discount
from b9_epe_dv01 import _epe_ref_profile, BUMP_BPS
from c4_model_v2 import IRSSurrogateV2

R_RECOVERY = 0.40
LAMBDA_HAZ = 0.01
N_MON = len(T_MONITOR)


def survival(t):
    return np.exp(-LAMBDA_HAZ * t)


def cva_from_epe(epe_profile):
    """
    epe_profile : (N, 21) EPE per sample per monitoring date (0 at
                  boundaries t_0 and t_20, matches T_MONITOR indexing).
    returns     : (N,) CVA per sample.
    """
    S = survival(T_MONITOR)              # (21,)
    weights = S[:-1] - S[1:]             # (20,) S(t_{n-1}) - S(t_n), n=1..20
    # interior monitoring dates are indices 1..19; weight for index k
    # (default between t_{k-1} and t_k) pairs with EPE(t_k).
    w_k = weights[:19]                    # weight applying to EPE(t_1..t_19)
    return (1.0 - R_RECOVERY) * (epe_profile[:, 1:20] * w_k[None, :]).sum(axis=1)


def build_holdout_curves(feds_full_path, feds_pools, n_holdout, seed):
    """Curves in feds_full_path whose date is not present in any of
    feds_pools (list of paths already used for training data)."""
    full = load_all_svensson_params(feds_full_path)
    used_dates = set()
    for pool_path in feds_pools:
        for p in load_all_svensson_params(pool_path):
            used_dates.add(p["date"])
    holdout = [p for p in full if p["date"] not in used_dates]
    print(f"Full pool: {len(full):,}  used in training: {len(used_dates):,}  "
          f"held-out available: {len(holdout):,}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(holdout), size=min(n_holdout, len(holdout)), replace=False)
    return [holdout[i] for i in idx], rng


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model_type", type=str, required=True, choices=["1a", "1b", "2"])
    p.add_argument("--feds_full",  type=str, default="data/feds200628.csv")
    p.add_argument("--feds_pools", type=str, nargs="+",
                   default=["data/feds_1000.csv", "data/feds_3000.csv"])
    p.add_argument("--n_holdout",  type=int, default=200)
    p.add_argument("--n_mc",       type=int, default=500)
    p.add_argument("--seed",       type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()

    holdout_params, rng = build_holdout_curves(
        args.feds_full, args.feds_pools, args.n_holdout, args.seed)
    N = len(holdout_params)

    X_disc = np.stack([market_discount(T_PAY, p).astype(np.float64) for p in holdout_params])
    a_vals   = rng.uniform(0.01, 0.30, size=N)
    sig_vals = rng.uniform(0.005, 0.030, size=N)
    X_scalar = np.stack([a_vals, sig_vals], axis=1)

    # --------------------------------------------------------- reference EPE/CVA
    print("Computing MC EPE reference (base) ...")
    EPE_ref_base = _epe_ref_profile(X_disc, X_scalar, bump=0.0, n_mc=args.n_mc, seed=args.seed)
    print("Computing MC EPE reference (bumped) ...")
    EPE_ref_up   = _epe_ref_profile(X_disc, X_scalar, bump=BUMP_BPS, n_mc=args.n_mc, seed=args.seed)

    CVA_ref_base = cva_from_epe(EPE_ref_base)
    CVA_ref_up   = cva_from_epe(EPE_ref_up)
    dCVA_ref     = CVA_ref_up - CVA_ref_base

    # ------------------------------------------------------------------ surrogate
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = IRSSurrogateV2(
        n_yields   = ckpt.get("n_yields",   20),
        n_scalar   = ckpt.get("n_scalar",    2),
        n_out      = ckpt.get("n_out",      21),
        hidden_dim = ckpt.get("hidden_dim", 128),
        n_layers   = ckpt.get("n_layers",    2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm_stats = ckpt.get("norm_stats", None)
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.6f})")

    x_disc_f32   = torch.tensor(X_disc,   dtype=torch.float32)
    x_scalar_f32 = torch.tensor(X_scalar, dtype=torch.float32)
    T_PAY_t32    = torch.tensor(T_PAY,    dtype=torch.float32)
    x_disc_up_f32 = x_disc_f32 * torch.exp(-BUMP_BPS * T_PAY_t32[None, :])

    def forward_raw(x_disc_raw):
        with torch.no_grad():
            if norm_stats is not None:
                xd_n = (x_disc_raw - norm_stats['xd_mean']) / norm_stats['xd_std']
                xs_n = (x_scalar_f32 - norm_stats['xs_mean']) / norm_stats['xs_std']
            else:
                xd_n = x_disc_raw
                xs_n = x_scalar_f32
            out = model(xd_n.unsqueeze(-1), xs_n)
            if norm_stats is not None:
                out = out * norm_stats['y_std'] + norm_stats['y_mean']
        return out.numpy().astype(np.float64)

    out_base = forward_raw(x_disc_f32)
    out_up   = forward_raw(x_disc_up_f32)

    if args.model_type == "2":
        EPE_surr_base = out_base
        EPE_surr_up   = out_up
    else:
        EPE_surr_base = np.maximum(out_base, 0.0)
        EPE_surr_up   = np.maximum(out_up, 0.0)

    CVA_surr_base = cva_from_epe(EPE_surr_base)
    CVA_surr_up   = cva_from_epe(EPE_surr_up)
    dCVA_surr     = CVA_surr_up - CVA_surr_base

    # ---------------------------------------------------------------------- report
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

    def corr(y_true, y_pred):
        if y_true.std() < 1e-30 or y_pred.std() < 1e-30:
            return float("nan")
        return np.corrcoef(y_true, y_pred)[0, 1]

    def mae(y_true, y_pred):
        return float(np.mean(np.abs(y_true - y_pred)))

    print()
    print(f"=== Test 1: held-out base CVA accuracy ({N} curves, never used in training) ===")
    print(f"  CVA_ref  mean: {CVA_ref_base.mean():.6e}   std: {CVA_ref_base.std():.6e}")
    print(f"  CVA_surr mean: {CVA_surr_base.mean():.6e}   std: {CVA_surr_base.std():.6e}")
    print(f"  R2:   {r2(CVA_ref_base, CVA_surr_base):.4f}")
    print(f"  corr: {corr(CVA_ref_base, CVA_surr_base):.4f}")
    print(f"  MAE:  {mae(CVA_ref_base, CVA_surr_base):.6e}  "
          f"({mae(CVA_ref_base, CVA_surr_base)*10000:.4f} bps of notional)")

    print()
    print(f"=== Test 2: CVA bump recovery (Delta_CVA, 1bp parallel shift, same {N} curves) ===")
    print(f"  dCVA_ref  mean: {dCVA_ref.mean():.6e}   std: {dCVA_ref.std():.6e}")
    print(f"  dCVA_surr mean: {dCVA_surr.mean():.6e}   std: {dCVA_surr.std():.6e}")
    print(f"  Scale ratio (surr/ref std): {dCVA_surr.std() / (dCVA_ref.std() + 1e-30):.4f}")
    print(f"  R2:   {r2(dCVA_ref, dCVA_surr):.4f}")
    print(f"  corr: {corr(dCVA_ref, dCVA_surr):.4f}")
    sign_match = float(np.mean(np.sign(dCVA_ref) == np.sign(dCVA_surr)))
    print(f"  Sign agreement: {sign_match*100:.1f}% of curves")


if __name__ == "__main__":
    main()
