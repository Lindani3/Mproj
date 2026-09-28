"""
diagnose_dv01_precision.py
===========================
Verifies whether a trained checkpoint's DV01 signal (as measured by
b8_dv01_profile.py's finite-difference bump-and-revalue) reflects a
genuinely learned local derivative, by comparing three independent
measurements per sample, per interior monitoring date:

  1. dv01_ref       : analytical target (ground truth), verified exact
                       against b8_dv01_profile.py's _ref_profile_from_disc.
  2. dv01_autograd  : EXACT derivative of the trained network's output
                       w.r.t. the RAW x_disc input, contracted with the
                       parallel-shift direction d(x_disc)/d(bump) =
                       -T_PAY * x_disc, i.e. no finite difference at all.
                       If the checkpoint has norm_stats, the normalisation
                       is included inside the autograd graph so the
                       gradient is still taken w.r.t. the raw input.
  3. dv01_fd[bump]  : finite-difference (pred_up - pred) / bump, for
                       bump in {1bp, 5bp, 25bp, 100bp}, matching what
                       b8_dv01_profile.py actually measures (same
                       normalisation handling as b8's _forward()).

Usage
-----
  python diagnose_dv01_precision.py --checkpoint data/best_model_1a_v22_sob.pt --data data/train_1a_v22.h5

Read-only. Does not modify b8_dv01_profile.py, any training script, or
any checkpoint.
"""

import argparse
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import h5py
import torch

from hw_utils import T_MONITOR, T_PAY, TAU, load_all_svensson_params, market_discount
import b8_dv01_profile as b8
from c4_model_v2 import IRSSurrogateV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default="data/best_model_1a_v2_dv01.pt")
    p.add_argument("--data",       type=str, default=None,
                   help="HDF5 file to draw real X_disc/X_scalar samples from. "
                        "If omitted, samples are built from feds_1000.csv directly.")
    p.add_argument("--feds",       type=str, default="data/feds_1000.csv")
    p.add_argument("--n_sample",   type=int, default=500)
    p.add_argument("--seed",       type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- data
    if args.data is not None:
        with h5py.File(args.data, "r") as hf:
            N_total = hf["X_disc"].shape[0]
            idx = rng.choice(N_total, size=min(args.n_sample, N_total), replace=False)
            idx.sort()
            X_disc   = hf["X_disc"][idx].astype(np.float64)
            X_scalar = hf["X_scalar"][idx].astype(np.float64)
        print(f"Sampled {len(idx)} rows from {args.data}")
    else:
        all_params = load_all_svensson_params(args.feds)
        idx = rng.choice(len(all_params), size=args.n_sample, replace=False)
        params_sub = [all_params[i] for i in idx]
        X_disc = np.stack([market_discount(T_PAY, p).astype(np.float64) for p in params_sub])
        a_vals   = rng.uniform(0.01, 0.30, size=args.n_sample)
        sig_vals = rng.uniform(0.005, 0.030, size=args.n_sample)
        X_scalar = np.stack([a_vals, sig_vals], axis=1)
        print(f"Built {args.n_sample} synthetic samples from {args.feds}")

    N_SAMPLE = X_disc.shape[0]

    # --------------------------------------------------------- analytical ref
    K_par = (1.0 - X_disc[:, -1]) / (TAU * X_disc.sum(axis=1))
    V_base_ref = b8._ref_profile_from_disc(X_disc, X_scalar, bump=0.0, K_par=K_par)
    V_up_ref   = b8._ref_profile_from_disc(X_disc, X_scalar, bump=b8.BUMP_BPS, K_par=K_par)
    dv01_ref = V_up_ref - V_base_ref   # (N, 21), float64

    # --------------------------------------------------------------- load model
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
    print(f"Normalisation stats: {'present' if norm_stats is not None else 'none'}")

    x_disc_f32   = torch.tensor(X_disc,   dtype=torch.float32)
    x_scalar_f32 = torch.tensor(X_scalar, dtype=torch.float32)
    T_PAY_t32    = torch.tensor(T_PAY,    dtype=torch.float32)

    if norm_stats is not None:
        xd_mean = norm_stats['xd_mean']
        xd_std  = norm_stats['xd_std']
        xs_mean = norm_stats['xs_mean']
        xs_std  = norm_stats['xs_std']
        y_mean  = norm_stats['y_mean']
        y_std   = norm_stats['y_std']
    else:
        xd_mean = xd_std = xs_mean = xs_std = y_mean = y_std = None

    def forward_raw(x_disc_raw, x_scalar_raw):
        """Forward pass taking RAW (unnormalised) inputs, handling
        normalisation internally exactly as b8_dv01_profile.py's
        _forward() does, and de-normalising the output back to raw
        price units."""
        if norm_stats is not None:
            xd_n = (x_disc_raw - xd_mean) / xd_std
            xs_n = (x_scalar_raw - xs_mean) / xs_std
        else:
            xd_n = x_disc_raw
            xs_n = x_scalar_raw
        out = model(xd_n.unsqueeze(-1), xs_n)
        if norm_stats is not None:
            out = out * y_std + y_mean
        return out

    # ---------------------------------------------------------- 1) exact autograd
    x_disc_grad = x_disc_f32.clone().requires_grad_(True)
    pred = forward_raw(x_disc_grad, x_scalar_f32)   # (N, 21)

    direction = -x_disc_f32 * T_PAY_t32[None, :]     # d(x_disc)/d(bump) at bump=0

    dv01_autograd = torch.zeros(N_SAMPLE, 21)
    for k in range(1, 20):
        grad_k = torch.autograd.grad(pred[:, k].sum(), x_disc_grad, retain_graph=True)[0]
        dv01_autograd[:, k] = (grad_k * direction).sum(dim=1)
    dv01_autograd = dv01_autograd.detach().numpy().astype(np.float64)

    # ---------------------------------------------------------- 2) finite diff
    def fd_dv01(bump):
        with torch.no_grad():
            x_up = x_disc_f32 * torch.exp(-bump * T_PAY_t32[None, :])
            pred_base = forward_raw(x_disc_f32, x_scalar_f32).numpy().astype(np.float64)
            pred_up   = forward_raw(x_up,        x_scalar_f32).numpy().astype(np.float64)
        return (pred_up - pred_base) / bump

    bumps = [1e-4, 5e-4, 25e-4, 100e-4]
    dv01_fd = {b: fd_dv01(b) for b in bumps}

    # ---------------------------------------------------------------------- report
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

    def corr(y_true, y_pred):
        if y_true.std() < 1e-30 or y_pred.std() < 1e-30:
            return float("nan")
        return np.corrcoef(y_true, y_pred)[0, 1]

    interior = list(range(1, 20))
    print()
    print("Pooled across all interior dates and samples:")
    print(f"{'method':>20}  {'R2':>10}  {'corr':>10}  {'std(pred)':>12}  {'std(ref)':>12}")

    flat_ref = dv01_ref[:, interior].flatten()
    print(f"{'analytical ref':>20}  {'--':>10}  {'--':>10}  {flat_ref.std():12.4e}  {flat_ref.std():12.4e}")

    flat_auto = dv01_autograd[:, interior].flatten()
    print(f"{'exact autograd':>20}  {r2(flat_ref, flat_auto):10.4f}  {corr(flat_ref, flat_auto):10.4f}  "
          f"{flat_auto.std():12.4e}  {flat_ref.std():12.4e}")

    for bmp in bumps:
        flat_fd = dv01_fd[bmp][:, interior].flatten()
        print(f"{'FD bump='+str(bmp):>20}  {r2(flat_ref, flat_fd):10.4f}  {corr(flat_ref, flat_fd):10.4f}  "
              f"{flat_fd.std():12.4e}  {flat_ref.std():12.4e}")

    print()
    print("Per-date detail:")
    print(f"{'t_k':>5}  {'ref std':>10}  {'auto std':>10}  {'auto R2':>10}  {'auto corr':>10}  "
          f"{'fd1bp R2':>10}  {'fd1bp corr':>10}  {'fd100bp R2':>11}  {'fd100bp corr':>12}")
    for k in interior:
        yt = dv01_ref[:, k]
        ya = dv01_autograd[:, k]
        yf1 = dv01_fd[1e-4][:, k]
        yf100 = dv01_fd[100e-4][:, k]
        print(f"{T_MONITOR[k]:5.1f}  {yt.std():10.4e}  {ya.std():10.4e}  {r2(yt,ya):10.4f}  {corr(yt,ya):10.4f}  "
              f"{r2(yt,yf1):10.4f}  {corr(yt,yf1):10.4f}  {r2(yt,yf100):11.4f}  {corr(yt,yf100):12.4f}")


if __name__ == "__main__":
    main()
