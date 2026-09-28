"""
b6_r2_profile_v2.py
===================
Per-date R2 evaluation for v2/v22 sequence-to-sequence surrogate models.

Each model maps (X_disc, X_scalar) -> 21-element profile.
R2(t_k) is computed across all N samples for each date index k,
excluding boundary dates t_0=0 and t_20=10 where labels are zero.

Usage
-----
  python b6_r2_profile_v2.py \
      --checkpoint  data/best_model_1a_v2.pt \
      --data        data/train_1a_v2.h5 \
      --label       1a_v2
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import T_MONITOR

CODE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Per-date R2 for v2/v22 profile models")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--label",      type=str, default="model",
                   help="Short label used in output filename, e.g. 1a_v2")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device",     type=str, default="cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--out_dir",    type=str,
                   default=os.path.join(CODE_DIR, "data"))
    return p.parse_args()


def main():
    import torch
    import h5py
    from c4_model_v2 import IRSSurrogateV2

    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading checkpoint: {args.checkpoint}")
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
    device = torch.device(args.device)
    model.to(device)
    print(f"  Epoch {ckpt['epoch']},  val MSE {ckpt['val_loss']:.6f}")

    # Sobolev checkpoints carry normalisation stats; standard ones do not.
    norm_stats = ckpt.get("norm_stats", None)
    if norm_stats is not None:
        print("  Normalisation stats found in checkpoint (Sobolev model).")

    print(f"Loading data: {args.data}")
    with h5py.File(args.data, "r") as hf:
        X_disc   = hf["X_disc"][:].astype(np.float32)    # (N, 20)
        X_scalar = hf["X_scalar"][:].astype(np.float32)  # (N, 2)
        y_true   = hf["y_price"][:].astype(np.float64)   # (N, 21)
    N = y_true.shape[0]
    print(f"  {N:,} samples")

    X_disc_t   = torch.from_numpy(X_disc).unsqueeze(-1)  # (N, 20, 1)
    X_scalar_t = torch.from_numpy(X_scalar)              # (N, 2)

    # Apply input normalisation for Sobolev models
    if norm_stats is not None:
        xd_mean = norm_stats['xd_mean']   # (1, 20)
        xd_std  = norm_stats['xd_std']
        xs_mean = norm_stats['xs_mean']   # (1, 2)
        xs_std  = norm_stats['xs_std']
        y_mean  = norm_stats['y_mean'].numpy()   # (1, 21)
        y_std   = norm_stats['y_std'].numpy()
        X_disc_n   = (torch.from_numpy(X_disc) - xd_mean) / xd_std
        X_scalar_n = (torch.from_numpy(X_scalar) - xs_mean) / xs_std
        X_disc_t   = X_disc_n.unsqueeze(-1)
        X_scalar_t = X_scalar_n
    else:
        y_mean = 0.0
        y_std  = 1.0

    preds = np.empty((N, 21), dtype=np.float64)
    bs = args.batch_size
    with torch.no_grad():
        for i in range(0, N, bs):
            xd  = X_disc_t[i:i+bs].to(device)
            xs  = X_scalar_t[i:i+bs].to(device)
            out = model(xd, xs).cpu().numpy().astype(np.float64)
            preds[i:i+bs] = out

    # Denormalise predictions back to original price units
    if norm_stats is not None:
        preds = preds * y_std + y_mean

    rows = []
    for k, t_k in enumerate(T_MONITOR):
        yt = y_true[:, k]
        yp = preds[:, k]
        if t_k < 1e-8 or t_k >= 10.0 - 1e-8:
            rows.append({"time": t_k, "n": N, "R2": np.nan})
            continue
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1e-20 else np.nan
        rows.append({"time": t_k, "n": N, "R2": float(r2)})

    df = pd.DataFrame(rows)
    final_r2 = float(np.nanmean(df["R2"]))

    hdr = f"{'time':>6}  {'n':>10}  {'R2':>10}"
    print()
    print(hdr)
    print("-" * len(hdr))
    for _, row in df.iterrows():
        r2_str = f"{row['R2']:.6f}" if not np.isnan(row["R2"]) else "     NaN"
        print(f"{row['time']:>6.1f}  {int(row['n']):>10,}  {r2_str:>10}")
    print("-" * len(hdr))
    print(f"{'Final R2 (mean)':>18}  {final_r2:.6f}")

    out = os.path.join(args.out_dir, f"r2_profile_{args.label}.csv")
    df.to_csv(out, index=False, float_format="%.6f")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
