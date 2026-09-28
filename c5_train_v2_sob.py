"""
c5_train_v2_sob.py
==================
Sobolev training for v2 surrogate models (sequence-to-sequence).

Loss function
-------------
    L = alpha * MSE(ỹ_pred, ỹ_true) + lambda * Sobolev_term

All inputs and outputs are normalised to zero mean / unit variance (computed
from the training split only).  The Jacobian labels are rescaled accordingly:

    d(ỹ_k)/d(x̃_j)  =  (dJ_kj) * sigma_xj / sigma_yk

This ensures the Sobolev term is dimensionless and comparable in magnitude to
the price MSE term, following the recommendation of Huge & Savine (2020).

The Sobolev term is estimated via random projection (one backward pass per
batch — unbiased and efficient):

    v ~ N(0,1)   shape (B, 21)
    vTy = sum(v * ỹ_pred)                    # scalar
    Jv_pred = grad(vTy, x̃_input)             # (B, d)
    Jv_true = einsum('bkd,bk->bd', J̃, v)    # (B, d)
    Sobolev_term = MSE(Jv_pred, Jv_true)

For Model 1a datasets: dJ_disc (N,21,20) and dJ_scalar (N,21,2) present.
For 1b / 2 datasets:  dJ_scalar (N,21,2) only.
The script detects this automatically from the HDF5 file.

Usage
-----
  python c5_train_v2_sob.py --model 1a --data data/train_1a_v2_sob.h5
  python c5_train_v2_sob.py --model 1b --data data/train_1b_v2_sob.h5
  python c5_train_v2_sob.py --model 2  --data data/train_2_v2_sob.h5
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_model_v2 import IRSSurrogateV2


def parse_args():
    p = argparse.ArgumentParser(description="GRU v2 Sobolev surrogate training")
    p.add_argument("--model",       type=str,   required=True, choices=["1a", "1b", "2"])
    p.add_argument("--data",        type=str,   required=True)
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--n_layers",    type=int,   default=2)
    p.add_argument("--lam",         type=float, default=1.0,
                   help="Weight on the Sobolev term (applied after normalisation)")
    p.add_argument("--alpha",       type=float, default=1.0,
                   help="Weight on the price MSE term")
    p.add_argument("--device",      type=str,   default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num_threads", type=int,   default=8)
    p.add_argument("--save",        type=str,   default=None)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_norm_stats(X_disc, X_scalar, y_price, tr_idx):
    """Compute normalisation statistics from the training split only."""
    eps = 1e-8
    xd_mean = X_disc[tr_idx].mean(dim=0, keepdim=True)
    xd_std  = X_disc[tr_idx].std(dim=0,  keepdim=True).clamp(min=eps)
    xs_mean = X_scalar[tr_idx].mean(dim=0, keepdim=True)
    xs_std  = X_scalar[tr_idx].std(dim=0,  keepdim=True).clamp(min=eps)
    y_mean  = y_price[tr_idx].mean(dim=0, keepdim=True)
    y_std   = y_price[tr_idx].std(dim=0,  keepdim=True).clamp(min=eps)
    return dict(xd_mean=xd_mean, xd_std=xd_std,
                xs_mean=xs_mean, xs_std=xs_std,
                y_mean=y_mean,   y_std=y_std)


def apply_normalisation(X_disc, X_scalar, y_price, dJ_disc, dJ_scalar,
                        has_disc, stats):
    """
    Normalise to zero mean / unit variance.

    Jacobian transformation:
        d(ỹ_k)/d(x̃_j) = dJ_kj * sigma_xj / sigma_yk

    dJ_disc   shape (N, 21, 20)  : axis -1 is input dim j
    dJ_scalar shape (N, 21,  2)  : axis -1 is input dim j
    xd_std    shape ( 1, 20)     → unsqueeze to (1,  1, 20)
    xs_std    shape ( 1,  2)     → unsqueeze to (1,  1,  2)
    y_std     shape ( 1, 21)     → unsqueeze to (1, 21,  1)
    """
    xd = (X_disc   - stats['xd_mean']) / stats['xd_std']
    xs = (X_scalar - stats['xs_mean']) / stats['xs_std']
    y  = (y_price  - stats['y_mean'])  / stats['y_std']

    if has_disc and dJ_disc is not None:
        dJd = dJ_disc * (stats['xd_std'].unsqueeze(1) /
                         stats['y_std'].unsqueeze(-1))
    else:
        dJd = None

    dJs = dJ_scalar * (stats['xs_std'].unsqueeze(1) /
                       stats['y_std'].unsqueeze(-1))
    return xd, xs, y, dJd, dJs


def load_dataset(path, batch_size, device):
    print(f"Loading {path} ...", flush=True)
    t0 = time.time()
    with h5py.File(path, "r") as hf:
        X_disc    = torch.from_numpy(hf["X_disc"][:].astype(np.float32))
        X_scalar  = torch.from_numpy(hf["X_scalar"][:].astype(np.float32))
        y_price   = torch.from_numpy(hf["y_price"][:].astype(np.float32))
        dJ_scalar = torch.from_numpy(hf["dJ_scalar"][:].astype(np.float32))
        has_disc  = "dJ_disc" in hf
        dJ_disc   = torch.from_numpy(hf["dJ_disc"][:].astype(np.float32)) if has_disc else None

    N       = len(y_price)
    idx     = torch.randperm(N)
    n_train = int(0.8 * N)
    tr, va  = idx[:n_train], idx[n_train:]

    stats = compute_norm_stats(X_disc, X_scalar, y_price, tr)
    print(f"  Normalisation: y_std mean={stats['y_std'].mean().item():.4f}  "
          f"xd_std mean={stats['xd_std'].mean().item():.4f}  "
          f"xs_std mean={stats['xs_std'].mean().item():.4f}", flush=True)

    Xd_n, Xs_n, y_n, dJd_n, dJs_n = apply_normalisation(
        X_disc, X_scalar, y_price, dJ_disc, dJ_scalar, has_disc, stats)

    def make_ds(sel):
        tensors = [Xd_n[sel], Xs_n[sel], y_n[sel], dJs_n[sel]]
        if has_disc:
            tensors.append(dJd_n[sel])
        return TensorDataset(*tensors)

    pin = (device.type == "cuda")
    tr_loader = DataLoader(make_ds(tr), batch_size=batch_size, shuffle=True,
                           pin_memory=pin, num_workers=4)
    va_loader = DataLoader(make_ds(va), batch_size=batch_size, shuffle=False,
                           pin_memory=pin, num_workers=4)

    print(f"  {N:,} samples  dJ_disc={'yes' if has_disc else 'no'}  ({time.time()-t0:.1f}s)",
          flush=True)
    return tr_loader, va_loader, has_disc, stats


def _compute_loss(model, x_disc, x_scalar, y_true, dJ_scalar,
                  dJ_disc, has_disc, lam, alpha):
    """
    Forward pass + Sobolev loss via random projection.
    All inputs and labels are already normalised.
    Returns (total_loss, price_mse_item).
    """
    if has_disc:
        xd = x_disc.detach().requires_grad_(True)
        xs = x_scalar.detach().requires_grad_(True)
    else:
        xd = x_disc.detach()
        xs = x_scalar.detach().requires_grad_(True)

    y_pred = model(xd.unsqueeze(-1), xs)          # (B, 21)
    price_loss = F.mse_loss(y_pred, y_true)

    v   = torch.randn_like(y_pred)                # (B, 21)
    vTy = (v * y_pred).sum()

    if has_disc:
        gd, gs = torch.autograd.grad(vTy, [xd, xs], create_graph=True)
        Jv_disc   = (dJ_disc   * v.unsqueeze(-1)).sum(dim=1)   # (B, 20)
        Jv_scalar = (dJ_scalar * v.unsqueeze(-1)).sum(dim=1)   # (B, 2)
        sob = F.mse_loss(gd, Jv_disc) + F.mse_loss(gs, Jv_scalar)
    else:
        gs, = torch.autograd.grad(vTy, [xs], create_graph=True)
        Jv_scalar = (dJ_scalar * v.unsqueeze(-1)).sum(dim=1)   # (B, 2)
        sob = F.mse_loss(gs, Jv_scalar)

    total = alpha * price_loss + lam * sob
    return total, price_loss.item()


def train_epoch(model, loader, optimiser, device, has_disc, lam, alpha):
    model.train()
    tot_loss, tot_price, n = 0.0, 0.0, 0
    for batch in loader:
        if has_disc:
            xd, xs, yt, djs, djd = [b.to(device) for b in batch]
        else:
            xd, xs, yt, djs = [b.to(device) for b in batch]
            djd = None

        optimiser.zero_grad()
        loss, price = _compute_loss(model, xd, xs, yt, djs, djd, has_disc, lam, alpha)
        loss.backward()
        optimiser.step()

        tot_loss  += loss.item()
        tot_price += price
        n += 1
    return tot_loss / max(n, 1), tot_price / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    """Validation MSE on normalised prices (for checkpoint selection)."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        xd, xs, yt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        pred = model(xd.unsqueeze(-1), xs)
        total += F.mse_loss(pred, yt).item()
        n += 1
    return total / max(n, 1)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    torch.set_num_threads(args.num_threads)

    if args.save is None:
        data_dir  = os.path.dirname(os.path.abspath(args.data))
        args.save = os.path.join(data_dir, f"best_model_{args.model}_v2_sob.pt")
    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    print(f"Model    : {args.model} v2 Sobolev")
    print(f"Data     : {args.data}")
    print(f"Device   : {device}")
    print(f"alpha={args.alpha}  lambda={args.lam}")
    print(f"Save     : {args.save}")

    tr_loader, va_loader, has_disc, stats = load_dataset(
        args.data, args.batch_size, device)
    print(f"Train: {len(tr_loader.dataset):,}  Val: {len(va_loader.dataset):,}")
    print(f"Sobolev inputs: {'disc + scalar' if has_disc else 'scalar only'}")

    model = IRSSurrogateV2(
        n_yields=20, n_scalar=2, n_out=21,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=10, factor=0.5,
    )

    # Move norm stats to device for checkpoint serialisation (keep on CPU)
    stats_cpu = {k: v.cpu() for k, v in stats.items()}

    best_val = float("inf")
    t0 = time.time()
    hdr = f"{'Epoch':>6}  {'Total':>12}  {'Price':>12}  {'Val MSE':>12}  {'LR':>10}  {'Time':>8}"
    print()
    print(hdr)
    print("-" * len(hdr))

    for epoch in range(1, args.epochs + 1):
        tr_tot, tr_pr = train_epoch(model, tr_loader, optimiser, device,
                                    has_disc, args.lam, args.alpha)
        va_loss = val_epoch(model, va_loader, device)

        scheduler.step(va_loss)
        lr   = optimiser.param_groups[0]["lr"]
        flag = ""

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "epoch":      epoch,
                "model_type": f"{args.model}_v2_sob",
                "n_yields":   20,
                "n_scalar":   2,
                "n_out":      21,
                "hidden_dim": args.hidden_dim,
                "n_layers":   args.n_layers,
                "val_loss":   best_val,
                "lam":        args.lam,
                "alpha":      args.alpha,
                "norm_stats": stats_cpu,
                "state_dict": model.state_dict(),
            }, args.save)
            flag = " *"

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_tot:>12.6f}  {tr_pr:>12.6f}  {va_loss:>12.6f}  "
              f"{lr:>10.2e}  {elapsed:>7.0f}s{flag}")

    print(f"\nBest val MSE: {best_val:.6f}   Checkpoint: {args.save}")


if __name__ == "__main__":
    main()
