"""
c5_train_v2.py
==============
Training script for v2 surrogate models (sequence-to-sequence).

Usage
-----
  python c5_train_v2.py --model 1a --data data/train_1a_v2.h5
  python c5_train_v2.py --model 1b --data data/train_1b_v2.h5
  python c5_train_v2.py --model 2  --data data/train_2_v2.h5

Input data (HDF5)
-----------------
  X_disc   (N, 20)   Svensson discount factors
  X_scalar (N,  2)   [a, sigma]
  y_price  (N, 21)   full label profile (V_affine, V_MC, or EPE per date)

All models use MSE loss over the full 21-element output profile.
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_model_v2 import IRSSurrogateV2, mse_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRU v2 surrogate training")
    p.add_argument("--model",      type=str, required=True, choices=["1a", "1b", "2"])
    p.add_argument("--data",       type=str, required=True)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--device",     type=str,   default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--num_threads",type=int,   default=8)
    p.add_argument("--save",       type=str,   default=None)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: str, batch_size: int, device: torch.device):
    print(f"Loading {path} ...", flush=True)
    t0 = time.time()
    with h5py.File(path, "r") as hf:
        X_disc   = torch.from_numpy(hf["X_disc"][:].astype(np.float32))    # (N, 20)
        X_scalar = torch.from_numpy(hf["X_scalar"][:].astype(np.float32))  # (N, 2)
        y_price  = torch.from_numpy(hf["y_price"][:].astype(np.float32))   # (N, 21)
    print(f"  Loaded {len(y_price):,} samples in {time.time()-t0:.1f}s", flush=True)

    X_disc_seq = X_disc.unsqueeze(-1)   # (N, 20, 1)

    N       = len(y_price)
    idx     = torch.randperm(N)
    n_train = int(0.8 * N)
    tr_idx  = idx[:n_train]
    va_idx  = idx[n_train:]

    tr_ds = TensorDataset(X_disc_seq[tr_idx], X_scalar[tr_idx], y_price[tr_idx])
    va_ds = TensorDataset(X_disc_seq[va_idx], X_scalar[va_idx], y_price[va_idx])

    pin = (device.type == "cuda")
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           pin_memory=pin, num_workers=4)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           pin_memory=pin, num_workers=4)
    return tr_loader, va_loader


def train_epoch(model, loader, optimiser, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x_disc, x_scalar, y_profile in loader:
        x_disc   = x_disc.to(device)
        x_scalar = x_scalar.to(device)
        y_profile = y_profile.to(device)

        optimiser.zero_grad()
        pred = model(x_disc, x_scalar)   # (B, 21)
        loss = mse_loss(pred, y_profile)
        loss.backward()
        optimiser.step()

        total += loss.item()
        n     += 1
    return total / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x_disc, x_scalar, y_profile in loader:
        x_disc    = x_disc.to(device)
        x_scalar  = x_scalar.to(device)
        y_profile = y_profile.to(device)

        pred = model(x_disc, x_scalar)
        loss = mse_loss(pred, y_profile)

        total += loss.item()
        n     += 1
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    torch.set_num_threads(args.num_threads)

    if args.save is None:
        data_dir  = os.path.dirname(os.path.abspath(args.data))
        args.save = os.path.join(data_dir, f"best_model_{args.model}_v2.pt")

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    print(f"Model type : {args.model} v2")
    print(f"Data file  : {args.data}")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.save}")

    tr_loader, va_loader = load_dataset(args.data, args.batch_size, device)
    print(f"Train: {len(tr_loader.dataset):,}  Val: {len(va_loader.dataset):,}")

    model = IRSSurrogateV2(
        n_yields   = 20,
        n_scalar   = 2,
        n_out      = 21,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=10, factor=0.5,
    )

    best_val = float("inf")
    t0       = time.time()
    header   = f"{'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>10}  {'Time':>8}"
    print()
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, tr_loader, optimiser, device)
        va_loss = val_epoch(model, va_loader, device)

        scheduler.step(va_loss)
        lr   = optimiser.param_groups[0]["lr"]
        flag = ""

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "epoch":      epoch,
                "model_type": args.model + "_v2",
                "n_yields":   20,
                "n_scalar":   2,
                "n_out":      21,
                "hidden_dim": args.hidden_dim,
                "n_layers":   args.n_layers,
                "val_loss":   best_val,
                "state_dict": model.state_dict(),
            }, args.save)
            flag = " *"

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_loss:>12.6f}  {va_loss:>12.6f}  "
              f"{lr:>10.2e}  {elapsed:>7.0f}s{flag}")

    print()
    print(f"Training complete. Best val MSE: {best_val:.6f}")
    print(f"Checkpoint: {args.save}")


if __name__ == "__main__":
    main()
