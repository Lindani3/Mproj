"""
c5_train_v2_dv01.py
====================
Staged experiment: does supervising the network on the analytical HW DV01
(the curve-derivative Jacobian, fixed-K convention) fix the DV01 R^2
collapse seen in b8_dv01_profile.py?

Identical to c5_train_v2.py (same architecture, optimiser, LR schedule,
train/val split) except for one addition: an analytical DV01 loss term,
computed on the fly, added to the value MSE.

The analytical_dv01_batch() function below is a batched torch translation
of b8_dv01_profile.py's _ref_profile_from_disc(), verified line-by-line
against it (fixed K from base curve; f0t always from base log-discounts;
f0t_bump = f0t + bump; single fused pass over base and bumped legs sharing
a, sigma-dependent terms). Do not vectorise the f0t computation across k in
a single precomputed array -- the per-k loop below is required because the
monitor-date index k does not align one-to-one with the pay-date column
index in a way that a flat slice can reproduce (verified: an earlier
vectorised attempt at this introduced a silent off-by-one in the forward
curve).

Usage
-----
  python c5_train_v2_dv01.py --model 1a --data data/train_1a_v2.h5
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c4_model_v2 import IRSSurrogateV2, mse_loss
from hw_utils import T_MONITOR, T_PAY, TAU

A_FLOOR = 1e-4
BUMP    = 1e-4  # 1bp, matches b8_dv01_profile.py's BUMP_BPS


# ---------------------------------------------------------------------------
# Analytical DV01 reference, batched (torch, differentiable-safe: called
# under no_grad for the target; the surrogate's own DV01 is obtained via a
# real second forward pass so it stays in the autograd graph).
# ---------------------------------------------------------------------------

def analytical_dv01_batch(x_disc: torch.Tensor, x_scalar: torch.Tensor,
                           T_PAY_t: torch.Tensor, T_MON_t: torch.Tensor,
                           tau: float) -> torch.Tensor:
    """
    x_disc   : (B, 20) base discount factors P(0, T_j)
    x_scalar : (B,  2) [a, sigma]
    returns  : (B, 21) analytical DV01 per monitoring date, fixed K from the
               base curve. Rows 0 and 20 are zero (boundary dates).
    """
    Bsz    = x_disc.shape[0]
    device = x_disc.device
    dtype  = x_disc.dtype

    a_raw = x_scalar[:, 0]
    sig   = x_scalar[:, 1]
    a     = torch.clamp(a_raw, min=A_FLOOR)

    K = (1.0 - x_disc[:, -1]) / (tau * x_disc.sum(dim=1))          # (B,)

    disc_up  = x_disc * torch.exp(-BUMP * T_PAY_t)                  # (B, 20)
    log_base = torch.log(x_disc)                                    # (B, 20) always base curve

    V_base = torch.zeros(Bsz, 21, device=device, dtype=dtype)
    V_up   = torch.zeros(Bsz, 21, device=device, dtype=dtype)

    for k in range(1, 20):
        t_k = float(T_MON_t[k])

        if k == 1:
            f0t = -(log_base[:, 1] - log_base[:, 0]) / tau
        else:
            f0t = -(log_base[:, k] - log_base[:, k - 2]) / (2.0 * tau)

        ea  = torch.exp(-a * t_k)
        e2a = torch.exp(-2.0 * a * t_k)
        s2  = (sig**2 / (2.0 * a)) * (1.0 - e2a)                    # (B,)

        rem_mask = T_PAY_t > t_k + 1e-9
        T_rem    = T_PAY_t[rem_mask]
        B_kj = ((1.0 - torch.exp(-a[:, None] * (T_rem[None, :] - t_k)))
                / a[:, None])                                        # (B, n_rem)

        # ---- base leg (bump = 0) ----
        mu_b = f0t + (sig**2 / (2.0 * a**2)) * (1.0 - ea)**2
        disc_rem_b = x_disc[:, rem_mask]
        disc_t_b   = x_disc[:, k - 1]
        lnA_b = (torch.log(disc_rem_b / disc_t_b[:, None])
                 + B_kj * f0t[:, None]
                 - (sig[:, None]**2 / (4.0 * a[:, None]))
                   * B_kj**2 * (1.0 - e2a[:, None]))
        E_P_b = torch.exp(lnA_b - B_kj * mu_b[:, None] + 0.5 * B_kj**2 * s2[:, None])
        V_base[:, k] = (1.0 - E_P_b[:, -1]) - K * tau * E_P_b.sum(dim=1)

        # ---- bumped leg ----
        f0t_up = f0t + BUMP
        mu_u   = f0t_up + (sig**2 / (2.0 * a**2)) * (1.0 - ea)**2
        disc_rem_u = disc_up[:, rem_mask]
        disc_t_u   = disc_up[:, k - 1]
        lnA_u = (torch.log(disc_rem_u / disc_t_u[:, None])
                 + B_kj * f0t_up[:, None]
                 - (sig[:, None]**2 / (4.0 * a[:, None]))
                   * B_kj**2 * (1.0 - e2a[:, None]))
        E_P_u = torch.exp(lnA_u - B_kj * mu_u[:, None] + 0.5 * B_kj**2 * s2[:, None])
        V_up[:, k] = (1.0 - E_P_u[:, -1]) - K * tau * E_P_u.sum(dim=1)

    return V_up - V_base


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRU v2 surrogate training with DV01 supervision")
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
    p.add_argument("--lambda_d",   type=float, default=None,
                   help="DV01 loss weight. If omitted, auto-computed from "
                        "the variance ratio of one training batch.")
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
        X_disc   = torch.from_numpy(hf["X_disc"][:].astype(np.float32))
        X_scalar = torch.from_numpy(hf["X_scalar"][:].astype(np.float32))
        y_price  = torch.from_numpy(hf["y_price"][:].astype(np.float32))
    print(f"  Loaded {len(y_price):,} samples in {time.time()-t0:.1f}s", flush=True)

    X_disc_seq = X_disc.unsqueeze(-1)

    N       = len(y_price)
    idx     = torch.randperm(N)
    n_train = int(0.8 * N)
    tr_idx  = idx[:n_train]
    va_idx  = idx[n_train:]

    tr_ds = TensorDataset(X_disc_seq[tr_idx], X_scalar[tr_idx], y_price[tr_idx], X_disc[tr_idx])
    va_ds = TensorDataset(X_disc_seq[va_idx], X_scalar[va_idx], y_price[va_idx], X_disc[va_idx])

    pin = (device.type == "cuda")
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           pin_memory=pin, num_workers=4)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                           pin_memory=pin, num_workers=4)
    return tr_loader, va_loader


def train_epoch(model, loader, optimiser, device, T_PAY_t, T_MON_t, tau,
                 lambda_d) -> tuple[float, float]:
    model.train()
    total_val, total_dv01, n = 0.0, 0.0, 0
    interior = slice(1, 20)
    for x_disc_seq, x_scalar, y_profile, x_disc in loader:
        x_disc_seq = x_disc_seq.to(device)
        x_scalar   = x_scalar.to(device)
        y_profile  = y_profile.to(device)
        x_disc     = x_disc.to(device)

        optimiser.zero_grad()

        pred = model(x_disc_seq, x_scalar)                          # (B, 21)
        loss_val = mse_loss(pred, y_profile)

        x_disc_up_seq = (x_disc * torch.exp(-BUMP * T_PAY_t)).unsqueeze(-1)
        pred_up   = model(x_disc_up_seq, x_scalar)                  # (B, 21)
        dv01_pred = (pred_up - pred) / BUMP

        with torch.no_grad():
            dv01_ref = analytical_dv01_batch(x_disc, x_scalar, T_PAY_t, T_MON_t, tau)

        loss_dv01 = F.mse_loss(dv01_pred[:, interior], dv01_ref[:, interior])
        loss = loss_val + lambda_d * loss_dv01
        loss.backward()
        optimiser.step()

        total_val  += loss_val.item()
        total_dv01 += loss_dv01.item()
        n += 1
    return total_val / max(n, 1), total_dv01 / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x_disc_seq, x_scalar, y_profile, _ in loader:
        x_disc_seq = x_disc_seq.to(device)
        x_scalar   = x_scalar.to(device)
        y_profile  = y_profile.to(device)

        pred = model(x_disc_seq, x_scalar)
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
        args.save = os.path.join(data_dir, f"best_model_{args.model}_v2_dv01.pt")

    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)

    print(f"Model type : {args.model} v2 + DV01 supervision")
    print(f"Data file  : {args.data}")
    print(f"Device     : {device}")
    print(f"Checkpoint : {args.save}")

    tr_loader, va_loader = load_dataset(args.data, args.batch_size, device)
    print(f"Train: {len(tr_loader.dataset):,}  Val: {len(va_loader.dataset):,}")

    T_PAY_t = torch.tensor(T_PAY, dtype=torch.float32, device=device)
    T_MON_t = torch.tensor(T_MONITOR, dtype=torch.float32, device=device)
    tau     = float(TAU)

    model = IRSSurrogateV2(
        n_yields   = 20,
        n_scalar   = 2,
        n_out      = 21,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ---- lambda_d: auto-compute from one batch if not given ----
    if args.lambda_d is None:
        x_disc_seq0, x_scalar0, y0, x_disc0 = next(iter(tr_loader))
        x_disc0   = x_disc0.to(device)
        x_scalar0 = x_scalar0.to(device)
        y0        = y0.to(device)
        with torch.no_grad():
            dv01_ref0 = analytical_dv01_batch(x_disc0, x_scalar0, T_PAY_t, T_MON_t, tau)
        interior = slice(1, 20)
        var_y    = y0[:, interior].var().item()
        var_dv01 = dv01_ref0[:, interior].var().item()
        lambda_d = var_y / (var_dv01 + 1e-30)
        print(f"Auto lambda_d = {lambda_d:.3e}  (var_y={var_y:.3e}, var_dv01={var_dv01:.3e})")
    else:
        lambda_d = args.lambda_d
        print(f"Using provided lambda_d = {lambda_d:.3e}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=10, factor=0.5,
    )

    best_val = float("inf")
    t0       = time.time()
    header   = (f"{'Epoch':>6}  {'Train MSE':>12}  {'Train DV01':>12}  "
                f"{'Val MSE':>12}  {'LR':>10}  {'Time':>8}")
    print()
    print(header)
    print("-" * len(header))

    for epoch in range(1, args.epochs + 1):
        tr_val, tr_dv01 = train_epoch(model, tr_loader, optimiser, device,
                                       T_PAY_t, T_MON_t, tau, lambda_d)
        va_loss = val_epoch(model, va_loader, device)

        scheduler.step(va_loss)
        lr   = optimiser.param_groups[0]["lr"]
        flag = ""

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "epoch":      epoch,
                "model_type": args.model + "_v2_dv01",
                "n_yields":   20,
                "n_scalar":   2,
                "n_out":      21,
                "hidden_dim": args.hidden_dim,
                "n_layers":   args.n_layers,
                "val_loss":   best_val,
                "lambda_d":   lambda_d,
                "state_dict": model.state_dict(),
            }, args.save)
            flag = " *"

        elapsed = time.time() - t0
        print(f"{epoch:>6}  {tr_val:>12.6f}  {tr_dv01:>12.6e}  "
              f"{va_loss:>12.6f}  {lr:>10.2e}  {elapsed:>7.0f}s{flag}")

    print()
    print(f"Training complete. Best val MSE: {best_val:.6f}")
    print(f"Checkpoint: {args.save}")


if __name__ == "__main__":
    main()
