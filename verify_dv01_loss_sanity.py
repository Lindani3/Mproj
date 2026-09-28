"""
verify_dv01_loss_sanity.py
===========================
One-off sanity check: does c5_train_v2_dv01.py's analytical_dv01_batch()
reproduce b8_dv01_profile.py's _ref_profile_from_disc() exactly?

Run this BEFORE any training with c5_train_v2_dv01.py. It should print
"allclose: True" and a max relative difference at float64 precision
(around 1e-10 or smaller). If it does not, do not train against it.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

from hw_utils import T_MONITOR, T_PAY, TAU, load_all_svensson_params, market_discount
import b8_dv01_profile as b8
from c5_train_v2_dv01 import analytical_dv01_batch

all_params = load_all_svensson_params("data/feds_1000.csv")
params_sub = all_params[:8]

X_disc = np.stack([market_discount(T_PAY, p).astype(np.float64) for p in params_sub])
a_vals   = np.array([0.05, 0.10, 0.15, 0.20, 0.08, 0.25, 0.12, 0.30])
sig_vals = np.array([0.010, 0.015, 0.020, 0.008, 0.025, 0.005, 0.030, 0.012])
X_scalar = np.stack([a_vals, sig_vals], axis=1)

K_par = (1.0 - X_disc[:, -1]) / (TAU * X_disc.sum(axis=1))
V_base_ref = b8._ref_profile_from_disc(X_disc, X_scalar, bump=0.0, K_par=K_par)
V_up_ref   = b8._ref_profile_from_disc(X_disc, X_scalar, bump=b8.BUMP_BPS, K_par=K_par)
dv01_ref_np = V_up_ref - V_base_ref

device = torch.device("cpu")
T_PAY_t = torch.tensor(T_PAY, dtype=torch.float64, device=device)
T_MON_t = torch.tensor(T_MONITOR, dtype=torch.float64, device=device)
x_disc_t   = torch.tensor(X_disc, dtype=torch.float64, device=device)
x_scalar_t = torch.tensor(X_scalar, dtype=torch.float64, device=device)

dv01_torch = analytical_dv01_batch(x_disc_t, x_scalar_t, T_PAY_t, T_MON_t, float(TAU)).numpy()

diff = np.abs(dv01_torch - dv01_ref_np)
rel  = diff / (np.abs(dv01_ref_np) + 1e-12)
mask = np.abs(dv01_ref_np) > 1e-8

print("max abs diff:", diff.max())
print("max rel diff (|ref|>1e-8):", rel[mask].max() if mask.any() else float("nan"))
print("allclose (atol=1e-8, rtol=1e-5):", np.allclose(dv01_torch, dv01_ref_np, atol=1e-8, rtol=1e-5))
print()
print("Sample row 0, interior dates:")
for k in range(1, 20):
    print(f"  t={T_MONITOR[k]:5.1f}  ref={dv01_ref_np[0,k]: .6e}  torch={dv01_torch[0,k]: .6e}")
