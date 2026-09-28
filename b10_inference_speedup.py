"""
b10_inference_speedup.py
=========================
Timing comparison at realistic industry scale: surrogate batch forward
pass vs. the real Monte Carlo reference procedure, for B = 5,000
scenarios (matching Green (2016), "XVA: Credit, Funding and Capital
Value Adjustments", p.385: "Assume that the XVA simulation uses 5000
paths..." -- also matches J = 5,000 already used throughout this study
for Monte Carlo label generation).

A single-sample (B=1) timing test understates the surrogate's real
advantage and overstates the reference's, since PyTorch has fixed
per-call overhead that dominates at B=1, while the surrogate's real
advantage is batched throughput (one forward pass over many scenarios
costs barely more than one scenario). This version times B=5,000
scenarios through both paths.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import load_all_svensson_params, par_swap_rate, T_PAY, T_MONITOR, market_discount
from c1_datagen_1b_v2 import mc_profile
from c4_model import IRSSurrogate

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
N_REPEATS = 5
B = 5000  # scenarios per batch, matching Green (2016) p.385's "5000 paths" example

feds_path = os.path.join(CODE_DIR, "data", "feds_1000.csv")
all_params = load_all_svensson_params(feds_path)
p = all_params[0]
a, sigma = 0.10, 0.015
K = par_swap_rate(p, T_PAY)

print(f"Representative curve: {feds_path}, param set 0")
print(f"Hull-White params: a={a}, sigma={sigma}, K_par={K:.6f}")
print(f"Batch size B = {B} scenarios (Green 2016, p.385)")

# -- reference: B independent mc_profile() calls, J = 5,000 paths each -------
# -- (the real procedure used to build Model 1b's training labels) -----------
J = 5000
rng = np.random.default_rng(42)

times_ref = []
for _ in range(N_REPEATS):
    t0 = time.perf_counter()
    for _ in range(B):
        y_ref = mc_profile(K, p, a, sigma, rng, J)
    times_ref.append(time.perf_counter() - t0)
t_ref = float(np.median(times_ref))
print(f"\nReference: {B} scenarios x mc_profile(J={J}): "
      f"median {t_ref:.3f} s over {N_REPEATS} repeats "
      f"({t_ref/B*1000:.4f} ms/scenario)")

# -- surrogate: std architecture, batched -------------------------------------
ckpt = torch.load(os.path.join(CODE_DIR, "data", "best_model_1a.pt"),
                   map_location="cpu", weights_only=False)
model = IRSSurrogate(n_yields=20, n_scalar=3, hidden_dim=128, n_layers=2)
model.load_state_dict(ckpt["state_dict"])
model.eval()

disc = market_discount(T_PAY, p).astype(np.float32)
x_disc_one = torch.from_numpy(disc).view(1, 20, 1)
x_disc_batch = x_disc_one.repeat(B, 1, 1)  # (B, 20, 1), same curve B times

active_dates = [t_k for t_k in T_MONITOR if not (t_k < 1e-8 or t_k >= 10.0 - 1e-8)]

times_std = []
with torch.no_grad():
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        for t_k in active_dates:
            x_scalar = torch.tensor([[a, sigma, float(t_k)]], dtype=torch.float32).repeat(B, 1)
            _ = model(x_disc_batch, x_scalar)
        times_std.append(time.perf_counter() - t0)
t_std = float(np.median(times_std))
print(f"\nSurrogate (std, batched, 21 forward calls over B={B}): "
      f"median {t_std:.3f} s over {N_REPEATS} repeats "
      f"({t_std/B*1000:.4f} ms/scenario)")
print(f"  -> speedup vs reference: {t_ref/t_std:,.1f}x")

# -- surrogate: v2/profile architecture, batched, single forward call --------
v2_ckpt_path = os.path.join(CODE_DIR, "data", "best_model_1a_v22.pt")
if os.path.exists(v2_ckpt_path):
    from c4_model_v2 import IRSSurrogateV2
    ckpt2 = torch.load(v2_ckpt_path, map_location="cpu", weights_only=False)
    model2 = IRSSurrogateV2(n_yields=20, n_scalar=2, n_out=21, hidden_dim=128, n_layers=2)
    model2.load_state_dict(ckpt2["state_dict"])
    model2.eval()
    x_scalar2 = torch.tensor([[a, sigma]], dtype=torch.float32).repeat(B, 1)
    times_v2 = []
    with torch.no_grad():
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            _ = model2(x_disc_batch, x_scalar2)
            times_v2.append(time.perf_counter() - t0)
    t_v2 = float(np.median(times_v2))
    print(f"\nSurrogate (v2/profile, batched, 1 forward call over B={B}): "
          f"median {t_v2:.4f} s over {N_REPEATS} repeats "
          f"({t_v2/B*1000:.4f} ms/scenario)")
    print(f"  -> speedup vs reference: {t_ref/t_v2:,.0f}x")
    print(f"  -> speedup vs std architecture: {t_std/t_v2:,.1f}x")
else:
    print(f"\n(v2 checkpoint not found at {v2_ckpt_path})")
