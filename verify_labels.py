"""
verify_labels.py
================
Verify Model 1a, 1b, and 2 labels on a single FEDS yield curve.

For each of 21 semi-annual monitoring dates t_k in [0.0, 0.5, ..., 10.0]:

  Model 1a : E[V_affine(t_k)]  -- analytical via MGF of r_{t_k} ~ N(mu, s^2)
  Model 1b : E[V_MC(t_k)]      -- outer-path average of inner-MC IRS values
  Model 2  : EPE(t_k)          -- E[max(V_MC(t_k), 0)]

Boundary checks:
  t=0.0  -> E[V] = 0  (ATM swap at inception)
  t=10.0 -> E[V] = 0  (swap expired)
  Model 1a should agree closely with Model 1b (within MC noise)
  EPE >= 0 everywhere

Usage:
    python verify_labels.py --feds_idx 0 --n_outer 2000 --m_inner 500 --seed 42
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    HW_A, HW_SIGMA, T_PAY, TAU, MATURITY, SUB_STEPS,
    _hw_B, _hw_lnA,
    compute_irs_pv, load_all_svensson_params, market_discount,
    par_swap_rate, simulate_hw_paths, svensson_forward, hw_theta,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "feds200628.csv")

# 21 semi-annual monitoring dates: 0.0, 0.5, 1.0, ..., 10.0
T_MONITOR_NEW = np.array([round(k * 0.5, 1) for k in range(21)])


# ---------------------------------------------------------------------------
# Model 1a: analytical expected IRS value via MGF
# ---------------------------------------------------------------------------

def irs_expected_affine(t_k: float, K: float, p: dict,
                        a: float, sigma: float) -> float:
    """
    Analytical E[V(t_k)] under Hull-White using the moment generating function
    of r_{t_k} ~ N(mu_{t_k}, s^2_{t_k}).

        mu  = f(0,t_k) + sigma^2/(2a^2) * (1 - exp(-a*t_k))^2
        s^2 = sigma^2/(2a) * (1 - exp(-2a*t_k))

        E[P(t_k, T_j)] = exp(lnA(t_k,T_j) - B_j*mu + 0.5*B_j^2*s^2)
        E[V(t_k)]      = [1 - E[P_n]] - K*tau * sum_{T_j > t_k} E[P_j]
    """
    if t_k < 1e-8:
        return 0.0          # ATM at inception: V(0, r0) = 0 by construction
    if t_k >= MATURITY - 1e-8:
        return 0.0          # swap expired

    f0t = float(svensson_forward(t_k, p))
    mu  = f0t + (sigma**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * t_k))**2
    s2  = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t_k))

    rem = T_PAY[T_PAY > t_k + 1e-9]
    if len(rem) == 0:
        return 0.0

    B   = _hw_B(t_k, rem, a)                       # (n_rem,)
    lnA = _hw_lnA(t_k, rem, p, a, sigma)           # (n_rem,)
    E_P = np.exp(lnA - B * mu + 0.5 * B**2 * s2)  # (n_rem,)

    return float((1.0 - E_P[-1]) - K * TAU * E_P.sum())


# ---------------------------------------------------------------------------
# Outer simulation (shared for Models 1b and 2)
# ---------------------------------------------------------------------------

def outer_simulation(r0: float, p: dict, a: float, sigma: float,
                     n_outer: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate n_outer Euler-Maruyama paths from t=0 to t=10.
    Records the short rate at each of the 21 semi-annual monitoring dates.
    Returns r_at_mon: shape (n_outer, 21).
    """
    dt      = TAU / SUB_STEPS
    t_max   = float(T_MONITOR_NEW[-1])
    grid    = np.arange(0.0, t_max + dt * 0.1, dt)
    n_steps = len(grid) - 1

    mon_to_step = {k: int(np.round(t / dt)) for k, t in enumerate(T_MONITOR_NEW)}
    step_to_mon = {v: k for k, v in mon_to_step.items()}

    theta_vec = hw_theta(grid[:-1], p, a, sigma)

    r        = np.full(n_outer, r0, dtype=float)
    r_at_mon = np.zeros((n_outer, len(T_MONITOR_NEW)), dtype=float)
    r_at_mon[:, 0] = r0      # t=0: all paths start at r0
    sqrt_dt  = np.sqrt(dt)

    for i in range(n_steps):
        r = r + (theta_vec[i] - a * r) * dt + sigma * sqrt_dt * rng.standard_normal(n_outer)
        if (i + 1) in step_to_mon:
            r_at_mon[:, step_to_mon[i + 1]] = r

    return r_at_mon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Verify labels for Models 1a, 1b, 2")
    parser.add_argument("--feds_idx", type=int,   default=0,       help="FEDS curve index")
    parser.add_argument("--n_outer",  type=int,   default=2000,    help="Outer paths")
    parser.add_argument("--m_inner",  type=int,   default=500,     help="Inner MC paths")
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--feds",     type=str,   default=FEDS_FILE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load curve
    all_params = load_all_svensson_params(args.feds)
    crv = all_params[args.feds_idx]
    r0  = float(svensson_forward(1e-8, crv))
    K   = par_swap_rate(crv)

    print(f"\n{'='*62}")
    print(f"FEDS index : {args.feds_idx}")
    print(f"Date       : {crv['date']}")
    print(f"r0         : {r0*100:.4f}%")
    print(f"K_par      : {K*100:.4f}%")
    print(f"{'='*62}")

    # X_disc: 20 semi-annual discount factors
    d_knots = market_discount(T_PAY, crv)
    print("\nX_disc  P(0,T) at semi-annual maturities:")
    for T_j, d in zip(T_PAY, d_knots):
        print(f"  P(0,{T_j:4.1f}) = {d:.6f}")

    # Outer simulation
    print(f"\nOuter simulation: {args.n_outer} paths x {args.m_inner} inner paths ...")
    r_outer = outer_simulation(r0, crv, HW_A, HW_SIGMA, args.n_outer, rng)

    # Labels
    print(f"\n{'t_k':>5}  {'1a E[V]':>12}  {'1b E[V_MC]':>12}  {'2 EPE':>12}  {'1a-1b':>10}")
    print("-" * 62)

    for ki, t_k in enumerate(T_MONITOR_NEW):

        EV_1a = irs_expected_affine(t_k, K, crv, HW_A, HW_SIGMA)

        if t_k < 1e-8 or t_k >= MATURITY - 1e-8:
            EV_1b, EPE_2 = 0.0, 0.0
        else:
            r_tk    = r_outer[:, ki]
            r_batch = np.repeat(r_tk, args.m_inner)

            r_at_pay, logD_at_pay, rem = simulate_hw_paths(
                t_eval=t_k, r_t_batch=r_batch,
                p=crv, a=HW_A, sigma=HW_SIGMA, rng=rng,
            )
            V_paths = compute_irs_pv(
                t_eval=t_k, r_t_batch=r_batch,
                r_at_pay=r_at_pay, logD_at_pay=logD_at_pay,
                rem=rem, K=K, p=crv, a=HW_A, sigma=HW_SIGMA,
            )
            V_outer = V_paths.reshape(args.n_outer, args.m_inner).mean(axis=1)
            EV_1b   = float(V_outer.mean())
            EPE_2   = float(np.maximum(V_outer, 0.0).mean())

        print(f"{t_k:>5.1f}  {EV_1a:>12.6f}  {EV_1b:>12.6f}  {EPE_2:>12.6f}  {(EV_1a-EV_1b):>10.6f}")

    print("-" * 62)
    print("\nChecks:")
    print("  t=0.0  : 1a and 1b should both be 0.0 (ATM at inception)")
    print("  t=10.0 : all should be 0.0 (swap expired)")
    print("  1a - 1b: should be close to 0 (within MC noise ~0.001)")
    print("  EPE    : should be >= 0 everywhere, hump-shaped profile")


if __name__ == "__main__":
    main()
