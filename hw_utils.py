"""
hw_utils.py
===========
Shared Hull-White / Svensson pricing utilities for the GRU surrogate pipeline.

All functions are parameter-aware: a and sigma are explicit arguments rather
than module-level globals, so Model 1a can sweep (a, sigma) without
reimporting.  market_discount_bumped supports Model 2 DV01 computation via
finite-difference bumps of individual zero-yield knots.

The sign convention in _hw_lnA follows Hull & White (1990):

    ln A(t,T) = ln[P(0,T)/P(0,t)] + B(t,T)*f(0,t)
                - (sigma^2 / 4a) * B(t,T)^2 * (1 - exp(-2a*t))

(positive sign before B*f0t, consistent with b3/b4).

References
----------
Hull, J. and White, A. (1990). Pricing interest-rate-derivative securities.
    Review of Financial Studies, 3(4), 573-592.
Svensson, L.E.O. (1994). Estimating and Interpreting Forward Interest Rates:
    Sweden 1992-1994. IMF Working Paper 94/114.
Gurkaynak, R.S., Sack, B. and Wright, J.H. (2007). The U.S. Treasury yield
    curve: 1961 to the present. Journal of Monetary Economics, 54(8), 2291-2304.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# IRS instrument constants (shared across all scripts)
# ---------------------------------------------------------------------------
MATURITY  = 10.0
TAU       = 0.5
SUB_STEPS = 12                                            # dt = TAU / SUB_STEPS
T_PAY     = np.arange(TAU, MATURITY + 1e-9, TAU)        # shape (20,)
T_MONITOR = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                      5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
T_KNOTS   = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 10.0])

HW_A     = 0.10
HW_SIGMA = 0.015


# ===========================================================================
# Svensson (1994) term structure
# ===========================================================================

def load_all_svensson_params(filepath: str) -> list[dict]:
    """
    Read all complete rows from the Federal Reserve GSW dataset.
    Returns a list of parameter dicts (yields converted from percent to decimal).
    """
    df = pd.read_csv(filepath, skiprows=9)
    df = df.dropna(subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "beta0": float(row["BETA0"]) / 100.0,
            "beta1": float(row["BETA1"]) / 100.0,
            "beta2": float(row["BETA2"]) / 100.0,
            "beta3": float(row["BETA3"]) / 100.0,
            "tau1":  float(row["TAU1"]),
            "tau2":  float(row["TAU2"]),
            "date":  str(row["Date"]),
        })
    return rows


def load_svensson_params(filepath: str) -> dict:
    """Return Svensson parameters from the last complete row (matches b3/b4)."""
    return load_all_svensson_params(filepath)[-1]


def svensson_yield(t: np.ndarray, p: dict) -> np.ndarray:
    """Continuously compounded zero-coupon yield y(0,t) in decimal form."""
    t  = np.asarray(t, dtype=float)
    ts = np.where(t < 1e-10, 1e-10, t)
    e1 = np.exp(-ts / p["tau1"])
    e2 = np.exp(-ts / p["tau2"])
    g1 = (1.0 - e1) / (ts / p["tau1"])
    g2 = (1.0 - e2) / (ts / p["tau2"])
    return (
        p["beta0"]
        + p["beta1"] * g1
        + p["beta2"] * (g1 - e1)
        + p["beta3"] * (g2 - e2)
    )


def svensson_forward(t: np.ndarray, p: dict) -> np.ndarray:
    """Instantaneous forward rate f(0,t) in decimal form."""
    t  = np.asarray(t, dtype=float)
    ts = np.where(t < 1e-10, 1e-10, t)
    e1 = np.exp(-ts / p["tau1"])
    e2 = np.exp(-ts / p["tau2"])
    return (
        p["beta0"]
        + p["beta1"] * e1
        + p["beta2"] * (ts / p["tau1"]) * e1
        + p["beta3"] * (ts / p["tau2"]) * e2
    )


def market_discount(t: np.ndarray, p: dict) -> np.ndarray:
    """Market discount factor P(0,t) = exp(-y(0,t)*t) from the Svensson fit."""
    t = np.asarray(t, dtype=float)
    return np.exp(-svensson_yield(t, p) * t)


def market_discount_bumped(
    t: np.ndarray,
    p: dict,
    bump_idx: int,
    bump_size: float,
    t_knots: np.ndarray = T_KNOTS,
) -> np.ndarray:
    """
    Discount factor with a single zero-yield knot bumped by bump_size.

    The Svensson-fitted zero yields are evaluated at t_knots; y[bump_idx] is
    shifted by bump_size; the bumped curve is linearly interpolated at
    arbitrary maturities t using numpy.interp (flat extrapolation outside
    the knot range).

    This is the building block for key-rate DV01 computation in Model 2.
    """
    t      = np.asarray(t, dtype=float)
    y_base = svensson_yield(t_knots, p)
    y_bump = y_base.copy()
    y_bump[bump_idx] += bump_size
    y_at_t = np.interp(t, t_knots, y_bump)
    return np.exp(-y_at_t * t)


# ===========================================================================
# Hull-White drift
# ===========================================================================

def hw_theta(t: np.ndarray, p: dict, a: float, sigma: float) -> np.ndarray:
    """
    Drift function theta(t) calibrated to the initial term structure.

        theta(t) = df(0,t)/dt + a*f(0,t) + (sigma^2/(2a))*(1 - exp(-2a*t))

    The time derivative is evaluated by central finite differences on the
    Svensson instantaneous forward.
    """
    t    = np.asarray(t, dtype=float)
    h    = 1e-5
    dfdt = (svensson_forward(t + h, p) - svensson_forward(t - h, p)) / (2.0 * h)
    return dfdt + a * svensson_forward(t, p) + (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))


# ===========================================================================
# Hull-White affine bond price
# ===========================================================================

def _hw_B(t: float, T: np.ndarray, a: float) -> np.ndarray:
    """B(t,T) = [1 - exp(-a*(T-t))] / a."""
    return (1.0 - np.exp(-a * (np.asarray(T, dtype=float) - t))) / a


def _hw_lnA(
    t: float,
    T: np.ndarray,
    p: dict,
    a: float,
    sigma: float,
    disc_fn=None,
) -> np.ndarray:
    """
    ln A(t,T) for the Hull-White affine bond price.

    disc_fn(t, p) replaces market_discount when bumping is required for DV01.
    The instantaneous forward f(0,t) always comes from the original Svensson
    fit; only the discount factors P(0,T) and P(0,t) are bumped.
    """
    if disc_fn is None:
        disc_fn = market_discount
    T   = np.asarray(T, dtype=float)
    B   = _hw_B(t, T, a)
    P0T = disc_fn(T, p)
    P0t = float(disc_fn(np.asarray([t]), p)[0])
    f0t = float(svensson_forward(t, p))
    return (
        np.log(P0T / P0t)
        + B * f0t
        - (sigma**2 / (4.0 * a)) * B**2 * (1.0 - np.exp(-2.0 * a * t))
    )


def hw_bond_price(
    t: float,
    T: np.ndarray,
    r_t,
    p: dict,
    a: float,
    sigma: float,
    disc_fn=None,
) -> np.ndarray:
    """
    P(t, T | r_t) via the Hull-White affine formula.

    r_t may be a scalar or a 1-D array of shape (M,). When r_t is (M,) and
    T is (n,), the output has shape (M, n).
    """
    T   = np.asarray(T, dtype=float)
    B   = _hw_B(t, T, a)
    lnA = _hw_lnA(t, T, p, a, sigma, disc_fn)
    r_t = np.asarray(r_t)
    if r_t.ndim == 0:
        return np.exp(lnA - B * r_t)
    return np.exp(lnA[np.newaxis, :] - np.outer(r_t, B))


# ===========================================================================
# IRS par rate
# ===========================================================================

def par_swap_rate(p: dict, t_pay: np.ndarray = T_PAY, tau: float = TAU) -> float:
    """Par fixed rate K at t=0 under single-curve pricing."""
    P = market_discount(t_pay, p)
    return float((1.0 - P[-1]) / (tau * P.sum()))


# ===========================================================================
# Closed-form (affine) IRS value
# ===========================================================================

def irs_expected_affine(
    t_k:   float,
    K:     float,
    p:     dict,
    a:     float,
    sigma: float,
    t_pay: np.ndarray = T_PAY,
    tau:   float = TAU,
    maturity: float = MATURITY,
) -> float:
    """
    Analytical E[V(t_k)] under Hull-White via the MGF of r_{t_k} ~ N(mu, s^2).

        mu  = f(0,t_k) + sigma^2/(2a^2) * (1 - exp(-a*t_k))^2
        s^2 = sigma^2/(2a) * (1 - exp(-2a*t_k))
        E[P(t_k,T_j)] = exp(lnA(t_k,T_j) - B_j*mu + 0.5*B_j^2*s^2)
        E[V(t_k)]     = [1 - E[P_n]] - K*tau * sum_{T_j>t_k} E[P_j]

    Returns 0 at boundaries (t_k=0 for ATM, t_k>=maturity for expired swap).
    """
    if t_k < 1e-8 or t_k >= maturity - 1e-8:
        return 0.0
    f0t = float(svensson_forward(t_k, p))
    mu  = f0t + (sigma**2 / (2.0 * a**2)) * (1.0 - np.exp(-a * t_k))**2
    s2  = (sigma**2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t_k))
    rem = t_pay[t_pay > t_k + 1e-9]
    if len(rem) == 0:
        return 0.0
    B   = _hw_B(t_k, rem, a)
    lnA = _hw_lnA(t_k, rem, p, a, sigma)
    E_P = np.exp(lnA - B * mu + 0.5 * B**2 * s2)
    return float((1.0 - E_P[-1]) - K * tau * E_P.sum())


def irs_affine(
    t_eval: float,
    r_t: float,
    K: float,
    p: dict,
    a: float,
    sigma: float,
    t_pay: np.ndarray = T_PAY,
    tau: float = TAU,
    disc_fn=None,
) -> float:
    """
    Closed-form payer IRS value at (t_eval, r_t).

        V = [1 - P(t, T_n)] - K*tau * sum_{T_j > t} P(t, T_j)
    """
    rem   = t_pay[t_pay > t_eval + 1e-9]
    bonds = hw_bond_price(t_eval, rem, r_t, p, a, sigma, disc_fn)
    return float((1.0 - bonds[-1]) - K * tau * bonds.sum())


# ===========================================================================
# Factored inner Monte Carlo (simulation + cashflow re-evaluation separated)
# ===========================================================================

def simulate_hw_paths(
    t_eval: float,
    r_t_batch: np.ndarray,
    p: dict,
    a: float,
    sigma: float,
    t_pay: np.ndarray = T_PAY,
    maturity: float = MATURITY,
    tau: float = TAU,
    sub_steps: int = SUB_STEPS,
    rng: np.random.Generator = None,
):
    """
    Simulate Euler-Maruyama paths from (t_eval, r_t_batch) to maturity.

    Parameters
    ----------
    r_t_batch : (M,) array of starting short rates.
    rng       : pre-seeded Generator; if None, uses numpy default.

    Returns
    -------
    r_at_pay    : (M, n_pay) short rate at each remaining payment date.
    logD_at_pay : (M, n_pay) accumulated log-discount at each payment date.
    rem         : (n_pay,) remaining payment dates > t_eval.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt    = tau / sub_steps
    rem   = t_pay[t_pay > t_eval + 1e-9]
    n_pay = len(rem)
    M     = len(r_t_batch)

    if n_pay == 0:
        return (
            np.zeros((M, 0), dtype=float),
            np.zeros((M, 0), dtype=float),
            rem,
        )

    grid    = np.arange(t_eval, maturity + dt * 0.1, dt)
    n_steps = len(grid) - 1

    pay_to_grid = {j: int(np.round((rem[j] - t_eval) / dt)) for j in range(n_pay)}
    grid_to_pay = {v: k for k, v in pay_to_grid.items()}

    theta_vec = hw_theta(grid[:-1], p, a, sigma)

    r    = r_t_batch.copy()
    logD = np.zeros(M, dtype=float)

    r_at_pay    = np.zeros((M, n_pay), dtype=float)
    logD_at_pay = np.zeros((M, n_pay), dtype=float)

    sqrt_dt = np.sqrt(dt)

    for i in range(n_steps):
        logD -= r * dt
        r     = r + (theta_vec[i] - a * r) * dt + sigma * sqrt_dt * rng.standard_normal(M)
        if (i + 1) in grid_to_pay:
            j                  = grid_to_pay[i + 1]
            r_at_pay[:, j]    = r
            logD_at_pay[:, j] = logD

    return r_at_pay, logD_at_pay, rem


def compute_irs_pv(
    t_eval: float,
    r_t_batch: np.ndarray,
    r_at_pay: np.ndarray,
    logD_at_pay: np.ndarray,
    rem: np.ndarray,
    K: float,
    p: dict,
    a: float,
    sigma: float,
    tau: float = TAU,
    disc_fn=None,
) -> np.ndarray:
    """
    Compute path-wise payer IRS PV from pre-simulated trajectories.

    Accepts the same (r_at_pay, logD_at_pay, rem) returned by
    simulate_hw_paths, plus an optional disc_fn for DV01 bump-and-reprice.

    Returns V_paths: shape (M,).
    """
    if disc_fn is None:
        disc_fn = market_discount

    M     = r_t_batch.shape[0]
    n_pay = len(rem)
    V_paths = np.zeros(M, dtype=float)

    for j in range(n_pay):
        T_jm1 = t_eval if j == 0 else rem[j - 1]
        T_j   = rem[j]

        r_jm1 = r_t_batch if j == 0 else r_at_pay[:, j - 1]

        B_val   = float(_hw_B(T_jm1, T_j, a))
        lnA_val = float(_hw_lnA(T_jm1, T_j, p, a, sigma, disc_fn))
        P_bond  = np.exp(lnA_val - B_val * r_jm1)

        net_cf = (1.0 / P_bond - 1.0) - K * tau
        disc   = np.exp(logD_at_pay[:, j])
        V_paths += net_cf * disc

    return V_paths


def irs_mc_inner(
    t_eval: float,
    r_t_batch: np.ndarray,
    K: float,
    p: dict,
    a: float,
    sigma: float,
    rng: np.random.Generator,
    disc_fn=None,
    t_pay: np.ndarray = T_PAY,
    maturity: float = MATURITY,
    tau: float = TAU,
    sub_steps: int = SUB_STEPS,
) -> np.ndarray:
    """
    Combined simulate + cashflow: returns V_paths of shape (M,).

    Convenience wrapper around simulate_hw_paths + compute_irs_pv for
    contexts that do not need to reuse the simulated trajectories.
    """
    r_at_pay, logD_at_pay, rem = simulate_hw_paths(
        t_eval, r_t_batch, p, a, sigma, t_pay, maturity, tau, sub_steps, rng
    )
    if len(rem) == 0:
        return np.zeros(len(r_t_batch), dtype=float)
    return compute_irs_pv(
        t_eval, r_t_batch, r_at_pay, logD_at_pay, rem, K, p, a, sigma, tau, disc_fn
    )
