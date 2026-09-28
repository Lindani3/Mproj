"""
Block 1 — Simulated Hull-White Short Rate Paths.

Simulates 5 Hull-White paths for a single (a, sigma) pair and plots them
alongside the corresponding discount factor curve from the Svensson initial
term structure.

Output: ../../results/b1_hw_paths.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]

HW_A     = 0.10
HW_SIGMA = 0.015
N_PATHS  = 50
T_END    = 10.0
N_STEPS  = 200
SEED     = 42


def svensson(T, b0, b1, b2, b3, t1, t2):
    T   = np.atleast_1d(np.asarray(T, dtype=float))
    s   = np.where(T < 1e-10, 1e-10, T)
    e1  = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


# ── Load FEDS ─────────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid    = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]

# Select a date with a clearly upward-sloping term structure:
# β1 < −1.5 means the short end is at least 1.5% below the long end.
steep    = valid[valid["BETA1"] < -1.5]
row      = steep.iloc[len(steep) // 2]   # pick from the middle to avoid extremes
cal_date = steep.index[len(steep) // 2]
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

# Build a ZeroCurve from the Svensson term structure so that θ(t) in the
# Hull-White model is fitted to the full initial yield curve, not a flat rate.
tenors     = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
params     = row[SVEN_COLS].values
zero_rates = svensson(tenors, *params) / 100.0   # convert from percent to decimal

def _tenor_date(rd, t):
    if t < 1.0:
        return rd + ql.Period(int(round(t * 12)), ql.Months)
    return rd + ql.Period(int(t), ql.Years)

ql_dates   = [_tenor_date(ref_date, t) for t in tenors]
zero_curve = ql.ZeroCurve(ql_dates, zero_rates.tolist(),
                           ql.Actual365Fixed(), ql.NullCalendar(),
                           ql.Linear(), ql.Continuous)
yts_handle = ql.YieldTermStructureHandle(zero_curve)

# r0 = lim_{T→0} y(0,T) = β0 + β1 (Diebold and Li, 2006)
r0 = (row["BETA0"] + row["BETA1"]) / 100.0

# ── Simulate paths ────────────────────────────────────────────────────────────
hw_process = ql.HullWhiteProcess(yts_handle, HW_A, HW_SIGMA)
colours    = plt.cm.tab10(np.linspace(0, 0.45, N_PATHS))

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(N_PATHS):
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(N_STEPS,
                                          ql.UniformRandomGenerator(SEED + i * 7))
    )
    path_gen = ql.GaussianPathGenerator(hw_process, T_END, N_STEPS, rng, False)
    path     = path_gen.next().value()
    n        = N_STEPS + 1
    t_vals   = np.array([path.time(j) for j in range(n)])
    r_vals   = np.array([path[j]      for j in range(n)])
    ax.plot(t_vals, r_vals * 100.0, lw=1.0, color=colours[i], alpha=0.7)

ax.axhline(r0 * 100.0, color="k", lw=1.2, ls=":", alpha=0.8,
           label=f"$r_0 = {r0*100:.2f}\\%$")
ax.set_xlabel("Time $t$ (years)", fontsize=11)
ax.set_ylabel("Short rate $r_t$ (%)", fontsize=11)
ax.set_xlim(0, T_END)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_title(
    f"Hull-White Simulated Short Rate Paths ({cal_date.strftime('%d %B %Y')})\n"
    f"$a={HW_A}$,  $\\sigma={HW_SIGMA}$,  $N={N_PATHS}$ paths",
    fontsize=10,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b1_hw_paths.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"  r0 = {r0*100:.4f}%  |  a = {HW_A}  |  sigma = {HW_SIGMA}")
