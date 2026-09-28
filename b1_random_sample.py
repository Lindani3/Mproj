"""
Block 1 — Five Randomly Sampled Historical Yield and Discount Factor Curves.

Samples 5 random rows from the Federal Reserve GSW dataset (feds200628.csv),
computes the Svensson yield curve and corresponding discount factor curve for
each, and plots both side by side.

Output: ../../results/b1_random_sample.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
SEED      = 42
N_SAMPLE  = 5
T_GRID    = np.linspace(0.25, 10.0, 200)


def svensson(T, b0, b1, b2, b3, t1, t2):
    T   = np.atleast_1d(np.asarray(T, dtype=float))
    s   = np.where(T < 1e-10, 1e-10, T)
    e1  = np.exp(-s / t1)
    e2  = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1)
    ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


# ── Load and sample ───────────────────────────────────────────────────────────
feds  = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds  = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]

rng    = np.random.default_rng(SEED)
idx    = rng.choice(len(valid), size=N_SAMPLE, replace=False)
sample = valid.iloc[sorted(idx)]

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
colors = plt.cm.tab10(np.linspace(0, 0.55, N_SAMPLE))

for (date, row), col in zip(sample.iterrows(), colors):
    params = row[SVEN_COLS].values
    y  = svensson(T_GRID, *params) / 100.0   # GSW stores rates in percent
    df = np.exp(-y * T_GRID)
    label = date.strftime("%b %Y")
    ax1.plot(T_GRID, y * 100, color=col, lw=1.8, label=label)
    ax2.plot(T_GRID, df,      color=col, lw=1.8, label=label)

ax1.set_xlabel("Maturity $T$ (years)", fontsize=11)
ax1.set_ylabel("Zero rate $y(0,T)$ (%)", fontsize=11)
ax1.set_title("Svensson Yield Curves — 5 randomly sampled historical dates",
              fontsize=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.25)
ax1.set_xlim(0, 10)

ax2.set_xlabel("Maturity $T$ (years)", fontsize=11)
ax2.set_ylabel("Discount factor $P(0,T)$", fontsize=11)
ax2.set_title("Discount Factor Curves — 5 randomly sampled historical dates",
              fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)
ax2.set_xlim(0, 10)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b1_random_sample.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

print("\nSampled dates and short-end rates:")
for date, row in sample.iterrows():
    r0 = (row["BETA0"] + row["BETA1"]) / 100.0
    print(f"  {date.strftime('%Y-%m-%d')}  r0 = {r0*100:.4f}%")
