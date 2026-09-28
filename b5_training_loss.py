"""
Figure -- Training Loss Curves: Models 1a, 1b, and 2.

Three vertically stacked panels sharing the x-axis. Each panel shows
training MSE and validation MSE on a log-scale y-axis. Thin vertical
dashed markers indicate learning-rate reductions triggered by
ReduceLROnPlateau (factor=0.5, patience=10, initial LR=1e-3).

Data: parsed at runtime from the real SLURM training logs for the
      "v2" (profile-design) pipeline, the pipeline actually used to
      produce the checkpoints and results reported in the dissertation:
          Model 1a: job 21190  (slurm/logs/train_1a_v2_21190.out)
          Model 1b: job 21205  (slurm/logs/train_1b_v2_21205.out)
          Model 2:  job 21206  (slurm/logs/train_2_v2_21206.out)
      Confirmed via sacct against checkpoint timestamps and against
      the already-published per-date R2/DV01 values (see r2/dv01 CSVs
      in data/r2/, which match the dissertation's Table exactly).

      NOTE: earlier versions of this script hardcoded plotted values
      as literal arrays and cited job IDs (21014/21204/21522) that
      turned out to be either the wrong ("std", pre-redesign) pipeline
      or non-existent. This version reads the real logs directly so
      the figure cannot silently drift from the underlying data again.

Output:
    ../Lindani_Thesis/Images/fig_training_loss.pdf
    ../Lindani_Thesis/Images/fig_training_loss.png
"""

import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.labelsize":   9,
    "axes.titlesize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "lines.linewidth":   1.4,
    "grid.linewidth":    0.5,
    "pdf.fonttype":      42,   # embeds fonts in PDF
    "ps.fonttype":       42,
})

# -- paths --------------------------------------------------------------------
HERE     = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(HERE, "..", "Lindani_Thesis", "Images")
LOG_DIR  = os.path.join(HERE, "slurm", "logs")
os.makedirs(IMG_DIR, exist_ok=True)

# -- colours --------------------------------------------------------------------
C_TRAIN = "#2166ac"   # steel blue  -- solid line
C_VAL   = "#d6604d"   # burnt sienna -- dashed line
C_LR    = "#969696"   # mid-grey    -- LR-reduction markers

FLOOR = 5e-7  # floor near-zero values before plotting on a log axis

EPOCH_LINE = re.compile(
    r"^\s*(\d+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+(\d+)s"
)


def parse_log(path):
    """Parse a SLURM training log into per-epoch arrays and LR-reduction epochs."""
    epochs, train, val, lr, time_s = [], [], [], [], []
    with open(path) as f:
        for line in f:
            m = EPOCH_LINE.match(line)
            if not m:
                continue
            ep, tr, va, learning_rate, t = m.groups()
            epochs.append(int(ep))
            train.append(float(tr))
            val.append(float(va))
            lr.append(float(learning_rate))
            time_s.append(int(t))

    epochs = np.array(epochs)
    train  = np.array(train)
    val    = np.array(val)
    lr     = np.array(lr)
    time_s = np.array(time_s)

    lr_epochs = [int(epochs[i]) for i in range(1, len(lr)) if lr[i] != lr[i - 1]]

    return {
        "epochs": epochs, "train": train, "val": val,
        "lr": lr, "time_s": time_s, "lr_epochs": lr_epochs,
    }


MODELS = [
    {
        "label":    "Model 1a",
        "subtitle": "Analytical IRS labels, 1{,}000 training curves",
        "logfile":  os.path.join(LOG_DIR, "train_1a_v2_21190.out"),
    },
    {
        "label":    "Model 1b",
        "subtitle": "Monte Carlo IRS labels, 1{,}000 training curves",
        "logfile":  os.path.join(LOG_DIR, "train_1b_v2_21205.out"),
    },
    {
        "label":    "Model 2",
        "subtitle": "Direct EPE labels, 1{,}000 training curves",
        "logfile":  os.path.join(LOG_DIR, "train_2_v2_21206.out"),
    },
]

for m in MODELS:
    parsed = parse_log(m["logfile"])
    assert len(parsed["epochs"]) == 200, (
        f"{m['label']}: expected 200 epochs, parsed {len(parsed['epochs'])} "
        f"from {m['logfile']}"
    )
    m.update(parsed)
    total_time = parsed["time_s"][-1]
    print(f"{m['label']}: {len(parsed['epochs'])} epochs parsed, "
          f"final train={parsed['train'][-1]:.2e}, val={parsed['val'][-1]:.2e}, "
          f"total time={total_time}s ({total_time/60:.1f} min), "
          f"LR reductions at epochs {parsed['lr_epochs']}")

# -- figure ---------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.5), sharex=True)
fig.subplots_adjust(hspace=0.10, left=0.11, right=0.97, top=0.97, bottom=0.07)

for ax, m in zip(axes, MODELS):
    epochs     = m["epochs"]
    train_plot = np.maximum(m["train"], FLOOR)
    val_plot   = np.maximum(m["val"],   FLOOR)

    first_lr = True
    for ep in m["lr_epochs"]:
        lbl = "LR halved" if first_lr else None
        ax.axvline(ep, color=C_LR, lw=0.8, ls="--", alpha=0.85,
                   zorder=1, label=lbl)
        first_lr = False

    ax.semilogy(epochs, train_plot, color=C_TRAIN, lw=1.4,
                label="Training MSE",   zorder=3)
    ax.semilogy(epochs, val_plot,   color=C_VAL,   lw=1.4, ls="--",
                label="Validation MSE", zorder=3)

    ax.text(0.02, 0.93, m["label"],
            transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="top", ha="left")

    ax.set_ylabel("MSE", labelpad=4)
    ax.set_ylim(FLOOR * 0.5, 8e-4)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation(labelOnlyBase=True))
    ax.grid(True, which="major", lw=0.4, alpha=0.35)
    ax.grid(True, which="minor", lw=0.25, alpha=0.18)
    ax.tick_params(axis="both", which="both", direction="in")

    ax.legend(loc="upper right", framealpha=0.85, edgecolor="none",
              handlelength=1.8, handletextpad=0.5, borderpad=0.4,
              ncol=3, columnspacing=1.0)

axes[-1].set_xlabel("Epoch")
axes[-1].set_xlim(1, 200)

# -- save -------------------------------------------------------------------------
for ext in ("pdf", "png"):
    out = os.path.join(IMG_DIR, f"fig_training_loss.{ext}")
    dpi = 300 if ext == "png" else None
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out}")

plt.close(fig)
