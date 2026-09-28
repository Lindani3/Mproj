#!/bin/bash
# ==============================================================================
# campus_run.sh
# ==============================================================================
# Run ON THE CLUSTER while on campus.
#
# What it does:
#   1. Downloads new scripts from GitHub (if not already present)
#   2. Runs R² and DV01 evaluations for all trained model variants
#   3. Saves all result CSVs to data/r2/
#
# How to use:
#   ssh lindani@146.141.21.100
#   cd ~/FinalResults/Code
#   wget -O campus_run.sh https://raw.githubusercontent.com/Lindani3/Mproj/main/campus_run.sh
#   bash campus_run.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " Campus evaluation run"
echo " Start: $(date)"
echo "============================================================"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate swapnet

mkdir -p data/r2

# Download new scripts from GitHub if missing
for f in b8_dv01_profile.py hw_utils.py c4_model_v2.py; do
    if [ ! -f "$f" ]; then
        echo "Downloading $f ..."
        wget -q -O "$f" "https://raw.githubusercontent.com/Lindani3/Mproj/main/$f"
    fi
done

# ------------------------------------------------------------------------------
run_r2() {
    local label=$1 ckpt=$2 data=$3 script=${4:-b6_r2_profile_v2.py}
    [ ! -f "$ckpt" ] && echo "  [SKIP] $label — $ckpt not found" && return
    [ ! -f "$data" ] && echo "  [SKIP] $label — $data not found" && return
    echo "--- R² : $label ---"
    python -u "$script" --checkpoint "$ckpt" --data "$data" --label "$label" --out_dir data/r2
}

run_dv01() {
    local label=$1 ckpt=$2 data=$3
    [ ! -f "$ckpt" ] && echo "  [SKIP] $label — $ckpt not found" && return
    [ ! -f "$data" ] && echo "  [SKIP] $label — $data not found" && return
    echo "--- DV01: $label ---"
    python -u b8_dv01_profile.py --checkpoint "$ckpt" --data "$data" --label "$label" --out_dir data/r2
}

# ==============================================================================
echo ""
echo "=============================="
echo " R² Profile Evaluations"
echo "=============================="

run_r2 1a_v2         data/best_model_1a_v2.pt        data/train_1a_v2.h5
run_r2 1b_v2         data/best_model_1b_v2.pt        data/train_1b_v2.h5
run_r2 2_v2          data/best_model_2_v2.pt         data/train_2_v2.h5
run_r2 1a_v22        data/best_model_1a_v22.pt       data/train_1a_v22.h5
run_r2 1b_v22        data/best_model_1b_v22.pt       data/train_1b_v22.h5
run_r2 2_v22         data/best_model_2_v22.pt        data/train_2_v22.h5
run_r2 1a_v2_sob     data/best_model_1a_v2_sob.pt    data/train_1a_v2_sob.h5
run_r2 1b_v2_sob     data/best_model_1b_v2_sob.pt    data/train_1b_v2_sob.h5
run_r2 2_v2_sob      data/best_model_2_v2_sob.pt     data/train_2_v2_sob.h5
run_r2 1b_v2_sob_pw  data/best_model_1b_v2_sob_pw.pt data/train_1b_v2_sob_pw.h5
run_r2 2_v2_sob_pw   data/best_model_2_v2_sob_pw.pt  data/train_2_v2_sob_pw.h5

# ==============================================================================
echo ""
echo "=============================="
echo " DV01 Profile Evaluations"
echo "=============================="

run_dv01 1a_v2        data/best_model_1a_v2.pt        data/train_1a_v2.h5
run_dv01 1b_v2        data/best_model_1b_v2.pt        data/train_1b_v2.h5
run_dv01 2_v2         data/best_model_2_v2.pt         data/train_2_v2.h5
run_dv01 1a_v22       data/best_model_1a_v22.pt       data/train_1a_v22.h5
run_dv01 1b_v22       data/best_model_1b_v22.pt       data/train_1b_v22.h5
run_dv01 2_v22        data/best_model_2_v22.pt        data/train_2_v22.h5
run_dv01 1b_v2_sob_pw data/best_model_1b_v2_sob_pw.pt data/train_1b_v2_sob_pw.h5
run_dv01 2_v2_sob_pw  data/best_model_2_v2_sob_pw.pt  data/train_2_v2_sob_pw.h5

# ==============================================================================
echo ""
echo "============================================================"
echo " Done: $(date)"
echo " Results saved to: $SCRIPT_DIR/data/r2/"
ls data/r2/*.csv 2>/dev/null | sed 's/^/   /' || echo "   (no CSVs found)"
echo "============================================================"
