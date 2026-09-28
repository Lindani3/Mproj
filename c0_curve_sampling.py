"""
c0_curve_sampling.py
====================
Select 1,000 representative yield curves from the full FEDS dataset
using PCA dimensionality reduction followed by K-means clustering.

Rationale
---------
The 16,160 FEDS curves live near a 3-dimensional subspace of R^20
(level, slope, curvature -- Litterman & Scheinkman 1991).  The
historical distribution is non-uniform: post-2008 low-rate curves
dominate.  Naive random sampling oversamples common regimes and
undersamples rare ones (inverted, high-rate) that drive CVA stress.

Strategy
--------
1. Compute P(0, T_PAY) for every FEDS curve  ->  (N, 20) matrix
2. Standardise and project onto 3 principal components
3. K-means with K=1,000 in the 3D PCA space
4. Per cluster: select the curve nearest to the centroid
5. Write: indices array, PCA summary, filtered FEDS CSV

Output
------
  data/feds_1000_indices.npy   integer indices into the full FEDS list
  data/feds_1000.csv           filtered FEDS CSV (drop-in for --feds)
  data/pca_summary.npz         scores, explained variance (for reporting)
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import T_PAY, load_all_svensson_params, market_discount

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "feds200628.csv")


def parse_args():
    p = argparse.ArgumentParser(description="PCA + K-means yield curve selection")
    p.add_argument("--feds",    type=str, default=FEDS_FILE)
    p.add_argument("--n",       type=int, default=1000,
                   help="Number of curves to select")
    p.add_argument("--n_pca",   type=int, default=3,
                   help="PCA components to retain")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--out_dir", type=str,
                   default=os.path.join(CODE_DIR, "data"))
    return p.parse_args()


def compute_discount_matrix(all_params):
    """Compute P(0, T_PAY) for every FEDS curve. Returns (N, 20)."""
    N = len(all_params)
    D = np.empty((N, len(T_PAY)), dtype=np.float64)
    for i, p in enumerate(all_params):
        D[i] = market_discount(T_PAY, p)
    return D


def select_curves(D, n_select, n_pca, seed):
    """
    Project D (N, 20) onto n_pca PCs, cluster with K-means,
    return indices of the curve nearest to each centroid.
    """
    scaler   = StandardScaler()
    D_scaled = scaler.fit_transform(D)

    pca    = PCA(n_components=n_pca, random_state=seed)
    scores = pca.fit_transform(D_scaled)          # (N, n_pca)
    ev     = pca.explained_variance_ratio_

    print("PCA explained variance:")
    cum = 0.0
    for k, v in enumerate(ev):
        cum += v
        print(f"  PC{k+1}: {v*100:.2f}%  (cumulative {cum*100:.2f}%)")

    print(f"\nRunning K-means (K={n_select}, seed={seed}) ...")
    t0 = time.time()
    km = KMeans(n_clusters=n_select, n_init=10, random_state=seed)
    km.fit(scores)
    print(f"  Done in {time.time()-t0:.1f}s")

    centroids    = km.cluster_centers_     # (n_select, n_pca)
    labels       = km.labels_              # (N,)
    selected_idx = np.empty(n_select, dtype=int)

    for k in range(n_select):
        cluster_idx     = np.where(labels == k)[0]
        cluster_pts     = scores[cluster_idx]
        dists           = np.linalg.norm(cluster_pts - centroids[k], axis=1)
        selected_idx[k] = cluster_idx[np.argmin(dists)]

    return selected_idx, scores, ev


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading FEDS curves from {args.feds} ...")
    all_params = load_all_svensson_params(args.feds)
    n_feds     = len(all_params)
    print(f"  {n_feds:,} curves loaded")

    print(f"\nComputing discount factor matrix ({n_feds} x {len(T_PAY)}) ...")
    D = compute_discount_matrix(all_params)

    selected_idx, scores, ev = select_curves(
        D, n_select=args.n, n_pca=args.n_pca, seed=args.seed
    )
    print(f"\nSelected {len(selected_idx)} representative curves")

    # Save indices
    idx_path = os.path.join(args.out_dir, "feds_1000_indices.npy")
    np.save(idx_path, selected_idx)
    print(f"Saved indices     : {idx_path}")

    # Save PCA summary for dissertation reporting
    pca_path = os.path.join(args.out_dir, "pca_summary.npz")
    np.savez(pca_path, scores=scores,
             explained_variance=ev, selected_idx=selected_idx)
    print(f"Saved PCA summary : {pca_path}")

    # Write filtered FEDS CSV (preserves original 9-row header)
    with open(args.feds) as f:
        raw_header = [f.readline() for _ in range(9)]

    df_full = pd.read_csv(args.feds, skiprows=9)
    df_full = df_full.dropna(
        subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    ).reset_index(drop=True)

    df_selected = df_full.iloc[selected_idx].reset_index(drop=True)

    out_csv = os.path.join(args.out_dir, "feds_1000.csv")
    with open(out_csv, "w") as f:
        for line in raw_header:
            f.write(line)
    df_selected.to_csv(out_csv, mode="a", index=False)
    print(f"Saved filtered CSV: {out_csv}  ({len(df_selected)} rows)")

    date_col = next((c for c in df_full.columns if "date" in c.lower()), None)
    if date_col:
        dates = df_full[date_col].iloc[selected_idx]
        print(f"Date range        : {dates.min()} to {dates.max()}")

    print(f"\nTo use: pass --feds {out_csv} to any datagen script.")


if __name__ == "__main__":
    main()
