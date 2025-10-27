# interpretability/loadings.py

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config import Config


# ------------------------------------------------------
# 1) Fit PCA on dataset
# ------------------------------------------------------
def fit_pca(X, n_components=0.98):
    """Fit PCA on features and return fitted PCA model + transformed data."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print(f"✅ PCA fitted. Original shape = {X.shape}, Reduced shape = {X_pca.shape}")
    return pca, X_pca


# ------------------------------------------------------
# 2) Compute PCA loadings (normalized correlations)
# ------------------------------------------------------
def compute_loadings(pca, feature_cols):
    """
    Compute PCA loadings (correlation of features with principal components).
    Range: [-1, 1].
    """
    loadings = pca.components_.T   # normalized loadings
    loading_df = pd.DataFrame(loadings, index=feature_cols,
                              columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])])
    print(f"✅ Loadings shape: {loading_df.shape}")
    return loading_df


# ------------------------------------------------------
# 3) Aggregate importance across PCs
# ------------------------------------------------------
def aggregate_importance(loading_df):
    """
    Aggregate feature importance by mean absolute loading across all PCs.
    """
    global_importance = loading_df.abs().mean(axis=1)
    return global_importance


# ------------------------------------------------------
# 4) Plot feature importance
# ------------------------------------------------------
def plot_importance(series, title, filename, color="royalblue", output_dir="output"):
    """Save barh plot for top/bottom features."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8), dpi=600)
    series.plot(kind="barh", color=color)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Mean Absolute Loading")
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved plot: {path}")


# ------------------------------------------------------
# 5) Main pipeline
# ------------------------------------------------------
def run_pca_loadings(data_path, n_components=0.98, output_dir="output"):
    """Run PCA loadings analysis and save top/bottom feature plots."""
    # Load dataset
    df = pd.read_csv(data_path)
    X = df.iloc[:, :52].values.astype(np.float32)   # first 52 features
    feature_cols = df.columns[:52]

    # Fit PCA
    pca, _ = fit_pca(X, n_components)

    # Compute loadings
    loading_df = compute_loadings(pca, feature_cols)

    # Aggregate importance
    global_importance = aggregate_importance(loading_df)

    # Top-30
    top30 = global_importance.sort_values(ascending=False).head(30)
    plot_importance(top30,
                    title="Top 30 Important Features (PCA Loadings)",
                    filename="pca_top30.png",
                    color="royalblue",
                    output_dir=output_dir)

    # Bottom-30
    bottom30 = global_importance.sort_values(ascending=True).head(30)
    plot_importance(bottom30,
                    title="Bottom 30 Least Important Features (PCA Loadings)",
                    filename="pca_bottom30.png",
                    color="darkorange",
                    output_dir=output_dir)


# ------------------------------------------------------
# 6) CLI entry
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA Loadings Interpretability")
    parser.add_argument("--data_path", type=str, default=Config().data_path,
                        help="Path to dataset CSV")
    parser.add_argument("--n_components", type=float, default=0.98,
                        help="PCA n_components (int or float for variance ratio)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save plots")

    args = parser.parse_args()
    run_pca_loadings(args.data_path, args.n_components, args.output_dir)