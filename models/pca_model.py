# models/pca_model.py

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def run(X_train, X_val, X_test,
        y_train, y_val, y_test,
        n_components=0.98,
        output_dir="output",
        output_name="pca_latent.csv",
        plot_name="pca_scree.png"):
    """
    Runs PCA on full dataset (train+val+test),
    saves reduced features + target to CSV, and saves scree plot.
    """

    # 1. Stack all splits to apply PCA consistently
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    # 2. Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_all)

    print(f"Original shape: {X_all.shape}")
    print(f"Reduced shape after PCA: {X_pca.shape}")

    # 3. Create DataFrame
    pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    pca_df["target"] = y_all

    # 4. Save reduced dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    pca_df.to_csv(output_path, index=False)
    print(f"✅ Saved PCA-reduced dataset to {output_path}")

    # 5. Save Scree Plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    plt.bar(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_*100, alpha=0.7,
            label='Individual Explained Variance')

    plt.plot(np.cumsum(pca.explained_variance_ratio_)*100,
             color='red', marker='o', label='Cumulative Explained Variance')

    plt.xlabel("Principal Component Index")
    plt.ylabel("Explained Variance (%)")
    plt.title("Scree Plot - PCA")
    plt.legend(loc='best')
    plt.tight_layout()

    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Scree plot saved to {plot_path}")

    return pca_df