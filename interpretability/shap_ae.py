# evaluation/shap_denseae.py
import os
import argparse
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Model
from config import Config


# ------------------------------------------------------
# 1) Load encoder model
# ------------------------------------------------------
def load_encoder_model(model_path):
    """Load trained DenseAE model and extract encoder part."""
    autoencoder = load_model(model_path, compile=False)  # <-- disable compile
    encoder = Model(inputs=autoencoder.input,
                    outputs=autoencoder.get_layer("latent").output)
    latent_dim = int(encoder.output_shape[-1])
    print(f"✅ Loaded DenseAE. Latent dimension = {latent_dim}")
    return encoder, latent_dim


# ------------------------------------------------------
# 2) Compute SHAP values
# ------------------------------------------------------
def compute_shap_values(encoder, X, background_size=90, explain_size=30):
    """Compute SHAP values for encoder model."""
    ENC_BG = min(background_size, X.shape[0])
    ENC_EXPL = min(explain_size, X.shape[0])
    background = X[:ENC_BG]
    X_explain = X[:ENC_EXPL]

    explainer = shap.GradientExplainer(encoder, background)
    shap_values_raw = explainer.shap_values(X_explain)

    if isinstance(shap_values_raw, list):
        sv_stack = np.stack(shap_values_raw, axis=-1)  # (samples, features, latent_dim)
    else:
        sv_stack = shap_values_raw

    print(f"✅ SHAP values computed. Shape = {sv_stack.shape}")
    return sv_stack


# ------------------------------------------------------
# 3) Aggregate global importance
# ------------------------------------------------------
def aggregate_importance(shap_values, feature_cols):
    """Aggregate SHAP values into global importance ranking."""
    encoder_importance = np.abs(shap_values).mean(axis=0)  # (features, latent_dim)
    global_importance = encoder_importance.mean(axis=1)    # (features,)
    return pd.Series(global_importance, index=feature_cols)


# ------------------------------------------------------
# 4) Plot importance
# ------------------------------------------------------
def plot_importance(series, title, filename, color, output_dir="output"):
    """Save horizontal bar plot for feature importance."""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 8), dpi=600)
    series.plot(kind="barh", color=color)
    plt.gca().invert_yaxis()
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved plot: {path}")


# ------------------------------------------------------
# 5) Main SHAP pipeline
# ------------------------------------------------------
def run_shap(model_path):
    """Run SHAP interpretability pipeline for DenseAE."""
    cfg = Config()

    # Load encoder model
    encoder, latent_dim = load_encoder_model(model_path)

    # Load dataset directly from config
    df = pd.read_csv(cfg.data_path)
    X = df.iloc[:, :52].values.astype(np.float32)
    feature_cols = df.columns[:52]

    # Compute SHAP values
    shap_values = compute_shap_values(encoder, X)

    # Aggregate importance
    global_importance = aggregate_importance(shap_values, feature_cols)

    # Top-30
    top30 = global_importance.sort_values(ascending=False).head(30)
    plot_importance(top30,
                    title="Top 30 Important Features (DenseAE - SHAP)",
                    filename="denseae_top30.png",
                    color="royalblue")

    # Bottom-30
    bottom30 = global_importance.sort_values(ascending=True).head(30)
    plot_importance(bottom30,
                    title="Bottom 30 Least Important Features (DenseAE - SHAP)",
                    filename="denseae_bottom30.png",
                    color="darkorange")


# ------------------------------------------------------
# 6) CLI entry
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP interpretability for DenseAE")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained DenseAE .h5 file (e.g., output/dense_autoencoder.h5)")
    args = parser.parse_args()

    run_shap(args.model_path)