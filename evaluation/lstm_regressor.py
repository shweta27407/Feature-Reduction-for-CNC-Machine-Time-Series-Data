# evaluation/lstm_regressor.py

import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras import layers, models, optimizers

from config import Config

# ------------------------------------------------------
# Reproducibility
# ------------------------------------------------------
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------
# 1) Split latent features into train/val/test
# ------------------------------------------------------
def split_latent_data(Z, y, split):
    """Chronologically split latent features + targets."""
    N = len(Z)
    train_end = int(split[0] * N)
    val_end = int((split[0] + split[1]) * N)

    Z_train, y_train = Z[:train_end], y[:train_end]
    Z_val,   y_val   = Z[train_end:val_end], y[train_end:val_end]
    Z_test,  y_test  = Z[val_end:], y[val_end:]

    return Z_train, Z_val, Z_test, y_train, y_val, y_test


# ------------------------------------------------------
# 2) Make sequences from latent features
# ------------------------------------------------------
def make_latent_sequences(Z, y, context):
    """Convert latent features into sequential format for LSTM regressor."""
    Xs, ys = [], []
    for i in range(context - 1, len(Z)):
        Xs.append(Z[i - context + 1:i + 1])   # shape: (context, latent_dim)
        ys.append(y[i])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# ------------------------------------------------------
# 3) Build the LSTM regressor model
# ------------------------------------------------------
def build_lstm_regressor(context, latent_dim):
    """Builds a simple LSTM regressor for latent features."""
    rin = layers.Input(shape=(context, latent_dim))
    h  = layers.LSTM(64, return_sequences=False)(rin)
    h  = layers.Dropout(0.2)(h)
    h  = layers.Dense(32, activation='relu')(h)
    rout = layers.Dense(1, activation=None)(h)

    model = models.Model(rin, rout)
    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    return model


# ------------------------------------------------------
# 4) Plot True vs Predicted
# ------------------------------------------------------
def plot_results(y_true, y_pred, model_type, output_dir):
    """Save True vs Predicted plot."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{model_type}_true_vs_pred.png")

    title_map = {
        "lstm_cnn": "LSTM-CNN - True vs Predicted",
        "dense_ae": "DenseAE - True vs Predicted",
        "pca": "PCA - True vs Predicted"
    }

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True", alpha=0.8)
    plt.plot(y_pred, label="Predicted", alpha=0.8)
    plt.title(title_map.get(model_type, "Model - True vs Predicted"))
    plt.xlabel("Sample")
    plt.ylabel("Target")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"📊 Plot saved to {plot_path}")


# ------------------------------------------------------
# 5) Main evaluation function
# ------------------------------------------------------
def run(latent_csv, model_type="dense_ae", output_dir="output"):
    """Train LSTM regressor on reduced (latent) features and plot results."""
    cfg = Config()

    # Load reduced dataset
    df = pd.read_csv(latent_csv)
    y = df["target"].values.astype(np.float32)

    # Remove "split" column if present
    if "split" in df.columns:
        df = df.drop(columns=["split"])

    Z = df.drop(columns=["target"]).values.astype(np.float32)

    # Split into train/val/test
    Z_train, Z_val, Z_test, y_train, y_val, y_test = split_latent_data(Z, y, cfg.split)

    # Choose context size
    if model_type == "lstm_cnn":
        context = cfg.context_lstm_cnn
    else:  # dense_ae or pca
        context = cfg.context_dense_pca

    # Build sequences
    Ztr_seq, ytr_seq = make_latent_sequences(Z_train, y_train, context)
    Zva_seq, yva_seq = make_latent_sequences(Z_val,   y_val,   context)
    Zte_seq, yte_seq = make_latent_sequences(Z_test,  y_test,  context)

    print(f"Latent sequence shapes: {Ztr_seq.shape}, {Zva_seq.shape}, {Zte_seq.shape}")

    # Build regressor
    regressor = build_lstm_regressor(context, Z.shape[1])

    # Train regressor
    start_time = time.time()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
    ]
    regressor.fit(
        Ztr_seq, ytr_seq,
        epochs=100, batch_size=32,
        validation_data=(Zva_seq, yva_seq),
        callbacks=callbacks, verbose=1, shuffle=True
    )
    training_time = time.time() - start_time

    # Evaluate
    y_pred = regressor.predict(Zte_seq, verbose=0).ravel()
    r2   = r2_score(yte_seq, y_pred)
    mae  = mean_absolute_error(yte_seq, y_pred)
    rmse = np.sqrt(mean_squared_error(yte_seq, y_pred))

    print(f"✅ Evaluation Results ({model_type} latent features):")
    print(f"R²: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | TrainTime(s): {training_time:.2f}")

    # Plot results
    plot_results(yte_seq, y_pred, model_type, output_dir)


# ------------------------------------------------------
# 6) CLI entry
# ------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate latent features with LSTM regressor")
    parser.add_argument("--latent_csv", type=str, required=True,
                        help="Path to latent CSV file (from output folder)")
    parser.add_argument("--model_type", type=str, required=True,
                        choices=["lstm_cnn", "dense_ae", "pca"],
                        help="Model type used to generate latent features")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Directory to save plots")

    args = parser.parse_args()
    run(args.latent_csv, model_type=args.model_type, output_dir=args.output_dir)