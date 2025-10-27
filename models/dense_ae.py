# models/dense_ae.py

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# ======================================================
# Build Dense Autoencoder
# ======================================================
def build_dense_autoencoder(input_dim, latent_dim=25):
    """
    Builds a simple Dense Autoencoder and its Encoder part separately.
    """
    inputs = layers.Input(shape=(input_dim,))
    e = layers.Dense(128, activation="relu")(inputs)
    e = layers.Dropout(0.2)(e)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(e)
    d = layers.Dense(128, activation="relu")(latent)
    d = layers.Dropout(0.2)(d)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(d)

    ae_model = models.Model(inputs, outputs, name="dense_autoencoder")
    ae_model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")

    encoder = models.Model(inputs, latent, name="dense_encoder")
    return ae_model, encoder


# ======================================================
# Save Autoencoder + Encoder Models
# ======================================================
def save_models(autoencoder, encoder, output_dir="output"):
    """
    Saves both the full autoencoder and the encoder (latent extractor).
    - Autoencoder: can be used for full reconstruction tasks.
    - Encoder:     can be used for interpretability (e.g. SHAP).
    """
    os.makedirs(output_dir, exist_ok=True)

    auto_path = os.path.join(output_dir, "dense_autoencoder.h5")
    enc_path  = os.path.join(output_dir, "dense_encoder.h5")

    autoencoder.save(auto_path)
    encoder.save(enc_path)

    print(f"✅ Saved Autoencoder model to {auto_path}")
    print(f"✅ Saved Encoder model to {enc_path}")


# ======================================================
# Run function
# ======================================================
def run(X_train, X_val, X_test,
        y_train, y_val, y_test,
        x_scaler, y_scaler,
        latent_dim=25,
        output_dir="output",
        output_name="dense_latent.csv"):
    """
    Trains Dense Autoencoder, saves latent features + targets into CSV,
    and also saves trained models for later use (e.g. SHAP interpretability).
    """
    input_dim = X_train.shape[1]

    # 1. Build & train AE
    ae_model, encoder = build_dense_autoencoder(input_dim, latent_dim=latent_dim)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    ]

    ae_model.fit(
        X_train, X_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=1,
        shuffle=False
    )

    # 2. Encode to latent features
    Z_train = encoder.predict(X_train, verbose=0)
    Z_val   = encoder.predict(X_val,   verbose=0)
    Z_test  = encoder.predict(X_test,  verbose=0)

    # 3. Save latent features + y into CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    df_train = pd.DataFrame(Z_train, columns=[f"latent_{i}" for i in range(Z_train.shape[1])])
    df_train["target"] = y_train
    df_train["split"] = "train"

    df_val = pd.DataFrame(Z_val, columns=[f"latent_{i}" for i in range(Z_val.shape[1])])
    df_val["target"] = y_val
    df_val["split"] = "val"

    df_test = pd.DataFrame(Z_test, columns=[f"latent_{i}" for i in range(Z_test.shape[1])])
    df_test["target"] = y_test
    df_test["split"] = "test"

    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    df_all.to_csv(output_path, index=False)
    print(f"✅ Saved reduced features to {output_path}")

    # 4. Save models for interpretability later
    save_models(ae_model, encoder, output_dir=output_dir)

    return df_all