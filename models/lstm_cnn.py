import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from data.dataloader import make_sequences   # ✅ use the common function

# ======================================================
# Build Autoencoder
# ======================================================
def build_autoencoder(timesteps, features, latent_dim=25, multitask_alpha=0.5):
    inputs = layers.Input(shape=(timesteps, features))

    # CNN encoder
    z = layers.Conv1D(64, kernel_size=5, padding='same')(inputs)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)

    z = layers.Conv1D(64, kernel_size=3, dilation_rate=2, padding='same')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Activation('relu')(z)
    z = layers.Dropout(0.1)(z)

    # RNN encoder
    z = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(z)
    z = layers.LSTM(64, return_sequences=False)(z)

    # Bottleneck (latent space)
    latent = layers.Dense(latent_dim, activation=None, name="latent")(z)
    latent = layers.LayerNormalization()(latent)
    latent = layers.Dropout(0.2)(latent)

    # Decoder
    d = layers.RepeatVector(timesteps)(latent)
    d = layers.LSTM(128, return_sequences=True)(d)
    d = layers.LSTM(64, return_sequences=True)(d)
    decoded = layers.TimeDistributed(layers.Dense(features, activation=None),
                                     name="reconstruction")(d)

    # Optional regression head
    reg_head = layers.Dense(64, activation='relu')(latent)
    reg_head = layers.Dropout(0.2)(reg_head)
    reg_out = layers.Dense(1, activation=None, name="latent_to_target")(reg_head)

    if multitask_alpha > 0.0:
        ae_model = models.Model(inputs, outputs=[decoded, reg_out], name="multitask_ae")
        ae_model.compile(
            optimizer=optimizers.Adam(1e-3),
            loss={"reconstruction": "mse", "latent_to_target": "mse"},
            loss_weights={"reconstruction": 1.0, "latent_to_target": multitask_alpha}
        )
    else:
        ae_model = models.Model(inputs, decoded, name="ae_only")
        ae_model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")

    encoder = models.Model(inputs, latent, name="encoder")
    return ae_model, encoder


# ======================================================
# Run function (for feature reduction only)
# ======================================================
import pandas as pd
import os

def run(X_train, X_val, X_test,
        y_train, y_val, y_test,
        x_scaler, y_scaler,
        window_size=60,
        latent_dim=25,
        multitask_alpha=0.5,
        output_dir="output",
        output_name="lstm_cnn_latent.csv"):
    """
    Trains LSTM+CNN Autoencoder and saves reduced features to CSV.
    """

    timesteps, features = X_train.shape[1], X_train.shape[2]

    # 2. Build & train AE
    ae_model, encoder = build_autoencoder(timesteps, features,
                                          latent_dim=latent_dim,
                                          multitask_alpha=multitask_alpha)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
    ]


    print("Final X_train shape before fit:", X_train.shape)
    print("Final y_train shape before fit:", y_train.shape)

    if multitask_alpha > 0.0:
        ae_model.fit(
            X_train, {"reconstruction": X_train, "latent_to_target": y_train},
            epochs=100, batch_size=64,
            validation_data=(X_val, {"reconstruction": X_val, "latent_to_target": y_val}),
            callbacks=callbacks, verbose=1, shuffle=True
        )
    else:
        ae_model.fit(
            X_train, X_train,
            epochs=100, batch_size=64,
            validation_data=(X_val, X_val),
            callbacks=callbacks, verbose=1, shuffle=True
        )

    # 3. Encode to latent features
    Z_train = encoder.predict(X_train, verbose=0)
    Z_val   = encoder.predict(X_val,   verbose=0)
    Z_test  = encoder.predict(X_test,  verbose=0)

    # 4. Save latent features + y into CSV
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)

    # concatenate latent features and targets
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
    return df_all