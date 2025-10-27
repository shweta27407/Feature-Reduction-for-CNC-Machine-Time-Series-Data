# data/dataloader.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def _check_split(split: Tuple[float, float, float]):
    if len(split) != 3:
        raise ValueError("split must be a 3-tuple: (train, val, test).")
    if not np.isclose(sum(split), 1.0):
        raise ValueError(f"split must sum to 1.0, got {split} (sum={sum(split):.3f}).")

def load_dataset(path: str,
                 split: Tuple[float, float, float],
                 feature_cols_slice: slice = slice(0, 52),
                 target_col: str = "CURRENT|6"):
    """
    Loads data, applies chronological split, scales on train only.
    Returns scaled arrays + scalers. No windowing here (done later).
    """
    _check_split(split)
    df = pd.read_csv(path)

    feature_cols = df.columns[feature_cols_slice]
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)

    N = len(X_raw)
    train_end = int(split[0] * N)
    val_end   = int((split[0] + split[1]) * N)

    X_train, y_train = X_raw[:train_end], y_raw[:train_end]
    X_val,   y_val   = X_raw[train_end:val_end], y_raw[train_end:val_end]
    X_test,  y_test  = X_raw[val_end:], y_raw[val_end:]

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))

    X_train = x_scaler.transform(X_train)
    X_val   = x_scaler.transform(X_val)
    X_test  = x_scaler.transform(X_test)

    y_train = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val   = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train, X_val, X_test, y_train, y_val, y_test, x_scaler, y_scaler



def make_sequences(X, y, window_size: int, predict_offset: int = 1):
    """
    Builds (X_seq, y_seq) for sequence models.
    y at index i targets the value at i+window_size-1+predict_offset.
    """
    Xs, ys = [], []
    max_i = len(X) - window_size - predict_offset + 1
    for i in range(max_i):
        Xs.append(X[i:i+window_size])
        ys.append(y[i + window_size - 1 + predict_offset])
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)