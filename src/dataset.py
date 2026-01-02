import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_jena_csv(csv_path: str) -> np.ndarray:
    """Load Jena climate CSV and return float32 numpy array of features (no Date Time col)."""
    df = pd.read_csv(csv_path)
    if "Date Time" in df.columns:
        df = df.drop(columns=["Date Time"])
    return df.values.astype(np.float32)


def time_split(data: np.ndarray, train_frac=0.7, val_frac=0.15):
    n = len(data)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def scale_splits(train: np.ndarray, val: np.ndarray, test: np.ndarray):
    """Fit scaler on train only; transform all splits."""
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train)
    val_s = scaler.transform(val)
    test_s = scaler.transform(test)
    return train_s, val_s, test_s, scaler


def make_windows(data: np.ndarray, window_size: int, horizon: int, target_col: int = 0):
    """
    Create sliding windows.
    X shape: (num_samples, window_size, num_features)
    y shape: (num_samples,) for horizon=1 else (num_samples, horizon)
    """
    X, y = [], []
    max_i = len(data) - window_size - horizon + 1
    for i in range(max_i):
        X.append(data[i:i + window_size])
        y_seq = data[i + window_size:i + window_size + horizon, target_col]
        y.append(y_seq if horizon > 1 else y_seq[0])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
