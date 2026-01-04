from pathlib import Path
import numpy as np

from src.dataset import load_jena_csv, time_split, scale_splits, make_windows
from src.evaluate import naive_last_value, compute_metrics

import tensorflow as tf


def main():
    csv_path = Path("data/jena_climate_2009_2016.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}")

    # Load + split + scale (fit on train only)
    data = load_jena_csv(str(csv_path))
    train, val, test = time_split(data, train_frac=0.7, val_frac=0.15)
    train_s, val_s, test_s, _ = scale_splits(train, val, test)

    # Same window/horizon as baseline + training
    window_size = 72
    horizon = 1
    target_col = 0

    X_test, y_test = make_windows(test_s, window_size, horizon, target_col=target_col)

    # Baseline
    y_pred_naive = naive_last_value(X_test, target_col=target_col)
    mae_naive, rmse_naive = compute_metrics(y_test, y_pred_naive)

    # LSTM
    model_path = Path("experiments/lstm_model.keras")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model at {model_path}. Run training first: python -m src.train"
        )

    model = tf.keras.models.load_model(model_path)
    y_pred_lstm = model.predict(X_test, verbose=0).reshape(-1)

    mae_lstm, rmse_lstm = compute_metrics(y_test, y_pred_lstm)

    print(f"Naive (test) | MAE: {mae_naive:.4f} | RMSE: {rmse_naive:.4f}")
    print(f"LSTM  (test) | MAE: {mae_lstm:.4f} | RMSE: {rmse_lstm:.4f}")

    # Write results markdown
    out = Path("experiments/results.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    content = f"""## Results

### Dataset
- Jena Climate (scaled using train-only statistics)
- Window: {window_size}, Horizon: {horizon}, Target col: {target_col}

### Test Metrics
| Model | MAE | RMSE |
|---|---:|---:|
| Naive last-value | {mae_naive:.4f} | {rmse_naive:.4f} |
| LSTM | {mae_lstm:.4f} | {rmse_lstm:.4f} |

Notes:
- Naive is a strong baseline for short-horizon forecasting.
- LSTM should beat naive if it learns temporal patterns beyond last value.
"""
    out.write_text(content)
    print(f"Wrote results to {out}")


if __name__ == "__main__":
    main()

