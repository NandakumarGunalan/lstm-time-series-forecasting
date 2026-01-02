from pathlib import Path
from src.dataset import load_jena_csv, time_split, scale_splits, make_windows
from src.evaluate import naive_last_value, compute_metrics


def main():
    csv_path = Path("data/jena_climate_2009_2016.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset at {csv_path}. Run the download step first.")

    data = load_jena_csv(str(csv_path))
    train, val, test = time_split(data, train_frac=0.7, val_frac=0.15)
    train_s, val_s, test_s, _ = scale_splits(train, val, test)

    window_size = 72
    horizon = 1
    target_col = 0

    X_test, y_test = make_windows(test_s, window_size, horizon, target_col=target_col)
    y_pred = naive_last_value(X_test, target_col=target_col)

    mae, rmse = compute_metrics(y_test, y_pred)
    print(f"Naive baseline (test) | MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # write results
    results_path = Path("experiments/results.md")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""## Baseline Results

Dataset: Jena Climate (scaled using train-only statistics)

Naive last-value predictor (window={window_size}, horizon={horizon}):

- MAE: {mae:.4f}
- RMSE: {rmse:.4f}

This baseline serves as a lower bound for LSTM performance.
"""
    results_path.write_text(content)
    print(f"Wrote results to {results_path}")


if __name__ == "__main__":
    main()
