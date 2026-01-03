from pathlib import Path
import tensorflow as tf

from src.dataset import load_jena_csv, time_split, scale_splits, make_windows
from src.model import build_lstm_model


def main():
    csv_path = Path("data/jena_climate_2009_2016.csv")
    if not csv_path.exists():
        raise FileNotFoundError("Dataset not found. Run baseline step first.")

    # Load + prepare data
    data = load_jena_csv(str(csv_path))
    train, val, test = time_split(data)
    train_s, val_s, test_s, _ = scale_splits(train, val, test)

    window_size = 72
    horizon = 1
    target_col = 0

    X_train, y_train = make_windows(train_s, window_size, horizon, target_col)
    X_val, y_val = make_windows(val_s, window_size, horizon, target_col)

    # Build model
    model = build_lstm_model(
        window_size=window_size,
        num_features=X_train.shape[2],
        hidden_size=64
    )

    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )
    ]

    # Train
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks
    )

    # Save model
    Path("experiments").mkdir(exist_ok=True)
    model.save("experiments/lstm_model.keras")

    print("Training complete. Model saved to experiments/lstm_model")


if __name__ == "__main__":
    main()

