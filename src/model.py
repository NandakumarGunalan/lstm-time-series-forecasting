import tensorflow as tf
from tensorflow.keras import layers, models


def build_lstm_model(
    window_size: int,
    num_features: int,
    hidden_size: int = 64,
    dropout: float = 0.2
):
    """
    Build a simple LSTM model for time-series forecasting.
    """
    model = models.Sequential([
        layers.Input(shape=(window_size, num_features)),
        layers.LSTM(hidden_size, dropout=dropout),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=["mae"]
    )

    return model

