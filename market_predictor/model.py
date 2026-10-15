"""Chronological train/validation/test splits with a training-only scaler."""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray
    train_end: int
    validation_end: int
    minimum: float
    span: float
    lookback: int


def prepare_windows(prices, lookback=30) -> Windows:
    prices = np.asarray(prices, dtype=float).reshape(-1)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Prices must be finite and positive.")
    n = len(prices)
    train_stop, val_stop = int(n * 0.7), int(n * 0.85)
    if not 5 <= lookback <= 120 or train_stop - lookback < 20 or n - val_stop < 10:
        raise ValueError("Insufficient history for this lookback; add data or shorten the window.")
    minimum = float(prices[:train_stop].min())
    span = max(float(np.ptp(prices[:train_stop])), 1e-8)
    scaled = (prices - minimum) / span
    x = np.asarray([scaled[i-lookback:i] for i in range(lookback, n)], dtype=np.float32)[..., None]
    return Windows(x, scaled[lookback:].astype(np.float32), train_stop-lookback,
                   val_stop-lookback, minimum, span, lookback)


def fit_model(windows: Windows, epochs=15, units=32, seed=42):
    if not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError("Epochs must be an integer between 1 and 100.")
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.layers.Input(shape=(windows.lookback, 1))
    hidden = tf.keras.layers.LSTM(units)(inputs)
    hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
    change = tf.keras.layers.Dense(1, kernel_initializer="zeros", name="predicted_change")(hidden)
    last_close = tf.keras.layers.Flatten()(
        tf.keras.layers.Cropping1D(cropping=(windows.lookback-1, 0))(inputs))
    # A persistence skip connection lets the recurrent network learn corrections
    # instead of relearning the price level; built-in layers serialize safely.
    output = tf.keras.layers.Add(name="next_close")([last_close, change])
    model = tf.keras.Model(inputs, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    t, v = windows.train_end, windows.validation_end
    history = model.fit(windows.x[:t], windows.y[:t],
                        validation_data=(windows.x[t:v], windows.y[t:v]),
                        epochs=epochs, batch_size=32, shuffle=False, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss", patience=5, restore_best_weights=True)])
    return model, history.history


def metrics(actual, predicted, previous):
    actual, predicted, previous = map(np.asarray, (actual, predicted, previous))
    errors = predicted - actual
    return {"MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2))),
            "MAPE (%)": float(np.mean(np.abs(errors / actual)) * 100),
            "Direction accuracy (%)": float(np.mean(np.sign(actual-previous) ==
                                                     np.sign(predicted-previous)) * 100)}


def evaluate(model, windows: Windows, frame: pd.DataFrame):
    start = windows.validation_end
    predicted = np.asarray(model(windows.x[start:], training=False)).reshape(-1)
    predicted = predicted * windows.span + windows.minimum
    offset = windows.lookback + start
    actual = frame.Close.to_numpy()[offset:]
    previous = frame.Close.to_numpy()[offset-1:-1]
    results = pd.DataFrame({"Actual": actual, "LSTM": predicted,
                            "Previous close baseline": previous}, index=frame.index[offset:])
    scores = pd.DataFrame({"LSTM": metrics(actual, predicted, previous),
                           "Previous close baseline": metrics(actual, previous, previous)}).T
    return results, scores


def forecast(model, windows: Windows, prices, horizon=10):
    if not 1 <= horizon <= 60:
        raise ValueError("Forecast horizon must be between 1 and 60 sessions.")
    sequence = (np.asarray(prices, dtype=float)[-windows.lookback:] - windows.minimum) / windows.span
    output = []
    for _ in range(horizon):
        value = float(np.asarray(model(sequence[None, :, None].astype(np.float32),
                                       training=False)).reshape(-1)[0])
        price = value * windows.span + windows.minimum
        if not np.isfinite(price) or price <= 0:
            raise ValueError("Model produced an invalid future price. Try more data or retrain.")
        output.append(price)
        sequence = np.append(sequence[1:], value)
    return np.asarray(output)
