import numpy as np
import pandas as pd
import pytest
from market_predictor.data import clean_prices, demo_prices, download_prices
from market_predictor.model import prepare_windows, metrics


def test_scaler_never_sees_validation_or_test():
    prices = np.arange(1, 201, dtype=float)
    windows = prepare_windows(prices, 20)
    prices[140:] = 100000
    changed = prepare_windows(prices, 20)
    assert changed.minimum == windows.minimum
    assert changed.span == windows.span
    np.testing.assert_array_equal(changed.x[:changed.train_end], windows.x[:windows.train_end])
    np.testing.assert_array_equal(changed.y[:changed.train_end], windows.y[:windows.train_end])
    assert changed.y[-1] > 1


def test_windows_align_with_targets():
    frame = demo_prices()
    w = prepare_windows(frame.Close, 30)
    assert w.x.shape == (570, 30, 1)
    assert w.y[0] * w.span + w.minimum == pytest.approx(frame.Close.iloc[30])
    assert w.x[0, -1, 0] * w.span + w.minimum == pytest.approx(frame.Close.iloc[29])


def test_cleaning_sorts_and_deduplicates():
    frame = demo_prices().reset_index()
    shuffled = pd.concat([frame.iloc[::-1], frame.iloc[:1]])
    cleaned = clean_prices(shuffled)
    assert len(cleaned) == len(frame)
    assert cleaned.index.is_monotonic_increasing


def test_multilevel_yahoo_columns():
    frame = demo_prices()
    frame.columns = pd.MultiIndex.from_tuples([("Close", "AAPL")])
    assert list(clean_prices(frame).columns) == ["Close"]


def test_empty_yahoo_response(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *args, **kwargs: pd.DataFrame())
    with pytest.raises(ValueError, match="no prices"):
        download_prices("INVALID", "2024-01-01", "2025-01-01")


def test_bad_csv_and_insufficient_history():
    with pytest.raises(ValueError):
        clean_prices(pd.DataFrame({"Price": [1, 2]}))
    with pytest.raises(ValueError):
        prepare_windows(np.arange(1, 51), 30)


def test_metrics_known_errors():
    scores = metrics([100, 200], [110, 190], [90, 210])
    assert scores["MAE"] == 10
    assert scores["RMSE"] == 10
    assert scores["MAPE (%)"] == pytest.approx(7.5)
    assert scores["Direction accuracy (%)"] == 100
