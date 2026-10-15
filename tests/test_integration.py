import numpy as np
import pandas as pd
import pytest
from market_predictor.data import demo_prices
from market_predictor.pipeline import run_experiment
from market_predictor.model import forecast


@pytest.mark.integration
def test_real_tensorflow_pipeline_and_reload(tmp_path):
    import tensorflow as tf
    result = run_experiment(demo_prices(160), lookback=10, epochs=1, horizon=3)
    assert len(result.future) == 3
    assert np.isfinite(result.scores.to_numpy()).all()
    assert (result.future.Forecast > 0).all()
    assert len(result.evaluation) == 24
    assert result.plan.orders <= 8
    result.model.save(tmp_path / "model.keras")
    loaded = tf.keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(forecast(loaded, result.windows, demo_prices(160).Close, 3),
                               result.future.Forecast, rtol=1e-5)


def test_backtest_uses_predicted_decisions(monkeypatch):
    import market_predictor.pipeline as pipeline
    frame = demo_prices(160)
    evaluation = pd.DataFrame({"Actual": [110., 90.], "LSTM": [120., 80.],
                              "Previous close baseline": [100., 110.]}, index=frame.index[-2:])
    monkeypatch.setattr(pipeline, "fit_model", lambda *a: (object(), {}))
    monkeypatch.setattr(pipeline, "evaluate", lambda *a: (evaluation, pd.DataFrame()))
    monkeypatch.setattr(pipeline, "forecast", lambda *a: np.array([100., 110.]))
    result = pipeline.run_experiment(frame, lookback=10, horizon=2, fee=.01)
    assert result.backtest.Exposure.tolist() == [1, 0]
    assert result.backtest.Equity.iloc[-1] == pytest.approx(10000 * 1.1 * .99**2)
