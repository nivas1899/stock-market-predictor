"""Shared dashboard/CLI pipeline and causal out-of-sample simulation."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .model import prepare_windows, fit_model, evaluate, forecast
from .strategy import plan_strategy, buy_and_hold


@dataclass
class Result:
    model: object
    windows: object
    history: dict
    evaluation: pd.DataFrame
    scores: pd.DataFrame
    future: pd.DataFrame
    plan: object
    backtest: pd.DataFrame
    summary: dict


def run_experiment(frame, lookback=30, epochs=15, horizon=10, capital=10000., fee=0.001,
                   max_orders=8):
    windows = prepare_windows(frame.Close, lookback)
    model, history = fit_model(windows, epochs)
    evaluation, scores = evaluate(model, windows, frame)
    predicted = forecast(model, windows, frame.Close, horizon)
    # Business-day labels are approximate; exchange holidays are not modeled.
    dates = pd.bdate_range(frame.index[-1] + pd.offsets.BDay(1), periods=horizon)
    future = pd.DataFrame({"Forecast": predicted}, index=dates)
    path = np.r_[frame.Close.iloc[-1], predicted]
    plan = plan_strategy(path, capital, fee, max_orders)
    equity, rows = capital, []
    # Each prediction uses only the lookback preceding its target. Positions are
    # closed daily: no future actual price enters the decision, and both fees apply.
    for date, row in evaluation.iterrows():
        decision = plan_strategy([row["Previous close baseline"], row.LSTM],
                                 equity, fee, min(max_orders, 2))
        exposure = int(decision.ledger.Exposure.iloc[0])
        if exposure:
            equity *= row.Actual / row["Previous close baseline"] * (1-fee)**2
        rows.append({"Date": date, "Equity": equity, "Exposure": exposure})
    backtest = pd.DataFrame(rows).set_index("Date")
    first_price = evaluation["Previous close baseline"].iloc[0]
    backtest["Buy and hold"] = capital * (1-fee) * evaluation.Actual / first_price
    backtest.loc[backtest.index[-1], "Buy and hold"] *= 1-fee
    equity_path = np.r_[capital, backtest.Equity.to_numpy()]
    drawdowns = equity_path / np.maximum.accumulate(equity_path) - 1
    summary = {"Test strategy return (%)": (equity / capital - 1) * 100,
               "Test buy-and-hold return (%)": (buy_and_hold(
                   [first_price, evaluation.Actual.iloc[-1]], capital, fee) / capital - 1) * 100,
               "Test maximum drawdown (%)": float(drawdowns.min() * 100),
               "Test round trips": int(backtest.Exposure.sum()),
               "Projected strategy return (%)": (plan.final_value / capital - 1) * 100,
               "Projected buy-and-hold return (%)": (buy_and_hold(path, capital, fee)/capital-1)*100}
    return Result(model, windows, history, evaluation, scores, future, plan, backtest, summary)
