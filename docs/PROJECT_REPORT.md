# Project report

**Title:** Stock Market Predictor using LSTM and Search Algorithms

**Techniques:** Recurrent Neural Networks, A* for strategy pathing

**Tools:** Python, TensorFlow/Keras, yfinance, pandas, NumPy, Streamlit, Plotly

## Abstract

This project connects a sequence-learning forecasting model with a search-based decision model. An LSTM predicts daily closing prices from historical windows. A* then finds a cash/long strategy over the predicted path, subject to transaction fees and an order limit. A dashboard presents historical prices, model errors, future forecasts, and strategy decisions. A separate causal historical simulation demonstrates the impact of forecast error and transaction costs. The project is intended for education and experimentation, not automated investment.

## Problem and objectives

Price forecasting and strategy planning are distinct tasks. A forecast may look plausible while providing little advantage over the previous closing price. Even a correct directional prediction may be unprofitable after fees. The objectives are to implement an actual LSTM; evaluate it without future-data leakage; demonstrate an explicitly defined A* search problem; and expose the results through a usable interface and reproducible CLI.

## System design

Data enters from Yahoo Finance, a local CSV, or a labeled synthetic generator. Validation normalizes dates, sorts observations, removes missing values and duplicate dates, and enforces positive finite closes. The model layer creates chronological windows and trains the LSTM. The experiment pipeline calculates held-out metrics, recursive forecasts, A* paths, and historical equity. The UI and CLI both call this shared pipeline.

The detailed search state, cost transformation, and admissible heuristic are documented in the repository README. The finite state graph tracks time, exposure, and order count. It supports exact optimization without exponential enumeration of all action paths.

## Evaluation design

The first 70% of observations supplies training targets and scaler parameters. The next 15% supplies validation targets for early stopping; the final 15% supplies test targets. Model weights are frozen for evaluation. Metrics include MAE, RMSE, MAPE, and directional accuracy. The previous-close baseline is reported alongside the LSTM. Trading evaluation charges both entry and exit fees, uses only predictions available at the decision time, and compares final wealth with buy and hold.

No fixed accuracy claim is made. Each experiment exports its actual metrics and configuration. A synthetic-data smoke test establishes that the software runs; it does not establish a market edge. A real-data experiment should specify the ticker, date range, hyperparameters, fees, and all test metrics, including cases where the baseline wins.

## Validation

Automated tests compare A* against exhaustive enumeration with different prices, fee levels, and order budgets. Other tests check scaler isolation, window/target alignment, date cleaning, missing Yahoo data, invalid input, transaction-cost accounting, TensorFlow training and serialization, and interaction with the Streamlit dashboard. GitHub Actions runs the test suite and a CLI smoke experiment.

## Limitations and future work

Only the closing price is used as an input feature. The model does not estimate calibrated prediction intervals. Recursive forecasts accumulate errors and the training/test regime may change over time. The historical strategy uses daily closed round trips and an idealized close-to-close execution convention; it does not validate the multi-session future plan. The optimizer is optimal for its forecast path, not for unknown market prices. Further work could add walk-forward retraining, uncertainty-aware planning, explicit slippage, exchange calendars, and comparisons against additional forecasting models. Such extensions should be evaluated on later, untouched data.

## Conclusion

The implementation demonstrates the requested integration of recurrent sequence learning and heuristic search in an executable research application. Its main output is a reproducible experiment with transparent assumptions, measurable forecast errors, and inspectable strategy paths.
