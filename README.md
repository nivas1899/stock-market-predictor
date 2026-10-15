# Stock Market Predictor using LSTM and Search Algorithms

A working Python research project combining **recurrent neural networks (LSTM)** for daily closing-price prediction with **A\*** search for strategy path planning. Built with **TensorFlow/Keras**, **yfinance**, and **Streamlit**.

## Features

- Interactive dashboard with historical prices, future forecasts, training curves, metrics, and strategy tables.
- Yahoo Finance adjusted daily closes, CSV uploads, and a deterministic **synthetic offline demo**.
- Chronological 70% training / 15% validation / 15% test split; training-only scaling and validation early stopping.
- MAE, RMSE, MAPE, and directional accuracy, compared with a previous-close baseline.
- Exact A* optimization of cash/long paths with proportional fees, an order budget, and terminal liquidation.
- A causal test simulation using one-session A* decisions and a buy-and-hold comparison.
- CLI exports: saved Keras model, scaler metadata, predictions, forecasts, strategy, metrics, and equity history.
- Automated tests, including exhaustive-search comparisons and actual TensorFlow training; GitHub Actions CI.

## Install and run

Use **64-bit Python 3.12**. A CPU is sufficient; TensorFlow installation is a large download.

```bash
git clone https://github.com/nivas1899/stock-market-predictor.git
cd stock-market-predictor
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS / Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). Leave **Synthetic offline demo** selected and click **Train & analyze** for an initial run. The demo does not download market data and its prices are explicitly synthetic. Use Yahoo Finance with a ticker such as `AAPL`, `MSFT`, or `RELIANCE.NS` for real price history. End dates are exclusive.

For an exact dependency snapshot from the tested Windows/Python 3.12 environment, use `requirements-lock.txt`. `requirements.txt` is the cross-platform installation entry point.

## Command-line examples

Run these with the virtual environment active, or replace `python` with its full venv path:

```bash
python train.py --demo --epochs 15 --horizon 10
python train.py --ticker AAPL --start 2020-01-01 --end 2026-01-01 --epochs 25
python train.py --ticker RELIANCE.NS --start 2020-01-01 --end 2026-01-01
python train.py --csv data/example.csv --lookback 20 --epochs 5 --horizon 5
python train.py --demo --fee 0.001 --capital 10000 --max-orders 8
```

Results are written to `artifacts/` (excluded from Git). `--fee 0.001` means 0.1% per order, equivalent to 10 basis points in the dashboard. Capital and returns use the ticker's quote currency; there is no currency conversion.

CSV format (at least 100 valid observations; larger lookbacks need more):

```csv
Date,Close
2024-01-02,100.50
2024-01-03,101.20
```

The two rows above only illustrate the format. A complete 160-row synthetic example is provided in `data/example.csv`. Prefer consistently adjusted prices for real CSV data. Rows are sorted, duplicate dates keep the last observation, and missing/invalid values are removed. Nonpositive or infinite prices are rejected.

## Method

The LSTM consumes the preceding `lookback` closing prices and predicts one next closing price. The recurrent branch is `Input → LSTM(32) → Dense(16, ReLU) → Dense(1)` and predicts a correction added to the last observed scaled close through a persistence skip connection. The correction layer starts at zero, so training begins at the baseline rather than an arbitrary price level. This improves short-run stability without guaranteeing improvement over the baseline. Training uses Adam and mean squared error. Validation loss controls early stopping. Training and validation are never shuffled.

Only the first 70% of prices determine the min/max scaler. Validation and test prices may legitimately scale outside `[0, 1]`. Test windows include previously observed closes, so evaluation measures **rolling one-step prediction**. Future forecasts recursively feed predicted values back into the model, so they are a different and harder task. The same model remains frozen after training; it is not silently refitted on the test set.

The previous-close baseline predicts the last observed close. A low error alone does not establish predictive value: compare the LSTM with this baseline. Direction accuracy treats a zero predicted change as a zero direction; this is relevant for the persistence baseline. No accuracy or profit level is guaranteed.

### A* formulation

Each state is `(interval, exposure, orders_used)`, with exposure 0 for cash and 1 for an all-capital long position. Changing exposure uses an order and pays a proportional fee. Any final long position is liquidated and that exit counts toward the budget. The optimizer starts in cash and ends in cash.

For a forecast log-return `r_i`, fee `f`, and `m_i` entry/exit orders on an interval:

```text
log_growth_i = exposure_i * r_i + m_i * log(1 - f)
B_i = abs(r_i) + 1
edge_cost_i = B_i - log_growth_i
h(state) = sum over remaining intervals of [B_i - max(0, r_i)]
```

All edge costs are nonnegative. Every complete path has the same sum of `B_i`, so minimizing total cost maximizes terminal log wealth. The heuristic relaxes fees and the switching budget, allowing free selection of the best exposure each interval. This gives a lower bound on remaining cost and is admissible and consistent. A heap-based A* search with best-cost pruning therefore finds an optimal path **for the supplied forecast and model constraints**. Tests compare it with all possible paths on small instances.

There are at most `2 × (horizon + 1) × (max_orders + 1)` states. Fractional shares are allowed. The model excludes leverage, shorting, slippage, spread, taxes, liquidity constraints, and cash interest. It does not connect to a broker.

### Historical simulation versus projected plan

- **Historical test simulation:** for each untouched test date, a one-session A* plan chooses cash or a round trip from the previous actual close to the next predicted close. The position is closed the next session; actual next prices determine realized equity. Each trade pays both fees. Decisions never use the next actual price. The order budget applies separately to each daily plan, so budgets below two disable trades. This is a deliberately simple policy, separate from the longer future plan.
- **Projected plan:** multi-session A* search over the recursive future forecast. These returns are hypothetical outcomes if that path occurred. They are not a historical backtest or evidence of future profit.

All historical decisions assume execution at the observed close used by the prediction and at the next close. This is an idealized research convention; real execution would add timing and trading costs. Future date labels approximate weekdays and do not apply exchange holiday calendars. Synthetic demo results cannot establish performance on real markets.

## Tests

```bash
python -m pytest -q
python -m pytest -q -m "not integration"
python train.py --demo --epochs 1 --horizon 3 --output artifacts/smoke
```

The full suite includes actual LSTM training, saved-model reload, dashboard interaction, scaler leakage checks, causal fee accounting, invalid-data handling, and A* versus exhaustive enumeration. CI uses offline synthetic data so Yahoo outages do not break the build.

## Project layout

```text
app.py                       Streamlit dashboard
train.py                     CLI and export pipeline
market_predictor/data.py     Yahoo / CSV / synthetic data validation
market_predictor/model.py    Splits, scaling, LSTM, metrics, recursive forecast
market_predictor/strategy.py A* planner and buy-and-hold benchmark
market_predictor/pipeline.py Shared experiment and causal test simulation
tests/                       Unit, integration, and dashboard tests
docs/PROJECT_REPORT.md       Academic project summary and limitations
data/example.csv             Complete synthetic CSV example
.github/workflows/tests.yml  Automated CI
```

## Troubleshooting

- **No prices from Yahoo:** check the symbol and dates, wait if rate limited, or use CSV/demo. No synthetic data is silently substituted for live data.
- **Insufficient history:** widen the date range or reduce the lookback. The training partition must contain at least 20 target windows.
- **TensorFlow DLL/import error on Windows:** use 64-bit Python 3.12 and install the Microsoft Visual C++ runtime as described in the [official TensorFlow installation guide](https://www.tensorflow.org/install/pip).
- **Training takes time:** lower epochs or lookback for a first run. Early stopping can stop before the chosen epoch limit.

Technical references: [Keras LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM), [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html). Yahoo data access is subject to the provider's terms; yfinance is an unofficial research tool.

## License

MIT. Educational research software; forecasts and simulations are not investment advice.
