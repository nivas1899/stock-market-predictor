# Local validation record

Verified on 15 October 2026 with Windows, Python 3.12.14, TensorFlow 2.20.0, and the package snapshot in `requirements-lock.txt`.

- Full automated suite: **30 passed**. Includes actual TensorFlow training, model serialization/reload, interactive dashboard training, and unit tests.
- Additional randomized check: **300 A* cases matched exhaustive enumeration**, across fee rates and order budgets.
- Live Yahoo Finance retrieval: **252 AAPL daily rows**, from 2024-01-01 inclusive to 2025-01-01 exclusive.
- Real-data CLI training and export: completed with 15 maximum epochs, 30-session lookback, and 5-session forecast horizon.

Example development-run test metrics:

| Model | MAE | RMSE | MAPE |
| --- | ---: | ---: | ---: |
| Residual LSTM | 1.8983 | 2.3866 | 0.7928% |
| Previous close | 1.9165 | 2.4044 | 0.8003% |

In that run the daily strategy stayed in cash after fees: strategy return 0%, compared with approximately 11.97% for buy and hold. The result illustrates why forecast error and trading performance must be reported separately.

These are development sanity-check results, not an independently held-out investment-performance claim. The architecture was revised during development after inspecting an earlier run. A formal research evaluation should freeze model choices and use a later untouched period. Results can vary with provider adjustments, numerical libraries, and hardware.

Streamlit's local event-loop sockets require permission outside this environment's restricted sandbox. The full suite passed with those local sockets allowed. Seven third-party Keras/NumPy deprecation warnings were observed; they did not fail model training or reload.
