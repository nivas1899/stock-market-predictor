"""Run with: streamlit run app.py"""
from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from market_predictor.data import clean_prices, demo_prices, download_prices
from market_predictor.pipeline import run_experiment

st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")
st.title("Stock Market Predictor")
st.caption("LSTM recurrent neural network · A* strategy search · TensorFlow / Keras")
st.info("Research and education project. Forecasts are uncertain; projected returns are simulations.")

with st.sidebar:
    st.header("Experiment settings")
    source = st.selectbox("Data source", ["Synthetic offline demo", "Yahoo Finance", "Upload CSV"])
    ticker = st.text_input("Ticker", "AAPL", help="Indian equities: RELIANCE.NS, TCS.NS")
    start = st.date_input("Start date", date.today() - timedelta(days=1095))
    end = st.date_input("End date (exclusive)", date.today())
    upload = st.file_uploader("CSV with Date, Close columns", type=["csv"]) if source == "Upload CSV" else None
    lookback = st.slider("Lookback sessions", 5, 120, 30)
    epochs = st.slider("Maximum training epochs", 1, 100, 15)
    horizon = st.slider("Forecast sessions", 1, 30, 10)
    capital = st.number_input("Initial capital (quote currency)", min_value=100., value=10000., step=100.)
    fee_bps = st.slider("Fee per order (basis points)", 0, 100, 10)
    max_orders = st.slider("Maximum orders per forecast plan", 0, 20, 8)
    run = st.button("Train & analyze", type="primary", use_container_width=True)

if run:
    try:
        with st.spinner("Loading prices, training the LSTM, and searching strategy paths…"):
            if source == "Synthetic offline demo":
                frame = demo_prices()
                label = "SYNTHETIC DEMO · not real market prices"
            elif source == "Yahoo Finance":
                frame = download_prices(ticker, str(start), str(end))
                label = f"{ticker.upper()} · Yahoo Finance adjusted daily close"
            else:
                if upload is None:
                    raise ValueError("Choose a CSV file before training.")
                frame = clean_prices(pd.read_csv(upload))
                label = f"Uploaded CSV · {upload.name}"
            result = run_experiment(frame, lookback, epochs, horizon, capital, fee_bps/10000, max_orders)
            st.session_state["experiment"] = (frame, result, label, capital)
    except Exception as exc:
        st.error(f"Experiment could not finish: {exc}")

if "experiment" not in st.session_state:
    st.subheader("From price history to a strategy path")
    st.write("Select demo, live data, or a CSV, then click **Train & analyze**. The offline demo works without a market-data connection.")
    st.markdown("1. Split history into 70% training, 15% validation, and 15% testing.\n2. Train an LSTM on past closing-price windows.\n3. Evaluate against the previous-close baseline.\n4. Use A* to plan cash/long decisions over the forecast, including fees.")
    st.stop()

frame, result, label, saved_capital = st.session_state["experiment"]
st.caption(f"Completed experiment: {label}. Change settings and retrain to update results.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest close", f"{frame.Close.iloc[-1]:,.2f}")
c2.metric("Test RMSE", f"{result.scores.loc['LSTM', 'RMSE']:.2f}")
c3.metric("Test strategy return", f"{result.summary['Test strategy return (%)']:.2f}%")
c4.metric("Projected plan value", f"{result.plan.final_value:,.2f}")
history_tab, evaluation_tab, strategy_tab, method_tab = st.tabs([
    "Prices & forecast", "Test evaluation", "A* strategy", "Method & exports"])

with history_tab:
    fig = go.Figure()
    fig.add_scatter(x=frame.index, y=frame.Close, name="History")
    fig.add_scatter(x=result.evaluation.index, y=result.evaluation.LSTM, name="Test one-step predictions")
    fig.add_scatter(x=result.future.index, y=result.future.Forecast, name="Recursive forecast", line={"dash": "dash"})
    fig.update_layout(xaxis_title="Date", yaxis_title="Price (quote currency)", height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Future dates approximate weekdays and may include exchange holidays. Recursive forecast errors compound.")
    st.dataframe(result.future, use_container_width=True)

with evaluation_tab:
    st.subheader("Untouched chronological test set")
    st.dataframe(result.scores, use_container_width=True)
    st.caption("Direction accuracy compares predicted/actual change with the previous close. A persistence baseline predicts zero change.")
    st.line_chart(result.evaluation)
    st.subheader("Causal daily A* simulation")
    st.write("At each test close, A* chooses cash or a one-session round trip using only the model's next-price prediction. Every executed round trip pays an entry and exit fee. The order budget applies per daily plan; fewer than two orders disables trading.")
    st.line_chart(result.backtest[["Equity", "Buy and hold"]])
    st.json({k: v for k, v in result.summary.items() if k.startswith("Test")})

with strategy_tab:
    st.subheader("Plan over forecast prices")
    st.warning("These are projected outcomes on the LSTM forecast, not realized profits or trade instructions.")
    st.write(f"A* expanded {result.plan.expanded} states and selected {result.plan.orders} orders, including final liquidation.")
    st.dataframe(result.plan.ledger, use_container_width=True, hide_index=True)
    st.line_chart(result.plan.ledger.set_index("Step")[["Equity"]])
    st.json({k: v for k, v in result.summary.items() if k.startswith("Projected")})

with method_tab:
    st.write("The scaler sees training data only. Validation controls early stopping; test results never select model weights. Test predictions are rolling one-step predictions with observed past closes; future forecasts feed predictions back into the input. The evaluated model stays frozen after training.")
    st.write("A* states track time, cash/long exposure, and orders used. Its admissible heuristic assumes cost-free switching with perfect knowledge of the forecast path. Search is exact for this finite model. It assumes fractional shares, zero cash interest, no shorting, no slippage, and proportional fees.")
    st.line_chart(pd.DataFrame(result.history).rename(columns={"loss": "Training MSE", "val_loss": "Validation MSE"}))
    for name, table in [("test_predictions", result.evaluation), ("future_forecast", result.future),
                         ("strategy_plan", result.plan.ledger), ("backtest", result.backtest),
                         ("metrics", result.scores)]:
        st.download_button(f"Download {name.replace('_', ' ')}", table.to_csv().encode(),
                           file_name=f"{name}.csv", mime="text/csv")
