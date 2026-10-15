"""Validated prices from Yahoo Finance, CSV, or an explicitly synthetic demo."""
import numpy as np
import pandas as pd


def clean_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if "Date" in frame.columns:
        frame.index = pd.to_datetime(frame.pop("Date"), errors="coerce", utc=True)
    else:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("CSV data must contain a Date column.")
        frame.index = pd.to_datetime(frame.index, errors="coerce", utc=True)
    if "Close" not in frame:
        raise ValueError("Data must contain Date and Close columns.")
    close = pd.to_numeric(frame["Close"], errors="coerce")
    frame = pd.DataFrame({"Close": close}, index=frame.index)
    frame = frame.loc[~frame.index.isna()].dropna()
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index()
    if len(frame) < 100:
        raise ValueError("Provide at least 100 valid daily closing prices.")
    if not np.isfinite(frame.Close).all() or (frame.Close <= 0).any():
        raise ValueError("Closing prices must be finite and positive.")
    frame.index = frame.index.tz_convert(None)
    frame.index.name = "Date"
    return frame


def demo_prices(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = 0.0004 + 0.003 * np.sin(np.arange(rows) / 18) + rng.normal(0, 0.009, rows)
    return clean_prices(pd.DataFrame({"Close": 100 * np.exp(np.cumsum(returns))},
                                    index=pd.bdate_range("2023-01-02", periods=rows)))


def download_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    ticker = ticker.strip().upper()
    if not ticker or len(ticker) > 30:
        raise ValueError("Enter a valid ticker, for example AAPL or RELIANCE.NS.")
    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError("Start date must precede end date (end is exclusive).")
    frame = yf.download(ticker, start=start, end=end, auto_adjust=True,
                        progress=False, threads=False, multi_level_index=False, timeout=20)
    if frame is None or frame.empty:
        raise ValueError("Yahoo Finance returned no prices. Check the ticker/dates or use CSV/demo.")
    return clean_prices(frame)
