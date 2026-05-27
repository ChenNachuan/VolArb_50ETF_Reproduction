"""Data loading and 5min-to-daily resampling."""

from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"


def load_etf_5min() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "50ETF_5min.parquet")


def load_options_5min() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "50ETF_options_5min.parquet")


def load_option_metadata() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "option_metadata.parquet")


def resample_to_daily(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Resample 5min OHLCV to daily using last bar of each trading day."""
    df = df_5min.copy()
    # Filter out zero-price bars (first bar of some days has open=0, close=0)
    df = df[df["close"] > 0].copy()
    df["date"] = df["kline_time"].dt.date

    daily = df.groupby("date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    return daily


def get_etf_daily(date_range: tuple[str, str] | None = None) -> pd.DataFrame:
    """Get 50ETF daily OHLCV, optionally filtered by date range.

    Uses pre-computed daily data (2010-2025) from combined 5min sources.
    Falls back to resampling from 5min if parquet not available.
    """
    daily_path = DATA_DIR / "etf_daily_510050.parquet"
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
    else:
        etf_5min = load_etf_5min()
        daily = resample_to_daily(etf_5min)
    if date_range:
        daily = daily.loc[date_range[0]:date_range[1]]
    return daily


def get_option_daily(code: str, date_range: tuple[str, str] | None = None) -> pd.DataFrame:
    """Get daily OHLCV for a single option contract."""
    opts = load_options_5min()
    contract = opts[opts["code"] == code].copy()
    if contract.empty:
        return pd.DataFrame()
    daily = resample_to_daily(contract)
    if date_range:
        daily = daily.loc[date_range[0]:date_range[1]]
    return daily
