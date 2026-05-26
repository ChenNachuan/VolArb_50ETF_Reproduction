"""Historical and realized volatility computation."""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def historical_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Rolling window historical volatility (annualized).

    Parameters
    ----------
    log_returns : daily log returns
    window : rolling window in trading days
    """
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def realized_volatility_5min(etf_5min: pd.DataFrame, window_days: int) -> pd.Series:
    """Realized volatility from 5min intraday data.

    Uses the close-to-close method on 5min bars, aggregated daily.
    """
    df = etf_5min.copy()
    df["date"] = df["kline_time"].dt.date
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # Daily realized variance: sum of squared intraday returns
    daily_rv = df.groupby("date")["log_ret"].apply(
        lambda x: np.sqrt(np.sum(x.dropna() ** 2))
    )
    # Annualize: multiply by sqrt(trading_days)
    daily_rv_annual = daily_rv * np.sqrt(TRADING_DAYS_PER_YEAR)
    return daily_rv_annual.rolling(window_days).mean()


def realized_volatility_daily(etf_daily: pd.DataFrame, window: int) -> pd.Series:
    """Realized volatility from daily close-to-close returns."""
    log_returns = np.log(etf_daily["close"] / etf_daily["close"].shift(1))
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
