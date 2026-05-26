"""Volatility cone construction."""

import numpy as np
import pandas as pd

from .volatility import historical_volatility

DEFAULT_WINDOWS = [5, 10, 20, 40, 60, 120]
DEFAULT_PERCENTILES = [10, 25, 50, 75, 85, 90]


def build_vol_cone(etf_daily: pd.DataFrame,
                   windows: list[int] | None = None,
                   percentiles: list[int] | None = None) -> pd.DataFrame:
    """Build volatility cone from ETF daily data.

    Returns DataFrame with windows as rows, percentiles as columns.
    Each value is the annualized volatility at that percentile for that window.
    """
    if windows is None:
        windows = DEFAULT_WINDOWS
    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    log_returns = np.log(etf_daily["close"] / etf_daily["close"].shift(1))

    cone = {}
    for w in windows:
        hv = historical_volatility(log_returns, w).dropna()
        cone[w] = {p: np.percentile(hv, p) for p in percentiles}

    return pd.DataFrame(cone, index=percentiles).T


def get_vol_threshold(cone: pd.DataFrame, window: int = 20,
                      percentile: int = 85) -> float:
    """Get the volatility threshold from the cone."""
    return cone.loc[window, percentile]


def compute_rolling_threshold(etf_daily: pd.DataFrame,
                              lookback_years: int = 2,
                              window: int = 20,
                              percentile: int = 85) -> pd.Series:
    """Compute rolling volatility threshold.

    For each date, uses the past lookback_years of data to compute
    the HV percentile, returning a time-varying threshold.

    Parameters
    ----------
    etf_daily : DataFrame with date index and close column
    lookback_years : years of historical data to use
    window : rolling window for HV calculation
    percentile : percentile threshold (e.g., 85)

    Returns
    -------
    Series with date index and threshold values
    """
    log_returns = np.log(etf_daily["close"] / etf_daily["close"].shift(1))
    lookback_days = int(lookback_years * 252)

    thresholds = {}
    dates = etf_daily.index

    for i, date in enumerate(dates):
        if i < lookback_days:
            continue
        # Use past lookback_days of returns
        recent_returns = log_returns.iloc[i - lookback_days:i]
        hv = recent_returns.rolling(window).std() * np.sqrt(252)
        hv = hv.dropna()
        if len(hv) > 0:
            thresholds[date] = np.percentile(hv, percentile)

    return pd.Series(thresholds)
