"""Entry/exit signal generation for volatility arbitrage."""

import pandas as pd
import numpy as np

from src.models.implied_vol import implied_vol


def compute_iv_series(options_daily: pd.DataFrame, etf_daily: pd.DataFrame,
                      r: float = 0.04) -> pd.DataFrame:
    """Compute implied volatility for each option on each day.

    Returns DataFrame with columns: date, code, iv, strike, expiry, T, option_price, etf_price
    """
    # Merge ETF price
    etf_prices = etf_daily[["close"]].rename(columns={"close": "etf_price"})
    merged = options_daily.merge(etf_prices, left_on="date", right_index=True, how="left")

    # Filter valid rows
    merged = merged[(merged["T"] > 0) & (merged["close"] > 0) & (merged["etf_price"] > 0)].copy()

    # Compute IV for each row
    ivs = []
    for _, row in merged.iterrows():
        try:
            iv = implied_vol(
                market_price=row["close"],
                S=row["etf_price"],
                K=row["EXERCISE_PRICE"],
                T=row["T"],
                r=r,
                option_type="call",
            )
            ivs.append(iv)
        except Exception:
            ivs.append(np.nan)

    merged["iv"] = ivs
    return merged[["date", "code", "iv", "EXPIRY_DATE",
                    "EXERCISE_PRICE", "T", "close", "etf_price"]].copy()


def generate_entry_signals(iv_df: pd.DataFrame,
                           vol_threshold: float,
                           expiry_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """Generate entry signals when IV > threshold.

    Parameters
    ----------
    iv_df : DataFrame with date, code, iv columns
    vol_threshold : annualized vol threshold (e.g., 0.2834)
    expiry_date : if provided, only trade options expiring on this date
    """
    df = iv_df.copy()
    if expiry_date is not None:
        df = df[df["EXPIRY_DATE"] == expiry_date]

    # Entry signal: IV > threshold
    df["signal"] = df["iv"] > vol_threshold
    return df
