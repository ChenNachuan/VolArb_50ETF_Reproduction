"""Data preprocessing: merge metadata, filter, align."""

from pathlib import Path

import pandas as pd
import numpy as np

from .load_data import get_etf_daily, load_options_5min, load_option_metadata

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "Data"
CACHE_FILE = DATA_DIR / "options_daily.parquet"


def prepare_etf_daily(start_date: str | None = None,
                      end_date: str | None = None) -> pd.DataFrame:
    """Load and prepare 50ETF daily data with log returns."""
    etf = get_etf_daily()
    if start_date:
        etf = etf.loc[start_date:]
    if end_date:
        etf = etf.loc[:end_date]
    etf["log_return"] = np.log(etf["close"] / etf["close"].shift(1))
    return etf


def _build_options_daily_cache():
    """Build and save daily options data from 5min data (one-time)."""
    meta = load_option_metadata()
    opts_5min = load_options_5min()
    opts_5min = opts_5min[opts_5min["code"].isin(meta["code"])].copy()
    opts_5min["date"] = opts_5min["kline_time"].dt.date

    daily_list = []
    for code, group in opts_5min.groupby("code"):
        df = group.copy()
        df = df[df["close"] > 0].copy()
        df["date"] = df["kline_time"].dt.date
        daily = df.groupby("date").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        ).reset_index()
        daily["code"] = code
        daily_list.append(daily)

    daily_all = pd.concat(daily_list).reset_index(drop=True)
    daily_all["date"] = pd.to_datetime(daily_all["date"])
    daily_all = daily_all.merge(
        meta[["code", "EXPIRY_DATE", "EXERCISE_PRICE", "CONTRACT_TYPE"]],
        on="code", how="left",
    )
    daily_all["T"] = (daily_all["EXPIRY_DATE"] - daily_all["date"]).dt.days / 365.25
    daily_all = daily_all[daily_all["T"] > 0]
    daily_all.to_parquet(CACHE_FILE, index=False)
    return daily_all


def prepare_options_daily(start_date: str | None = None,
                         end_date: str | None = None,
                         option_type: str = "C") -> pd.DataFrame:
    """Load options daily data, filtered by type and date range.

    Uses cached parquet if available (fast), otherwise builds from 5min data.
    """
    if CACHE_FILE.exists():
        daily = pd.read_parquet(CACHE_FILE)
    else:
        daily = _build_options_daily_cache()

    # Filter to specified option type
    daily = daily[daily["CONTRACT_TYPE"] == option_type].copy()

    # Filter by date range
    if start_date:
        daily = daily[daily["date"] >= pd.Timestamp(start_date)]
    if end_date:
        daily = daily[daily["date"] <= pd.Timestamp(end_date)]

    return daily.sort_values(["date", "code"]).reset_index(drop=True)


GREEKS_CACHE_FILE = DATA_DIR / "options_greeks.parquet"
GREEKS_5MIN_CACHE_FILE = DATA_DIR / "options_greeks_5min.parquet"


def prepare_options_daily_5min_iv(start_date: str | None = None,
                                  end_date: str | None = None,
                                  option_type: str = "C",
                                  r: float = 0.04,
                                  force_rebuild: bool = False) -> pd.DataFrame:
    """Compute daily IV using 5-minute close prices (more accurate than daily aggregation).

    Uses the last 5-minute bar of each day (14:55) as the closing price.
    """
    if not force_rebuild and GREEKS_5MIN_CACHE_FILE.exists():
        df = pd.read_parquet(GREEKS_5MIN_CACHE_FILE)
        df = df[df["CONTRACT_TYPE"] == option_type].copy()
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values(["date", "code"]).reset_index(drop=True)

    from ..models.implied_vol import implied_vol_vec
    from ..models.greeks import delta_vec, vega_vec

    # Load data
    meta = load_option_metadata()
    opts_5min = load_options_5min()
    opts_5min = opts_5min[opts_5min["code"].isin(meta["code"])].copy()
    opts_5min["date"] = pd.to_datetime(opts_5min["kline_time"].dt.date)

    # Get last bar of each day for each contract (14:55 closing)
    opts_5min["time"] = opts_5min["kline_time"].dt.time
    daily_close = opts_5min.sort_values("kline_time").groupby(["code", "date"]).last().reset_index()

    # Merge metadata
    daily_close = daily_close.merge(
        meta[["code", "EXPIRY_DATE", "EXERCISE_PRICE", "CONTRACT_TYPE"]],
        on="code", how="left",
    )
    daily_close["T"] = (daily_close["EXPIRY_DATE"] - daily_close["date"]).dt.days / 365.25
    daily_close = daily_close[daily_close["T"] > 0].copy()

    # Merge ETF price
    etf = prepare_etf_daily()
    etf_prices = etf[["close"]].rename(columns={"close": "etf_price"})
    daily_close = daily_close.merge(etf_prices, left_on="date", right_index=True, how="left")
    daily_close = daily_close[(daily_close["close"] > 0) & (daily_close["etf_price"] > 0)].copy()

    # Vectorized IV for calls and puts
    for opt_type_label, contract_type in [("call", "C"), ("put", "P")]:
        mask = daily_close["CONTRACT_TYPE"] == contract_type
        sub = daily_close.loc[mask]
        if len(sub) == 0:
            continue
        iv = implied_vol_vec(
            sub["close"].values, sub["etf_price"].values,
            sub["EXERCISE_PRICE"].values, sub["T"].values, r, opt_type_label,
        )
        daily_close.loc[mask, "iv"] = iv
        daily_close.loc[mask, "delta"] = delta_vec(
            sub["etf_price"].values, sub["EXERCISE_PRICE"].values,
            sub["T"].values, r, iv, opt_type_label,
        )
        daily_close.loc[mask, "vega"] = vega_vec(
            sub["etf_price"].values, sub["EXERCISE_PRICE"].values,
            sub["T"].values, r, iv,
        )

    # Save cache
    daily_close.to_parquet(GREEKS_5MIN_CACHE_FILE, index=False)

    # Filter
    result = daily_close[daily_close["CONTRACT_TYPE"] == option_type].copy()
    if start_date:
        result = result[result["date"] >= pd.Timestamp(start_date)]
    if end_date:
        result = result[result["date"] <= pd.Timestamp(end_date)]
    return result.sort_values(["date", "code"]).reset_index(drop=True)


def prepare_options_daily_with_greeks(start_date: str | None = None,
                                      end_date: str | None = None,
                                      option_type: str = "C",
                                      r: float = 0.04,
                                      force_rebuild: bool = False) -> pd.DataFrame:
    """Load options daily data with pre-computed IV, Delta, Vega.

    Uses cached parquet if available. Computes vectorized Greeks on first run.
    """
    if not force_rebuild and GREEKS_CACHE_FILE.exists():
        df = pd.read_parquet(GREEKS_CACHE_FILE)
        df = df[df["CONTRACT_TYPE"] == option_type].copy()
        if start_date:
            df = df[df["date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["date"] <= pd.Timestamp(end_date)]
        return df.sort_values(["date", "code"]).reset_index(drop=True)

    from ..models.implied_vol import implied_vol_vec
    from ..models.greeks import delta_vec, vega_vec

    # Load all options (both C and P)
    all_daily = pd.concat([
        prepare_options_daily(option_type="C"),
        prepare_options_daily(option_type="P"),
    ], ignore_index=True)

    etf = prepare_etf_daily()
    etf_prices = etf[["close"]].rename(columns={"close": "etf_price"})
    merged = all_daily.merge(etf_prices, left_on="date", right_index=True, how="left")
    merged = merged[(merged["T"] > 0) & (merged["close"] > 0) & (merged["etf_price"] > 0)].copy()

    # Vectorized IV for calls and puts separately
    for opt_type_label, contract_type in [("call", "C"), ("put", "P")]:
        mask = merged["CONTRACT_TYPE"] == contract_type
        sub = merged.loc[mask]
        if len(sub) == 0:
            continue
        iv = implied_vol_vec(
            sub["close"].values, sub["etf_price"].values,
            sub["EXERCISE_PRICE"].values, sub["T"].values, r, opt_type_label,
        )
        merged.loc[mask, "iv"] = iv
        merged.loc[mask, "delta"] = delta_vec(
            sub["etf_price"].values, sub["EXERCISE_PRICE"].values,
            sub["T"].values, r, iv, opt_type_label,
        )
        merged.loc[mask, "vega"] = vega_vec(
            sub["etf_price"].values, sub["EXERCISE_PRICE"].values,
            sub["T"].values, r, iv,
        )

    # Save full cache
    merged.to_parquet(GREEKS_CACHE_FILE, index=False)

    # Filter for requested type and date range
    result = merged[merged["CONTRACT_TYPE"] == option_type].copy()
    if start_date:
        result = result[result["date"] >= pd.Timestamp(start_date)]
    if end_date:
        result = result[result["date"] <= pd.Timestamp(end_date)]
    return result.sort_values(["date", "code"]).reset_index(drop=True)
