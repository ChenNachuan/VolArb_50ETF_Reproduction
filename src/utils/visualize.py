"""Visualization utilities for volatility arbitrage analysis."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import pathlib

_setup_done = False

_FONT_SANS_SERIF = [
    "PingFang HK", "Heiti TC", "STHeiti", "Lantinghei SC", "Songti SC", "Arial Unicode MS", "DejaVu Sans"
]


def setup_chinese_font():
    """Configure matplotlib for Chinese text rendering on macOS.

    Clears the font cache and rebuilds the font manager so that macOS
    system fonts (PingFang HK, Heiti TC, etc.) are properly discovered.
    Idempotent — safe to call multiple times.
    """
    global _setup_done
    if _setup_done:
        return

    # Clear all font-list caches (version-agnostic glob)
    cache_dir = pathlib.Path(matplotlib.get_cachedir())
    for f in cache_dir.glob("fontlist*.json"):
        f.unlink(missing_ok=True)

    # Rebuild the font manager from scratch
    fm._load_fontmanager(try_read_cache=False)

    plt.rcParams["font.sans-serif"] = _FONT_SANS_SERIF
    plt.rcParams["axes.unicode_minus"] = False

    _setup_done = True


def plot_vol_cone(cone: pd.DataFrame, title: str = "50ETF Volatility Cone"):
    """Plot volatility cone chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    windows = cone.index
    for col in cone.columns:
        ax.plot(windows, cone[col] * 100, marker="o", label=f"{col}%")
    ax.set_xlabel("Window (trading days)")
    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_iv_timeseries(iv_df: pd.DataFrame, vol_threshold: float,
                       title: str = "Call Option IV vs Threshold"):
    """Plot IV time series for each call with threshold line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for code in iv_df["code"].unique():
        data = iv_df[iv_df["code"] == code]
        label = f'K={data["EXERCISE_PRICE"].iloc[0]:.2f}'
        ax.plot(data["date"], data["iv"] * 100, label=label, alpha=0.7)
    ax.axhline(y=vol_threshold * 100, color="red", linestyle="--",
               linewidth=2, label=f"Threshold ({vol_threshold*100:.1f}%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Implied Volatility (%)")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_trade_distribution(trades: list, title: str = "Trade Entry Distribution"):
    """Plot bar chart of trades by entry date."""
    if not trades:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return fig

    dates = [t.entry_date for t in trades]
    date_counts = pd.Series(dates).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(date_counts.index, date_counts.values, color="steelblue", alpha=0.8)
    ax.set_xlabel("Entry Date")
    ax.set_ylabel("Number of Trades")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m/%d"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    return fig


def plot_equity_curve(equity: pd.Series, title: str = "Strategy Equity Curve"):
    """Plot NAV/equity curve."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity.index, equity.values, linewidth=1.5, color="steelblue")
    ax.fill_between(equity.index, equity.values, alpha=0.2, color="steelblue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative P&L")
    ax.set_title(title)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_comparison(results: dict, title: str = "Hold-to-Maturity vs Early Close"):
    """Plot comparison of two backtest results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Equity curves
    for label, result in results.items():
        if not result.equity_curve.empty:
            axes[0].plot(result.equity_curve.index, result.equity_curve.values,
                        label=label, linewidth=1.5)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Cumulative P&L")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Summary stats bar chart
    metrics = ["win_rate", "avg_holding_days", "annualized_return"]
    x = np.arange(len(metrics))
    width = 0.35
    for i, (label, result) in enumerate(results.items()):
        values = [result.summary.get(m, 0) * 100 if "rate" in m or "return" in m
                  else result.summary.get(m, 0) for m in metrics]
        axes[1].bar(x + i * width, values, width, label=label, alpha=0.8)
    axes[1].set_title("Strategy Comparison")
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(["Win Rate (%)", "Avg Holding Days", "Annualized Return (%)"])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig
