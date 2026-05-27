"""Backtest framework for volatility arbitrage strategy."""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from ..models.bsm import bsm_price
from .hedging import Trade, CONTRACT_MULTIPLIER


@dataclass
class TradeRecord:
    code: str
    strike: float
    expiry: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None = None
    entry_iv: float = 0.0
    entry_price: float = 0.0
    entry_etf: float = 0.0
    holding_days: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    total_cost: float = 0.0


@dataclass
class BacktestResult:
    trades: list[TradeRecord]
    equity_curve: pd.Series
    summary: dict


class VolArbBacktest:
    """Backtest engine for vol arbitrage on 50ETF options."""

    def __init__(self, vol_threshold: float | None = None, r: float = 0.04,
                 commission_rate: float = 0.0005, min_t_days: int = 10,
                 close_t_days: int = 5, rolling_threshold: pd.Series | None = None,
                 hedge: bool = True, predicted_vol: float = 0.217,
                 option_type: str = "call"):
        self.vol_threshold = vol_threshold
        self.r = r
        self.commission_rate = commission_rate
        self.min_t_days = min_t_days
        self.close_t_days = close_t_days
        self.rolling_threshold = rolling_threshold
        self.hedge = hedge
        self.predicted_vol = predicted_vol
        self.option_type = option_type  # "call" or "put"

    def run(self, options_daily: pd.DataFrame, etf_daily: pd.DataFrame,
            expiry_date: pd.Timestamp | None = None,
            early_close: bool = False) -> BacktestResult:
        """Run backtest.

        Parameters
        ----------
        options_daily : DataFrame with date, code, close, EXERCISE_PRICE, EXPIRY_DATE, T
        etf_daily : DataFrame with date index and close column
        expiry_date : if provided, only trade this expiry
        early_close : if True, close when cumulative P&L >= expected P&L
        """
        etf = etf_daily[["close"]].copy()
        etf.columns = ["etf_price"]

        # Pre-group options by date for O(1) lookup
        if expiry_date is not None:
            opts_filtered = options_daily[options_daily["EXPIRY_DATE"] == expiry_date]
        else:
            opts_filtered = options_daily
        date_groups = {d: g for d, g in opts_filtered.groupby("date")}

        # Pre-build code lookup for each date group
        date_code_groups = {}
        for d, g in date_groups.items():
            date_code_groups[d] = {row["code"]: row for _, row in g.iterrows()}

        all_dates = sorted(date_groups.keys())
        trade_records = []
        open_trades = []  # List of (Trade, TradeRecord) tuples
        daily_pnl = []

        for date in all_dates:
            if date not in etf.index:
                continue
            etf_price = etf.loc[date, "etf_price"]
            day_codes = date_code_groups.get(date, {})

            # Update existing trades
            closed_indices = []
            for i, (trade, record) in enumerate(open_trades):
                row = day_codes.get(trade.code)
                if row is None:
                    continue
                T = row["T"]
                opt_price = row["close"]

                if T <= 0 or opt_price <= 0:
                    # Expired - close
                    result = trade.close_position(etf_price, opt_price, trade.entry_iv)
                    record.exit_date = date
                    record.holding_days = (date - record.entry_date).days
                    record.pnl = trade.cumulative_pnl
                    record.pnl_pct = trade.cumulative_pnl / (trade.option_price * 10000)
                    record.total_cost = trade.total_cost
                    closed_indices.append(i)
                    continue

                # Use pre-computed IV from data, or compute on the fly
                current_iv = row.get("iv", np.nan)
                if np.isnan(current_iv) or current_iv <= 0:
                    from ..models.implied_vol import implied_vol
                    try:
                        current_iv = implied_vol(opt_price, etf_price, trade.strike,
                                                T, self.r, trade.option_type)
                    except Exception:
                        current_iv = trade.entry_iv
                    if np.isnan(current_iv) or current_iv <= 0:
                        current_iv = trade.entry_iv

                if self.hedge:
                    # Daily rebalance
                    trade.daily_rebalance(etf_price, opt_price, T, current_iv)
                else:
                    # No hedging - only check for early close
                    if not trade.closed and T < trade.close_t_days / 365.25:
                        trade.close_position(etf_price, opt_price, current_iv)
                        record.exit_date = date
                        record.holding_days = (date - record.entry_date).days
                        record.pnl = trade.cumulative_pnl
                        record.pnl_pct = trade.cumulative_pnl / (trade.option_price * 10000)
                        record.total_cost = trade.total_cost
                        closed_indices.append(i)

                # Check early close
                if early_close and trade.cumulative_pnl >= trade.expected_pnl:
                    result = trade.close_position(etf_price, opt_price, current_iv)
                    record.exit_date = date
                    record.holding_days = (date - record.entry_date).days
                    record.pnl = trade.cumulative_pnl
                    record.pnl_pct = trade.cumulative_pnl / (trade.option_price * 10000)
                    record.total_cost = trade.total_cost
                    closed_indices.append(i)

            for i in sorted(closed_indices, reverse=True):
                open_trades.pop(i)

            # Get threshold for this date (rolling or fixed)
            if self.rolling_threshold is not None:
                if date in self.rolling_threshold.index:
                    threshold = self.rolling_threshold.loc[date]
                else:
                    continue  # No threshold available for this date
            else:
                threshold = self.vol_threshold

            # Check for new entries (allow re-entry on different days, prevent same-day duplicate)
            today_codes = set()
            for code, row in day_codes.items():
                T = row["T"]
                K = row["EXERCISE_PRICE"]
                opt_price = row["close"]

                if T <= 0 or opt_price <= 0 or T < self.min_t_days / 365.25:
                    continue
                if code in today_codes:
                    continue
                today_codes.add(code)

                iv = row.get("iv", np.nan)
                if np.isnan(iv) or iv > 1.0:
                    # Compute IV on the fly if not pre-computed
                    from ..models.implied_vol import implied_vol
                    try:
                        iv = implied_vol(opt_price, etf_price, K, T, self.r, self.option_type)
                    except Exception:
                        continue
                    if np.isnan(iv) or iv > 1.0:
                        continue

                if iv > threshold:
                    trade = Trade(
                        code=code,
                        strike=K,
                        expiry=row["EXPIRY_DATE"],
                        entry_date=date,
                        entry_iv=iv,
                        etf_price=etf_price,
                        option_price=opt_price,
                        r=self.r,
                        commission_rate=self.commission_rate,
                        close_t_days=self.close_t_days,
                        predicted_vol=self.predicted_vol,
                        option_type=self.option_type,
                    )
                    # Expected PnL: BSM price at entry_iv - BSM price at predicted_vol
                    T_years = (row["EXPIRY_DATE"] - date).days / 365.25
                    c_entry = bsm_price(etf_price, K, T_years, self.r, iv, self.option_type)
                    c_predicted = bsm_price(etf_price, K, T_years, self.r, self.predicted_vol, self.option_type)
                    trade.expected_pnl = (c_entry - c_predicted) * CONTRACT_MULTIPLIER

                    record = TradeRecord(
                        code=code,
                        strike=K,
                        expiry=row["EXPIRY_DATE"],
                        entry_date=date,
                        entry_iv=iv,
                        entry_price=opt_price,
                        entry_etf=etf_price,
                    )

                    open_trades.append((trade, record))
                    trade_records.append(record)

            # Record daily P&L
            total_pnl = sum(t[0].cumulative_pnl for t in open_trades)
            daily_pnl.append({"date": date, "pnl": total_pnl})

        # Close remaining trades
        for trade, record in open_trades:
            last_date = all_dates[-1]
            if last_date in etf.index:
                etf_price = etf.loc[last_date, "etf_price"]
                last_codes = date_code_groups.get(last_date, {})
                row = last_codes.get(trade.code)
                if row is not None:
                    opt_price = row["close"]
                    trade.close_position(etf_price, opt_price, trade.entry_iv)
                    record.exit_date = last_date
                    record.holding_days = (last_date - record.entry_date).days
                    record.pnl = trade.cumulative_pnl
                    record.pnl_pct = trade.cumulative_pnl / (trade.option_price * 10000)
                    record.total_cost = trade.total_cost

        equity_df = pd.DataFrame(daily_pnl).set_index("date") if daily_pnl else pd.DataFrame()
        equity = equity_df["pnl"] if not equity_df.empty else pd.Series(dtype=float)

        summary = self._compute_summary(trade_records, equity)
        return BacktestResult(trades=trade_records, equity_curve=equity, summary=summary)

    def _compute_summary(self, trades: list[TradeRecord],
                         equity: pd.Series) -> dict:
        if not trades:
            return {"num_trades": 0}

        pnls = [t.pnl for t in trades]
        holding_days = [t.holding_days for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max drawdown from equity curve
        max_dd = 0.0
        if not equity.empty:
            cummax = equity.cummax()
            drawdown = equity - cummax
            max_dd = drawdown.min()

        # Annualized return (simple)
        closed_trades = [t for t in trades if t.exit_date is not None]
        if len(closed_trades) > 1:
            total_days = (closed_trades[-1].exit_date - closed_trades[0].entry_date).days
        else:
            total_days = 30
        total_pnl = sum(pnls)
        total_investment = sum(t.entry_price * 10000 for t in trades)
        total_return = total_pnl / total_investment if total_investment > 0 else 0
        annualized_return = total_return * 365 / total_days if total_days > 0 else 0

        return {
            "num_trades": len(trades),
            "num_wins": len(wins),
            "num_losses": len(losses),
            "win_rate": len(wins) / len(trades),
            "avg_holding_days": np.mean(holding_days),
            "min_holding_days": min(holding_days),
            "max_holding_days": max(holding_days),
            "avg_pnl": np.mean(pnls),
            "avg_pnl_pct": np.mean([t.pnl_pct for t in trades]),
            "total_pnl": total_pnl,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_dd,
        }
