"""Delta hedging engine for volatility arbitrage."""

import numpy as np
import pandas as pd

from ..models.greeks import delta as bsm_delta
from ..models.bsm import bsm_price

CONTRACT_MULTIPLIER = 10000  # 50ETF option contract multiplier


class Trade:
    """Single volatility arbitrage trade: short call + delta hedge."""

    def __init__(self, code: str, strike: float, expiry: pd.Timestamp,
                 entry_date: pd.Timestamp, entry_iv: float,
                 etf_price: float, option_price: float,
                 r: float = 0.04, commission_rate: float = 0.0005,
                 close_t_days: int = 5,
                 predicted_vol: float = 0.217,
                 option_type: str = "call"):
        self.code = code
        self.strike = strike
        self.expiry = expiry
        self.entry_date = entry_date
        self.entry_iv = entry_iv
        self.r = r
        self.commission_rate = commission_rate
        self.close_t_days = close_t_days
        self.predicted_vol = predicted_vol  # Forecast vol for delta hedging
        self.option_type = option_type  # "call" or "put"

        # Initial positions
        self.option_price = option_price  # Short 1 option
        self.etf_price = etf_price
        self.T = (expiry - entry_date).days / 365.25

        # Delta hedge: use predicted vol (not entry_iv) for delta calculation
        self.current_delta = bsm_delta(etf_price, strike, self.T, r, predicted_vol, option_type)
        self.spot_shares = self.current_delta * CONTRACT_MULTIPLIER
        self.total_cost = commission_rate * self.spot_shares * etf_price

        # Daily tracking
        self.daily_pnl = []
        self.cumulative_pnl = 0.0
        self.closed = False

    def daily_rebalance(self, new_etf_price: float, new_option_price: float,
                        new_T: float, current_iv: float) -> dict:
        """Compute daily P&L and rebalance delta.

        P&L model: Market-to-Market option P&L + delta hedge P&L
        - option_pnl: (old_market_price - new_market_price) * CONTRACT_MULTIPLIER
        - spot_pnl: spot_shares * (new_S - old_S)
        Delta hedging uses predicted_vol (not entry_iv).
        """
        # Near-maturity protection: close position when T < MIN_T_DAYS
        if new_T < self.close_t_days / 365.25:
            return self.close_position(new_etf_price, new_option_price, current_iv)

        # Option P&L: actual market price change (short call profits when price drops)
        option_pnl = (self.option_price - new_option_price) * CONTRACT_MULTIPLIER

        # Spot P&L (long Delta shares)
        spot_pnl = self.spot_shares * (new_etf_price - self.etf_price)

        # Financing cost (cost of buying ETF with borrowed money)
        financing_cost = self.r / 252 * self.spot_shares * self.etf_price

        # Transaction cost for rebalancing (use predicted vol for delta)
        new_delta = bsm_delta(new_etf_price, self.strike, new_T, self.r, self.predicted_vol, self.option_type)
        delta_change = new_delta - self.current_delta
        rebalance_cost = self.commission_rate * abs(delta_change) * CONTRACT_MULTIPLIER * new_etf_price

        # Total daily P&L
        daily_pnl = option_pnl + spot_pnl - financing_cost - rebalance_cost
        self.cumulative_pnl += daily_pnl

        # Update state
        self.spot_shares = new_delta * CONTRACT_MULTIPLIER
        self.current_delta = new_delta
        self.option_price = new_option_price
        self.etf_price = new_etf_price
        self.T = new_T
        self.total_cost += rebalance_cost + financing_cost

        result = {
            "option_pnl": option_pnl,
            "spot_pnl": spot_pnl,
            "financing_cost": financing_cost,
            "rebalance_cost": rebalance_cost,
            "daily_pnl": daily_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "current_delta": self.current_delta,
            "total_cost": self.total_cost,
        }
        self.daily_pnl.append(result)
        return result

    def close_position(self, final_etf_price: float, final_option_price: float,
                       final_iv: float) -> dict:
        """Close the position at maturity or early close."""
        if self.closed:
            return {"cumulative_pnl": self.cumulative_pnl}

        # Final option P&L: actual market price change
        option_pnl = (self.option_price - final_option_price) * CONTRACT_MULTIPLIER

        # Final spot P&L
        spot_pnl = self.spot_shares * (final_etf_price - self.etf_price)

        # Close-out cost (sell the ETF shares)
        close_cost = self.commission_rate * self.spot_shares * final_etf_price

        final_pnl = option_pnl + spot_pnl - close_cost
        self.cumulative_pnl += final_pnl
        self.closed = True

        return {
            "option_pnl": option_pnl,
            "spot_pnl": spot_pnl,
            "close_cost": close_cost,
            "final_pnl": final_pnl,
            "cumulative_pnl": self.cumulative_pnl,
            "total_cost": self.total_cost + close_cost,
        }
