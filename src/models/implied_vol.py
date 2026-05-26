"""Implied volatility solver using Newton-Raphson and bisection."""

import numpy as np
from scipy.stats import norm

from .bsm import bsm_price, bsm_price_vec


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = "call") -> float:
    """Solve for implied volatility given market price.

    Uses bisection method. Returns NaN if market price is below intrinsic value.
    """
    if T <= 0:
        raise ValueError("Time to expiry must be positive")

    # Check if market price is below intrinsic value
    if option_type == "call":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - 1e-6:
        return np.nan

    return _bisection_iv(market_price, S, K, T, r, option_type)


def implied_vol_vec(market_price, S, K, T, r, option_type="call", max_iter=100):
    """Vectorized IV solver using iterative bisection on arrays.

    All inputs should be numpy arrays of the same length.
    Returns array of IV values (NaN where solver fails).
    """
    market_price = np.asarray(market_price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)

    lo = np.full_like(market_price, 0.001)
    hi = np.full_like(market_price, 5.0)

    # Mask for valid entries
    valid = (T > 0) & (market_price > 0)

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        price = bsm_price_vec(S, K, T, r, mid, option_type)
        diff = price - market_price
        lo = np.where(valid & (diff < 0), mid, lo)
        hi = np.where(valid & (diff > 0), mid, hi)

    sigma = (lo + hi) / 2
    result = np.where(valid, sigma, np.nan)
    return result


def _newton_raphson_iv(market_price: float, S: float, K: float, T: float,
                       r: float, option_type: str, x0: float = 0.3,
                       tol: float = 1e-8, max_iter: int = 100) -> float:
    """Newton-Raphson IV solver."""
    sigma = x0
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        # Vega: dC/dsigma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        if vega < 1e-12:
            break
        sigma -= diff / vega
        sigma = np.clip(sigma, 0.001, 5.0)
    return sigma


def _bisection_iv(market_price: float, S: float, K: float, T: float,
                  r: float, option_type: str,
                  lo: float = 0.001, hi: float = 5.0,
                  tol: float = 1e-8, max_iter: int = 200) -> float:
    """Bisection IV solver (robust fallback)."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = bsm_price(S, K, T, r, mid, option_type)
        diff = price - market_price
        if abs(diff) < tol or (hi - lo) / 2.0 < tol:
            return mid
        # Check which side the market price falls on
        price_lo = bsm_price(S, K, T, r, lo, option_type)
        if (price_lo - market_price) * diff < 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0
