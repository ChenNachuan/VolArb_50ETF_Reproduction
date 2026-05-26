"""Greeks computation using BSM model."""

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def _d1_d2_vec(S, K, T, r, sigma):
    """Vectorized d1/d2 computation."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def delta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def delta_vec(S, K, T, r, sigma, option_type="call"):
    """Vectorized delta."""
    d1, _ = _d1_d2_vec(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega per 1% change in volatility."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * np.sqrt(T) * norm.pdf(d1) / 100.0


def vega_vec(S, K, T, r, sigma):
    """Vectorized vega per 1% change in volatility."""
    d1, _ = _d1_d2_vec(S, K, T, r, sigma)
    return np.asarray(S) * np.sqrt(np.asarray(T)) * norm.pdf(d1) / 100.0


def theta(S: float, K: float, T: float, r: float, sigma: float,
          option_type: str = "call") -> float:
    """Theta per calendar day."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
        return (term1 + term2) / 365.0
    else:
        term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365.0
