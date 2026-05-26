"""Black-Scholes-Merton option pricing model."""

import numpy as np
from scipy.stats import norm


def bsm_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call") -> float:
    """Compute BSM option price.

    Parameters
    ----------
    S : underlying price
    K : strike price
    T : time to expiry in years
    r : risk-free rate (annualized)
    sigma : volatility (annualized)
    option_type : 'call' or 'put'
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * T), 0.0)
        else:
            return max(K * np.exp(-r * T) - S, 0.0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option_type: {option_type}")


def bsm_price_vec(S, K, T, r, sigma, option_type="call"):
    """Vectorized BSM pricing. All inputs are numpy arrays."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
