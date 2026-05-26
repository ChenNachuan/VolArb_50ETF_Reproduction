"""Tests for BSM pricing and IV solver."""

import numpy as np
import pytest

from src.models.bsm import bsm_price
from src.models.implied_vol import implied_vol


class TestBSMPrice:
    def test_at_the_money_call(self):
        # ATM call with known analytical value approximation
        price = bsm_price(S=2.50, K=2.50, T=30 / 365, r=0.04, sigma=0.30,
                          option_type="call")
        assert 0.02 < price < 0.10

    def test_deep_itm_call(self):
        price = bsm_price(S=3.0, K=2.0, T=30 / 365, r=0.04, sigma=0.30,
                          option_type="call")
        assert price > 0.95  # Should be close to intrinsic

    def test_deep_otm_call(self):
        price = bsm_price(S=2.0, K=3.0, T=30 / 365, r=0.04, sigma=0.30,
                          option_type="call")
        assert price < 0.01

    def test_put_call_parity(self):
        S, K, T, r, sigma = 2.50, 2.45, 30 / 365, 0.04, 0.30
        call = bsm_price(S, K, T, r, sigma, "call")
        put = bsm_price(S, K, T, r, sigma, "put")
        # C - P = S - K * exp(-r*T)
        assert abs((call - put) - (S - K * np.exp(-r * T))) < 1e-10

    def test_zero_vol(self):
        # With sigma=0, call payoff is max(S-K, 0) discounted
        S, K, T, r = 2.50, 2.40, 30 / 365, 0.04
        price = bsm_price(S, K, T, r, sigma=0, option_type="call")
        expected = max(S - K * np.exp(-r * T), 0.0)
        assert abs(price - expected) < 1e-10

    def test_zero_time(self):
        # At expiry, option value = intrinsic
        S, K, r, sigma = 2.50, 2.40, 0.04, 0.30
        price = bsm_price(S, K, T=0, r=r, sigma=sigma, option_type="call")
        assert abs(price - max(S - K, 0.0)) < 1e-10


class TestImpliedVol:
    def test_roundtrip_call(self):
        # Price an option, then recover IV
        S, K, T, r, sigma_true = 2.50, 2.45, 30 / 365, 0.04, 0.30
        market_price = bsm_price(S, K, T, r, sigma_true, "call")
        iv = implied_vol(market_price, S, K, T, r, "call")
        assert abs(iv - sigma_true) < 1e-6

    def test_roundtrip_put(self):
        S, K, T, r, sigma_true = 2.50, 2.45, 30 / 365, 0.04, 0.25
        market_price = bsm_price(S, K, T, r, sigma_true, "put")
        iv = implied_vol(market_price, S, K, T, r, "put")
        assert abs(iv - sigma_true) < 1e-6

    def test_high_vol(self):
        S, K, T, r, sigma_true = 2.50, 2.45, 30 / 365, 0.04, 1.0
        market_price = bsm_price(S, K, T, r, sigma_true, "call")
        iv = implied_vol(market_price, S, K, T, r, "call")
        assert abs(iv - sigma_true) < 1e-4

    def test_low_vol(self):
        S, K, T, r, sigma_true = 2.50, 2.45, 30 / 365, 0.04, 0.05
        market_price = bsm_price(S, K, T, r, sigma_true, "call")
        iv = implied_vol(market_price, S, K, T, r, "call")
        assert abs(iv - sigma_true) < 1e-4
