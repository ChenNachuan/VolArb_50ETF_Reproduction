"""Tests for Greeks computation."""

import numpy as np
import pytest

from src.models.bsm import bsm_price
from src.models.greeks import delta, gamma, vega, theta


S, K, T, r, sigma = 2.50, 2.45, 30 / 365, 0.04, 0.30


class TestDelta:
    def test_call_delta_range(self):
        d = delta(S, K, T, r, sigma, "call")
        assert 0 < d < 1

    def test_put_delta_range(self):
        d = delta(S, K, T, r, sigma, "put")
        assert -1 < d < 0

    def test_put_call_delta_sum(self):
        d_call = delta(S, K, T, r, sigma, "call")
        d_put = delta(S, K, T, r, sigma, "put")
        assert abs((d_call - d_put) - 1.0) < 1e-10

    def test_finite_difference(self):
        ds = 0.001
        price_up = bsm_price(S + ds, K, T, r, sigma, "call")
        price_dn = bsm_price(S - ds, K, T, r, sigma, "call")
        fd_delta = (price_up - price_dn) / (2 * ds)
        assert abs(delta(S, K, T, r, sigma, "call") - fd_delta) < 1e-4


class TestGamma:
    def test_gamma_positive(self):
        g = gamma(S, K, T, r, sigma)
        assert g > 0

    def test_finite_difference(self):
        ds = 0.001
        delta_up = delta(S + ds, K, T, r, sigma, "call")
        delta_dn = delta(S - ds, K, T, r, sigma, "call")
        fd_gamma = (delta_up - delta_dn) / (2 * ds)
        assert abs(gamma(S, K, T, r, sigma) - fd_gamma) < 1e-3


class TestVega:
    def test_vega_positive(self):
        v = vega(S, K, T, r, sigma)
        assert v > 0

    def test_finite_difference(self):
        ds = 0.001
        price_up = bsm_price(S, K, T, r, sigma + ds, "call")
        price_dn = bsm_price(S, K, T, r, sigma - ds, "call")
        fd_vega = (price_up - price_dn) / (2 * ds) / 100.0
        assert abs(vega(S, K, T, r, sigma) - fd_vega) < 1e-4
