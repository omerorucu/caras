# -*- coding: utf-8 -*-
"""Verification of the continuous-map agreement statistics."""

from __future__ import division

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import regression as reg  # noqa: E402


def test_perfect_agreement():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    r = reg.continuous_agreement(x, x.copy(), bootstrap=0)
    s = r.stats
    assert abs(s["rmse"]) < 1e-12
    assert abs(s["mae"]) < 1e-12
    assert abs(s["bias"]) < 1e-12
    assert abs(s["r2_nash_sutcliffe"] - 1.0) < 1e-12
    assert abs(s["pearson_r2"] - 1.0) < 1e-12
    assert abs(s["lins_ccc"] - 1.0) < 1e-12
    assert abs(s["willmott_d"] - 1.0) < 1e-12
    assert abs(s["ols_slope"] - 1.0) < 1e-9
    assert abs(s["ols_intercept"]) < 1e-9


def test_hand_computed_errors():
    """obs = 1..4, map = obs + (1, -1, 2, -2)

        bias = (1 - 1 + 2 - 2)/4 = 0
        MAE  = (1 + 1 + 2 + 2)/4 = 1.5
        RMSE = sqrt((1 + 1 + 4 + 4)/4) = sqrt(2.5)
    """
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([2.0, 1.0, 5.0, 2.0])
    s = reg.continuous_agreement(obs, pred, bootstrap=0).stats
    assert abs(s["bias"] - 0.0) < 1e-12
    assert abs(s["mae"] - 1.5) < 1e-12
    assert abs(s["rmse"] - math.sqrt(2.5)) < 1e-12
    # SST = sum (obs - 2.5)^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5 ; SSE = 10
    assert abs(s["r2_nash_sutcliffe"] - (1.0 - 10.0 / 5.0)) < 1e-12


def test_constant_offset_separates_r2_from_pearson():
    """A pure +10 offset leaves r^2 at 1 but destroys the determination."""
    obs = np.arange(1.0, 21.0)
    pred = obs + 10.0
    s = reg.continuous_agreement(obs, pred, bootstrap=0).stats
    assert abs(s["pearson_r2"] - 1.0) < 1e-12
    assert s["r2_nash_sutcliffe"] < 0.0
    assert abs(s["bias"] - 10.0) < 1e-12
    assert s["lins_ccc"] < 1.0
    # the systematic share of the error must dominate
    assert s["systematic_share"] > 0.99


def test_slope_test_detects_compression():
    rng = np.random.RandomState(11)
    obs = rng.uniform(0, 100, 400)
    pred = 0.5 * obs + 25.0 + rng.normal(0, 1.0, 400)
    r = reg.continuous_agreement(obs, pred, bootstrap=0)
    assert abs(r.stats["ols_slope"] - 0.5) < 0.02
    assert r.stats["t_slope_vs_1"] < -10
    assert any("slope differs significantly" in w for w in r.warnings)


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    rng = np.random.RandomState(5)
    obs = rng.uniform(0, 50, 150)
    pred = obs + rng.normal(0, 3, 150)
    a = reg.continuous_agreement(obs, pred, bootstrap=300, seed=123)
    b = reg.continuous_agreement(obs, pred, bootstrap=300, seed=123)
    c = reg.continuous_agreement(obs, pred, bootstrap=300, seed=124)
    assert a.bootstrap["rmse"]["ci_lower"] == b.bootstrap["rmse"]["ci_lower"]
    assert a.bootstrap["rmse"]["ci_lower"] != c.bootstrap["rmse"]["ci_lower"]
    ci = a.bootstrap["rmse"]
    assert ci["ci_lower"] <= a.stats["rmse"] <= ci["ci_upper"]
    ci = a.bootstrap["bias"]
    assert ci["ci_lower"] <= a.stats["bias"] <= ci["ci_upper"]


def test_willmott_and_ccc_bounds():
    rng = np.random.RandomState(2)
    obs = rng.uniform(0, 10, 120)
    pred = rng.uniform(0, 10, 120)
    s = reg.continuous_agreement(obs, pred, bootstrap=0).stats
    assert 0.0 <= s["willmott_d"] <= 1.0
    assert -1.0 <= s["willmott_d1"] <= 1.0
    assert -1.0 <= s["lins_ccc"] <= 1.0


def test_nan_pairs_are_dropped():
    obs = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    pred = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    s = reg.continuous_agreement(obs, pred, bootstrap=0).stats
    assert s["n"] == 3


def test_too_few_pairs_raises():
    try:
        reg.continuous_agreement([1.0, 2.0], [1.0, 2.0], bootstrap=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for fewer than three pairs")
