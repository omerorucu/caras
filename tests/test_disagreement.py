# -*- coding: utf-8 -*-
"""Verification of the quantity / allocation / exchange / shift decomposition.

The components are defined so that they must add up exactly; the tests below
check those identities and then confirm the intended behaviour on three
constructed matrices whose disagreement is, by design, purely of one kind.
"""

from __future__ import division

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import disagreement as dis  # noqa: E402


def _check_identities(d):
    assert abs(d.quantity + d.allocation - d.total) < 1e-12
    assert abs(d.exchange + d.shift - d.allocation) < 1e-12
    assert d.exchange <= d.allocation + 1e-12
    assert d.quantity >= -1e-12 and d.allocation >= -1e-12
    assert abs(d.allocation_from_category_sum - d.allocation) < 1e-9
    assert d.to_dict()["identity_check"]["closes"] is True


def test_perfect_agreement():
    p = np.array([[0.4, 0.0], [0.0, 0.6]])
    d = dis.disagreement(p)
    assert abs(d.total) < 1e-15
    assert abs(d.quantity) < 1e-15
    assert abs(d.allocation) < 1e-15
    _check_identities(d)


def test_pure_quantity_disagreement():
    """All errors run one way, so the class totals differ and nothing swaps.

        map class 1 total = 0.5, reference class 1 total = 0.3
        Q = (|0.5-0.3| + |0.5-0.7|)/2 = 0.2 ; and A must be 0.
    """
    p = np.array([[0.3, 0.2],
                  [0.0, 0.5]])
    d = dis.disagreement(p)
    assert abs(d.total - 0.2) < 1e-12
    assert abs(d.quantity - 0.2) < 1e-12
    assert abs(d.allocation) < 1e-12
    _check_identities(d)


def test_pure_allocation_by_exchange():
    """Symmetric off-diagonal cells: the amounts match, only the places differ.

        row sums = column sums = (0.5, 0.5) so Q = 0
        D = 0.1 + 0.1 = 0.2, entirely exchange.
    """
    p = np.array([[0.4, 0.1],
                  [0.1, 0.4]])
    d = dis.disagreement(p)
    assert abs(d.quantity) < 1e-12
    assert abs(d.allocation - 0.2) < 1e-12
    assert abs(d.exchange - 0.2) < 1e-12
    assert abs(d.shift) < 1e-12
    _check_identities(d)


def test_shift_component():
    """A three-class cycle produces allocation disagreement with no exchange.

    Class 1 is mapped where 2 belongs, 2 where 3 belongs, 3 where 1 belongs.
    No pair swaps reciprocally, so exchange is zero and the whole allocation
    disagreement is shift.
    """
    p = np.array([[0.2, 0.1, 0.0],
                  [0.0, 0.2, 0.1],
                  [0.1, 0.0, 0.3]])
    d = dis.disagreement(p)
    assert abs(d.quantity) < 1e-12
    assert abs(d.exchange) < 1e-12
    assert abs(d.shift - d.allocation) < 1e-12
    assert d.allocation > 0
    _check_identities(d)


def test_matches_one_minus_overall_accuracy():
    rng = np.random.RandomState(3)
    for _ in range(50):
        p = rng.random_sample((4, 4))
        p = p / p.sum()
        d = dis.disagreement(p)
        oa = float(np.trace(p))
        assert abs(d.total - (1.0 - oa)) < 1e-12
        _check_identities(d)


def test_unnormalised_input_is_rescaled():
    counts = np.array([[80.0, 20.0], [10.0, 90.0]])
    d1 = dis.disagreement(counts)
    d2 = dis.disagreement(counts / counts.sum())
    assert abs(d1.total - d2.total) < 1e-14
    assert abs(d1.quantity - d2.quantity) < 1e-14


def test_per_category_rows():
    p = np.array([[0.3, 0.2], [0.0, 0.5]])
    d = dis.disagreement(p, [1, 2], ["water", "land"])
    assert [r["label"] for r in d.per_category] == ["water", "land"]
    assert abs(d.per_category[0]["map_proportion"] - 0.5) < 1e-15
    assert abs(d.per_category[0]["reference_proportion"] - 0.3) < 1e-15
