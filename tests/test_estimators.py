# -*- coding: utf-8 -*-
"""
Verification of the design-based estimators.

The checks are of three kinds:

1. **Analytic identities** that must hold exactly for any input - the
   stratified estimator collapsing to the equal-probability estimator when the
   weights match the allocation, proportions summing to one, and so on.
2. **An independent re-implementation** of the published variance formulas,
   written out literally from Olofsson et al. (2014) inside the test, compared
   against the vectorised implementation in the module.
3. **A fully hand-worked example** whose intermediate values are written into
   the test as literal numbers, so that a reader can verify the arithmetic on
   paper.

Run without pytest:  ``python tests/run_tests.py``
"""

from __future__ import division

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import estimators as est  # noqa: E402

CATS = [1, 2]
LABS = ["Forest", "Non-forest"]

# hand-worked stratified example ------------------------------------------
#   rows = map class, columns = reference class
#   n_1. = n_2. = 100 ; weights from a census of the map
HAND_COUNTS = np.array([[80.0, 20.0],
                        [10.0, 90.0]])
HAND_W = np.array([0.2, 0.8])
HAND_AREA = 1000.0          # hectares


# ---------------------------------------------------------------------------
def test_srs_basic():
    r = est.srs_estimate(HAND_COUNTS, CATS, LABS)
    # OA = (80 + 90) / 200
    assert abs(r.overall.value - 0.85) < 1e-12
    # UA of map class 1 = 80/100 ; PA of reference class 1 = 80/90
    assert abs(r.users[0].value - 0.80) < 1e-12
    assert abs(r.producers[0].value - 80.0 / 90.0) < 1e-12
    # V(OA) = OA(1-OA)/(n-1)
    assert abs(r.overall.se - math.sqrt(0.85 * 0.15 / 199.0)) < 1e-15
    # area proportions must sum to one
    assert abs(sum(e.value for e in r.area_proportions) - 1.0) < 1e-12


def test_srs_ratio_estimator_identity():
    """The binomial SE of UA is the ratio-estimator SE, not a shortcut.

    For binary indicators y_u <= x_u the general ratio-estimator variance

        V(R) = (1/(n xbar^2))[s2_y + R^2 s2_x - 2 R s_xy]

    reduces algebraically to R(1-R) * n / (n_x (n-1)).  The general expression
    is evaluated here on explicit indicator vectors and compared with that
    closed form; the value CARAS reports, R(1-R)/(n_x - 1), is then checked to
    be marginally conservative and to agree to order 1/n^2.
    """
    n_total, n_x, n_y = 200, 100, 80          # x: mapped as class 1; y: also correct
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    x[:n_x] = 1.0
    y[:n_y] = 1.0
    R = y.sum() / x.sum()

    xbar, ybar = x.mean(), y.mean()
    s2y = np.sum((y - ybar) ** 2) / (n_total - 1)
    s2x = np.sum((x - xbar) ** 2) / (n_total - 1)
    sxy = np.sum((y - ybar) * (x - xbar)) / (n_total - 1)
    v_general = (1.0 / xbar ** 2) * (1.0 / n_total) * (
        s2y + R * R * s2x - 2.0 * R * sxy)

    v_closed = R * (1.0 - R) * n_total / (n_x * (n_total - 1.0))
    assert abs(v_general - v_closed) < 1e-15, (v_general, v_closed)

    # what CARAS actually reports for this cell
    counts = np.array([[80.0, 20.0], [10.0, 90.0]])
    r = est.srs_estimate(counts, CATS, LABS)
    v_caras = r.users[0].se ** 2
    assert v_caras >= v_general - 1e-15                    # conservative
    assert abs(v_caras - v_general) < 1e-4                 # agrees to O(1/n^2)


def test_stratified_reduces_to_srs_point_estimates():
    """With W_i = n_i./n the stratified estimator must reproduce the SRS one.

    Point estimates coincide exactly.  The *variance* estimators legitimately
    differ - one conditions on the stratum sizes, the other does not - so only
    the point estimates are compared.
    """
    counts = np.array([[97.0, 3.0, 2.0],
                       [5.0, 279.0, 15.0],
                       [8.0, 1.0, 141.0]])
    cats, labs = [1, 2, 3], ["a", "b", "c"]
    srs = est.srs_estimate(counts, cats, labs)
    W = counts.sum(axis=1) / counts.sum()
    strat = est.stratified_estimate(counts, W, cats, labs)

    assert abs(strat.overall.value - srs.overall.value) < 1e-12
    for a, b in zip(strat.users, srs.users):
        assert abs(a.value - b.value) < 1e-12
    for a, b in zip(strat.producers, srs.producers):
        assert abs(a.value - b.value) < 1e-12
    for a, b in zip(strat.area_proportions, srs.area_proportions):
        assert abs(a.value - b.value) < 1e-12
    assert np.allclose(strat.proportions, srs.proportions, atol=1e-12)


def test_stratified_hand_worked_example():
    """Every number below is derived by hand in the module docstring.

        q   = [[0.8, 0.2], [0.1, 0.9]]
        p   = W_i * q_ij = [[0.16, 0.04], [0.08, 0.72]]
        OA  = 0.16 + 0.72 = 0.88
        V(OA) = 0.2^2*0.8*0.2/99 + 0.8^2*0.9*0.1/99 = 0.064/99
        p_.1 = 0.24 ; PA_1 = 0.16/0.24 = 2/3
    """
    r = est.stratified_estimate(HAND_COUNTS, HAND_W, CATS, LABS,
                                area_total=HAND_AREA, area_unit="ha")

    assert np.allclose(r.proportions, [[0.16, 0.04], [0.08, 0.72]], atol=1e-15)
    assert abs(r.overall.value - 0.88) < 1e-15
    assert abs(r.overall.se - math.sqrt(0.064 / 99.0)) < 1e-15

    assert abs(r.users[0].value - 0.8) < 1e-15
    assert abs(r.users[0].se - math.sqrt(0.8 * 0.2 / 99.0)) < 1e-15
    assert abs(r.users[1].value - 0.9) < 1e-15

    assert abs(r.producers[0].value - 2.0 / 3.0) < 1e-12
    assert abs(r.producers[1].value - 0.72 / 0.76) < 1e-12

    # Olofsson Eq. 7 written out literally for reference class 1
    pa = 2.0 / 3.0
    v = ((0.2 ** 2) * (1 - pa) ** 2 * 0.8 * 0.2 / 99.0
         + (pa ** 2) * (0.8 ** 2) * 0.1 * 0.9 / 99.0) / (0.24 ** 2)
    assert abs(r.producers[0].se - math.sqrt(v)) < 1e-14

    # area adjustment
    areas = r.areas()
    assert abs(areas[0]["adjusted_proportion"] - 0.24) < 1e-15
    assert abs(areas[0]["map_proportion"] - 0.2) < 1e-15
    assert abs(areas[0]["adjusted_area"] - 240.0) < 1e-12
    assert abs(areas[0]["map_area"] - 200.0) < 1e-12
    assert abs(sum(a["adjusted_area"] for a in areas) - HAND_AREA) < 1e-9
    # SE of p_.1 : Eq. 10
    v_p = ((0.2 ** 2) * 0.8 * 0.2 / 99.0 + (0.8 ** 2) * 0.1 * 0.9 / 99.0)
    assert abs(areas[0]["proportion_se"] - math.sqrt(v_p)) < 1e-15


def test_stratified_differs_from_unweighted():
    """The whole point of the fix: ignoring the weights changes the answer."""
    srs = est.srs_estimate(HAND_COUNTS, CATS, LABS)
    strat = est.stratified_estimate(HAND_COUNTS, HAND_W, CATS, LABS)
    assert abs(strat.overall.value - srs.overall.value) > 0.02
    # and the rare class is over-represented in the raw sample
    assert srs.area_proportions[0].value > strat.area_proportions[0].value


def test_stratified_transposed_axis():
    """Stratifying on the reference map swaps the roles of UA and PA."""
    counts = np.array([[70.0, 30.0], [20.0, 80.0]])
    W = np.array([0.35, 0.65])
    a = est.stratified_estimate(counts, W, CATS, LABS, strata_axis="reference")
    b = est.stratified_estimate(counts.T, W, CATS, LABS, strata_axis="map")
    assert abs(a.overall.value - b.overall.value) < 1e-14
    for x, y in zip(a.users, b.producers):
        assert abs(x.value - y.value) < 1e-14
    for x, y in zip(a.producers, b.users):
        assert abs(x.value - y.value) < 1e-14


def test_perfect_and_degenerate():
    perfect = np.array([[50.0, 0.0], [0.0, 50.0]])
    r = est.srs_estimate(perfect, CATS, LABS)
    assert abs(r.overall.value - 1.0) < 1e-15
    assert abs(r.overall.se) < 1e-15
    assert abs(r.kappa() - 1.0) < 1e-12

    # a class with no reference unit must not raise, only report nan
    thin = np.array([[10.0, 0.0], [3.0, 0.0]])
    r2 = est.srs_estimate(thin, CATS, LABS)
    assert not np.isfinite(r2.producers[1].value)
    assert any("No reference unit of class" in w for w in r2.warnings)


def test_kappa_hand_value():
    """2x2 example: OA = 0.85, Pe = 0.5*0.45 + 0.5*0.55 = 0.5, kappa = 0.7."""
    counts = np.array([[40.0, 10.0], [5.0, 45.0]])
    r = est.srs_estimate(counts, CATS, LABS)
    p = counts / counts.sum()
    po = float(np.trace(p))
    pe = float(np.sum(p.sum(1) * p.sum(0)))
    assert abs(r.kappa() - (po - pe) / (1 - pe)) < 1e-14
    assert abs(po - 0.85) < 1e-15


def test_wilson_interval():
    lo, hi = est.wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 0.20          # never degenerate at 0/n
    lo, hi = est.wilson_interval(10, 20)
    assert abs((lo + hi) / 2.0 - 0.5) < 1e-12     # symmetric at p = 0.5
    lo, hi = est.wilson_interval(19, 20)
    assert hi < 1.0


def test_sample_size_round_trip():
    """The size returned for a target SE must actually achieve it."""
    W = np.array([0.05, 0.15, 0.80])
    U = np.array([0.70, 0.80, 0.95])
    target = 0.01
    n = est.sample_size_olofsson(W, target, U)
    # achieved SE under proportional allocation, Olofsson Eq. 5
    alloc = np.maximum(np.floor(W * n).astype(int), 2)
    var = np.sum((W ** 2) * U * (1 - U) / (alloc - 1))
    assert math.sqrt(var) <= target * 1.15, (math.sqrt(var), target)
    # a stricter target must ask for more units
    assert est.sample_size_olofsson(W, target / 2.0, U) > n


def test_sample_size_for_class():
    n = est.sample_size_for_class(0.05, 0.8)
    assert abs(n - math.ceil(0.8 * 0.2 / 0.05 ** 2)) < 1e-9


def test_allocation_schemes():
    W = np.array([0.02, 0.08, 0.90])
    for scheme in ("equal", "proportional", "neyman", "olofsson"):
        alloc = est.allocate_sample(W, 600, scheme=scheme, minimum=30)
        assert alloc.sum() == 600, (scheme, alloc)
        assert alloc.min() >= 30, (scheme, alloc)
    prop = est.allocate_sample(W, 600, "proportional", minimum=0)
    assert prop.sum() == 600
    assert prop[2] > prop[1] > prop[0]
    eq = est.allocate_sample(W, 600, "equal", minimum=0)
    assert max(eq) - min(eq) <= 1


def test_estimate_container():
    e = est.Estimate(0.9, 0.02, n=100, successes=90)
    lo, hi = e.ci()
    assert lo < 0.9 < hi
    assert abs(e.margin() - 1.959963984540054 * 0.02) < 1e-15
    assert e.normal_approximation_ok is True
    small = est.Estimate(0.02, 0.01, n=50, successes=1)
    assert small.normal_approximation_ok is False
    wlo, whi = small.wilson()
    assert 0.0 <= wlo < whi <= 1.0
    d = e.to_dict()
    assert set(["value", "se", "ci_lower", "ci_upper", "n"]).issubset(d)
