# -*- coding: utf-8 -*-
"""
CARAS core :: design-based accuracy and area estimators
=======================================================

Estimators for thematic map validation that follow the accuracy-assessment
"good practice" literature.  All estimators are *design-based*: the sampling
design determines the estimator, and every point estimate is accompanied by
an estimated standard error so that a confidence interval can be reported.

Implemented references
----------------------
Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E. &
    Wulder, M.A. (2014). Good practices for estimating area and assessing
    accuracy of land change. *Remote Sensing of Environment*, 148, 42-57.
    https://doi.org/10.1016/j.rse.2014.02.015
    -> Eqs. 1-13 (stratified estimator, variances, area adjustment,
    sample size).
Olofsson, P., Foody, G.M., Stehman, S.V. & Woodcock, C.E. (2013). Making
    better use of accuracy data in land change studies. *RSE*, 129, 122-131.
Card, D.H. (1982). Using known map category marginal frequencies to improve
    estimates of thematic map accuracy. *PE&RS*, 48, 431-439.
Stehman, S.V. (2014). Estimating area and map accuracy for stratified random
    sampling when the strata are different from the map classes. *IJRS*,
    35, 4923-4939.
Stehman, S.V. & Foody, G.M. (2019). Key issues in rigorous accuracy
    assessment of land cover products. *RSE*, 231, 111199.
Wilson, E.B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22, 209-212.  -> score interval.
Cohen, J. (1960). A coefficient of agreement for nominal scales.
    *Educational and Psychological Measurement*, 20, 37-46.

Matrix convention
-----------------
Throughout this module the confusion / error matrix is stored with

    rows    = MAP classes  (the classification being assessed)
    columns = REFERENCE classes (ground truth)

This is the convention used by Olofsson et al. (2014) and it is the one the
variance formulas below assume.  With this orientation:

    User's accuracy   (UA_i) = correct / row total    = 1 - commission error
    Producer's accur. (PA_j) = correct / column total = 1 - omission error

The reporting layer states the orientation explicitly so that the matrix can
never be read the wrong way round.
"""

from __future__ import division

import math

import numpy as np

__all__ = [
    "Estimate",
    "AccuracyResult",
    "srs_estimate",
    "stratified_estimate",
    "kappa_from_proportions",
    "wilson_interval",
    "sample_size_olofsson",
    "sample_size_for_class",
    "allocate_sample",
    "KAPPA_CAUTION",
    "Z_95",
]

#: Two-sided 95 % normal quantile.
Z_95 = 1.959963984540054
_Z_DEFAULT = Z_95

#: Text reproduced in every report next to Cohen's kappa.
KAPPA_CAUTION = (
    "Cohen's kappa is reported for backward comparability only. Pontius & "
    "Millones (2011, IJRS 32:4407-4429) and Foody (2020, RSE 239:111630) show "
    "that kappa adds no information beyond overall accuracy for map "
    "assessment and that its chance correction is not defensible in this "
    "context. Interpret the quantity / allocation disagreement components "
    "instead. No verbal benchmark scale (e.g. Landis & Koch 1977, which was "
    "devised for clinical inter-rater agreement) is applied here."
)


# ---------------------------------------------------------------------------
# small containers
# ---------------------------------------------------------------------------
class Estimate(object):
    """A point estimate with its estimated standard error.

    ``se`` is ``nan`` when the design does not permit a variance estimate
    (typically a stratum with fewer than two sample units).
    """

    __slots__ = ("value", "se", "n", "successes", "note")

    def __init__(self, value, se=float("nan"), n=None, successes=None, note=None):
        self.value = float(value) if value is not None else float("nan")
        self.se = float(se) if se is not None else float("nan")
        self.n = n
        self.successes = successes
        self.note = note

    def ci(self, z=_Z_DEFAULT, clip=(0.0, 1.0)):
        """Wald confidence interval, optionally clipped to a valid range."""
        if not np.isfinite(self.se):
            return (float("nan"), float("nan"))
        lo = self.value - z * self.se
        hi = self.value + z * self.se
        if clip is not None:
            lo = max(clip[0], lo)
            hi = min(clip[1], hi)
        return (lo, hi)

    def margin(self, z=_Z_DEFAULT):
        return z * self.se if np.isfinite(self.se) else float("nan")

    @property
    def normal_approximation_ok(self):
        """Heuristic guard for the Wald interval (n*p and n*(1-p) >= 5)."""
        if self.n is None or not np.isfinite(self.value):
            return None
        return bool(self.n * self.value >= 5.0 and self.n * (1.0 - self.value) >= 5.0)

    def wilson(self, z=_Z_DEFAULT):
        """Wilson score interval, available when raw counts are known."""
        if self.n is None or self.successes is None:
            return (float("nan"), float("nan"))
        return wilson_interval(self.successes, self.n, z)

    def to_dict(self, z=_Z_DEFAULT):
        lo, hi = self.ci(z)
        wlo, whi = self.wilson(z)
        return {
            "value": self.value,
            "se": self.se,
            "ci_lower": lo,
            "ci_upper": hi,
            "margin_of_error": self.margin(z),
            "wilson_lower": wlo,
            "wilson_upper": whi,
            "n": self.n,
            "normal_approximation_ok": self.normal_approximation_ok,
            "note": self.note,
        }

    def __repr__(self):  # pragma: no cover - debugging aid
        return "Estimate(%.6g +/- %.3g)" % (self.value, self.se)


class AccuracyResult(object):
    """Complete design-based accuracy / area estimate for one comparison."""

    def __init__(self, categories, labels, counts, proportions, design,
                 overall, users, producers, area_proportions,
                 map_proportions, weights=None, area_total=None,
                 pixel_area=None, area_unit=None, warnings=None, f1=None):
        self.categories = list(categories)
        self.labels = list(labels)
        self.counts = np.asarray(counts)
        self.proportions = np.asarray(proportions, dtype=float)
        self.design = design                      # srs | systematic | stratified
        self.overall = overall                    # Estimate
        self.users = list(users)                  # [Estimate]
        self.producers = list(producers)          # [Estimate]
        self.area_proportions = list(area_proportions)   # reference-class share
        self.map_proportions = list(map_proportions)     # uncorrected map share
        self.weights = None if weights is None else np.asarray(weights, float)
        self.area_total = area_total
        self.pixel_area = pixel_area
        self.area_unit = area_unit
        self.warnings = list(warnings or [])
        self.f1 = list(f1 or [])

    @property
    def n(self):
        return int(self.counts.sum())

    @property
    def n_by_map_class(self):
        return self.counts.sum(axis=1).astype(int)

    @property
    def n_by_reference_class(self):
        return self.counts.sum(axis=0).astype(int)

    def commission(self, i):
        return 1.0 - self.users[i].value

    def omission(self, j):
        return 1.0 - self.producers[j].value

    def areas(self, z=_Z_DEFAULT):
        """Bias-adjusted areas (Olofsson Eqs. 9-10) as a list of dicts."""
        out = []
        for j, est in enumerate(self.area_proportions):
            row = {
                "category": self.categories[j],
                "label": self.labels[j],
                "map_proportion": self.map_proportions[j],
                "adjusted_proportion": est.value,
                "proportion_se": est.se,
            }
            lo, hi = est.ci(z)
            row["proportion_ci_lower"] = lo
            row["proportion_ci_upper"] = hi
            if self.area_total is not None:
                row["map_area"] = self.map_proportions[j] * self.area_total
                row["adjusted_area"] = est.value * self.area_total
                row["area_se"] = est.se * self.area_total
                row["area_margin"] = est.margin(z) * self.area_total
                row["area_ci_lower"] = lo * self.area_total
                row["area_ci_upper"] = hi * self.area_total
            out.append(row)
        return out

    def macro_f1(self):
        vals = [e.value for e in self.f1 if e is not None and np.isfinite(e.value)]
        return float(np.mean(vals)) if vals else float("nan")

    def weighted_f1(self):
        """F1 averaged with the estimated reference-class area proportions."""
        num = 0.0
        den = 0.0
        for j, e in enumerate(self.f1):
            w = self.area_proportions[j].value
            if e is not None and np.isfinite(e.value) and np.isfinite(w):
                num += w * e.value
                den += w
        return num / den if den > 0 else float("nan")

    def macro_users(self):
        vals = [e.value for e in self.users if np.isfinite(e.value)]
        return float(np.mean(vals)) if vals else float("nan")

    def macro_producers(self):
        vals = [e.value for e in self.producers if np.isfinite(e.value)]
        return float(np.mean(vals)) if vals else float("nan")

    def kappa(self):
        return kappa_from_proportions(self.proportions)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_ratio(num, den):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    out = np.zeros_like(num, dtype=float)
    ok = np.broadcast_to(den > 0, num.shape)
    denb = np.broadcast_to(den, num.shape)
    out[ok] = num[ok] / denb[ok]
    out[~ok] = np.nan
    return out


def _binomial_se(p, n):
    """SE of a sample proportion with an n-1 denominator (Olofsson Eqs. 5-6)."""
    if n is None or n < 2 or not np.isfinite(p):
        return float("nan")
    var = p * (1.0 - p) / (float(n) - 1.0)
    return math.sqrt(var) if var >= 0 else float("nan")


def wilson_interval(successes, n, z=_Z_DEFAULT):
    """Wilson (1927) score interval - better small-sample coverage than Wald."""
    if n is None or n <= 0:
        return (float("nan"), float("nan"))
    p = float(successes) / float(n)
    d = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / d
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _f1_estimates(ua, pa):
    """Per-class F1 from UA and PA with a delta-method standard error.

    F1 = 2*UA*PA / (UA + PA).  UA and PA come from different marginals of the
    same matrix and are not independent; the SE below assumes independence and
    is therefore flagged as an approximation wherever it is reported.
    """
    out = []
    for u, p in zip(ua, pa):
        if not (np.isfinite(u.value) and np.isfinite(p.value)) or (u.value + p.value) <= 0:
            out.append(Estimate(float("nan"), float("nan"),
                                note="undefined (empty class)"))
            continue
        s = u.value + p.value
        f1 = 2.0 * u.value * p.value / s
        du = 2.0 * p.value * p.value / (s * s)
        dp = 2.0 * u.value * u.value / (s * s)
        if np.isfinite(u.se) and np.isfinite(p.se):
            se = math.sqrt((du * u.se) ** 2 + (dp * p.se) ** 2)
        else:
            se = float("nan")
        out.append(Estimate(f1, se, note="delta-method SE, independence assumed"))
    return out


def kappa_from_proportions(p):
    """Cohen's kappa computed from an estimated area-proportion matrix.

    Using the design-consistent proportion matrix rather than raw counts keeps
    kappa consistent with the other estimates under a stratified design.  See
    :data:`KAPPA_CAUTION` before reporting it.
    """
    p = np.asarray(p, float)
    total = p.sum()
    if total <= 0:
        return float("nan")
    p = p / total
    po = float(np.trace(p))
    pe = float(np.sum(p.sum(axis=1) * p.sum(axis=0)))
    if abs(1.0 - pe) < 1e-15:
        return float("nan")
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# estimator 1 :: equal-probability designs (simple random / systematic)
# ---------------------------------------------------------------------------
def srs_estimate(counts, categories, labels, area_total=None,
                 pixel_area=None, area_unit=None, design="srs"):
    """Estimator for simple random (or systematic) sampling of pixels.

    Every sample unit carries the same inclusion probability, so the sample
    counts are already an unbiased estimator of the population proportions:
    ``p_ij = n_ij / n``.

    Variances
    ---------
    Overall accuracy is a sample proportion, so ``V(OA) = OA(1-OA)/(n-1)``.

    User's and producer's accuracies are *ratio* estimators (Card 1982;
    Stehman 1997).  For binary indicator variables with ``y_u <= x_u`` the
    general ratio-estimator variance

        V(R) = (1/(n xbar^2))[s2_y + R^2 s2_x - 2 R s_xy]

    collapses algebraically to ``R(1-R) * n / (n_x (n-1))``, i.e. to the
    familiar binomial form ``R(1-R)/n_x`` up to the finite-sample factor
    ``n/(n-1)``.  CARAS reports the marginally more conservative
    ``R(1-R)/(n_x - 1)``, which agrees with the ratio estimator to order
    ``1/n^2``.  The algebra is verified numerically in
    ``tests/test_estimators.py``.

    For systematic sampling no design-unbiased variance estimator exists from a
    single sample; treating the sample as simple random is the standard
    conservative practice (Stehman & Foody 2019) and is flagged in the report.
    """
    counts = np.asarray(counts, float)
    k = counts.shape[0]
    n = counts.sum()
    warnings = []
    if n <= 1:
        raise ValueError("At least two sample units are required.")

    p = counts / n
    n_i = counts.sum(axis=1)      # map class totals   (rows)
    n_j = counts.sum(axis=0)      # reference totals   (columns)
    diag = np.diag(counts)

    oa_val = float(np.trace(p))
    overall = Estimate(oa_val, _binomial_se(oa_val, n), n=int(n),
                       successes=int(round(oa_val * n)))

    users, producers = [], []
    for i in range(k):
        if n_i[i] >= 1:
            ua = diag[i] / n_i[i]
            users.append(Estimate(ua, _binomial_se(ua, n_i[i]), n=int(n_i[i]),
                                  successes=int(diag[i])))
        else:
            users.append(Estimate(float("nan"), float("nan"), n=0,
                                  note="no sample unit mapped as this class"))
        if n_j[i] >= 1:
            pa = diag[i] / n_j[i]
            producers.append(Estimate(pa, _binomial_se(pa, n_j[i]), n=int(n_j[i]),
                                      successes=int(diag[i])))
        else:
            producers.append(Estimate(float("nan"), float("nan"), n=0,
                                      note="no reference unit of this class"))

    area_props = []
    for j in range(k):
        pj = n_j[j] / n
        area_props.append(Estimate(pj, _binomial_se(pj, n), n=int(n),
                                   successes=int(n_j[j])))
    map_props = [float(v) for v in (n_i / n)]

    for i in range(k):
        if n_i[i] == 0:
            warnings.append(
                "No sample unit was mapped as class '%s'; its user's accuracy "
                "cannot be estimated." % labels[i])
        elif n_i[i] < 2:
            warnings.append(
                "Map class '%s' has %d sample unit(s); its user's accuracy has "
                "no variance estimate." % (labels[i], int(n_i[i])))
        if n_j[i] == 0:
            warnings.append(
                "No reference unit of class '%s' was drawn; its producer's "
                "accuracy and adjusted area cannot be estimated."
                % labels[i])
        elif n_j[i] < 2:
            warnings.append(
                "Reference class '%s' has %d sample unit(s); its producer's "
                "accuracy has no variance estimate." % (labels[i], int(n_j[i])))
    if design == "systematic":
        warnings.append(
            "Systematic sampling: variances are estimated under a simple random "
            "sampling assumption. This is standard conservative practice but is "
            "only approximate when the landscape carries periodic structure "
            "(Stehman & Foody 2019).")

    return AccuracyResult(
        categories=categories, labels=labels, counts=counts.astype(int),
        proportions=p, design=design, overall=overall, users=users,
        producers=producers, area_proportions=area_props,
        map_proportions=map_props, weights=None, area_total=area_total,
        pixel_area=pixel_area, area_unit=area_unit, warnings=warnings,
        f1=_f1_estimates(users, producers))


# ---------------------------------------------------------------------------
# estimator 2 :: stratified sampling (Olofsson et al. 2014)
# ---------------------------------------------------------------------------
def stratified_estimate(counts, weights, categories, labels,
                        strata_axis="map", area_total=None, pixel_area=None,
                        area_unit=None):
    """Stratified estimator of Olofsson et al. (2014), Eqs. 4-10.

    Parameters
    ----------
    counts : (k, k) array
        Sample counts, rows = map classes, columns = reference classes.
    weights : (k,) array
        Stratum weights ``W_h = N_h / N`` obtained from a full census of the
        stratification raster.  Rescaled internally so that they sum to 1.
    strata_axis : {'map', 'reference'}
        Which raster supplied the strata.  ``'map'`` is the standard case and
        is what the published formulas assume.  ``'reference'`` is handled by
        running the same algebra on the transposed matrix and swapping the
        user's / producer's roles back afterwards; overall accuracy and the
        estimated proportion matrix are invariant under that operation.

    Notes
    -----
    The estimated area proportions are

        p_ij = W_i * n_ij / n_i.                                      (Eq. 4)

    from which overall accuracy, user's and producer's accuracy follow.  The
    reason unweighted sample counts must never be used with a stratified
    design is that the allocation of units across strata is chosen by the
    analyst and in general does *not* reproduce the class proportions of the
    map, so the raw sample over-represents rare classes.
    """
    counts = np.asarray(counts, float)
    weights = np.asarray(weights, float)
    if strata_axis not in ("map", "reference"):
        raise ValueError("strata_axis must be 'map' or 'reference'")

    transposed = strata_axis == "reference"
    work = counts.T.copy() if transposed else counts.copy()

    k = work.shape[0]
    if weights.shape[0] != k:
        raise ValueError("weights and matrix dimensions disagree")
    if weights.sum() <= 0:
        raise ValueError("stratum weights must be positive")
    W = weights / weights.sum()

    n_h = work.sum(axis=1)                 # units drawn per stratum
    warnings = []
    for i in range(k):
        if n_h[i] == 0 and W[i] > 0:
            warnings.append(
                "Stratum '%s' covers %.4f%% of the map but received no sample "
                "unit; its contribution to every estimate is missing and the "
                "results are conditional on that omission."
                % (labels[i], 100.0 * W[i]))

    q = _safe_ratio(work, n_h[:, None])    # q_ij = n_ij / n_i.
    q_filled = np.nan_to_num(q, nan=0.0)
    p = W[:, None] * q_filled              # Eq. 4  (strata in rows)

    oa_val = float(np.trace(p))

    # --- V(OA) : Olofsson Eq. 5 -------------------------------------------
    var_oa = 0.0
    oa_ok = True
    for h in range(k):
        if n_h[h] < 2:
            if W[h] > 0 and n_h[h] > 0:
                oa_ok = False
            continue
        u_h = q_filled[h, h]
        var_oa += (W[h] ** 2) * u_h * (1.0 - u_h) / (n_h[h] - 1.0)
    se_oa = math.sqrt(var_oa) if (oa_ok and var_oa >= 0) else float("nan")
    overall = Estimate(oa_val, se_oa, n=int(work.sum()))

    # --- accuracy of the stratified axis : Eq. 6 --------------------------
    strat_users = []
    for h in range(k):
        if n_h[h] >= 1:
            u_h = q_filled[h, h]
            strat_users.append(Estimate(u_h, _binomial_se(u_h, n_h[h]),
                                        n=int(n_h[h]),
                                        successes=int(work[h, h])))
        else:
            strat_users.append(Estimate(float("nan"), float("nan"), n=0,
                                        note="stratum not sampled"))

    # --- proportions and accuracy of the non-stratified axis --------------
    p_col = p.sum(axis=0)                 # p_.j
    strat_producers = []
    other_props = []
    for j in range(k):
        # V(p_.j) : Olofsson Eq. 10, expressed as a proportion
        var_pj = 0.0
        ok = True
        for h in range(k):
            if n_h[h] < 2:
                if W[h] > 0 and n_h[h] > 0:
                    ok = False
                continue
            qhj = q_filled[h, j]
            var_pj += (W[h] ** 2) * qhj * (1.0 - qhj) / (n_h[h] - 1.0)
        se_pj = math.sqrt(var_pj) if (ok and var_pj >= 0) else float("nan")
        other_props.append(Estimate(p_col[j], se_pj, n=int(work.sum())))

        # producer's accuracy of column class j : Eq. 7
        if p_col[j] <= 0:
            strat_producers.append(Estimate(float("nan"), float("nan"), n=0,
                                            note="class absent from the estimate"))
            continue
        pa = p[j, j] / p_col[j]
        var_pa = 0.0
        ok = True
        if n_h[j] >= 2:
            u_j = q_filled[j, j]
            var_pa += (W[j] ** 2) * ((1.0 - pa) ** 2) * u_j * (1.0 - u_j) / (n_h[j] - 1.0)
        elif W[j] > 0 and n_h[j] > 0:
            ok = False
        for h in range(k):
            if h == j:
                continue
            if n_h[h] < 2:
                if W[h] > 0 and n_h[h] > 0:
                    ok = False
                continue
            qhj = q_filled[h, j]
            var_pa += (pa ** 2) * (W[h] ** 2) * qhj * (1.0 - qhj) / (n_h[h] - 1.0)
        var_pa = var_pa / (p_col[j] ** 2)
        se_pa = math.sqrt(var_pa) if (ok and var_pa >= 0) else float("nan")
        strat_producers.append(Estimate(pa, se_pa, n=int(work[:, j].sum())))

    # --- map results back to (map rows, reference columns) ----------------
    if transposed:
        proportions = p.T
        users = strat_producers            # producer's of the transposed run
        producers = strat_users
        # reference classes were the strata, so their shares are a census
        area_props = [Estimate(float(W[j]), 0.0, n=int(work.sum()),
                               note="census of the stratification raster")
                      for j in range(k)]
        map_props = [e.value for e in other_props]
    else:
        proportions = p
        users = strat_users
        producers = strat_producers
        area_props = other_props           # reference-class area shares
        map_props = [float(v) for v in W]

    for h in range(k):
        if 0 < n_h[h] < 2:
            warnings.append(
                "Stratum '%s' contains a single sample unit; no variance can be "
                "estimated from it and the affected standard errors are "
                "reported as undefined." % labels[h])

    return AccuracyResult(
        categories=categories, labels=labels, counts=counts.astype(int),
        proportions=proportions, design="stratified", overall=overall,
        users=users, producers=producers, area_proportions=area_props,
        map_proportions=map_props, weights=W, area_total=area_total,
        pixel_area=pixel_area, area_unit=area_unit, warnings=warnings,
        f1=_f1_estimates(users, producers))


# ---------------------------------------------------------------------------
# sample size planning
# ---------------------------------------------------------------------------
def sample_size_olofsson(weights, target_se_oa, expected_ua=None):
    """Total sample size for a target standard error of overall accuracy.

    Olofsson et al. (2014), Eq. 13::

        n = ( sum_i W_i * S_i / S(O) )^2 ,      S_i = sqrt(U_i (1 - U_i))

    ``expected_ua`` is the analyst's prior guess of each class's user's
    accuracy.  When omitted a conservative 0.5 is used for every class
    (maximum variance), which yields the largest - safest - sample size.
    """
    W = np.asarray(weights, float)
    if W.sum() <= 0:
        raise ValueError("weights must be positive")
    W = W / W.sum()
    if expected_ua is None:
        U = np.full(W.shape, 0.5)
    else:
        U = np.clip(np.asarray(expected_ua, float), 0.0, 1.0)
    S = np.sqrt(U * (1.0 - U))
    if target_se_oa <= 0:
        raise ValueError("target standard error must be positive")
    n = (float(np.sum(W * S)) / float(target_se_oa)) ** 2
    return int(math.ceil(n))


def sample_size_for_class(target_se, expected_accuracy=0.7):
    """Units needed in one class for a target SE of its UA or PA (Cochran)."""
    p = min(max(float(expected_accuracy), 1e-6), 1.0 - 1e-6)
    if target_se <= 0:
        raise ValueError("target standard error must be positive")
    return int(math.ceil(p * (1.0 - p) / (float(target_se) ** 2)))


def allocate_sample(weights, n_total, scheme="proportional", minimum=0,
                    expected_ua=None):
    """Distribute ``n_total`` sample units across strata.

    scheme
    ------
    ``'equal'``
        n/k in every stratum.  Maximises the precision of rare-class user's
        accuracy but is the least efficient choice for overall accuracy.
    ``'proportional'``
        n_h proportional to W_h.  Self-weighting: the sample then behaves like
        a simple random sample.
    ``'neyman'``
        n_h proportional to ``W_h * S_h`` with ``S_h = sqrt(U_h (1-U_h))``;
        optimal for the precision of overall accuracy (Cochran 1977).
    ``'olofsson'``
        Proportional allocation with a floor for rare classes - the compromise
        recommended in Olofsson et al. (2014), section 5.1.2.

    Rounding uses the largest-remainder method so the requested total is met
    exactly whenever the floor permits it.
    """
    W = np.asarray(weights, float)
    if W.sum() <= 0:
        raise ValueError("weights must be positive")
    W = W / W.sum()
    k = W.size
    n_total = int(n_total)
    if n_total < k:
        raise ValueError("total sample size must be at least the number of strata")

    if scheme == "equal":
        raw = np.full(k, n_total / float(k))
    elif scheme in ("proportional", "olofsson"):
        raw = W * n_total
    elif scheme == "neyman":
        U = np.full(k, 0.5) if expected_ua is None else np.clip(
            np.asarray(expected_ua, float), 0.0, 1.0)
        s = W * np.sqrt(U * (1.0 - U))
        raw = (s / s.sum()) * n_total if s.sum() > 0 else W * n_total
    else:
        raise ValueError("unknown allocation scheme: %s" % scheme)

    floor_ = max(int(minimum), 1)
    if scheme == "olofsson" and minimum <= 0:
        floor_ = max(1, min(20, n_total // max(k, 1)))
    floor_ = min(floor_, max(1, n_total // k))

    alloc = np.maximum(np.floor(raw).astype(int), floor_)
    remainder = n_total - int(alloc.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for idx in order[:remainder]:
            alloc[idx] += 1
        if remainder > order.size:
            alloc[int(np.argmax(W))] += remainder - order.size
    elif remainder < 0:
        deficit = -remainder
        order = list(np.argsort(-alloc))
        i = 0
        while deficit > 0 and i < 100 * k:
            idx = order[i % k]
            if alloc[idx] > floor_:
                alloc[idx] -= 1
                deficit -= 1
            i += 1
    return alloc.astype(int)
