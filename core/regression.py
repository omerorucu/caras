# -*- coding: utf-8 -*-
"""
CARAS core :: agreement statistics for continuous maps
======================================================

Validation statistics for *continuous* (interval / ratio scale) rasters -
biomass, canopy cover, LST, NDVI, elevation, model output, and so on.

Design rule enforced by CARAS
-----------------------------
These statistics are computed **only** when the analyst declares the data
continuous.  Applying RMSE, R^2, MAE or bias to nominal class codes produces
numbers that depend entirely on the arbitrary integer labels assigned to the
categories, and CARAS v1.0 did exactly that.  Version 2 refuses to.

What is reported and why
------------------------
``r2_nash_sutcliffe``
    ``1 - SSE/SST`` - the coefficient of determination in the
    prediction-validation sense (identical to the Nash-Sutcliffe efficiency,
    Nash & Sutcliffe 1970).  It can be negative.
``pearson_r`` / ``pearson_r2``
    Correlation of the two variables and its square.  This is *not* the same
    quantity as above and the two are routinely confused in the literature;
    ``pearson_r2`` ignores any bias or scaling error, so a badly biased map
    can still score 0.99.  Both are printed side by side deliberately.
``rmse`` / ``mae`` / ``bias``
    Standard error magnitudes.  ``bias`` (mean error) is
    ``mean(map - reference)``: positive means the map overestimates.
``rmse_systematic`` / ``rmse_unsystematic``
    Willmott's (1981) decomposition using the ordinary least squares fit of
    map on reference.  A large systematic share indicates a calibration
    problem that a linear correction could remove; a large unsystematic share
    indicates irreducible noise.
``willmott_d`` / ``willmott_d1``
    Index of agreement (Willmott 1981) and its refined version
    (Willmott et al. 2012), which is less dominated by large outliers.
``lins_ccc``
    Lin's (1989) concordance correlation coefficient - agreement with the 1:1
    line, combining precision and accuracy in one number.
``ols_slope`` / ``ols_intercept``
    Regression of map on reference with standard errors and t tests against
    the ideal values (slope = 1, intercept = 0).  A slope significantly below
    1 is the classic signature of regression-to-the-mean in model output.

Confidence intervals are obtained by a seeded non-parametric bootstrap
(Efron & Tibshirani 1993) so that they are reproducible.

References
----------
Willmott, C.J. (1981). On the validation of models. *Physical Geography*,
    2, 184-194.
Willmott, C.J., Robeson, S.M. & Matsuura, K. (2012). A refined index of model
    performance. *International Journal of Climatology*, 32, 2088-2094.
Lin, L.I. (1989). A concordance correlation coefficient to evaluate
    reproducibility. *Biometrics*, 45, 255-268.
Nash, J.E. & Sutcliffe, J.V. (1970). River flow forecasting through
    conceptual models. *Journal of Hydrology*, 10, 282-290.
Ji, L. & Gallo, K. (2006). An agreement coefficient for image comparison.
    *PE&RS*, 72, 823-833.
Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*.
"""

from __future__ import division

import math

import numpy as np

__all__ = ["ContinuousResult", "continuous_agreement"]

_Z_95 = 1.959963984540054


class ContinuousResult(object):
    """Container for continuous-map agreement statistics."""

    def __init__(self, stats, bootstrap=None, warnings=None):
        self.stats = dict(stats)
        self.bootstrap = dict(bootstrap or {})
        self.warnings = list(warnings or [])

    def __getitem__(self, key):
        return self.stats[key]

    def get(self, key, default=None):
        return self.stats.get(key, default)

    def to_dict(self):
        return {
            "statistics": self.stats,
            "bootstrap_ci": self.bootstrap,
            "warnings": self.warnings,
        }


def _ols(x, y):
    """Ordinary least squares fit y = a + b x with standard errors."""
    n = x.size
    if n < 3:
        return {}
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0:
        return {}
    b = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    a = float(y.mean() - b * x.mean())
    resid = y - (a + b * x)
    dof = n - 2
    s2 = float(np.sum(resid ** 2) / dof) if dof > 0 else float("nan")
    se_b = math.sqrt(s2 / sxx) if np.isfinite(s2) and s2 >= 0 else float("nan")
    se_a = (math.sqrt(s2 * (1.0 / n + x.mean() ** 2 / sxx))
            if np.isfinite(s2) and s2 >= 0 else float("nan"))
    out = {
        "ols_slope": b,
        "ols_intercept": a,
        "ols_slope_se": se_b,
        "ols_intercept_se": se_a,
        "ols_residual_sd": math.sqrt(s2) if np.isfinite(s2) and s2 >= 0 else float("nan"),
        "ols_dof": dof,
    }
    if np.isfinite(se_b) and se_b > 0:
        out["t_slope_vs_1"] = (b - 1.0) / se_b
    if np.isfinite(se_a) and se_a > 0:
        out["t_intercept_vs_0"] = a / se_a
    return out


def _willmott(obs, pred):
    """Index of agreement d (1981) and refined d1 (2012)."""
    obs_mean = obs.mean()
    denom = np.sum((np.abs(pred - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    d = 1.0 - np.sum((pred - obs) ** 2) / denom if denom > 0 else float("nan")

    num1 = float(np.sum(np.abs(pred - obs)))
    den1 = 2.0 * float(np.sum(np.abs(obs - obs_mean)))
    if den1 <= 0:
        d1 = float("nan")
    elif num1 <= den1:
        d1 = 1.0 - num1 / den1
    else:
        d1 = den1 / num1 - 1.0
    return float(d), float(d1)


def _lins_ccc(obs, pred):
    n = obs.size
    if n < 2:
        return float("nan")
    mo, mp = obs.mean(), pred.mean()
    vo = float(np.sum((obs - mo) ** 2) / n)
    vp = float(np.sum((pred - mp) ** 2) / n)
    cov = float(np.sum((obs - mo) * (pred - mp)) / n)
    den = vo + vp + (mo - mp) ** 2
    return 2.0 * cov / den if den > 0 else float("nan")


def _core_stats(obs, pred):
    """All point statistics for one (possibly resampled) pair of vectors."""
    n = obs.size
    err = pred - obs
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))

    out = {
        "n": int(n),
        "reference_mean": float(obs.mean()),
        "map_mean": float(pred.mean()),
        "reference_sd": float(obs.std(ddof=1)) if n > 1 else float("nan"),
        "map_sd": float(pred.std(ddof=1)) if n > 1 else float("nan"),
        "bias": float(err.mean()),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(sse / n)),
        "r2_nash_sutcliffe": float(1.0 - sse / sst) if sst > 0 else float("nan"),
    }

    if n > 1 and obs.std() > 0 and pred.std() > 0:
        r = float(np.corrcoef(obs, pred)[0, 1])
    else:
        r = float("nan")
    out["pearson_r"] = r
    out["pearson_r2"] = r * r if np.isfinite(r) else float("nan")

    # normalised errors
    rng = float(obs.max() - obs.min())
    out["nrmse_range"] = out["rmse"] / rng if rng > 0 else float("nan")
    out["nrmse_mean"] = (out["rmse"] / abs(out["reference_mean"])
                         if abs(out["reference_mean"]) > 0 else float("nan"))
    if np.all(np.abs(obs) > 0):
        out["mape_percent"] = float(np.mean(np.abs(err / obs)) * 100.0)
    else:
        out["mape_percent"] = float("nan")

    d, d1 = _willmott(obs, pred)
    out["willmott_d"] = d
    out["willmott_d1"] = d1
    out["lins_ccc"] = _lins_ccc(obs, pred)

    ols = _ols(obs, pred)
    out.update(ols)

    # Willmott (1981) RMSE decomposition using the OLS fit
    if "ols_slope" in ols:
        fitted = ols["ols_intercept"] + ols["ols_slope"] * obs
        mse_s = float(np.mean((fitted - obs) ** 2))
        mse_u = float(np.mean((pred - fitted) ** 2))
        out["rmse_systematic"] = math.sqrt(mse_s)
        out["rmse_unsystematic"] = math.sqrt(mse_u)
        total = mse_s + mse_u
        out["systematic_share"] = mse_s / total if total > 0 else float("nan")
    return out


def continuous_agreement(reference, mapped, bootstrap=2000, seed=None,
                         z=_Z_95):
    """Full agreement assessment for two continuous variables.

    Parameters
    ----------
    reference, mapped : 1-D arrays of equal length
        Paired values at the validation sample locations.  ``reference`` is
        the ground-truth / observed variable, ``mapped`` the value being
        assessed.
    bootstrap : int
        Number of bootstrap resamples for confidence intervals; 0 disables.
    seed : int or None
        Seed for the bootstrap, stored in the report so the interval can be
        reproduced exactly.
    """
    obs = np.asarray(reference, dtype=float).ravel()
    pred = np.asarray(mapped, dtype=float).ravel()
    if obs.size != pred.size:
        raise ValueError("reference and map vectors must have equal length")
    good = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[good], pred[good]
    n = obs.size
    warnings = []
    if n < 3:
        raise ValueError("at least three valid pairs are required")
    if n < 30:
        warnings.append(
            "Only %d valid pairs: bootstrap intervals are wide and the "
            "normality of the OLS t tests is not guaranteed." % n)

    stats = _core_stats(obs, pred)

    # ------------------------------------------------------------------
    # seeded non-parametric bootstrap
    # ------------------------------------------------------------------
    ci = {}
    if bootstrap and bootstrap >= 100:
        rng = np.random.RandomState(seed if seed is not None else 0)
        keys = ["rmse", "mae", "bias", "r2_nash_sutcliffe", "pearson_r",
                "lins_ccc", "willmott_d", "ols_slope", "ols_intercept"]
        draws = dict((k, []) for k in keys)
        idx_all = np.arange(n)
        for _ in range(int(bootstrap)):
            idx = rng.choice(idx_all, size=n, replace=True)
            try:
                s = _core_stats(obs[idx], pred[idx])
            except Exception:
                continue
            for k in keys:
                v = s.get(k, float("nan"))
                if np.isfinite(v):
                    draws[k].append(v)
        for k in keys:
            arr = np.asarray(draws[k], dtype=float)
            if arr.size >= 20:
                lo, hi = np.percentile(arr, [2.5, 97.5])
                ci[k] = {
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "bootstrap_se": float(arr.std(ddof=1)),
                    "replicates": int(arr.size),
                }
        ci["_meta"] = {
            "method": "non-parametric percentile bootstrap",
            "replicates_requested": int(bootstrap),
            "seed": seed,
            "level": 0.95,
        }
    elif bootstrap:
        warnings.append(
            "Bootstrap skipped: at least 100 replicates are required for a "
            "usable percentile interval.")

    # interpretation aids -------------------------------------------------
    if np.isfinite(stats.get("pearson_r2", float("nan"))) and \
            np.isfinite(stats.get("r2_nash_sutcliffe", float("nan"))):
        gap = stats["pearson_r2"] - stats["r2_nash_sutcliffe"]
        if gap > 0.05:
            warnings.append(
                "Squared Pearson r exceeds the coefficient of determination by "
                "%.3f. The map is well correlated with the reference but "
                "systematically offset or mis-scaled; report the coefficient "
                "of determination, not r-squared." % gap)
    if "t_slope_vs_1" in stats and abs(stats["t_slope_vs_1"]) > 1.96:
        warnings.append(
            "OLS slope differs significantly from 1 (t = %.2f). The map "
            "compresses or expands the range of the reference variable."
            % stats["t_slope_vs_1"])
    if "t_intercept_vs_0" in stats and abs(stats["t_intercept_vs_0"]) > 1.96:
        warnings.append(
            "OLS intercept differs significantly from 0 (t = %.2f); an "
            "additive offset is present." % stats["t_intercept_vs_0"])
    if np.isfinite(stats.get("systematic_share", float("nan"))) and \
            stats["systematic_share"] > 0.5:
        warnings.append(
            "%.0f%% of the mean squared error is systematic and could in "
            "principle be removed by a linear recalibration."
            % (100.0 * stats["systematic_share"]))

    return ContinuousResult(stats, ci, warnings)
