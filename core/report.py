# -*- coding: utf-8 -*-
"""
CARAS core :: reporting
=======================

Turns an :class:`caras.core.analysis.AnalysisResult` into the four artefacts a
paper actually needs:

``text_report``
    A complete, self-contained plain-text report - every estimate with its
    confidence interval, both error matrices with their marginals, the area
    adjustment, the disagreement components, the limitations, and a
    reproducibility block that names the seed and the software versions.
``json_report``
    The same content as machine-readable provenance, so a reviewer can
    recompute anything without re-running QGIS.
``html_report``
    A printable version of the text report.  All user-supplied strings pass
    through ``html.escape`` - version 1 injected class names straight into the
    markup.
``matrix_csv``
    The sample-count matrix and the estimated area-proportion matrix, ready
    for a supplementary table.

A short *methods paragraph* is generated as well: a citable description of
exactly what was done, with the references filled in, that can be pasted into a
manuscript and edited.
"""

from __future__ import division

import json
import math
from html import escape as _esc

import numpy as np

from .estimators import KAPPA_CAUTION

__all__ = ["text_report", "json_report", "html_report", "matrix_csv",
           "methods_paragraph", "REFERENCES"]


REFERENCES = [
    ("Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E. & "
     "Wulder, M.A. (2014). Good practices for estimating area and assessing "
     "accuracy of land change. Remote Sensing of Environment, 148, 42-57. "
     "https://doi.org/10.1016/j.rse.2014.02.015"),
    ("Stehman, S.V. & Foody, G.M. (2019). Key issues in rigorous accuracy "
     "assessment of land cover products. Remote Sensing of Environment, 231, "
     "111199. https://doi.org/10.1016/j.rse.2019.05.018"),
    ("Stehman, S.V. (2009). Sampling designs for accuracy assessment of land "
     "cover. International Journal of Remote Sensing, 30, 5243-5272."),
    ("Card, D.H. (1982). Using known map category marginal frequencies to "
     "improve estimates of thematic map accuracy. Photogrammetric Engineering "
     "and Remote Sensing, 48, 431-439."),
    ("Pontius, R.G. Jr. & Millones, M. (2011). Death to Kappa: birth of "
     "quantity disagreement and allocation disagreement for accuracy "
     "assessment. International Journal of Remote Sensing, 32, 4407-4429."),
    ("Pontius, R.G. Jr. & Santacruz, A. (2014). Quantity, exchange and shift "
     "components of difference in a square contingency table. International "
     "Journal of Remote Sensing, 35, 7543-7554."),
    ("Foody, G.M. (2020). Explaining the unsuitability of the kappa "
     "coefficient in the assessment and comparison of the accuracy of "
     "thematic maps. Remote Sensing of Environment, 239, 111630."),
    ("Congalton, R.G. & Green, K. (2019). Assessing the Accuracy of Remotely "
     "Sensed Data: Principles and Practices, 3rd ed. CRC Press."),
    ("Willmott, C.J. (1981). On the validation of models. Physical Geography, "
     "2, 184-194."),
    ("Lin, L.I. (1989). A concordance correlation coefficient to evaluate "
     "reproducibility. Biometrics, 45, 255-268."),
]

_SEP = "=" * 78
_SUB = "-" * 78


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------
def _f(x, nd=4):
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "n/a"
    return ("%." + str(nd) + "f") % x


def _pct(x, nd=2):
    if x is None or not np.isfinite(float(x)):
        return "n/a"
    return ("%." + str(nd) + "f") % (100.0 * float(x)) + " %"


def _ci(est, z, nd=4):
    lo, hi = est.ci(z)
    if not np.isfinite(lo):
        return "  (no variance estimate)"
    return "[%s, %s]" % (_f(lo, nd), _f(hi, nd))


def _ci_best(est, z, nd=4):
    """Interval to print, preferring Wilson where the Wald one is unusable.

    A class whose sample units are all correct gives UA = 1 with a Wald
    standard error of exactly zero, i.e. the interval [1, 1] - a claim of
    perfect certainty from as few as five units.  Whenever the normal
    approximation fails (n*p < 5 or n*(1-p) < 5) the Wilson score interval is
    printed instead and flagged with an asterisk.
    """
    if est.normal_approximation_ok is False:
        wlo, whi = est.wilson(z)
        if np.isfinite(wlo):
            return "[%s, %s]*" % (_f(wlo, nd), _f(whi, nd)), True
    return _ci(est, z, nd), False


#: Footnote emitted whenever a Wilson interval was substituted.
WILSON_NOTE = (
    "* Wilson score interval: the normal approximation is unreliable for this "
    "class (fewer than five expected successes or failures), and the Wald "
    "interval would understate the uncertainty - at an accuracy of exactly 1.0 "
    "it collapses to zero width.")


def _estimate_line(name, est, z, width=30):
    body = "%s  +/- %s   %s" % (_f(est.value), _f(est.margin(z)), _ci(est, z))
    extra = ""
    if est.normal_approximation_ok is False:
        wlo, whi = est.wilson(z)
        if np.isfinite(wlo):
            extra = "   Wilson: [%s, %s]" % (_f(wlo), _f(whi))
        else:
            extra = "   (normal approximation unreliable)"
    return "%-*s : %s%s" % (width, name, body, extra)


def _matrix_block(matrix, labels, row_title, col_title, fmt="%d",
                  colwidth=13, with_marginals=True):
    """Error matrix with row/column totals, printed as fixed-width text."""
    k = len(labels)
    short = [str(l)[:colwidth - 2] for l in labels]
    lines = []
    head = "%-24s" % ("%s \\ %s" % (row_title[:10], col_title[:10]))
    for s in short:
        head += "%*s" % (colwidth, s)
    if with_marginals:
        head += "%*s" % (colwidth, "TOTAL")
    lines.append(head)
    lines.append(_SUB)
    for i in range(k):
        row = "%-24s" % short[i]
        for j in range(k):
            row += "%*s" % (colwidth, fmt % matrix[i][j])
        if with_marginals:
            row += "%*s" % (colwidth, fmt % sum(matrix[i][j] for j in range(k)))
        lines.append(row)
    if with_marginals:
        lines.append(_SUB)
        row = "%-24s" % "TOTAL"
        grand = 0
        for j in range(k):
            s = sum(matrix[i][j] for i in range(k))
            grand += s
            row += "%*s" % (colwidth, fmt % s)
        row += "%*s" % (colwidth, fmt % grand)
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# methods paragraph
# ---------------------------------------------------------------------------
def methods_paragraph(result):
    """A citable description of the run, for a manuscript methods section."""
    cfg = result.config
    acc = result.accuracy
    parts = []
    ref = result.reference_info
    cls = result.classified_info

    design_names = {
        "random": "simple random sampling of pixels",
        "systematic": "systematic sampling with a random start",
        "stratified": "stratified random sampling",
        "points": "an externally supplied set of validation points",
    }
    design = design_names.get(cfg.method, cfg.method)

    if cfg.method == "stratified":
        strat = ("the classified map" if cfg.strata_source == "map"
                 else "the reference map")
        alloc = {"proportional": "proportional", "equal": "equal",
                 "neyman": "Neyman (optimal)",
                 "olofsson": "proportional with a floor for rare classes"}
        parts.append(
            "Validation units were selected by %s, using the classes of %s as "
            "strata and a %s allocation of %d units."
            % (design, strat, alloc.get(cfg.allocation, cfg.allocation),
               cfg.n_points))
        parts.append(
            "Stratum weights were obtained from a complete census of the "
            "stratification raster, and accuracy, area proportions and their "
            "standard errors were estimated with the design-based stratified "
            "estimator of Olofsson et al. (2014, Eqs. 4-10).")
    else:
        parts.append(
            "Validation units were selected by %s (%d units requested, %d "
            "realised on pixels carrying valid data in both maps)."
            % (design, cfg.n_points, len(result.sample)))
        parts.append(
            "Accuracy and area proportions were estimated with the "
            "equal-probability design-based estimator, standard errors "
            "following Card (1982) and Stehman (2009).")

    parts.append(
        "Sampling used pseudo-random number seed %s, so the sample is exactly "
        "reproducible." % (cfg.seed if cfg.seed is not None else "unset"))

    if acc is not None:
        parts.append(
            "Overall accuracy was %s (%.0f%% CI %s), and class areas were "
            "adjusted for classification error following Olofsson et al. "
            "(2014, Eqs. 9-10)."
            % (_pct(acc.overall.value), 100 * cfg.confidence_level,
               _ci(acc.overall, cfg.z, 3)))
        parts.append(
            "Disagreement was decomposed into quantity and allocation "
            "components (Pontius & Millones 2011), the latter further split "
            "into exchange and shift (Pontius & Santacruz 2014).")
    if result.continuous is not None:
        s = result.continuous.stats
        parts.append(
            "Agreement between the continuous maps was summarised by the "
            "coefficient of determination (%s), RMSE (%s) and mean error "
            "(%s), with 95%% confidence intervals from a seeded "
            "non-parametric bootstrap of %d replicates."
            % (_f(s.get("r2_nash_sutcliffe"), 3), _f(s.get("rmse"), 3),
               _f(s.get("bias"), 3), cfg.bootstrap))

    parts.append(
        "All computations were carried out in CARAS %s inside QGIS "
        "(reference raster: %s, %s; classified raster: %s, %s)."
        % (result.provenance.get("caras_version", ""), ref.name,
           ref.crs.authid() or ref.crs.description(), cls.name,
           cls.crs.authid() or cls.crs.description()))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------
def text_report(result):
    cfg = result.config
    z = cfg.z
    lvl = 100 * cfg.confidence_level
    L = []
    add = L.append

    add(_SEP)
    add("CARAS - Classification Accuracy and Regression Assessment Suite")
    add("Design-based validation report")
    add(_SEP)
    add("Generated            : %s" % result.provenance.get("timestamp"))
    add("CARAS version        : %s" % result.provenance.get("caras_version"))
    add("Confidence level     : %.0f %%   (z = %.4f)" % (lvl, z))
    add("")

    # -- 1. data ----------------------------------------------------------
    add(_SUB)
    add("1. DATA")
    add(_SUB)
    for tag, info in (("Reference map", result.reference_info),
                      ("Classified map", result.classified_info)):
        d = info.to_dict()
        add("%s" % tag)
        add("    layer            : %s" % d["name"])
        add("    source           : %s" % d["source"])
        add("    CRS              : %s (%s)" % (d["crs"], d["crs_description"]))
        add("    size             : %d x %d px, band %d of %d"
            % (d["width"], d["height"], d["band"], d["band_count"]))
        add("    pixel size       : %.6g x %.6g %s"
            % (d["resolution_x"], d["resolution_y"], d["map_units"]))
        add("    data type        : %s   NoData: %s"
            % (d["data_type"], d["nodata"]))
    win = result.provenance.get("analysis_window", {})
    add("Analysis window      : rows %s-%s, cols %s-%s (%s px of the "
        "reference grid)" % (win.get("row0"), win.get("row1"),
                             win.get("col0"), win.get("col1"),
                             win.get("pixels")))
    add("")

    # -- 2. diagnostics ---------------------------------------------------
    add(_SUB)
    add("2. PRE-FLIGHT DIAGNOSTICS")
    add(_SUB)
    for d in result.diagnostics:
        add("[%-7s] %s" % (d.level.upper(), d.message))
    add("")

    # -- 3. design --------------------------------------------------------
    add(_SUB)
    add("3. SAMPLING DESIGN")
    add(_SUB)
    smp = result.provenance.get("sample", {})
    add("Method               : %s" % smp.get("method"))
    add("Random seed          : %s" % smp.get("seed"))
    add("Units requested      : %s" % smp.get("n_requested"))
    add("Units realised       : %s" % smp.get("n_achieved"))
    if smp.get("allocation"):
        add("Allocation scheme    : %s" % cfg.allocation)
        names = {}
        if result.accuracy is not None:
            names = dict(zip(result.accuracy.categories, result.accuracy.labels))
        add("Units per stratum    : %s" % ", ".join(
            "%s=%d" % (names.get(s, s), a)
            for s, a in zip(smp.get("stratum_values") or [],
                            smp.get("allocation"))))
    if cfg.min_separation:
        add("Minimum separation   : %g map units" % cfg.min_separation)
    for note in smp.get("notes", []):
        add("  * %s" % note)
    add("")

    if result.accuracy is not None:
        _text_categorical(add, result, z, lvl)
    if result.continuous is not None:
        _text_continuous(add, result)

    # -- limitations ------------------------------------------------------
    add(_SUB)
    add("%d. LIMITATIONS AND CAVEATS" % (9 if result.accuracy is not None else 5))
    add(_SUB)
    seen = set()
    n = 0
    for w in result.warnings:
        if w in seen:
            continue
        seen.add(w)
        n += 1
        add("%2d. %s" % (n, _wrap(w, 74, "    ")))
    if n == 0:
        add("No caveats were raised by the diagnostics.")
    add("")

    # -- methods paragraph ------------------------------------------------
    add(_SUB)
    add("METHODS PARAGRAPH (ready to adapt for a manuscript)")
    add(_SUB)
    add(_wrap(methods_paragraph(result), 76, ""))
    add("")

    # -- reproducibility --------------------------------------------------
    add(_SUB)
    add("REPRODUCIBILITY")
    add(_SUB)
    p = result.provenance
    add("CARAS               : %s" % p.get("caras_version"))
    add("QGIS                : %s" % p.get("qgis_version", "n/a"))
    add("Python / NumPy      : %s / %s" % (p.get("python"), p.get("numpy")))
    add("Platform            : %s" % p.get("platform"))
    add("Random seed         : %s" % cfg.seed)
    add("Bootstrap replicates: %s" % cfg.bootstrap)
    add("The JSON export of this report contains the complete configuration, "
        "the class mapping and the raster census, which is sufficient to "
        "reproduce every number above.")
    add("")

    add(_SUB)
    add("REFERENCES")
    add(_SUB)
    for r in REFERENCES:
        add(_wrap(r, 74, "    "))
    add("")
    add(_SEP)
    add("CARAS %s - GNU GPL v3 or later - Omer K. Orucu"
        % p.get("caras_version"))
    add(_SEP)
    return "\n".join(L)


def _text_categorical(add, result, z, lvl):
    acc = result.accuracy
    cfg = result.config
    labels = acc.labels

    add(_SUB)
    add("4. ERROR MATRIX - SAMPLE COUNTS")
    add(_SUB)
    add("Rows = MAP (classified) class, columns = REFERENCE class.")
    add("")
    add(_matrix_block(acc.counts.tolist(), labels, "MAP", "REFERENCE"))
    add("")

    add(_SUB)
    add("5. ERROR MATRIX - ESTIMATED AREA PROPORTIONS")
    add(_SUB)
    if acc.design == "stratified":
        add("Cells are p_ij = W_i * n_ij / n_i. (Olofsson et al. 2014, Eq. 4).")
        add("Stratum weights W_i come from a full census of the "
            "stratification raster:")
        add("    " + ", ".join("%s=%.6f" % (labels[i], acc.weights[i])
                               for i in range(len(labels))))
        add("Using the raw counts above instead of these proportions would "
            "over-weight rare classes and bias every estimate.")
    else:
        add("Cells are p_ij = n_ij / n; under an equal-probability design the "
            "sample proportions already estimate the area proportions.")
    add("")
    add(_matrix_block([[acc.proportions[i][j] for j in range(len(labels))]
                       for i in range(len(labels))],
                      labels, "MAP", "REFERENCE", fmt="%.6f", colwidth=13))
    add("")

    add(_SUB)
    add("6. ACCURACY ESTIMATES (%.0f %% confidence intervals)" % lvl)
    add(_SUB)
    add(_estimate_line("Overall accuracy", acc.overall, z))
    add("")
    wilson_used = False
    add("%-24s %10s %10s %23s %8s" % ("Class", "UA", "+/-", "CI", "n(map)"))
    add(_SUB)
    for i, lab in enumerate(labels):
        e = acc.users[i]
        text, flagged = _ci_best(e, z)
        wilson_used = wilson_used or flagged
        add("%-24s %10s %10s %23s %8d"
            % (str(lab)[:24], _f(e.value), _f(e.margin(z)), text,
               acc.n_by_map_class[i]))
    add("")
    add("%-24s %10s %10s %23s %8s" % ("Class", "PA", "+/-", "CI", "n(ref)"))
    add(_SUB)
    for j, lab in enumerate(labels):
        e = acc.producers[j]
        text, flagged = _ci_best(e, z)
        wilson_used = wilson_used or flagged
        add("%-24s %10s %10s %23s %8d"
            % (str(lab)[:24], _f(e.value), _f(e.margin(z)), text,
               acc.n_by_reference_class[j]))
    if wilson_used:
        add("")
        add(_wrap(WILSON_NOTE, 74, "  "))
    add("")
    add("%-24s %12s %12s %12s %12s"
        % ("Class", "Commission", "Omission", "F1", "F1 +/-"))
    add(_SUB)
    for i, lab in enumerate(labels):
        f1 = acc.f1[i] if i < len(acc.f1) else None
        add("%-24s %12s %12s %12s %12s"
            % (str(lab)[:24], _f(acc.commission(i)), _f(acc.omission(i)),
               _f(f1.value if f1 else None),
               _f(f1.margin(z) if f1 else None)))
    add("")
    add("Macro F1              : %s" % _f(acc.macro_f1()))
    add("Area-weighted F1      : %s" % _f(acc.weighted_f1()))
    add("Macro user's accuracy : %s" % _f(acc.macro_users()))
    add("Macro producer's acc. : %s" % _f(acc.macro_producers()))
    add("Note: user's accuracy = 1 - commission error; producer's accuracy = "
        "1 - omission error.")
    add("F1 standard errors use the delta method and assume UA and PA are "
        "independent, which they are not exactly; treat them as indicative.")
    add("")

    add(_SUB)
    add("7. AREA ESTIMATION")
    add(_SUB)
    unit = acc.area_unit or "units"
    add(_wrap(
        "Adjusted areas remove the bias that pixel counting leaves in the map "
        "(Olofsson et al. 2014, Eqs. 9-10). Where the mapped area falls "
        "outside the confidence interval of the adjusted area, pixel counting "
        "alone would have been misleading for that class.", 76, ""))
    add("")
    add("%-22s %14s %14s %14s %20s"
        % ("Class", "Map area", "Adjusted", "+/- (%.0f%%)" % lvl, "CI"))
    add("%-22s %14s %14s %14s %20s" % ("", unit, unit, unit, unit))
    add(_SUB)
    for row in acc.areas(z):
        if "adjusted_area" in row:
            add("%-22s %14s %14s %14s %20s"
                % (str(row["label"])[:22], _f(row["map_area"], 2),
                   _f(row["adjusted_area"], 2), _f(row["area_margin"], 2),
                   "[%s, %s]" % (_f(row["area_ci_lower"], 2),
                                 _f(row["area_ci_upper"], 2))))
        else:
            add("%-22s %14s %14s %14s %20s"
                % (str(row["label"])[:22], _f(row["map_proportion"]),
                   _f(row["adjusted_proportion"]), _f(row["proportion_se"] * z),
                   "n/a"))
    add(_SUB)
    add("%-22s %14s %14s" % ("TOTAL", _f(sum(
        r.get("map_area", 0.0) for r in acc.areas(z)), 2),
        _f(sum(r.get("adjusted_area", 0.0) for r in acc.areas(z)), 2)))
    if result.reference_info.is_geographic or result.classified_info.is_geographic:
        add("")
        add("Areas are given in pixels because at least one raster uses a "
            "geographic CRS, where pixel ground area varies with latitude.")
    add("")

    d = result.disagreement
    if d is not None:
        add(_SUB)
        add("8. DISAGREEMENT COMPONENTS (replaces kappa)")
        add(_SUB)
        add("Total disagreement          : %s   (= 1 - overall accuracy)"
            % _f(d.total))
        add("  Quantity disagreement     : %s   (%s of the total)"
            % (_f(d.quantity), _pct(d.quantity / d.total if d.total else 0)))
        add("  Allocation disagreement   : %s   (%s of the total)"
            % (_f(d.allocation),
               _pct(d.allocation / d.total if d.total else 0)))
        add("      of which exchange     : %s" % _f(d.exchange))
        add("      of which shift        : %s" % _f(d.shift))
        add("")
        add("%-24s %12s %12s %12s %12s"
            % ("Class", "Quantity", "Allocation", "Exchange", "Shift"))
        add(_SUB)
        for row in d.per_category:
            add("%-24s %12s %12s %12s %12s"
                % (str(row["label"])[:24], _f(row["quantity"]),
                   _f(row["allocation"]), _f(row["exchange"]),
                   _f(row["shift"])))
        add("")
        add("Quantity disagreement is a difference in how much of a class "
            "exists; allocation disagreement is a difference in where it is. "
            "A map dominated by quantity disagreement needs recalibration of "
            "class totals; one dominated by allocation disagreement needs "
            "better spatial discrimination.")
        add("")
        add("Cohen's kappa               : %s" % _f(acc.kappa()))
        add(_wrap(KAPPA_CAUTION, 74, "    "))
        add("")


def _text_continuous(add, result):
    c = result.continuous
    s = c.stats
    add(_SUB)
    add("4. CONTINUOUS AGREEMENT STATISTICS")
    add(_SUB)
    add("Validation pairs           : %d" % s["n"])
    add("Reference mean (sd)        : %s (%s)"
        % (_f(s["reference_mean"]), _f(s["reference_sd"])))
    add("Map mean (sd)              : %s (%s)"
        % (_f(s["map_mean"]), _f(s["map_sd"])))
    add("")
    add("Coefficient of determination (1 - SSE/SST) : %s"
        % _f(s["r2_nash_sutcliffe"]))
    add("Squared Pearson correlation (r^2)          : %s" % _f(s["pearson_r2"]))
    add("  These are different quantities. r^2 ignores bias and scaling; the "
        "coefficient of determination does not. Report the latter.")
    add("")
    add("RMSE                       : %s" % _f(s["rmse"]))
    add("  systematic part          : %s" % _f(s.get("rmse_systematic")))
    add("  unsystematic part        : %s" % _f(s.get("rmse_unsystematic")))
    add("  systematic share of MSE  : %s" % _pct(s.get("systematic_share")))
    add("MAE                        : %s" % _f(s["mae"]))
    add("Mean error (bias)          : %s   %s"
        % (_f(s["bias"]),
           "(map overestimates)" if s["bias"] > 0 else
           "(map underestimates)" if s["bias"] < 0 else "(no mean offset)"))
    add("NRMSE (range / mean)       : %s / %s"
        % (_f(s.get("nrmse_range")), _f(s.get("nrmse_mean"))))
    add("MAPE                       : %s %%" % _f(s.get("mape_percent"), 2))
    add("")
    add("Willmott index of agreement d  : %s" % _f(s["willmott_d"]))
    add("Refined index d1 (2012)        : %s" % _f(s["willmott_d1"]))
    add("Lin concordance correlation    : %s" % _f(s["lins_ccc"]))
    add("")
    add("OLS regression of map on reference:")
    add("  slope                    : %s (SE %s)  t vs 1 = %s"
        % (_f(s.get("ols_slope")), _f(s.get("ols_slope_se")),
           _f(s.get("t_slope_vs_1"), 2)))
    add("  intercept                : %s (SE %s)  t vs 0 = %s"
        % (_f(s.get("ols_intercept")), _f(s.get("ols_intercept_se")),
           _f(s.get("t_intercept_vs_0"), 2)))
    add("  residual sd              : %s  (df = %s)"
        % (_f(s.get("ols_residual_sd")), s.get("ols_dof")))
    if c.bootstrap:
        add("")
        add("Bootstrap %d%% confidence intervals (%s replicates, seed %s):"
            % (95, c.bootstrap.get("_meta", {}).get("replicates_requested"),
               c.bootstrap.get("_meta", {}).get("seed")))
        add("%-24s %14s %14s %12s" % ("Statistic", "CI lower", "CI upper", "SE"))
        add(_SUB)
        for k, v in c.bootstrap.items():
            if k.startswith("_"):
                continue
            add("%-24s %14s %14s %12s"
                % (k, _f(v["ci_lower"]), _f(v["ci_upper"]),
                   _f(v["bootstrap_se"])))
    add("")


def _wrap(text, width, indent):
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    if not lines:
        return ""
    return ("\n" + indent).join(lines)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------
def json_report(result, indent=2):
    cfg = result.config
    z = cfg.z
    out = {
        "software": {
            "name": "CARAS - Classification Accuracy and Regression "
                    "Assessment Suite",
            "version": result.provenance.get("caras_version"),
            "license": "GPL-3.0-or-later",
            "author": "Omer K. Orucu",
        },
        "provenance": result.provenance,
        "warnings": list(dict.fromkeys(result.warnings)),
        "methods_paragraph": methods_paragraph(result),
        "references": REFERENCES,
    }

    acc = result.accuracy
    if acc is not None:
        out["categorical"] = {
            "categories": acc.categories,
            "labels": acc.labels,
            "design": acc.design,
            "matrix_orientation": "rows = map (classified) class, "
                                  "columns = reference class",
            "counts": acc.counts.tolist(),
            "estimated_area_proportions": acc.proportions.tolist(),
            "stratum_weights": (None if acc.weights is None
                                else [float(w) for w in acc.weights]),
            "overall_accuracy": acc.overall.to_dict(z),
            "users_accuracy": [e.to_dict(z) for e in acc.users],
            "producers_accuracy": [e.to_dict(z) for e in acc.producers],
            "f1_per_class": [e.to_dict(z) for e in acc.f1],
            "macro_f1": acc.macro_f1(),
            "area_weighted_f1": acc.weighted_f1(),
            "commission_error": [acc.commission(i) for i in range(len(acc.labels))],
            "omission_error": [acc.omission(j) for j in range(len(acc.labels))],
            "area_estimation": acc.areas(z),
            "area_unit": acc.area_unit,
            "area_total": acc.area_total,
            "cohens_kappa": acc.kappa(),
            "cohens_kappa_caution": KAPPA_CAUTION,
        }
        if result.disagreement is not None:
            out["categorical"]["disagreement"] = result.disagreement.to_dict()
    if result.continuous is not None:
        out["continuous"] = result.continuous.to_dict()

    # JSON has no NaN or Infinity; emit null instead so the file stays valid
    # for every strict parser (R jsonlite, pandas.read_json, jq).
    return json.dumps(_sanitize(out), indent=indent, ensure_ascii=False,
                      allow_nan=False, default=_default)


def _sanitize(obj):
    """Recursively replace non-finite floats with None and NumPy types with
    their Python equivalents."""
    if isinstance(obj, dict):
        return dict((str(k), _sanitize(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    return obj


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------
def matrix_csv(result):
    acc = result.accuracy
    if acc is None:
        return "no categorical result\n"
    labels = acc.labels
    lines = []
    lines.append("# CARAS error matrix. Rows = MAP class, columns = "
                 "REFERENCE class.")
    lines.append("# Section 1: sample counts")
    lines.append("map_class," + ",".join(_csv(l) for l in labels) + ",row_total")
    for i, lab in enumerate(labels):
        row = acc.counts[i]
        lines.append(_csv(lab) + "," + ",".join(str(int(v)) for v in row)
                     + "," + str(int(row.sum())))
    lines.append("column_total," + ",".join(
        str(int(acc.counts[:, j].sum())) for j in range(len(labels)))
        + "," + str(int(acc.counts.sum())))
    lines.append("")
    lines.append("# Section 2: estimated area proportions")
    lines.append("map_class," + ",".join(_csv(l) for l in labels) + ",row_total")
    for i, lab in enumerate(labels):
        row = acc.proportions[i]
        lines.append(_csv(lab) + "," + ",".join("%.10f" % v for v in row)
                     + ",%.10f" % row.sum())
    lines.append("column_total," + ",".join(
        "%.10f" % acc.proportions[:, j].sum() for j in range(len(labels)))
        + ",%.10f" % acc.proportions.sum())
    lines.append("")
    lines.append("# Section 3: class estimates")
    z = result.config.z
    lines.append("class,users_accuracy,ua_se,ua_ci_lower,ua_ci_upper,"
                 "producers_accuracy,pa_se,pa_ci_lower,pa_ci_upper,"
                 "map_area,adjusted_area,adjusted_area_se,area_unit")
    areas = acc.areas(z)
    for i, lab in enumerate(labels):
        u, p = acc.users[i], acc.producers[i]
        ul, uh = u.ci(z)
        pl, ph = p.ci(z)
        a = areas[i]
        lines.append(",".join([
            _csv(lab), "%.8f" % u.value, "%.8f" % u.se, "%.8f" % ul,
            "%.8f" % uh, "%.8f" % p.value, "%.8f" % p.se, "%.8f" % pl,
            "%.8f" % ph,
            "%.6f" % a.get("map_area", float("nan")),
            "%.6f" % a.get("adjusted_area", float("nan")),
            "%.6f" % a.get("area_se", float("nan")),
            _csv(acc.area_unit or "")]))
    return "\n".join(lines) + "\n"


def _csv(value):
    s = str(value)
    if any(ch in s for ch in ',"\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
_CSS = """
:root{--ink:#1b2733;--muted:#5b6b7b;--line:#d7dee5;--accent:#1f6f8b;
      --warn:#8a5a00;--bad:#8c2f24;--bg:#fff;--soft:#f4f7f9;}
*{box-sizing:border-box}
body{font:15px/1.55 "Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
     background:var(--soft);margin:0;padding:32px 16px}
.wrap{max-width:1080px;margin:0 auto;background:var(--bg);padding:36px 40px;
      border:1px solid var(--line);border-radius:6px}
h1{font-size:24px;margin:0 0 4px;color:var(--accent)}
h2{font-size:17px;margin:34px 0 10px;padding-bottom:6px;
   border-bottom:2px solid var(--accent)}
h3{font-size:14px;margin:20px 0 6px;color:var(--muted);
   text-transform:uppercase;letter-spacing:.04em}
p.sub{color:var(--muted);margin:0 0 18px}
table{border-collapse:collapse;width:100%;margin:12px 0 18px;font-size:13.5px}
th,td{border:1px solid var(--line);padding:6px 9px;text-align:right}
th{background:var(--soft);font-weight:600;text-align:center}
td.l,th.l{text-align:left}
tr.total td{font-weight:600;background:var(--soft)}
td.diag{background:#eef6f0}
.kv{display:grid;grid-template-columns:220px 1fr;gap:2px 14px;font-size:13.5px}
.kv div:nth-child(odd){color:var(--muted)}
.note{background:var(--soft);border-left:4px solid var(--accent);
      padding:10px 14px;margin:12px 0;font-size:13.5px}
.warn{border-left-color:var(--warn)}
.bad{border-left-color:var(--bad)}
ol.caveats li{margin-bottom:7px;font-size:13.5px}
code,pre{font-family:Consolas,Menlo,monospace;font-size:12.5px}
pre{background:var(--soft);padding:12px;border:1px solid var(--line);
    overflow-x:auto;white-space:pre-wrap}
footer{margin-top:34px;padding-top:14px;border-top:1px solid var(--line);
       color:var(--muted);font-size:12.5px}
@media print{body{background:#fff;padding:0}.wrap{border:0;padding:0}}
"""


def html_report(result):
    cfg = result.config
    z = cfg.z
    lvl = 100 * cfg.confidence_level
    acc = result.accuracy
    H = []
    a = H.append

    a("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>")
    a("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    a("<title>CARAS validation report</title><style>%s</style></head><body>"
      % _CSS)
    a("<div class='wrap'>")
    a("<h1>CARAS validation report</h1>")
    a("<p class='sub'>Design-based accuracy and area assessment &middot; "
      "generated %s &middot; CARAS %s</p>"
      % (_esc(str(result.provenance.get("timestamp"))),
         _esc(str(result.provenance.get("caras_version")))))

    # data
    a("<h2>1. Data</h2><div class='kv'>")
    for tag, info in (("Reference map", result.reference_info),
                      ("Classified map", result.classified_info)):
        d = info.to_dict()
        a("<div>%s</div><div>%s &mdash; %s, %d&times;%d px, %.6g %s pixels</div>"
          % (_esc(tag), _esc(d["name"]), _esc(str(d["crs"])), d["width"],
             d["height"], d["resolution_x"], _esc(d["map_units"])))
    a("<div>Confidence level</div><div>%.0f %%</div>" % lvl)
    a("<div>Random seed</div><div>%s</div>" % _esc(str(cfg.seed)))
    a("<div>Sampling design</div><div>%s, %d units requested, %d realised</div>"
      % (_esc(str(cfg.method)), cfg.n_points, len(result.sample)))
    a("</div>")

    # diagnostics
    a("<h2>2. Pre-flight diagnostics</h2>")
    for d in result.diagnostics:
        cls_ = {"error": "note bad", "warning": "note warn"}.get(d.level, "note")
        a("<div class='%s'><strong>%s</strong> &mdash; %s</div>"
          % (cls_, _esc(d.level.upper()), _esc(d.message)))

    if acc is not None:
        labels = [_esc(str(l)) for l in acc.labels]
        k = len(labels)

        a("<h2>3. Error matrix &mdash; sample counts</h2>")
        a("<p class='sub'>Rows = map (classified) class, columns = reference "
          "class.</p>")
        a(_html_matrix(acc.counts.tolist(), labels, "%d"))

        a("<h2>4. Error matrix &mdash; estimated area proportions</h2>")
        if acc.design == "stratified":
            a("<div class='note'>Cells are "
              "<code>p<sub>ij</sub> = W<sub>i</sub> n<sub>ij</sub> / "
              "n<sub>i&middot;</sub></code> (Olofsson et al. 2014, Eq. 4). "
              "Stratum weights come from a full census of the stratification "
              "raster; using raw counts would over-weight rare classes and "
              "bias every estimate.</div>")
        else:
            a("<div class='note'>Under an equal-probability design the sample "
              "proportions already estimate the area proportions.</div>")
        a(_html_matrix(acc.proportions.tolist(), labels, "%.6f"))

        a("<h2>5. Accuracy estimates (%.0f %% confidence intervals)</h2>" % lvl)
        a("<div class='note'>Overall accuracy <strong>%s</strong> "
          "&plusmn; %s &nbsp; %s</div>"
          % (_f(acc.overall.value), _f(acc.overall.margin(z)),
             _esc(_ci(acc.overall, z))))
        a("<table><tr><th class='l'>Class</th><th>User's</th><th>CI</th>"
          "<th>n (map)</th><th>Producer's</th><th>CI</th><th>n (ref)</th>"
          "<th>F1</th><th>Commission</th><th>Omission</th></tr>")
        wilson_used = False
        for i in range(k):
            u, p = acc.users[i], acc.producers[i]
            f1 = acc.f1[i] if i < len(acc.f1) else None
            u_txt, u_flag = _ci_best(u, z, 3)
            p_txt, p_flag = _ci_best(p, z, 3)
            wilson_used = wilson_used or u_flag or p_flag
            a("<tr><td class='l'>%s</td><td>%s</td><td>%s</td><td>%d</td>"
              "<td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td>"
              "<td>%s</td></tr>"
              % (labels[i], _f(u.value), _esc(u_txt),
                 acc.n_by_map_class[i], _f(p.value), _esc(p_txt),
                 acc.n_by_reference_class[i],
                 _f(f1.value if f1 else None), _f(acc.commission(i)),
                 _f(acc.omission(i))))
        a("<tr class='total'><td class='l'>Macro / weighted</td><td>%s</td>"
          "<td></td><td>%d</td><td>%s</td><td></td><td>%d</td><td>%s</td>"
          "<td></td><td></td></tr>"
          % (_f(acc.macro_users()), int(acc.counts.sum()),
             _f(acc.macro_producers()), int(acc.counts.sum()),
             _f(acc.macro_f1())))
        a("</table>")
        if wilson_used:
            a("<div class='note warn'>%s</div>" % _esc(WILSON_NOTE))

        a("<h2>6. Area estimation</h2>")
        a("<div class='note'>Adjusted areas remove the bias left by pixel "
          "counting (Olofsson et al. 2014, Eqs. 9&ndash;10). Where the mapped "
          "area falls outside the confidence interval of the adjusted area, "
          "pixel counting alone would have misled.</div>")
        unit = _esc(str(acc.area_unit or ""))
        a("<table><tr><th class='l'>Class</th><th>Map area (%s)</th>"
          "<th>Adjusted (%s)</th><th>&plusmn; %.0f %%</th><th>CI</th>"
          "<th>Map share</th><th>Adjusted share</th></tr>" % (unit, unit, lvl))
        for row in acc.areas(z):
            a("<tr><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td>"
              "<td>%s</td><td>%s</td><td>%s</td></tr>"
              % (_esc(str(row["label"])), _f(row.get("map_area"), 2),
                 _f(row.get("adjusted_area"), 2), _f(row.get("area_margin"), 2),
                 "[%s, %s]" % (_f(row.get("area_ci_lower"), 2),
                               _f(row.get("area_ci_upper"), 2)),
                 _pct(row["map_proportion"]),
                 _pct(row["adjusted_proportion"])))
        a("</table>")

        d = result.disagreement
        if d is not None:
            a("<h2>7. Disagreement components</h2>")
            a("<table><tr><th class='l'>Component</th><th>Value</th>"
              "<th>Share of disagreement</th></tr>")
            for name, val in (("Total disagreement (1 - OA)", d.total),
                              ("Quantity", d.quantity),
                              ("Allocation", d.allocation),
                              ("&nbsp;&nbsp;of which exchange", d.exchange),
                              ("&nbsp;&nbsp;of which shift", d.shift)):
                a("<tr><td class='l'>%s</td><td>%s</td><td>%s</td></tr>"
                  % (name, _f(val), _pct(val / d.total if d.total else 0)))
            a("</table>")
            a("<table><tr><th class='l'>Class</th><th>Quantity</th>"
              "<th>Allocation</th><th>Exchange</th><th>Shift</th></tr>")
            for row in d.per_category:
                a("<tr><td class='l'>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s</td></tr>"
                  % (_esc(str(row["label"])), _f(row["quantity"]),
                     _f(row["allocation"]), _f(row["exchange"]),
                     _f(row["shift"])))
            a("</table>")
            a("<div class='note warn'><strong>Cohen&rsquo;s kappa = %s.</strong> %s</div>"
              % (_f(acc.kappa()), _esc(KAPPA_CAUTION)))

    if result.continuous is not None:
        s = result.continuous.stats
        a("<h2>3. Continuous agreement statistics</h2>")
        a("<table><tr><th class='l'>Statistic</th><th>Value</th>"
          "<th>Bootstrap CI</th></tr>")
        order = [("Validation pairs", "n", 0),
                 ("Coefficient of determination", "r2_nash_sutcliffe", 4),
                 ("Squared Pearson r", "pearson_r2", 4),
                 ("RMSE", "rmse", 4), ("RMSE systematic", "rmse_systematic", 4),
                 ("RMSE unsystematic", "rmse_unsystematic", 4),
                 ("MAE", "mae", 4), ("Mean error (bias)", "bias", 4),
                 ("Willmott d", "willmott_d", 4),
                 ("Refined d1", "willmott_d1", 4),
                 ("Lin CCC", "lins_ccc", 4),
                 ("OLS slope", "ols_slope", 4),
                 ("OLS intercept", "ols_intercept", 4)]
        for name, key, nd in order:
            ci = result.continuous.bootstrap.get(key)
            ci_txt = ("[%s, %s]" % (_f(ci["ci_lower"], nd), _f(ci["ci_upper"], nd))
                      if ci else "")
            a("<tr><td class='l'>%s</td><td>%s</td><td>%s</td></tr>"
              % (_esc(name), _f(s.get(key), nd), _esc(ci_txt)))
        a("</table>")
        a("<div class='note'>The coefficient of determination and the squared "
          "Pearson correlation are different quantities: r&sup2; is blind to "
          "bias and to a wrong slope. Report the coefficient of "
          "determination.</div>")

    # caveats
    a("<h2>Limitations and caveats</h2><ol class='caveats'>")
    seen = set()
    for w in result.warnings:
        if w in seen:
            continue
        seen.add(w)
        a("<li>%s</li>" % _esc(w))
    if not seen:
        a("<li>No caveats were raised by the diagnostics.</li>")
    a("</ol>")

    a("<h2>Methods paragraph</h2>")
    a("<p>%s</p>" % _esc(methods_paragraph(result)))

    a("<h2>Reproducibility</h2><div class='kv'>")
    p = result.provenance
    for label, key in (("CARAS", "caras_version"), ("QGIS", "qgis_version"),
                       ("Python", "python"), ("NumPy", "numpy"),
                       ("Platform", "platform"), ("Timestamp", "timestamp")):
        a("<div>%s</div><div>%s</div>" % (_esc(label), _esc(str(p.get(key, "n/a")))))
    a("<div>Random seed</div><div>%s</div>" % _esc(str(cfg.seed)))
    a("</div>")

    a("<h2>References</h2><ol>")
    for r in REFERENCES:
        a("<li>%s</li>" % _esc(r))
    a("</ol>")

    a("<footer>Generated by CARAS %s &mdash; Classification Accuracy and "
      "Regression Assessment Suite &mdash; GNU GPL v3 or later &mdash; "
      "&Ouml;mer K. &Ouml;r&uuml;c&uuml;</footer>"
      % _esc(str(p.get("caras_version"))))
    a("</div></body></html>")
    return "\n".join(H)


def _html_matrix(matrix, labels, fmt):
    k = len(labels)
    out = ["<table><tr><th class='l'>Map \\ Reference</th>"]
    for l in labels:
        out.append("<th>%s</th>" % l)
    out.append("<th>Total</th></tr>")
    for i in range(k):
        out.append("<tr><td class='l'>%s</td>" % labels[i])
        for j in range(k):
            css = " class='diag'" if i == j else ""
            out.append("<td%s>%s</td>" % (css, fmt % matrix[i][j]))
        out.append("<td>%s</td></tr>" % (fmt % sum(matrix[i])))
    out.append("<tr class='total'><td class='l'>Total</td>")
    grand = 0.0
    for j in range(k):
        s = sum(matrix[i][j] for i in range(k))
        grand += s
        out.append("<td>%s</td>" % (fmt % s))
    out.append("<td>%s</td></tr></table>" % (fmt % grand))
    return "".join(out)
