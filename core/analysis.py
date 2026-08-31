# -*- coding: utf-8 -*-
"""
CARAS core :: analysis orchestration
====================================

Ties the raster access, sampling designs and estimators together into one
reproducible run, and records the complete provenance of that run so the
numbers in a paper can be regenerated from the report alone.

The class-mapping idea that made CARAS v1 useful is kept and strengthened: two
rasters with unrelated legends (a model output against CORINE, ESA WorldCover
against a national product) are folded onto a common set of categories before
anything statistical happens.  What changed is everything downstream of that
fold - the design, the estimator and the uncertainty.
"""

from __future__ import division

import math
import platform
import sys
from datetime import datetime

import numpy as np

from . import disagreement as dis
from . import estimators as est
from . import raster as rio
from . import regression as reg
from . import sampling as smp

__all__ = ["AnalysisConfig", "AnalysisResult", "ClassMapping", "run_analysis",
           "CARAS_VERSION"]

CARAS_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
class ClassMapping(object):
    """Fold raw pixel values of one raster onto analysis categories."""

    def __init__(self, value_to_category=None, labels=None):
        self.value_to_category = dict(value_to_category or {})
        self.labels = dict(labels or {})

    def categories(self):
        return sorted(set(self.value_to_category.values()))

    def members(self, category):
        return [v for v, c in self.value_to_category.items() if c == category]

    def apply(self, values):
        """Vectorised mapping; unmapped values become NaN."""
        values = np.asarray(values, dtype=float)
        out = np.full(values.shape, np.nan)
        for raw, cat in self.value_to_category.items():
            out[values == raw] = cat
        return out

    def to_dict(self):
        return {
            "value_to_category": dict((str(k), int(v))
                                      for k, v in self.value_to_category.items()),
            "labels": dict((str(k), v) for k, v in self.labels.items()),
        }


class AnalysisConfig(object):
    """Everything the analyst chose, in one serialisable object."""

    def __init__(self, reference_layer, classified_layer, mode="categorical",
                 reference_band=1, classified_band=1, method="random",
                 n_points=500, seed=42, allocation="proportional",
                 min_per_stratum=0, strata_source="map", min_separation=0.0,
                 reference_mapping=None, classified_mapping=None,
                 category_labels=None, points=None, points_crs=None,
                 points_reference_values=None, points_ids=None,
                 declared_design="srs", bootstrap=2000,
                 confidence_level=0.95, chunk_rows=rio.DEFAULT_CHUNK_ROWS):
        self.reference_layer = reference_layer
        self.classified_layer = classified_layer
        self.reference_band = int(reference_band)
        self.classified_band = int(classified_band)
        self.mode = mode                      # categorical | continuous
        self.method = method                  # random | systematic | stratified | points
        self.n_points = int(n_points)
        self.seed = seed
        self.allocation = allocation
        self.min_per_stratum = int(min_per_stratum)
        self.strata_source = strata_source    # map | reference
        self.min_separation = float(min_separation or 0.0)
        self.reference_mapping = reference_mapping or ClassMapping()
        self.classified_mapping = classified_mapping or ClassMapping()
        self.category_labels = dict(category_labels or {})
        self.points = points                  # (xs, ys) in points_crs
        self.points_crs = points_crs
        self.points_reference_values = points_reference_values
        self.points_ids = points_ids
        self.declared_design = declared_design
        self.bootstrap = int(bootstrap)
        self.confidence_level = float(confidence_level)
        self.chunk_rows = int(chunk_rows)

    @property
    def z(self):
        # only the two conventional levels are offered by the UI
        return {0.90: 1.6448536269514722,
                0.95: est.Z_95,
                0.99: 2.5758293035489004}.get(round(self.confidence_level, 2),
                                              est.Z_95)

    def to_dict(self):
        return {
            "caras_version": CARAS_VERSION,
            "mode": self.mode,
            "sampling_method": self.method,
            "n_points_requested": self.n_points,
            "random_seed": self.seed,
            "allocation": self.allocation,
            "minimum_per_stratum": self.min_per_stratum,
            "strata_source": self.strata_source,
            "minimum_separation_map_units": self.min_separation,
            "declared_design": self.declared_design,
            "bootstrap_replicates": self.bootstrap,
            "confidence_level": self.confidence_level,
            "reference_band": self.reference_band,
            "classified_band": self.classified_band,
            "reference_mapping": self.reference_mapping.to_dict(),
            "classified_mapping": self.classified_mapping.to_dict(),
            "category_labels": dict((str(k), v)
                                    for k, v in self.category_labels.items()),
        }


class AnalysisResult(object):
    """Outcome of one run, ready for the reporting layer."""

    def __init__(self):
        self.config = None
        self.reference_info = None
        self.classified_info = None
        self.diagnostics = []
        self.window = None
        self.reference_census = None
        self.classified_census = None
        self.sample = None
        self.reference_values = None      # raw pixel values at the sample
        self.classified_values = None
        self.reference_categories = None  # mapped categories
        self.classified_categories = None
        self.accuracy = None              # estimators.AccuracyResult
        self.disagreement = None          # disagreement.Disagreement
        self.continuous = None            # regression.ContinuousResult
        self.provenance = {}
        self.warnings = []
        self.timestamp = None

    @property
    def is_categorical(self):
        return self.accuracy is not None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _PairExtractor(object):
    """Cached extraction of the value pair at candidate pixels."""

    def __init__(self, ref_info, cls_info, coord_crs, ref_mapping=None,
                 cls_mapping=None, categorical=True):
        self.ref = ref_info
        self.cls = cls_info
        self.to_ref = rio.make_transform(coord_crs, ref_info.crs)
        self.to_cls = rio.make_transform(coord_crs, cls_info.crs)
        self.cache = {}
        self.ref_mapping = ref_mapping
        self.cls_mapping = cls_mapping
        self.categorical = categorical

    def fetch(self, rows, cols, xs, ys):
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        miss_idx = [i for i in range(len(rows))
                    if (int(rows[i]), int(cols[i])) not in self.cache]
        if miss_idx:
            mx = np.asarray([xs[i] for i in miss_idx], dtype=float)
            my = np.asarray([ys[i] for i in miss_idx], dtype=float)
            rv = rio.sample_values(self.ref, mx, my, transform=self.to_ref)
            cv = rio.sample_values(self.cls, mx, my, transform=self.to_cls)
            for k, i in enumerate(miss_idx):
                self.cache[(int(rows[i]), int(cols[i]))] = (rv[k], cv[k])
        out_r = np.empty(len(rows), dtype=float)
        out_c = np.empty(len(rows), dtype=float)
        for i in range(len(rows)):
            a, b = self.cache[(int(rows[i]), int(cols[i]))]
            out_r[i], out_c[i] = a, b
        return out_r, out_c

    def validator(self, rows, cols, xs, ys):
        rv, cv = self.fetch(rows, cols, xs, ys)
        ok = np.isfinite(rv) & np.isfinite(cv)
        if self.categorical:
            # a value the analyst did not map is not part of the population
            if self.ref_mapping is not None:
                ok &= np.isfinite(self.ref_mapping.apply(rv))
            if self.cls_mapping is not None:
                ok &= np.isfinite(self.cls_mapping.apply(cv))
        return ok


def _confusion(ref_cat, cls_cat, categories):
    """Counts with rows = MAP class, columns = REFERENCE class."""
    k = len(categories)
    index = dict((c, i) for i, c in enumerate(categories))
    cm = np.zeros((k, k), dtype=np.int64)
    for r, c in zip(ref_cat, cls_cat):
        if not (np.isfinite(r) and np.isfinite(c)):
            continue
        ri = index.get(int(r))
        ci = index.get(int(c))
        if ri is None or ci is None:
            continue
        cm[ci, ri] += 1          # row = map, column = reference
    return cm


def _category_weights(census, mapping, categories):
    """Pixel count and area share of every category from a raster census."""
    counts = census["counts"]
    per_cat = dict((c, 0) for c in categories)
    unmapped = 0
    for value, cnt in counts.items():
        cat = mapping.value_to_category.get(value)
        if cat is None:
            unmapped += cnt
        else:
            per_cat[cat] = per_cat.get(cat, 0) + int(cnt)
    total = sum(per_cat.values())
    weights = np.array([per_cat[c] / float(total) if total else 0.0
                        for c in categories], dtype=float)
    return per_cat, weights, unmapped, total


def _noop(*_args, **_kwargs):
    return None


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_analysis(config, progress=None, status=None, cancel=None,
                 census_cache=None):
    """Execute a complete, reproducible validation run.

    ``progress(fraction)`` and ``status(text)`` are optional UI callbacks;
    ``cancel()`` may return True to abort.  ``census_cache`` lets the caller
    reuse the censuses it already computed for the class-mapping dialog.
    """
    progress = progress or _noop
    status = status or _noop
    cancel = cancel or (lambda: False)

    res = AnalysisResult()
    res.config = config
    res.timestamp = datetime.now()

    status("Reading raster metadata")
    ref = rio.describe(config.reference_layer, config.reference_band)
    cls = rio.describe(config.classified_layer, config.classified_band)
    res.reference_info = ref
    res.classified_info = cls
    progress(0.02)

    status("Checking raster compatibility")
    res.diagnostics = rio.check_pair(ref, cls)
    blocking = [d for d in res.diagnostics if d.is_blocking]
    if blocking:
        raise ValueError("; ".join(d.message for d in blocking))

    window = smp.intersection_window(ref, cls)
    res.window = window
    if window.pixel_count <= 0:
        raise ValueError("The two rasters share no common area.")
    progress(0.05)

    categorical = config.mode == "categorical"

    # Stratification by the map class - the design Olofsson et al. (2014)
    # assume - draws its units on the classified grid; every other design
    # draws them on the reference grid.  The extractor is told which CRS the
    # sample coordinates arrive in so that a mixed-CRS pair is transformed
    # rather than mis-indexed.
    strata_on_map = (config.method == "stratified"
                     and config.strata_source == "map" and categorical)
    sampling_grid = cls if strata_on_map else ref
    extractor = _PairExtractor(
        ref, cls, sampling_grid.crs,
        config.reference_mapping if categorical else None,
        config.classified_mapping if categorical else None,
        categorical=categorical)

    # ------------------------------------------------------------------
    # census (categorical mode only - it supplies the stratum weights and
    # the uncorrected map areas)
    # ------------------------------------------------------------------
    if categorical:
        if census_cache and "reference" in census_cache:
            res.reference_census = census_cache["reference"]
        else:
            status("Census of the reference raster")
            res.reference_census = rio.value_census(
                ref, chunk_rows=config.chunk_rows, window=window,
                progress=lambda f: progress(0.05 + 0.15 * f), cancel=cancel)
        if census_cache and "classified" in census_cache:
            res.classified_census = census_cache["classified"]
        else:
            status("Census of the classified raster")
            res.classified_census = rio.value_census(
                cls, chunk_rows=config.chunk_rows,
                window=smp.intersection_window(cls, ref),
                progress=lambda f: progress(0.20 + 0.15 * f), cancel=cancel)
    progress(0.35)

    categories = sorted(set(list(config.reference_mapping.categories())
                            + list(config.classified_mapping.categories()))) \
        if categorical else []
    labels = [config.category_labels.get(c, "Category %d" % c)
              for c in categories]

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    status("Drawing the sample")
    strata_defs = None
    weights = None
    grid_for_strata = None

    if config.method == "points":
        xs, ys = config.points
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if config.points_crs is not None:
            tr = rio.make_transform(config.points_crs, ref.crs)
            if tr is not None:
                from qgis.core import QgsPointXY
                pts = [tr.transform(QgsPointXY(float(x), float(y)))
                       for x, y in zip(xs, ys)]
                xs = np.asarray([p.x() for p in pts])
                ys = np.asarray([p.y() for p in pts])
        sample = smp.sample_from_points(ref, xs, ys, validator=extractor.validator)
    elif config.method == "stratified":
        if not categorical:
            raise ValueError(
                "Stratified sampling needs categories; it is not available in "
                "continuous mode. Use simple random or systematic sampling.")
        if config.strata_source == "map":
            grid_for_strata = cls
            census = res.classified_census
            mapping = config.classified_mapping
        else:
            grid_for_strata = ref
            census = res.reference_census
            mapping = config.reference_mapping
        per_cat, weights, unmapped, total = _category_weights(
            census, mapping, categories)
        if unmapped:
            res.warnings.append(
                "%d pixels (%.3f%% of the stratification raster) carry values "
                "that were not assigned to any category and are excluded from "
                "the target population."
                % (unmapped, 100.0 * unmapped / max(total + unmapped, 1)))
        strata_defs = [(c, mapping.members(c), per_cat.get(c, 0))
                       for c in categories]
        live = [i for i, d in enumerate(strata_defs) if d[2] > 0]
        if not live:
            raise ValueError("No stratum contains any pixel.")
        alloc_full = np.zeros(len(categories), dtype=int)
        sub_weights = np.array([strata_defs[i][2] for i in live], dtype=float)
        sub_alloc = est.allocate_sample(
            sub_weights, config.n_points, scheme=config.allocation,
            minimum=config.min_per_stratum)
        for k, i in enumerate(live):
            alloc_full[i] = sub_alloc[k]
        if grid_for_strata is cls:
            # strata live on the classified grid; sample there, then read back
            sample = smp.stratified_sample(
                cls, strata_defs, alloc_full, seed=config.seed,
                window=smp.intersection_window(cls, ref),
                validator=None, chunk_rows=config.chunk_rows,
                progress=lambda f: progress(0.35 + 0.25 * f), cancel=cancel)
            keep = extractor.validator(sample.rows, sample.cols,
                                       sample.xs, sample.ys)
            dropped = int((~keep).sum())
            sample = sample.subset(keep)
            if dropped:
                sample.notes.append(
                    "%d drawn units lacked valid data in the reference raster "
                    "and were removed." % dropped)
        else:
            sample = smp.stratified_sample(
                ref, strata_defs, alloc_full, seed=config.seed,
                window=window, validator=extractor.validator,
                chunk_rows=config.chunk_rows,
                progress=lambda f: progress(0.35 + 0.25 * f), cancel=cancel)
    elif config.method == "systematic":
        sample = smp.systematic_sample(
            ref, config.n_points, seed=config.seed, window=window,
            validator=extractor.validator,
            progress=lambda f: progress(0.35 + 0.25 * f))
    else:
        sample = smp.simple_random_sample(
            ref, config.n_points, seed=config.seed, window=window,
            validator=extractor.validator,
            progress=lambda f: progress(0.35 + 0.25 * f))

    if config.min_separation > 0:
        sample = smp.apply_min_separation(sample, config.min_separation, ref)
    res.sample = sample
    if len(sample) < 2:
        raise ValueError(
            "Only %d usable validation unit(s) could be placed. Check the "
            "NoData settings, the class mapping and the overlap of the two "
            "rasters." % len(sample))
    progress(0.62)

    # ------------------------------------------------------------------
    # value extraction
    # ------------------------------------------------------------------
    status("Extracting values at the sample")
    if config.method == "stratified" and grid_for_strata is cls:
        rv, cv = extractor.fetch(sample.rows, sample.cols, sample.xs, sample.ys)
    else:
        rv, cv = extractor.fetch(sample.rows, sample.cols, sample.xs, sample.ys)

    if config.method == "points" and config.points_reference_values is not None:
        supplied = np.asarray(config.points_reference_values, dtype=float)
        if supplied.size == rv.size:
            rv = supplied
        else:
            res.warnings.append(
                "The supplied reference values could not be matched one to one "
                "with the retained points; the reference raster was used "
                "instead.")
    res.reference_values = rv
    res.classified_values = cv
    progress(0.7)

    # ------------------------------------------------------------------
    # estimation
    # ------------------------------------------------------------------
    if categorical:
        status("Estimating accuracy and area")
        ref_cat = config.reference_mapping.apply(rv)
        cls_cat = config.classified_mapping.apply(cv)
        res.reference_categories = ref_cat
        res.classified_categories = cls_cat

        cm = _confusion(ref_cat, cls_cat, categories)
        if cm.sum() < 2:
            raise ValueError("Fewer than two units survived the class mapping.")

        pixel_area = cls.pixel_area
        if res.classified_census is not None:
            valid_pixels = res.classified_census["valid_pixels"]
        else:
            valid_pixels = window.pixel_count
        area_value, area_unit = rio.area_conversions(
            cls, valid_pixels * pixel_area)
        if area_value is None:
            area_total, area_unit = float(valid_pixels), "pixels"
        else:
            area_total = area_value

        if config.method == "stratified":
            res.accuracy = est.stratified_estimate(
                cm, weights, categories, labels,
                strata_axis=("map" if config.strata_source == "map"
                             else "reference"),
                area_total=area_total, pixel_area=pixel_area,
                area_unit=area_unit)
        else:
            design = "systematic" if config.method == "systematic" else "srs"
            if config.method == "points":
                design = config.declared_design or "srs"
            res.accuracy = est.srs_estimate(
                cm, categories, labels, area_total=area_total,
                pixel_area=pixel_area, area_unit=area_unit, design=design)

        res.disagreement = dis.disagreement(res.accuracy.proportions,
                                            categories, labels)

        # uncorrected map proportions from the census, for comparison
        if res.classified_census is not None:
            per_cat, w_map, _un, _tot = _category_weights(
                res.classified_census, config.classified_mapping, categories)
            res.provenance["map_class_pixels"] = dict(
                (str(c), int(per_cat.get(c, 0))) for c in categories)
            if config.method != "stratified":
                res.accuracy.map_proportions = [float(v) for v in w_map]
    else:
        status("Computing continuous agreement statistics")
        res.continuous = reg.continuous_agreement(
            rv, cv, bootstrap=config.bootstrap, seed=config.seed, z=config.z)
    progress(0.9)

    # ------------------------------------------------------------------
    # provenance
    # ------------------------------------------------------------------
    status("Assembling provenance")
    res.provenance.update({
        "caras_version": CARAS_VERSION,
        "timestamp": res.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "reference_raster": ref.to_dict(),
        "classified_raster": cls.to_dict(),
        "analysis_window": window.to_dict(),
        "window_is_full_reference": window.is_full(ref),
        "configuration": config.to_dict(),
        "sample": sample.to_dict(),
        "diagnostics": [d.to_dict() for d in res.diagnostics],
    })
    try:
        from qgis.core import Qgis
        res.provenance["qgis_version"] = Qgis.QGIS_VERSION
    except Exception:
        pass
    if categorical:
        res.provenance["reference_census"] = _census_summary(res.reference_census)
        res.provenance["classified_census"] = _census_summary(res.classified_census)
        if weights is not None:
            res.provenance["stratum_weights"] = [float(w) for w in weights]

    for d in res.diagnostics:
        if d.level == "warning":
            res.warnings.append(d.message)
    res.warnings.extend(sample.notes)
    if res.accuracy is not None:
        res.warnings.extend(res.accuracy.warnings)
    if res.continuous is not None:
        res.warnings.extend(res.continuous.warnings)
    _add_power_warnings(res)

    progress(1.0)
    status("Done")
    return res


def _census_summary(census):
    if not census:
        return None
    out = dict((k, v) for k, v in census.items() if k != "counts")
    out["counts"] = dict((str(k), int(v)) for k, v in census["counts"].items())
    return out


def _add_power_warnings(res):
    """Flag sample sizes that cannot support the claims being made."""
    acc = res.accuracy
    if acc is None:
        return
    thin = [acc.labels[i] for i, n in enumerate(acc.n_by_reference_class)
            if 0 < n < 20]
    if thin:
        res.warnings.append(
            "Fewer than 20 reference units in: %s. Congalton & Green (2019) "
            "recommend at least 50 units per class (20-30 for rare classes); "
            "the producer's accuracy of these classes is barely constrained."
            % ", ".join(str(t) for t in thin))
    absent = [acc.labels[i] for i, n in enumerate(acc.n_by_reference_class)
              if n == 0]
    if absent:
        res.warnings.append(
            "No reference unit was drawn for: %s. Their producer's accuracy "
            "and adjusted area cannot be estimated from this sample."
            % ", ".join(str(a) for a in absent))
    if np.isfinite(acc.overall.se):
        moe = acc.overall.margin(res.config.z)
        if moe > 0.05:
            weights = (acc.weights if acc.weights is not None
                       else np.asarray(acc.map_proportions, dtype=float))
            if not np.isfinite(weights).all() or weights.sum() <= 0:
                weights = np.ones(len(acc.labels))
            target = est.sample_size_olofsson(
                weights,
                target_se_oa=0.05 / res.config.z,
                expected_ua=[e.value if np.isfinite(e.value) else 0.5
                             for e in acc.users])
            res.warnings.append(
                "The %.0f%% margin of error on overall accuracy is "
                "+/- %.1f percentage points. About %d units would be needed "
                "for +/- 5 points at the same accuracies (Olofsson et al. "
                "2014, Eq. 13)."
                % (100 * res.config.confidence_level, 100 * moe, target))
