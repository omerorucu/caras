# -*- coding: utf-8 -*-
"""
CARAS core :: probability sampling designs
==========================================

Every design implemented here is a *probability* sample: each pixel of the
target population has a known, positive inclusion probability, which is what
makes the design-based estimators in :mod:`caras.core.estimators` valid.

Target population
-----------------
The population is the set of pixels that (a) lie inside the intersection of
the two rasters and (b) carry valid data in **both** of them.  Sampling
therefore continues until the requested number of usable pairs is reached, and
the realised sample size is reported next to the requested one.  Version 1
drew points first and discarded invalid ones afterwards, so the achieved
sample size - and hence the precision - was uncontrolled.

Fixes over version 1
--------------------
* Sampling is uniform over **pixels**, not over coordinates.  Drawing
  coordinates uniformly is only equal-area in a projected CRS; in a geographic
  CRS it over-samples high latitudes.
* Pixels are drawn **without replacement**, so no location can enter the
  sample twice (pseudo-replication).
* Systematic sampling uses a **random start**, which is what makes it a
  probability design rather than a single deterministic realisation.
* Stratified sampling supports equal, proportional, Neyman and
  Olofsson-floor allocation instead of hard-wired equal allocation, and the
  stratum weights are carried into the estimator.
* Every design accepts a **seed**, and the seed is written into the report.

References
----------
Cochran, W.G. (1977). *Sampling Techniques*, 3rd ed. Wiley.
Olofsson, P. et al. (2014). *RSE*, 148, 42-57 - sections 4 and 5.
Stehman, S.V. (2009). Sampling designs for accuracy assessment of land cover.
    *International Journal of Remote Sensing*, 30, 5243-5272.
"""

from __future__ import division

import math

import numpy as np

from . import raster as rio

__all__ = [
    "SampleSet", "Window", "intersection_window", "simple_random_sample",
    "systematic_sample", "stratified_sample", "sample_from_points",
    "apply_min_separation",
]


class Window(object):
    """Half-open pixel window ``[row0, row1) x [col0, col1)`` on a grid."""

    def __init__(self, row0, row1, col0, col1):
        self.row0 = int(max(0, row0))
        self.row1 = int(row1)
        self.col0 = int(max(0, col0))
        self.col1 = int(col1)

    @property
    def height(self):
        return max(0, self.row1 - self.row0)

    @property
    def width(self):
        return max(0, self.col1 - self.col0)

    @property
    def pixel_count(self):
        return self.height * self.width

    def is_full(self, info):
        return (self.row0 == 0 and self.col0 == 0
                and self.row1 == info.height and self.col1 == info.width)

    def to_dict(self):
        return {"row0": self.row0, "row1": self.row1,
                "col0": self.col0, "col1": self.col1,
                "pixels": self.pixel_count}


class SampleSet(object):
    """A realised probability sample on the reference grid."""

    def __init__(self, rows, cols, xs, ys, method, seed, n_requested,
                 strata=None, stratum_values=None, allocation=None,
                 notes=None, draws=None):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.xs = np.asarray(xs, dtype=np.float64)
        self.ys = np.asarray(ys, dtype=np.float64)
        self.method = method
        self.seed = seed
        self.n_requested = int(n_requested)
        self.strata = None if strata is None else np.asarray(strata)
        self.stratum_values = stratum_values
        self.allocation = allocation
        self.notes = list(notes or [])
        self.draws = draws           # candidate pixels examined

    def __len__(self):
        return int(self.rows.size)

    @property
    def n_achieved(self):
        return len(self)

    def subset(self, mask):
        mask = np.asarray(mask, dtype=bool)
        return SampleSet(
            self.rows[mask], self.cols[mask], self.xs[mask], self.ys[mask],
            self.method, self.seed, self.n_requested,
            None if self.strata is None else self.strata[mask],
            self.stratum_values, self.allocation, self.notes, self.draws)

    def to_dict(self):
        return {
            "method": self.method,
            "seed": self.seed,
            "n_requested": self.n_requested,
            "n_achieved": self.n_achieved,
            "allocation": (None if self.allocation is None
                           else [int(a) for a in self.allocation]),
            "stratum_values": self.stratum_values,
            "candidate_pixels_examined": self.draws,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# window
# ---------------------------------------------------------------------------
def intersection_window(ref, cls):
    """Pixel window on the reference grid covered by both rasters.

    The map raster extent is transformed into the reference CRS first when the
    two differ, so a mixed-CRS pair is handled correctly instead of silently
    producing nonsense pixel indices.
    """
    other = cls.extent
    tr = rio.make_transform(cls.crs, ref.crs)
    if tr is not None:
        try:
            other = tr.transformBoundingBox(cls.extent)
        except Exception:
            other = cls.extent
    inter = ref.extent.intersect(other)
    if inter.isEmpty():
        return Window(0, 0, 0, 0)

    col0 = int(math.floor((inter.xMinimum() - ref.extent.xMinimum()) / ref.res_x))
    col1 = int(math.ceil((inter.xMaximum() - ref.extent.xMinimum()) / ref.res_x))
    row0 = int(math.floor((ref.extent.yMaximum() - inter.yMaximum()) / ref.res_y))
    row1 = int(math.ceil((ref.extent.yMaximum() - inter.yMinimum()) / ref.res_y))
    return Window(max(0, row0), min(ref.height, row1),
                  max(0, col0), min(ref.width, col1))


def _centres(ref, rows, cols):
    return rio.pixel_centres(ref, rows, cols)


# ---------------------------------------------------------------------------
# design 1 :: simple random sampling of pixels
# ---------------------------------------------------------------------------
def simple_random_sample(ref, n, seed=None, window=None, validator=None,
                         max_rounds=40, progress=None):
    """Simple random sample of ``n`` pixels, without replacement.

    ``validator(rows, cols, xs, ys) -> boolean mask`` decides which candidate
    pixels carry usable data in both rasters.  Candidates are drawn in rounds
    until ``n`` valid pixels are collected or the population is exhausted; the
    resulting sample is a simple random sample of the *valid* population,
    which is the population every estimate then refers to.
    """
    window = window or Window(0, ref.height, 0, ref.width)
    if window.pixel_count <= 0:
        raise ValueError("the two rasters have no common area to sample")
    rng = np.random.RandomState(_as_seed(seed))

    n = int(n)
    npix = window.pixel_count
    if n > npix:
        raise ValueError("requested %d units but the common area holds only "
                         "%d pixels" % (n, npix))

    seen = set()
    keep_r, keep_c, keep_x, keep_y = [], [], [], []
    draws = 0
    rate = 1.0
    for round_ in range(max_rounds):
        missing = n - len(keep_r)
        if missing <= 0:
            break
        batch = int(min(npix, max(missing * 2, missing / max(rate, 0.02))))
        batch = int(min(batch, 4 * npix))
        lin = rng.randint(0, npix, size=batch)
        lin = np.unique(lin)
        if seen:
            lin = np.array([v for v in lin.tolist() if v not in seen],
                           dtype=np.int64)
        if lin.size == 0:
            continue
        seen.update(lin.tolist())
        draws += int(lin.size)

        rows = window.row0 + (lin // window.width)
        cols = window.col0 + (lin % window.width)
        xs, ys = _centres(ref, rows, cols)
        if validator is not None:
            ok = validator(rows, cols, xs, ys)
            rows, cols, xs, ys = rows[ok], cols[ok], xs[ok], ys[ok]
        if draws > 0:
            rate = max(len(keep_r) + rows.size, 1) / float(draws)

        take = min(missing, rows.size)
        keep_r.extend(rows[:take].tolist())
        keep_c.extend(cols[:take].tolist())
        keep_x.extend(xs[:take].tolist())
        keep_y.extend(ys[:take].tolist())
        if progress is not None:
            progress(min(1.0, len(keep_r) / float(n)))
        if len(seen) >= npix:
            break

    notes = []
    if len(keep_r) < n:
        notes.append(
            "Only %d of the %d requested units could be placed on pixels that "
            "hold valid data in both rasters (%d candidate pixels examined). "
            "All estimates refer to the valid population and their standard "
            "errors reflect the achieved sample size."
            % (len(keep_r), n, draws))
    return SampleSet(keep_r, keep_c, keep_x, keep_y, "simple_random",
                     _as_seed(seed), n, notes=notes, draws=draws)


# ---------------------------------------------------------------------------
# design 2 :: systematic sampling with a random start
# ---------------------------------------------------------------------------
def systematic_sample(ref, n, seed=None, window=None, validator=None,
                      progress=None):
    """Aligned systematic sample with a random start.

    The grid interval is chosen so that the *full* window would yield roughly
    ``n`` points; the random start makes every pixel's inclusion probability
    equal, which a fixed origin does not.  The realised size differs from the
    request because the grid must have an integer interval - the achieved size
    is reported rather than silently substituted (version 1 returned
    ``floor(sqrt(n))^2`` points without saying so).
    """
    window = window or Window(0, ref.height, 0, ref.width)
    if window.pixel_count <= 0:
        raise ValueError("the two rasters have no common area to sample")
    rng = np.random.RandomState(_as_seed(seed))
    n = int(n)

    step = max(1.0, math.sqrt(window.pixel_count / float(n)))
    step_i = max(1, int(round(step)))
    start_r = int(rng.randint(0, step_i))
    start_c = int(rng.randint(0, step_i))

    rr = np.arange(window.row0 + start_r, window.row1, step_i)
    cc = np.arange(window.col0 + start_c, window.col1, step_i)
    grid_r, grid_c = np.meshgrid(rr, cc, indexing="ij")
    rows = grid_r.ravel()
    cols = grid_c.ravel()
    xs, ys = _centres(ref, rows, cols)
    drawn = rows.size

    if validator is not None:
        ok = validator(rows, cols, xs, ys)
        rows, cols, xs, ys = rows[ok], cols[ok], xs[ok], ys[ok]
    if progress is not None:
        progress(1.0)

    notes = [
        "Systematic grid interval %d px with a random start at (%d, %d); the "
        "grid produced %d candidate locations, of which %d carry valid data in "
        "both rasters." % (step_i, start_r, start_c, drawn, rows.size),
        "No design-unbiased variance estimator exists for a single systematic "
        "sample; standard errors are computed under a simple random sampling "
        "assumption, which is the standard conservative choice.",
    ]
    return SampleSet(rows, cols, xs, ys, "systematic", _as_seed(seed), n,
                     notes=notes, draws=int(drawn))


# ---------------------------------------------------------------------------
# design 3 :: stratified random sampling
# ---------------------------------------------------------------------------
def stratified_sample(grid, strata_defs, allocation, seed=None,
                      window=None, validator=None, chunk_rows=None,
                      progress=None, cancel=None):
    """Stratified random sample drawn by rank selection in a streaming pass.

    Parameters
    ----------
    grid : LayerInfo
        The raster that carries the strata (and the grid the sample lives on).
    strata_defs : list of (stratum_id, member_values, N_h)
        One entry per stratum.  ``member_values`` is the list of raw pixel
        values folded into that stratum by the class mapping, ``N_h`` its pixel
        count from the census.
    allocation : sequence of int
        Units to draw from each stratum, in ``strata_defs`` order.

    Method
    ------
    For stratum ``h`` holding ``N_h`` pixels, ``n_h`` distinct ranks are drawn
    uniformly from ``[0, N_h)``.  A single streaming pass then counts the
    pixels of each stratum and keeps those whose running rank was selected.
    This is an exact simple random sample within every stratum, uses memory
    proportional to the sample rather than to the raster, and never holds the
    pixel coordinates of the whole map in RAM.
    """
    ref = grid
    window = window or Window(0, ref.height, 0, ref.width)
    rng = np.random.RandomState(_as_seed(seed))

    targets = {}
    members = {}
    notes = []
    realised_plan = []
    for (sid, member_values, N_h), n_h in zip(strata_defs, allocation):
        N_h = int(N_h)
        n_h = int(n_h)
        if N_h <= 0:
            realised_plan.append(0)
            continue
        if n_h > N_h:
            notes.append(
                "Stratum %s holds only %d pixels; the allocation of %d units "
                "was reduced accordingly." % (sid, N_h, n_h))
            n_h = N_h
        if n_h <= 0:
            realised_plan.append(0)
            continue
        ranks = rng.choice(N_h, size=n_h, replace=False)
        ranks.sort()
        targets[sid] = ranks
        members[sid] = np.asarray(member_values, dtype=np.float64)
        realised_plan.append(n_h)

    seen = dict((v, 0) for v in targets)
    picked_rows, picked_cols, picked_strata = [], [], []

    kwargs = {}
    if chunk_rows:
        kwargs["chunk_rows"] = chunk_rows
    for row0, arr in rio.iter_blocks(ref, window=window, progress=progress,
                                     **kwargs):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        flat_arr = arr.ravel()
        for sid, ranks in targets.items():
            offset = seen[sid]
            vals = members[sid]
            if vals.size == 1:
                hit = flat_arr == vals[0]
            else:
                hit = np.isin(flat_arr, vals)
            flat = np.flatnonzero(hit)
            if flat.size == 0:
                continue
            lo = np.searchsorted(ranks, offset, side="left")
            hi = np.searchsorted(ranks, offset + flat.size, side="left")
            if hi > lo:
                local = ranks[lo:hi] - offset
                sel = flat[local]
                r = window.row0 + row0 + (sel // window.width)
                c = window.col0 + (sel % window.width)
                picked_rows.extend(r.tolist())
                picked_cols.extend(c.tolist())
                picked_strata.extend([sid] * int(sel.size))
            seen[sid] = offset + int(flat.size)

    rows = np.asarray(picked_rows, dtype=np.int64)
    cols = np.asarray(picked_cols, dtype=np.int64)
    strata = np.asarray(picked_strata)
    xs, ys = _centres(ref, rows, cols)

    if validator is not None and rows.size:
        ok = validator(rows, cols, xs, ys)
        dropped = int((~ok).sum())
        rows, cols, xs, ys, strata = rows[ok], cols[ok], xs[ok], ys[ok], strata[ok]
        if dropped:
            notes.append(
                "%d drawn units fell on pixels without valid data in the other "
                "raster and were removed; the realised allocation therefore "
                "differs from the plan and is the one carried into the "
                "estimator." % dropped)

    notes.append(
        "Stratum weights come from a full census of the stratification raster "
        "over the analysed window, as required by Olofsson et al. (2014); the "
        "estimator reweights the sample back to those proportions.")
    return SampleSet(rows, cols, xs, ys, "stratified", _as_seed(seed),
                     int(np.sum(allocation)), strata=strata,
                     stratum_values=[d[0] for d in strata_defs],
                     allocation=[int(a) for a in realised_plan], notes=notes,
                     draws=int(rows.size))


# ---------------------------------------------------------------------------
# design 4 :: analyst-supplied points
# ---------------------------------------------------------------------------
def sample_from_points(ref, xs, ys, validator=None):
    """Wrap externally supplied coordinates (CSV / field campaign) as a sample.

    No design assumption is made here: the report states that inference rests
    on whatever design produced those points, and the standard errors are
    computed under the design the analyst declares.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    rows = np.empty(xs.size, dtype=np.int64)
    cols = np.empty(xs.size, dtype=np.int64)
    inside = np.zeros(xs.size, dtype=bool)
    for i in range(xs.size):
        rc = rio.coord_to_pixel(ref, xs[i], ys[i])
        if rc is None:
            continue
        rows[i], cols[i] = rc
        inside[i] = True
    rows, cols, xs, ys = rows[inside], cols[inside], xs[inside], ys[inside]
    notes = []
    dropped = int((~inside).sum())
    if dropped:
        notes.append("%d supplied points fall outside the raster and were "
                     "discarded." % dropped)
    if validator is not None and rows.size:
        ok = validator(rows, cols, xs, ys)
        d2 = int((~ok).sum())
        rows, cols, xs, ys = rows[ok], cols[ok], xs[ok], ys[ok]
        if d2:
            notes.append("%d supplied points fall on NoData and were "
                         "discarded." % d2)
    notes.append(
        "Inference from supplied points is valid only under the design that "
        "generated them. Declare that design; if the points were not selected "
        "by a probability rule, the standard errors below are descriptive "
        "rather than inferential.")
    return SampleSet(rows, cols, xs, ys, "supplied_points", None,
                     int(xs.size + dropped), notes=notes, draws=None)


# ---------------------------------------------------------------------------
# post-processing
# ---------------------------------------------------------------------------
def apply_min_separation(sample, min_distance, ref=None):
    """Greedily thin the sample so that no two points are closer than a limit.

    Spatial autocorrelation makes neighbouring validation units carry less
    independent information than the nominal sample size implies, and a
    minimum separation distance is the usual field-survey remedy.

    Caveat, reported verbatim: thinning changes the inclusion probabilities of
    the design.  The estimator can no longer be called strictly design
    unbiased, and the standard errors become approximate.  Use it when the
    points must also be visited in the field; prefer a larger sample when the
    goal is purely statistical.
    """
    if min_distance is None or min_distance <= 0 or len(sample) == 0:
        return sample
    xs, ys = sample.xs, sample.ys
    keep = []
    kept_x, kept_y = [], []
    d2 = float(min_distance) ** 2
    for i in range(xs.size):
        if kept_x:
            dx = np.asarray(kept_x) - xs[i]
            dy = np.asarray(kept_y) - ys[i]
            if np.any(dx * dx + dy * dy < d2):
                continue
        keep.append(i)
        kept_x.append(xs[i])
        kept_y.append(ys[i])
    mask = np.zeros(xs.size, dtype=bool)
    mask[keep] = True
    out = sample.subset(mask)
    removed = int(xs.size - mask.sum())
    out.notes = list(sample.notes) + [
        "Minimum separation of %g map units removed %d of %d units. This "
        "alters the inclusion probabilities of the design: the estimates below "
        "are no longer strictly design unbiased and their standard errors are "
        "approximate." % (min_distance, removed, xs.size)]
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_seed(seed):
    if seed is None:
        return None
    return int(seed) % (2 ** 31 - 1)


def _native(v):
    try:
        return v.item()
    except AttributeError:
        return v


def _fmt(v):
    f = float(v)
    return str(int(f)) if f == int(f) else ("%g" % f)
