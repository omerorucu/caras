# -*- coding: utf-8 -*-
"""
CARAS core :: raster access, geometry and pair diagnostics
==========================================================

Everything that touches a ``QgsRasterLayer`` lives here so that the
statistical modules stay pure NumPy and can be unit tested without QGIS.

Design decisions
----------------
*Memory.*  Version 1 of CARAS loaded every raster into a full ``float64``
array (a 10 980 x 10 980 Sentinel-2 tile costs ~964 MB that way).  This module
never holds more than one horizontal block at a time: the census that produces
the stratum weights, and the rank-based sample selection, are both single
streaming passes with bounded memory.

*Point queries.*  Scattered value look-ups use
``QgsRasterDataProvider.sample()`` (a windowed GDAL read), which is O(number of
points) instead of O(number of pixels).

*Geometry.*  Coordinates are always pixel **centres**
(``x = xmin + (col + 0.5) * res_x``).  Version 1 emitted pixel corners, which
placed every exported validation point half a pixel away from the cell it
described.

*CRS.*  A reference / map pair in different coordinate reference systems is no
longer silently mis-sampled: the pair is diagnosed up front and, if the analyst
proceeds, sample coordinates are transformed with a proper
``QgsCoordinateTransform`` before the second raster is queried.
"""

from __future__ import division

import math

import numpy as np

from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsPointXY, QgsProject, QgsRasterLayer, QgsRectangle,
                       QgsUnitTypes)

__all__ = [
    "LayerInfo", "Diagnostic", "describe", "iter_blocks", "value_census",
    "sample_values", "pixel_centre", "coord_to_pixel", "check_pair",
    "make_transform", "area_conversions", "DEFAULT_CHUNK_ROWS",
    "MAX_UNIQUE_VALUES",
]

#: Rows read per streaming block.  ~256 rows x 20 000 px x 8 B = ~41 MB peak.
DEFAULT_CHUNK_ROWS = 256

#: Above this many distinct values a raster is treated as continuous.
MAX_UNIQUE_VALUES = 512

# Qgis.DataType numeric codes; stable across QGIS 3.x and 4.x.
_DTYPE_MAP = {
    1: np.uint8, 2: np.uint16, 3: np.int16, 4: np.uint32,
    5: np.int32, 6: np.float32, 7: np.float64, 14: np.int8,
}
_DTYPE_NAMES = {
    1: "Byte", 2: "UInt16", 3: "Int16", 4: "UInt32", 5: "Int32",
    6: "Float32", 7: "Float64", 14: "Int8",
}


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------
class LayerInfo(object):
    """Immutable snapshot of the geometry and metadata of one raster band."""

    def __init__(self, layer, band=1):
        self.layer = layer
        self.band = int(band)
        self.name = layer.name()
        self.source = layer.source()
        self.crs = layer.crs()
        self.extent = QgsRectangle(layer.extent())
        self.width = int(layer.width())
        self.height = int(layer.height())
        self.res_x = float(layer.rasterUnitsPerPixelX())
        self.res_y = float(layer.rasterUnitsPerPixelY())
        provider = layer.dataProvider()
        self.dtype_code = int(provider.dataType(self.band))
        self.dtype_name = _DTYPE_NAMES.get(self.dtype_code, "unknown")
        try:
            self.has_nodata = bool(provider.sourceHasNoDataValue(self.band))
            self.nodata = (float(provider.sourceNoDataValue(self.band))
                           if self.has_nodata else None)
        except Exception:
            self.has_nodata = False
            self.nodata = None
        self.band_count = int(layer.bandCount())

    # -- units ------------------------------------------------------------
    @property
    def is_geographic(self):
        return bool(self.crs.isGeographic())

    @property
    def map_units(self):
        try:
            return QgsUnitTypes.toString(self.crs.mapUnits())
        except Exception:
            return "map units"

    @property
    def pixel_area(self):
        """Area of one pixel in squared map units."""
        return self.res_x * self.res_y

    def to_dict(self):
        return {
            "name": self.name,
            "source": self.source,
            "band": self.band,
            "band_count": self.band_count,
            "crs": self.crs.authid() or self.crs.description(),
            "crs_description": self.crs.description(),
            "is_geographic": self.is_geographic,
            "map_units": self.map_units,
            "width": self.width,
            "height": self.height,
            "pixel_count": self.width * self.height,
            "resolution_x": self.res_x,
            "resolution_y": self.res_y,
            "pixel_area_map_units2": self.pixel_area,
            "data_type": self.dtype_name,
            "nodata": self.nodata,
            "extent": {
                "xmin": self.extent.xMinimum(), "ymin": self.extent.yMinimum(),
                "xmax": self.extent.xMaximum(), "ymax": self.extent.yMaximum(),
            },
        }


class Diagnostic(object):
    """One item of the pre-flight compatibility report."""

    LEVELS = ("info", "warning", "error")

    def __init__(self, level, code, message, detail=None):
        if level not in self.LEVELS:
            raise ValueError("unknown diagnostic level: %s" % level)
        self.level = level
        self.code = code
        self.message = message
        self.detail = detail

    @property
    def is_blocking(self):
        return self.level == "error"

    def to_dict(self):
        return {"level": self.level, "code": self.code,
                "message": self.message, "detail": self.detail}

    def __repr__(self):  # pragma: no cover
        return "[%s] %s" % (self.level.upper(), self.message)


def describe(layer, band=1):
    return LayerInfo(layer, band)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def pixel_centre(info, row, col):
    """Map coordinate of the *centre* of pixel (row, col)."""
    x = info.extent.xMinimum() + (col + 0.5) * info.res_x
    y = info.extent.yMaximum() - (row + 0.5) * info.res_y
    return (x, y)


def pixel_centres(info, rows, cols):
    rows = np.asarray(rows, dtype=float)
    cols = np.asarray(cols, dtype=float)
    x = info.extent.xMinimum() + (cols + 0.5) * info.res_x
    y = info.extent.yMaximum() - (rows + 0.5) * info.res_y
    return x, y


def coord_to_pixel(info, x, y):
    """(row, col) containing the map coordinate, or ``None`` if outside."""
    col = int(math.floor((x - info.extent.xMinimum()) / info.res_x))
    row = int(math.floor((info.extent.yMaximum() - y) / info.res_y))
    if 0 <= col < info.width and 0 <= row < info.height:
        return (row, col)
    return None


def make_transform(src_crs, dst_crs):
    """Coordinate transform, or ``None`` when the two CRS agree."""
    if src_crs == dst_crs:
        return None
    if src_crs.authid() and src_crs.authid() == dst_crs.authid():
        return None
    return QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())


def area_conversions(info, area_map_units2):
    """Convert an area in squared map units to friendly units.

    Returns ``(value, unit_label)``.  For a geographic CRS the conversion is
    refused - squared degrees are not an area - and the caller is expected to
    report pixel counts instead.
    """
    if info.is_geographic:
        return (None, None)
    try:
        factor = QgsUnitTypes.fromUnitToUnitFactor(
            info.crs.mapUnits(), QgsUnitTypes.DistanceMeters)
    except Exception:
        factor = 1.0
    m2 = area_map_units2 * (factor ** 2)
    if m2 >= 1e7:
        return (m2 / 1e6, "km2")
    if m2 >= 1e4:
        return (m2 / 1e4, "ha")
    return (m2, "m2")


# ---------------------------------------------------------------------------
# streaming access
# ---------------------------------------------------------------------------
def iter_blocks(info, chunk_rows=DEFAULT_CHUNK_ROWS, progress=None,
                window=None):
    """Yield ``(row0, array)`` blocks of the band as float64 with NaN NoData.

    ``window`` restricts the traversal to a pixel window; ``row0`` is then
    counted from the top of the *window*, and the array holds only the window
    columns.  ``progress`` is an optional callable receiving a float in [0, 1].
    """
    provider = info.layer.dataProvider()
    np_dtype = _DTYPE_MAP.get(info.dtype_code)
    ymax = info.extent.yMaximum()
    xmin_grid = info.extent.xMinimum()

    if window is None:
        wrow0, wrow1, wcol0, wcol1 = 0, info.height, 0, info.width
    else:
        wrow0, wrow1 = int(window.row0), int(window.row1)
        wcol0, wcol1 = int(window.col0), int(window.col1)
    w_height = max(0, wrow1 - wrow0)
    w_width = max(0, wcol1 - wcol0)
    if w_height == 0 or w_width == 0:
        return

    xmin = xmin_grid + wcol0 * info.res_x
    xmax = xmin_grid + wcol1 * info.res_x

    row0 = 0
    while row0 < w_height:
        nrows = min(int(chunk_rows), w_height - row0)
        top = ymax - (wrow0 + row0) * info.res_y
        bottom = ymax - (wrow0 + row0 + nrows) * info.res_y
        rect = QgsRectangle(xmin, bottom, xmax, top)
        block = provider.block(info.band, rect, w_width, nrows)

        arr = None
        if np_dtype is not None:
            try:
                raw = np.frombuffer(block.data(), dtype=np_dtype)
                if raw.size == w_width * nrows:
                    arr = raw.reshape((nrows, w_width)).astype(np.float64)
            except (ValueError, TypeError):
                arr = None
        if arr is None:                              # safe, slow fallback
            arr = np.empty((nrows, w_width), dtype=np.float64)
            for r in range(nrows):
                for c in range(w_width):
                    arr[r, c] = block.value(r, c)

        if block.hasNoDataValue():
            nd = block.noDataValue()
            if np.isfinite(nd):
                arr[arr == nd] = np.nan

        if progress is not None:
            progress(min(1.0, (row0 + nrows) / float(w_height)))
        yield row0, arr
        row0 += nrows


def value_census(info, chunk_rows=DEFAULT_CHUNK_ROWS,
                 max_unique=MAX_UNIQUE_VALUES, progress=None,
                 cancel=None, window=None):
    """Full census of the band: distinct values, their pixel counts, validity.

    A census - not a sample - is what the stratified estimator needs: the
    stratum weights ``W_h = N_h / N`` must come from the complete map
    (Olofsson et al. 2014, section 4.3).

    Returns a dict with ``counts`` (value -> pixel count, ordered by value),
    ``valid_pixels``, ``total_pixels``, ``nodata_pixels``, ``continuous``
    (True when the number of distinct values exceeded ``max_unique``) and
    ``min``/``max``.
    """
    counts = {}
    valid = 0
    vmin = float("inf")
    vmax = float("-inf")
    continuous = False

    for _row0, arr in iter_blocks(info, chunk_rows, progress, window=window):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        valid += int(finite.size)
        vmin = min(vmin, float(finite.min()))
        vmax = max(vmax, float(finite.max()))
        if continuous:
            continue
        vals, cnts = np.unique(finite, return_counts=True)
        if vals.size > max_unique:
            continuous = True
            counts = {}
            continue
        for v, c in zip(vals.tolist(), cnts.tolist()):
            counts[v] = counts.get(v, 0) + int(c)
        if len(counts) > max_unique:
            continuous = True
            counts = {}

    if window is None:
        total = info.width * info.height
    else:
        total = max(0, window.row1 - window.row0) * max(0, window.col1 - window.col0)
    ordered = dict(sorted(counts.items())) if counts else {}
    return {
        "counts": ordered,
        "values": sorted(ordered.keys()),
        "valid_pixels": valid,
        "total_pixels": total,
        "nodata_pixels": total - valid,
        "continuous": continuous,
        "min": None if vmin == float("inf") else vmin,
        "max": None if vmax == float("-inf") else vmax,
        "distinct_count": len(ordered) if not continuous else None,
    }


def sample_values(info, xs, ys, transform=None):
    """Value of the band at map coordinates, using windowed point reads.

    ``xs``/``ys`` are in the CRS of the *sampling* raster; ``transform`` (a
    ``QgsCoordinateTransform``) is applied before querying when the target
    raster uses a different CRS.  Returns a float64 array with NaN wherever the
    point falls outside the raster or on NoData.
    """
    provider = info.layer.dataProvider()
    n = len(xs)
    out = np.full(n, np.nan, dtype=np.float64)
    nodata = info.nodata
    for i in range(n):
        x, y = float(xs[i]), float(ys[i])
        if transform is not None:
            try:
                pt = transform.transform(QgsPointXY(x, y))
                x, y = pt.x(), pt.y()
            except Exception:
                continue
        if not info.extent.contains(QgsPointXY(x, y)):
            continue
        try:
            val, ok = provider.sample(QgsPointXY(x, y), info.band)
        except Exception:
            val, ok = None, False
        if not ok or val is None:
            continue
        val = float(val)
        if not np.isfinite(val):
            continue
        if nodata is not None and np.isfinite(nodata) and val == nodata:
            continue
        out[i] = val
    return out


# ---------------------------------------------------------------------------
# pair diagnostics
# ---------------------------------------------------------------------------
def check_pair(ref, cls):
    """Pre-flight compatibility check of a reference / map pair.

    Returns a list of :class:`Diagnostic`.  Anything at ``error`` level must be
    resolved before an analysis can be defended; ``warning`` items are recorded
    verbatim in the report so that a reader knows what the numbers rest on.
    """
    out = []

    # --- CRS -------------------------------------------------------------
    same_crs = (ref.crs == cls.crs) or (
        bool(ref.crs.authid()) and ref.crs.authid() == cls.crs.authid())
    if not same_crs:
        out.append(Diagnostic(
            "warning", "crs_mismatch",
            "The two rasters use different coordinate reference systems "
            "(%s vs %s). Sample coordinates will be transformed on the fly; "
            "reprojection introduces a sub-pixel positional error that is not "
            "propagated into the reported standard errors."
            % (ref.crs.authid() or ref.crs.description(),
               cls.crs.authid() or cls.crs.description())))
    if ref.crs.isGeographic() or cls.crs.isGeographic():
        out.append(Diagnostic(
            "warning", "geographic_crs",
            "At least one raster uses a geographic CRS. Pixels then vary in "
            "ground area with latitude, so area estimates are reported in "
            "pixels rather than hectares, and equal-area inference requires an "
            "equal-area projection."))

    # --- overlap ---------------------------------------------------------
    other_extent = cls.extent
    if not same_crs:
        tr = make_transform(cls.crs, ref.crs)
        if tr is not None:
            try:
                other_extent = tr.transformBoundingBox(cls.extent)
            except Exception:
                out.append(Diagnostic(
                    "error", "crs_transform_failed",
                    "The map raster extent could not be transformed into the "
                    "reference CRS. One of the two CRS definitions is invalid."))
    inter = ref.extent.intersect(other_extent)
    if inter.isEmpty():
        out.append(Diagnostic(
            "error", "no_overlap",
            "The two rasters do not overlap in the reference CRS. Check the "
            "CRS assignment of both layers before continuing."))
    else:
        share_ref = inter.area() / ref.extent.area() if ref.extent.area() > 0 else 0.0
        share_cls = inter.area() / other_extent.area() if other_extent.area() > 0 else 0.0
        smallest = min(share_ref, share_cls)
        if smallest < 0.999:
            level = "warning" if smallest > 0.5 else "error"
            out.append(Diagnostic(
                level, "partial_overlap",
                "The rasters overlap over only %.1f%% of the smaller extent. "
                "Sampling is restricted to the intersection, and every "
                "estimate refers to that intersection, not to the full map."
                % (100.0 * smallest)))

    # --- resolution ------------------------------------------------------
    # Pixel sizes are only comparable in the same units, so when the two CRS
    # differ the map raster's resolution is expressed in the reference CRS by
    # dividing its transformed extent by its pixel count.  Comparing 30 metres
    # against 0.00032 degrees as raw numbers would report a mismatch factor of
    # ~94 000 for two rasters that in fact describe the same ground cell.
    cls_res_x, cls_res_y = cls.res_x, cls.res_y
    if not same_crs and not inter.isEmpty() and cls.width and cls.height:
        cls_res_x = other_extent.width() / float(cls.width)
        cls_res_y = other_extent.height() / float(cls.height)
    rx = max(ref.res_x, cls_res_x) / max(min(ref.res_x, cls_res_x), 1e-15)
    ry = max(ref.res_y, cls_res_y) / max(min(ref.res_y, cls_res_y), 1e-15)
    ratio = max(rx, ry)
    if ratio > 1.02:
        level = "warning" if ratio <= 2.0 else "error"
        approx = ("" if same_crs else
                  ", the latter approximated in the reference CRS")
        out.append(Diagnostic(
            level, "resolution_mismatch",
            "Pixel sizes differ by a factor of %.2f (%.4g x %.4g vs "
            "%.4g x %.4g%s). The two maps then describe different spatial "
            "supports; a point-to-point comparison confounds classification "
            "error with the change of support. Resample to a common grid, or "
            "record the mismatch explicitly in the methods section."
            % (ratio, ref.res_x, ref.res_y, cls_res_x, cls_res_y, approx)))
    if abs(ref.res_x - ref.res_y) > 1e-9 * max(ref.res_x, 1.0) or \
            abs(cls.res_x - cls.res_y) > 1e-9 * max(cls.res_x, 1.0):
        out.append(Diagnostic(
            "info", "anisotropic_pixels",
            "Pixels are not square; pixel-count areas remain valid, but any "
            "distance-based option (minimum separation) uses map units."))

    # --- alignment -------------------------------------------------------
    if same_crs and ratio <= 1.001:
        dx = abs((ref.extent.xMinimum() - cls.extent.xMinimum()) / ref.res_x)
        dy = abs((ref.extent.yMaximum() - cls.extent.yMaximum()) / ref.res_y)
        off = max(dx - math.floor(dx), dy - math.floor(dy))
        if min(off, 1.0 - off) > 0.01:
            out.append(Diagnostic(
                "warning", "grid_offset",
                "The two grids are offset by a fraction of a pixel "
                "(%.2f px in x, %.2f px in y). Nearest-cell extraction then "
                "mixes neighbouring cells near class boundaries."
                % (dx - math.floor(dx), dy - math.floor(dy))))

    # --- bands / nodata ---------------------------------------------------
    for tag, info in (("reference", ref), ("map", cls)):
        if info.band_count > 1:
            out.append(Diagnostic(
                "info", "multiband",
                "The %s raster has %d bands; band %d is used."
                % (tag, info.band_count, info.band)))
        if not info.has_nodata:
            out.append(Diagnostic(
                "info", "no_nodata",
                "The %s raster declares no NoData value. Every pixel is "
                "treated as valid data; if the file uses a fill value such as "
                "0 or -9999, declare it in the layer properties first - CARAS "
                "deliberately no longer hard-codes -9999." % tag))

    if not out:
        out.append(Diagnostic("info", "ok",
                              "The two rasters share CRS, extent and grid."))
    return out
