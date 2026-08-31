# -*- coding: utf-8 -*-
"""
End-to-end integration test inside a real QGIS environment.

Unlike the rest of the suite this file needs QGIS and GDAL, so it is skipped by
``run_tests.py`` when they are absent.  Run it from a QGIS Python console or
from the shell::

    "C:/Program Files/QGIS 3.40.13/bin/python-qgis-ltr.bat" tests/test_qgis_integration.py
    /usr/bin/python3 tests/test_qgis_integration.py        # Linux, QGIS on PATH

It builds synthetic rasters with a known truth, runs every sampling design
through the full pipeline, and checks the invariants that a QGIS-free unit test
cannot reach: streaming block reads, the census, the pixel-centre geometry, the
CRS transform path, and the reports.
"""

from __future__ import division, print_function

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QGIS_AVAILABLE = True
try:
    from qgis.core import QgsApplication, QgsRasterLayer
    from osgeo import gdal, osr
except ImportError:                                    # pragma: no cover
    QGIS_AVAILABLE = False

WIDTH, HEIGHT = 300, 220
PIXEL = 30.0
ORIGIN_X, ORIGIN_Y = 400000.0, 4200000.0
NODATA_THEMATIC = 255


# ---------------------------------------------------------------------------
# synthetic data with a known truth
# ---------------------------------------------------------------------------
def _write_tif(path, array, epsg, nodata, dtype):
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, array.shape[1], array.shape[0], 1, dtype)
    ds.SetGeoTransform([ORIGIN_X, PIXEL, 0.0, ORIGIN_Y, 0.0, -PIXEL])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array)
    band.FlushCache()
    ds = None
    return path


def make_thematic_pair(folder, seed=7):
    """Reference with a deliberately rare class, plus a map with known errors.

    Class 3 covers about 2 % of the reference. A stratified design with a floor
    is the only way to say anything about it, and an unweighted estimator would
    over-state its influence on overall accuracy - which is exactly what the
    integration test then checks.
    """
    rng = np.random.RandomState(seed)
    ref = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    ref[: HEIGHT // 2, :] = 1                       # class 1, ~50 %
    ref[HEIGHT // 2:, :] = 2                        # class 2, ~48 %
    ref[10:24, 10:110] = 3                          # class 3, rare
    ref[:12, :12] = NODATA_THEMATIC                 # a NoData corner

    cls = ref.copy()
    valid = ref != NODATA_THEMATIC
    # 8 % of class 1 confused with class 2, 5 % the other way,
    # and half of the rare class missed as class 2 (an omission error)
    flip12 = valid & (ref == 1) & (rng.random_sample(ref.shape) < 0.08)
    flip21 = valid & (ref == 2) & (rng.random_sample(ref.shape) < 0.05)
    miss3 = valid & (ref == 3) & (rng.random_sample(ref.shape) < 0.50)
    cls[flip12] = 2
    cls[flip21] = 1
    cls[miss3] = 2
    # and a few false positives of the rare class inside class 2
    false3 = valid & (ref == 2) & (rng.random_sample(ref.shape) < 0.002)
    cls[false3] = 3

    ref_path = _write_tif(os.path.join(folder, "reference.tif"), ref, 32636,
                          NODATA_THEMATIC, gdal.GDT_Byte)
    cls_path = _write_tif(os.path.join(folder, "classified.tif"), cls, 32636,
                          NODATA_THEMATIC, gdal.GDT_Byte)
    return ref_path, cls_path, ref, cls


def make_continuous_pair(folder, seed=11):
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    truth = 20.0 + 0.05 * xx + 0.03 * yy + rng.normal(0, 1.5, (HEIGHT, WIDTH))
    model = 5.0 + 0.75 * truth + rng.normal(0, 2.0, (HEIGHT, WIDTH))
    a = _write_tif(os.path.join(folder, "truth.tif"), truth.astype(np.float32),
                   32636, -9999.0, gdal.GDT_Float32)
    b = _write_tif(os.path.join(folder, "model.tif"), model.astype(np.float32),
                   32636, -9999.0, gdal.GDT_Float32)
    return a, b


# ---------------------------------------------------------------------------
# the test body
# ---------------------------------------------------------------------------
def run(verbose=True):
    from core import analysis as ana
    from core import raster as rio
    from core import report as rpt
    from core import sampling as smp

    folder = tempfile.mkdtemp(prefix="caras_it_")
    failures = []

    def check(name, condition, detail=""):
        if condition:
            if verbose:
                print("   ok    %s" % name)
        else:
            failures.append("%s %s" % (name, detail))
            print("   FAIL  %s %s" % (name, detail))

    try:
        ref_path, cls_path, ref_arr, cls_arr = make_thematic_pair(folder)
        ref_layer = QgsRasterLayer(ref_path, "reference")
        cls_layer = QgsRasterLayer(cls_path, "classified")
        check("rasters load", ref_layer.isValid() and cls_layer.isValid())

        ref_info = rio.describe(ref_layer)
        cls_info = rio.describe(cls_layer)

        # -- geometry ---------------------------------------------------
        x, y = rio.pixel_centre(ref_info, 0, 0)
        check("pixel centre, not corner",
              abs(x - (ORIGIN_X + PIXEL / 2)) < 1e-9
              and abs(y - (ORIGIN_Y - PIXEL / 2)) < 1e-9, (x, y))
        check("coord round trip", rio.coord_to_pixel(ref_info, x, y) == (0, 0))

        # -- diagnostics -------------------------------------------------
        diags = rio.check_pair(ref_info, cls_info)
        check("identical grids raise nothing blocking",
              not any(d.is_blocking for d in diags),
              [d.message for d in diags if d.is_blocking])

        # -- census against a direct NumPy count -------------------------
        census = rio.value_census(ref_info)
        truth_valid = ref_arr[ref_arr != NODATA_THEMATIC]
        vals, cnts = np.unique(truth_valid, return_counts=True)
        expected = dict(zip(vals.tolist(), cnts.tolist()))
        got = dict((int(k), int(v)) for k, v in census["counts"].items())
        check("census reproduces a direct NumPy count", got == expected,
              (got, expected))
        check("census excludes NoData",
              census["valid_pixels"] == int(truth_valid.size))

        # -- streaming blocks tile the raster ----------------------------
        seen = 0
        for _row0, block in rio.iter_blocks(ref_info, chunk_rows=37):
            seen += block.shape[0]
        check("blocks tile the raster exactly", seen == HEIGHT, seen)

        # -- windowed census over the intersection -----------------------
        window = smp.intersection_window(ref_info, cls_info)
        check("full overlap gives the full window", window.is_full(ref_info))

        # -- categorical, stratified ------------------------------------
        mapping = ana.ClassMapping({1.0: 1, 2.0: 2, 3.0: 3},
                                   {1: "Class 1", 2: "Class 2", 3: "Class 3"})
        labels = {1: "Class 1", 2: "Class 2", 3: "Class 3"}
        cfg = ana.AnalysisConfig(
            reference_layer=ref_layer, classified_layer=cls_layer,
            mode="categorical", method="stratified", n_points=600, seed=42,
            allocation="olofsson", min_per_stratum=40, strata_source="map",
            reference_mapping=mapping, classified_mapping=mapping,
            category_labels=labels)
        res = ana.run_analysis(cfg)
        acc = res.accuracy
        check("stratified run produced an estimate", acc is not None)
        check("sample size is close to the request",
              abs(len(res.sample) - 600) <= 60, len(res.sample))
        check("rare class actually got units",
              acc.n_by_map_class[2] >= 30, acc.n_by_map_class[2])
        check("weights sum to one", abs(acc.weights.sum() - 1.0) < 1e-12)
        check("proportion matrix sums to one",
              abs(acc.proportions.sum() - 1.0) < 1e-9, acc.proportions.sum())
        check("area proportions sum to one",
              abs(sum(e.value for e in acc.area_proportions) - 1.0) < 1e-9)
        check("overall accuracy has a standard error",
              np.isfinite(acc.overall.se) and acc.overall.se > 0)
        lo, hi = acc.overall.ci()
        check("overall accuracy interval brackets the estimate",
              lo <= acc.overall.value <= hi, (lo, acc.overall.value, hi))

        # the design-based OA must be close to the true pixel-level OA
        valid = (ref_arr != NODATA_THEMATIC)
        true_oa = float((ref_arr[valid] == cls_arr[valid]).mean())
        check("design-based OA is within its own interval of the truth",
              lo - 0.02 <= true_oa <= hi + 0.02,
              (true_oa, lo, hi))

        # the adjusted area of the rare class must beat pixel counting
        areas = acc.areas()
        true_share3 = float((ref_arr[valid] == 3).mean())
        adj3 = areas[2]["adjusted_proportion"]
        map3 = areas[2]["map_proportion"]
        check("adjusted area of the rare class is closer to the truth "
              "than the mapped area",
              abs(adj3 - true_share3) <= abs(map3 - true_share3) + 1e-9,
              (map3, adj3, true_share3))

        # -- disagreement identities on real output ----------------------
        d = res.disagreement
        check("Q + A = 1 - OA",
              abs(d.quantity + d.allocation - d.total) < 1e-12)
        check("E + S = A", abs(d.exchange + d.shift - d.allocation) < 1e-12)

        # -- reproducibility ---------------------------------------------
        res2 = ana.run_analysis(cfg)
        check("same seed gives the same overall accuracy",
              abs(res2.accuracy.overall.value - acc.overall.value) < 1e-15)
        cfg.seed = 43
        res3 = ana.run_analysis(cfg)
        check("a different seed moves the estimate",
              res3.accuracy.overall.value != acc.overall.value)
        cfg.seed = 42

        # -- simple random and systematic --------------------------------
        for method in ("random", "systematic"):
            cfg2 = ana.AnalysisConfig(
                reference_layer=ref_layer, classified_layer=cls_layer,
                mode="categorical", method=method, n_points=500, seed=5,
                reference_mapping=mapping, classified_mapping=mapping,
                category_labels=labels)
            r = ana.run_analysis(cfg2)
            check("%s design runs" % method, r.accuracy is not None)
            check("%s draws no duplicate pixel" % method,
                  len(set(zip(r.sample.rows.tolist(), r.sample.cols.tolist())))
                  == len(r.sample))
            check("%s never lands on NoData" % method,
                  bool(np.all(np.isfinite(r.reference_values))))
            olo, ohi = r.accuracy.overall.ci()
            check("%s interval covers the true OA" % method,
                  olo - 0.03 <= true_oa <= ohi + 0.03, (olo, true_oa, ohi))

        check("simple random hits the requested size",
              abs(len(ana.run_analysis(ana.AnalysisConfig(
                  reference_layer=ref_layer, classified_layer=cls_layer,
                  mode="categorical", method="random", n_points=400, seed=3,
                  reference_mapping=mapping, classified_mapping=mapping,
                  category_labels=labels)).sample) - 400) <= 1)

        # -- reports ------------------------------------------------------
        txt = rpt.text_report(res)
        check("text report is substantial", len(txt) > 4000, len(txt))
        check("text report names the seed", "42" in txt)
        html = rpt.html_report(res)
        check("html is well formed", html.startswith("<!DOCTYPE html>")
              and html.rstrip().endswith("</html>"))
        import json
        payload = json.loads(rpt.json_report(res))
        check("json carries the census",
              "classified_census" in payload["provenance"])
        check("json carries the adjusted areas",
              len(payload["categorical"]["area_estimation"]) == 3)
        csv = rpt.matrix_csv(res)
        check("csv has three sections", csv.count("# Section") == 3)

        # -- continuous mode ---------------------------------------------
        t_path, m_path = make_continuous_pair(folder)
        t_layer = QgsRasterLayer(t_path, "truth")
        m_layer = QgsRasterLayer(m_path, "model")
        cfg3 = ana.AnalysisConfig(
            reference_layer=t_layer, classified_layer=m_layer,
            mode="continuous", method="random", n_points=800, seed=9,
            bootstrap=400)
        rc = ana.run_analysis(cfg3)
        check("continuous run produced statistics", rc.continuous is not None)
        s = rc.continuous.stats
        check("continuous slope recovers the simulated 0.75",
              abs(s["ols_slope"] - 0.75) < 0.08, s["ols_slope"])
        check("continuous mode reports no accuracy table",
              rc.accuracy is None)
        check("bootstrap interval brackets RMSE",
              rc.continuous.bootstrap["rmse"]["ci_lower"] <= s["rmse"]
              <= rc.continuous.bootstrap["rmse"]["ci_upper"])
        ctxt = rpt.text_report(rc)
        check("continuous report has no error matrix",
              "ERROR MATRIX" not in ctxt)

        # -- mixed CRS ----------------------------------------------------
        wgs = os.path.join(folder, "classified_4326.tif")
        gdal.Warp(wgs, cls_path, dstSRS="EPSG:4326",
                  resampleAlg="near", dstNodata=NODATA_THEMATIC)
        wgs_layer = QgsRasterLayer(wgs, "classified_wgs84")
        if wgs_layer.isValid():
            wgs_info = rio.describe(wgs_layer)
            dd = rio.check_pair(ref_info, wgs_info)
            codes = [x.code for x in dd]
            check("mixed CRS is flagged, not ignored", "crs_mismatch" in codes,
                  codes)
            check("mixed CRS still finds the overlap",
                  smp.intersection_window(ref_info, wgs_info).pixel_count > 0)
            cfg4 = ana.AnalysisConfig(
                reference_layer=ref_layer, classified_layer=wgs_layer,
                mode="categorical", method="random", n_points=300, seed=4,
                reference_mapping=mapping, classified_mapping=mapping,
                category_labels=labels)
            r4 = ana.run_analysis(cfg4)
            check("mixed CRS run agrees with the same-CRS run",
                  abs(r4.accuracy.overall.value - true_oa) < 0.06,
                  (r4.accuracy.overall.value, true_oa))

    finally:
        shutil.rmtree(folder, ignore_errors=True)

    return failures


def test_qgis_integration():
    if not QGIS_AVAILABLE:                             # pragma: no cover
        print("QGIS or GDAL not importable - integration test skipped")
        return
    failures = run(verbose=False)
    assert not failures, "\n".join(failures)


if __name__ == "__main__":                             # pragma: no cover
    if not QGIS_AVAILABLE:
        print("QGIS or GDAL not importable - nothing to do")
        sys.exit(0)
    app = QgsApplication([], False)
    app.initQgis()
    print("CARAS integration test\n" + "=" * 66)
    problems = run()
    print("=" * 66)
    print("%d failure(s)" % len(problems))
    app.exitQgis()
    sys.exit(1 if problems else 0)
