# -*- coding: utf-8 -*-
"""Reporting tests.

The reporting layer must never require QGIS, must escape user-supplied text,
and must not emit placeholder prose (version 1 shipped the line "Detailed
analysis results are available in the complete report" in every HTML export).
"""

from __future__ import division

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import disagreement as dis      # noqa: E402
from core import estimators as est        # noqa: E402
from core import regression as reg        # noqa: E402
from core import report as rep            # noqa: E402


# ---------------------------------------------------------------------------
# minimal stand-ins for the QGIS-backed objects
# ---------------------------------------------------------------------------
class _CRS(object):
    def __init__(self, authid="EPSG:32636"):
        self._a = authid

    def authid(self):
        return self._a

    def description(self):
        return "WGS 84 / UTM zone 36N"

    def isGeographic(self):
        return False


class _Info(object):
    def __init__(self, name):
        self.name = name
        self.crs = _CRS()
        self.is_geographic = False

    def to_dict(self):
        return {
            "name": self.name, "source": "/data/%s.tif" % self.name,
            "crs": "EPSG:32636", "crs_description": "WGS 84 / UTM 36N",
            "width": 2000, "height": 1500, "band": 1, "band_count": 1,
            "resolution_x": 30.0, "resolution_y": 30.0, "map_units": "meters",
            "data_type": "Byte", "nodata": 255,
        }


class _Diag(object):
    def __init__(self, level, message):
        self.level = level
        self.message = message

    @property
    def is_blocking(self):
        return self.level == "error"

    def to_dict(self):
        return {"level": self.level, "message": self.message}


class _Cfg(object):
    method = "stratified"
    n_points = 400
    seed = 42
    allocation = "proportional"
    strata_source = "map"
    min_separation = 0.0
    bootstrap = 500
    confidence_level = 0.95
    z = est.Z_95
    mode = "categorical"


class _Sample(object):
    def __len__(self):
        return 400


class _Result(object):
    pass


def _make_result(labels=None, continuous=False):
    labels = labels or ["Forest <b>", 'Water "blue"', "Urban & built"]
    counts = np.array([[120.0, 8.0, 4.0],
                       [6.0, 95.0, 9.0],
                       [5.0, 7.0, 146.0]])
    W = np.array([0.55, 0.10, 0.35])
    cats = [1, 2, 3]
    acc = est.stratified_estimate(counts, W, cats, labels,
                                  area_total=125000.0, area_unit="ha")
    r = _Result()
    r.config = _Cfg()
    r.reference_info = _Info("reference_2020")
    r.classified_info = _Info("classified_2020")
    r.diagnostics = [_Diag("info", "The two rasters share CRS, extent and grid."),
                     _Diag("warning", "Pixels are not square.")]
    r.sample = _Sample()
    r.accuracy = acc
    r.disagreement = dis.disagreement(acc.proportions, cats, labels)
    r.continuous = None
    r.warnings = list(acc.warnings) + ["A caveat with <angle> brackets & an ampersand."]
    r.provenance = {
        "caras_version": "2.0.0", "timestamp": "2026-08-30 12:00:00",
        "python": "3.12.0", "numpy": np.__version__, "platform": "test",
        "qgis_version": "3.40.13",
        "analysis_window": {"row0": 0, "row1": 1500, "col0": 0, "col1": 2000,
                            "pixels": 3000000},
        "sample": {"method": "stratified", "seed": 42, "n_requested": 400,
                   "n_achieved": 400, "allocation": [220, 40, 140],
                   "stratum_values": [1, 2, 3], "notes": ["a note"]},
    }
    if continuous:
        rng = np.random.RandomState(4)
        x = rng.uniform(0, 100, 120)
        y = 0.9 * x + 3 + rng.normal(0, 4, 120)
        r.continuous = reg.continuous_agreement(x, y, bootstrap=200, seed=1)
        r.accuracy = None
        r.disagreement = None
        r.config.mode = "continuous"
        r.warnings = list(r.continuous.warnings)
    return r


# ---------------------------------------------------------------------------
def test_text_report_contains_the_required_sections():
    txt = rep.text_report(_make_result())
    for needle in ["CARAS", "PRE-FLIGHT DIAGNOSTICS", "SAMPLING DESIGN",
                   "ERROR MATRIX - SAMPLE COUNTS",
                   "ERROR MATRIX - ESTIMATED AREA PROPORTIONS",
                   "ACCURACY ESTIMATES", "AREA ESTIMATION",
                   "DISAGREEMENT COMPONENTS", "LIMITATIONS AND CAVEATS",
                   "METHODS PARAGRAPH", "REPRODUCIBILITY", "REFERENCES",
                   "Olofsson"]:
        assert needle in txt, needle
    # confidence intervals must actually be printed, per class and overall
    assert txt.count("+/-") >= 3
    assert txt.count("[") >= 2 * 3 + 3
    # the matrices must carry marginals
    assert "TOTAL" in txt
    # section numbers must be unique
    import re
    heads = re.findall(r"^(\d+)\. [A-Z]", txt, re.M)
    assert len(heads) == len(set(heads)), heads


def test_text_report_states_matrix_orientation():
    txt = rep.text_report(_make_result())
    assert "Rows = MAP (classified) class, columns = REFERENCE class." in txt


def test_kappa_is_reported_with_the_caution_and_no_benchmark_scale():
    txt = rep.text_report(_make_result())
    assert "Cohen's kappa" in txt
    assert "Pontius" in txt
    for banned in ["Almost Perfect", "Substantial", "Slight", "Landis & Koch "
                   "scale"]:
        assert banned not in txt, banned


def test_html_escapes_user_strings_and_has_no_placeholder():
    html = rep.html_report(_make_result())
    assert "<b>" not in html.replace("&lt;b&gt;", "")
    assert "&lt;b&gt;" in html
    assert "&amp;" in html
    assert "Detailed analysis results are available" not in html
    assert html.strip().startswith("<!DOCTYPE html>")
    assert html.strip().endswith("</html>")


def test_json_is_valid_and_carries_provenance():
    data = json.loads(rep.json_report(_make_result()))
    assert data["software"]["version"] == "2.0.0"
    assert data["provenance"]["sample"]["seed"] == 42
    cat = data["categorical"]
    assert cat["matrix_orientation"].startswith("rows = map")
    assert "ci_lower" in cat["overall_accuracy"]
    assert len(cat["users_accuracy"]) == 3
    assert cat["stratum_weights"] is not None
    assert "disagreement" in cat
    assert "methods_paragraph" in data
    assert len(data["references"]) >= 8
    # every area row must expose an interval
    for row in cat["area_estimation"]:
        assert "area_ci_lower" in row and "area_ci_upper" in row


def test_json_handles_non_finite_values():
    r = _make_result()
    r.accuracy.users[1] = est.Estimate(float("nan"), float("nan"), n=0)
    text = rep.json_report(r)
    assert "NaN" not in text
    json.loads(text)


def test_csv_matrix_round_trip():
    csv = rep.matrix_csv(_make_result())
    lines = [l for l in csv.splitlines() if l and not l.startswith("#")]
    header = lines[0].split(",")
    assert header[0] == "map_class" and header[-1] == "row_total"
    # a label containing a comma or quote must be quoted
    assert '"Water ""blue"""' in csv
    assert "Section 2" in csv
    assert "Section 3" in csv


def test_continuous_report():
    r = _make_result(continuous=True)
    txt = rep.text_report(r)
    assert "CONTINUOUS AGREEMENT STATISTICS" in txt
    assert "Coefficient of determination" in txt
    assert "Squared Pearson correlation" in txt
    assert "Bootstrap" in txt
    html = rep.html_report(r)
    assert "Continuous agreement statistics" in html
    data = json.loads(rep.json_report(r))
    assert "continuous" in data


def test_methods_paragraph_mentions_seed_and_design():
    para = rep.methods_paragraph(_make_result())
    assert "seed 42" in para
    assert "stratified" in para
    assert "Olofsson" in para
