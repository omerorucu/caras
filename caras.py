# -*- coding: utf-8 -*-
"""
CARAS - Classification Accuracy and Regression Assessment Suite
===============================================================

QGIS user interface.  All statistics live in :mod:`caras.core`; this module
only collects the analyst's choices, drives the run and renders the result.

The interface is deliberately ordered as a four-step workflow, because the
scientific content of the tool depends on the steps being taken in order:

    1. Data      - pick the rasters, declare categorical or continuous, and
                   read the pre-flight diagnostics before anything else.
    2. Classes   - fold the raw pixel values of both rasters onto shared
                   categories; the census behind this step also supplies the
                   stratum weights.
    3. Design    - choose a probability sampling design, a seed and, if
                   stratified, an allocation; the sample-size planner says
                   what precision the chosen n can deliver.
    4. Results   - estimates with confidence intervals, adjusted areas,
                   disagreement components, caveats, and exports.

Copyright (C) 2026 Omer K. Orucu.  Licensed under the GNU General Public
License version 3 or (at your option) any later version.
"""

import os
import traceback
from datetime import datetime

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QColor, QIcon
from qgis.PyQt.QtWidgets import (
    QAction, QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

from qgis.core import (QgsCoordinateReferenceSystem,
                       QgsCoordinateTransformContext, QgsFeature, QgsField,
                       QgsGeometry, QgsPointXY, QgsProject, QgsRasterLayer,
                       QgsVectorFileWriter, QgsVectorLayer)

import numpy as np

from .core import analysis as ana
from .core import estimators as est
from .core import raster as rio
from .core import report as rpt
from .core.sampling import intersection_window

PLUGIN_NAME = "CARAS"
VERSION = ana.CARAS_VERSION

_MONO = "Consolas, 'DejaVu Sans Mono', Menlo, monospace"


# ---------------------------------------------------------------------------
# Qt 5 / Qt 6 helpers
# ---------------------------------------------------------------------------
def _exec(dialog):
    return dialog.exec() if hasattr(dialog, "exec") else dialog.exec_()


def _make_field(name, kind):
    """QgsField that works on QGIS 3 (QVariant) and QGIS 4 (QMetaType)."""
    try:
        from qgis.PyQt.QtCore import QVariant
        mapping = {"int": QVariant.Int, "double": QVariant.Double,
                   "string": QVariant.String}
        return QgsField(name, mapping[kind])
    except ImportError:
        from qgis.PyQt.QtCore import QMetaType
        mapping = {"int": QMetaType.Type.Int, "double": QMetaType.Type.Double,
                   "string": QMetaType.Type.QString}
        return QgsField(name, mapping[kind])


def _stretch(header):
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def _bold(widget):
    f = widget.font()
    f.setBold(True)
    widget.setFont(f)
    return widget


def _fmt_value(v):
    f = float(v)
    return str(int(f)) if f == int(f) else ("%g" % f)


# ---------------------------------------------------------------------------
# class mapping dialog
# ---------------------------------------------------------------------------
class ClassMappingDialog(QDialog):
    """Fold the raw pixel values of both rasters onto shared categories.

    The census counts are shown next to every value, because the decision that
    matters - which raw values are worth keeping as separate categories - is
    driven by how much of the map they cover.  Values left unassigned are
    excluded from the target population, and the dialog says how much that is.
    """

    def __init__(self, ref_census, cls_census, parent=None,
                 ref_name="Reference", cls_name="Classified"):
        QDialog.__init__(self, parent)
        self.setWindowTitle("CARAS - class mapping")
        self.resize(1080, 720)
        self.ref_values = list(ref_census["values"])
        self.cls_values = list(cls_census["values"])
        self.ref_counts = dict(ref_census["counts"])
        self.cls_counts = dict(cls_census["counts"])
        self.ref_total = max(1, sum(self.ref_counts.values()))
        self.cls_total = max(1, sum(self.cls_counts.values()))
        self.ref_name = ref_name
        self.cls_name = cls_name
        self.reference_mapping = None
        self.classified_mapping = None
        self.labels = None
        self._build()
        self.auto_map_identical()

    # -- ui ---------------------------------------------------------------
    def _build(self):
        layout = QVBoxLayout(self)
        title = QLabel("Assign the pixel values of both rasters to shared "
                       "analysis categories")
        title.setStyleSheet("font-size:14px;font-weight:600;")
        layout.addWidget(title)

        info = QLabel(
            "Values that mean the same thing must receive the same category "
            "number. Set the category to 0 to exclude a value from the "
            "analysis entirely - excluded pixels leave the target population, "
            "and the report records how much of the map that was. Category "
            "labels are taken from the reference raster where the two "
            "disagree.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#4a5a68;background:#f2f6f8;padding:9px;"
                           "border-radius:4px;")
        layout.addWidget(info)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.ref_table = self._make_table(
            "Reference raster - %s" % self.ref_name, self.ref_values,
            self.ref_counts, self.ref_total, split)
        self.cls_table = self._make_table(
            "Classified raster - %s" % self.cls_name, self.cls_values,
            self.cls_counts, self.cls_total, split)
        layout.addWidget(split, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Quick mapping:"))
        b1 = QPushButton("Identical values share a category")
        b1.setToolTip("Recommended when both rasters use the same legend.")
        b1.clicked.connect(self.auto_map_identical)
        b2 = QPushButton("Sequential (by rank)")
        b2.setToolTip("Match the sorted value lists position by position; use "
                      "only when the two legends are known to be ordered the "
                      "same way.")
        b2.clicked.connect(self.auto_map_sequential)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch()
        layout.addLayout(row)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color:#4a5a68;")
        layout.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_table(self, caption, values, counts, total, parent):
        box = QGroupBox(caption, parent)
        lay = QVBoxLayout(box)
        table = QTableWidget(len(values), 5)
        table.setHorizontalHeaderLabels(
            ["Pixel value", "Pixels", "% of raster", "Label", "Category"])
        for i, val in enumerate(values):
            item = QTableWidgetItem(_fmt_value(val))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(i, 0, item)

            cnt = int(counts.get(val, 0))
            c_item = QTableWidgetItem("{:,}".format(cnt))
            c_item.setFlags(c_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            c_item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(i, 1, c_item)

            share = 100.0 * cnt / total
            p_item = QTableWidgetItem("%.3f" % share)
            p_item.setFlags(p_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                    | Qt.AlignmentFlag.AlignVCenter)
            if share < 0.5:
                p_item.setForeground(QColor("#8a5a00"))
                p_item.setToolTip("Rare class: a stratified design with a "
                                  "floor of 20-50 units is usually needed to "
                                  "estimate its accuracy at all.")
            table.setItem(i, 2, p_item)

            table.setCellWidget(i, 3, QLineEdit("Class %s" % _fmt_value(val)))
            spin = QSpinBox()
            spin.setRange(0, 999)
            spin.setSpecialValueText("exclude")
            spin.valueChanged.connect(self._refresh_summary)
            table.setCellWidget(i, 4, spin)
        _stretch(table.horizontalHeader())
        lay.addWidget(table)
        return table

    # -- quick mappings ---------------------------------------------------
    def auto_map_identical(self):
        registry = {}
        counter = [1]

        def cat_for(v):
            if v not in registry:
                registry[v] = counter[0]
                counter[0] += 1
            return registry[v]

        for table, values in ((self.ref_table, self.ref_values),
                              (self.cls_table, self.cls_values)):
            for i, v in enumerate(values):
                table.cellWidget(i, 4).setValue(cat_for(v))
        self._refresh_summary()

    def auto_map_sequential(self):
        for table, values in ((self.ref_table, self.ref_values),
                              (self.cls_table, self.cls_values)):
            for i in range(len(values)):
                table.cellWidget(i, 4).setValue(i + 1)
        self._refresh_summary()

    # -- result -----------------------------------------------------------
    def _collect(self, table, values, counts, total):
        mapping, labels, excluded = {}, {}, 0
        for i, v in enumerate(values):
            cat = table.cellWidget(i, 4).value()
            if cat == 0:
                excluded += int(counts.get(v, 0))
                continue
            mapping[v] = cat
            text = table.cellWidget(i, 3).text().strip()
            if cat not in labels and text:
                labels[cat] = text
        return mapping, labels, 100.0 * excluded / total

    def _refresh_summary(self):
        try:
            rm, _rl, rex = self._collect(self.ref_table, self.ref_values,
                                         self.ref_counts, self.ref_total)
            cm, _cl, cex = self._collect(self.cls_table, self.cls_values,
                                         self.cls_counts, self.cls_total)
        except Exception:
            return
        cats = sorted(set(list(rm.values()) + list(cm.values())))
        missing = [c for c in cats
                   if c not in set(rm.values()) or c not in set(cm.values())]
        msg = ("%d categories defined. Excluded from the population: %.3f %% "
               "of the reference raster, %.3f %% of the classified raster."
               % (len(cats), rex, cex))
        if missing:
            msg += ("  Categories missing from one raster: %s. That is legal - "
                    "their row or column of the matrix will simply be empty."
                    % ", ".join(str(c) for c in missing))
        self.summary.setText(msg)

    def _accept(self):
        rm, rl, _rex = self._collect(self.ref_table, self.ref_values,
                                     self.ref_counts, self.ref_total)
        cm, cl, _cex = self._collect(self.cls_table, self.cls_values,
                                     self.cls_counts, self.cls_total)
        if not rm or not cm:
            QMessageBox.warning(self, PLUGIN_NAME,
                                "At least one category must remain in each "
                                "raster.")
            return
        labels = {}
        for cat in sorted(set(list(rm.values()) + list(cm.values()))):
            labels[cat] = rl.get(cat) or cl.get(cat) or ("Category %d" % cat)
        seen = {}
        for cat in sorted(labels):
            name = labels[cat]
            if name in seen:
                QMessageBox.warning(
                    self, PLUGIN_NAME,
                    "Categories %d and %d share the label '%s'. Distinct "
                    "labels are required so that per-class results cannot be "
                    "confused." % (seen[name], cat, name))
                return
            seen[name] = cat
        self.reference_mapping = ana.ClassMapping(rm, rl)
        self.classified_mapping = ana.ClassMapping(cm, cl)
        self.labels = labels
        self.accept()


# ---------------------------------------------------------------------------
# sample size planner
# ---------------------------------------------------------------------------
class SampleSizeDialog(QDialog):
    """Olofsson et al. (2014) Eq. 13, exposed as an interactive planner."""

    def __init__(self, weights, labels, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle("CARAS - sample size planner")
        self.resize(700, 540)
        self.weights = np.asarray(weights, dtype=float)
        self.labels = list(labels)
        self.chosen = None
        self.n = 0
        self._build()
        self._recompute()

    def _build(self):
        lay = QVBoxLayout(self)
        note = QLabel(
            "The total sample size follows from the precision you want on "
            "overall accuracy: n = (sum_i W_i S_i / S(O))^2 with "
            "S_i = sqrt(U_i (1 - U_i)) (Olofsson et al. 2014, Eq. 13). Enter "
            "the user's accuracy you expect for each class; 0.5 is the safe, "
            "maximum-variance guess.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5a68;background:#f2f6f8;padding:9px;"
                           "border-radius:4px;")
        lay.addWidget(note)

        form = QFormLayout()
        self.target = QDoubleSpinBox()
        self.target.setRange(0.1, 25.0)
        self.target.setValue(2.0)
        self.target.setSuffix(" percentage points")
        self.target.setDecimals(1)
        self.target.setSingleStep(0.5)
        self.target.valueChanged.connect(self._recompute)
        form.addRow("Target margin of error (95 %) on overall accuracy:",
                    self.target)
        lay.addLayout(form)

        self.table = QTableWidget(len(self.labels), 3)
        self.table.setHorizontalHeaderLabels(
            ["Class", "Area share W", "Expected user's accuracy"])
        for i, lab in enumerate(self.labels):
            item = QTableWidgetItem(str(lab))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 0, item)
            w = QTableWidgetItem("%.6f" % self.weights[i])
            w.setFlags(w.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 1, w)
            spin = QDoubleSpinBox()
            spin.setRange(0.05, 0.999)
            spin.setDecimals(2)
            spin.setSingleStep(0.05)
            spin.setValue(0.80)
            spin.valueChanged.connect(self._recompute)
            self.table.setCellWidget(i, 2, spin)
        _stretch(self.table.horizontalHeader())
        lay.addWidget(self.table, 1)

        self.result = QLabel("")
        self.result.setWordWrap(True)
        self.result.setStyleSheet("font-size:13px;padding:8px;"
                                  "background:#eef4f7;border-radius:4px;")
        lay.addWidget(self.result)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Use this sample size")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _expected(self):
        return [self.table.cellWidget(i, 2).value()
                for i in range(len(self.labels))]

    def _recompute(self):
        se = (self.target.value() / 100.0) / est.Z_95
        try:
            n = est.sample_size_olofsson(self.weights, se, self._expected())
            alloc_p = est.allocate_sample(self.weights, n, "proportional")
            alloc_n = est.allocate_sample(self.weights, n, "neyman",
                                          expected_ua=self._expected())
        except Exception as exc:
            self.result.setText(str(exc))
            return
        self.n = n
        thin = [self.labels[i] for i in range(len(self.labels))
                if alloc_p[i] < 20]
        text = ("<b>%d validation units</b> are needed for a +/- %.1f point "
                "margin of error on overall accuracy.<br>Proportional "
                "allocation: %s<br>Neyman allocation: %s"
                % (n, self.target.value(),
                   ", ".join("%s=%d" % (self.labels[i], alloc_p[i])
                             for i in range(len(self.labels))),
                   ", ".join("%s=%d" % (self.labels[i], alloc_n[i])
                             for i in range(len(self.labels)))))
        if thin:
            text += ("<br><span style='color:#8a5a00'>Proportional allocation "
                     "leaves fewer than 20 units in: %s. Raise the per-stratum "
                     "floor if the accuracy of those classes matters.</span>"
                     % ", ".join(str(t) for t in thin))
        self.result.setText(text)

    def _accept(self):
        self.chosen = self.n
        self.accept()


# ---------------------------------------------------------------------------
# main dialog
# ---------------------------------------------------------------------------
class CARASDialog(QDialog):

    def __init__(self, iface, parent=None):
        QDialog.__init__(self, parent)
        self.iface = iface
        self.setWindowTitle("%s %s - Classification Accuracy and Regression "
                            "Assessment Suite" % (PLUGIN_NAME, VERSION))
        self.resize(1180, 860)

        self.ref_info = None
        self.cls_info = None
        self.ref_census = None
        self.cls_census = None
        self.reference_mapping = None
        self.classified_mapping = None
        self.category_labels = None
        self.result = None
        self._export_buttons = []

        self._build()
        self.reload_layers()

    # -- construction -----------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)

        header = QLabel("CARAS %s - design-based accuracy, area and agreement "
                        "assessment" % VERSION)
        header.setStyleSheet("font-size:16px;font-weight:600;color:#1f6f8b;")
        root.addWidget(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._tab_data(), "1. Data")
        self.tabs.addTab(self._tab_classes(), "2. Classes")
        self.tabs.addTab(self._tab_design(), "3. Design")
        self.tabs.addTab(self._tab_results(), "4. Results")
        for i in (1, 2, 3):
            self.tabs.setTabEnabled(i, False)
        root.addWidget(self.tabs, 1)

        bar = QHBoxLayout()
        self.status = QLabel("")
        self.status.setStyleSheet("color:#4a5a68;")
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bar.addWidget(self.status, 1)
        bar.addWidget(self.progress, 2)

        self.run_button = QPushButton("Run analysis")
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet(
            "QPushButton{background:#1f6f8b;color:white;font-weight:600;"
            "padding:7px 18px;border-radius:3px;}"
            "QPushButton:disabled{background:#adbcc6;}")
        self.run_button.clicked.connect(self.run_analysis)
        bar.addWidget(self.run_button)

        close = QPushButton("Close")
        close.clicked.connect(self.reject)
        bar.addWidget(close)
        root.addLayout(bar)

    # -- tab 1 ------------------------------------------------------------
    def _tab_data(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        mode_box = QGroupBox("Measurement scale of the data")
        ml = QVBoxLayout(mode_box)
        self.mode_categorical = QRadioButton(
            "Categorical (thematic classes) - accuracy, adjusted area and "
            "disagreement")
        self.mode_categorical.setChecked(True)
        self.mode_continuous = QRadioButton(
            "Continuous (biomass, cover, LST, model output) - RMSE, bias, "
            "agreement")
        for b in (self.mode_categorical, self.mode_continuous):
            b.toggled.connect(self._mode_changed)
            ml.addWidget(b)
        note = QLabel(
            "This choice is enforced. RMSE, R-squared and bias computed on "
            "nominal class codes depend entirely on the arbitrary integers "
            "used as labels, so CARAS refuses to report them for categorical "
            "data - a defect of version 1 that this release removes.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a5a00;")
        ml.addWidget(note)
        lay.addWidget(mode_box)

        maps = QGroupBox("Rasters")
        form = QFormLayout(maps)
        self.ref_combo, ref_row, self.ref_band = self._layer_row()
        self.cls_combo, cls_row, self.cls_band = self._layer_row()
        form.addRow("Reference map (ground truth):", ref_row)
        form.addRow("Classified map (being assessed):", cls_row)
        refresh = QPushButton("Reload layer list")
        refresh.clicked.connect(self.reload_layers)
        inspect = QPushButton("Inspect rasters and continue")
        inspect.setStyleSheet("font-weight:600;padding:5px 14px;")
        inspect.clicked.connect(self.inspect_rasters)
        row = QHBoxLayout()
        row.addWidget(refresh)
        row.addStretch()
        row.addWidget(inspect)
        form.addRow("", self._wrap(row))
        lay.addWidget(maps)

        diag = QGroupBox("Pre-flight diagnostics")
        dl = QVBoxLayout(diag)
        self.diag_text = QTextEdit()
        self.diag_text.setReadOnly(True)
        self.diag_text.setStyleSheet("font-family:%s;font-size:12px;" % _MONO)
        dl.addWidget(self.diag_text)
        lay.addWidget(diag, 1)
        return w

    def _layer_row(self):
        combo = QComboBox()
        combo.setMinimumWidth(380)
        browse = QPushButton("...")
        browse.setFixedWidth(34)
        browse.setToolTip("Open a raster file from disk")
        browse.clicked.connect(lambda: self.browse_raster(combo))
        band = QSpinBox()
        band.setRange(1, 999)
        band.setPrefix("band ")
        band.setFixedWidth(90)
        row = QHBoxLayout()
        row.addWidget(combo, 1)
        row.addWidget(browse)
        row.addWidget(band)
        return combo, self._wrap(row), band

    @staticmethod
    def _wrap(layout):
        w = QWidget()
        w.setLayout(layout)
        return w

    # -- tab 2 ------------------------------------------------------------
    def _tab_classes(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        note = QLabel(
            "The census below is a complete pass over both rasters inside the "
            "common area. It supplies the class inventory, the uncorrected map "
            "areas, and - for a stratified design - the stratum weights that "
            "the estimator needs.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5a68;background:#f2f6f8;padding:9px;"
                           "border-radius:4px;")
        lay.addWidget(note)

        self.census_text = QTextEdit()
        self.census_text.setReadOnly(True)
        self.census_text.setStyleSheet("font-family:%s;font-size:12px;" % _MONO)
        lay.addWidget(self.census_text, 1)

        row = QHBoxLayout()
        self.map_button = QPushButton("Edit class mapping...")
        self.map_button.clicked.connect(self.edit_mapping)
        row.addWidget(self.map_button)
        row.addStretch()
        lay.addLayout(row)

        self.mapping_text = QTextEdit()
        self.mapping_text.setReadOnly(True)
        self.mapping_text.setMaximumHeight(190)
        self.mapping_text.setStyleSheet("font-family:%s;font-size:12px;" % _MONO)
        lay.addWidget(self.mapping_text)
        return w

    # -- tab 3 ------------------------------------------------------------
    def _tab_design(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        w = QWidget()
        lay = QVBoxLayout(w)

        design = QGroupBox("Sampling design")
        dl = QVBoxLayout(design)
        self.method_group = QButtonGroup(self)
        self.rb_random = QRadioButton(
            "Simple random - equal inclusion probability for every pixel")
        self.rb_random.setChecked(True)
        self.rb_systematic = QRadioButton(
            "Systematic with a random start - even spatial coverage")
        self.rb_stratified = QRadioButton(
            "Stratified random - required if rare classes matter (recommended)")
        self.rb_points = QRadioButton("Use my own points from a CSV file")
        for i, b in enumerate((self.rb_random, self.rb_systematic,
                               self.rb_stratified, self.rb_points), start=1):
            self.method_group.addButton(b, i)
            b.toggled.connect(self._method_changed)
            dl.addWidget(b)
        lay.addWidget(design)

        common = QGroupBox("Sample")
        cf = QFormLayout(common)
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 2000000)
        self.n_points.setValue(500)
        self.n_points.setSingleStep(50)
        size_row = QHBoxLayout()
        size_row.addWidget(self.n_points)
        self.size_button = QPushButton("Sample size planner...")
        self.size_button.clicked.connect(self.open_size_planner)
        size_row.addWidget(self.size_button)
        size_row.addStretch()
        cf.addRow("Number of validation units:", self._wrap(size_row))

        self.seed = QSpinBox()
        self.seed.setRange(0, 2 ** 31 - 2)
        self.seed.setValue(42)
        self.seed_check = QCheckBox("fix the seed (reproducible sample)")
        self.seed_check.setChecked(True)
        self.seed_check.toggled.connect(self.seed.setEnabled)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.seed)
        seed_row.addWidget(self.seed_check)
        seed_row.addStretch()
        cf.addRow("Random seed:", self._wrap(seed_row))

        self.min_sep = QDoubleSpinBox()
        self.min_sep.setRange(0.0, 1e9)
        self.min_sep.setDecimals(2)
        self.min_sep.setValue(0.0)
        self.min_sep.setSuffix(" map units")
        self.min_sep.setToolTip(
            "Thinning the sample changes the inclusion probabilities; the "
            "report says so wherever it is used. 0 disables it.")
        cf.addRow("Minimum separation between units:", self.min_sep)

        self.conf_level = QComboBox()
        self.conf_level.addItem("95 %", 0.95)
        self.conf_level.addItem("90 %", 0.90)
        self.conf_level.addItem("99 %", 0.99)
        cf.addRow("Confidence level:", self.conf_level)
        lay.addWidget(common)

        self.strat_box = QGroupBox("Stratification")
        sf = QFormLayout(self.strat_box)
        self.strata_source = QComboBox()
        self.strata_source.addItem(
            "Classified map (standard, Olofsson et al. 2014)", "map")
        self.strata_source.addItem("Reference map", "reference")
        sf.addRow("Strata come from:", self.strata_source)
        self.allocation = QComboBox()
        for label, key in (("Proportional (self-weighting)", "proportional"),
                           ("Proportional with a floor for rare classes",
                            "olofsson"),
                           ("Neyman / optimal for overall accuracy", "neyman"),
                           ("Equal per stratum", "equal")):
            self.allocation.addItem(label, key)
        sf.addRow("Allocation:", self.allocation)
        self.min_per_stratum = QSpinBox()
        self.min_per_stratum.setRange(0, 100000)
        self.min_per_stratum.setValue(30)
        sf.addRow("Minimum units per stratum:", self.min_per_stratum)
        hint = QLabel(
            "Stratum weights are taken from the census, and the estimator "
            "reweights the sample back to the map proportions (Olofsson et al. "
            "2014, Eqs. 4-10). Without that step a stratified sample "
            "over-represents rare classes and every accuracy figure is biased.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#4a5a68;")
        sf.addRow("", hint)
        self.strat_box.setEnabled(False)
        lay.addWidget(self.strat_box)

        self.csv_box = QGroupBox("Validation points from a CSV file")
        pf = QFormLayout(self.csv_box)
        self.csv_path = QLineEdit()
        self.csv_path.setReadOnly(True)
        pick = QPushButton("Browse...")
        pick.clicked.connect(self.browse_csv)
        crow = QHBoxLayout()
        crow.addWidget(self.csv_path, 1)
        crow.addWidget(pick)
        pf.addRow("File (id,x,y,reference_value):", self._wrap(crow))
        self.csv_crs = QComboBox()
        self.csv_crs.addItem("WGS 84 (EPSG:4326)", "EPSG:4326")
        self.csv_crs.addItem("Same as the reference raster", "layer")
        pf.addRow("Coordinates are in:", self.csv_crs)
        self.csv_use_values = QCheckBox(
            "Use the reference_value column instead of the reference raster")
        self.csv_use_values.setChecked(True)
        pf.addRow("", self.csv_use_values)
        self.declared_design = QComboBox()
        self.declared_design.addItem("Simple random / equal probability", "srs")
        self.declared_design.addItem("Systematic", "systematic")
        pf.addRow("Design that produced the points:", self.declared_design)
        warn = QLabel(
            "Standard errors are only inferential if the points were drawn by "
            "a probability rule. Purposive or convenience points give "
            "descriptive numbers, and the report states that.")
        warn.setWordWrap(True)
        warn.setStyleSheet("color:#8a5a00;")
        pf.addRow("", warn)
        self.csv_box.setEnabled(False)
        lay.addWidget(self.csv_box)

        self.cont_box = QGroupBox("Continuous mode")
        cl = QFormLayout(self.cont_box)
        self.bootstrap = QSpinBox()
        self.bootstrap.setRange(0, 100000)
        self.bootstrap.setValue(2000)
        self.bootstrap.setSingleStep(500)
        cl.addRow("Bootstrap replicates for confidence intervals:",
                  self.bootstrap)
        self.cont_box.setEnabled(False)
        lay.addWidget(self.cont_box)

        lay.addStretch()
        outer.setWidget(w)
        return outer

    # -- tab 4 ------------------------------------------------------------
    def _tab_results(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.result_tabs = QTabWidget()

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.summary_text.setStyleSheet("font-family:%s;font-size:12px;" % _MONO)
        self.result_tabs.addTab(self.summary_text, "Full report")

        self.matrix_table = QTableWidget()
        self.result_tabs.addTab(self._panel(
            self.matrix_table,
            "Rows = map (classified) class, columns = reference class. The "
            "upper block holds the sample counts, the lower block the "
            "estimated area proportions."), "Error matrix")

        self.class_table = QTableWidget()
        self.result_tabs.addTab(self._panel(
            self.class_table,
            "User's accuracy = 1 - commission error; producer's accuracy = "
            "1 - omission error. Intervals are at the confidence level chosen "
            "on the Design tab; an asterisk marks a Wilson score interval, "
            "substituted where the normal approximation is unreliable (for an "
            "accuracy of exactly 1.0 the usual interval would collapse to zero "
            "width)."), "Per class")

        self.area_table = QTableWidget()
        self.result_tabs.addTab(self._panel(
            self.area_table,
            "Adjusted areas remove the bias left by pixel counting (Olofsson "
            "et al. 2014, Eqs. 9-10). A mapped area outside the interval of "
            "the adjusted area is highlighted: pixel counting alone would have "
            "misled."), "Area")

        self.dis_table = QTableWidget()
        self.result_tabs.addTab(self._panel(
            self.dis_table,
            "Quantity disagreement is a difference in how much of a class "
            "exists; allocation disagreement is a difference in where it is "
            "(Pontius & Millones 2011)."), "Disagreement")

        self.caveat_text = QTextEdit()
        self.caveat_text.setReadOnly(True)
        self.result_tabs.addTab(self.caveat_text, "Caveats")
        lay.addWidget(self.result_tabs, 1)

        row = QHBoxLayout()
        for label, slot in (("Export report (TXT)", self.export_txt),
                            ("Export JSON", self.export_json),
                            ("Export HTML", self.export_html),
                            ("Export matrix (CSV)", self.export_csv),
                            ("Export validation points", self.export_points)):
            b = QPushButton(label)
            b.clicked.connect(slot)
            b.setEnabled(False)
            row.addWidget(b)
            self._export_buttons.append(b)
        row.addStretch()
        lay.addLayout(row)
        return w

    @staticmethod
    def _panel(table, caption):
        w = QWidget()
        lay = QVBoxLayout(w)
        note = QLabel(caption)
        note.setWordWrap(True)
        note.setStyleSheet("color:#4a5a68;background:#f2f6f8;padding:7px;"
                           "border-radius:4px;")
        lay.addWidget(note)
        lay.addWidget(table, 1)
        return w

    # -- reactions --------------------------------------------------------
    def _mode_changed(self):
        categorical = self.mode_categorical.isChecked()
        self.cont_box.setEnabled(not categorical)
        self.rb_stratified.setEnabled(categorical)
        if not categorical and self.rb_stratified.isChecked():
            self.rb_random.setChecked(True)

    def _method_changed(self):
        self.strat_box.setEnabled(self.rb_stratified.isChecked())
        self.csv_box.setEnabled(self.rb_points.isChecked())
        self.n_points.setEnabled(not self.rb_points.isChecked())

    # -- layers -----------------------------------------------------------
    def reload_layers(self):
        for combo in (self.ref_combo, self.cls_combo):
            current = combo.currentData()
            combo.clear()
            for layer in QgsProject.instance().mapLayers().values():
                if isinstance(layer, QgsRasterLayer) and layer.isValid():
                    combo.addItem(layer.name(), layer)
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def browse_raster(self, combo):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a raster file", "",
            "Raster files (*.tif *.tiff *.img *.asc *.vrt *.jp2 *.bil *.dat);;"
            "All files (*.*)")
        if not path:
            return
        for i in range(combo.count()):
            layer = combo.itemData(i)
            if layer is not None and layer.source() == path:
                combo.setCurrentIndex(i)
                return
        layer = QgsRasterLayer(path, os.path.splitext(os.path.basename(path))[0])
        if not layer.isValid():
            QMessageBox.critical(self, PLUGIN_NAME,
                                 "The raster could not be opened:\n%s" % path)
            return
        QgsProject.instance().addMapLayer(layer)
        combo.addItem(layer.name(), layer)
        combo.setCurrentIndex(combo.count() - 1)

    def browse_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the validation point file", "",
            "CSV files (*.csv *.txt);;All files (*.*)")
        if path:
            self.csv_path.setText(path)

    # -- step 1 -> 2 ------------------------------------------------------
    def inspect_rasters(self):
        ref_layer = self.ref_combo.currentData()
        cls_layer = self.cls_combo.currentData()
        if ref_layer is None or cls_layer is None:
            QMessageBox.warning(self, PLUGIN_NAME,
                                "Select both a reference and a classified "
                                "raster.")
            return
        if (ref_layer is cls_layer
                and self.ref_band.value() == self.cls_band.value()):
            QMessageBox.warning(self, PLUGIN_NAME,
                                "Both selections point at the same band of the "
                                "same raster.")
            return
        try:
            self.ref_info = rio.describe(ref_layer, self.ref_band.value())
            self.cls_info = rio.describe(cls_layer, self.cls_band.value())
        except Exception as exc:
            QMessageBox.critical(self, PLUGIN_NAME, str(exc))
            return

        diags = rio.check_pair(self.ref_info, self.cls_info)
        lines = ["[%-7s] %s" % (d.level.upper(), d.message) for d in diags]
        lines.append("")
        for tag, info in (("REFERENCE ", self.ref_info),
                          ("CLASSIFIED", self.cls_info)):
            i = info.to_dict()
            lines.append("%s  %s" % (tag, i["name"]))
            lines.append("    CRS %s | %d x %d px | pixel %.6g x %.6g %s | %s "
                         "| NoData %s"
                         % (i["crs"], i["width"], i["height"],
                            i["resolution_x"], i["resolution_y"],
                            i["map_units"], i["data_type"], i["nodata"]))
        self.diag_text.setPlainText("\n".join(lines))

        if any(d.is_blocking for d in diags):
            QMessageBox.critical(
                self, PLUGIN_NAME,
                "The two rasters cannot be compared as they stand:\n\n"
                + "\n\n".join(d.message for d in diags if d.is_blocking))
            return

        if self.mode_categorical.isChecked():
            if not self._run_census():
                return
            self.tabs.setTabEnabled(1, True)
            self.tabs.setCurrentIndex(1)
            if self.reference_mapping is None:
                self.edit_mapping()
        else:
            self.reference_mapping = ana.ClassMapping()
            self.classified_mapping = ana.ClassMapping()
            self.category_labels = {}
            self.tabs.setTabEnabled(2, True)
            self.tabs.setCurrentIndex(2)
            self.run_button.setEnabled(True)

    def _run_census(self):
        self.progress.setVisible(True)
        self.progress.setValue(0)
        try:
            self._set_status("Census of the reference raster...")
            self.ref_census = rio.value_census(
                self.ref_info,
                window=intersection_window(self.ref_info, self.cls_info),
                progress=lambda f: self._tick(50.0 * f))
            self._set_status("Census of the classified raster...")
            self.cls_census = rio.value_census(
                self.cls_info,
                window=intersection_window(self.cls_info, self.ref_info),
                progress=lambda f: self._tick(50.0 + 50.0 * f))
        except Exception as exc:
            _log(traceback.format_exc())
            QMessageBox.critical(self, PLUGIN_NAME,
                                 "The raster census failed:\n%s" % exc)
            return False
        finally:
            self.progress.setVisible(False)
            self._set_status("")

        for tag, census in (("reference", self.ref_census),
                            ("classified", self.cls_census)):
            if census["continuous"]:
                QMessageBox.warning(
                    self, PLUGIN_NAME,
                    "The %s raster holds more than %d distinct values, so it "
                    "is not a thematic map. Switch to continuous mode, or "
                    "reclassify it first." % (tag, rio.MAX_UNIQUE_VALUES))
                return False
            if census["valid_pixels"] == 0:
                QMessageBox.warning(
                    self, PLUGIN_NAME,
                    "The %s raster holds no valid pixel inside the common "
                    "area." % tag)
                return False
        self._show_census()
        return True

    def _show_census(self):
        lines = []
        for tag, info, census in (
                ("REFERENCE", self.ref_info, self.ref_census),
                ("CLASSIFIED", self.cls_info, self.cls_census)):
            total = max(1, census["valid_pixels"])
            lines.append("%s  %s" % (tag, info.name))
            lines.append("    valid pixels %s   NoData %s   distinct values %s"
                         % ("{:,}".format(census["valid_pixels"]),
                            "{:,}".format(census["nodata_pixels"]),
                            census["distinct_count"]))
            lines.append("    %-14s %14s %10s" % ("value", "pixels", "% valid"))
            for value, count in census["counts"].items():
                lines.append("    %-14s %14s %9.4f"
                             % (_fmt_value(value), "{:,}".format(count),
                                100.0 * count / total))
            lines.append("")
        self.census_text.setPlainText("\n".join(lines))

    def edit_mapping(self):
        if not self.ref_census or not self.cls_census:
            QMessageBox.warning(self, PLUGIN_NAME,
                                "Run the raster inspection first.")
            return
        dlg = ClassMappingDialog(self.ref_census, self.cls_census, self,
                                 self.ref_info.name, self.cls_info.name)
        if _exec(dlg) != QDialog.DialogCode.Accepted:
            return
        self.reference_mapping = dlg.reference_mapping
        self.classified_mapping = dlg.classified_mapping
        self.category_labels = dlg.labels

        lines = ["CATEGORY  LABEL", "-" * 60]
        for cat in sorted(self.category_labels):
            lines.append("%-9d %s" % (cat, self.category_labels[cat]))
        lines.append("")
        for tag, mapping in (("reference", self.reference_mapping),
                             ("classified", self.classified_mapping)):
            pairs = sorted(mapping.value_to_category.items())
            lines.append("%s raster: %s"
                         % (tag, ", ".join("%s->%d" % (_fmt_value(v), c)
                                           for v, c in pairs)))
        self.mapping_text.setPlainText("\n".join(lines))

        self.tabs.setTabEnabled(2, True)
        self.tabs.setCurrentIndex(2)
        self.run_button.setEnabled(True)

    # -- sample size planner ----------------------------------------------
    def open_size_planner(self):
        if not self.category_labels or not self.cls_census:
            QMessageBox.information(
                self, PLUGIN_NAME,
                "The planner needs the class census. Inspect the rasters and "
                "define the class mapping first.")
            return
        cats = sorted(self.category_labels)
        use_map = self.strata_source.currentData() == "map"
        source = self.cls_census if use_map else self.ref_census
        mapping = self.classified_mapping if use_map else self.reference_mapping
        per_cat = dict((c, 0) for c in cats)
        for value, cnt in source["counts"].items():
            cat = mapping.value_to_category.get(value)
            if cat is not None:
                per_cat[cat] = per_cat.get(cat, 0) + int(cnt)
        total = max(1, sum(per_cat.values()))
        weights = [per_cat[c] / float(total) for c in cats]
        labels = [self.category_labels[c] for c in cats]
        dlg = SampleSizeDialog(weights, labels, self)
        if _exec(dlg) == QDialog.DialogCode.Accepted and dlg.chosen:
            self.n_points.setValue(int(dlg.chosen))

    # -- run ---------------------------------------------------------------
    def _tick(self, percent):
        self.progress.setValue(int(max(0, min(100, percent))))
        QApplication.processEvents()

    def _set_status(self, text):
        self.status.setText(text or "")
        QApplication.processEvents()

    def _collect_config(self):
        method = {1: "random", 2: "systematic", 3: "stratified",
                  4: "points"}[self.method_group.checkedId()]
        points = ref_values = ids = points_crs = None
        if method == "points":
            path = self.csv_path.text().strip()
            if not path:
                raise ValueError("Select a CSV file of validation points.")
            ids, xs, ys, ref_values = _read_points_csv(path)
            if not xs:
                raise ValueError("No usable rows were found in the CSV file.")
            points = (xs, ys)
            key = self.csv_crs.currentData()
            points_crs = (self.ref_info.crs if key == "layer"
                          else QgsCoordinateReferenceSystem(key))
            if not self.csv_use_values.isChecked():
                ref_values = None

        return ana.AnalysisConfig(
            reference_layer=self.ref_combo.currentData(),
            classified_layer=self.cls_combo.currentData(),
            reference_band=self.ref_band.value(),
            classified_band=self.cls_band.value(),
            mode=("categorical" if self.mode_categorical.isChecked()
                  else "continuous"),
            method=method,
            n_points=self.n_points.value(),
            seed=(self.seed.value() if self.seed_check.isChecked() else None),
            allocation=self.allocation.currentData(),
            min_per_stratum=self.min_per_stratum.value(),
            strata_source=self.strata_source.currentData(),
            min_separation=self.min_sep.value(),
            reference_mapping=self.reference_mapping,
            classified_mapping=self.classified_mapping,
            category_labels=self.category_labels,
            points=points, points_crs=points_crs,
            points_reference_values=ref_values, points_ids=ids,
            declared_design=self.declared_design.currentData(),
            bootstrap=self.bootstrap.value(),
            confidence_level=self.conf_level.currentData())

    def run_analysis(self):
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.warning(self, PLUGIN_NAME, str(exc))
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.run_button.setEnabled(False)
        cache = None
        if config.mode == "categorical" and self.ref_census and self.cls_census:
            cache = {"reference": self.ref_census,
                     "classified": self.cls_census}
        try:
            self.result = ana.run_analysis(
                config, progress=lambda f: self._tick(100.0 * f),
                status=self._set_status, census_cache=cache)
        except Exception as exc:
            _log(traceback.format_exc())
            QMessageBox.critical(
                self, PLUGIN_NAME,
                "The analysis could not be completed:\n\n%s" % exc)
            return
        finally:
            self.progress.setVisible(False)
            self._set_status("")
            self.run_button.setEnabled(True)

        self._render_results()
        self.tabs.setTabEnabled(3, True)
        self.tabs.setCurrentIndex(3)
        for b in self._export_buttons:
            b.setEnabled(True)

    # -- rendering ---------------------------------------------------------
    def _render_results(self):
        res = self.result
        self.summary_text.setPlainText(rpt.text_report(res))
        unique = list(dict.fromkeys(res.warnings))
        self.caveat_text.setPlainText(
            "\n\n".join("%d. %s" % (i + 1, w) for i, w in enumerate(unique))
            or "No caveats were raised by the diagnostics.")

        for table in (self.matrix_table, self.class_table, self.area_table,
                      self.dis_table):
            table.clear()
            table.setRowCount(0)
            table.setColumnCount(0)

        acc = res.accuracy
        if acc is None:
            self.result_tabs.setCurrentIndex(0)
            return
        labels = [str(l) for l in acc.labels]
        k = len(labels)
        z = res.config.z

        # matrix -----------------------------------------------------------
        self.matrix_table.setColumnCount(k + 2)
        self.matrix_table.setRowCount(2 * (k + 2))
        self.matrix_table.setHorizontalHeaderLabels(
            ["Map \\ Reference"] + labels + ["Total"])
        row = 0
        for block, matrix, fmt in (
                ("SAMPLE COUNTS", acc.counts, "%d"),
                ("ESTIMATED AREA PROPORTIONS", acc.proportions, "%.6f")):
            self.matrix_table.setItem(row, 0, _bold_item(block, left=True))
            row += 1
            for i in range(k):
                self.matrix_table.setItem(row, 0, QTableWidgetItem(labels[i]))
                total = 0.0
                for j in range(k):
                    v = matrix[i][j]
                    total += v
                    item = QTableWidgetItem(fmt % v)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                    if i == j:
                        item.setBackground(QColor("#e6f2ea"))
                    self.matrix_table.setItem(row, j + 1, item)
                self.matrix_table.setItem(row, k + 1, _bold_item(fmt % total))
                row += 1
            self.matrix_table.setItem(row, 0, _bold_item("Total", left=True))
            grand = 0.0
            for j in range(k):
                s = sum(matrix[i][j] for i in range(k))
                grand += s
                self.matrix_table.setItem(row, j + 1, _bold_item(fmt % s))
            self.matrix_table.setItem(row, k + 1, _bold_item(fmt % grand))
            row += 1
        _stretch(self.matrix_table.horizontalHeader())

        # per class ---------------------------------------------------------
        headers = ["Class", "n (map)", "n (ref)", "User's acc.", "UA interval",
                   "Producer's acc.", "PA interval", "F1", "Commission",
                   "Omission"]
        self.class_table.setColumnCount(len(headers))
        self.class_table.setRowCount(k)
        self.class_table.setHorizontalHeaderLabels(headers)
        for i in range(k):
            u, p = acc.users[i], acc.producers[i]
            f1 = acc.f1[i] if i < len(acc.f1) else None
            cells = [labels[i], str(acc.n_by_map_class[i]),
                     str(acc.n_by_reference_class[i]),
                     _n(u.value), _interval(u, z), _n(p.value), _interval(p, z),
                     _n(f1.value if f1 is not None else None),
                     _n(acc.commission(i)), _n(acc.omission(i))]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if j in (1, 2) and acc.n_by_reference_class[i] < 20:
                    item.setForeground(QColor("#8a5a00"))
                    item.setToolTip("Fewer than 20 reference units: this "
                                    "class's producer's accuracy is barely "
                                    "constrained.")
                self.class_table.setItem(i, j, item)
        _stretch(self.class_table.horizontalHeader())

        # area ---------------------------------------------------------------
        unit = acc.area_unit or ""
        headers = ["Class", "Map area (%s)" % unit, "Adjusted area (%s)" % unit,
                   "Margin", "CI lower", "CI upper", "Map share",
                   "Adjusted share"]
        rows = acc.areas(z)
        self.area_table.setColumnCount(len(headers))
        self.area_table.setRowCount(len(rows))
        self.area_table.setHorizontalHeaderLabels(headers)
        for i, r in enumerate(rows):
            cells = [str(r["label"]), _n(r.get("map_area"), 2),
                     _n(r.get("adjusted_area"), 2), _n(r.get("area_margin"), 2),
                     _n(r.get("area_ci_lower"), 2),
                     _n(r.get("area_ci_upper"), 2),
                     _pctd(r["map_proportion"]),
                     _pctd(r["adjusted_proportion"])]
            outside = False
            try:
                outside = not (r["area_ci_lower"] <= r["map_area"]
                               <= r["area_ci_upper"])
            except (KeyError, TypeError):
                outside = False
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if outside:
                    item.setBackground(QColor("#fdf1e3"))
                    item.setToolTip("The mapped area falls outside the "
                                    "confidence interval of the adjusted "
                                    "area: pixel counting is misleading for "
                                    "this class.")
                self.area_table.setItem(i, j, item)
        _stretch(self.area_table.horizontalHeader())

        # disagreement --------------------------------------------------------
        d = res.disagreement
        if d is not None:
            headers = ["Class", "Quantity", "Allocation", "Exchange", "Shift"]
            self.dis_table.setColumnCount(len(headers))
            self.dis_table.setRowCount(len(d.per_category) + 2)
            self.dis_table.setHorizontalHeaderLabels(headers)
            for i, r in enumerate(d.per_category):
                cells = [str(r["label"]), _n(r["quantity"]),
                         _n(r["allocation"]), _n(r["exchange"]), _n(r["shift"])]
                for j, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    if j:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                              | Qt.AlignmentFlag.AlignVCenter)
                    self.dis_table.setItem(i, j, item)
            base = len(d.per_category)
            self.dis_table.setItem(base, 0, _bold_item("TOTAL", left=True))
            for j, v in enumerate((d.quantity, d.allocation, d.exchange,
                                   d.shift), start=1):
                self.dis_table.setItem(base, j, _bold_item(_n(v)))
            self.dis_table.setItem(
                base + 1, 0,
                _bold_item("Total disagreement (1 - OA)", left=True))
            self.dis_table.setItem(base + 1, 1, _bold_item(_n(d.total)))
            _stretch(self.dis_table.horizontalHeader())

    # -- exports ------------------------------------------------------------
    def _save_dialog(self, caption, default_ext, filter_):
        name = "caras_%s_%s%s" % (
            self.result.classified_info.name.replace(" ", "_"),
            datetime.now().strftime("%Y%m%d_%H%M%S"), default_ext)
        path, _ = QFileDialog.getSaveFileName(self, caption, name, filter_)
        return path

    def _write(self, path, text):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except OSError as exc:
            QMessageBox.critical(self, PLUGIN_NAME,
                                 "The file could not be written:\n%s" % exc)
            return
        QMessageBox.information(self, PLUGIN_NAME, "Saved:\n%s" % path)

    def export_txt(self):
        path = self._save_dialog("Save the report", ".txt", "Text (*.txt)")
        if path:
            self._write(path, rpt.text_report(self.result))

    def export_json(self):
        path = self._save_dialog("Save the JSON report", ".json",
                                 "JSON (*.json)")
        if path:
            self._write(path, rpt.json_report(self.result))

    def export_html(self):
        path = self._save_dialog("Save the HTML report", ".html",
                                 "HTML (*.html)")
        if path:
            self._write(path, rpt.html_report(self.result))

    def export_csv(self):
        path = self._save_dialog("Save the error matrix", ".csv", "CSV (*.csv)")
        if path:
            self._write(path, rpt.matrix_csv(self.result))

    def export_points(self):
        res = self.result
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the validation points",
            "caras_points_%s.gpkg" % datetime.now().strftime("%Y%m%d_%H%M%S"),
            "GeoPackage (*.gpkg);;ESRI Shapefile (*.shp)")
        if not path:
            return
        driver = "ESRI Shapefile" if path.lower().endswith(".shp") else "GPKG"
        on_map_grid = (res.config.method == "stratified"
                       and res.config.strata_source == "map")
        grid = res.classified_info if on_map_grid else res.reference_info
        layer = QgsVectorLayer("Point?crs=%s" % grid.crs.authid(),
                               "CARAS validation points", "memory")
        provider = layer.dataProvider()
        fields = [_make_field("unit_id", "int"), _make_field("x", "double"),
                  _make_field("y", "double"), _make_field("ref_value", "double"),
                  _make_field("map_value", "double")]
        categorical = res.accuracy is not None
        has_strata = res.sample.strata is not None
        if categorical:
            fields += [_make_field("ref_cat", "int"),
                       _make_field("map_cat", "int"),
                       _make_field("ref_label", "string"),
                       _make_field("map_label", "string"),
                       _make_field("agreement", "string")]
            if has_strata:
                fields.append(_make_field("stratum", "string"))
        else:
            fields.append(_make_field("residual", "double"))
        provider.addAttributes(fields)
        layer.updateFields()

        labels = res.config.category_labels or {}
        feats = []
        for i in range(len(res.sample)):
            f = QgsFeature()
            f.setGeometry(QgsGeometry.fromPointXY(
                QgsPointXY(float(res.sample.xs[i]), float(res.sample.ys[i]))))
            attrs = [i + 1, float(res.sample.xs[i]), float(res.sample.ys[i]),
                     _safe_float(res.reference_values[i]),
                     _safe_float(res.classified_values[i])]
            if categorical:
                rc = res.reference_categories[i]
                mc = res.classified_categories[i]
                rc_i = int(rc) if np.isfinite(rc) else None
                mc_i = int(mc) if np.isfinite(mc) else None
                attrs += [rc_i, mc_i, labels.get(rc_i, ""),
                          labels.get(mc_i, ""),
                          ("agree" if (rc_i is not None and rc_i == mc_i)
                           else "disagree")]
                if has_strata:
                    attrs.append(str(res.sample.strata[i]))
            else:
                rv = _safe_float(res.reference_values[i])
                mv = _safe_float(res.classified_values[i])
                attrs.append(None if (rv is None or mv is None) else mv - rv)
            f.setAttributes(attrs)
            feats.append(f)
        provider.addFeatures(feats)

        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = driver
        options.fileEncoding = "UTF-8"
        try:
            err = QgsVectorFileWriter.writeAsVectorFormatV3(
                layer, path, QgsCoordinateTransformContext(), options)
        except AttributeError:
            err = QgsVectorFileWriter.writeAsVectorFormatV2(
                layer, path, QgsCoordinateTransformContext(), options)
        if err[0] != QgsVectorFileWriter.WriterError.NoError:
            QMessageBox.critical(self, PLUGIN_NAME,
                                 "The points could not be written:\n%s"
                                 % err[1])
            return
        saved = QgsVectorLayer(path, "CARAS validation points", "ogr")
        if saved.isValid():
            QgsProject.instance().addMapLayer(saved)
        QMessageBox.information(self, PLUGIN_NAME, "Saved:\n%s" % path)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _bold_item(text, left=False):
    item = QTableWidgetItem(text)
    _bold(item)
    item.setTextAlignment(
        (Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight)
        | Qt.AlignmentFlag.AlignVCenter)
    return item


def _n(value, nd=4):
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    return ("%." + str(nd) + "f") % v


def _pctd(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return "%.2f %%" % (100.0 * v)


def _interval(estimate, z):
    """Confidence interval, falling back to Wilson where Wald is unusable.

    A class whose sample units are all correct yields a Wald standard error of
    exactly zero, i.e. the interval [1, 1]; the Wilson score interval is
    substituted there and marked with an asterisk.
    """
    if estimate.normal_approximation_ok is False:
        wlo, whi = estimate.wilson(z)
        if np.isfinite(wlo):
            return "[%.4f, %.4f] *" % (wlo, whi)
    lo, hi = estimate.ci(z)
    if not np.isfinite(lo):
        return "n/a"
    return "[%.4f, %.4f]" % (lo, hi)


def _safe_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _read_points_csv(path):
    """Read ``id,x,y,reference_value``; extra columns are ignored."""
    ids, xs, ys, vals = [], [], [], []
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline().strip()
        sep = ";" if header.count(";") > header.count(",") else ","
        cols = [c.strip().lower() for c in header.split(sep)]
        try:
            i_id, i_x, i_y = cols.index("id"), cols.index("x"), cols.index("y")
        except ValueError:
            raise ValueError("The CSV header must contain the columns id, x, y "
                             "and (optionally) reference_value.")
        i_v = cols.index("reference_value") if "reference_value" in cols else None
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(sep)]
            if len(parts) <= max(i_id, i_x, i_y):
                continue
            try:
                x = float(parts[i_x])
                y = float(parts[i_y])
            except ValueError:
                continue
            ids.append(parts[i_id])
            xs.append(x)
            ys.append(y)
            if i_v is not None and len(parts) > i_v:
                try:
                    vals.append(float(parts[i_v]))
                except ValueError:
                    vals.append(float("nan"))
            else:
                vals.append(float("nan"))
    if i_v is None or all(not np.isfinite(v) for v in vals):
        vals = None
    return ids, xs, ys, vals


def _log(message):
    try:
        from qgis.core import Qgis, QgsMessageLog
        QgsMessageLog.logMessage(message, PLUGIN_NAME,
                                 Qgis.MessageLevel.Warning)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# plugin entry point
# ---------------------------------------------------------------------------
class CARASPlugin(object):

    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
        self.action = QAction(icon, "CARAS - accuracy assessment",
                              self.iface.mainWindow())
        self.action.setToolTip(
            "CARAS: design-based accuracy, area and agreement assessment "
            "between two rasters")
        self.action.triggered.connect(self.run)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("CARAS", self.action)

    def unload(self):
        if self.action is not None:
            self.iface.removeToolBarIcon(self.action)
            self.iface.removePluginRasterMenu("CARAS", self.action)
            self.action = None
        self.dialog = None

    def run(self):
        if self.dialog is None:
            self.dialog = CARASDialog(self.iface, self.iface.mainWindow())
        self.dialog.reload_layers()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()


def classFactory(iface):
    return CARASPlugin(iface)
