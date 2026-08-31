# -*- coding: utf-8 -*-
"""
CARAS core package
==================

The statistical core is deliberately split from the QGIS user interface:

============================  ====================  =========================
module                        depends on QGIS       responsibility
============================  ====================  =========================
``estimators``                no                    design-based accuracy and
                                                    area estimators with
                                                    variances
``disagreement``              no                    quantity / allocation /
                                                    exchange / shift
``regression``                no                    continuous-map agreement
``report``                    no                    TXT / JSON / HTML / CSV
``raster``                    yes                   streaming raster access,
                                                    geometry, diagnostics
``sampling``                  yes                   probability sampling
                                                    designs
``analysis``                  yes                   orchestration, provenance
============================  ====================  =========================

Everything in the first group is importable and unit-testable without a QGIS
installation, which is what makes ``tests/`` runnable in plain CPython.  Submodules
are therefore *not* imported eagerly here.
"""

__version__ = "2.0.0"
__author__ = "Omer K. Orucu"
__license__ = "GPL-3.0-or-later"

#: Modules that can be imported without QGIS present.
PURE_MODULES = ("estimators", "disagreement", "regression", "report")

#: Modules that require a running QGIS environment.
QGIS_MODULES = ("raster", "sampling", "analysis")
