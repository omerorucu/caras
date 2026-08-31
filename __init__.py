# -*- coding: utf-8 -*-
"""
CARAS - Classification Accuracy and Regression Assessment Suite
===============================================================

Design-based accuracy assessment, bias-adjusted area estimation and
continuous-map agreement analysis between two rasters, for QGIS 3.16+ and 4.x.

Copyright (C) 2026 Omer K. Orucu <omerorucu@sdu.edu.tr>

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.  It is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.  A
copy of the licence ships with this plugin as ``LICENSE``.
"""

__version__ = "2.0.0"
__author__ = "Omer K. Orucu"
__email__ = "omerorucu@sdu.edu.tr"
__license__ = "GPL-3.0-or-later"


def classFactory(iface):  # pragma: no cover - QGIS entry point
    from .caras import CARASPlugin
    return CARASPlugin(iface)
