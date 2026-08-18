# -*- coding: utf-8 -*-
"""
CARAS - Classification Accuracy and Regression Assessment Suite
Herhangi iki raster harita için doğrulama analizi
QGIS 3.x ve 4.0 uyumlu / Compatible with QGIS 3.x and 4.0
"""

def classFactory(iface):
    from .caras import CARASPlugin
    return CARASPlugin(iface)
