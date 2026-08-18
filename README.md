# CARAS — Classification Accuracy and Regression Assessment Suite

CARAS is a general-purpose QGIS plugin for performing comprehensive accuracy and regression assessment between two raster maps (e.g., a classified map vs. a reference/ground truth map).

## Features

- **Sampling methods**: random, stratified, systematic, or points imported from a CSV file
- **Browse raster from disk**: in addition to layers already loaded in the project, the "…" button lets you open a raster file directly from folder
- **Class mapping interface**: map pixel values from the reference and classified maps to comparable categories
- **Accuracy metrics**: Overall Accuracy, Cohen's Kappa, F1-Score (macro & weighted), Precision, Recall, Confusion Matrix, Producer's & User's Accuracy
- **Regression statistics**: R², RMSE, MAE, Bias (for both raw pixel values and mapped categories)
- **Report export**: TXT, JSON, HTML
- **Validation points export**: Shapefile

## Requirements

- QGIS 3.x or 4.0
- Python packages: `numpy`, `scikit-learn` (must be installed in QGIS's own Python environment)

```
pip install numpy scikit-learn
```

## Installation

1. Download or clone this repository
2. Copy the `caras` folder into your QGIS plugins directory:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` (or `QGIS4\...`)
   - Linux/macOS: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
3. Start QGIS and enable CARAS from **Plugins → Manage and Install Plugins**

## Usage

1. Click the CARAS icon in the toolbar
2. Select the reference and classified map from the layer list, or open one from disk with "…"
3. Choose a sampling method and number of points (or select a CSV file)
4. Click **Run CARAS Analysis**, then complete the class mapping in the dialog that appears
5. Review the results, and optionally export the report (TXT/JSON/HTML) or the validation points (Shapefile)

For using predefined points from a CSV file, see [CSV_NOKTA_KULLANIMI.md](CSV_NOKTA_KULLANIMI.md).

## License

Author: Ömer K. ÖRÜCÜ — omerorucu@sdu.edu.tr

This plugin was developed with the assistance of DeepSeek AI and Claude AI (Anthropic).
