# Changelog

All notable changes to CARAS are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.0.0] — 2026-08-30

A scientific rebuild. **Results are not comparable with 1.0.0**: several
estimates that version 1 reported were biased or undefined, and version 2
computes them differently on purpose. Re-run any analysis produced with 1.0.0.

### Added — statistical substance

- **Design-based estimation** (Olofsson et al. 2014, *RSE* 148:42–57, Eqs. 4–10).
  A stratified sample is now reweighted by stratum weights taken from a full
  raster census, so accuracy and area refer to the map rather than to the
  sample.
- **Confidence intervals everywhere**: overall accuracy, user's and producer's
  accuracy, F1 and class areas each carry a standard error and an interval, at
  a selectable 90 / 95 / 99 % level. Wilson score intervals are substituted
  automatically where the normal approximation fails — a class with every unit
  correct no longer reports the interval [1, 1].
- **Bias-adjusted area estimation** with confidence intervals (Eqs. 9–10),
  shown next to the uncorrected pixel-count area and flagged when the latter
  falls outside the interval.
- **Quantity, allocation, exchange and shift disagreement** (Pontius & Millones
  2011; Pontius & Santacruz 2014), computed on the estimated area-proportion
  matrix.
- **Sample size planner** (Eq. 13) with proportional, Neyman, equal and
  rare-class-floor allocation schemes.
- **Continuous mode**: RMSE with its systematic / unsystematic decomposition
  (Willmott 1981), MAE, mean error, the coefficient of determination reported
  separately from squared Pearson *r*, Willmott's *d* and *d1*, Lin's
  concordance correlation, an OLS slope and intercept with *t* tests against 1
  and 0, and seeded bootstrap confidence intervals.
- **Pre-flight diagnostics** for CRS, extent overlap, grid offset, resolution
  mismatch, multiband rasters and undeclared NoData, with blocking versus
  advisory levels.
- **Provenance and reproducibility**: every report records the seed, software
  versions, raster metadata, analysis window, census, class mapping and the
  realised sample, plus a ready-to-adapt methods paragraph and the reference
  list.
- **Test suite** of 38 checks that runs without QGIS (`python tests/run_tests.py`),
  plus a QGIS integration test of 45 end-to-end checks against synthetic data
  with a known truth. Verified on QGIS 3.40.13 (Qt5) and 4.2.0 (Qt6).
- `LICENSE` (GPL-3.0-or-later), `CITATION.cff`, this changelog.
- CSV error-matrix export and GeoPackage validation-point export.

### Fixed — defects that changed results

- **Stratified samples were estimated from raw counts.** Every accuracy figure
  and every class area under a stratified design in 1.0.0 was biased toward
  rare classes.
- **No CRS check between the two rasters.** Coordinates from one raster were
  fed into the other's extent arithmetic; a mixed-CRS pair produced silently
  wrong pixel values and a plausible-looking accuracy table. Sample coordinates
  are now transformed with `QgsCoordinateTransform`, and the mismatch is
  reported.
- **Resolution mismatch was compared in incompatible units.** Two rasters
  describing the same ground cell in different CRS appeared to differ by a
  factor of ~94 000. The map raster's resolution is now expressed in the
  reference CRS before comparison.
- **Regression statistics on nominal class codes.** `r2_cat`, `rmse_cat`,
  `mae_cat` and `bias_cat` depended entirely on the arbitrary integers the
  analyst typed into the mapping dialog. A categorical / continuous mode switch
  now prevents this.
- **Sampling was uniform in coordinate space**, which over-samples high
  latitudes in a geographic CRS; it is now uniform over pixels.
- **Pixels could be drawn more than once** (pseudo-replication); sampling is now
  without replacement.
- **The achieved sample size was uncontrolled** — invalid points were discarded
  after the draw. Sampling now continues until *n* usable pairs exist, and the
  realised size is reported.
- **Systematic sampling had a fixed origin** and silently returned
  `floor(√n)²` points. It now uses a random start and reports its realised size.
- **Exported validation points sat on pixel corners**, half a pixel from the
  cell they described. All coordinates are pixel centres.
- **Exported points were written in the classified raster's CRS** even when the
  coordinates had been generated on the reference grid.
- **`-9999` was hard-coded as NoData** in addition to the declared value,
  silently discarding legitimate data. Only the declared NoData is honoured.
- **The HTML report contained a placeholder section** ("Detailed analysis
  results are available in the complete report") and injected user-supplied
  class names into the markup unescaped.
- Duplicate class labels are rejected instead of silently collapsing rows.
- `classification_report` key collisions could silently drop a class from the
  per-class table.

### Changed

- **Cohen's kappa is no longer a headline metric.** It is still reported for
  comparability, but the Landis & Koch verbal scale — devised for clinical
  inter-rater agreement, with no basis in map accuracy assessment — has been
  removed, and the caution of Pontius & Millones (2011) and Foody (2020) is
  printed alongside the value.
- **Matrix orientation is now rows = map, columns = reference**, the convention
  the published variance formulas assume, and it is stated in every output.
- Error matrices carry row, column and grand totals.
- **Memory**: rasters are read in streaming blocks instead of being loaded whole
  as `float64` (a 10 980² Sentinel-2 tile cost ~964 MB in 1.0.0). Scattered
  point reads are O(*n*) instead of O(pixels).
- The interface is a four-step workflow (Data → Classes → Design → Results)
  that enforces the order the statistics depend on.
- The class-mapping dialog shows the pixel count and area share of every raw
  value, and allows values to be excluded from the target population.
- Minimum requirement raised to QGIS 3.16; QGIS 4.x supported.

### Removed

- **`scikit-learn` dependency.** NumPy alone is enough; the plugin no longer
  fails to load when scikit-learn is missing from the QGIS Python environment.
- The Landis & Koch kappa benchmark scale.
- The regression block for categorical data.

## [1.0.0]

- Initial release: random / stratified / systematic sampling, CSV point import,
  class mapping, overall accuracy, Cohen's kappa, F1, precision, recall,
  confusion matrix, R², RMSE, MAE, bias, and TXT / JSON / HTML / Shapefile
  export.
