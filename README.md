# CARAS — Classification Accuracy and Regression Assessment Suite

A QGIS plugin for **statistically defensible** validation of one raster map against another.

CARAS implements the accuracy-assessment good practices of Olofsson et al. (2014): a probability
sampling design, a **design-based estimator matched to that design**, a **standard error for every
number it prints**, and **bias-adjusted areas** — the three things that separate a validation you can
publish from a confusion matrix you cannot.

---

## Why version 2 exists

Version 1.0.0 computed the usual metrics correctly but was not defensible as a scientific
instrument. Version 2.0.0 is a rebuild of everything downstream of the class mapping.

| Problem in 1.0.0 | What 2.0.0 does |
|---|---|
| No standard errors — `OA = 0.87` with no `±` | Every estimate carries an SE and a confidence interval; Wilson intervals where the normal approximation fails |
| Stratified samples estimated from **raw counts**, over-weighting rare classes → biased OA, PA and areas | Stratum weights from a full raster census; the Olofsson (2014) Eqs. 4–10 estimator |
| No area estimation | Bias-adjusted areas with confidence intervals; the mapped area is flagged when it falls outside that interval |
| Kappa foregrounded with the Landis & Koch verbal scale | Quantity / allocation / exchange / shift components (Pontius & Millones 2011). Kappa retained, benchmark scale removed, caution printed |
| RMSE, R², bias computed on **nominal class codes** | A categorical / continuous mode switch; regression statistics are refused for nominal data |
| **No CRS check between the two rasters** → silently wrong pixels | Pre-flight diagnostics; a mixed-CRS pair is transformed with `QgsCoordinateTransform` |
| Uniform sampling in **coordinate** space (latitude bias in geographic CRS), duplicates allowed, achieved *n* uncontrolled | Uniform over **pixels**, without replacement, restricted to the common valid area, sampling continues until *n* usable pairs exist |
| Systematic sampling: fixed origin, silently returned `floor(√n)²` points | Random start (a genuine probability design); realised size reported |
| Stratified allocation hard-wired to equal | Proportional, Neyman, equal, or proportional with a floor for rare classes; interactive sample size planner |
| No seed → results not reproducible | Seed exposed in the UI and written into every report |
| Whole raster loaded as `float64` (~964 MB for a Sentinel-2 tile) | Streaming block reads; scattered point queries are O(*n*), not O(pixels) |
| `-9999` hard-coded as NoData | Only the declared NoData value is honoured |
| HTML report contained a placeholder and unescaped class names | Reports rewritten with full provenance, a limitations section and a methods paragraph; all user text escaped |
| No licence, no tests, `scikit-learn` required | GPL-3.0, 37 checks that run without QGIS, NumPy only |

---

## What it reports

**Categorical maps**

- Error matrix as sample counts **and** as estimated area proportions, both with marginals,
  rows = map class, columns = reference class (stated in the output so it cannot be misread)
- Overall accuracy, user's and producer's accuracy, commission and omission error, F1 — each with a
  confidence interval and the sample size behind it
- Bias-adjusted class areas with confidence intervals, next to the uncorrected pixel-count areas
- Quantity, allocation, exchange and shift disagreement
- Cohen's kappa, with the reason not to lead with it
- A limitations section: thin classes, unsampled strata, thinning, CRS and support mismatches

**Continuous maps** (biomass, cover, LST, model output)

- RMSE with its systematic / unsystematic split (Willmott 1981), MAE, mean error
- The coefficient of determination reported **separately from** squared Pearson *r*, because they
  are different quantities and only one of them notices bias
- Willmott's *d* and refined *d1*, Lin's concordance correlation
- OLS slope and intercept with *t* tests against 1 and 0 — the standard signature of
  regression-to-the-mean in model output
- Seeded bootstrap confidence intervals for all of the above

---

## Workflow

1. **Data** — pick the two rasters, declare categorical or continuous, read the pre-flight
   diagnostics (CRS, overlap, grid offset, resolution, NoData).
2. **Classes** — a full census of both rasters inside the common area; fold raw pixel values onto
   shared categories, with pixel counts and area shares shown next to each value.
3. **Design** — sampling design, sample size (with the planner), seed, allocation, confidence level.
4. **Results** — estimates, matrices, areas, disagreement, caveats; export TXT / JSON / HTML / CSV
   and the validation points as GeoPackage or Shapefile.

---

## Installation

Requires QGIS 3.16 or newer (QGIS 4.x supported) and NumPy, which QGIS already ships.
There is no other dependency.

1. Copy the `caras` folder into your QGIS plugin directory:
   - Windows: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\` (or `QGIS4\…`)
   - Linux / macOS: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
2. Start QGIS and enable **CARAS** in *Plugins → Manage and Install Plugins*.

---

## Validation points from a CSV file

A header with `id,x,y` is required; `reference_value` is optional and, when present, can be used
instead of the reference raster. Coordinates are read either as WGS 84 or in the reference raster's
CRS, selected in the interface.

```csv
id,x,y,reference_value
P001,412300.5,4189200.5,1
P002,415120.0,4191840.0,3
```

Inference from supplied points is only valid under the design that generated them. CARAS asks which
design that was and records the answer; for purposive or convenience points the reported standard
errors are descriptive, and the report says so.

---

## Tests

The statistical core imports no QGIS module, so the suite runs in plain CPython:

```bash
python tests/run_tests.py     # or: pytest tests -q
```

The 37 unit checks are of three kinds: analytic identities that must hold exactly (the stratified
estimator collapsing to the equal-probability one when the weights match the allocation;
`Q + A = 1 − OA`; `E + S = A`; areas summing to the total), an independent re-implementation of the
published variance formulas written out literally inside the test, and a fully hand-worked
stratified example whose intermediate values appear in the test as literal numbers so they can be
checked on paper.

A separate integration test builds synthetic rasters with a **known** truth — including a class
covering 2 % of the map, a NoData corner and a deliberate 50 % omission error — and runs 45
end-to-end checks through the real QGIS API: the census against a direct NumPy count, pixel-centre
geometry, absence of duplicate or NoData units, seed reproducibility, coverage of the true overall
accuracy by the reported interval, the adjusted area beating pixel counting for the rare class, the
mixed-CRS path, and all four export formats. It skips itself outside QGIS:

```bash
"C:/Program Files/QGIS 3.40.13/bin/python-qgis-ltr.bat" tests/test_qgis_integration.py
```

Verified on QGIS 3.40.13 (Qt5) and QGIS 4.2.0 (Qt6).

---

## Citation

If CARAS contributes to a publication, please cite the software (see `CITATION.cff`) **and** the
methodological sources it implements — above all Olofsson et al. (2014), which supplies the
estimator, the variance formulas and the area adjustment.

Key references, all cited in every report CARAS produces:

- Olofsson, P., Foody, G.M., Herold, M., Stehman, S.V., Woodcock, C.E. & Wulder, M.A. (2014).
  Good practices for estimating area and assessing accuracy of land change.
  *Remote Sensing of Environment*, 148, 42–57. <https://doi.org/10.1016/j.rse.2014.02.015>
- Stehman, S.V. & Foody, G.M. (2019). Key issues in rigorous accuracy assessment of land cover
  products. *Remote Sensing of Environment*, 231, 111199.
- Pontius, R.G. Jr. & Millones, M. (2011). Death to Kappa. *IJRS*, 32, 4407–4429.
- Pontius, R.G. Jr. & Santacruz, A. (2014). Quantity, exchange and shift components. *IJRS*, 35,
  7543–7554.
- Foody, G.M. (2020). Explaining the unsuitability of the kappa coefficient. *RSE*, 239, 111630.
- Card, D.H. (1982). Using known map category marginal frequencies. *PE&RS*, 48, 431–439.
- Congalton, R.G. & Green, K. (2019). *Assessing the Accuracy of Remotely Sensed Data*, 3rd ed.
- Willmott, C.J. (1981). On the validation of models. *Physical Geography*, 2, 184–194.
- Lin, L.I. (1989). A concordance correlation coefficient. *Biometrics*, 45, 255–268.

---

## Related tools

[AcATaMa](https://plugins.qgis.org/plugins/AcATaMa/) also implements the Olofsson framework and is
the reference point for anyone doing this work in QGIS. CARAS differs in taking **two complete
rasters** as its input and providing a class-mapping step, so maps built on unrelated legends
(a model output against CORINE, ESA WorldCover against a national product) can be compared directly,
and in covering categorical and continuous validation in one interface.

---

## Licence

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

Author: Ömer K. Örücü — omerorucu@sdu.edu.tr — ORCID
[0000-0002-2162-7553](https://orcid.org/0000-0002-2162-7553)
Department of Landscape Architecture, Faculty of Architecture, Süleyman Demirel University,
Isparta, Türkiye.

Parts of this software were developed with the assistance of AI coding tools; all statistical
methods, formulas and their verification are the author's responsibility and are traceable to the
cited literature.
