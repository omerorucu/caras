# Using your own validation points

CARAS can take validation units from a CSV file instead of drawing them itself — field campaign
points, an existing reference sample, or a sample designed in another tool.

## File format

A header row is required. `id`, `x` and `y` must be present; `reference_value` is optional.
Column order does not matter, extra columns are ignored, and both `,` and `;` are accepted as the
separator (whichever occurs more often in the header wins).

```csv
id,x,y,reference_value
P001,412300.5,4189200.5,1
P002,415120.0,4191840.0,3
P003,418004.2,4193117.8,2
```

| Column | Meaning |
|---|---|
| `id` | Any unique label. Kept for your own bookkeeping; the export renumbers units from 1. |
| `x`, `y` | Coordinates, either in WGS 84 (EPSG:4326) or in the reference raster's CRS — you choose which in the **Design** tab. |
| `reference_value` | Optional. The ground-truth value observed at that location. When present you can use it *instead of* the reference raster, which is the normal case for field data. Integer or decimal. |

Points that fall outside the raster, or on NoData in either raster, are dropped and the count is
reported. Values that were not assigned to a category in the class mapping are dropped as well —
those pixels are not part of the target population.

## What CARAS needs to know, and why

Before running, the **Design** tab asks which sampling design produced the points.

This is not bureaucracy. A confidence interval is a statement about a *sampling design*: it says
how much the estimate would move if the design were repeated. If the points were selected by a
probability rule — every unit in the map had a known, positive chance of being chosen — then the
standard errors CARAS prints are inferential and can go into a paper. If the points were chosen
because they were accessible, or because they looked interesting, or because someone had already
been there, then no design exists to repeat, and the same numbers are **descriptive only**. CARAS
prints exactly that sentence in the limitations section of the report rather than letting the
distinction disappear.

If your points came from a stratified design, the strata weights are not recoverable from the CSV
alone. Either draw the sample inside CARAS (Design → Stratified random), or treat the supplied
points as equal-probability and note the limitation.

## Coordinate reference systems

`x`/`y` are interpreted in whichever CRS you select, and transformed to the raster CRS with a
proper `QgsCoordinateTransform`. Mixing a WGS 84 CSV with a projected raster is therefore safe.
Reprojection introduces a sub-pixel positional error that is **not** propagated into the reported
standard errors; when the pixel size is close to the positional accuracy of the field GPS, say so
in the methods section.

## Sample files

`sample_points_integer.csv`, `sample_points_float.csv` and `sample_validation_points.csv` in the
plugin folder are minimal, valid examples using WGS 84 coordinates.
