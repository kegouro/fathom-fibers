# Headless API

The headless API has no Qt or SPMKit dependency. `FathomEngine` accepts files or
NumPy arrays and always requires explicit physical calibration for arrays.

## Arrays and manual geometry

```python
import numpy as np

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.model import Calibration

engine = FathomEngine()
image = engine.from_array(
    np.zeros((512, 768), dtype=np.uint8),
    calibration=Calibration(5e-9, 7e-9, "experiment-log"),
    image_id="sample-A-field-01",
)

distance = engine.measure(
    image,
    "DISTANCE",
    {"p1": (10.0, 20.0), "p2": (40.0, 60.0)},
)
print(distance.primary_value, distance.primary_unit)
```

Supported interactive kinds are `PROJECTED_WIDTH`, `DISTANCE`,
`POLYLINE_LENGTH`, `ANGLE`, `RECTANGLE_AREA`, `POLYGON_AREA` and
`INTENSITY_PROFILE`. Derived values are returned read-only by convention; project
mutation belongs to `ProjectSession`.

## Zeiss TIFF

```python
image = engine.open_image("sample.tif")
print(image.calibration.pixel_size_x_m)
print(image.footer_bounds)
print(image.source_sha256)
```

Generic rasters without embedded calibration require
`manual_pixel_size_m=...`.

## Assisted Fathom

```python
result = engine.run_fathom(
    image,
    roi_bbox=(100, 100, 700, 700),
    options={"n_sections": 3, "polarity": "auto"},
)

for candidate in result.candidates:
    print(candidate.candidate_id, candidate.quality_flags)
```

Candidates are proposals. `ProjectSession.apply_fathom_result` persists them with
status `PROPOSED`; they do not enter primary statistics until explicitly accepted.

## SIMPoly

```python
source_result, source_intermediates = engine.run_simpoly(
    image,
    profile="SIMPOLY_SOURCE_COMPAT_V1",
)

controlled_result, controlled_intermediates = engine.run_simpoly(
    image,
    profile="SIMPOLY_CONTROLLED_INPUT_V1",
    roi_bbox=(100, 100, 700, 700),
)
```

Source-compatible mode applies the fixed first-channel/bottom-90-row source rule.
Controlled-input mode runs the downstream pipeline over the supplied common ROI.
An optional Boolean `valid_mask` may be applied in controlled mode; its use is
flagged in provenance. For anisotropic pixels, SIMPoly remains a pixel-domain
algorithm and the adapter explicitly flags use of X calibration.

The typed result distinguishes Gaussian center, source-reported `c1/2`,
mathematical `c1/sqrt(2)`, arithmetic mean, median, valid count, foreground
fraction, skeleton count and flags.

## Comparison

```python
comparison = engine.compare_methods(
    image,
    roi_bbox=(100, 100, 700, 700),
    manual_measurements=project.records,
)

for row in comparison.rows:
    print(row.method, row.estimand, row.n, row.main_reported_px)
```

Rows retain explicit estimands. A numerical difference does not imply that
SIMPoly's fitted distribution center, Fathom sections and manual accepted
measurements estimate the same physical quantity.

