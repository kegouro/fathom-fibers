# Unified Method Comparison

Every method observes the same calibrated image while retaining its own native
estimand and limitation. Agreement is not truth.

## Method contract

`MethodResult` records method/version, image/ROI, calibration, capabilities,
native result, optional native and common distributions, optional geometry,
flags, confidence, runtime and provenance. A backend must leave unsupported
fields absent and set its capability state accordingly.

The comparison contract uses `COMMON_LENGTH_WEIGHTED_DIAMETER` in physical units.
SIMPoly samples are weighted by calibrated skeleton arclength. Fathom Local
sections share their candidate centreline's physical length. Neither rule makes
the estimators interchangeable; it only gives an explicit common representation.

`CONSENSUS_PSEUDO_REFERENCE_V1` is the median of participating inverse CDFs on a
fixed quantile grid, with equal method weight. It records excluded methods and a
MAD disagreement envelope. It is not ground truth.

## Current methods

| Method | Native estimand | Common output | Status |
|---|---|---|---|
| MATLAB SIMPoly | `SIMPOLY_NATIVE_GAUSS1` | EDT samples from validated cache | cache-dependent |
| Python SIMPoly | `SIMPOLY_NATIVE_GAUSS1` | calibrated skeleton-length weighted EDT samples | partial MATLAB compatibility; `bwskel` divergence |
| Fathom Local | `FATHOM_NATIVE_LOCAL` | centreline-length weighted proposed sections | reviewable assisted output |
| Manual 5×5 | `MANUAL_5X5_REFERENCE` | only after accepted measurements | currently `NOT_MEASURED` |
| FATHOM_FIELD_GRAPH_V1 | none yet | none yet | `EXPERIMENTAL_NOT_YET_MEASURING` |

## Reproduction

```bash
fathom-fibers methods list
fathom-fibers compare --image image.tif --matlab-cache-root .validation/real-tiff-campaign
fathom-fibers campaign unified --dataset /path/to/tiffs \
  --matlab-cache-root /path/to/.validation/real-tiff-campaign
fathom-fibers campaign unified-report
```

The report is generated headlessly at
`.validation/unified-method-comparison/latest/index.html`.
