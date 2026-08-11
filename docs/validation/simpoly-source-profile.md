# SIMPoly source-compatible profile

## Canonical source

The canonical source supplied for this audit is:

```text
/home/kegouro/Downloads/sj-m-1-ten-10.1089_ten.tec.2020.0304.m
SHA-256: 6cbb827ebfa92a2f951d3fd06cb3561d81854ddd8fc4fc9f8f7bb1151ad1f446
```

It is byte-identical to the ignored repository evidence copy at
`.reference/simpoly/original/SIMPolyMatlabCode.m`. The external source is never
modified or packaged. The normalized working copy is not byte-identical and is
not the canonical evidence.

## Profiles

`SIMPOLY_SOURCE_COMPAT_V1` preserves the source ordering and literal decisions,
including first-channel selection, fixed 90-row footer crop, threshold computed
from the cropped original and applied to the equalized image, the morphology
sequence, equal-foreground-count median stopping rule, 55 px skeleton guard,
`2*bwdist(~BW)`, automatic histogram, two prepended zero samples, `gauss1`, `b1`
as the main reported result, and `c1/2` as the source-reported standard deviation.

`SIMPOLY_CONTROLLED_INPUT_V1` runs the same downstream implementation on a
caller-provided image body/ROI. It deliberately bypasses the source's fixed
footer crop so methods can be compared over a common spatial domain.

If a positive conversion in µm/px is provided, it is applied to diameters before
histogramming and fitting, as in the MATLAB source. Pixel equivalents are retained
separately in the typed result.

## Evidence classifications

The implementation exposes `SIMPOLY_STAGE_PARITY`. The labels mean:

| Label | Meaning |
|---|---|
| `EXACT_SOURCE_RULE` | Literal control flow or parameter is confirmed in the source. |
| `EXACT_FORMULA` | The displayed arithmetic formula is identical. |
| `TESTED_INTERNAL_SEMANTICS` | Python behavior is unit-tested, without claiming MATLAB parity. |
| `CLOSE_REIMPLEMENTATION` | Closely corresponding library operation, not externally cross-validated. |
| `VERSION_DEPENDENT` | Behavior depends on MATLAB/toolbox or Python library version. |
| `MATLAB_PARITY_UNVERIFIED` | No executable MATLAB oracle proves stage equivalence. |

Notably, CLAHE, histogram equalization, Canny, skeletonization, `bwmorph`
thinning/thickening/branchpoints/spur, and nonlinear fitting are not called exact
MATLAB semantics. The source uses MATLAB R2020a toolboxes; this Python environment
uses NumPy 2.5.1, SciPy 1.18.0 and scikit-image 0.26.0.

## Corrected source-rule divergences

The 2026-08-11 audit corrected these deviations in the prior Python port:

- `bwareaopen(..., 20)` now removes areas strictly below 20 and retains area 20;
- automatic bin selection replaces a hard-coded 30-bin histogram;
- the fit uses left bin edges, not bin centers;
- µm/px conversion is applied before histogramming and fitting;
- integer intensity normalization uses the full dtype range, including uint16;
- the non-source 500-iteration median-filter cap was removed.

Automatic binning remains `VERSION_DEPENDENT`: NumPy's `"auto"` selector is not
claimed to be MATLAB R2020a's exact histogram selector. SciPy curve fitting is a
`MATLAB_PARITY_UNVERIFIED` implementation of the exact `gauss1` formula.

## Scientific non-claims

This profile is source compatible, not validated ground truth and not proven
exact MATLAB parity. Finite Python tests establish internal behavior only. The
source-compatible pixel domain should be presented as approximately 10–100 px,
and results remain projected 2D geometry. SIMPoly's fitted distribution center
and Fathom's manual or section-based widths need not estimate the same quantity.

## Historical literature implementation

`simpoly_compat.py` remains available as
`SIMPOLY_LITERATURE_REIMPLEMENTATION_V1` because the existing synthetic published-
benchmark campaign depends on it. It is not an alias for either source-compatible
profile. `SIMPOLY_COMPAT_DEVIATIONS` classifies its dynamic footer, omitted
histogram equalization, different morphology/thresholds, fixed 40-bin histogram
and mathematical Gaussian fit as `CLOSE_REIMPLEMENTATION` or
`MATLAB_PARITY_UNVERIFIED`. Its algorithm was not changed to improve the currently
failing benchmark gate.
