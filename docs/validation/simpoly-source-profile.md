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
| `BITWISE_PARITY` | R2026a probe with a common input produced identical discrete arrays/scalars. |
| `NUMERICAL_PARITY` | R2026a probe agreed within a stated floating-point tolerance, not bitwise. |
| `CROSS_VALIDATED_WITH_TOLERANCE` | Real MATLAB and Python outputs were compared and the residual is quantified. |

The executable oracle is MATLAB 26.1.0.3312084 (R2026a) Update 4 with Image
Processing, Curve Fitting and Signal Processing Toolboxes 26.1. The supplied source
mentions R2020a, so the evidence is release-qualified. Python uses NumPy 2.5.1,
SciPy 1.18.0 and scikit-image 0.26.0.

## R2026a operation probes

Each row feeds Python the exact MATLAB input from the preceding stage.

Stage hashes serialize 2-D arrays in row-major/C order and little-endian byte
order. Logical masks are first converted to `uint8`. Integer hashes are exact
evidence; floating arrays are judged by max/mean absolute error, RMSE and explicit
`rtol`/`atol`, never by hash alone.

| Source rule | Python implementation | MATLAB probe status | Real TIFF status | Classification | Known difference |
|---|---|---|---|---|---|
| crop first channel, remove 90 rows | array slice | identical | `ZEISS_001` identical | `BITWISE_PARITY` | none observed |
| `adapthisteq(I)` | scikit CLAHE, default 8×8 tile grid | normalized MAE 0.00725 | first divergence | `CROSS_VALIDATED_WITH_TOLERANCE` | interpolation/clip implementation differs |
| `histeq` default 64 levels | cumulative-error transform | identical with common CLAHE | isolated real array identical | `BITWISE_PARITY` | full pipeline inherits CLAHE differences |
| erosion, disk 5 | R2026a 9×9/69-pixel footprint | identical | isolated real array identical | `BITWISE_PARITY` | MATLAB default is not `skimage.disk(5)` |
| reconstruction | `morphology.reconstruction` | identical | isolated real array identical | `BITWISE_PARITY` | dependency emits a NumPy 2.5 warning |
| Canny `[0.2 0.4]` | scikit Canny | 334,947 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | gradient/smoothing/NMS differ |
| `bwareaopen(...,20)` | 8-connected filter | identical | isolated real array identical | `BITWISE_PARITY` | none observed |
| edge thicken 1 | padded dual thinning | 973 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | network subiterations differ |
| `graythresh(I)+0.1` | discrete-class Otsu | identical scalar | isolated real scalar identical | `BITWISE_PARITY` | none observed |
| threshold on `Ihist` | direct comparison | identical | isolated real mask identical | `BITWISE_PARITY` | none observed |
| close disk 1 | morphological closing | 173 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | border semantics |
| clean/fill/majority | iterative 3×3 rules | identical | isolated real masks identical | `BITWISE_PARITY` | none observed |
| thin 4 | scikit thin | identical | isolated real mask identical | `BITWISE_PARITY` | none on tested input |
| median loop | SciPy median, count stop | identical; 67 iterations | isolated real mask identical | `BITWISE_PARITY` | stop is count equality, not array equality |
| thicken 4 | padded dual thinning | 83 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | network semantics |
| `bwskel` | scikit skeletonize | 85,692 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | algorithm/tie-breaking differs |
| branchpoints | R2026a-derived local LUT | 424 pixels differ on real skeleton | mismatch | `CLOSE_REIMPLEMENTATION` | plus probe exact; network context differs |
| branch guard disk 3,n=0 | exact footprint dilation | identical | isolated real mask identical | `BITWISE_PARITY` | none observed |
| spur 1 | endpoint templates | 23,653 pixels differ | mismatch | `CLOSE_REIMPLEMENTATION` | endpoint/subiteration semantics differ |
| `bwdist` and doubled EDT | SciPy EDT | max error ≤1.52e-5 px | numerical agreement | `NUMERICAL_PARITY` | MATLAB single vs Python double |
| automatic histogram | NumPy auto histogram | campaign comparison | mismatch possible | `VERSION_DEPENDENT` | MATLAB graphics auto-binning retained by oracle |
| `gauss1` | SciPy nonlinear fit | campaign comparison | mismatch possible | `MATLAB_PARITY_UNVERIFIED` | optimizer differs |

## Corrected source-rule divergences

The 2026-08-11 audit corrected these deviations in the prior Python port:

- `bwareaopen(..., 20)` now removes areas strictly below 20 and retains area 20;
- automatic bin selection replaces a hard-coded 30-bin histogram;
- the fit uses left bin edges, not bin centers;
- µm/px conversion is applied before histogramming and fitting;
- integer intensity normalization uses the full dtype range, including uint16;
- the non-source 500-iteration median-filter cap was removed.
- MATLAB CLAHE is an 8×8 tile grid, not an 8-pixel kernel;
- `histeq(I)` uses MATLAB's 64-level cumulative-error transform;
- R2026a disk 5 uses the probed 9×9/69-pixel neighborhood;
- `bwareaopen` uses default 8-connectivity;
- the non-source 100-iteration cap on `majority(...,500)` was removed;
- `graythresh` uses MATLAB's discrete image-class Otsu level;
- padding removes finite-image border artifacts from dual thinning.

Automatic binning remains `VERSION_DEPENDENT`: NumPy's `"auto"` selector is not
claimed to be MATLAB's graphics histogram selector. Canny, complex thickening,
`bwskel`, network branchpoints/spur and fitting are not called exact MATLAB semantics.

### Reconstruction deprecation warning

scikit-image 0.26 still exposes `skimage.morphology.reconstruction` as the public
API; it dispatches internally to `morphology.grayreconstruct`, whose assignment to
`ndarray.shape` triggers a NumPy 2.5 deprecation warning. There is no documented
replacement public API in the installed version. The call is therefore retained:
switching primitives without a MATLAB probe would mix dependency migration with a
scientific change. The isolated R2026a reconstruction array is bitwise identical.

## Frozen 16-TIFF campaign result

Dataset `ZEISS_PVDF_2026-07-30` completed all 16 MATLAB SOURCE_COMPAT, 16 MATLAB
CONTROLLED_INPUT, 32 corresponding Python SIMPoly and 16 Fathom attempts. No case
was silently excluded. CONTROLLED_INPUT uses the Zeiss reader's 2071-row body;
SOURCE_COMPAT uses the literal 90-row crop (2214 rows), so only the former is a
controlled method comparison with Fathom.

For CONTROLLED_INPUT, absolute MATLAB/Python `b1` difference was mean 10.586%,
median 9.170%, P90 19.046% and maximum 30.182%; only 2/16 were within 5%. For
SOURCE_COMPAT it was mean 11.687%, median 8.780%, P90 15.856% and maximum 50.402%;
1/16 was within 5%. The campaign therefore records `MATLAB_PYTHON_PARITY_FAIL`.

All 16 controlled cases first diverged at CLAHE. Across them, CLAHE normalized
MAE ranged 0.00516–0.02875; Canny Dice had median 0.508 (range 0.281–0.658);
final valid-skeleton Dice had median 0.180 (range 0.037–0.218). Raw skeleton
displacement had median 1 px in the median case and P95 displacement median 5.465
px, with a worst P95 of 43.012 px. No histogram or Gaussian fit was exactly equal.

Fathom/SIMPoly absolute method difference had mean 38.056% and median 37.777%.
This is reported as a difference between estimands, not an accuracy error and not
a target for tuning Fathom. The ignored local HTML report and 16-row review queue
are under `.validation/real-tiff-campaign/latest/` and
`.validation/real-tiff-campaign/review_queue.csv`.

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
