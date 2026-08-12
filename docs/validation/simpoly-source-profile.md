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
| `KNOWN_LIBRARY_DIVERGENCE` | Common-input probes establish a reproducible difference in the available library primitive. |
| `UNRESOLVED` | Evidence is insufficient to assign a stronger classification. |

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
| `adapthisteq(I)` | dedicated R2026a-compatible 8×8 implementation | canonical uint8 probes exact | 16/16 bitwise | `BITWISE_PARITY` | non-divisible small tiles can retain ±1 LSB residuals; uint16 remains unresolved |
| `histeq` default 64 levels | cumulative-error transform | identical with common CLAHE | isolated real array identical | `BITWISE_PARITY` | full pipeline inherits CLAHE differences |
| erosion, disk 5 | R2026a 9×9/69-pixel footprint | identical | isolated real array identical | `BITWISE_PARITY` | MATLAB default is not `skimage.disk(5)` |
| reconstruction | `morphology.reconstruction` | identical | isolated real array identical | `BITWISE_PARITY` | dependency emits a NumPy 2.5 warning |
| Canny `[0.2 0.4]` | dedicated derivative-of-Gaussian/NMS/hysteresis implementation | 10/10 synthetic probes bitwise | 12/16 bitwise; 5 pixels differ in total | `CROSS_VALIDATED_WITH_TOLERANCE` | isolated boundary/tie pixels in four real arrays |
| `bwareaopen(...,20)` | 8-connected filter | identical | isolated real array identical | `BITWISE_PARITY` | none observed |
| edge thicken 1 | R2026a local LUT/subiterations | all 512 neighborhoods exact | 16/16 bitwise on common input | `BITWISE_PARITY` | none observed |
| `graythresh(I)+0.1` | discrete-class Otsu | identical scalar | isolated real scalar identical | `BITWISE_PARITY` | none observed |
| threshold on `Ihist` | strict `Ihist > level` comparison | threshold-tie probe exact | 16/16 bitwise on common input | `BITWISE_PARITY` | prior `>=` affected one canonical TIFF |
| close disk 1 | padded dilation/erosion with R2026a crop semantics | border probes exact | 16/16 bitwise on common input | `BITWISE_PARITY` | none observed |
| clean/fill/majority | iterative 3×3 rules | identical | isolated real masks identical | `BITWISE_PARITY` | none observed |
| thin 4 | scikit thin | identical | isolated real mask identical | `BITWISE_PARITY` | none on tested input |
| median loop | SciPy median, count stop | identical; 67 iterations | isolated real mask identical | `BITWISE_PARITY` | stop is count equality, not array equality |
| thicken 4 | R2026a local LUT/subiterations | all 512 neighborhoods exact | 16/16 bitwise on common input | `BITWISE_PARITY` | none observed |
| `bwskel` | scikit Zhang skeletonization | 74-shape corpus compared with four candidates | median Dice 0.233 on common real masks | `KNOWN_LIBRARY_DIVERGENCE` | no tested candidate reproduces MATLAB tie-breaking/topology |
| branchpoints | R2026a local LUT | all 512 neighborhoods exact | 16/16 bitwise on common skeleton input | `BITWISE_PARITY` | end-to-end input inherits `bwskel` |
| branch guard disk 3,n=0 | exact footprint dilation | identical | isolated real mask identical | `BITWISE_PARITY` | none observed |
| spur 1 | R2026a local LUT/subiterations | all 512 neighborhoods exact | 16/16 bitwise on common skeleton input | `BITWISE_PARITY` | end-to-end input inherits `bwskel` |
| `bwdist` and doubled EDT | SciPy EDT | max error ≤1.52e-5 px | numerical agreement | `NUMERICAL_PARITY` | MATLAB single vs Python double |
| automatic histogram | Scott width with MATLAB nice-edge rounding | 8/8 vectors match `histogram` and `histcounts` | 16/16 counts match bitwise; edges match numerically on common diameters | `NUMERICAL_PARITY` | float edge storage is compared with explicit tolerance; release-qualified to R2026a |
| `gauss1` | exact formula, unconstrained `a1`/`b1`, nonnegative `c1` | synthetic and real common histograms | real `b1` max absolute residual 0.000610 | `NUMERICAL_PARITY` | optimizer/start-point paths can differ for degenerate fits |

Probe counts used in this audit are 15 CLAHE patterns, 10 Canny patterns,
3,584 exhaustive 3×3 morphology neighborhoods (512 for each of seven
operations), 74 `bwskel` shapes, 8 histogram/fit vectors, and all 16 canonical
TIFF bodies. Real-image common-input checks are reported separately from
end-to-end propagation.

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
- padding removes finite-image border artifacts from dual thinning;
- `adapthisteq` now reproduces R2026a tile padding, clipping, redistribution,
  class quantization and bilinear tile-centre interpolation;
- Canny now reproduces R2026a's default `sqrt(2)` smoothing, derivative kernels,
  normalized gradient, interpolated non-maximum suppression and hysteresis;
- `thin`, both `thicken` stages, `branchpoints` and `spur` use exhaustively probed
  local lookup rules and subiteration ordering;
- closing reproduces the R2026a finite-image padding/cropping convention;
- automatic histogram edges reproduce the R2026a Scott-width and nice-edge rule;
- `gauss1` no longer applies non-source bounds to amplitude or centre;
- `imbinarize` now uses the demonstrated strict-foreground rule (`Ihist > level`),
  excluding samples exactly equal to the threshold.

`bwskel` remains the material compatibility boundary. Zhang, Lee 2-D, Lee 3-D,
converged `thin`, and medial-axis candidates were tested on the same MATLAB
outputs; none justified replacement of the existing source profile primitive.
The converged-thin candidate improved overlap on these images but was rejected
because better Dice is not evidence that it implements `bwskel`.

This boundary is explicitly release-dependent. MathWorks documents a changed
2-D `bwskel` implementation in R2026a: it now skeletonizes in two dimensions,
uses 4-connectivity, and no longer pads a 2-D image into a 3-D volume. Therefore
the Lee-3D behavior associated with earlier releases is not a valid R2026a
substitute. The present evidence targets the installed R2026a Update 4 oracle.

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

Before this parity batch, CONTROLLED_INPUT absolute MATLAB/Python `b1` difference
was mean 10.586%, median 9.170%, P90 19.046% and maximum 30.182%; 2/16 were within
5%. After the demonstrated corrections it is mean 7.555%, median 4.780%, P90
9.149% and maximum 37.318%; 9/16 are within 5% and 15/16 within 10%.

For literal SOURCE_COMPAT, the final mean is 11.841%, median 5.290%, P90 13.968%
and maximum 92.760%; 8/16 are within 5% and 13/16 within 10%. The large worst-case
relative differences occur where MATLAB's fitted centre is close to zero, but
they remain material and are not hidden. The campaign therefore remains
`MATLAB_PYTHON_PARITY_FAIL` at the whole-profile level.

Before correction all 16 controlled cases first diverged at CLAHE. Afterwards,
12 first diverge at `bwskel` and four at one- or two-pixel Canny ties (five Canny
pixels in total). CLAHE is bitwise on 16/16, as are threshold, closing, clean,
fill, majority, thin, median loop and final thickening. Final raw-skeleton Dice
has median 0.233; median displacement is 1 px and median P95 displacement is
4.357 px (worst P95 33.302 px). The valid-skeleton Dice median is 0.236.

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

## R2026a references

- [Contrast-limited adaptive histogram equalization (`adapthisteq`)](https://www.mathworks.com/help/images/ref/adapthisteq.html)
- [Canny and other edge detectors (`edge`)](https://www.mathworks.com/help/images/ref/edge.html)
- [Binary morphology (`bwmorph`)](https://www.mathworks.com/help/images/ref/bwmorph.html)
- [R2026a 2-D skeletonization behavior (`bwskel`)](https://www.mathworks.com/help/images/ref/bwskel.html)
- [Automatic histogram binning (`histogram`)](https://www.mathworks.com/help/matlab/ref/matlab.graphics.chart.primitive.histogram.html)
- [Gaussian curve models (`gauss1`)](https://www.mathworks.com/help/curvefit/gaussian.html)

## Historical literature implementation

`simpoly_compat.py` remains available as
`SIMPOLY_LITERATURE_REIMPLEMENTATION_V1` because the existing synthetic published-
benchmark campaign depends on it. It is not an alias for either source-compatible
profile. `SIMPOLY_COMPAT_DEVIATIONS` classifies its dynamic footer, omitted
histogram equalization, different morphology/thresholds, fixed 40-bin histogram
and mathematical Gaussian fit as `CLOSE_REIMPLEMENTATION` or
`MATLAB_PARITY_UNVERIFIED`. Its algorithm was not changed to improve the currently
failing benchmark gate.
