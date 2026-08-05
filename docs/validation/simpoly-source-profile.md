# SIMPoly Source Code Method Profile

## Overview
This document provides a line-by-line audit of the canonical MATLAB source file `SIMPolyMatlabCode.m` (SHA-256: `6cbb827ebfa92a2f951d3fd06cb3561d81854ddd8fc4fc9f8f7bb1151ad1f446`) and maps its operations to the Python reimplementation in `fathom_fibers_quick.oracles.simpoly_source`.

## Line-by-Line MATLAB Source Audit

| Line Range | MATLAB Operation | Inputs / Parameters | Purpose / Semantics |
|---|---|---|---|
| **L4–8** | `imread(fullfile(p,file))` | User file dialog | Loads target micrograph image. |
| **L10–58** | `questdlg` & `inputdlg` | `conv` ($\mu m/\text{px}$) | Interactive pixel calibration dialog or scale bar distance tool. |
| **L60** | `I = I(1:end-90, :, 1);` | First 2D plane, drop 90 bottom rows | Strips SEM footer info (90 bottom pixel rows). |
| **L67–68** | `adapthisteq` + `histeq` | Default CLAHE + histogram equalization | Sequential contrast enhancement. |
| **L71–74** | `imerode` + `imreconstruct` | Disk SE $r=5$ | Grayscale erosion followed by morphological reconstruction. |
| **L76–82** | `edge` + `bwareaopen` + `bwmorph` | Canny `[0.2 0.4]`, min area 20, thicken 1 | Edge candidate detection, removes small edges $< 20\text{ px}$, thickens edges by 1 iteration. |
| **L85–86** | `graythresh(I) + 0.1` + `imbinarize` | Otsu on original `I` + 0.1 offset, applied to `Ihist` | Threshold computed from un-enhanced crop `I`, applied to contrast-enhanced `Ihist`. |
| **L90** | `imclose(BW, strel('disk', 1))` | Disk SE $r=1$ | Morphological closing on binary mask. |
| **L93–96** | `bwmorph` sequence | `clean` (100k), `fill` (5000), `majority` (500), `thin` (4) | Sequential binary morphological cleanup & thinning. |
| **L100–106** | `medfilt2` loop | `sum(sum(BWf)) ~= sum(sum(BW))` | Iterative 3x3 median filter until **foreground sum** stops changing. |
| **L110** | `bwmorph(BW, 'thicken', 4)` | 4 iterations | Re-thickens binary mask after median filtering. |
| **L112–117** | `bwskel` + `branchpoints` + `spur` | `branchpoints` dilated disk $r=3$, `spur` 1 iteration | Axial skeletonization, branchpoint guard removal ($r=3$), 1-iteration spur trimming. |
| **L120–130** | `F = bwdist(E)` | Cutoff distance $> 55\text{ px}$ | Filters skeleton pixels whose distance to Canny edges exceeds $55\text{ px}$. |
| **L141–144** | `Dist = 2*bwdist(~BW)` | `diameters = Dist(SK)` | Evaluates local diameter $2 \times \text{EDT}$ at valid skeleton pixels. |
| **L148–155** | `histogram` + `findpeaks` | Automatic histogram binning | Binning of local diameter array. |
| **L158–160** | `y = [0 0 h.Values]`, `x = [...]`, `fit(..., 'gauss1')` | $y = a_1 \exp\left(-\left(\frac{x - b_1}{c_1}\right)^2\right)$ | Prepends two zero-count samples before fitting 1-term Gaussian. |
| **L169–175** | `ave = f.b1`, `stdev = f.c1/2` | Output variables | Source reports `b1` as Average Diameter and `c1/2` as Standard Deviation. |

## Compatibility Profile & Known Deviations Table

| MATLAB Source Component | Python Reimplementation | Compatibility Level | Known Deviation | Test Coverage |
|---|---|---|---|---|
| Footer Crop `I(1:end-90,:,1)` | `I[0:-90, :, 0]` (`SOURCE_COMPAT_V1`) | `EXACT_SEMANTICS` | `CONTROLLED_INPUT_V1` bypasses 90-row crop to use Zeiss footer detector. | `test_fixed_90_row_crop` |
| Otsu Offset `graythresh(I) + 0.1` | `filters.threshold_otsu(I_crop) + 0.1` | `EXACT_SEMANTICS` | None | `test_threshold_otsu_offset` |
| Morphological Operations | `bwmorph_clean`, `bwmorph_fill`, `bwmorph_majority`, `bwmorph_thin`, `bwmorph_thicken` | `CLOSE_REIMPLEMENTATION` | Implemented via explicit 3x3 binary lookup / SciPy filters. | `test_bwmorph_semantics` |
| Median Filter Equal-Count Stop | `medfilt_loop_until_count_equal` | `EXACT_SEMANTICS` | Stopped by foreground pixel sum equality (not array equality). | `test_median_equal_count_stop` |
| Branchpoint Guard | `bwmorph_branchpoints` + `dilation(disk 3)` | `EXACT_SEMANTICS` | Removes skeleton pixels within 3-px radius of 3x3 junction points. | `test_branchpoint_guard` |
| Edge Distance Cleanup | `distance_transform_edt(~canny_edges) <= 55` | `EXACT_SEMANTICS` | Removes skeleton points $> 55\text{ px}$ from Canny edge map. | `test_edge_distance_cleanup` |
| Gaussian Fit Parameters | `fit_1d_gaussian_source_compat` | `CLOSE_REIMPLEMENTATION` | `b1` = Gaussian center, `c1/2` = Source StDev, `c1/sqrt(2)` = Mathematical Sigma. | `test_gaussian_fit_semantics` |
