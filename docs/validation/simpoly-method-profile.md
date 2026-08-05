# SIMPoly Algorithm Method Profile

## Overview
This document summarizes the exact algorithmic steps, parameters, and design choices of **SIMPoly** (*Semiautomated Image Measurements of Polymers*) as published by Murphy et al. (2020), DOI: [10.1089/ten.tec.2020.0304](https://doi.org/10.1089/ten.tec.2020.0304).

## Algorithmic Pipeline & Audit

| Step | Operation | MATLAB Function | Parameters / Settings | Purpose / Description |
|---|---|---|---|---|
| **1** | Footer Exclusion & Scale Calibration | User Interaction / Dialog | Interactive scale bar length ($\mu m$) & pixel distance | Converts pixel magnitudes to physical micrometers ($\mu m$). |
| **2** | Contrast Enhancement | `adapthisteq` | Contrast-Limited Adaptive Histogram Equalization (CLAHE), default tiles `[8 8]`, clip limit `0.01` | Enhances local contrast of fibers relative to background. |
| **3** | Morphological Reconstruction | `imreconstruct` | Morphological erosion using disk structuring element ($r = 3 - 5\text{ px}$) as marker | Removes background noise while preserving fiber shapes. |
| **4** | Edge Localization | `edge` | Canny edge detector with automatic or user-defined thresholds | Identifies local fiber boundary candidates. |
| **5** | Global Binarization | `graythresh` + `im2bw` | Otsu global intensity thresholding | Produces initial binary foreground mask. |
| **6** | Component Filtering | `bwareaopen` | Minimum area threshold ($A_{\min} \approx 50 - 100\text{ px}$) | Removes small background noise components. |
| **7** | Morphological Closing & Smoothing | `imclose` + `medfilt2` | Disk SE ($r = 2 - 3\text{ px}$) and 2D median filter ($3 \times 3$ or $5 \times 5$ kernel) | Fills internal holes and smooths fiber boundaries. |
| **8** | Compensatory Dilation | `imdilate` | Disk SE ($r = 1 - 2\text{ px}$) | Recovers boundary pixels lost during smoothing filters. |
| **9** | Edge Overlay Cleanup | Edge map logical AND | Intersection of dilated mask with Canny edge map | Suppresses spurious non-boundary pixels. |
| **10** | Axial Skeletonization | `bwskel` / `bwmorph` | Infinite thinning (`'thin', Inf`) | Extracts 1-pixel wide fiber centerlines. |
| **11** | Distance Transform | `bwdist` | Euclidean Distance Transform (EDT) on inverse mask `~mask` | Calculates shortest distance from each foreground pixel to background. |
| **12** | Local Diameter Extraction | Skeleton masking | $d(x,y) = 2 \times \text{EDT}(x,y)$ at skeleton pixels | Evaluates local projected fiber diameter per skeleton pixel. |
| **13** | Histogram & Gaussian Fitting | `histogram` + `fit` | 1D Gaussian model: $y = A \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ | Extracts global peak diameter (`SIMPOLY_GAUSSIAN_CENTER` $\mu$) and distribution width ($\sigma$). |

## Estimand Definitions
- **`SIMPOLY_GAUSSIAN_CENTER`**: Peak ($\mu$) of 1D Gaussian fit on the skeleton pixel diameter distribution (primary reported metric in paper).
- **`SIMPOLY_GAUSSIAN_SIGMA`**: Standard deviation ($\sigma$) of the fitted Gaussian distribution.
- **`SKELETON_PIXEL_MEAN`**: Arithmetic mean of local diameters evaluated at skeleton pixels.
- **`SKELETON_PIXEL_MEDIAN`**: Median of local diameters evaluated at skeleton pixels.
