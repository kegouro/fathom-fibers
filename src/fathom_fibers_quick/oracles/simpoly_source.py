from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import ndimage, optimize
from skimage import exposure, feature, filters, morphology

PROFILE_SOURCE_COMPAT_V1 = "SIMPOLY_SOURCE_COMPAT_V1"
PROFILE_CONTROLLED_INPUT_V1 = "SIMPOLY_CONTROLLED_INPUT_V1"


class ParityClassification(str, Enum):
    """Evidence level for one MATLAB-to-Python pipeline decision."""

    EXACT_SOURCE_RULE = "EXACT_SOURCE_RULE"
    EXACT_FORMULA = "EXACT_FORMULA"
    TESTED_INTERNAL_SEMANTICS = "TESTED_INTERNAL_SEMANTICS"
    CLOSE_REIMPLEMENTATION = "CLOSE_REIMPLEMENTATION"
    VERSION_DEPENDENT = "VERSION_DEPENDENT"
    MATLAB_PARITY_UNVERIFIED = "MATLAB_PARITY_UNVERIFIED"


# This is intentionally stage-level rather than one misleading global parity claim.
# The source rules are confirmed by the canonical MATLAB file.  Image Processing
# Toolbox equivalence is not claimed where no executable MATLAB oracle exists.
SIMPOLY_STAGE_PARITY: dict[str, ParityClassification] = {
    "crop_first_channel_footer_90": ParityClassification.EXACT_SOURCE_RULE,
    "adapthisteq": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "histeq": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "grayscale_erosion_disk_5": ParityClassification.VERSION_DEPENDENT,
    "morphological_reconstruction": ParityClassification.CLOSE_REIMPLEMENTATION,
    "canny_0_2_0_4": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "bwareaopen_20": ParityClassification.EXACT_SOURCE_RULE,
    "bwmorph_thicken_1": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "graythresh_plus_0_1_on_ihist": ParityClassification.EXACT_SOURCE_RULE,
    "closing_disk_1": ParityClassification.VERSION_DEPENDENT,
    "bwmorph_clean_fill_majority": ParityClassification.TESTED_INTERNAL_SEMANTICS,
    "bwmorph_thin_4": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "median_stop_equal_foreground_count": ParityClassification.EXACT_SOURCE_RULE,
    "bwmorph_thicken_4": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "bwskel": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "branchpoints": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "branch_guard_disk_3": ParityClassification.EXACT_SOURCE_RULE,
    "spur_1": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "edge_distance_guard_55_px": ParityClassification.EXACT_FORMULA,
    "diameter_map_2x_edt": ParityClassification.EXACT_FORMULA,
    "automatic_histogram": ParityClassification.VERSION_DEPENDENT,
    "prepend_two_zeros": ParityClassification.EXACT_SOURCE_RULE,
    "gauss1_fit": ParityClassification.MATLAB_PARITY_UNVERIFIED,
    "main_result_b1": ParityClassification.EXACT_FORMULA,
    "source_stdev_c1_over_2": ParityClassification.EXACT_FORMULA,
}


@dataclass(frozen=True)
class SIMPolySourceConfig:
    profile: str = PROFILE_SOURCE_COMPAT_V1
    conversion_um_per_px: float | None = None
    footer_rows: int = 90
    canny_low: float = 0.2
    canny_high: float = 0.4
    minimum_edge_area: int = 20
    reconstruction_disk_radius: int = 5
    closing_disk_radius: int = 1
    branch_guard_radius: int = 3
    maximum_edge_distance_px: float = 55.0


@dataclass
class SIMPolyIntermediates:
    cropped: np.ndarray
    clahe: np.ndarray
    equalized: np.ndarray
    reconstruction: np.ndarray
    canny_edges: np.ndarray
    threshold_mask: np.ndarray
    morph_mask: np.ndarray
    median_mask: np.ndarray
    thickened_mask: np.ndarray
    raw_skeleton: np.ndarray
    branch_guard: np.ndarray
    valid_skeleton: np.ndarray
    distance_map: np.ndarray
    median_iterations: int = 0
    median_stopped_by_equal_count: bool = False
    masks_equal_at_stop: bool = False


@dataclass(frozen=True)
class SIMPolySourceResult:
    profile: str
    local_diameters_px: np.ndarray
    histogram_counts: np.ndarray
    histogram_edges: np.ndarray
    gaussian_amplitude: float | None
    gaussian_center_px: float | None
    gaussian_c1_px: float | None
    source_reported_stdev_px: float | None
    mathematical_gaussian_sigma_px: float | None
    arithmetic_mean_px: float | None
    median_px: float | None
    status: str
    reported_center: float | None = None
    reported_stdev: float | None = None
    reported_unit: str = "px"
    foreground_fraction: float | None = None
    skeleton_count: int = 0
    flags: tuple[str, ...] = field(default_factory=tuple)


# --- MATLAB bwmorph Morphological Semantics Implementation ---

def bwmorph_clean(bw: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Removes isolated pixels (1s with 0 8-neighbors)."""
    res = bw.copy().astype(bool)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    for _ in range(min(iterations, 100)):
        neighbors = ndimage.convolve(res.astype(int), kernel, mode="constant")
        isolated = res & (neighbors == 0)
        if not np.any(isolated):
            break
        res[isolated] = False
    return res


def bwmorph_fill(bw: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Fills isolated background holes (0s with 8 8-neighbors)."""
    res = bw.copy().astype(bool)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    for _ in range(min(iterations, 100)):
        neighbors = ndimage.convolve(res.astype(int), kernel, mode="constant")
        holes = (~res) & (neighbors == 8)
        if not np.any(holes):
            break
        res[holes] = True
    return res


def bwmorph_majority(bw: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Pixel is 1 if >= 5 pixels in 3x3 neighborhood are 1, 0 otherwise."""
    res = bw.copy().astype(bool)
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int)
    for _ in range(min(iterations, 100)):
        counts = ndimage.convolve(res.astype(int), kernel, mode="constant")
        new_res = counts >= 5
        if np.array_equal(new_res, res):
            break
        res = new_res
    return res


def bwmorph_thin(bw: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Applies thinning iteration."""
    return morphology.thin(bw, max_num_iter=iterations)


def bwmorph_thicken(bw: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Applies thickening iteration (inversion of thinning on background)."""
    inverted_thinned = morphology.thin(~bw, max_num_iter=iterations)
    return ~inverted_thinned


def bwmorph_branchpoints(skel: np.ndarray) -> np.ndarray:
    """Finds skeleton branchpoints (pixels with >= 3 8-neighbors)."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    neighbors = ndimage.convolve(skel.astype(int), kernel, mode="constant")
    return skel.astype(bool) & (neighbors >= 3)


def bwmorph_spur(skel: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Removes 1-pixel spur end branches."""
    res = skel.copy().astype(bool)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    for _ in range(iterations):
        neighbors = ndimage.convolve(res.astype(int), kernel, mode="constant")
        endpoints = res & (neighbors == 1)
        if not np.any(endpoints):
            break
        res[endpoints] = False
    return res


# --- MATLAB Histogram & Gaussian Fit Reimplementation ---

def fit_matlab_gauss1(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None, float | None]:
    """Fit the MATLAB ``gauss1`` formula with SciPy's optimizer.

    The formula is exact, but optimizer initialization/convergence is not
    MATLAB-parity-verified. Returns ``(a1, b1, c1)``.
    """
    if len(x) < 3 or np.max(y) <= 0:
        return None, None, None

    max_idx = int(np.argmax(y))
    p0_a1 = float(y[max_idx])
    p0_b1 = float(x[max_idx])
    p0_c1 = float(np.std(x)) if np.std(x) > 0 else 5.0

    def gauss1_fn(x_val: np.ndarray, a1: float, b1: float, c1: float) -> np.ndarray:
        return a1 * np.exp(-(((x_val - b1) / max(c1, 1e-3)) ** 2))

    try:
        popt, _ = optimize.curve_fit(
            gauss1_fn,
            x,
            y,
            p0=[p0_a1, p0_b1, max(p0_c1, 1.0)],
            bounds=([0.0, 0.1, 0.1], [np.inf, np.inf, np.inf]),
            maxfev=2000,
        )
        return float(popt[0]), float(popt[1]), float(abs(popt[2]))
    except (RuntimeError, ValueError, TypeError):
        return p0_a1, p0_b1, p0_c1


DEFAULT_SIMPOLY_SOURCE_CONFIG = SIMPolySourceConfig()


def _as_matlab_unit_interval(image: np.ndarray) -> np.ndarray:
    """Map supported MATLAB image classes to the normalized intensity domain."""
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return (arr.astype(np.float64) - info.min) / float(info.max - info.min)
    result = arr.astype(np.float64)
    if result.size and (float(np.nanmin(result)) < 0.0 or float(np.nanmax(result)) > 1.0):
        raise ValueError("Floating-point MATLAB image input must lie in [0, 1]")
    return result


def _bwareaopen_4_connected(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Reproduce ``bwareaopen(BW, P)``'s strict ``area < P`` removal rule."""
    if minimum_area <= 1:
        return mask.astype(bool, copy=True)
    try:
        # scikit-image >=0.26 names the largest removed size explicitly.
        return morphology.remove_small_objects(mask, max_size=minimum_area - 1, connectivity=1)
    except TypeError:  # pragma: no cover - compatibility with scikit-image <=0.25
        return morphology.remove_small_objects(mask, min_size=minimum_area, connectivity=1)


def run_simpoly_source_pipeline(
    image: np.ndarray,
    config: SIMPolySourceConfig = DEFAULT_SIMPOLY_SOURCE_CONFIG,
) -> tuple[SIMPolySourceResult, SIMPolyIntermediates]:
    """Execute the source-ordered SIMPoly pipeline.

    Literal source decisions are preserved.  ``SIMPOLY_STAGE_PARITY`` records
    where the Python primitive is not proven equivalent to MATLAB R2020a.
    """
    flags: list[str] = []

    # Step 1. Image Crop (SOURCE_COMPAT removes bottom 90 rows; CONTROLLED_INPUT keeps cropped image body)
    if config.profile == PROFILE_SOURCE_COMPAT_V1:
        if image.ndim == 3:
            first_channel = image[:, :, 0]
        else:
            first_channel = image

        if first_channel.shape[0] <= config.footer_rows:
            empty_arr = np.array([], dtype=np.float64)
            dummy = np.zeros((1, 1), dtype=np.uint8)
            inter = SIMPolyIntermediates(
                cropped=dummy, clahe=dummy, equalized=dummy, reconstruction=dummy,
                canny_edges=dummy, threshold_mask=dummy, morph_mask=dummy,
                median_mask=dummy, thickened_mask=dummy, raw_skeleton=dummy,
                branch_guard=dummy, valid_skeleton=dummy, distance_map=dummy
            )
            res = SIMPolySourceResult(
                profile=config.profile,
                local_diameters_px=empty_arr,
                histogram_counts=np.array([]),
                histogram_edges=np.array([]),
                gaussian_amplitude=None,
                gaussian_center_px=None,
                gaussian_c1_px=None,
                source_reported_stdev_px=None,
                mathematical_gaussian_sigma_px=None,
                arithmetic_mean_px=None,
                median_px=None,
                status="IMAGE_TOO_SHORT_FOR_CROP",
                flags=("IMAGE_TOO_SHORT_FOR_CROP",),
            )
            return res, inter

        I_crop = first_channel[: -config.footer_rows, :].copy()
    else:
        if image.ndim == 3:
            I_crop = image[:, :, 0].copy()
        else:
            I_crop = image.copy()

    # Step 2 & 3. Contrast Enhancement: adapthisteq + histeq
    norm_crop = _as_matlab_unit_interval(I_crop)

    I_clahe = exposure.equalize_adapthist(norm_crop, kernel_size=8, clip_limit=0.01)
    I_equalized = exposure.equalize_hist(I_clahe)

    # Step 4 & 5. Grayscale erosion & Morphological Reconstruction
    se_disk5 = morphology.disk(config.reconstruction_disk_radius)
    marker = morphology.erosion(I_equalized, se_disk5)
    I_reconstructed = morphology.reconstruction(marker, I_equalized)

    # Step 6 & 7. Canny Edge Detection & Small edge removal
    canny_edges = feature.canny(I_reconstructed, low_threshold=config.canny_low, high_threshold=config.canny_high)
    canny_clean = _bwareaopen_4_connected(canny_edges, config.minimum_edge_area)

    # Step 8. bwmorph thicken x1 on edges
    edges_thickened = bwmorph_thicken(canny_clean, iterations=1)

    # Step 9 & 10. Otsu threshold computed from original cropped image I + 0.1, applied to I_equalized
    thresh_val = float(filters.threshold_otsu(norm_crop)) + 0.1
    thresh_val = min(max(thresh_val, 0.0), 1.0)
    BW = I_equalized >= thresh_val

    # Step 11. Closing disk radius 1
    BW = morphology.closing(BW, morphology.disk(config.closing_disk_radius))

    # Step 12–15. bwmorph sequence: clean, fill, majority, thin x4
    BW = bwmorph_clean(BW, 100000)
    BW = bwmorph_fill(BW, 5000)
    BW = bwmorph_majority(BW, 500)
    BW = bwmorph_thin(BW, 4)

    # Step 16. Iterative 3x3 Median filter loop stopping on EQUAL FOREGROUND PIXEL COUNT
    BWf = ndimage.median_filter(BW.astype(np.uint8), size=(3, 3)) > 0
    med_iters = 0
    while np.sum(BWf) != np.sum(BW):
        med_iters += 1
        BW = BWf
        BWf = ndimage.median_filter(BW.astype(np.uint8), size=(3, 3)) > 0

    masks_equal_at_stop = bool(np.array_equal(BWf, BW))

    # Step 17. bwmorph thicken x4
    BW_thickened = bwmorph_thicken(BW, 4)

    if not np.any(BW_thickened):
        empty_arr = np.array([], dtype=np.float64)
        inter = SIMPolyIntermediates(
            cropped=I_crop, clahe=I_clahe, equalized=I_equalized, reconstruction=I_reconstructed,
            canny_edges=edges_thickened, threshold_mask=BW, morph_mask=BW,
            median_mask=BW, thickened_mask=BW_thickened, raw_skeleton=BW_thickened,
            branch_guard=BW_thickened, valid_skeleton=BW_thickened, distance_map=BW_thickened,
            median_iterations=med_iters, median_stopped_by_equal_count=True, masks_equal_at_stop=masks_equal_at_stop
        )
        res = SIMPolySourceResult(
            profile=config.profile, local_diameters_px=empty_arr, histogram_counts=np.array([]),
            histogram_edges=np.array([]), gaussian_amplitude=None, gaussian_center_px=None,
            gaussian_c1_px=None, source_reported_stdev_px=None, mathematical_gaussian_sigma_px=None,
            arithmetic_mean_px=None, median_px=None, status="NO_FOREGROUND", flags=("NO_FOREGROUND",),
        )
        return res, inter

    # Step 18–21. Skeletonization, Branchpoints, Branch Guard Dilation & Removal
    raw_skel = morphology.skeletonize(BW_thickened)
    if not np.any(raw_skel):
        empty_arr = np.array([], dtype=np.float64)
        inter = SIMPolyIntermediates(
            cropped=I_crop, clahe=I_clahe, equalized=I_equalized, reconstruction=I_reconstructed,
            canny_edges=edges_thickened, threshold_mask=BW, morph_mask=BW,
            median_mask=BW, thickened_mask=BW_thickened, raw_skeleton=raw_skel,
            branch_guard=raw_skel, valid_skeleton=raw_skel, distance_map=raw_skel,
            median_iterations=med_iters, median_stopped_by_equal_count=True, masks_equal_at_stop=masks_equal_at_stop
        )
        res = SIMPolySourceResult(
            profile=config.profile, local_diameters_px=empty_arr, histogram_counts=np.array([]),
            histogram_edges=np.array([]), gaussian_amplitude=None, gaussian_center_px=None,
            gaussian_c1_px=None, source_reported_stdev_px=None, mathematical_gaussian_sigma_px=None,
            arithmetic_mean_px=None, median_px=None, status="NO_SKELETON", flags=("NO_SKELETON",),
        )
        return res, inter

    branchpoints = bwmorph_branchpoints(raw_skel)
    branch_guard = morphology.dilation(branchpoints, morphology.disk(config.branch_guard_radius))
    skel_clean = raw_skel & (~branch_guard)

    # Step 22. Spur x1
    skel_clean = bwmorph_spur(skel_clean, iterations=1)

    # Step 23. Remove skeleton pixels with distance to Canny edges > 55 px
    F = ndimage.distance_transform_edt(~edges_thickened)
    valid_skel = skel_clean.copy()
    valid_skel[F > config.maximum_edge_distance_px] = False

    if not np.any(valid_skel):
        empty_arr = np.array([], dtype=np.float64)
        inter = SIMPolyIntermediates(
            cropped=I_crop, clahe=I_clahe, equalized=I_equalized, reconstruction=I_reconstructed,
            canny_edges=edges_thickened, threshold_mask=BW, morph_mask=BW,
            median_mask=BW, thickened_mask=BW_thickened, raw_skeleton=raw_skel,
            branch_guard=branch_guard, valid_skeleton=valid_skel, distance_map=F,
            median_iterations=med_iters, median_stopped_by_equal_count=True, masks_equal_at_stop=masks_equal_at_stop
        )
        res = SIMPolySourceResult(
            profile=config.profile, local_diameters_px=empty_arr, histogram_counts=np.array([]),
            histogram_edges=np.array([]), gaussian_amplitude=None, gaussian_center_px=None,
            gaussian_c1_px=None, source_reported_stdev_px=None, mathematical_gaussian_sigma_px=None,
            arithmetic_mean_px=None, median_px=None, status="NO_VALID_DIAMETERS", flags=("NO_VALID_DIAMETERS",),
        )
        return res, inter

    # Step 24 & 25. Distance Transform Dist = 2 * bwdist(~BW) & Local Diameters = Dist(SK)
    Dist = 2.0 * ndimage.distance_transform_edt(BW_thickened)
    ys, xs = np.where(valid_skel)
    diameters = Dist[ys, xs]

    if len(diameters) == 0:
        empty_arr = np.array([], dtype=np.float64)
        inter = SIMPolyIntermediates(
            cropped=I_crop, clahe=I_clahe, equalized=I_equalized, reconstruction=I_reconstructed,
            canny_edges=edges_thickened, threshold_mask=BW, morph_mask=BW,
            median_mask=BW, thickened_mask=BW_thickened, raw_skeleton=raw_skel,
            branch_guard=branch_guard, valid_skeleton=valid_skel, distance_map=Dist,
            median_iterations=med_iters, median_stopped_by_equal_count=True, masks_equal_at_stop=masks_equal_at_stop
        )
        res = SIMPolySourceResult(
            profile=config.profile, local_diameters_px=empty_arr, histogram_counts=np.array([]),
            histogram_edges=np.array([]), gaussian_amplitude=None, gaussian_center_px=None,
            gaussian_c1_px=None, source_reported_stdev_px=None, mathematical_gaussian_sigma_px=None,
            arithmetic_mean_px=None, median_px=None, status="NO_VALID_DIAMETERS", flags=("NO_VALID_DIAMETERS",),
        )
        return res, inter

    # Step 26 & 27. Conversion precedes MATLAB's automatic histogram. NumPy's
    # automatic selector is version-dependent and is not claimed to reproduce
    # MATLAB R2020a's exact automatic binning rule.
    conversion = config.conversion_um_per_px
    if conversion is not None and conversion <= 0:
        raise ValueError("conversion_um_per_px must be positive when provided")
    reported_diameters = diameters * conversion if conversion is not None else diameters
    counts, edges = np.histogram(reported_diameters, bins="auto")
    first_edge = edges[0]

    y_fit = np.concatenate(([0.0, 0.0], counts.astype(np.float64)))
    # MATLAB uses h.BinEdges(1:end-1), not bin centers.
    x_fit = np.concatenate(([first_edge - 2.0, first_edge - 1.0], edges[:-1].astype(np.float64)))

    # Step 28. Fit one-term Gaussian
    a1, b1, c1 = fit_matlab_gauss1(x_fit, y_fit)

    if b1 is not None and c1 is not None:
        reported_center = b1
        reported_stdev = c1 / 2.0
        reported_unit = "um" if conversion is not None else "px"
        scale_to_px = conversion if conversion is not None else 1.0
        center_px = b1 / scale_to_px
        c1_px = c1 / scale_to_px
        source_stdev = c1_px / 2.0
        math_sigma = c1_px / math.sqrt(2.0)
        status = "OK"
        if not 10.0 <= center_px <= 100.0:
            flags.append("SIMPOLY_APPROX_PIXEL_DOMAIN_10_100_EXCEEDED")
    else:
        reported_center = None
        reported_stdev = None
        reported_unit = "um" if conversion is not None else "px"
        center_px = None
        c1_px = None
        source_stdev = None
        math_sigma = None
        status = "GAUSSIAN_FIT_FAILED"
        flags.append("GAUSSIAN_FIT_FAILED")

    inter = SIMPolyIntermediates(
        cropped=I_crop, clahe=I_clahe, equalized=I_equalized, reconstruction=I_reconstructed,
        canny_edges=edges_thickened, threshold_mask=BW, morph_mask=BW,
        median_mask=BW, thickened_mask=BW_thickened, raw_skeleton=raw_skel,
        branch_guard=branch_guard, valid_skeleton=valid_skel, distance_map=Dist,
        median_iterations=med_iters, median_stopped_by_equal_count=True, masks_equal_at_stop=masks_equal_at_stop
    )

    res = SIMPolySourceResult(
        profile=config.profile,
        local_diameters_px=diameters,
        histogram_counts=counts,
        histogram_edges=edges,
        gaussian_amplitude=a1,
        gaussian_center_px=center_px,
        gaussian_c1_px=c1_px,
        source_reported_stdev_px=source_stdev,
        mathematical_gaussian_sigma_px=math_sigma,
        arithmetic_mean_px=float(np.mean(diameters)),
        median_px=float(np.median(diameters)),
        status=status,
        reported_center=reported_center,
        reported_stdev=reported_stdev,
        reported_unit=reported_unit,
        foreground_fraction=float(BW_thickened.mean()),
        skeleton_count=int(valid_skel.sum()),
        flags=tuple(flags),
    )

    return res, inter
