from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, optimize
from skimage import exposure, feature, filters, morphology

PROFILE_SOURCE_COMPAT_V1 = "SIMPOLY_SOURCE_COMPAT_V1"
PROFILE_CONTROLLED_INPUT_V1 = "SIMPOLY_CONTROLLED_INPUT_V1"


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
    """Fits MATLAB gauss1 model: y = a1 * exp(-((x - b1) / c1)^2).

    Returns (a1, b1, c1).
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


def run_simpoly_source_pipeline(
    image: np.ndarray,
    config: SIMPolySourceConfig = DEFAULT_SIMPOLY_SOURCE_CONFIG,
) -> tuple[SIMPolySourceResult, SIMPolyIntermediates]:
    """Executes the exact SIMPoly source pipeline following SIMPolyMatlabCode.m."""
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
    norm_crop = I_crop.astype(np.float32)
    if norm_crop.max() > 1.0:
        norm_crop /= 255.0

    I_clahe = exposure.equalize_adapthist(norm_crop, kernel_size=8, clip_limit=0.01)
    I_equalized = exposure.equalize_hist(I_clahe)

    # Step 4 & 5. Grayscale erosion & Morphological Reconstruction
    se_disk5 = morphology.disk(config.reconstruction_disk_radius)
    marker = morphology.erosion(I_equalized, se_disk5)
    I_reconstructed = morphology.reconstruction(marker, I_equalized)

    # Step 6 & 7. Canny Edge Detection & Small edge removal
    canny_edges = feature.canny(I_reconstructed, low_threshold=config.canny_low, high_threshold=config.canny_high)
    try:
        canny_clean = morphology.remove_small_objects(canny_edges, max_size=config.minimum_edge_area)
    except TypeError:
        canny_clean = morphology.remove_small_objects(canny_edges, min_size=config.minimum_edge_area)

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
        if med_iters >= 500:
            break

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

    # Step 26 & 27. Automatic Histogram & Prepending two zero-count samples
    counts, edges = np.histogram(diameters, bins=30)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    first_edge = edges[0]

    y_fit = np.concatenate(([0.0, 0.0], counts.astype(np.float64)))
    x_fit = np.concatenate(([first_edge - 2.0, first_edge - 1.0], bin_centers.astype(np.float64)))

    # Step 28. Fit one-term Gaussian
    a1, b1, c1 = fit_matlab_gauss1(x_fit, y_fit)

    if b1 is not None and c1 is not None:
        source_stdev = c1 / 2.0
        math_sigma = c1 / math.sqrt(2.0)
        status = "OK"
    else:
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
        gaussian_center_px=b1,
        gaussian_c1_px=c1,
        source_reported_stdev_px=source_stdev,
        mathematical_gaussian_sigma_px=math_sigma,
        arithmetic_mean_px=float(np.mean(diameters)),
        median_px=float(np.median(diameters)),
        status=status,
        flags=tuple(flags),
    )

    return res, inter
