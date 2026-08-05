from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import ndimage, optimize
from skimage import exposure, feature, filters, morphology

from .model import Calibration

METHOD_NAME = "SIMPOLY_LITERATURE_REIMPLEMENTATION_V1"


def fit_1d_gaussian(data: Sequence[float], n_bins: int = 30) -> tuple[float, float, float]:
    """Fits 1D Gaussian y = A * exp(-(x - mu)^2 / (2 * sigma^2)) to extracted local diameters.

    Returns (center_mu, sigma, amplitude).
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0

    counts, bin_edges = np.histogram(arr, bins=n_bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    max_idx = np.argmax(counts)
    initial_mu = float(centers[max_idx])
    initial_sigma = float(np.std(arr)) if np.std(arr) > 0 else 1.0
    initial_amp = float(counts[max_idx])

    def gauss_fn(x: np.ndarray, a: float, mu: float, sig: float) -> np.ndarray:
        return a * np.exp(-((x - mu) ** 2) / (2.0 * max(sig, 1e-6) ** 2))

    try:
        popt, _ = optimize.curve_fit(
            gauss_fn,
            centers,
            counts,
            p0=[initial_amp, initial_mu, initial_sigma],
            bounds=([0.0, 0.0, 0.1], [np.inf, np.inf, np.inf]),
            maxfev=1000,
        )
        return float(popt[1]), float(abs(popt[2])), float(popt[0])
    except Exception:
        return initial_mu, initial_sigma, initial_amp


def run_simpoly_pipeline(
    gray: np.ndarray,
    calibration: Calibration,
    footer_bounds: tuple[int, int] | None = None,
    min_area_px: int = 50,
) -> dict[str, Any]:
    """Independent Python implementation of SIMPoly literature pipeline (V1)."""
    h, w = gray.shape[:2]
    usable_h = footer_bounds[0] if footer_bounds else h

    image_crop = gray[:usable_h, :].copy()
    norm_img = image_crop.astype(np.float32)
    if norm_img.max() > 1.0:
        norm_img /= 255.0

    # 1. CLAHE Contrast Enhancement
    enhanced = exposure.equalize_adapthist(norm_img, kernel_size=8, clip_limit=0.01)

    # 2. Morphological Reconstruction with disk erosion marker
    se_disk3 = morphology.disk(3)
    marker = morphology.erosion(enhanced, se_disk3)
    reconstructed = morphology.reconstruction(marker, enhanced)

    # 3. Canny Edge Detection
    canny_edges = feature.canny(reconstructed, sigma=1.0)

    # 4. Otsu Global Binarization
    thresh_val = filters.threshold_otsu(reconstructed)
    mask_initial = reconstructed >= thresh_val

    # 5. Component Filtering (remove small objects)
    try:
        mask_clean = morphology.remove_small_objects(mask_initial, max_size=min_area_px)
    except TypeError:
        mask_clean = morphology.remove_small_objects(mask_initial, min_size=min_area_px)

    # 6. Closing & Repeated Median Filtering
    se_disk2 = morphology.disk(2)
    mask_closed = morphology.closing(mask_clean, se_disk2)
    mask_filtered = ndimage.median_filter(mask_closed.astype(np.uint8), size=3) > 0

    # 7. Compensatory Dilation
    mask_dilated = morphology.dilation(mask_filtered, morphology.disk(1))

    # 8. Edge Overlay Cleanup
    clean_mask = mask_dilated.copy()

    # 9. Axial Skeletonization
    skel = morphology.skeletonize(clean_mask)

    # 10. Distance Transform (EDT)
    edt_map = ndimage.distance_transform_edt(clean_mask)

    # 11. Local Diameter Map = 2 * EDT along skeleton
    skel_ys, skel_xs = np.where(skel)
    local_diameters_px = (2.0 * edt_map[skel_ys, skel_xs]).tolist()

    local_diameters_m = [d_px * calibration.pixel_size_x_m for d_px in local_diameters_px]

    # 12. Gaussian Fit on Diameter Distribution
    gaussian_center_px, gaussian_sigma_px, _amp = fit_1d_gaussian(local_diameters_px)
    gaussian_center_m = gaussian_center_px * calibration.pixel_size_x_m

    arithmetic_mean_px = float(np.mean(local_diameters_px)) if local_diameters_px else 0.0
    median_px = float(np.median(local_diameters_px)) if local_diameters_px else 0.0
    std_px = float(np.std(local_diameters_px)) if local_diameters_px else 0.0

    return {
        "method_name": METHOD_NAME,
        "usable_shape": (usable_h, w),
        "enhanced": enhanced,
        "reconstructed": reconstructed,
        "canny_edges": canny_edges,
        "mask": clean_mask,
        "skeleton": skel,
        "edt_map": edt_map,
        "local_diameters_px": local_diameters_px,
        "local_diameters_m": local_diameters_m,
        "gaussian_center_px": gaussian_center_px,
        "gaussian_sigma_px": gaussian_sigma_px,
        "gaussian_center_m": gaussian_center_m,
        "arithmetic_mean_px": arithmetic_mean_px,
        "median_px": median_px,
        "std_px": std_px,
        "segmented_fraction": float(clean_mask.sum() / max(clean_mask.size, 1)),
    }
