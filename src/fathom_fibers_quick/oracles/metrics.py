from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np
from scipy import stats


class FailureTaxonomy(str, Enum):
    MATLAB_VERSION_DIFFERENCE = "MATLAB_VERSION_DIFFERENCE"
    TOOLBOX_VERSION_DIFFERENCE = "TOOLBOX_VERSION_DIFFERENCE"
    DATASET_DIFFERENCE = "DATASET_DIFFERENCE"
    INTERACTIVE_PARAMETER_DIFFERENCE = "INTERACTIVE_PARAMETER_DIFFERENCE"
    FIT_DIFFERENCE = "FIT_DIFFERENCE"
    ENHANCEMENT_DIFFERENCE = "ENHANCEMENT_DIFFERENCE"
    RECONSTRUCTION_DIFFERENCE = "RECONSTRUCTION_DIFFERENCE"
    EDGE_DIFFERENCE = "EDGE_DIFFERENCE"
    THRESHOLD_DIFFERENCE = "THRESHOLD_DIFFERENCE"
    MORPHOLOGY_DIFFERENCE = "MORPHOLOGY_DIFFERENCE"
    SKELETON_DIFFERENCE = "SKELETON_DIFFERENCE"
    DISTANCE_TRANSFORM_DIFFERENCE = "DISTANCE_TRANSFORM_DIFFERENCE"
    GAUSSIAN_FIT_DIFFERENCE = "GAUSSIAN_FIT_DIFFERENCE"
    UNKNOWN = "UNKNOWN"


def compute_mask_metrics(mask1: np.ndarray, mask2: np.ndarray) -> dict[str, float]:
    """Calculates IoU, Dice, foreground fractions, and structural parity between two binary masks."""
    m1 = np.asarray(mask1, dtype=bool)
    m2 = np.asarray(mask2, dtype=bool)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    iou = float(intersection / max(union, 1))
    dice = float(2 * intersection / max(m1.sum() + m2.sum(), 1))

    return {
        "iou": iou,
        "dice": dice,
        "foreground_fraction_1": float(m1.mean()),
        "foreground_fraction_2": float(m2.mean()),
    }


def compute_skeleton_metrics(skel1: np.ndarray, skel2: np.ndarray) -> dict[str, float]:
    """Calculates pixel count, overlap fraction, and structural similarity between two skeletons."""
    s1 = np.asarray(skel1, dtype=bool)
    s2 = np.asarray(skel2, dtype=bool)

    overlap = np.logical_and(s1, s2).sum()
    total_px = max(s1.sum() + s2.sum() - overlap, 1)

    return {
        "pixel_count_1": float(s1.sum()),
        "pixel_count_2": float(s2.sum()),
        "overlap_fraction": float(overlap / max(s1.sum(), 1)),
        "jaccard_skeleton": float(overlap / total_px),
    }


def compute_diameter_metrics(
    diams1: Sequence[float],
    diams2: Sequence[float],
    center1: float | None = None,
    center2: float | None = None,
) -> dict[str, float | None]:
    """Calculates MAE, RMSE, P90 relative error, Gaussian center difference, and Wasserstein distance."""
    d1 = np.asarray(diams1, dtype=np.float64)
    d2 = np.asarray(diams2, dtype=np.float64)

    if len(d1) == 0 or len(d2) == 0:
        return {
            "difference_of_gaussian_center": None,
            "difference_of_mean": None,
            "difference_of_median": None,
            "difference_of_std": None,
            "mae": None,
            "rmse": None,
            "p90_relative_error": None,
            "wasserstein_distance": None,
        }

    diff_center = (center1 - center2) if (center1 is not None and center2 is not None) else None
    diff_mean = float(np.mean(d1) - np.mean(d2))
    diff_median = float(np.median(d1) - np.median(d2))
    diff_std = float(np.std(d1) - np.std(d2))

    # Element-wise MAE / RMSE if lengths match
    if len(d1) == len(d2):
        abs_err = np.abs(d1 - d2)
        mae = float(np.mean(abs_err))
        rmse = float(np.sqrt(np.mean(abs_err**2)))
        rel_err = abs_err / np.maximum(d2, 1e-12)
        p90_rel = float(np.percentile(rel_err, 90))
    else:
        mae = float(abs(np.mean(d1) - np.mean(d2)))
        rmse = mae
        p90_rel = float(mae / max(np.mean(d2), 1e-12))

    # Wasserstein distance between distributions
    w_dist = float(stats.wasserstein_distance(d1, d2))

    return {
        "difference_of_gaussian_center": diff_center,
        "difference_of_mean": diff_mean,
        "difference_of_median": diff_median,
        "difference_of_std": diff_std,
        "mae": mae,
        "rmse": rmse,
        "p90_relative_error": p90_rel,
        "wasserstein_distance": w_dist,
    }
