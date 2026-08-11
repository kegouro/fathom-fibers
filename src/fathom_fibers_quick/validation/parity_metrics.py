from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage, stats

STAGE_ORDER = (
    "INPUT",
    "CROP",
    "CLAHE",
    "HISTEQ",
    "EROSION",
    "RECONSTRUCTION",
    "CANNY",
    "AREA_FILTER",
    "EDGE_THICKEN",
    "OTSU",
    "THRESHOLD",
    "CLOSING",
    "CLEAN",
    "FILL",
    "MAJORITY",
    "THIN",
    "MEDIAN_LOOP",
    "THICKEN",
    "SKELETON",
    "BRANCHPOINTS",
    "BRANCH_GUARD",
    "SPUR",
    "EDGE_DISTANCE_FILTER",
    "EDT",
    "DIAMETERS",
    "HISTOGRAM",
    "GAUSSIAN_FIT",
)


def boolean_parity(matlab: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    left = np.asarray(matlab, dtype=bool)
    right = np.asarray(python, dtype=bool)
    if left.shape != right.shape:
        return {"shape_equal": False, "matlab_shape": left.shape, "python_shape": right.shape}
    intersection = int(np.count_nonzero(left & right))
    union = int(np.count_nonzero(left | right))
    left_n = int(np.count_nonzero(left))
    right_n = int(np.count_nonzero(right))
    different = int(np.count_nonzero(left != right))
    return {
        "shape_equal": True,
        "exact_equal": different == 0,
        "different_pixels": different,
        "different_fraction": different / left.size if left.size else 0.0,
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
        "dice": 2 * intersection / (left_n + right_n) if left_n + right_n else 1.0,
        "precision": intersection / right_n if right_n else (1.0 if left_n == 0 else 0.0),
        "recall": intersection / left_n if left_n else (1.0 if right_n == 0 else 0.0),
        "matlab_foreground": left_n,
        "python_foreground": right_n,
    }


def float_parity(
    matlab: np.ndarray,
    python: np.ndarray,
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> dict[str, Any]:
    left = np.asarray(matlab)
    right = np.asarray(python)
    result: dict[str, Any] = {
        "shape_equal": left.shape == right.shape,
        "matlab_shape": left.shape,
        "python_shape": right.shape,
        "matlab_dtype": str(left.dtype),
        "python_dtype": str(right.dtype),
    }
    if left.shape != right.shape:
        return result
    delta = left.astype(np.float64) - right.astype(np.float64)
    denominator = np.maximum(np.abs(left.astype(np.float64)), np.finfo(float).eps)
    result.update(
        max_abs_difference=float(np.max(np.abs(delta), initial=0.0)),
        mean_abs_difference=float(np.mean(np.abs(delta))) if delta.size else 0.0,
        rmse=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        max_relative_difference=float(np.max(np.abs(delta) / denominator, initial=0.0)),
        allclose=bool(np.allclose(left, right, rtol=rtol, atol=atol, equal_nan=True)),
        rtol=rtol,
        atol=atol,
    )
    return result


def skeleton_parity(matlab: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    result = boolean_parity(matlab, python)
    left = np.asarray(matlab, dtype=bool)
    right = np.asarray(python, dtype=bool)
    if left.shape != right.shape:
        return result
    distances_to_right = (
        ndimage.distance_transform_edt(~right)[left] if left.any() else np.array([])
    )
    distances_to_left = (
        ndimage.distance_transform_edt(~left)[right] if right.any() else np.array([])
    )
    distances = np.concatenate((distances_to_right, distances_to_left))
    result.update(
        overlap=int(np.count_nonzero(left & right)),
        median_skeleton_displacement=float(np.median(distances)) if distances.size else 0.0,
        p95_skeleton_displacement=float(np.quantile(distances, 0.95)) if distances.size else 0.0,
    )
    return result


def diameter_parity(matlab: np.ndarray, python: np.ndarray) -> dict[str, Any]:
    left = np.asarray(matlab, dtype=float).ravel()
    right = np.asarray(python, dtype=float).ravel()

    def summary(values: np.ndarray) -> dict[str, Any]:
        if not values.size:
            return {"n": 0, "mean": None, "median": None, "std": None, "quantiles": {}}
        return {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "quantiles": {
                str(q): float(np.quantile(values, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
            },
        }

    result = {"matlab": summary(left), "python": summary(right)}
    if left.size and right.size:
        result["ks_statistic"] = float(stats.ks_2samp(left, right).statistic)
        result["wasserstein_distance"] = float(stats.wasserstein_distance(left, right))
    else:
        result["ks_statistic"] = None
        result["wasserstein_distance"] = None
    result["max_aligned_difference"] = (
        float(np.max(np.abs(left - right))) if left.shape == right.shape and left.size else None
    )
    return result


def histogram_parity(
    matlab_counts: np.ndarray,
    matlab_edges: np.ndarray,
    python_counts: np.ndarray,
    python_edges: np.ndarray,
) -> dict[str, Any]:
    mc, me = np.ravel(matlab_counts), np.ravel(matlab_edges)
    pc, pe = np.ravel(python_counts), np.ravel(python_edges)
    return {
        "matlab_bin_count": int(mc.size),
        "python_bin_count": int(pc.size),
        "counts_equal": bool(np.array_equal(mc, pc)),
        "counts_max_difference": float(np.max(np.abs(mc - pc)))
        if mc.shape == pc.shape and mc.size
        else None,
        "edges_equal": bool(np.array_equal(me, pe)),
        "edges_max_difference": float(np.max(np.abs(me - pe)))
        if me.shape == pe.shape and me.size
        else None,
    }


def fit_parity(matlab: Mapping[str, float], python: Mapping[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("a1", "b1", "c1"):
        left, right = float(matlab[name]), float(python[name])
        absolute = abs(left - right)
        result[name] = {
            "matlab": left,
            "python": right,
            "absolute_difference": absolute,
            "relative_difference": absolute / abs(left) if left else None,
        }
    return result


def first_divergence(stage_matches: Mapping[str, bool]) -> str:
    for stage in STAGE_ORDER:
        if stage in stage_matches and not stage_matches[stage]:
            return stage
    return "MATCHED"


def first_divergence_record(
    stage_matches: Mapping[str, bool],
    stage_metrics: Mapping[str, Any],
    hypotheses: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    stage = first_divergence(stage_matches)
    return {
        "first_divergence": stage,
        "cause_hypothesis": (hypotheses or {}).get(
            stage, "No causal claim; inspect stage evidence."
        ),
        "evidence": stage_metrics.get(stage, {}),
    }
