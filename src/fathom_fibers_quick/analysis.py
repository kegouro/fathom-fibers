from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, sobel
from scipy.signal import find_peaks

from .model import Measurement


@dataclass(frozen=True, slots=True)
class GeometricValidationResult:
    valid: bool
    reason: str = ""


def validate_measurement_geometry(
    p1: tuple[float, float],
    p2: tuple[float, float],
    width_px: int,
    height_px: int,
    footer_bounds: tuple[int, int] | None = None,
    min_length_px: float = 2.0,
    max_length_px: float | None = None,
) -> GeometricValidationResult:
    if not (math.isfinite(p1[0]) and math.isfinite(p1[1])):
        return GeometricValidationResult(False, "Coordenada p1 no finita.")
    if not (math.isfinite(p2[0]) and math.isfinite(p2[1])):
        return GeometricValidationResult(False, "Coordenada p2 no finita.")

    if not (0 <= p1[0] < width_px and 0 <= p1[1] < height_px):
        return GeometricValidationResult(False, "El extremo p1 queda fuera de la imagen.")
    if not (0 <= p2[0] < width_px and 0 <= p2[1] < height_px):
        return GeometricValidationResult(False, "El extremo p2 queda fuera de la imagen.")

    if footer_bounds is not None:
        y0, y1 = footer_bounds
        if (y0 <= p1[1] <= y1) or (y0 <= p2[1] <= y1):
            return GeometricValidationResult(False, "La sección termina dentro del footer excluido.")

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length_px = math.hypot(dx, dy)
    if length_px <= 0:
        return GeometricValidationResult(False, "El ancho proyectado debe ser un valor positivo.")
    if length_px < min_length_px:
        return GeometricValidationResult(
            False, f"El ancho ({length_px:.1f} px) es inferior al mínimo permitido ({min_length_px} px)."
        )
    if max_length_px is not None and length_px > max_length_px:
        return GeometricValidationResult(
            False, f"El ancho ({length_px:.1f} px) supera el límite del dominio de búsqueda ({max_length_px:.1f} px)."
        )

    return GeometricValidationResult(True)


def format_length_m(value_m: float) -> str:
    if value_m >= 1e-3:
        return f"{value_m * 1e3:.3f} mm"
    return f"{value_m * 1e9:.1f} nm"


def sample_profile(
    gray: np.ndarray,
    center: tuple[float, float],
    direction_xy: tuple[float, float],
    half_length: float,
    step: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    direction = np.asarray(direction_xy, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Direction vector cannot be zero")
    direction /= norm
    offsets = np.arange(-half_length, half_length + step, step)
    xs = center[0] + offsets * direction[0]
    ys = center[1] + offsets * direction[1]
    profile = map_coordinates(gray, [ys, xs], order=1, mode="nearest")
    return offsets, profile


def estimate_local_normal(
    gray: np.ndarray,
    point: tuple[float, float],
    radius: int = 24,
) -> tuple[tuple[float, float], float]:
    x, y = point
    x0 = max(0, round(x) - radius)
    x1 = min(gray.shape[1], round(x) + radius + 1)
    y0 = max(0, round(y) - radius)
    y1 = min(gray.shape[0], round(y) + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 25:
        raise ValueError("Point is too close to the image edge")
    smooth = gaussian_filter(patch.astype(float), 1.2)
    gx = sobel(smooth, axis=1)
    gy = sobel(smooth, axis=0)
    jxx = float(np.sum(gx * gx))
    jxy = float(np.sum(gx * gy))
    jyy = float(np.sum(gy * gy))
    matrix = np.array([[jxx, jxy], [jxy, jyy]], dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    normal = vectors[:, int(np.argmax(values))]
    anisotropy = float((values[-1] - values[0]) / (values[-1] + values[0] + 1e-12))
    return (float(normal[0]), float(normal[1])), max(0.0, min(1.0, anisotropy))


def _best_edge_pair(offsets: np.ndarray, profile: np.ndarray, center_tolerance: float = 3.0):
    smooth = gaussian_filter1d(profile.astype(float), 1.5)
    gradient = np.gradient(smooth, offsets)
    absolute = np.abs(gradient)
    prominence = max(float(np.std(gradient)) * 0.7, float(np.max(absolute)) * 0.04, 1e-9)
    peaks, _props = find_peaks(absolute, prominence=prominence, distance=max(2, int(2 / (offsets[1] - offsets[0]))))
    left = [int(i) for i in peaks if offsets[i] < -center_tolerance]
    right = [int(i) for i in peaks if offsets[i] > center_tolerance]
    if not left or not right:
        raise ValueError("No stable pair of edges was found")
    scale = float(np.percentile(absolute, 95) + 1e-12)
    best = None
    for li in left:
        for ri in right:
            width = offsets[ri] - offsets[li]
            if width < 3:
                continue
            sign_opposition = 1.0 if gradient[li] * gradient[ri] < 0 else 0.35
            edge_score = (absolute[li] + absolute[ri]) / (2 * scale)
            centered = math.exp(-abs((offsets[li] + offsets[ri]) / 2) / max(width, 1.0))
            span_penalty = math.exp(-width / max(offsets[-1] - offsets[0], 1.0) * 0.4)
            score = sign_opposition * edge_score * centered * span_penalty
            if best is None or score > best[0]:
                best = (score, li, ri, smooth, gradient)
    if best is None:
        raise ValueError("No plausible edge pair was found")
    return best


def one_click_measurement(
    gray: np.ndarray,
    point: tuple[float, float],
    search_radius_px: float = 60.0,
    footer_bounds: tuple[int, int] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    height, width = gray.shape[:2]
    normal, orientation_confidence = estimate_local_normal(gray, point)
    offsets, profile = sample_profile(gray, point, normal, search_radius_px)
    score, li, ri, _smooth, _gradient = _best_edge_pair(offsets, profile)
    p1 = (point[0] + offsets[li] * normal[0], point[1] + offsets[li] * normal[1])
    p2 = (point[0] + offsets[ri] * normal[0], point[1] + offsets[ri] * normal[1])

    validation = validate_measurement_geometry(
        p1, p2, width_px=width, height_px=height, footer_bounds=footer_bounds, max_length_px=2.0 * search_radius_px
    )
    if not validation.valid:
        raise ValueError(validation.reason)

    confidence = max(0.0, min(1.0, 0.55 * orientation_confidence + 0.45 * min(score, 1.0)))
    return p1, p2, confidence


def snap_two_click_edges(
    gray: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    footer_bounds: tuple[int, int] | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    height, width = gray.shape[:2]
    p1a = np.asarray(p1, dtype=float)
    p2a = np.asarray(p2, dtype=float)
    vector = p2a - p1a
    length = float(np.linalg.norm(vector))
    if length < 3:
        raise ValueError("The approximate diameter is too short")
    direction = vector / length
    center = tuple((p1a + p2a) / 2)
    half = max(length * 0.9, length / 2 + 8)
    offsets, profile = sample_profile(gray, center, tuple(direction), half)
    score, li, ri, _smooth, _gradient = _best_edge_pair(offsets, profile, center_tolerance=1.0)
    width_px = offsets[ri] - offsets[li]
    if width_px < 0.35 * length or width_px > 2.5 * length:
        return p1, p2, 0.25
    snapped1 = (center[0] + offsets[li] * direction[0], center[1] + offsets[li] * direction[1])
    snapped2 = (center[0] + offsets[ri] * direction[0], center[1] + offsets[ri] * direction[1])

    validation = validate_measurement_geometry(
        snapped1, snapped2, width_px=width, height_px=height, footer_bounds=footer_bounds
    )
    if not validation.valid:
        raise ValueError(validation.reason)

    return snapped1, snapped2, max(0.0, min(1.0, score))


def fiber_level_summary(measurements: Iterable[Measurement]) -> dict[str, float | int | None]:
    """Calculate summary statistics over the accepted median width of each fiber."""
    valid_by_fiber: dict[str, list[float]] = defaultdict(list)
    total_valid_measurements = 0
    for m in measurements:
        if m.accepted and np.isfinite(m.width_m) and m.width_m > 0:
            valid_by_fiber[m.fiber_id].append(m.width_m)
            total_valid_measurements += 1

    if not valid_by_fiber:
        return {
            "n_measurements": 0,
            "n_fibers": 0,
            "mean_m": None,
            "median_m": None,
            "min_m": None,
            "max_m": None,
            "std_m": None,
            "p05_m": None,
            "p95_m": None,
        }

    medians = np.asarray([float(np.median(vals)) for vals in valid_by_fiber.values()], dtype=float)
    return {
        "n_measurements": total_valid_measurements,
        "n_fibers": int(medians.size),
        "mean_m": float(medians.mean()),
        "median_m": float(np.median(medians)),
        "min_m": float(medians.min()),
        "max_m": float(medians.max()),
        "std_m": float(medians.std(ddof=1)) if medians.size > 1 else 0.0,
        "p05_m": float(np.quantile(medians, 0.05)),
        "p95_m": float(np.quantile(medians, 0.95)),
    }


def section_level_summary(measurements: Iterable[Measurement]) -> dict[str, float | int | None]:
    """Calculate summary statistics over all accepted individual sections."""
    return measurement_statistics(measurements)


def measurement_statistics(measurements: Iterable[Measurement]) -> dict[str, float | int | None]:
    valid = [m for m in measurements if m.accepted and np.isfinite(m.width_m) and m.width_m > 0]
    values = np.asarray([m.width_m for m in valid], dtype=float)
    fibers = {m.fiber_id for m in valid}
    if values.size == 0:
        return {
            "n_measurements": 0,
            "n_fibers": 0,
            "mean_m": None,
            "median_m": None,
            "min_m": None,
            "max_m": None,
            "std_m": None,
            "p05_m": None,
            "p95_m": None,
        }
    return {
        "n_measurements": int(values.size),
        "n_fibers": len(fibers),
        "mean_m": float(values.mean()),
        "median_m": float(np.median(values)),
        "min_m": float(values.min()),
        "max_m": float(values.max()),
        "std_m": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p05_m": float(np.quantile(values, 0.05)),
        "p95_m": float(np.quantile(values, 0.95)),
    }


def fiber_statistics(measurements: Iterable[Measurement]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for measurement in measurements:
        if measurement.accepted and measurement.width_m > 0:
            grouped[measurement.fiber_id].append(measurement.width_m)
    output: dict[str, dict[str, float | int]] = {}
    for fiber_id, values_list in grouped.items():
        values = np.asarray(values_list, dtype=float)
        output[fiber_id] = {
            "n": int(values.size),
            "mean_m": float(values.mean()),
            "median_m": float(np.median(values)),
            "min_m": float(values.min()),
            "max_m": float(values.max()),
            "std_m": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
    return output


def get_fiber_extrema(measurements: Iterable[Measurement], fiber_id: str) -> dict[str, list[str]]:
    """Returns mapping of measurement_id -> list of labels ('MIN', 'MED', 'MAX') for fiber_id."""
    accepted = [m for m in measurements if m.fiber_id == fiber_id and m.accepted and math.isfinite(m.width_m) and m.width_m > 0]
    if not accepted:
        return {}

    ordered = sorted(accepted, key=lambda m: m.width_m)
    extrema: dict[str, list[str]] = {}

    if len(ordered) == 1:
        extrema[ordered[0].measurement_id] = ["MIN", "MED", "MAX"]
        return extrema

    if len(ordered) == 2:
        extrema[ordered[0].measurement_id] = ["MIN"]
        extrema[ordered[1].measurement_id] = ["MAX"]
        return extrema

    median_val = float(np.median([m.width_m for m in ordered]))
    median_item = min(ordered, key=lambda m: abs(m.width_m - median_val))

    for m_item, label in ((ordered[0], "MIN"), (median_item, "MED"), (ordered[-1], "MAX")):
        extrema.setdefault(m_item.measurement_id, []).append(label)

    return extrema


def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    if k == 1:
        centers = np.array([values.mean()])
        labels = np.zeros(values.size, dtype=int)
        rss = float(np.sum((values - centers[0]) ** 2))
        return centers, labels, rss
    quantiles = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(values, quantiles).astype(float)
    labels = np.zeros(values.size, dtype=int)
    for _ in range(max_iter):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1)
        new_centers = np.array([
            values[new_labels == index].mean() if np.any(new_labels == index) else centers[index]
            for index in range(k)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            break
        labels, centers = new_labels, new_centers
    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    labels = np.array([remap[int(label)] for label in labels])
    centers = centers[order]
    rss = float(np.sum((values - centers[labels]) ** 2))
    return centers, labels, rss


def classify_fibers(
    measurements: list[Measurement],
    requested_k: int | None = None,
    maximum_k: int = 4,
) -> dict[str, int]:
    stats = fiber_statistics(measurements)
    fiber_ids = sorted(stats)
    if not fiber_ids:
        return {}
    values = np.log(np.asarray([stats[f]["median_m"] for f in fiber_ids], dtype=float))
    max_k = min(maximum_k, len(values))
    if requested_k is not None:
        k = max(1, min(int(requested_k), max_k))
        _centers, labels, _rss = _kmeans_1d(values, k)
    else:
        best = None
        for candidate_k in range(1, max_k + 1):
            centers, labels_candidate, rss = _kmeans_1d(values, candidate_k)
            variance = max(rss / max(len(values), 1), 1e-12)
            parameter_count = 2 * candidate_k
            bic = len(values) * math.log(variance) + parameter_count * math.log(max(len(values), 2))
            bic += candidate_k * 1.5
            if best is None or bic < best[0]:
                best = (bic, centers, labels_candidate)
        assert best is not None
        labels = best[2]
    mapping = {fiber_id: int(label) for fiber_id, label in zip(fiber_ids, labels, strict=True)}
    for measurement in measurements:
        measurement.group = mapping.get(measurement.fiber_id)
    return mapping


def classify_fibers_manual(
    measurements: list[Measurement],
    ranges: list[tuple[str, float, float]],
) -> dict[str, int]:
    """Classify fibers into manual ranges based on each fiber's accepted median width.

    ranges is a list of tuples: (group_name, min_width_m, max_width_m)
    """
    stats = fiber_statistics(measurements)
    mapping: dict[str, int] = {}
    for fiber_id, fiber_info in stats.items():
        median_m = float(fiber_info["median_m"])
        assigned_group = None
        for group_idx, (_name, min_m, max_m) in enumerate(ranges):
            if min_m <= median_m <= max_m:
                assigned_group = group_idx
                break
        if assigned_group is not None:
            mapping[fiber_id] = assigned_group

    for measurement in measurements:
        measurement.group = mapping.get(measurement.fiber_id)

    return mapping


def compute_histogram_data(
    measurements: Iterable[Measurement],
    mode: str = "fiber",
    n_bins: int = 10,
) -> dict[str, Any]:
    """Computes bin edges, counts, and items for interactive histogram rendering."""
    if mode == "fiber":
        fibers = fiber_statistics(measurements)
        items = [(fid, float(info["median_m"])) for fid, info in fibers.items()]
    else:
        items = [
            (m.measurement_id, m.width_m)
            for m in measurements
            if m.accepted and math.isfinite(m.width_m) and m.width_m > 0
        ]

    if not items:
        return {
            "mode": mode,
            "values": np.array([]),
            "counts": np.array([]),
            "bin_edges": np.array([]),
            "items": [],
            "mean_m": None,
            "median_m": None,
        }

    vals = np.array([val for _, val in items], dtype=float)
    counts, bin_edges = np.histogram(vals, bins=max(1, n_bins))

    return {
        "mode": mode,
        "values": vals,
        "counts": counts,
        "bin_edges": bin_edges,
        "items": items,
        "mean_m": float(vals.mean()),
        "median_m": float(np.median(vals)),
    }
