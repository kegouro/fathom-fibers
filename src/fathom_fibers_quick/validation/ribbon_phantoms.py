"""Synthetic ribbon phantoms and known-truth evaluation helpers.

Shared by the oriented-ribbon test suite and the headless validation report.
Truth is defined operationally: the value each estimator returns when the
centerline is exactly at the known true centerline (same mask, same normal
convention, same algorithms).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from fathom_fibers_quick.core.centerline_refinement import refine_centerline
from fathom_fibers_quick.core.fiber_field import (
    IntensityProfileSampler,
    OrientedBoundaryEngine,
)
from fathom_fibers_quick.core.oriented_ribbon import compute_midpoint_observations
from fathom_fibers_quick.core.refined_remeasurement import remeasure_refined_centerline

PX = 5e-8
PY = 5e-8
TRUE_HALF_PX = 10.0
SEED_OFFSET_PX = 3.0


def straight_phantom(
    *,
    px: float = PX,
    py: float = PY,
    noise_seed: int | None = None,
    variable_radius: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Straight ribbon; mask at true center, seed offset +3 px along +n.

    Returns (mask, body, skeleton, samples, true_xy).  The band extends far
    beyond the sample columns so end caps never bias the EDT.
    """
    true_center = 47.0
    seed = true_center + SEED_OFFSET_PX
    height, width = 100, 200
    band_start, band_end = 0, 200
    col_start, col_end = 55, 145
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    half_rows = np.full(band_end - band_start, TRUE_HALF_PX)
    if variable_radius:
        cols = np.arange(band_start, band_end)
        half_rows = TRUE_HALF_PX * (1.0 + 0.2 * np.sin(2.0 * np.pi * cols / (band_end - band_start)))
    mask = np.zeros((height, width), dtype=bool)
    for index, col in enumerate(range(band_start, band_end)):
        low = round(true_center - half_rows[index])
        high = round(true_center + half_rows[index])
        mask[low:high + 1, col] = True
    body = np.where(mask, 200.0, 40.0).astype(float)
    skeleton = np.zeros((height, width), dtype=bool)
    skeleton[round(seed), col_start:col_end] = True
    n = col_end - col_start
    cols = np.arange(col_start, col_end)
    mid_rows = np.full(n, true_center)
    if rng is not None:
        mid_rows = mid_rows + rng.normal(0.0, 0.5, n)
    minus_rows = mid_rows - TRUE_HALF_PX
    plus_rows = mid_rows + TRUE_HALF_PX
    samples: dict[str, np.ndarray] = {
        "x_m": cols * px,
        "y_m": np.full(n, seed * py),
        "qx": np.ones(n),
        "qy": np.zeros(n),
        "coherence": np.ones(n),
        "normal_xy": np.tile(np.array([0.0, 1.0]), (n, 1)),
        "minus_xy_m": np.column_stack((cols * px, minus_rows * py)),
        "plus_xy_m": np.column_stack((cols * px, plus_rows * py)),
        "radius_minus_um": np.abs(seed - minus_rows) * py * 1e6,
        "radius_plus_um": np.abs(plus_rows - seed) * py * 1e6,
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": -(np.abs(seed - minus_rows) * py) * 1e6,
        "profile_plus_u_um": (np.abs(plus_rows - seed) * py) * 1e6,
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": np.full(n, round(seed)),
        "seed_col": cols,
    }
    true_xy = np.column_stack((cols * px, mid_rows * py))
    return mask, body, skeleton, samples, true_xy


def arc_phantom(
    *,
    px: float = PX,
    py: float = PY,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Annulus ribbon; true band at R +/- 10, seed skeleton at R + 3.

    The seed skeleton comes from skeletonize of a band centered at R + 3
    (a clean digital arc), while mask/body/EDT describe the true band at R.
    """
    radius = 400.0
    center_px = np.array([300.0, 360.0])
    height = int(center_px[1] + radius + 10 + 40)
    width = int(center_px[0] + radius + 10 + 40)
    rows, cols = np.mgrid[0:height, 0:width]
    distance = np.hypot((cols - center_px[0]) * px, (rows - center_px[1]) * py)
    true_inner = (radius - TRUE_HALF_PX) * py
    true_outer = (radius + TRUE_HALF_PX) * py
    mask = (distance >= true_inner) & (distance <= true_outer)
    body = np.where(mask, 200.0, 40.0).astype(float)
    seed_band_center = radius + SEED_OFFSET_PX
    seed_band = (
        (distance >= (seed_band_center - TRUE_HALF_PX) * py)
        & (distance <= (seed_band_center + TRUE_HALF_PX) * py)
    )
    skeleton = skeletonize(seed_band)
    chain_pixels = np.argwhere(skeleton)
    rows_px = chain_pixels[:, 0]
    cols_px = chain_pixels[:, 1]
    phi = np.arctan2(rows_px - center_px[1], cols_px - center_px[0])
    in_range = (phi >= 0.2) & (phi <= 1.4)
    rows_px = rows_px[in_range]
    cols_px = cols_px[in_range]
    phi = phi[in_range]
    n = rows_px.size
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    mid_radius = radius + (rng.normal(0.0, 0.5, n) if rng is not None else np.zeros(n))
    mid_rows = center_px[1] + mid_radius * np.sin(phi)
    mid_cols = center_px[0] + mid_radius * np.cos(phi)
    true_xy = np.column_stack((mid_cols * px, mid_rows * py))
    seed_xy = np.column_stack((cols_px * px, rows_px * py))
    normals = np.column_stack((mid_cols - center_px[0], mid_rows - center_px[1]))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    r_minus = (SEED_OFFSET_PX + TRUE_HALF_PX) * py
    r_plus = (TRUE_HALF_PX - SEED_OFFSET_PX) * py
    samples: dict[str, np.ndarray] = {
        "x_m": seed_xy[:, 0],
        "y_m": seed_xy[:, 1],
        "qx": np.ones(n),
        "qy": np.zeros(n),
        "coherence": np.ones(n),
        "normal_xy": normals,
        "minus_xy_m": seed_xy - r_minus * normals,
        "plus_xy_m": seed_xy + r_plus * normals,
        "radius_minus_um": np.full(n, r_minus * 1e6),
        "radius_plus_um": np.full(n, r_plus * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, -r_minus * 1e6),
        "profile_plus_u_um": np.full(n, r_plus * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": rows_px,
        "seed_col": cols_px,
    }
    return mask, body, skeleton, samples, true_xy


def rotated_phantom(angle_deg: float, *, px: float = PX, py: float = PY):
    """Rotated straight ribbon; seed displaced 3 px along the rotated normal."""
    theta = math.radians(angle_deg)
    tangent = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    length_px = 80
    height = width = length_px + 60
    start = np.array([30.0, 50.0])
    true_start = start - SEED_OFFSET_PX * normal
    rows_grid, cols_grid = np.mgrid[0:height, 0:width]
    xy = np.column_stack((cols_grid.ravel() * px, rows_grid.ravel() * py))
    offset_m = (
        (xy[:, 0] - true_start[0] * px) * normal[0]
        + (xy[:, 1] - true_start[1] * py) * normal[1]
    )
    mask = np.abs(offset_m) <= TRUE_HALF_PX * py
    mask = mask.reshape(height, width)
    body = np.where(mask, 200.0, 40.0).astype(float)
    skeleton = np.zeros((height, width), dtype=bool)
    pixels: list[tuple[int, int]] = []
    for index in range(length_px):
        position = start + index * tangent
        pixel = (round(position[1]), round(position[0]))
        if 0 <= pixel[0] < height and 0 <= pixel[1] < width:
            skeleton[pixel] = True
            if not pixels or pixels[-1] != pixel:
                pixels.append(pixel)
    rows = np.asarray([p[0] for p in pixels])
    cols = np.asarray([p[1] for p in pixels])
    n = rows.size
    seed_xy = np.asarray([start + index * tangent for index in range(length_px)], dtype=float)[:n]
    r_minus = (SEED_OFFSET_PX + TRUE_HALF_PX) * py
    r_plus = (TRUE_HALF_PX - SEED_OFFSET_PX) * py
    samples: dict[str, np.ndarray] = {
        "x_m": seed_xy[:, 0] * px,
        "y_m": seed_xy[:, 1] * py,
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.ones(n),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": seed_xy * np.asarray((px, py)) - r_minus * normal[None, :],
        "plus_xy_m": seed_xy * np.asarray((px, py)) + r_plus * normal[None, :],
        "radius_minus_um": np.full(n, r_minus * 1e6),
        "radius_plus_um": np.full(n, r_plus * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, -r_minus * 1e6),
        "profile_plus_u_um": np.full(n, r_plus * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": rows,
        "seed_col": cols,
    }
    true_xy = (seed_xy - SEED_OFFSET_PX * normal[None, :]) * np.asarray((px, py))
    return mask, body, skeleton, samples, true_xy


def raw_measurements(
    mask: np.ndarray,
    body: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    px: float,
    py: float,
) -> dict[str, np.ndarray]:
    """The un-refined estimators at the (displaced) seed centerline."""
    edt = ndimage.distance_transform_edt(mask, sampling=(py, px))
    radius = ndimage.map_coordinates(
        edt, [samples["y_m"] / py, samples["x_m"] / px], order=1, mode="nearest"
    )
    edt_um = 2.0 * radius * 1e6
    engine = OrientedBoundaryEngine()
    paired = engine.pair_centers(
        np.column_stack((samples["x_m"], samples["y_m"])),
        np.asarray(samples["normal_xy"], float),
        2.0 * radius,
        np.asarray(samples["coherence"], float),
        mask=mask,
        pixel_size_xy_m=(px, py),
    )
    profile = IntensityProfileSampler().refine(
        body, paired, pixel_size_xy_m=(px, py)
    )
    return {
        "edt_um": edt_um,
        "edge_um": np.asarray(paired.diameter_m) * 1e6,
        "edge_accepted": np.asarray(paired.accepted, bool),
        "profile_um": np.asarray(profile.diameter_m) * 1e6,
        "profile_accepted": np.asarray(profile.accepted, bool),
        "asymmetry": np.asarray(paired.asymmetry),
    }


def refined_pipeline(
    mask: np.ndarray,
    body: np.ndarray,
    skeleton: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    px: float,
    py: float,
):
    ribbon = compute_midpoint_observations(samples)
    smooth = refine_centerline(ribbon, samples, skeleton, pixel_size_xy_m=(px, py))
    edt = ndimage.distance_transform_edt(mask, sampling=(py, px))
    rem = remeasure_refined_centerline(
        smooth,
        mask=mask,
        body=body,
        edt_radius_m=edt,
        pixel_size_xy_m=(px, py),
        roi_origin_px=(0, 0),
        raw_coherence=np.asarray(samples["coherence"], float),
        raw_normal_xy=np.asarray(samples["normal_xy"], float),
        raw_qx=np.asarray(samples["qx"], float),
        raw_qy=np.asarray(samples["qy"], float),
    )
    return ribbon, smooth, rem


def mae(values: np.ndarray, truth_um: np.ndarray) -> float:
    return float(np.nanmean(np.abs(values - truth_um)))


def run_case(mask, body, skeleton, samples, true_xy, *, px: float = PX, py: float = PY):
    """Evaluate raw and refined estimators against the known true centerline."""
    n = samples["x_m"].size
    edt = ndimage.distance_transform_edt(np.asarray(mask, bool), sampling=(py, px))
    edt_truth = (
        2.0
        * ndimage.map_coordinates(
            edt, [true_xy[:, 1] / py, true_xy[:, 0] / px], order=1, mode="nearest"
        )
        * 1e6
    )
    engine = OrientedBoundaryEngine()
    sampler = IntensityProfileSampler()
    normals = np.asarray(samples["normal_xy"], float)
    paired_true = engine.pair_centers(
        true_xy,
        normals,
        edt_truth * 1e-6,
        np.asarray(samples["coherence"], float),
        mask=mask,
        pixel_size_xy_m=(px, py),
    )
    edge_truth = np.asarray(paired_true.diameter_m) * 1e6
    profile_truth = np.asarray(
        sampler.refine(body, paired_true, pixel_size_xy_m=(px, py)).diameter_m
    ) * 1e6
    raw = raw_measurements(mask, body, samples, px=px, py=py)
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=px, py=py)
    supported = rem.refined_mask
    edge_ok = rem.refined_edge_accepted
    profile_ok = rem.refined_profile_accepted
    seed_delta = np.hypot(
        samples["x_m"] - true_xy[:, 0],
        samples["y_m"] - true_xy[:, 1],
    )
    refined_pos = _smooth.refined_xy_m
    refined_delta = np.hypot(
        refined_pos[supported, 0] - true_xy[supported, 0],
        refined_pos[supported, 1] - true_xy[supported, 1],
    )
    return {
        "n": n,
        "supported": int(np.sum(supported)),
        "coverage": float(np.mean(supported)),
        "center_seed_mae": float(np.mean(seed_delta) * 1e6),
        "center_refined_mae": float(np.mean(refined_delta) * 1e6),
        "edt_raw_mae": mae(raw["edt_um"], edt_truth),
        "edt_refined_mae": mae(rem.refined_edt_um[supported], edt_truth[supported]),
        "edge_raw_mae": mae(raw["edge_um"], edge_truth),
        "edge_refined_mae": mae(rem.refined_edge_um[edge_ok], edge_truth[edge_ok]),
        "profile_raw_mae": mae(raw["profile_um"], profile_truth),
        "profile_refined_mae": mae(rem.refined_profile_um[profile_ok], profile_truth[profile_ok]),
        "asymmetry_raw": float(np.nanmedian(raw["asymmetry"])),
        "asymmetry_refined": float(np.nanmedian(rem.refined_asymmetry[edge_ok])),
        "rem": rem,
        "smooth": _smooth,
    }


__all__ = [
    "PX",
    "PY",
    "SEED_OFFSET_PX",
    "TRUE_HALF_PX",
    "arc_phantom",
    "mae",
    "raw_measurements",
    "refined_pipeline",
    "rotated_phantom",
    "run_case",
    "straight_phantom",
]
