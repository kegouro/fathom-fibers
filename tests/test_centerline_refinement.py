from __future__ import annotations

import math

import numpy as np
import pytest

from fathom_fibers_quick.core.centerline_refinement import (
    FLAG_NO_REFINEMENT,
    FLAG_SEGMENT_TOO_SHORT,
    STAGE,
    CenterlineSegment,
    order_seed_runs,
    refine_centerline,
)
from fathom_fibers_quick.core.oriented_ribbon import (
    CenterlineRefinementConfig,
    compute_midpoint_observations,
)

PX = 5e-8
PY = 5e-8


def build_straight_phantom(
    *,
    seed_row_px: int = 50,
    true_row_px: float = 47.0,
    half_width_px: float = 10.0,
    col_start: int = 1,
    col_end: int = 100,
    px: float = PX,
    py: float = PY,
    noise_seed: int | None = None,
    reject: slice | None = None,
    skeleton_rows: tuple[int, ...] = (50,),
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Horizontal ribbon phantom; returns (skeleton, samples, true_xy)."""
    height = max(skeleton_rows) + 20
    width = col_end + 20
    skeleton = np.zeros((height, width), dtype=bool)
    for row in skeleton_rows:
        skeleton[row, col_start:col_end] = True
    n = len(range(col_start, col_end))
    cols = np.arange(col_start, col_end)
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    mid_rows = (
        np.full(n, true_row_px)
        if rng is None
        else true_row_px + rng.normal(0.0, 0.6, n)
    )
    minus_rows = mid_rows - half_width_px
    plus_rows = mid_rows + half_width_px
    r_minus = np.abs(seed_row_px - minus_rows) * py
    r_plus = np.abs(plus_rows - seed_row_px) * py
    samples: dict[str, np.ndarray] = {
        "x_m": cols * px,
        "y_m": np.full(n, seed_row_px * py),
        "qx": np.ones(n),
        "qy": np.zeros(n),
        "coherence": np.ones(n),
        "normal_xy": np.tile(np.array([0.0, 1.0]), (n, 1)),
        "minus_xy_m": np.column_stack((cols * px, minus_rows * py)),
        "plus_xy_m": np.column_stack((cols * px, plus_rows * py)),
        "radius_minus_um": r_minus * 1e6,
        "radius_plus_um": r_plus * 1e6,
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": -(r_minus) * 1e6,
        "profile_plus_u_um": r_plus * 1e6,
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": np.full(n, seed_row_px),
        "seed_col": cols,
    }
    if reject is not None:
        samples["edge_accepted"][reject] = False
        samples["profile_accepted"][reject] = False
    true_xy = np.column_stack((cols * px, np.full(n, true_row_px * py)))
    return skeleton, samples, true_xy


def run_refinement(
    skeleton: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    pixel_size_xy_m: tuple[float, float] = (PX, PY),
    config: CenterlineRefinementConfig | None = None,
):
    base = compute_midpoint_observations(samples)
    return refine_centerline(
        base, samples, skeleton, pixel_size_xy_m=pixel_size_xy_m, config=config
    )


def mae_um(refined: np.ndarray, true_xy: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm(refined[mask] - true_xy[mask], axis=1).mean() * 1e6)


# ------------------------------------------------------------------- helpers


def test_seed_run_ordering_rejects_raster_order():
    """Samples arrive in raster order; ordering must recover path order."""
    skeleton, samples, _true_xy = build_straight_phantom()
    assert not np.array_equal(np.diff(samples["seed_col"]), np.ones(samples["seed_col"].size - 1)) is False
    result = run_refinement(skeleton, samples)
    assert result.summary["segment_count"] == 1
    assert result.refined_mask.sum() > 0.9 * samples["x_m"].size


def test_order_seed_runs_basic_chain_and_cut_points():
    skeleton = np.zeros((20, 40), dtype=bool)
    skeleton[10, 5:35] = True  # straight chain; endpoints are degree 1
    skeleton[11:13, 20] = True  # vertical stub makes (10,20) a degree-3 junction
    runs = order_seed_runs(skeleton, pixel_size_xy_m=(PX, PY))
    # the horizontal chain is split at the junction: the walk never bridges
    # the degree-3 pixel (10, 20), so no run contains it
    for run in runs:
        # no non-trivial run may contain the junction pixel (10, 20)
        if run.rows.size >= 2:
            assert not np.any((run.rows == 10) & (run.cols == 20))
    horizontal = [run for run in runs if np.all(run.rows == 10) and run.rows.size >= 2]
    assert len(horizontal) == 2  # left and right of the junction
    min_cols = sorted(run.cols.min() for run in horizontal)
    max_cols = sorted(run.cols.max() for run in horizontal)
    assert min_cols[0] == 6 and max_cols[-1] == 33


def test_order_seed_runs_arc_length_is_physical():
    skeleton = np.zeros((20, 20), dtype=bool)
    for index in range(10):
        skeleton[index, index] = True  # diagonal chain
    runs = order_seed_runs(skeleton, pixel_size_xy_m=(2.0, 5.0))
    assert len(runs) == 1
    # endpoints (0,0) and (9,9) are degree-1 cut points: 8 pixels, 7 steps
    expected = 7.0 * math.sqrt(2.0**2 + 5.0**2)
    assert runs[0].s_m[-1] == pytest.approx(expected, rel=1e-12)
    # first + last run pixel is direction invariant: (1,1) + (8,8)
    np.testing.assert_allclose(
        runs[0].xy_m[0] + runs[0].xy_m[-1], np.array([9.0 * 2.0, 9.0 * 5.0]), atol=1e-12
    )
    # every step is a physical diagonal of px, py
    steps = np.linalg.norm(np.diff(runs[0].xy_m, axis=0), axis=1)
    np.testing.assert_allclose(steps, np.full(steps.size, math.hypot(2.0, 5.0)), atol=1e-12)


# ----------------------------------------------------------------- straight


def test_straight_offset_refined_mae_below_seed():
    skeleton, samples, true_xy = build_straight_phantom()
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    seed_mae = mae_um(np.column_stack((samples["x_m"], samples["y_m"])), true_xy, mask)
    mid_mae = mae_um(result.preferred_midpoint_xy_m, true_xy, mask)
    refined_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert seed_mae == pytest.approx(3.0 * PY * 1e6, rel=1e-6)
    assert mid_mae < 1e-9
    assert refined_mae < seed_mae
    assert refined_mae < 1e-4 * seed_mae  # noiseless: essentially exact
    assert result.summary["segment_count"] == 1
    assert result.metadata["stage"] == STAGE


def test_straight_noisy_midpoints_are_smoothed():
    skeleton, samples, true_xy = build_straight_phantom(noise_seed=42)
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    raw_mae = mae_um(result.preferred_midpoint_xy_m, true_xy, mask)
    smooth_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert smooth_mae <= raw_mae
    assert smooth_mae < raw_mae * 0.5
    # smooth curve must not oscillate around the noisy midpoints
    deviations = np.abs(result.refined_xy_m[mask, 1] - true_xy[mask, 1])
    assert deviations.max() < 2.0 * PY * 1e6


def test_straight_smoothness_no_oscillation():
    skeleton, samples, true_xy = build_straight_phantom(noise_seed=3)
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    refined = result.refined_xy_m[mask]
    # perpendicular deviation from the straight true line
    deviation = np.abs(refined[:, 1] - true_xy[mask, 1])
    assert deviation.max() < 0.75 * PY * 1e6
    # second-difference norm remains small relative to the coordinate scale
    second_diff = np.diff(refined[:, 1], 2)
    assert np.linalg.norm(second_diff) < 0.05 * (refined[:, 1].max() - refined[:, 1].min()) + 1e-9


# ------------------------------------------------------------------- curved


def build_arc_phantom(
    *,
    radius_px: float = 400.0,
    offset_px: float = 3.0,
    half_width_px: float = 10.0,
    phi_range: tuple[float, float] = (0.2, 1.4),
    px: float = PX,
    py: float = PY,
    noise_seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Circular ribbon phantom; seed at radius+offset, true centerline radius.

    The ribbon is rendered as a filled annulus band and thinned with
    ``skimage.skeletonize`` so the seed is a real digital medial axis.  Each
    seed pixel carries boundaries at the true radius +/- half width along the
    radial normal; sample positions are the seed pixel positions.
    """
    from skimage.morphology import skeletonize

    center_px = np.array([300.0, 360.0])
    height = int(center_px[1] + radius_px + half_width_px + 40)
    width = int(center_px[0] + radius_px + half_width_px + 40)
    rows, cols = np.mgrid[0:height, 0:width]
    distance = np.hypot((cols - center_px[0]) * px, (rows - center_px[1]) * py)
    band_center = radius_px + offset_px
    inner = (band_center - half_width_px) * py
    outer = (band_center + half_width_px) * py
    band = (distance >= inner) & (distance <= outer)
    skeleton = skeletonize(band)
    chain_pixels = np.argwhere(skeleton)
    rng = np.random.default_rng(noise_seed) if noise_seed is not None else None
    noise = (
        np.zeros(chain_pixels.shape[0])
        if rng is None
        else rng.normal(0.0, 0.5, chain_pixels.shape[0])
    )
    rows_px = chain_pixels[:, 0]
    cols_px = chain_pixels[:, 1]
    phi = np.arctan2(rows_px - center_px[1], cols_px - center_px[0])
    in_range = (phi >= phi_range[0]) & (phi <= phi_range[1])
    rows_px = rows_px[in_range]
    cols_px = cols_px[in_range]
    phi = phi[in_range]
    noise = noise[in_range]
    n = rows_px.size
    mid_radius_px = radius_px + noise
    mid_rows = center_px[1] + mid_radius_px * np.sin(phi)
    mid_cols = center_px[0] + mid_radius_px * np.cos(phi)
    true_xy = np.column_stack((mid_cols * px, mid_rows * py))
    seed_xy = np.column_stack((cols_px * px, rows_px * py))
    normals = np.column_stack((mid_cols - center_px[0], mid_rows - center_px[1]))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    r_minus = (offset_px + half_width_px) * py
    r_plus = (half_width_px - offset_px) * py
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
    return skeleton, samples, true_xy


# ------------------------------------------------------------------- curved


def test_curved_offset_refined_mae_below_seed():
    skeleton, samples, true_xy = build_arc_phantom()
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    assert mask.sum() > 0.8 * true_xy.shape[0]
    seed_mae = mae_um(np.column_stack((samples["x_m"], samples["y_m"])), true_xy, mask)
    refined_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert seed_mae == pytest.approx(3.0 * PY * 1e6, rel=0.05)
    assert refined_mae < seed_mae
    assert refined_mae < seed_mae * 0.25
    assert result.summary["segment_count"] >= 1


def test_curved_noisy_midpoints_are_smoothed():
    skeleton, samples, true_xy = build_arc_phantom(noise_seed=11)
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    raw_mae = mae_um(result.preferred_midpoint_xy_m, true_xy, mask)
    smooth_mae = mae_um(result.refined_xy_m, true_xy, mask)
    # GCV keeps the arc's curvature, so the smoothing gain on this phantom is
    # modest; the requirement is that smoothing never makes it worse and the
    # residual stays well below a pixel
    assert smooth_mae <= raw_mae
    assert smooth_mae < 0.6 * PY * 1e6


# ------------------------------------------------------------ variable radius


def test_variable_radius_does_not_displace_centerline():
    skeleton, samples, true_xy = build_straight_phantom()
    n = samples["x_m"].size
    cols = np.asarray(samples["seed_col"], dtype=float)
    length = float(cols.max() - cols.min()) * PX
    radius = 10.0 * PY * (1.0 + 0.2 * np.sin(2.0 * np.pi * cols * PX / length))
    mid_rows = np.full(n, 47.0)
    samples["minus_xy_m"] = np.column_stack((samples["minus_xy_m"][:, 0], (mid_rows - radius / PY) * PY))
    samples["plus_xy_m"] = np.column_stack((samples["plus_xy_m"][:, 0], (mid_rows + radius / PY) * PY))
    samples["radius_minus_um"] = np.abs(50.0 - mid_rows + radius / PY) * PY * 1e6
    samples["radius_plus_um"] = np.abs(mid_rows + radius / PY - 50.0) * PY * 1e6
    samples["profile_minus_u_um"] = -(samples["radius_minus_um"])
    samples["profile_plus_u_um"] = samples["radius_plus_um"]
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    seed_mae = mae_um(np.column_stack((samples["x_m"], samples["y_m"])), true_xy, mask)
    refined_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert refined_mae < seed_mae
    assert refined_mae < 1e-4 * seed_mae
    # the refinement is not driven by width: the midpoint shift stays tiny.
    # the residual pattern at ~1e-15 m is numerical noise; only a strong
    # artificial displacement would pass the MAE gate above
    radius_norm = (radius / PY - 10.0) / 10.0
    shift = np.abs(result.refined_xy_m[mask, 1] - true_xy[mask, 1])
    correlation = np.corrcoef(radius_norm[mask], shift)[0, 1]
    assert abs(correlation) < 0.8


# ------------------------------------------------------------ gap and branch


def test_missing_observations_split_into_two_segments():
    skeleton, samples, _true_xy = build_straight_phantom()
    reject = slice(30, 45)
    samples["edge_accepted"][reject] = False
    samples["profile_accepted"][reject] = False
    result = run_refinement(skeleton, samples)
    assert result.summary["segment_count"] == 2
    ids = result.segment_ids
    assert set(ids[ids >= 0].tolist()) == {0, 1}
    # no refined points inside the gap; run endpoints are cut points
    assert not np.any(result.refined_mask[30:45])
    assert np.all(result.refined_mask[1:30])
    assert np.all(result.refined_mask[45:-1])


def test_branch_junction_cuts_run_no_spline_through_junction():
    height, width = 90, 120
    skeleton = np.zeros((height, width), dtype=bool)
    skeleton[50, 10:110] = True
    skeleton[45:50, 60] = True  # vertical stub connects to junction (50, 60)
    n = 100
    cols = np.arange(10, 110)
    samples = {
        "x_m": cols * PX,
        "y_m": np.full(n, 50.0 * PY),
        "qx": np.ones(n),
        "qy": np.zeros(n),
        "coherence": np.ones(n),
        "normal_xy": np.tile(np.array([0.0, 1.0]), (n, 1)),
        "minus_xy_m": np.column_stack((cols * PX, np.full(n, 37.0 * PY))),
        "plus_xy_m": np.column_stack((cols * PX, np.full(n, 57.0 * PY))),
        "radius_minus_um": np.full(n, 13.0 * PY * 1e6),
        "radius_plus_um": np.full(n, 7.0 * PY * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, -13.0 * PY * 1e6),
        "profile_plus_u_um": np.full(n, 7.0 * PY * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": np.full(n, 50),
        "seed_col": cols,
    }
    result = run_refinement(skeleton, samples)
    # horizontal line is split by the degree-3 junction into two segments
    assert result.summary["segment_count"] == 2
    junction_sample = 60 - 10  # pixel (50, 60) is a junction -> no refinement
    assert not result.refined_mask[junction_sample]
    # the pixels diagonally adjacent to the stub are cut points too
    assert not result.refined_mask[junction_sample - 1]
    assert not result.refined_mask[junction_sample + 1]
    left = result.refined_mask[1:junction_sample - 1]
    right = result.refined_mask[junction_sample + 2:-1]
    assert np.all(left) and np.all(right)


# ----------------------------------------------------------------- rotation


def rotated_line(angle_deg: float):
    theta = math.radians(angle_deg)
    tangent = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    length_px = 80
    height = width = length_px + 60
    skeleton = np.zeros((height, width), dtype=bool)
    pixels: list[tuple[int, int]] = []
    start = np.array([30.0, 50.0])
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
    r_minus = 13.0 * PY
    r_plus = 7.0 * PY
    samples = {
        "x_m": seed_xy[:, 0] * PX,
        "y_m": seed_xy[:, 1] * PY,
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.ones(n),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": seed_xy * np.asarray((PX, PY)) - r_minus * normal[None, :],
        "plus_xy_m": seed_xy * np.asarray((PX, PY)) + r_plus * normal[None, :],
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
    true_xy = (seed_xy - 3.0 * normal[None, :]) * np.asarray((PX, PY))
    return skeleton, samples, true_xy


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
def test_rotation_no_orientation_bias(angle_deg: float):
    skeleton, samples, true_xy = rotated_line(angle_deg)
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    assert mask.sum() > 0.7 * samples["x_m"].size
    seed_mae = mae_um(np.column_stack((samples["x_m"], samples["y_m"])), true_xy, mask)
    refined_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert refined_mae < seed_mae
    # staircase arc-length parameterization adds a small (<0.15 px) wiggle;
    # the refinement must still improve by an order of magnitude
    assert refined_mae < 0.1 * seed_mae


def test_rotation_mae_spread_is_small():
    mae_values = []
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        skeleton, samples, true_xy = rotated_line(angle)
        result = run_refinement(skeleton, samples)
        mask = result.refined_mask
        mae_values.append(mae_um(result.refined_xy_m, true_xy, mask))
    max_mae = max(mae_values)
    # no orientation introduces a strong bias: every angle stays below 0.15 px
    # while the seed offset is 3 px (a 20x+ improvement at every angle)
    assert max_mae < 0.15 * PY * 1e6
    assert max_mae < 0.05 * 3.0 * PY * 1e6


def test_anisotropic_calibration_physical_arc_length():
    px, py = 2.0, 5.0
    skeleton = np.zeros((20, 20), dtype=bool)
    for index in range(12):
        skeleton[index, index] = True
    runs = order_seed_runs(skeleton, pixel_size_xy_m=(px, py))
    # endpoints (0,0) and (11,11) are degree-1 cut points, so 9 steps remain
    assert runs[0].s_m[-1] == pytest.approx(9.0 * math.hypot(px, py), rel=1e-12)


def test_anisotropic_straight_refinement_is_physically_correct():
    px, py = 2.0, 5.0
    skeleton, samples, true_xy = build_straight_phantom(px=px, py=py)
    result = run_refinement(skeleton, samples, pixel_size_xy_m=(px, py))
    mask = result.refined_mask
    seed_mae = mae_um(np.column_stack((samples["x_m"], samples["y_m"])), true_xy, mask)
    refined_mae = mae_um(result.refined_xy_m, true_xy, mask)
    assert seed_mae == pytest.approx(3.0 * py * 1e6, rel=1e-6)
    assert refined_mae < seed_mae
    assert refined_mae < 1e-4 * seed_mae
    segment = result.segments[0]
    assert segment.length_m == pytest.approx(segment.s_m[-1] - segment.s_m[0], rel=1e-12)
    # 97 run pixels (endpoints excluded) at one physical step of px each
    assert segment.length_m == pytest.approx(96.0 * px, rel=1e-9)


# ------------------------------------------------------------ small segments


def test_too_short_segment_flagged_not_extrapolated():
    skeleton = np.zeros((30, 40), dtype=bool)
    skeleton[15, 5:35] = True
    cols = np.arange(5, 35)
    n = cols.size
    samples = {
        "x_m": cols * PX,
        "y_m": np.full(n, 15.0 * PY),
        "qx": np.ones(n),
        "qy": np.zeros(n),
        "coherence": np.ones(n),
        "normal_xy": np.tile(np.array([0.0, 1.0]), (n, 1)),
        "minus_xy_m": np.column_stack((cols * PX, np.full(n, 12.0 * PY))),
        "plus_xy_m": np.column_stack((cols * PX, np.full(n, 18.0 * PY))),
        "radius_minus_um": np.full(n, 6.0 * PY * 1e6),
        "radius_plus_um": np.full(n, 3.0 * PY * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, -6.0 * PY * 1e6),
        "profile_plus_u_um": np.full(n, 3.0 * PY * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 10.0),
        "seed_row": np.full(n, 15),
        "seed_col": cols,
    }
    config = CenterlineRefinementConfig(min_segment_points=40)
    result = run_refinement(skeleton, samples, config=config)
    assert result.summary["segment_count"] == 0
    assert not np.any(result.refined_mask)
    assert FLAG_SEGMENT_TOO_SHORT in result.flags


# ------------------------------------------------------------- sanity checks


def test_segments_are_typed_and_frozen():
    skeleton, samples, _true_xy = build_straight_phantom()
    result = run_refinement(skeleton, samples)
    segment = result.segments[0]
    assert isinstance(segment, CenterlineSegment)
    assert segment.source_indices.size >= 5
    assert segment.refined_xy_m.shape == (segment.source_indices.size, 2)
    assert segment.midpoint_source.size == segment.source_indices.size
    with pytest.raises((AttributeError, TypeError)):
        segment.refined_xy_m = None


def test_refined_xy_matches_original_indexing():
    skeleton, samples, _true_xy = build_straight_phantom()
    result = run_refinement(skeleton, samples)
    assert result.refined_xy_m.shape == (samples["x_m"].size, 2)
    assert result.refined_mask.shape == (samples["x_m"].size,)
    assert result.segment_ids.shape == (samples["x_m"].size,)
    for index in np.flatnonzero(result.refined_mask):
        assert np.isfinite(result.refined_xy_m[index]).all()
        assert result.segment_ids[index] >= 0
    for index in np.flatnonzero(~result.refined_mask):
        assert not np.isfinite(result.refined_xy_m[index]).any()


def test_smooth_shift_diagnostics_use_existing_normal_and_tangent():
    skeleton, samples, _true_xy = build_straight_phantom()
    result = run_refinement(skeleton, samples)
    mask = result.refined_mask
    normal = samples["normal_xy"][mask]
    tangent = np.column_stack((normal[:, 1], -normal[:, 0]))
    delta = result.refined_xy_m[mask] - np.column_stack((samples["x_m"][mask], samples["y_m"][mask]))
    expected_magnitude = np.linalg.norm(delta, axis=1) * 1e6
    np.testing.assert_allclose(result.smooth_shift_um[mask], expected_magnitude, rtol=1e-9)
    np.testing.assert_allclose(
        result.smooth_normal_shift_um[mask],
        (delta[:, 0] * normal[:, 0] + delta[:, 1] * normal[:, 1]) * 1e6,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        result.smooth_tangential_shift_um[mask],
        (delta[:, 0] * tangent[:, 0] + delta[:, 1] * tangent[:, 1]) * 1e6,
        rtol=1e-9,
    )


def test_missing_seed_keys_raise():
    skeleton, samples, _true_xy = build_straight_phantom()
    base = compute_midpoint_observations(samples)
    del samples["seed_row"]
    with pytest.raises(ValueError):
        refine_centerline(base, samples, skeleton, pixel_size_xy_m=(PX, PY))


def test_empty_skeleton_produces_no_segments():
    skeleton, samples, _true_xy = build_straight_phantom()
    base = compute_midpoint_observations(samples)
    result = refine_centerline(
        base, samples, np.zeros_like(skeleton), pixel_size_xy_m=(PX, PY)
    )
    assert result.summary["segment_count"] == 0
    assert not np.any(result.refined_mask)
    assert FLAG_NO_REFINEMENT in result.flags


def test_smoothing_strength_override_is_finite():
    skeleton, samples, _true_xy = build_straight_phantom(noise_seed=5)
    result = run_refinement(
        skeleton, samples, config=CenterlineRefinementConfig(smoothing_strength=5.0)
    )
    mask = result.refined_mask
    assert np.isfinite(result.refined_xy_m[mask]).all()
    assert result.summary["smooth_coverage"] > 0.5


def test_summary_contains_stage_two_keys():
    skeleton, samples, _true_xy = build_straight_phantom()
    result = run_refinement(skeleton, samples)
    for key in (
        "observation_coverage",
        "smooth_coverage",
        "segment_count",
        "accepted_observation_count",
        "smoothed_sample_count",
        "median_observed_shift_um",
        "median_smooth_shift_um",
        "p90_smooth_shift_um",
        "median_smooth_shift_fraction_of_width",
    ):
        assert key in result.summary, key
    assert result.summary["smoothed_sample_count"] == int(result.refined_mask.sum())
