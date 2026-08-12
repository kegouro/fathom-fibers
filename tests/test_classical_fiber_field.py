from __future__ import annotations

import numpy as np

from fathom_fibers_quick.core.fiber_field import (
    ClassicalFiberField,
    OrientedBoundaryEngine,
    double_angle_orientation,
)


def _field(mask: np.ndarray, *, spacing: tuple[float, float] = (1.0, 1.0)):
    return ClassicalFiberField(sigma_derivative=1.0, sigma_tensor=3.0).infer_field(
        mask.astype(float), mask=mask, pixel_size_xy_m=spacing
    )


def test_double_angle_representation_is_exactly_pi_periodic():
    theta = np.array([-1.2, -0.1, 0.0, 0.7, 2.3])
    qx, qy = double_angle_orientation(theta)
    shifted_x, shifted_y = double_angle_orientation(theta + np.pi)
    assert np.array_equal(qx, shifted_x)
    assert np.array_equal(qy, shifted_y)


def test_horizontal_vertical_and_diagonal_fields_have_expected_axis_encoding():
    horizontal = np.zeros((80, 120), dtype=bool)
    horizontal[30:50, 10:110] = True
    vertical = np.zeros((120, 80), dtype=bool)
    vertical[10:110, 30:50] = True
    diagonal = np.zeros((100, 100), dtype=bool)
    for offset in range(-8, 9):
        rows = np.arange(15, 85)
        diagonal[rows, rows + offset] = True

    h, v, d = _field(horizontal), _field(vertical), _field(diagonal)
    for result in (h, v, d):
        assert result.centerline is not None
        assert float(np.mean(result.coherence[result.centerline])) > 0.7
    assert float(np.mean(h.orientation_qx[h.centerline])) > 0.8
    assert float(np.mean(v.orientation_qx[v.centerline])) < -0.8
    assert float(np.mean(d.orientation_qy[d.centerline])) > 0.7
    assert np.isclose(np.median(h.diameter_px[h.centerline]), 20.0, atol=1.0)


def test_anisotropic_edt_uses_row_y_then_column_x_sampling():
    mask = np.zeros((31, 41), dtype=bool)
    mask[10:21, 5:36] = True
    result = _field(mask, spacing=(1.0, 2.0))
    # The limiting half-width is five row steps at 2 m each, not five pixels.
    assert np.isclose(result.radius_m[15, 20], 12.0, atol=1e-12)
    assert np.isclose(result.diameter_m[15, 20], 24.0, atol=1e-12)


def test_crossing_is_finite_and_does_not_claim_graph_topology():
    mask = np.zeros((101, 101), dtype=bool)
    mask[43:58, 10:91] = True
    mask[10:91, 43:58] = True
    result = _field(mask)
    assert result.centerline[50, 50]
    assert np.isfinite(result.coherence[50, 50])
    assert result.metadata["centerline_algorithm"] == "SKIMAGE_SKELETONIZE_MASK_BASELINE"


def test_curved_fiber_produces_finite_axis_and_edt_fields():
    yy, xx = np.ogrid[:121, :121]
    radius = np.hypot(xx - 60, yy - 60)
    mask = (radius >= 34) & (radius <= 46) & (yy < 65)
    result = _field(mask)
    support = result.centerline
    assert support.sum() > 30
    assert np.all(np.isfinite(result.orientation_qx[support]))
    assert np.all(np.isfinite(result.orientation_qy[support]))
    assert np.all(result.diameter_px[support] > 0)


def test_paired_edges_are_rotation_stable_and_agree_with_edt_on_ideal_bars():
    engine = OrientedBoundaryEngine(low_coherence=0.0, high_asymmetry=0.8)
    widths = []
    for shape, paint in (
        ((100, 140), lambda mask: mask.__setitem__((slice(40, 60), slice(15, 125)), True)),
        ((140, 100), lambda mask: mask.__setitem__((slice(15, 125), slice(40, 60)), True)),
        # Coordinate offsets are divided by sqrt(2) in physical normal width.
        ((120, 120), lambda mask: [mask.__setitem__((np.arange(20, 100), np.arange(20, 100) + offset), True) for offset in range(-14, 15)]),
    ):
        mask = np.zeros(shape, dtype=bool)
        paint(mask)
        field = _field(mask)
        paired = engine.pair_centerline(
            mask=mask, centerline=field.centerline, orientation_qx=field.orientation_qx,
            orientation_qy=field.orientation_qy, coherence=field.coherence,
            edt_diameter_m=field.diameter_m, pixel_size_xy_m=(1.0, 1.0),
        )
        values = paired.diameter_m[paired.accepted]
        edt = paired.edt_diameter_m[paired.accepted]
        assert values.size > 10
        widths.append(float(np.median(values)))
        assert np.isclose(np.median(values), np.median(edt), atol=1.5)
    assert max(widths) - min(widths) < 3.0


def test_paired_edges_respect_anisotropic_physical_spacing():
    mask = np.zeros((61, 101), dtype=bool)
    mask[20:41, 10:91] = True
    field = _field(mask, spacing=(1.0, 2.0))
    paired = OrientedBoundaryEngine(low_coherence=0.0, high_asymmetry=0.8).pair_centerline(
        mask=mask, centerline=field.centerline, orientation_qx=field.orientation_qx,
        orientation_qy=field.orientation_qy, coherence=field.coherence,
        edt_diameter_m=field.diameter_m, pixel_size_xy_m=(1.0, 2.0),
    )
    assert np.isclose(np.median(paired.diameter_m[paired.accepted]), 42.0, atol=2.0)


def test_crossing_marks_ambiguous_or_abstains_from_paired_widths():
    mask = np.zeros((101, 101), dtype=bool)
    mask[43:58, 10:91] = True
    mask[10:91, 43:58] = True
    field = _field(mask)
    paired = OrientedBoundaryEngine().pair_centerline(
        mask=mask, centerline=field.centerline, orientation_qx=field.orientation_qx,
        orientation_qy=field.orientation_qy, coherence=field.coherence,
        edt_diameter_m=field.diameter_m, pixel_size_xy_m=(1.0, 1.0),
    )
    center = np.argmin(np.sum((paired.center_xy_m - np.array([50.0, 50.0])) ** 2, axis=1))
    assert not paired.accepted[center] or "AMBIGUOUS_LOCAL_WIDTH" in paired.flags[center]
