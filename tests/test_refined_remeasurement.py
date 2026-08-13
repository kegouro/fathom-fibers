from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from fathom_fibers_quick.core.refined_remeasurement import (
    FLAG_ORIENTATION_DISAGREEMENT,
    RefinedRemeasurement,
)
from fathom_fibers_quick.validation.ribbon_phantoms import (
    PX,
    PY,
    arc_phantom,
    refined_pipeline,
    rotated_phantom,
    run_case,
    straight_phantom,
)

# ------------------------------------------------------------------- gates


def test_edt_refined_improves_on_straight_offset():
    mask, body, skeleton, samples, true_xy = straight_phantom()
    case = run_case(mask, body, skeleton, samples, true_xy)
    assert case["coverage"] > 0.8
    assert case["edt_refined_mae"] < case["edt_raw_mae"]
    # raw EDT is pulled toward the nearest boundary: the seed is 3 px from
    # the true center, so the raw diameter error is about 2 * 3 px
    assert case["edt_raw_mae"] > 4.0 * PY * 1e6
    assert case["edt_refined_mae"] < 0.3 * PY * 1e6


def test_edt_refined_improves_on_curved_offset():
    mask, body, skeleton, samples, true_xy = arc_phantom()
    case = run_case(mask, body, skeleton, samples, true_xy)
    assert case["coverage"] > 0.8
    assert case["edt_refined_mae"] < case["edt_raw_mae"]
    assert case["edt_refined_mae"] < 0.2 * case["edt_raw_mae"]


def test_edge_and_profile_do_not_degrade():
    mask, body, skeleton, samples, true_xy = straight_phantom()
    case = run_case(mask, body, skeleton, samples, true_xy)
    # paired edge is offset-insensitive by construction
    assert case["edge_refined_mae"] <= case["edge_raw_mae"] + 1e-6
    assert case["edge_refined_mae"] < 0.05 * PY * 1e6
    assert case["profile_refined_mae"] <= case["profile_raw_mae"] + 1e-6
    assert case["profile_refined_mae"] < 0.05 * PY * 1e6


def test_asymmetry_reduced_after_recentering():
    mask, body, skeleton, samples, true_xy = straight_phantom()
    case = run_case(mask, body, skeleton, samples, true_xy)
    # raw pairing from the displaced seed has 13/7 radii; refined is 10/10
    assert case["asymmetry_raw"] == pytest.approx(6.0 / 20.0, rel=0.1)
    assert case["asymmetry_refined"] < case["asymmetry_raw"]
    assert case["asymmetry_refined"] < 0.05


def test_variable_radius_follows_truth():
    mask, body, skeleton, samples, true_xy = straight_phantom(variable_radius=True)
    case = run_case(mask, body, skeleton, samples, true_xy)
    assert case["edt_refined_mae"] < case["edt_raw_mae"]
    # refined edge follows the known radius variation (truth from run_case)
    assert case["edge_refined_mae"] < 0.5 * PY * 1e6


def test_rotation_no_orientation_bias_in_remeasurement():
    maes = []
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        mask, body, skeleton, samples, true_xy = rotated_phantom(angle)
        case = run_case(mask, body, skeleton, samples, true_xy)
        assert case["coverage"] > 0.7
        assert case["edt_refined_mae"] < case["edt_raw_mae"]
        maes.append(case["edt_refined_mae"])
    assert max(maes) < 0.3 * PY * 1e6
    # no strong orientation bias: worst angle stays a small fraction of the
    # 6 px raw EDT bias that the refinement removes
    assert max(maes) < 0.1 * 6.0 * PY * 1e6


@pytest.mark.parametrize("noise_seed", [3, 42])
def test_noisy_midpoints_remain_stable(noise_seed: int):
    mask, body, skeleton, samples, true_xy = straight_phantom(noise_seed=noise_seed)
    case = run_case(mask, body, skeleton, samples, true_xy)
    assert case["edt_refined_mae"] < case["edt_raw_mae"]
    assert case["edge_refined_mae"] < 0.15 * PY * 1e6
    assert case["profile_refined_mae"] < 0.15 * PY * 1e6


# ------------------------------------------------------------- unit checks


def test_refined_tangent_central_difference_and_normal_sign():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    supported = rem.refined_mask
    tangent = rem.refined_tangent_xy[supported]
    normal = rem.refined_normal_xy[supported]
    raw_normal = np.asarray(samples["normal_xy"])[supported]
    # unit tangents along +x for the horizontal ribbon
    np.testing.assert_allclose(np.linalg.norm(tangent, axis=1), 1.0, atol=1e-9)
    assert np.all(np.abs(tangent[:, 1]) < 1e-6)
    # normal sign aligned with the raw normal: dot >= 0
    dots = normal[:, 0] * raw_normal[:, 0] + normal[:, 1] * raw_normal[:, 1]
    assert np.all(dots >= 0.0)
    np.testing.assert_allclose(normal[:, 1], 1.0, atol=1e-6)


def test_edt_subpixel_interpolation_and_anisotropy():
    # anisotropic calibration: px != py; the refined center sits at the true
    # center, so the refined EDT matches the phantom's own EDT there
    mask, body, skeleton, samples, true_xy = straight_phantom(px=2.0, py=5.0)
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=2.0, py=5.0)
    supported = rem.refined_mask
    edt = ndimage.distance_transform_edt(mask, sampling=(5.0, 2.0))
    row_frac = true_xy[supported, 1] / 5.0
    col_frac = true_xy[supported, 0] / 2.0
    expected = (
        2.0 * ndimage.map_coordinates(edt, [row_frac, col_frac], order=1, mode="nearest") * 1e6
    )
    np.testing.assert_allclose(rem.refined_edt_um[supported], expected, atol=1e-9)
    # physical value: radius 11 raster pixels (10 + half pixel) at py = 5 m
    assert np.median(rem.refined_edt_um[supported]) == pytest.approx(2.0 * 11.0 * 5.0 * 1e6, rel=0.02)
    # and the raw EDT at the displaced seed is biased (radius 8)
    case = run_case(mask, body, skeleton, samples, true_xy, px=2.0, py=5.0)
    assert case["edt_refined_mae"] < case["edt_raw_mae"]
    assert case["edt_refined_mae"] < 1e-6


def test_arc_length_weights_match_segment_geometry():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    supported = rem.refined_mask
    weights = rem.refined_arc_weight_m[supported]
    assert np.all(weights > 0)
    # one horizontal step of px per interior sample
    assert np.median(weights) == pytest.approx(PX, rel=0.2)
    # total weight approximates the supported chain length
    assert np.sum(weights) == pytest.approx((np.sum(supported) - 1) * PX, rel=0.05)


def test_refined_distributions_use_arc_length_weights():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    for name in (
        "FATHOM_FIELD_REFINED_EDT_DIAMETER",
        "FATHOM_FIELD_REFINED_EDGE_DIAMETER",
        "FATHOM_FIELD_REFINED_PROFILE_DIAMETER",
    ):
        distribution = rem.distributions[name]
        assert distribution.unit == "um"
        assert distribution.diameter.size > 10
        # weights sum to the physical supported arc length
        assert np.sum(distribution.weight) == pytest.approx(np.sum(rem.refined_arc_weight_m), rel=1e-9)


def test_raw_measurements_unavailable_outside_support():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    unsupported = ~rem.refined_mask
    assert np.any(unsupported)
    assert np.all(~np.isfinite(rem.refined_edt_um[unsupported]))
    assert not np.any(rem.refined_edge_accepted[unsupported])


def test_crossing_abstention_no_refined_diameter():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    samples["edge_flags"] = np.full(samples["x_m"].size, "POSSIBLE_CROSSING", dtype="<U80")
    samples["edge_accepted"] = np.zeros(samples["x_m"].size, bool)
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    assert not np.any(rem.refined_mask)
    assert not np.any(rem.refined_edge_accepted)
    assert np.all(~np.isfinite(rem.refined_edt_um))


def test_orientation_disagreement_diagnostic():
    mask, body, skeleton, samples, _true_xy = rotated_phantom(30.0)
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    supported = rem.refined_mask
    disagreement = rem.axis_disagreement_deg[supported]
    # refined tangent should agree axially with the field double-angle axis
    assert np.nanmedian(disagreement) < 2.0
    assert FLAG_ORIENTATION_DISAGREEMENT not in rem.flags


def test_remeasurement_result_typed_and_frozen():
    mask, body, skeleton, samples, _true_xy = straight_phantom()
    _ribbon, _smooth, rem = refined_pipeline(mask, body, skeleton, samples, px=PX, py=PY)
    assert isinstance(rem, RefinedRemeasurement)
    assert rem.metadata["stage"] == "REFINED_REMEASUREMENT"
    with pytest.raises((AttributeError, TypeError)):
        rem.refined_edt_um = None


def test_full_cache_round_trip_preserves_refined_arrays(tmp_path):
    import tempfile

    from fathom_fibers_quick.api import FathomEngine
    from fathom_fibers_quick.core.methods import MethodId
    from fathom_fibers_quick.model import Calibration
    from fathom_fibers_quick.workspace import WorkspaceCache

    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels,
        calibration=Calibration(5e-9, 5e-9, "test"),
        image_id="synthetic",
    )
    comparison = engine.compare_all_methods(image)
    with tempfile.TemporaryDirectory() as tmp:
        cache = WorkspaceCache(tmp)
        cache.store_comparison("synthetic", comparison)
        loaded = cache.load_comparison("synthetic")
        assert loaded is not None
        original = next(
            r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
        )
        restored = next(
            r for r in loaded.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
        )
        assert set(original.local_samples) == set(restored.local_samples)
        for key in (
            "refined_edt_um",
            "refined_edge_um",
            "refined_profile_um",
            "refined_edge_accepted",
            "refined_edge_flags",
            "residual_center_shift_um",
            "refined_arc_weight_m",
            "refined_normal_xy",
        ):
            left = np.asarray(original.local_samples[key])
            right = np.asarray(restored.local_samples[key])
            if left.dtype.kind in {"U", "S"}:
                assert np.array_equal(left, right), key
            else:
                np.testing.assert_allclose(left, right, equal_nan=True)
        # the three refined secondary distributions survive the round trip
        for name in (
            "FATHOM_FIELD_REFINED_EDT_DIAMETER",
            "FATHOM_FIELD_REFINED_EDGE_DIAMETER",
            "FATHOM_FIELD_REFINED_PROFILE_DIAMETER",
        ):
            assert name in restored.secondary_distributions
            np.testing.assert_allclose(
                original.secondary_distributions[name].diameter,
                restored.secondary_distributions[name].diameter,
            )
