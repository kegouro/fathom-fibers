from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.core.methods import MethodId
from fathom_fibers_quick.core.oriented_ribbon import (
    ALGORITHM_ID,
    STAGE,
    BoundaryMidpointObservation,
    CenterlineRefinementResult,
    compute_midpoint_observations,
)
from fathom_fibers_quick.model import Calibration

PX = 5e-8  # metres
PY = 5e-8


def straight_samples(
    *,
    true_center_px: float = 50.0,
    seed_offset_px: float = 3.0,
    half_width_px: float = 10.0,
    n: int = 3,
    x_px: float = 100.0,
    profile: bool = True,
) -> dict[str, np.ndarray]:
    """Horizontal ribbon; seed displaced by seed_offset_px along +n from truth.

    n = (0, 1): true center y=true_center_px, seed y=true_center_px+offset,
    boundaries at true_center_px +/- half_width_px.
    """
    true_center_y = true_center_px * PY
    seed_y = (true_center_px + seed_offset_px) * PY
    r_minus = (half_width_px + seed_offset_px) * PY
    r_plus = (half_width_px - seed_offset_px) * PY
    samples: dict[str, np.ndarray] = {
        "x_m": np.full(n, x_px * PX),
        "y_m": np.full(n, seed_y),
        "qx": np.full(n, 1.0),
        "qy": np.full(n, 0.0),
        "coherence": np.full(n, 1.0),
        "normal_xy": np.tile(np.array([0.0, 1.0]), (n, 1)),
        "minus_xy_m": np.column_stack((np.full(n, x_px * PX), np.full(n, true_center_y - half_width_px * PY))),
        "plus_xy_m": np.column_stack((np.full(n, x_px * PX), np.full(n, true_center_y + half_width_px * PY))),
        "radius_minus_um": np.full(n, r_minus * 1e6),
        "radius_plus_um": np.full(n, r_plus * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
    }
    if profile:
        u_minus = -(half_width_px + seed_offset_px) * PY
        u_plus = (half_width_px - seed_offset_px) * PY
        samples.update(
            {
                "profile_minus_u_um": np.full(n, u_minus * 1e6),
                "profile_plus_u_um": np.full(n, u_plus * 1e6),
                "profile_accepted": np.ones(n, bool),
                "profile_flags": np.full(n, "", dtype="<U80"),
                "profile_gradient_snr": np.full(n, 12.0),
            }
        )
    return samples


# --------------------------------------------------------------------- Test A


def test_straight_offset_midpoint_recovers_true_center():
    result = compute_midpoint_observations(straight_samples())
    assert isinstance(result, CenterlineRefinementResult)
    assert result.metadata["algorithm"] == ALGORITHM_ID
    assert result.metadata["stage"] == STAGE
    assert result.refined_xy_m is None
    observation = result.observations[0]
    midpoint = observation.mask_midpoint_xy_m
    assert midpoint is not None
    seed_error_px = abs(observation.original_xy_m[1] - 50.0 * PY) / PY
    midpoint_error_px = abs(midpoint[1] - 50.0 * PY) / PY
    assert seed_error_px == pytest.approx(3.0, abs=1e-9)
    assert midpoint_error_px < 1e-9
    assert midpoint_error_px < seed_error_px
    assert observation.mask_width_um == pytest.approx(20.0 * PY * 1e6, rel=1e-9)
    assert observation.signed_normal_shift_mask_um == pytest.approx(-3.0 * PY * 1e6, rel=1e-9)
    assert observation.shift_mask_um == pytest.approx(3.0 * PY * 1e6, rel=1e-9)
    assert observation.preferred_midpoint_source == "PROFILE"
    assert observation.accepted
    assert result.coverage_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------- Test B


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
def test_rotation_recovered_shift_is_angle_independent(angle_deg: float):
    theta = math.radians(angle_deg)
    tangent = np.array([math.cos(theta), math.sin(theta)])
    normal = np.array([-math.sin(theta), math.cos(theta)])
    injected_px = 3.0
    half_width_px = 10.0
    c0_m = np.array([50.0 * PX, 80.0 * PY])
    n = 5
    x_offsets = np.linspace(-20.0, 20.0, n)
    centers = c0_m[None, :] + np.outer(x_offsets * PX, tangent)
    samples: dict[str, np.ndarray] = {
        "x_m": centers[:, 0],
        "y_m": centers[:, 1],
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.full(n, 1.0),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": centers - (half_width_px + injected_px) * PY * normal[None, :],
        "plus_xy_m": centers + (half_width_px - injected_px) * PY * normal[None, :],
        "radius_minus_um": np.full(n, (half_width_px + injected_px) * PY * 1e6),
        "radius_plus_um": np.full(n, (half_width_px - injected_px) * PY * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
    }
    result = compute_midpoint_observations(samples)
    for observation in result.observations:
        midpoint = observation.mask_midpoint_xy_m
        assert midpoint is not None
        recovered = np.array(observation.shift_mask_um) * 1e-6
        assert recovered == pytest.approx(injected_px * PY, rel=1e-9, abs=1e-15)
        # direction: the shift must point from seed toward the true center,
        # i.e. opposite the injected seed offset, consistently with the normal
        expected_delta = np.array(observation.signed_normal_shift_mask_um) * 1e-6 * normal
        np.testing.assert_allclose(expected_delta, midpoint - np.array(observation.original_xy_m), atol=1e-15)
        assert observation.signed_normal_shift_mask_um == pytest.approx(-injected_px * PY * 1e6, rel=1e-9)
        # tangent reconstruction rule: t = (n_y, -n_x) equals analytic tangent
        np.testing.assert_allclose(observation.tangent_xy, tangent, atol=1e-12)
        np.testing.assert_allclose(observation.normal_xy, normal, atol=1e-12)
        assert observation.accepted
    assert result.coverage_fraction == pytest.approx(1.0)


# --------------------------------------------------------------------- Test C


def test_anisotropic_pixels_physical_geometry_correct():
    px, py = 2.0, 5.0  # physical metres per pixel, anisotropic
    theta = math.radians(30.0)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    injected_px = 3.0
    half_width_px = 10.0
    n = 2
    samples: dict[str, np.ndarray] = {
        "x_m": np.full(n, 7.0 * px),
        "y_m": np.full(n, 11.0 * py),
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.full(n, 0.9),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": (np.array([7.0 * px, 11.0 * py]) - (half_width_px + injected_px) * py * normal)[None, :] * np.ones((n, 1)),
        "plus_xy_m": (np.array([7.0 * px, 11.0 * py]) + (half_width_px - injected_px) * py * normal)[None, :] * np.ones((n, 1)),
        "radius_minus_um": np.full(n, (half_width_px + injected_px) * py * 1e6),
        "radius_plus_um": np.full(n, (half_width_px - injected_px) * py * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
    }
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    midpoint = observation.mask_midpoint_xy_m
    assert midpoint is not None
    # physical midpoint: seed - injected*normal (seed displaced +3px along normal)
    expected_midpoint = np.array([7.0 * px, 11.0 * py]) - injected_px * py * normal
    np.testing.assert_allclose(midpoint, expected_midpoint, atol=1e-12)
    # physical width = 20 * py (distances along the normal are in y-scale only)
    assert observation.mask_width_um == pytest.approx(20.0 * py * 1e6, rel=1e-9)
    assert observation.shift_mask_um == pytest.approx(injected_px * py * 1e6, rel=1e-9)
    assert observation.signed_normal_shift_mask_um == pytest.approx(-injected_px * py * 1e6, rel=1e-9)


# --------------------------------------------------------------------- Test D


def test_crossing_abstention():
    samples = straight_samples()
    samples["edge_flags"] = np.full(samples["x_m"].size, "POSSIBLE_CROSSING", dtype="<U80")
    samples.pop("profile_minus_u_um")
    samples.pop("profile_plus_u_um")
    samples.pop("profile_accepted")
    samples.pop("profile_flags")
    result = compute_midpoint_observations(samples)
    for observation in result.observations:
        assert not observation.accepted
        assert observation.preferred_midpoint_source is None
        assert observation.preferred_midpoint_xy_m is None
        assert "POSSIBLE_CROSSING" in observation.flags
    assert result.coverage_fraction == 0.0
    assert result.summary["accepted_count"] == 0


def test_ambiguous_local_width_abstention():
    samples = straight_samples()
    samples["edge_flags"] = np.full(samples["x_m"].size, "HIGH_ASYMMETRY;AMBIGUOUS_LOCAL_WIDTH", dtype="<U80")
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert not observation.accepted
    assert observation.preferred_midpoint_source is None
    assert "AMBIGUOUS_LOCAL_WIDTH" in observation.flags


def test_profile_ambiguity_falls_back_to_mask():
    samples = straight_samples()
    samples["profile_flags"] = np.full(samples["x_m"].size, "PROFILE_AMBIGUOUS_EDGE", dtype="<U80")
    samples["profile_accepted"] = np.zeros(samples["x_m"].size, bool)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert observation.accepted
    assert observation.preferred_midpoint_source == "MASK"
    assert observation.profile_midpoint_xy_m is None
    assert "PROFILE_AMBIGUOUS_EDGE" in observation.flags


# --------------------------------------------------------------- identities


def test_mask_geometric_identities():
    samples = straight_samples(profile=False)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    midpoint = observation.mask_midpoint_xy_m
    minus = observation.mask_minus_xy_m
    plus = observation.mask_plus_xy_m
    c0 = observation.original_xy_m
    n_vector = observation.normal_xy
    t_vector = observation.tangent_xy
    r_minus = samples["radius_minus_um"][0] * 1e-6
    r_plus = samples["radius_plus_um"][0] * 1e-6

    # m = (p- + p+)/2
    np.testing.assert_allclose(midpoint, 0.5 * (np.array(minus) + np.array(plus)), atol=1e-15)
    # width = ||p+ - p-||
    np.testing.assert_allclose(
        observation.mask_width_um, math.hypot(plus[0] - minus[0], plus[1] - minus[1]) * 1e6, atol=1e-9
    )
    # delta = ((r+ - r-)/2) * n
    expected_delta = ((r_plus - r_minus) / 2.0) * np.array(n_vector)
    np.testing.assert_allclose(np.array(midpoint) - np.array(c0), expected_delta, atol=1e-15)
    # normal shift = dot(m - c0, n); tangential = dot(m - c0, t)
    delta = np.array(midpoint) - np.array(c0)
    np.testing.assert_allclose(
        observation.signed_normal_shift_mask_um * 1e-6, np.dot(delta, n_vector), atol=1e-15
    )
    np.testing.assert_allclose(
        observation.tangential_shift_mask_um * 1e-6, np.dot(delta, t_vector), atol=1e-15
    )


def test_profile_reconstructed_width_matches_diameter():
    samples = straight_samples()
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    minus = observation.profile_minus_xy_m
    plus = observation.profile_plus_xy_m
    width = math.hypot(plus[0] - minus[0], plus[1] - minus[1]) * 1e6
    assert observation.profile_width_um == pytest.approx(width, rel=1e-12)
    expected_diameter = (samples["profile_plus_u_um"][0] - samples["profile_minus_u_um"][0]) * 1e-6
    assert observation.profile_width_um == pytest.approx(expected_diameter * 1e6, rel=1e-12)
    # reconstructed positions sit on the normal: c0 + u * n
    c0 = np.array(observation.original_xy_m)
    n_vector = np.array(observation.normal_xy)
    np.testing.assert_allclose(
        minus, c0 + samples["profile_minus_u_um"][0] * 1e-6 * n_vector, atol=1e-15
    )
    np.testing.assert_allclose(
        plus, c0 + samples["profile_plus_u_um"][0] * 1e-6 * n_vector, atol=1e-15
    )
    assert observation.preferred_midpoint_source == "PROFILE"


def test_preferred_midpoint_prefers_profile_over_mask():
    samples = straight_samples()
    result = compute_midpoint_observations(samples)
    assert all(observation.preferred_midpoint_source == "PROFILE" for observation in result.observations)
    profile_mid = result.observations[0].profile_midpoint_xy_m
    assert profile_mid is not None
    np.testing.assert_allclose(
        result.preferred_midpoint_xy_m,
        np.tile(profile_mid, (result.preferred_midpoint_xy_m.shape[0], 1)),
        atol=1e-15,
    )
    assert set(result.midpoint_source) == {"PROFILE"}


# -------------------------------------------------------------- edge cases


def test_missing_mask_boundary_is_rejected_with_flag():
    samples = straight_samples(profile=False)
    samples["plus_xy_m"][0] = np.nan
    samples["radius_plus_um"][0] = np.nan
    samples["edge_accepted"][0] = False
    result = compute_midpoint_observations(samples)
    first = result.observations[0]
    assert not first.accepted
    assert first.preferred_midpoint_source is None
    assert "MISSING_POSITIVE_EDGE" in first.flags
    second = result.observations[1]
    assert second.accepted
    assert second.preferred_midpoint_source == "MASK"


def test_nan_and_inf_geometry_do_not_crash():
    samples = straight_samples(profile=False)
    samples["x_m"][1] = np.nan
    samples["normal_xy"][2] = [np.inf, 0.0]
    samples["minus_xy_m"][2] = [np.nan, np.nan]
    result = compute_midpoint_observations(samples)
    assert result.observations[0].accepted
    assert result.observations[2].flags  # flagged, not crashed
    assert not result.observations[2].accepted


def test_zero_width_is_rejected():
    samples = straight_samples(profile=False)
    samples["plus_xy_m"] = samples["minus_xy_m"].copy()
    samples["radius_plus_um"] = samples["radius_minus_um"].copy()
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert not observation.accepted
    assert "MIDPOINT_ZERO_WIDTH" in observation.flags
    assert observation.preferred_midpoint_source is None


def test_invalid_normal_is_rejected_with_flag():
    samples = straight_samples(profile=False)
    samples["normal_xy"] = np.zeros((samples["x_m"].size, 2))
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert not observation.accepted
    assert "MIDPOINT_INVALID_NORMAL" in observation.flags


def test_low_coherence_rejects_with_flag():
    samples = straight_samples()
    samples["coherence"] = np.full(samples["x_m"].size, 0.02)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert not observation.accepted
    assert observation.preferred_midpoint_source is None
    assert "LOW_ORIENTATION_COHERENCE" in observation.flags
    assert observation.refinement_confidence == 0.0


def test_low_coherence_flag_from_field_is_honored():
    samples = straight_samples()
    samples["coherence"] = np.full(samples["x_m"].size, 0.5)
    samples["edge_flags"] = np.full(samples["x_m"].size, "LOW_ORIENTATION_COHERENCE", dtype="<U80")
    result = compute_midpoint_observations(samples)
    assert not result.observations[0].accepted


def test_rejected_edge_gives_zero_confidence_but_mask_midpoint_diagnostics():
    samples = straight_samples(profile=False)
    samples["edge_accepted"] = np.zeros(samples["x_m"].size, bool)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert not observation.accepted
    assert observation.preferred_midpoint_source is None
    assert observation.refinement_confidence == 0.0
    # raw boundary geometry diagnostics remain available, not accepted
    assert observation.mask_minus_xy_m is not None
    assert observation.mask_plus_xy_m is not None
    assert observation.mask_midpoint_xy_m is None


def test_missing_profile_keys_treated_as_absent():
    samples = straight_samples(profile=False)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert observation.profile_midpoint_xy_m is None
    assert observation.preferred_midpoint_source == "MASK"
    assert observation.accepted


def test_missing_required_key_raises():
    samples = straight_samples(profile=False)
    del samples["normal_xy"]
    with pytest.raises(ValueError):
        compute_midpoint_observations(samples)


def test_shape_mismatch_raises():
    samples = straight_samples(n=3)
    samples["coherence"] = np.zeros(4)
    with pytest.raises(ValueError):
        compute_midpoint_observations(samples)


def test_tangential_mismatch_diagnostic_flag_does_not_reject():
    samples = straight_samples(profile=False)
    samples["minus_xy_m"] = samples["minus_xy_m"].copy()
    samples["plus_xy_m"] = samples["plus_xy_m"].copy()
    # shift both boundaries tangentially so the width is preserved and the
    # midpoint gains a purely tangential component
    samples["minus_xy_m"][:, 0] += 40.0 * PX
    samples["plus_xy_m"][:, 0] += 40.0 * PX
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert abs(observation.tangential_shift_mask_um) > 1.0
    assert "MIDPOINT_TANGENTIAL_MISMATCH" in observation.flags
    assert observation.accepted  # diagnostic only


def test_mask_profile_disagreement_is_scored_not_flagged_by_default():
    samples = straight_samples()
    samples["profile_minus_u_um"] = samples["profile_minus_u_um"] - 2.0
    samples["profile_plus_u_um"] = samples["profile_plus_u_um"] - 2.0
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert observation.mask_profile_center_disagreement_m == pytest.approx(2.0e-6, rel=1e-9)
    assert observation.mask_profile_center_disagreement_fraction is not None
    assert not any(flag.startswith("MASK_PROFILE") for flag in observation.flags)


# ----------------------------------------------------------- field integration


def test_field_adapter_adds_ribbon_fields_additively():
    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels,
        calibration=Calibration(5e-9, 5e-9, "test"),
        image_id="synthetic",
    )
    comparison = engine.compare_all_methods(image)
    field = next(
        result for result in comparison.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )
    samples = field.local_samples
    n = int(samples["x_m"].size)
    for key in (
        "refine_accepted",
        "refine_confidence",
        "midpoint_mask_x_m",
        "midpoint_mask_y_m",
        "midpoint_profile_x_m",
        "midpoint_profile_y_m",
        "midpoint_preferred_x_m",
        "midpoint_preferred_y_m",
        "center_shift_um",
        "center_shift_signed_um",
        "center_shift_tangent_um",
        "midpoint_source",
    ):
        assert key in samples, key
        assert np.asarray(samples[key]).shape[0] == n, key
    assert field.native_statistics["refine_accepted_count"] > 0
    assert 0.0 <= field.native_statistics["refine_coverage_fraction"] <= 1.0
    assert "refine_median_shift_um" in field.native_statistics
    # existing science untouched: d_EDT / d_edge / d_profile still reported
    assert field.common_distribution is not None
    assert "FATHOM_FIELD_PAIRED_EDGE_DIAMETER" in field.secondary_distributions
    assert "FATHOM_FIELD_PROFILE_DIAMETER" in field.secondary_distributions


def test_field_adapter_ribbon_uses_realistic_shift_signs():
    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels,
        calibration=Calibration(5e-9, 5e-9, "test"),
        image_id="synthetic",
    )
    comparison = engine.compare_all_methods(image)
    field = next(
        result for result in comparison.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )
    samples = field.local_samples
    accepted = samples["refine_accepted"]
    assert np.any(accepted)
    # preferred midpoints must lie between the mask boundaries
    mask_x, mask_y = samples["midpoint_preferred_x_m"], samples["midpoint_preferred_y_m"]
    finite = np.isfinite(mask_x) & np.isfinite(mask_y) & accepted
    assert np.any(finite)
    c0 = np.column_stack((samples["x_m"], samples["y_m"]))
    delta = np.column_stack((mask_x - c0[:, 0], mask_y - c0[:, 1]))
    normal = samples["normal_xy"]
    signed = delta[finite, 0] * normal[finite, 0] + delta[finite, 1] * normal[finite, 1]
    np.testing.assert_allclose(signed, samples["center_shift_signed_um"][finite] * 1e-6, atol=1e-12)


def test_full_cache_round_trip_keeps_ribbon_fields():
    import tempfile

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
        cache.store_comparison("ribbon", comparison)
        loaded = cache.load_comparison("ribbon")
        assert loaded is not None
        original = next(
            r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
        )
        restored = next(
            r for r in loaded.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
        )
        assert set(original.local_samples) == set(restored.local_samples)
        for key in ("refine_accepted", "center_shift_um", "midpoint_source"):
            left = np.asarray(original.local_samples[key])
            right = np.asarray(restored.local_samples[key])
            if left.dtype.kind in {"U", "S"}:
                assert np.array_equal(left, right), key
            else:
                np.testing.assert_allclose(left, right, equal_nan=True)


def test_observations_are_frozen_typed_objects():
    samples = straight_samples()
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert isinstance(observation, BoundaryMidpointObservation)
    with pytest.raises((AttributeError, TypeError)):
        observation.mask_midpoint_xy_m = (0.0, 0.0)


def test_include_observations_false_skips_object_build():
    samples = straight_samples()
    result = compute_midpoint_observations(samples, include_observations=False)
    assert result.observations == ()
    assert result.preferred_midpoint_xy_m.shape == (samples["x_m"].size, 2)
    assert result.midpoint_source.size == samples["x_m"].size


def test_oriented_ribbon_core_imports_without_qt(tmp_path):
    import subprocess
    import sys

    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    code = (
        "import sys; import fathom_fibers_quick.core.oriented_ribbon; "
        "assert 'PySide6' not in sys.modules; print('qt-free ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-free ok" in result.stdout


def test_zero_shift_remains_finite():
    """A perfectly centered ribbon must report a finite 0.0 shift, not NaN."""
    samples = straight_samples(profile=False, seed_offset_px=0.0)
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    assert observation.accepted
    assert observation.shift_mask_um == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(result.shift_um).all()
    assert result.summary["median_shift_um"] == pytest.approx(0.0, abs=1e-9)


# ------------------------------------------------------- profile preflight


def _rotated_profile_samples(angle_deg: float, n: int = 4, profile_shift_um: float = 2.0) -> dict[str, np.ndarray]:
    theta = math.radians(angle_deg)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    c0_m = np.array([50.0 * PX, 80.0 * PY])
    half_width_px = 10.0
    offset_m = profile_shift_um * 1e-6  # seed displaced along +n by 2 µm
    centers = c0_m[None, :] + np.linspace(-15.0, 15.0, n)[:, None] * PX * np.array([math.cos(theta), math.sin(theta)])
    u_minus = -(half_width_px * PY + offset_m)
    u_plus = half_width_px * PY - offset_m
    samples: dict[str, np.ndarray] = {
        "x_m": centers[:, 0],
        "y_m": centers[:, 1],
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.full(n, 0.9),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": centers + u_minus * normal[None, :],
        "plus_xy_m": centers + u_plus * normal[None, :],
        "radius_minus_um": np.full(n, -u_minus * 1e6),
        "radius_plus_um": np.full(n, u_plus * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, u_minus * 1e6),
        "profile_plus_u_um": np.full(n, u_plus * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 9.0),
    }
    return samples


@pytest.mark.parametrize("angle_deg", [0.0, 15.0, 30.0, 45.0, 60.0, 90.0])
def test_profile_midpoint_rotation_preflight(angle_deg: float):
    """PROFILE midpoint reconstruction is rotation invariant in physical units."""
    samples = _rotated_profile_samples(angle_deg)
    result = compute_midpoint_observations(samples)
    for observation in result.observations:
        midpoint = observation.profile_midpoint_xy_m
        assert midpoint is not None
        assert observation.preferred_midpoint_source == "PROFILE"
        # seed displaced +2 µm along n → profile midpoint sits 2 µm opposite
        shift = np.array(observation.shift_profile_um) * 1e-6
        assert shift == pytest.approx(2.0e-6, rel=1e-9)
        assert observation.signed_normal_shift_profile_um == pytest.approx(-2.0, rel=1e-9)
        # width = 2 * 10px * PY regardless of angle
        assert observation.profile_width_um == pytest.approx(20.0 * PY * 1e6, rel=1e-9)
        # reconstruction identity: p_profile = c0 + u * n
        c0 = np.array(observation.original_xy_m)
        nv = np.array(observation.normal_xy)
        np.testing.assert_allclose(
            midpoint,
            c0 + 0.5 * (samples["profile_minus_u_um"][0] + samples["profile_plus_u_um"][0]) * 1e-6 * nv,
            atol=1e-15,
        )


def test_profile_anisotropy_preflight():
    """PROFILE midpoint is correct under anisotropic calibration (px != py)."""
    py = 5.0
    theta = math.radians(60.0)
    normal = np.array([-math.sin(theta), math.cos(theta)])
    n = 3
    centers = np.full((n, 2), 0.0)
    u_minus = -11.0 * py
    u_plus = 9.0 * py
    samples: dict[str, np.ndarray] = {
        "x_m": centers[:, 0],
        "y_m": centers[:, 1],
        "qx": np.full(n, math.cos(2 * theta)),
        "qy": np.full(n, math.sin(2 * theta)),
        "coherence": np.full(n, 0.9),
        "normal_xy": np.tile(normal, (n, 1)),
        "minus_xy_m": centers + u_minus * normal[None, :],
        "plus_xy_m": centers + u_plus * normal[None, :],
        "radius_minus_um": np.full(n, -u_minus * 1e6),
        "radius_plus_um": np.full(n, u_plus * 1e6),
        "edge_accepted": np.ones(n, bool),
        "edge_flags": np.full(n, "", dtype="<U80"),
        "profile_minus_u_um": np.full(n, u_minus * 1e6),
        "profile_plus_u_um": np.full(n, u_plus * 1e6),
        "profile_accepted": np.ones(n, bool),
        "profile_flags": np.full(n, "", dtype="<U80"),
        "profile_gradient_snr": np.full(n, 9.0),
    }
    result = compute_midpoint_observations(samples)
    observation = result.observations[0]
    midpoint = observation.profile_midpoint_xy_m
    assert midpoint is not None
    # profile midpoint = c0 + (u_minus + u_plus)/2 * n = -1 * py * n (physical)
    expected = 0.5 * (u_minus + u_plus) * normal
    np.testing.assert_allclose(midpoint, expected, atol=1e-15)
    assert observation.profile_width_um == pytest.approx((u_plus - u_minus) * 1e6, rel=1e-12)
    assert observation.shift_profile_um == pytest.approx(py * 1e6, rel=1e-9)
