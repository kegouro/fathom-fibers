from __future__ import annotations

import numpy as np

from fathom_fibers_quick.auto_roi import (
    PRESET_HIGH_MAG_FINE,
    PRESET_MID_MAG_GENERAL,
    AutoFiberCandidate,
    ResolutionPreset,
    analyze_roi,
    check_resolution_resolvability,
    get_preset_for_calibration,
    otsu_threshold,
)
from fathom_fibers_quick.model import Calibration, ImageDocument, Measurement, Project


def test_otsu_threshold_synthetic():
    gray = np.full((100, 100), 50, dtype=np.float32)
    gray[30:70, 30:70] = 200
    th = otsu_threshold(gray)
    assert 90 <= th <= 160


def test_gradient_background_and_irregular_illumination():
    y, x = np.ogrid[:150, :150]
    gradient_bg = (x * 0.5 + y * 0.3).astype(np.float32)
    gray = gradient_bg.copy()
    # Bright fiber
    gray[20:130, 65:85] += 120.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 140, 140),
        calibration=cal,
        threshold_method="Local Adaptativo",
        polarity="bright",
    )
    assert len(candidates) >= 1
    assert candidates[0].median_width_m is not None


def test_curved_component_tracing():
    gray = np.full((160, 160), 20.0, dtype=np.float32)
    # Draw curved arc
    for y_idx in range(20, 140):
        x_idx = int(70 + 25 * np.sin((y_idx - 20) / 30.0))
        gray[y_idx, max(0, x_idx - 6) : min(160, x_idx + 6)] = 220.0

    cal = Calibration(1e-9, 1e-9, "test")
    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(5, 5, 155, 155),
        calibration=cal,
        allow_curved_trace=True,
    )
    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.curved_trace_used or "CURVED_TRACE_USED" in cand.quality_flags


def test_merged_fibers_likely_merged_flag():
    gray = np.full((160, 160), 20.0, dtype=np.float32)
    # Huge merged block
    gray[30:130, 40:120] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 150, 150),
        calibration=cal,
        preset=PRESET_MID_MAG_GENERAL,
    )
    assert len(candidates) >= 1
    assert "LIKELY_MERGED" in candidates[0].quality_flags


def test_resolution_resolvability_gate():
    roi = np.full((100, 100), 50, dtype=np.float32)
    cal_low = Calibration(50e-9, 50e-9, "low_mag")  # 50nm/px -> expected 50nm fiber is 1px
    status, msg = check_resolution_resolvability(roi, cal_low, expected_width_m=50e-9)
    assert status == "RESOLUTION_INSUFFICIENT"
    assert "Resolución insuficiente" in msg

    roi_high = np.full((100, 100), 50, dtype=np.float32)
    roi_high[30:70, 40:60] = 200.0
    cal_high = Calibration(1e-9, 1e-9, "high_mag")  # 1nm/px -> expected 50nm fiber is 50px
    status_high, _msg_high = check_resolution_resolvability(roi_high, cal_high, expected_width_m=50e-9)
    assert status_high == "RESOLUTION_OK"


def test_preset_selection_logic():
    p1 = get_preset_for_calibration(Calibration(2e-9, 2e-9, "t"))
    assert p1.name == "HIGH_MAG_FINE"

    p2 = get_preset_for_calibration(Calibration(15e-9, 15e-9, "t"))
    assert p2.name == "MID_MAG_GENERAL"

    p3 = get_preset_for_calibration(Calibration(100e-9, 100e-9, "t"))
    assert p3.name == "LOW_MAG_NETWORK"


def test_dense_network_safety_zero_high_confidence():
    gray = np.full((150, 150), 20.0, dtype=np.float32)
    gray[20:130, 60:80] = 220.0
    cal_low = Calibration(180e-9, 180e-9, "low_mag")

    candidates, summary = analyze_roi(gray, (10, 10, 140, 140), cal_low)
    assert summary.high_confidence == 0
    assert summary.resolution_status == "RESOLUTION_INSUFFICIENT"
    for cand in candidates:
        assert cand.confidence_level == "Baja"


def test_preset_json_round_trip():
    p = PRESET_HIGH_MAG_FINE
    d = p.to_dict()
    restored = ResolutionPreset.from_dict(d)
    assert restored.name == p.name
    assert restored.min_area_px == p.min_area_px


def test_straight_bright_fiber_on_dark_bg():
    gray = np.full((150, 150), 20.0, dtype=np.float32)
    gray[20:130, 65:85] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 140, 140),
        calibration=cal,
        polarity="bright",
        min_area_px=30,
        min_elongation=2.0,
        n_sections=3,
    )
    assert len(candidates) >= 1
    cand = candidates[0]
    assert "LOW_ELONGATION" not in cand.quality_flags
    assert cand.confidence_level in {"Alta", "Media"}
    assert len(cand.proposed_measurements) >= 2


def test_dark_fiber_on_bright_bg():
    gray = np.full((150, 150), 230.0, dtype=np.float32)
    gray[20:130, 65:85] = 30.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 140, 140),
        calibration=cal,
        polarity="auto",
        min_area_px=30,
        min_elongation=2.0,
        n_sections=3,
    )
    assert len(candidates) >= 1
    cand = candidates[0]
    assert len(cand.proposed_measurements) >= 2


def test_two_parallel_separated_fibers():
    gray = np.full((160, 160), 20.0, dtype=np.float32)
    gray[20:140, 35:48] = 210.0
    gray[20:140, 110:123] = 210.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 150, 150),
        calibration=cal,
        polarity="bright",
        min_area_px=30,
        min_elongation=2.0,
    )
    assert len(candidates) == 2


def test_circular_object_fails_low_elongation():
    gray = np.full((120, 120), 20.0, dtype=np.float32)
    y, x = np.ogrid[:120, :120]
    mask = (x - 60) ** 2 + (y - 60) ** 2 <= 25 ** 2
    gray[mask] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 110, 110),
        calibration=cal,
        polarity="bright",
        min_elongation=2.5,
    )
    assert len(candidates) == 1
    assert "LOW_ELONGATION" in candidates[0].quality_flags


def test_component_touches_roi_edge():
    gray = np.full((120, 120), 20.0, dtype=np.float32)
    gray[0:80, 55:70] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(0, 0, 100, 100),
        calibration=cal,
        polarity="bright",
    )
    assert len(candidates) >= 1
    assert "TOUCHES_ROI_EDGE" in candidates[0].quality_flags


def test_too_small_component():
    gray = np.full((100, 100), 20.0, dtype=np.float32)
    gray[45:48, 45:48] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 90, 90),
        calibration=cal,
        min_area_px=50,
    )
    assert len(candidates) == 1
    assert "TOO_SMALL" in candidates[0].quality_flags


def test_anisotropic_pixels_physical_pca():
    gray = np.full((150, 150), 20.0, dtype=np.float32)
    gray[20:130, 60:80] = 220.0
    cal = Calibration(2e-9, 1e-9, "test")

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(10, 10, 140, 140),
        calibration=cal,
        polarity="bright",
    )
    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.median_width_m is not None
    assert 30e-9 <= cand.median_width_m <= 50e-9


def test_candidate_acceptance_to_project(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 200, 200, cal)
    proj = Project(1, img)

    gray = np.full((200, 200), 20.0, dtype=np.float32)
    gray[30:160, 90:110] = 220.0

    candidates, _summary = analyze_roi(
        gray=gray,
        roi_bbox=(20, 20, 180, 180),
        calibration=cal,
    )
    assert len(candidates) >= 1
    cand = candidates[0]

    fiber_id = proj.get_next_fiber_id()
    for pm in cand.proposed_measurements:
        m = Measurement(
            measurement_id=proj.next_measurement_id(),
            fiber_id=fiber_id,
            p1=pm.p1,
            p2=pm.p2,
            width_m=pm.width_m,
            method="AUTO_ROI_COMPONENT",
            confidence=cand.confidence_score,
        )
        proj.measurements.append(m)

    assert len(proj.measurements) == len(cand.proposed_measurements)
    assert proj.measurements[0].method == "AUTO_ROI_COMPONENT"
    assert proj.measurements[0].fiber_id == "F001"


def test_rejected_candidate_does_not_affect_statistics(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 200, 200, cal)
    proj = Project(1, img)

    AutoFiberCandidate(
        candidate_id="C001",
        roi_bbox=(10, 10, 50, 50),
        component_label=1,
        centerline_points=[],
        proposed_measurements=[],
        median_width_m=10e-9,
        confidence_score=0.2,
        status="REJECTED",
    )

    assert len(proj.measurements) == 0


def test_reproducibility_same_input():
    gray = np.full((120, 120), 20.0, dtype=np.float32)
    gray[20:100, 50:65] = 220.0
    cal = Calibration(1e-9, 1e-9, "test")

    c1, _s1 = analyze_roi(gray, (10, 10, 110, 110), cal)
    c2, _s2 = analyze_roi(gray, (10, 10, 110, 110), cal)

    assert len(c1) == len(c2)
    assert c1[0].confidence_score == c2[0].confidence_score
    assert c1[0].median_width_m == c2[0].median_width_m
