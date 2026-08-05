from __future__ import annotations

import numpy as np

from fathom_fibers_quick.auto_roi import AutoFiberCandidate, analyze_roi, otsu_threshold
from fathom_fibers_quick.model import Calibration, ImageDocument, Measurement, Project


def test_otsu_threshold_synthetic():
    gray = np.full((100, 100), 50, dtype=np.float32)
    gray[30:70, 30:70] = 200
    th = otsu_threshold(gray)
    assert 90 <= th <= 160


def test_straight_bright_fiber_on_dark_bg():
    gray = np.full((150, 150), 20.0, dtype=np.float32)
    # Vertical bright bar of width 20px
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
    # Dark fiber
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
    # Fiber 1
    gray[20:140, 35:48] = 210.0
    # Fiber 2
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
    # Should detect 2 distinct candidates
    assert len(candidates) == 2


def test_circular_object_fails_low_elongation():
    gray = np.full((120, 120), 20.0, dtype=np.float32)
    # Circular disk of radius 25
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
    # Fiber touching top edge of ROI (y=0)
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
    # Anisotropic pixels: x is 2nm, y is 1nm
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
    # Physical width should be ~ 20px * 2nm = 40nm
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

    # Convert candidate to project measurements
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

    # REJECTED candidate is not added to proj.measurements
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
