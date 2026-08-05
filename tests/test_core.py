from __future__ import annotations

import numpy as np
import pytest

from fathom_fibers_quick.analysis import (
    classify_fibers,
    fiber_level_summary,
    measurement_statistics,
    one_click_measurement,
    section_level_summary,
    validate_measurement_geometry,
)
from fathom_fibers_quick.model import Calibration, ImageDocument, Measurement, Project
from fathom_fibers_quick.project_io import (
    SourceVerificationStatus,
    load_project,
    recalculate_and_validate_project,
    save_project,
    verify_project_source,
)
from fathom_fibers_quick.zeiss import _flatten_cz_sem, detect_footer, file_sha256


def test_root_execution_imports():
    import fathom_fibers_quick
    assert fathom_fibers_quick.__name__ == "fathom_fibers_quick"


def test_zeiss_flatten_and_calibration_fields():
    raw = {
        "ap_image_pixel_size": ("Image Pixel Size", 52.04, "nm"),
        "ap_mag": ("Mag", "1.73 K X"),
    }
    flat = _flatten_cz_sem(raw)
    assert flat["ap_image_pixel_size"] == 52.04
    assert flat["ap_image_pixel_size__unit"] == "nm"
    assert flat["ap_mag"] == "1.73 K X"


def test_footer_detection_preserves_rows_below_band():
    gray = np.full((300, 400), 80, dtype=np.uint8)
    gray[220:270] = 250
    gray[275:] = 100
    bounds = detect_footer(gray)
    assert bounds is not None
    assert bounds[0] <= 220
    assert bounds[1] < 280


def test_measurement_statistics_and_classification():
    measurements = [
        Measurement("M1", "F001", (0, 0), (1, 0), 1.0e-6, "manual"),
        Measurement("M2", "F001", (0, 0), (1, 0), 1.2e-6, "manual"),
        Measurement("M3", "F002", (0, 0), (1, 0), 4.0e-6, "manual"),
        Measurement("M4", "F002", (0, 0), (1, 0), 4.2e-6, "manual"),
    ]
    stats = measurement_statistics(measurements)
    assert stats["n_fibers"] == 2
    assert stats["n_measurements"] == 4
    mapping = classify_fibers(measurements, requested_k=2)
    assert mapping["F001"] != mapping["F002"]
    assert measurements[0].group == measurements[1].group


def test_project_round_trip(tmp_path):
    image_file = tmp_path / "image.tif"
    image_file.write_bytes(b"dummy image data")
    image_hash = file_sha256(image_file)

    image = ImageDocument(
        path=str(image_file),
        width_px=100,
        height_px=80,
        calibration=Calibration(1e-9, 1e-9, "test"),
        footer_bounds=(70, 75),
        source_sha256=image_hash,
    )
    project = Project(1, image, [Measurement("M0001", "F001", (1.5, 2.5), (3.5, 2.5), 2e-9, "manual")])
    path = save_project(project, tmp_path / "sample")
    loaded = load_project(path)
    assert loaded.image.footer_bounds == (70, 75)
    assert loaded.measurements[0].p1 == (1.5, 2.5)


def test_verify_project_source(tmp_path):
    image_file = tmp_path / "image.tif"
    image_file.write_bytes(b"test image bytes")
    actual_hash = file_sha256(image_file)

    # MATCH
    img = ImageDocument(str(image_file), 100, 100, Calibration(1e-9, 1e-9, "t"), source_sha256=actual_hash)
    proj = Project(1, img)
    res_match = verify_project_source(proj)
    assert res_match.status == SourceVerificationStatus.MATCH

    # MISSING
    img_missing = ImageDocument(str(tmp_path / "nonexistent.tif"), 100, 100, Calibration(1e-9, 1e-9, "t"), source_sha256=actual_hash)
    proj_missing = Project(1, img_missing)
    res_missing = verify_project_source(proj_missing)
    assert res_missing.status == SourceVerificationStatus.MISSING

    # MISMATCH
    img_mismatch = ImageDocument(str(image_file), 100, 100, Calibration(1e-9, 1e-9, "t"), source_sha256="wrong_hash_000")
    proj_mismatch = Project(1, img_mismatch)
    res_mismatch = verify_project_source(proj_mismatch)
    assert res_mismatch.status == SourceVerificationStatus.MISMATCH

    # UNVERIFIED
    img_unverified = ImageDocument(str(image_file), 100, 100, Calibration(1e-9, 1e-9, "t"), source_sha256=None)
    proj_unverified = Project(1, img_unverified)
    res_unverified = verify_project_source(proj_unverified)
    assert res_unverified.status == SourceVerificationStatus.UNVERIFIED


def test_recalculate_width_m_and_corrections(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0) # geometric distance = 10.0 * 1e-9 = 10e-9
    wrong_width = 999.0e-6 # deliberately incorrect stored width

    img = ImageDocument(str(tmp_path / "img.tif"), 100, 100, cal)
    m = Measurement("M0001", "F001", p1, p2, wrong_width, "manual")
    proj = Project(1, img, [m])

    corrections = recalculate_and_validate_project(proj)
    assert corrections == 1
    assert abs(proj.measurements[0].width_m - 10.0e-9) < 1e-15


def test_invalid_calibration():
    with pytest.raises(ValueError, match="Invalid pixel_size_x_m"):
        Calibration(0.0, 1e-9, "test")

    with pytest.raises(ValueError, match="Invalid pixel_size_x_m"):
        Calibration(-1e-9, 1e-9, "test")

    with pytest.raises(ValueError, match="Invalid pixel_size_x_m"):
        Calibration(float("nan"), 1e-9, "test")


def test_fiber_level_summary_equal_weighting():
    # Fiber F001 has 10 measurements of 1.0 µm
    m_f1 = [
        Measurement(f"M1_{i}", "F001", (0, 0), (1, 0), 1.0e-6, "manual")
        for i in range(10)
    ]
    # Fiber F002 has 1 measurement of 3.0 µm
    m_f2 = [
        Measurement("M2_0", "F002", (0, 0), (3, 0), 3.0e-6, "manual")
    ]
    all_measurements = m_f1 + m_f2

    f_stats = fiber_level_summary(all_measurements)
    s_stats = section_level_summary(all_measurements)

    assert f_stats["n_fibers"] == 2
    assert f_stats["n_measurements"] == 11
    # Fiber medians: F001 -> 1.0µm, F002 -> 3.0µm. Mean of medians = (1.0 + 3.0)/2 = 2.0µm
    assert abs(f_stats["mean_m"] - 2.0e-6) < 1e-12

    # Section level mean is heavily weighted towards F001: (10*1.0 + 1*3.0)/11 = 1.1818 µm
    assert s_stats["mean_m"] < 1.5e-6


def test_geometric_validation():
    # Out of bounds
    res_oob = validate_measurement_geometry((-5, 10), (50, 50), 100, 100)
    assert not res_oob.valid
    assert "fuera de la imagen" in res_oob.reason

    # Inside footer
    res_footer = validate_measurement_geometry((50, 50), (50, 105), 100, 200, footer_bounds=(100, 120))
    assert not res_footer.valid
    assert "footer" in res_footer.reason

    # Sub-minimum length
    res_min = validate_measurement_geometry((10, 10), (10.5, 10.0), 100, 100, min_length_px=2.0)
    assert not res_min.valid
    assert "inferior al mínimo" in res_min.reason

    # Exceed search radius
    res_max = validate_measurement_geometry((10, 10), (90, 10), 100, 100, max_length_px=50.0)
    assert not res_max.valid
    assert "supera el límite" in res_max.reason

    # Valid
    res_ok = validate_measurement_geometry((10, 10), (50, 50), 100, 100)
    assert res_ok.valid


def test_one_click_detects_bright_bar():
    image = np.full((160, 160), 20.0, dtype=np.float32)
    image[:, 70:91] = 220.0
    p1, p2, confidence = one_click_measurement(image, (80, 80), search_radius_px=45)
    width = np.linalg.norm(np.asarray(p2) - np.asarray(p1))
    assert 17 <= width <= 25
    assert confidence >= 0
