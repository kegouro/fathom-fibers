from __future__ import annotations

import numpy as np
import pytest

from fathom_fibers_quick.analysis import (
    classify_fibers,
    classify_fibers_manual,
    compute_histogram_data,
    fiber_level_summary,
    get_fiber_extrema,
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
    p2 = (10.0, 0.0)
    wrong_width = 999.0e-6

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
    m_f1 = [Measurement(f"M1_{i}", "F001", (0, 0), (1, 0), 1.0e-6, "manual") for i in range(10)]
    m_f2 = [Measurement("M2_0", "F002", (0, 0), (3, 0), 3.0e-6, "manual")]
    all_measurements = m_f1 + m_f2

    f_stats = fiber_level_summary(all_measurements)
    s_stats = section_level_summary(all_measurements)

    assert f_stats["n_fibers"] == 2
    assert f_stats["n_measurements"] == 11
    assert abs(f_stats["mean_m"] - 2.0e-6) < 1e-12
    assert s_stats["mean_m"] < 1.5e-6


def test_geometric_validation():
    res_oob = validate_measurement_geometry((-5, 10), (50, 50), 100, 100)
    assert not res_oob.valid
    assert "fuera de la imagen" in res_oob.reason

    res_footer = validate_measurement_geometry((50, 50), (50, 105), 100, 200, footer_bounds=(100, 120))
    assert not res_footer.valid
    assert "footer" in res_footer.reason

    res_min = validate_measurement_geometry((10, 10), (10.5, 10.0), 100, 100, min_length_px=2.0)
    assert not res_min.valid
    assert "inferior al mínimo" in res_min.reason

    res_max = validate_measurement_geometry((10, 10), (90, 10), 100, 100, max_length_px=50.0)
    assert not res_max.valid
    assert "supera el límite" in res_max.reason

    res_ok = validate_measurement_geometry((10, 10), (50, 50), 100, 100)
    assert res_ok.valid


def test_one_click_detects_bright_bar():
    image = np.full((160, 160), 20.0, dtype=np.float32)
    image[:, 70:91] = 220.0
    p1, p2, confidence = one_click_measurement(image, (80, 80), search_radius_px=45)
    width = np.linalg.norm(np.asarray(p2) - np.asarray(p1))
    assert 17 <= width <= 25
    assert confidence >= 0


def test_monotonic_fiber_id_generation():
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument("/tmp/dummy.tif", 100, 100, cal)
    proj = Project(1, img)

    id1 = proj.get_next_fiber_id()
    id2 = proj.get_next_fiber_id()
    assert id1 == "F001"
    assert id2 == "F002"


def test_protocol_completion_calculation():
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument("/tmp/dummy.tif", 100, 100, cal)
    proj = Project(1, img, target_sections=3)

    assert not proj.is_fiber_complete("F001")

    proj.measurements.append(Measurement("M1", "F001", (0, 0), (10, 0), 10e-9, "manual", accepted=True))
    proj.measurements.append(Measurement("M2", "F001", (0, 0), (12, 0), 12e-9, "manual", accepted=True))
    assert not proj.is_fiber_complete("F001")

    proj.measurements.append(Measurement("M3", "F001", (0, 0), (11, 0), 11e-9, "manual", accepted=True))
    assert proj.is_fiber_complete("F001")


def test_fiber_extrema_detection():
    measurements = [
        Measurement("M1", "F001", (0, 0), (1, 0), 1.0e-6, "manual", accepted=True),
        Measurement("M2", "F001", (0, 0), (2, 0), 2.0e-6, "manual", accepted=True),
        Measurement("M3", "F001", (0, 0), (3, 0), 3.0e-6, "manual", accepted=True),
    ]
    extrema = get_fiber_extrema(measurements, "F001")
    assert extrema["M1"] == ["MIN"]
    assert extrema["M2"] == ["MED"]
    assert extrema["M3"] == ["MAX"]


def test_manual_classification_by_fiber_median():
    measurements = [
        Measurement("M1", "F001", (0, 0), (1, 0), 0.5e-6, "manual", accepted=True),
        Measurement("M2", "F002", (0, 0), (2, 0), 2.0e-6, "manual", accepted=True),
    ]
    ranges = [
        ("Finas", 0.0, 1.0e-6),
        ("Medias", 1.0e-6, 3.0e-6),
    ]
    mapping = classify_fibers_manual(measurements, ranges)
    assert mapping["F001"] == 0
    assert mapping["F002"] == 1


def test_old_project_json_backward_compatibility(tmp_path):
    old_json_content = """{
        "schema_version": 1,
        "image": {
            "path": "/tmp/test.tif",
            "width_px": 100,
            "height_px": 100,
            "calibration": {"pixel_size_x_m": 1e-9, "pixel_size_y_m": 1e-9, "source": "test"}
        },
        "measurements": [
            {"measurement_id": "M0001", "fiber_id": "F001", "p1": [0, 0], "p2": [10, 0], "width_m": 10e-9, "method": "manual"}
        ]
    }"""
    p_file = tmp_path / "old.fiberquick.json"
    p_file.write_text(old_json_content, encoding="utf-8")

    loaded = load_project(p_file)
    assert loaded.target_sections == 5
    assert loaded.active_fiber_id == "F001"
    assert loaded.next_fiber_counter == 2


def test_histogram_data_binning():
    measurements = [
        Measurement("M1", "F001", (0, 0), (1, 0), 1.0e-6, "manual", accepted=True),
        Measurement("M2", "F002", (0, 0), (2, 0), 2.0e-6, "manual", accepted=True),
    ]
    h_data = compute_histogram_data(measurements, mode="fiber", n_bins=2)
    assert len(h_data["items"]) == 2
    assert h_data["counts"].sum() == 2
