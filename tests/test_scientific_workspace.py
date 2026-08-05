from __future__ import annotations

import math

import numpy as np

from fathom_fibers_quick.autosave import check_has_autosave, clear_autosave, perform_atomic_autosave
from fathom_fibers_quick.exporters import export_csv, export_profile_csv
from fathom_fibers_quick.history import Command, HistoryManager
from fathom_fibers_quick.measurement_geometry import (
    compute_angle_geometry,
    compute_area_roi_geometry,
    compute_line_geometry,
    compute_polyline_geometry,
    compute_profile_geometry,
)
from fathom_fibers_quick.measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementSource,
    MeasurementStatus,
    normalize_tags,
)
from fathom_fibers_quick.model import Calibration, ImageDocument, Project
from fathom_fibers_quick.project_io import (
    load_project,
    save_project,
)


def test_measurement_record_legacy_migration():
    # Legacy data dict without new kind/status fields
    legacy_data = {
        "measurement_id": "M0005",
        "fiber_id": "F002",
        "p1": [10.0, 20.0],
        "p2": [10.0, 50.0],
        "width_m": 30e-9,
        "method": "ASSISTED_LOCAL_ONE_CLICK",
        "accepted": True,
    }
    rec = MeasurementRecord.from_dict(legacy_data)
    assert rec.kind == MeasurementKind.PROJECTED_WIDTH
    assert rec.status == MeasurementStatus.ACCEPTED
    assert rec.source == MeasurementSource.ASSISTED
    assert rec.p1 == (10.0, 20.0)
    assert rec.width_m == 30e-9
    assert rec.is_included_in_statistics is True


def test_normalize_tags():
    tags = [" control ", "bead", "CONTROL", "zona-central, alta-mag ", ""]
    norm = normalize_tags(tags)
    assert norm == ["control", "bead", "zona-central", "alta-mag"]


def test_line_geometry_square_and_anisotropic():
    cal_sq = Calibration(1e-9, 1e-9, "test")
    line_sq = compute_line_geometry((0.0, 0.0), (3.0, 4.0), cal_sq)
    assert math.isclose(line_sq["length_m"], 5e-9)
    assert math.isclose(line_sq["delta_x_m"], 3e-9)
    assert math.isclose(line_sq["delta_y_m"], 4e-9)

    cal_aniso = Calibration(2e-9, 1e-9, "aniso")
    line_aniso = compute_line_geometry((0.0, 0.0), (3.0, 4.0), cal_aniso)
    # dx_m = 3*2 = 6nm, dy_m = 4*1 = 4nm -> length = sqrt(36+16) = sqrt(52) nm
    expected_m = math.hypot(6e-9, 4e-9)
    assert math.isclose(line_aniso["length_m"], expected_m)


def test_polyline_geometry_and_tortuosity():
    cal = Calibration(1e-9, 1e-9, "test")
    pts = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0)]
    poly = compute_polyline_geometry(pts, cal)
    assert math.isclose(poly["total_length_m"], 20e-9)
    expected_direct = math.hypot(10e-9, 10e-9)
    assert math.isclose(poly["direct_distance_m"], expected_direct)
    assert poly["tortuosity"] is not None
    assert poly["tortuosity"] > 1.0

    # Degenerate polyline (same endpoints) -> tortuosity is None
    deg_poly = compute_polyline_geometry([(5.0, 5.0), (5.0, 5.0)], cal)
    assert deg_poly["tortuosity"] is None


def test_angle_geometry_90deg_and_zero_arm():
    cal = Calibration(1e-9, 1e-9, "test")
    # 90 degree angle A(10,0) -> B(0,0) -> C(0,10)
    ang = compute_angle_geometry((10.0, 0.0), (0.0, 0.0), (0.0, 10.0), cal)
    assert ang["interior_angle_deg"] is not None
    assert math.isclose(ang["interior_angle_deg"], 90.0)
    assert math.isclose(ang["acute_angle_deg"], 90.0)

    # Zero length arm B -> C (same point) -> returns None
    zero_ang = compute_angle_geometry((10.0, 0.0), (0.0, 0.0), (0.0, 0.0), cal)
    assert zero_ang["interior_angle_deg"] is None


def test_area_roi_rectangle_and_polygon():
    gray = np.full((100, 100), 50.0, dtype=np.float32)
    gray[20:60, 20:60] = 200.0
    cal = Calibration(1e-9, 1e-9, "test")

    rect_info = compute_area_roi_geometry(gray, cal, bbox=(20, 20, 60, 60))
    assert rect_info["valid_pixel_count"] == 1600
    assert math.isclose(rect_info["area_m2"], 1600 * 1e-18)
    assert math.isclose(rect_info["mean_intensity_au"], 200.0)

    # Polygon area
    poly_pts = [(20.0, 20.0), (60.0, 20.0), (60.0, 60.0), (20.0, 60.0)]
    poly_info = compute_area_roi_geometry(gray, cal, polygon=poly_pts)
    assert poly_info["valid_pixel_count"] > 1400
    assert poly_info["mean_intensity_au"] > 190.0


def test_area_roi_footer_exclusion():
    gray = np.full((100, 100), 50.0, dtype=np.float32)
    cal = Calibration(1e-9, 1e-9, "test")
    footer_bounds = (80, 100)

    area_info = compute_area_roi_geometry(gray, cal, bbox=(10, 10, 90, 95), footer_bounds=footer_bounds)
    assert area_info["excluded_pixel_count"] > 0
    assert area_info["excluded_fraction"] > 0.0


def test_profile_geometry():
    gray = np.full((100, 100), 10.0, dtype=np.float32)
    gray[20:80, 40:60] = 180.0
    cal = Calibration(1e-9, 1e-9, "test")

    prof = compute_profile_geometry(gray, (10.0, 50.0), (90.0, 50.0), cal, bandwidth_px=3)
    assert prof["samples_count"] >= 80
    assert len(prof["profile_raw"]) == prof["samples_count"]
    assert prof["max_intensity"] > 170.0


def test_status_inclusion_rules():
    r_acc = MeasurementRecord("M1", MeasurementKind.PROJECTED_WIDTH, "M1", status=MeasurementStatus.ACCEPTED)
    r_edit = MeasurementRecord("M2", MeasurementKind.PROJECTED_WIDTH, "M2", status=MeasurementStatus.MANUALLY_EDITED)
    r_rej = MeasurementRecord("M3", MeasurementKind.PROJECTED_WIDTH, "M3", status=MeasurementStatus.REJECTED)
    r_prop = MeasurementRecord("M4", MeasurementKind.PROJECTED_WIDTH, "M4", status=MeasurementStatus.PROPOSED)

    assert r_acc.is_included_in_statistics is True
    assert r_edit.is_included_in_statistics is True
    assert r_rej.is_included_in_statistics is False
    assert r_prop.is_included_in_statistics is False


def test_history_undo_redo():
    history = HistoryManager()
    state = {"count": 0}

    def inc():
        state["count"] += 1

    def dec():
        state["count"] -= 1

    cmd = Command("Increment", inc, dec)
    history.push_and_execute(cmd)
    assert state["count"] == 1
    assert history.can_undo() is True

    history.undo()
    assert state["count"] == 0
    assert history.can_redo() is True

    history.redo()
    assert state["count"] == 1


def test_autosave_atomic_and_recovery(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 100, 100, cal, source_sha256="testsha123")
    proj = Project(2, img, project_path=str(tmp_path / "proj.fiberquick.json"))

    # Save main project
    save_project(proj, proj.project_path)

    # Perform autosave
    auto_path = perform_atomic_autosave(proj)
    assert auto_path.exists()

    # Recovery check
    has_auto, _p, _mtime = check_has_autosave(proj)
    assert has_auto is True or auto_path.exists()

    # Clear autosave
    clear_autosave(proj)
    assert not auto_path.exists()


def test_unified_csv_and_profile_exporter(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 100, 100, cal)
    proj = Project(2, img)

    rec1 = MeasurementRecord(
        measurement_id="M0001",
        kind=MeasurementKind.PROJECTED_WIDTH,
        name="Ancho M0001",
        status=MeasurementStatus.ACCEPTED,
        fiber_id="F001",
        geometry={"p1": (0, 0), "p2": (0, 10)},
        values={"width_m": 10e-9, "length_m": 10e-9},
    )
    rec2 = MeasurementRecord(
        measurement_id="M0002",
        kind=MeasurementKind.INTENSITY_PROFILE,
        name="Perfil M0002",
        status=MeasurementStatus.ACCEPTED,
        geometry={"p1": (0, 0), "p2": (10, 10)},
        values={"length_m": 14e-9, "distance_m": [0.0, 14e-9], "profile_raw": [10.0, 20.0], "profile_smoothed": [10.0, 20.0]},
    )
    proj.records = [rec1, rec2]

    csv_path = tmp_path / "unified.csv"
    export_csv(proj, csv_path)
    assert csv_path.exists()

    prof_csv_path = tmp_path / "profile.csv"
    export_profile_csv(rec2, prof_csv_path)
    assert prof_csv_path.exists()


def test_recalculate_derived_values_on_load(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 100, 100, cal)
    proj = Project(2, img)

    rec = MeasurementRecord(
        measurement_id="M0001",
        kind=MeasurementKind.PROJECTED_WIDTH,
        name="Ancho M0001",
        geometry={"p1": (0, 0), "p2": (0, 10)},
        values={"width_m": 999.0},  # Stale value
    )
    proj.records = [rec]

    p_path = save_project(proj, tmp_path / "stale")
    reloaded = load_project(p_path)
    # Recalculated value should be 10e-9 m
    assert math.isclose(reloaded.records[0].values["width_m"], 10e-9)
