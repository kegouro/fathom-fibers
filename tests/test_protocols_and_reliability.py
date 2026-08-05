from __future__ import annotations

import math
from pathlib import Path

from fathom_fibers_quick.autosave import (
    get_autosave_dir,
)
from fathom_fibers_quick.exporters import export_html_report
from fathom_fibers_quick.hierarchy import compute_hierarchical_statistics, hierarchical_bootstrap
from fathom_fibers_quick.history import Command, HistoryManager
from fathom_fibers_quick.measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementStatus,
)
from fathom_fibers_quick.model import Calibration, ImageDocument, Project
from fathom_fibers_quick.protocols import (
    PRESET_PVDF_5_SECTIONS,
    MeasurementProtocol,
)
from fathom_fibers_quick.repeatability import (
    analyze_repeatability,
    compare_automatic_and_manual,
    create_blind_session,
)
from fathom_fibers_quick.scientific_validation import (
    compute_measurement_uncertainty,
    validate_resolution,
)


def test_autosave_path_independent_and_migration(tmp_path, monkeypatch):
    # Mock home to test migration
    fake_home = tmp_path / "fake_home"
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    legacy_dir = fake_home / ".gemini" / "antigravity-cli" / "autosaves"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    old_file = legacy_dir / "sample_nosha.fiberquick.autosave.json"
    old_file.write_text('{"test": "legacy"}', encoding="utf-8")

    # Call get_autosave_dir which triggers migration
    new_dir = get_autosave_dir()

    assert not old_file.exists()
    assert (new_dir / "sample_nosha.fiberquick.autosave.json").exists()
    assert "antigravity" not in str(new_dir)


def test_protocol_snapshot_immutability():
    proto1 = PRESET_PVDF_5_SECTIONS
    rec = MeasurementRecord(
        measurement_id="M0001",
        kind=MeasurementKind.PROJECTED_WIDTH,
        name="Ancho M0001",
        protocol_snapshot=proto1.to_dict(),
    )

    # Modify original protocol definition
    modified_proto = MeasurementProtocol.from_dict(proto1.to_dict())
    modified_proto.sections_per_fiber = 99

    # Record snapshot remains untouched at 5
    assert rec.protocol_snapshot["sections_per_fiber"] == 5


def test_resolution_flags():
    cal = Calibration(10e-9, 10e-9, "test")  # 10 nm/px

    # 1. Resolved: 60 nm -> 6 px
    st_res, px_res, _flags_res = validate_resolution(60e-9, cal)
    assert st_res == "RESOLVED"
    assert math.isclose(px_res, 6.0)

    # 2. Marginal: 35 nm -> 3.5 px
    st_mar, _px_mar, flags_mar = validate_resolution(35e-9, cal)
    assert st_mar == "MARGINAL"
    assert "RESOLUTION_MARGINAL" in flags_mar

    # 3. Unresolved: 15 nm -> 1.5 px
    st_unres, _px_unres, flags_unres = validate_resolution(15e-9, cal)
    assert st_unres == "UNRESOLVED"
    assert "RESOLUTION_INSUFFICIENT" in flags_unres


def test_uncertainty_components_and_quadrature():
    cal = Calibration(1e-9, 1e-9, "test", confidence=0.95)
    rec = MeasurementRecord(
        measurement_id="M001",
        kind=MeasurementKind.PROJECTED_WIDTH,
        name="M1",
        geometry={"p1": (0, 0), "p2": (0, 10)},
        values={"width_m": 10e-9},
    )

    unc = compute_measurement_uncertainty(rec, cal)
    assert unc.calibration_m is not None
    assert unc.edge_localization_m is not None
    assert unc.combined_standard_m is not None
    # Quadrature combination check u_c = sqrt(u_cal^2 + u_edge^2)
    expected_c = math.sqrt(unc.calibration_m**2 + unc.edge_localization_m**2)
    assert math.isclose(unc.combined_standard_m, expected_c)


def test_blind_repeatability_study_and_insufficient_n():
    rec1 = MeasurementRecord("M1", MeasurementKind.PROJECTED_WIDTH, "M1", values={"width_m": 100e-9})
    rec2 = MeasurementRecord("M2", MeasurementKind.PROJECTED_WIDTH, "M2", values={"width_m": 150e-9})

    session = create_blind_session([rec1, rec2], seed=123)
    assert len(session.items) == 2
    # Blind items hide original values during input
    assert session.items[0].measured_value_m is None

    # Before measurements -> N insufficient
    res_insuf = analyze_repeatability(session)
    assert res_insuf["status"] == "N_INSUFFICIENT"

    # Simulate completed measurements
    session.items[0].measured_value_m = 102e-9
    session.items[1].measured_value_m = 148e-9

    res_suf = analyze_repeatability(session)
    assert res_suf["status"] == "SUFFICIENT"
    assert res_suf["s_intra_m"] > 0.0
    assert res_suf["cv"] > 0.0


def test_automatic_vs_manual_comparison():
    rec_auto = MeasurementRecord(
        "M_AUTO_1",
        MeasurementKind.PROJECTED_WIDTH,
        "Auto 1",
        source="AUTO_ROI_COMPONENT",
        geometry={"p1": (10, 10), "p2": (10, 30)},
        values={"width_m": 20e-9},
    )
    rec_man = MeasurementRecord(
        "M_MAN_1",
        MeasurementKind.PROJECTED_WIDTH,
        "Man 1",
        source="MANUAL",
        status=MeasurementStatus.ACCEPTED,
        geometry={"p1": (11, 10), "p2": (11, 30)},
        values={"width_m": 22e-9},
    )

    pairs = compare_automatic_and_manual([rec_auto, rec_man])
    assert len(pairs) == 1
    assert pairs[0]["reference_label"] == "Referencia manual revisada"
    assert math.isclose(pairs[0]["absolute_difference_m"], 2e-9)


def test_hierarchical_statistics_and_bootstrap():
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument("img1.tif", 100, 100, cal)
    proj = Project(3, img)

    r1 = MeasurementRecord("M1", MeasurementKind.PROJECTED_WIDTH, "M1", fiber_id="F001", status=MeasurementStatus.ACCEPTED, values={"width_m": 10e-9})
    r2 = MeasurementRecord("M2", MeasurementKind.PROJECTED_WIDTH, "M2", fiber_id="F001", status=MeasurementStatus.ACCEPTED, values={"width_m": 12e-9})
    r3 = MeasurementRecord("M3", MeasurementKind.PROJECTED_WIDTH, "M3", fiber_id="F002", status=MeasurementStatus.ACCEPTED, values={"width_m": 20e-9})
    proj.records = [r1, r2, r3]

    stats = compute_hierarchical_statistics(proj)
    assert stats["section_level"]["n"] == 3
    assert stats["fiber_level"]["n"] == 2
    assert stats["sample_level"]["n_fibers"] == 2

    boot = hierarchical_bootstrap(proj, n_bootstraps=100, seed=42)
    assert boot["bootstrap_mean_m"] is not None
    assert boot["ci_lower_m"] <= boot["ci_upper_m"]


def test_undo_redo_history_audit():
    history = HistoryManager()
    state = {"protocol": "PVDF_5_SECTIONS"}

    def set_p():
        state["protocol"] = "PVDF_3_SECTIONS"

    def reset_p():
        state["protocol"] = "PVDF_5_SECTIONS"

    cmd = Command("Change protocol", set_p, reset_p)
    history.push_and_execute(cmd)
    assert state["protocol"] == "PVDF_3_SECTIONS"

    history.undo()
    assert state["protocol"] == "PVDF_5_SECTIONS"

    history.redo()
    assert state["protocol"] == "PVDF_3_SECTIONS"


def test_html_report_export(tmp_path):
    cal = Calibration(1e-9, 1e-9, "test")
    img = ImageDocument(str(tmp_path / "img.tif"), 100, 100, cal)
    proj = Project(3, img)

    rec = MeasurementRecord("M1", MeasurementKind.PROJECTED_WIDTH, "M1", fiber_id="F001", values={"width_m": 10e-9})
    proj.records = [rec]

    report_path = tmp_path / "test_report.html"
    export_html_report(proj, "dummy_annotated.png", report_path)

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "geometría proyectada 2D" in content
    assert "Software Provenance" in content
