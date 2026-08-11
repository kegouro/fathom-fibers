from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.auto_roi import (
    AutoFiberCandidate,
    AutoROISummary,
    ProposedMeasurement,
)
from fathom_fibers_quick.core import Calibration, MeasurementKind, MeasurementStatus
from fathom_fibers_quick.core.contracts import FathomAnalysisResult
from fathom_fibers_quick.project_io import load_project, save_project


def synthetic_image() -> np.ndarray:
    image = np.zeros((128, 160), dtype=np.uint8)
    image[45:65, 15:145] = 220
    return image


def test_core_imports_without_qt():
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import fathom_fibers_quick.core; "
                "print(any(n.startswith(('PySide6', 'PyQt6')) for n in sys.modules))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert probe.stdout.strip() == "False"


def test_array_api_measurement_with_anisotropic_calibration():
    engine = FathomEngine()
    image = engine.from_array(
        synthetic_image(),
        calibration=Calibration(2e-9, 4e-9, "test"),
    )

    result = engine.measure(
        image,
        MeasurementKind.DISTANCE,
        {"p1": (1.0, 2.0), "p2": (4.0, 6.0)},
    )

    assert result.primary_value == np.hypot(6e-9, 16e-9)
    assert result.values["length_px"] == 5.0


def test_session_command_undo_redo_and_round_trip(tmp_path):
    engine = FathomEngine()
    image = engine.from_array(
        synthetic_image(),
        calibration=Calibration(5e-9, 5e-9, "test"),
        image_id="synthetic",
    )
    session = ProjectSession(engine)
    project = session.new_from_image(image)

    record = session.create_measurement(
        MeasurementKind.PROJECTED_WIDTH,
        {"p1": (30.0, 45.0), "p2": (30.0, 65.0)},
    )
    assert record in project.records
    assert session.dirty

    session.undo()
    assert not project.records
    session.redo()
    assert project.records == [record]

    path = session.save(tmp_path / "session")
    reopened = load_project(path)
    assert reopened.records[0].measurement_id == record.measurement_id
    assert reopened.history_metadata


def test_automatic_result_enters_as_proposed():
    engine = FathomEngine()
    image = engine.from_array(
        synthetic_image(), calibration=Calibration(5e-9, 5e-9, "test")
    )
    session = ProjectSession(engine)
    session.new_from_image(image)
    proposal = ProposedMeasurement(
        p1=(30.0, 45.0),
        p2=(30.0, 65.0),
        center=(30.0, 55.0),
        width_m=100e-9,
    )
    candidate = AutoFiberCandidate(
        candidate_id="C001",
        roi_bbox=(0, 0, 160, 128),
        component_label=1,
        centerline_points=[(20.0, 55.0), (140.0, 55.0)],
        proposed_measurements=[proposal],
        median_width_m=100e-9,
        confidence_score=0.9,
    )
    result = FathomAnalysisResult(
        "FATHOM_ASSISTED_ROI",
        (0, 0, 160, 128),
        (candidate,),
        AutoROISummary(1, 1, 1, 0, 0),
    )

    records = session.apply_fathom_result(result)

    assert len(records) == 1
    assert records[0].status == MeasurementStatus.PROPOSED
    assert not records[0].is_included_in_statistics


def test_legacy_save_creates_recoverable_backup(tmp_path):
    engine = FathomEngine()
    image = engine.from_array(
        synthetic_image(), calibration=Calibration(5e-9, 5e-9, "test")
    )
    session = ProjectSession(engine)
    project = session.new_from_image(image)
    path = tmp_path / "legacy.fiberquick.json"
    payload = project.to_dict()
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    save_project(project, path)

    backup = path.with_name(f"{path.name}.schema-v2.bak")
    assert backup.exists()
    assert json.loads(backup.read_text(encoding="utf-8"))["schema_version"] == 2
