from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.measurement_records import MeasurementStatus
from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.oracles.simpoly_source import PROFILE_CONTROLLED_INPUT_V1
from fathom_fibers_quick.ui.main_window import MainWindow

pytestmark = [pytest.mark.qt, pytest.mark.usefixtures("qapp")]


def make_session() -> ProjectSession:
    pixels = np.zeros((160, 220), dtype=np.uint8)
    pixels[55:85, 20:200] = 220
    engine = FathomEngine()
    image = engine.from_array(
        pixels,
        calibration=Calibration(4e-9, 5e-9, "synthetic-ui"),
        image_id="synthetic-ui",
    )
    session = ProjectSession(engine)
    session.new_from_image(image)
    return session


def make_window(qtbot) -> MainWindow:
    session = make_session()
    window = MainWindow(session, smoke_test=True)
    qtbot.addWidget(window)
    window.viewer.set_image(session.image)
    window.show()
    return window


def test_launch_open_synthetic_create_line_and_selection_sync(qtbot):
    window = make_window(qtbot)
    tool = window.viewer.tools.tools["projected_width"]
    tool.mouse_press(QPointF(80.0, 55.0), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    tool.mouse_move(QPointF(80.0, 85.0), Qt.MouseButton.LeftButton)
    tool.mouse_release(QPointF(80.0, 85.0), Qt.MouseButton.LeftButton)

    assert len(window.session.project.records) == 1
    record = window.session.project.records[0]
    assert window.results_panel.model.rowCount() == 1
    assert window.viewer.image is window.session.image

    window.session.select(record.measurement_id)
    assert window.results_panel.selected_ids() == [record.measurement_id]
    assert window.inspector_panel.title.text().startswith(record.measurement_id)


def test_edit_metadata_undo_redo(qtbot):
    window = make_window(qtbot)
    record = window.session.create_measurement(
        "DISTANCE",
        {"p1": (10.0, 10.0), "p2": (30.0, 10.0)},
    )
    index = window.results_panel.model.index(0, 1)
    assert window.results_panel.model.setData(index, "Gauge A")
    assert record.name == "Gauge A"

    window.history_bridge.undo()
    assert record.name != "Gauge A"
    window.history_bridge.redo()
    assert record.name == "Gauge A"


def test_save_reopen_and_simpoly_proposal(qtbot, tmp_path):
    window = make_window(qtbot)
    window.session.create_measurement(
        "PROJECTED_WIDTH",
        {"p1": (70.0, 55.0), "p2": (70.0, 85.0)},
    )
    project_path = window.session.save(tmp_path / "ui-roundtrip")

    reopened = ProjectSession(FathomEngine())
    # Array-backed projects intentionally have no source file. Verify persistence
    # directly and keep the UI source-path behavior covered by real TIFF smoke.
    from fathom_fibers_quick.project_io import load_project

    loaded = load_project(project_path)
    assert len(loaded.records) == 1

    result, _intermediates = window.session.engine.run_simpoly(
        window.session.image,
        profile=PROFILE_CONTROLLED_INPUT_V1,
        roi_bbox=(0, 0, 220, 150),
    )
    record = window.session.apply_simpoly_result(
        result,
        profile=PROFILE_CONTROLLED_INPUT_V1,
        roi_bbox=(0, 0, 220, 150),
    )
    assert record.status == MeasurementStatus.PROPOSED
    assert window.results_panel.model.rowCount() == 2
    assert reopened.project is None


def test_close_cleanly(qtbot):
    window = make_window(qtbot)
    window.close()
    assert not window.isVisible()


def test_batch_review_loads_exact_16_case_manifest(qtbot, tmp_path):
    from fathom_fibers_quick.ui.widgets import BatchReviewPanel

    cases = [
        {
            "case_id": f"ZEISS_{index:03d}",
            "filename": f"image-{index}.tif",
            "absolute_path": f"/tmp/image-{index}.tif",
        }
        for index in range(1, 17)
    ]
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {"dataset_id": "ZEISS_PVDF_2026-07-30", "case_count": 16, "cases": cases}
        ),
        encoding="utf-8",
    )
    panel = BatchReviewPanel()
    qtbot.addWidget(panel)
    panel.load_manifest(manifest)
    assert panel.position.text().startswith("Image 1 / 16")
    panel.next()
    assert "ZEISS_002" in panel.position.text()
    assert panel.current_progress.text() == "Current image measurements: 0 / 25"


def test_batch_grid_measurement_records_protocol_provenance(qtbot, tmp_path):
    window = make_window(qtbot)
    cases = [
        {
            "case_id": f"ZEISS_{index:03d}",
            "filename": f"image-{index}.tif",
            "absolute_path": f"/tmp/image-{index}.tif",
        }
        for index in range(1, 17)
    ]
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        __import__("json").dumps(
            {"dataset_id": "ZEISS_PVDF_2026-07-30", "case_count": 16, "cases": cases}
        ),
        encoding="utf-8",
    )
    window.batch_review_panel.load_manifest(manifest)
    window.batch_review_panel.grid.setCurrentCell(1, 2)
    window._create_measurement("PROJECTED_WIDTH", {"p1": (70.0, 55.0), "p2": (70.0, 85.0)})
    record = window.session.project.records[-1]
    assert record.protocol_snapshot["protocol_id"] == "MANUAL_5X5_REFERENCE"
    assert record.protocol_snapshot["grid_position"] == "R2C3"
    cell = window.batch_review_panel.reviews["ZEISS_001"].cell(1, 2)
    assert cell.measurement_id == record.measurement_id
    assert cell.status.value == "MEASURED"
