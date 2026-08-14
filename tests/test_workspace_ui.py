from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.core.methods import (
    Estimand,
    MethodCapabilities,
    MethodId,
    MethodResult,
    MethodStatus,
)
from fathom_fibers_quick.ui.main_window import MainWindow
from fathom_fibers_quick.ui.workspace_controller import WorkspaceController
from fathom_fibers_quick.unified_comparison import compare_method_results
from fathom_fibers_quick.workspace import (
    VALIDATION_ROOT,
    WorkspaceCache,
    WorkspaceDataset,
    WorkspaceImage,
)

pytestmark = [pytest.mark.qt, pytest.mark.usefixtures("qapp")]

PIXEL = 5e-9


def make_dataset(tmp_path: Path, count: int = 16) -> WorkspaceDataset:
    import tifffile

    images: list[WorkspaceImage] = []
    for index in range(1, count + 1):
        pixels = np.zeros((48, 64), dtype=np.uint8)
        pixels[18:30, 8:56] = 200
        path = tmp_path / f"PVDF Jose_{index:02d}.tif"
        tifffile.imwrite(path, pixels)
        images.append(
            WorkspaceImage(f"ZEISS_{index:03d}", path.name, path, sha256=f"sha{index:03d}")
        )
    return WorkspaceDataset("ZEISS_PVDF_2026-07-30", tuple(images), None, tmp_path)


def make_session_and_cache(tmp_path: Path) -> tuple[ProjectSession, WorkspaceDataset, WorkspaceCache]:
    engine = FathomEngine()
    session = ProjectSession(engine)
    dataset = make_dataset(tmp_path)
    cache = WorkspaceCache(tmp_path)
    return session, dataset, cache


def make_window(qtbot, tmp_path, *, precompute=True) -> tuple[MainWindow, WorkspaceDataset]:
    session, dataset, cache = make_session_and_cache(tmp_path)
    if precompute:
        image = session.engine.open_image(
            dataset.images[0].absolute_path, manual_pixel_size_m=PIXEL
        )
        comparison = session.engine.compare_all_methods(image)
        cache.store_comparison(dataset.images[0].stem, comparison)
    controller = WorkspaceController(
        session,
        pixel_size_m_fallback=PIXEL,
        cache_root=tmp_path,
    )
    window = MainWindow(session, smoke_test=True, workspace_controller=controller)
    controller.set_dataset(dataset)
    qtbot.addWidget(window)
    window.show()
    return window, dataset


def test_main_window_constructs_with_workspace(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    assert window.workspace.dataset is not None
    assert len(window.workspace.dataset.images) == 16
    assert window.viewer.image is not None
    window.close()


def test_dataset_panel_populates_16_images(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    root = window.dataset_panel.tree.topLevelItem(0)
    assert root is not None
    assert root.childCount() == 16
    assert "complete" in root.child(0).text(1)
    assert "not analyzed" in root.child(15).text(1)


def test_image_selection_changes_workspace(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.workspace.select_image(1)
    assert window.workspace.current_image.stem == "PVDF Jose_02"
    assert window.viewer.image.image_id == "PVDF Jose_02.tif"
    assert window.session.image is not None


def test_cached_method_results_populate_panels(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    comparison = window.workspace.comparison
    assert comparison is not None
    assert window.summary_panel.table.rowCount() >= 5
    assert window.methods_panel.table.rowCount() >= 5
    assert window.quality_panel.table.rowCount() >= 5
    field = window.workspace.results[MethodId.FATHOM_FIELD_GRAPH_V1]
    assert window.measurements_panel.field_model.n == field.local_samples["x_m"].size
    assert "edges" in window.overlay_layers.visible


def test_matlab_b1_only_does_not_crash_distributions(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    comparison = window.workspace.comparison
    results = []
    matlab = MethodResult(
        MethodId.MATLAB_SIMPOLY,
        "MATLAB_R2026A_SOURCE_COMPAT",
        "PVDF Jose_01.tif",
        {"pixel_size_x_m": PIXEL},
        None,
        "um",
        MethodCapabilities(),
        MethodStatus.COMPLETE,
        Estimand.SIMPOLY_NATIVE_GAUSS1,
        0.8017,
        {"gauss_b1": 0.8017},
        quality_flags=("MATLAB_RAW_DIAMETERS_UNAVAILABLE",),
        provenance={"matlab_version": "R2026a"},
    )
    for result in comparison.results:
        if result.method_id != MethodId.MATLAB_SIMPOLY:
            results.append(result)
    b1_comparison = compare_method_results((matlab, *results))
    window.distributions_panel.set_comparison(b1_comparison)
    assert "b1" in window.distributions_panel.matlab_note.text()
    assert window.distributions_panel.histogram is not None
    assert window.distributions_panel.summary_table.rowCount() > 0


def test_distributions_field_estimator_variants(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.distributions_panel.series_combo.setCurrentText("All")
    series = window.distributions_panel._all_series()
    names = {name for name, _dist in series}
    assert "Fathom Field (EDT)" in names
    assert "Fathom Local" in names
    assert "Python SIMPoly" in names
    assert "Consensus" in names
    window.distributions_panel.series_combo.setCurrentText("Field Paired Edge")
    assert window.distributions_panel.histogram is not None


def test_measurement_selection_synchronization(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    field = window.workspace.results[MethodId.FATHOM_FIELD_GRAPH_V1]
    count = int(field.local_samples["x_m"].size)
    assert count > 0
    window._select_field_sample(count // 2)
    assert window._field_sample_index == count // 2
    assert "Position" in window.workspace_inspector.selection.toPlainText()
    window._select_field_sample(0)
    assert window._field_sample_index == 0
    window._select_record(None)
    assert window._field_sample_index is None


def test_manual_progress_state_and_record(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    assert window.manual_panel.case_id == "ZEISS_001"
    assert window.manual_panel.active_cell is not None
    row, column = window.manual_panel.active_cell
    window._show_manual_5x5()
    window.session.create_measurement(
        "PROJECTED_WIDTH",
        {"p1": (8.0, 20.0), "p2": (8.0, 28.0)},
    )
    record = window.session.project.records[-1]
    assert window._try_record_manual_5x5(record) is True
    store = window.workspace.manual
    assert store.reviews["ZEISS_001"].measurement_count == 1
    assert store.reviews["ZEISS_001"].cell(row, column).status.value == "MEASURED"
    assert store.path.exists()
    assert record.protocol_snapshot["protocol_id"] == "MANUAL_5X5_REFERENCE"


def test_manual_autosave_survives_reload(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.session.create_measurement(
        "PROJECTED_WIDTH",
        {"p1": (8.0, 20.0), "p2": (8.0, 28.0)},
    )
    record = window.session.project.records[-1]
    window._try_record_manual_5x5(record)
    store_path = window.workspace.manual.path
    stored = json.loads(store_path.read_text())
    assert stored["reviews"]["ZEISS_001"]["cells"][0]["status"] == "MEASURED"
    assert "timestamp" in stored["reviews"]["ZEISS_001"]["cells"][0]


def test_report_action_dispatch(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)

    paths: list[str] = []
    window.workspace.reportReady.connect(paths.append)
    window._generate_report()
    qtbot.waitUntil(lambda: bool(paths), timeout=20000)
    assert paths and Path(paths[0]).exists()
    text = Path(paths[0]).read_text()
    assert "Method summary" in text


def test_worker_completion_updates_ui(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path, precompute=False)
    assert window.workspace.comparison is None
    window.workspace.run_current_image()
    qtbot.waitUntil(lambda: window.workspace.comparison is not None, timeout=60000)
    assert window.summary_panel.table.rowCount() >= 5
    assert window.workspace.cache.has_full("PVDF Jose_01")


def test_summary_only_payload_does_not_crash(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path, precompute=False)
    runs = tmp_path / VALIDATION_ROOT / "runs"
    runs.mkdir(parents=True)
    (runs / "PVDF Jose_01.json").write_text(
        json.dumps(
            {
                "image_id": "PVDF Jose_01.tif",
                "results": [
                    {"method_id": "PYTHON_SIMPOLY", "status": "COMPLETE", "native_result": 0.84}
                ],
            }
        )
    )
    window.workspace.select_image(0)
    assert window.workspace.summary_payload is not None
    assert window.workspace.comparison is None
    assert "summary cache" in window.dataset_panel.tree.topLevelItem(0).child(0).text(1)


def test_empty_states_do_not_crash(qtbot):
    session = ProjectSession(FathomEngine())
    window = MainWindow(session, smoke_test=True)
    qtbot.addWidget(window)
    window.show()
    assert window.workspace.dataset is None
    assert window.summary_panel.info is not None
    window._generate_report()
    window._generate_dataset_report()
    window.workspace.run_current_image()
    window.close()


def test_ribbon_series_in_distributions(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    series = window.distributions_panel._all_series()
    names = {name for name, _dist in series}
    assert "Ribbon Refined EDT" in names
    assert "Ribbon Refined Edge" in names
    assert "Ribbon Refined Profile" in names
    assert "Fathom Field (EDT)" in names  # raw family still present


def test_distribution_preset_raw_vs_refined_edt(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.distributions_panel.preset_combo.setCurrentText("RAW vs REFINED EDT")
    names = {name for name, _dist in window.distributions_panel._all_series()}
    assert {"Fathom Field (EDT)", "Ribbon Refined EDT"} <= names


def test_summary_panel_field_family_rows(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    labels = []
    for row in range(window.summary_panel.table.rowCount()):
        labels.append(window.summary_panel.table.item(row, 0).text())
    joined = " | ".join(labels)
    assert "Fathom Field / Raw EDT" in joined
    assert "Fathom Field / Ribbon EDT" in joined
    assert "Fathom Field / Ribbon Edge" in joined
    assert "Fathom Field / Ribbon Profile" in joined


def test_quality_panel_ribbon_rows(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    assert not window.quality_panel.ribbon_table.isHidden()
    labels = [
        window.quality_panel.ribbon_table.item(row, 0).text()
        for row in range(window.quality_panel.ribbon_table.rowCount())
    ]
    assert any("Supported centerline coverage" in label for label in labels)
    assert any("Residual center shift" in label for label in labels)


def test_measurements_table_refined_columns(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    headers = window.measurements_panel.field_model.HEADERS
    assert "refined EDT (nm)" in headers
    assert "refined Edge (nm)" in headers
    assert "residual shift" in headers
    assert "refinement" in headers


def test_inspector_refinement_section(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window._select_field_sample(0)
    text = window.workspace_inspector.selection.toPlainText()
    assert "CENTERLINE REFINEMENT" in text
    assert "RAW VS REFINED WIDTHS" in text
    assert "Residual shift" in text


def test_overlay_panel_ribbon_layers(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    for key in ("refined_centerline", "midpoints", "refined_edges", "rejected_refined"):
        assert key in window.overlay_panel.checks, key
    assert window.overlay_panel.checks["refined_centerline"].isEnabled()


def test_missing_ribbon_cache_does_not_crash(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path, precompute=False)
    window.distributions_panel.set_comparison(None)
    window.quality_panel.set_comparison(None)
    window.summary_panel.set_comparison(None)
    assert window.quality_panel.ribbon_table.rowCount() == 0


def test_export_bundle_writes_required_files(qtbot, tmp_path):
    from fathom_fibers_quick.export_bundle import export_analysis_bundle
    from fathom_fibers_quick.workspace import WorkspaceCache

    window, dataset = make_window(qtbot, tmp_path)
    cache = WorkspaceCache(tmp_path)
    assert cache.has_full("PVDF Jose_01")
    root = export_analysis_bundle(
        tmp_path,
        dataset=dataset,
        manual_store=window.workspace.manual,
        output_dir=tmp_path / "bundle",
    )
    assert (root / "README.md").exists()
    assert (root / "results/dataset_summary.csv").exists()
    assert (root / "results/measurements.csv").exists()
    assert (root / "results/provenance.json").exists()
    assert (root / "results/method_results.json").exists()
    assert (root / "report/index.html").exists()
    summary = (root / "results/dataset_summary.csv").read_text()
    assert "Refined EDT" in summary
    provenance = json.loads((root / "results/provenance.json").read_text())
    assert provenance["version"] == "0.2.0rc1"
    assert "git_commit" in provenance
