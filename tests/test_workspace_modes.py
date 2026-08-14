from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.ui.main_window import MainWindow
from fathom_fibers_quick.ui.workspace_controller import WorkspaceController
from fathom_fibers_quick.workspace import (
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
        images.append(WorkspaceImage(f"ZEISS_{index:03d}", path.name, path, sha256=f"sha{index:03d}"))
    return WorkspaceDataset("ZEISS_PVDF_2026-07-30", tuple(images), None, tmp_path)


def make_window(qtbot, tmp_path, *, precompute=True) -> tuple[MainWindow, WorkspaceDataset]:
    engine = FathomEngine()
    session = ProjectSession(engine)
    dataset = make_dataset(tmp_path)
    cache = WorkspaceCache(tmp_path)
    if precompute:
        image = session.engine.open_image(dataset.images[0].absolute_path, manual_pixel_size_m=PIXEL)
        comparison = session.engine.compare_all_methods(image)
        cache.store_comparison(dataset.images[0].stem, comparison)
    controller = WorkspaceController(session, pixel_size_m_fallback=PIXEL, cache_root=tmp_path)
    window = MainWindow(session, smoke_test=True, workspace_controller=controller)
    controller.set_dataset(dataset)
    qtbot.addWidget(window)
    window.show()
    return window, dataset


def test_default_empty_state_landing(qtbot, tmp_path):
    session = ProjectSession(FathomEngine())
    window = MainWindow(session, smoke_test=True)
    qtbot.addWidget(window)
    window.show()
    assert window.central_stack.currentWidget() is window.landing
    assert not window.dataset_dock.isVisible()
    assert not window.bottom_dock.isVisible()
    assert window.workspace.dataset is None


def test_analyze_mode_layout(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("analyze")
    assert window.central_stack.currentWidget() is window.viewer
    assert window.dataset_dock.isVisible()
    assert window.overlay_dock.isVisible()
    assert window.inspector_dock.isVisible()
    assert not window.project_dock.isVisible()
    visible_tabs = [
        window.bottom_tabs.tabText(index)
        for index in range(window.bottom_tabs.count())
        if window.bottom_tabs.isTabVisible(index)
    ]
    assert set(visible_tabs) == {"DISTRIBUTIONS", "COMPARE METHODS", "QUALITY"}
    window.bottom_tabs.setCurrentWidget(window.distributions_panel)
    window.bottom_tabs.setCurrentWidget(window.comparison_panel)
    window.bottom_tabs.setCurrentWidget(window.quality_panel)


def test_manual_mode_focus(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("manual")
    assert window.bottom_tabs.currentWidget() is window.manual_panel
    visible_tabs = [
        window.bottom_tabs.tabText(index)
        for index in range(window.bottom_tabs.count())
        if window.bottom_tabs.isTabVisible(index)
    ]
    assert visible_tabs == ["MANUAL 5×5"]
    assert not window.overlay_dock.isVisible()
    assert not window.inspector_dock.isVisible()
    assert window.manual_panel.active_cell is not None
    # progress text present
    assert "Point" in window.manual_panel.progress.text()


def test_manual_mode_keyboard_mapping(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("manual")
    panel = window.manual_panel
    first = panel.active_cell
    enter = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
    # without an accepted measurement the same pending target stays current
    panel.keyPressEvent(enter)
    assert panel.active_cell == first
    # after a measurement the next target advances
    cell = window.workspace.manual.reviews["ZEISS_001"].cell(*first)
    cell.status = __import__("fathom_fibers_quick.validation.manual_review", fromlist=["GridCellStatus"]).GridCellStatus.MEASURED
    panel.keyPressEvent(enter)
    assert panel.active_cell != first
    backspace = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(backspace)
    assert panel.active_cell == first
    delete = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(delete)  # must not raise
    escape = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(escape)


def test_report_mode_actions_reachable(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("report")
    assert window.report_header.isVisible()
    assert window.dataset_report_action.isEnabled()
    assert window.export_bundle_action.isEnabled()
    assert window.report_action.isEnabled()
    visible_tabs = [
        window.bottom_tabs.tabText(index)
        for index in range(window.bottom_tabs.count())
        if window.bottom_tabs.isTabVisible(index)
    ]
    assert "SUMMARY" in visible_tabs
    assert "MANUAL 5×5" in visible_tabs


def test_advanced_mode_preserves_technical_tabs(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("advanced")
    assert window.project_dock.isVisible()
    assert window.overlay_dock.isVisible()
    for index in range(window.bottom_tabs.count()):
        assert window.bottom_tabs.isTabVisible(index)
    window.bottom_tabs.setCurrentWidget(window.history_panel)
    window.bottom_tabs.setCurrentWidget(window.batch_review_panel)
    window.bottom_tabs.setCurrentWidget(window.analysis_panel)


def test_mode_switching_does_not_rerun_analysis(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    results_before = window.workspace.results
    window.set_mode("manual")
    window.set_mode("report")
    window.set_mode("advanced")
    window.set_mode("analyze")
    results_after = window.workspace.results
    assert set(results_before) == set(results_after)
    for method in results_before:
        assert results_before[method].status == results_after[method].status


def test_layers_hide_unavailable(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path, precompute=False)
    window.set_mode("analyze")
    # without results, refined overlays are hidden rather than grayed out
    assert not window.overlay_panel.checks["refined_centerline"].isVisible()
    assert window.overlay_panel.checks["manual"].isVisible()


def test_inspector_summary_and_selection(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    # no selection: image summary tab is current
    assert window.inspector_tabs.currentWidget() is window.image_summary_panel
    assert "Ribbon" in window.image_summary_panel.stat_table.item(2, 0).text()
    # selecting a field sample reveals the workspace inspector
    window._select_field_sample(0)
    assert window.inspector_tabs.currentWidget() is window.workspace_inspector
    assert "CENTERLINE REFINEMENT" in window.workspace_inspector.selection.toPlainText()


def test_dataset_header_cta(qtbot, tmp_path):
    engine = FathomEngine()
    session = ProjectSession(engine)
    dataset = make_dataset(tmp_path)
    cache = WorkspaceCache(tmp_path)
    for image in dataset.images:
        scientific = session.engine.open_image(image.absolute_path, manual_pixel_size_m=PIXEL)
        cache.store_comparison(image.stem, session.engine.compare_all_methods(scientific))
    controller = WorkspaceController(session, pixel_size_m_fallback=PIXEL, cache_root=tmp_path)
    window = MainWindow(session, smoke_test=True, workspace_controller=controller)
    controller.set_dataset(dataset)
    qtbot.addWidget(window)
    window.show()
    assert window.dataset_panel.cta_button.text() == "Explore Results"
    assert "16 images" in window.dataset_panel.header_subtitle.text()
    window.set_mode("manual")
    window.dataset_panel.cta_button.click()
    # explore returns to analyze + distributions
    assert window._mode == "analyze"
    assert window.bottom_tabs.currentWidget() is window.distributions_panel


def test_mode_shortcuts_registered(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    assert window.mode_actions["analyze"].shortcut().toString() == "A"
    assert window.mode_actions["manual"].shortcut().toString() == "M"
    assert window.mode_actions["report"].shortcut().toString() == "R"


def test_manual_feedback_flash(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.manual_panel.flash_feedback("0.842 µm · Saved ✓")
    assert "Saved" in window.manual_panel.feedback_label.text()
    qtbot.wait(2500)
    assert window.manual_panel.feedback_label.text() == ""
