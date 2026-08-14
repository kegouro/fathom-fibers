from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QPushButton

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.ui.help import (
    PAGE_QUICK_START,
    PAGE_SHORTCUTS,
    PAGE_USER_GUIDE,
    HelpDialog,
)
from fathom_fibers_quick.ui.main_window import MainWindow
from fathom_fibers_quick.ui.workspace_controller import WorkspaceController
from fathom_fibers_quick.workspace import (
    WorkspaceCache,
    WorkspaceDataset,
    WorkspaceImage,
)

pytestmark = [pytest.mark.qt, pytest.mark.usefixtures("qapp")]

PIXEL = 5e-9


def make_window(qtbot, tmp_path, *, precompute=True) -> tuple[MainWindow, WorkspaceDataset]:
    import tifffile

    engine = FathomEngine()
    session = ProjectSession(engine)
    images = []
    for index in range(1, 3):
        pixels = np.zeros((48, 64), dtype=np.uint8)
        pixels[18:30, 8:56] = 200
        path = tmp_path / f"PVDF Jose_{index:02d}.tif"
        tifffile.imwrite(path, pixels)
        images.append(WorkspaceImage(f"ZEISS_{index:03d}", path.name, path))
    dataset = WorkspaceDataset("ZEISS_PVDF_2026-07-30", tuple(images), None, tmp_path)
    cache = WorkspaceCache(tmp_path)
    if precompute:
        scientific = session.engine.open_image(images[0].absolute_path, manual_pixel_size_m=PIXEL)
        cache.store_comparison(images[0].stem, session.engine.compare_all_methods(scientific))
    controller = WorkspaceController(session, pixel_size_m_fallback=PIXEL, cache_root=tmp_path)
    window = MainWindow(session, smoke_test=True, workspace_controller=controller)
    controller.set_dataset(dataset)
    qtbot.addWidget(window)
    window.show()
    return window, dataset


def test_help_menu_contains_entries(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    texts = [action.text() for action in window.help_menu.actions() if action.text()]
    assert "Quick Start" in texts
    assert "User Guide" in texts
    assert "Methods Guide" in texts
    assert "Keyboard Shortcuts" in texts


def test_quick_start_opens_without_dataset(qtbot):
    session = ProjectSession(FathomEngine())
    window = MainWindow(session, smoke_test=True)
    qtbot.addWidget(window)
    window.show()
    dialog = HelpDialog(window, page=PAGE_QUICK_START)
    assert "Quick Start" in dialog.quick_start.toPlainText()
    assert "Open Dataset" in dialog.quick_start.toPlainText()
    dialog.close()


def test_quick_start_opens_with_dataset(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    dialog = HelpDialog(window, page=PAGE_QUICK_START)
    assert "REPORT" in dialog.quick_start.toPlainText()
    dialog.close()


def test_help_does_not_trigger_analysis(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path, precompute=False)
    dialog = HelpDialog(window, page=PAGE_QUICK_START)
    assert window.workspace.comparison is None
    dialog.pages.setCurrentRow(PAGE_USER_GUIDE)
    dialog.pages.setCurrentRow(PAGE_SHORTCUTS)
    dialog.close()
    assert window.workspace.comparison is None


def test_user_guide_and_shortcuts_pages_render(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    dialog = HelpDialog(window, page=PAGE_USER_GUIDE)
    assert "Run missing" in dialog.user_guide.toPlainText()
    dialog.pages.setCurrentRow(PAGE_SHORTCUTS)
    assert "Manual 5×5" in dialog.shortcuts.toPlainText()
    dialog.close()


def test_tooltips_exist_for_major_actions(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    for action in (
        window.open_dataset_action,
        window.run_all_action,
        window.manual_5x5_action,
        window.dataset_report_action,
        window.export_bundle_action,
    ):
        assert action.toolTip(), action.text()
    assert window.mode_actions["manual"].toolTip()
    assert window.mode_actions["report"].toolTip()


def test_run_missing_button_tooltip(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    assert window.dataset_panel.run_missing_button.toolTip()


def test_landing_quick_start_link(qtbot, tmp_path):
    session = ProjectSession(FathomEngine())
    window = MainWindow(session, smoke_test=True)
    qtbot.addWidget(window)
    window.show()
    buttons = window.landing.findChildren(QPushButton)
    texts = {button.text() for button in buttons}
    assert "Open Dataset" in texts
    assert "Quick Start" in texts


def test_quick_start_qsettings_persistence(qtbot, tmp_path, monkeypatch):
    window, _dataset = make_window(qtbot, tmp_path)
    recorded: dict[str, object] = {}

    class FakeSettings:
        def __init__(self, *_args):
            pass

        def setValue(self, key, value) -> None:
            recorded[key] = value

        def value(self, key, default=None):
            return recorded.get(key, default)

    monkeypatch.setattr("fathom_fibers_quick.ui.main_window.QSettings", FakeSettings)
    window._smoke_test = False
    window._set_quick_start_seen(True)
    assert recorded.get("help/quick_start_seen") is True
    assert window._quick_start_seen is True
    window._set_quick_start_seen(False)
    assert window._quick_start_seen is False


def test_manual_help_does_not_modify_measurements(qtbot, tmp_path):
    window, _dataset = make_window(qtbot, tmp_path)
    window.set_mode("manual")
    before = window.workspace.manual.total_measured
    dialog = HelpDialog(window, page=PAGE_QUICK_START)
    dialog.close()
    assert window.workspace.manual.total_measured == before
    assert window._mode == "manual"


def test_user_guide_documentation_exists():
    repo = Path(__file__).resolve().parents[1]
    guide = repo / "docs/USER_GUIDE.md"
    assert guide.exists()
    text = guide.read_text(encoding="utf-8")
    for section in (
        "Five-minute Quick Start",
        "The normal workflow",
        "Analyze workspace",
        "Manual 5×5 workspace",
        "Report workspace",
        "Advanced workspace",
        "Methods explained",
        "Distribution plots",
        "Layers / overlays",
        "Inspector",
        "Quality indicators",
        "Reports and exports",
        "Keyboard shortcuts",
        "Common questions / troubleshooting",
    ):
        assert section in text, section
