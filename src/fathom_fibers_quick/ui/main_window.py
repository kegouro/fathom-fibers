from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QTabWidget,
    QToolBar,
    QWidget,
)

from ..api import FathomEngine
from ..application import ProjectSession
from ..autosave import perform_atomic_autosave
from ..core.contracts import FathomAnalysisResult, MethodComparisonResult
from ..exporters import export_csv
from ..measurement_records import MeasurementStatus
from ..oracles.simpoly_source import PROFILE_CONTROLLED_INPUT_V1, PROFILE_SOURCE_COMPAT_V1
from ..project_io import SourceVerificationStatus, verify_project_source
from .commands import HistoryBridge
from .tasks import AnalysisTask
from .widgets import (
    AnalysisPanel,
    BatchReviewPanel,
    ComparisonPanel,
    InspectorPanel,
    ProjectPanel,
    ResultsPanel,
)
from .widgets.image_viewer import ScientificImageView
from .widgets.panels import HistoryPanel

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Scientific desktop shell; all domain mutations delegate to ProjectSession."""

    def __init__(
        self,
        session: ProjectSession | None = None,
        *,
        initial_path: str | None = None,
        smoke_test: bool = False,
    ) -> None:
        super().__init__()
        self.session = session or ProjectSession(FathomEngine())
        self.session.subscribe(self._session_event)
        self.history_bridge = HistoryBridge(self.session)
        self.thread_pool = QThreadPool(self)
        self.active_task: AnalysisTask | None = None
        self._selection_sync = False
        self._smoke_test = smoke_test
        self.setWindowTitle("Fathom Fibers")
        self.resize(1500, 920)
        self.setMinimumSize(1050, 680)

        self.viewer = ScientificImageView()
        self.setCentralWidget(self.viewer)
        self.project_panel = ProjectPanel(self.session)
        self.results_panel = ResultsPanel(self.session)
        self.inspector_panel = InspectorPanel()
        self.analysis_panel = AnalysisPanel()
        self.comparison_panel = ComparisonPanel()
        self.batch_review_panel = BatchReviewPanel()
        self.history_panel = HistoryPanel(self.session)
        self._build_docks()
        self._build_actions()
        self._build_menus_and_toolbars()
        self._build_status_bar()
        self._wire()
        self._start_autosave_timer()
        self._refresh_all()
        campaign_manifest = Path.cwd() / ".validation/real-tiff-campaign/dataset_manifest.json"
        if campaign_manifest.exists():
            try:
                self.batch_review_panel.load_manifest(campaign_manifest)
            except Exception:
                logger.exception("Could not load local batch review manifest")

        if initial_path:
            QTimer.singleShot(0, lambda: self.open_path(initial_path))

    def _build_docks(self) -> None:
        project_dock = QDockWidget("PROJECT", self)
        project_dock.setObjectName("projectDock")
        project_dock.setWidget(self.project_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project_dock)

        inspector_tabs = QTabWidget()
        inspector_tabs.addTab(self.inspector_panel, "Measurement")
        inspector_tabs.addTab(self._display_panel(), "Display")
        inspector_dock = QDockWidget("INSPECTOR", self)
        inspector_dock.setObjectName("inspectorDock")
        inspector_dock.setWidget(inspector_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, inspector_dock)

        bottom_tabs = QTabWidget()
        bottom_tabs.addTab(self.results_panel, "RESULTS / MEASUREMENTS")
        bottom_tabs.addTab(self.history_panel, "HISTORY")
        bottom_tabs.addTab(self.analysis_panel, "ANALYSIS")
        bottom_tabs.addTab(self.comparison_panel, "COMPARE METHODS")
        bottom_tabs.addTab(self.batch_review_panel, "BATCH MEASUREMENT REVIEW")
        self.bottom_tabs = bottom_tabs
        bottom_dock = QDockWidget("RESULTS / HISTORY / ANALYSIS", self)
        bottom_dock.setObjectName("resultsDock")
        bottom_dock.setWidget(bottom_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, bottom_dock)
        bottom_dock.setMinimumHeight(250)

    def _display_panel(self) -> QWidget:
        panel = QWidget()
        form = QFormLayout(panel)
        self.calibration_label = QLabel("No image")
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        self.gamma_slider = QSlider(Qt.Orientation.Horizontal)
        self.gamma_slider.setRange(20, 300)
        self.gamma_slider.setValue(100)
        self.invert_check = QCheckBox("Invert display")
        self.footer_check = QCheckBox("Show footer for inspection")
        self.footer_check.setChecked(True)
        note = QLabel("Display adjustments never modify scientific pixel data.")
        note.setWordWrap(True)
        form.addRow("Calibration", self.calibration_label)
        form.addRow("Brightness", self.brightness_slider)
        form.addRow("Contrast", self.contrast_slider)
        form.addRow("Gamma", self.gamma_slider)
        form.addRow(self.invert_check)
        form.addRow(self.footer_check)
        form.addRow(note)
        return panel

    def _action(
        self,
        text: str,
        slot,
        shortcut: str | QKeySequence | None = None,
        *,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(slot)
        return action

    def _build_actions(self) -> None:
        self.open_image_action = self._action("Open image…", self.open_image_dialog, "Ctrl+O")
        self.open_project_action = self._action("Open project…", self.open_project_dialog)
        self.save_action = self._action("Save project", self.save_project, "Ctrl+S")
        self.save_as_action = self._action("Save project as…", self.save_project_as, "Ctrl+Shift+S")
        self.export_action = self._action("Export CSV…", self.export_csv_dialog, "Ctrl+E")
        self.quit_action = self._action("Quit", self.close, "Ctrl+Q")
        self.undo_action = self._action("Undo", self.history_bridge.undo, "Ctrl+Z")
        self.redo_action = self._action("Redo", self.history_bridge.redo, "Ctrl+Shift+Z")
        self.redo_action.setShortcuts([QKeySequence("Ctrl+Shift+Z"), QKeySequence("Ctrl+Y")])
        self.delete_action = self._action("Delete selected", self._delete_selected, "Delete")
        self.fit_action = self._action("Fit image", self.viewer.fit_to_window, "F")
        self.actual_action = self._action("1:1 pixels", self.viewer.actual_pixels, "1")
        self.reset_action = self._action("Reset view", self.viewer.reset_view, "0")

        self.tool_group = QActionGroup(self)
        self.tool_group.setExclusive(True)
        specs = (
            ("Select", "select", "V"),
            ("Pan", "pan", "H"),
            ("Projected width", "projected_width", "M"),
            ("Distance", "distance", "D"),
            ("Polyline", "polyline", "P"),
            ("Angle", "angle", "G"),
            ("Rectangle ROI", "rectangle_roi", "R"),
            ("Polygon ROI", "polygon_roi", "Y"),
            ("Intensity profile", "intensity_profile", "L"),
        )
        self.tool_actions: dict[str, QAction] = {}
        for text, name, shortcut in specs:
            action = self._action(
                text,
                lambda _checked=False, tool=name: self.viewer.activate_tool(tool),
                shortcut,
                checkable=True,
            )
            self.tool_group.addAction(action)
            self.tool_actions[name] = action
        self.tool_actions["select"].setChecked(True)
        self.cancel_action = self._action("Cancel current tool", self.viewer.tools.cancel, "Esc")

    def _build_menus_and_toolbars(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.open_image_action,
            self.open_project_action,
            self.save_action,
            self.save_as_action,
            self.export_action,
            self.quit_action,
        ):
            file_menu.addAction(action)
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addActions((self.undo_action, self.redo_action, self.delete_action))
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addActions((self.fit_action, self.actual_action, self.reset_action))
        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addActions(self.tool_group.actions())
        tools_menu.addSeparator()
        tools_menu.addAction(self.cancel_action)

        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.addActions((self.open_image_action, self.save_action))
        toolbar.addSeparator()
        toolbar.addActions((self.undo_action, self.redo_action))
        toolbar.addSeparator()
        toolbar.addActions(self.tool_group.actions())
        toolbar.addSeparator()
        toolbar.addActions((self.fit_action, self.actual_action))
        self.addToolBar(toolbar)

    def _build_status_bar(self) -> None:
        self.scientific_notice = QLabel(
            "Measurements represent projected 2D geometry. Automatic results require review."
        )
        self.coordinate_label = QLabel("x —  y —  value —")
        self.statusBar().addWidget(self.scientific_notice, 1)
        self.statusBar().addPermanentWidget(self.coordinate_label)

    def _wire(self) -> None:
        self.viewer.measurementRequested.connect(self._create_measurement)
        self.viewer.roiDrawn.connect(self.session.set_roi)
        self.viewer.recordSelected.connect(self._select_record)
        self.viewer.geometryEdited.connect(self._edit_geometry)
        self.viewer.coordinateChanged.connect(self._show_coordinate)
        self.project_panel.recordSelected.connect(self._select_record)
        self.project_panel.focusRequested.connect(self.viewer.focus_record)
        self.results_panel.recordSelected.connect(self._select_record)
        self.results_panel.focusRequested.connect(self.viewer.focus_record)
        self.results_panel.acceptRequested.connect(
            lambda ids: self._set_status(ids, MeasurementStatus.ACCEPTED)
        )
        self.results_panel.rejectRequested.connect(
            lambda ids: self._set_status(ids, MeasurementStatus.REJECTED)
        )
        self.results_panel.deleteRequested.connect(self._delete_ids)
        self.results_panel.exportRequested.connect(self._export_selected)
        self.analysis_panel.runRequested.connect(self._run_analysis)
        self.analysis_panel.cancelRequested.connect(self._cancel_task)
        self.comparison_panel.runRequested.connect(self._run_comparison)
        self.batch_review_panel.imageRequested.connect(self.open_path)
        self.batch_review_panel.runRequested.connect(self._run_batch_action)
        self.batch_review_panel.saveRequested.connect(self.save_project)
        self.history_bridge.stateChanged.connect(self._history_state)
        self.brightness_slider.valueChanged.connect(self._display_changed)
        self.contrast_slider.valueChanged.connect(self._display_changed)
        self.gamma_slider.valueChanged.connect(self._display_changed)
        self.invert_check.toggled.connect(self._display_changed)
        self.footer_check.toggled.connect(self.viewer.set_footer_visible)

    def _history_state(self, can_undo: bool, can_redo: bool) -> None:
        self.undo_action.setEnabled(can_undo)
        self.redo_action.setEnabled(can_redo)

    def _display_changed(self, *_args: Any) -> None:
        self.viewer.set_display_adjustments(
            brightness=self.brightness_slider.value() / 100.0,
            contrast=self.contrast_slider.value() / 100.0,
            gamma=self.gamma_slider.value() / 100.0,
            inverted=self.invert_check.isChecked(),
        )

    def _session_event(self, event: str) -> None:
        if event == "selection":
            self._sync_selection()
        else:
            self._refresh_all()

    def _refresh_all(self) -> None:
        project = self.session.project
        self._selection_sync = True
        try:
            self.viewer.set_records(
                project.records if project else [],
                self.session.selected_record_id,
            )
            self.project_panel.refresh()
            self.results_panel.refresh()
            self.history_panel.refresh()
            self.results_panel.select_record(self.session.selected_record_id)
            self.viewer.set_selected_record(self.session.selected_record_id)
            self.inspector_panel.set_record(self.session.selected_record())
        finally:
            self._selection_sync = False
        self._update_calibration()
        self._update_title()
        self.history_bridge.refresh()

    def _sync_selection(self) -> None:
        if self._selection_sync:
            return
        self._selection_sync = True
        try:
            selected = self.session.selected_record_id
            self.viewer.set_selected_record(selected)
            self.results_panel.select_record(selected)
            self.inspector_panel.set_record(self.session.selected_record())
        finally:
            self._selection_sync = False

    def _select_record(self, record_id: str | None) -> None:
        if self._selection_sync or record_id == self.session.selected_record_id:
            return
        self.session.select(record_id)

    def _update_title(self) -> None:
        project = self.session.project
        name = Path(project.project_path or project.image.path).name if project else "No project"
        marker = " *" if self.session.dirty else ""
        self.setWindowTitle(f"Fathom Fibers — {name}{marker}")

    def _update_calibration(self) -> None:
        if self.session.image is None:
            self.calibration_label.setText("No image")
            return
        calibration = self.session.image.calibration
        self.calibration_label.setText(
            f"{calibration.pixel_size_x_m * 1e9:.5g} × "
            f"{calibration.pixel_size_y_m * 1e9:.5g} nm/px\n{calibration.source}"
        )

    def _show_coordinate(self, x, y, pixel, physical_x, physical_y) -> None:
        value = "—" if pixel is None else f"{pixel:.5g}"
        self.coordinate_label.setText(
            f"x {x:.1f}  y {y:.1f}  value {value}  |  "
            f"{physical_x * 1e6:.4g}, {physical_y * 1e6:.4g} µm"
        )

    def open_image_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open microscopy image",
            "",
            "Images (*.tif *.tiff *.png *.jpg *.jpeg);;All files (*)",
        )
        if path:
            self.open_image(path)

    def open_project_dialog(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Fathom project",
            "",
            "Fathom projects (*.fiberquick.json);;JSON (*.json)",
        )
        if path:
            self.open_project(path)

    def open_path(self, path: str) -> None:
        if path.lower().endswith((".fiberquick.json", ".fiberquick.autosave.json")):
            self.open_project(path)
        else:
            self.open_image(path)

    def open_image(self, path: str) -> None:
        if not self._confirm_discard():
            return
        try:
            try:
                self.session.open_image(path)
            except ValueError as exc:
                if "calibration" not in str(exc).lower():
                    raise
                nm_per_px, ok = QInputDialog.getDouble(
                    self,
                    "Physical calibration",
                    "Pixel size (nm/px):",
                    1.0,
                    1e-9,
                    1e9,
                    8,
                )
                if not ok:
                    return
                self.session.open_image(path, manual_pixel_size_m=nm_per_px * 1e-9)
            self.viewer.set_image(self.session.image)
            self.statusBar().showMessage(f"Opened {path}", 5000)
        except Exception as exc:
            logger.exception("Failed to open image %s", path)
            QMessageBox.critical(self, "Open image failed", str(exc))

    def open_project(self, path: str) -> None:
        if not self._confirm_discard():
            return
        try:
            project = self.session.open_project(path)
            verification = verify_project_source(project)
            self.viewer.set_image(self.session.image)
            if verification.status not in {
                SourceVerificationStatus.MATCH,
                SourceVerificationStatus.UNVERIFIED,
            }:
                QMessageBox.warning(self, "Source verification", verification.message)
            self.statusBar().showMessage(f"Opened project {path}", 5000)
        except Exception as exc:
            logger.exception("Failed to open project %s", path)
            QMessageBox.critical(self, "Open project failed", str(exc))

    def save_project(self) -> None:
        if self.session.project is None:
            return
        if not self.session.project.project_path:
            self.save_project_as()
            return
        self._save_to(self.session.project.project_path)

    def save_project_as(self) -> None:
        if self.session.project is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Save Fathom project",
            "project.fiberquick.json",
            "Fathom projects (*.fiberquick.json)",
        )
        if path:
            self._save_to(path)

    def _save_to(self, path: str) -> None:
        try:
            saved = self.session.save(path)
            self.statusBar().showMessage(f"Saved {saved}", 5000)
        except Exception as exc:
            logger.exception("Project save failed")
            QMessageBox.critical(self, "Save failed", str(exc))

    def export_csv_dialog(self) -> None:
        if self.session.project is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export measurements",
            "measurements.csv",
            "CSV (*.csv)",
        )
        if path:
            export_csv(self.session.project, path)
            self.statusBar().showMessage(f"Exported {path}", 5000)

    def _export_selected(self, record_ids: list[str]) -> None:
        if not record_ids or self.session.project is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export selected measurements",
            "selected-measurements.csv",
            "CSV (*.csv)",
        )
        if not path:
            return
        project = copy.copy(self.session.project)
        selected = set(record_ids)
        project.records = [r for r in self.session.project.records if r.measurement_id in selected]
        export_csv(project, path)

    def _create_measurement(self, kind: str, geometry: dict[str, Any]) -> None:
        try:
            record = self.session.create_measurement(kind, geometry)
            case = self.batch_review_panel.current_case()
            grid = self.batch_review_panel.active_grid_position()
            if kind == "PROJECTED_WIDTH" and case is not None and grid is not None:
                position = f"R{grid[0] + 1}C{grid[1] + 1}"
                self.session.annotate_manual_grid(
                    record.measurement_id, case_id=case["case_id"], grid_position=position
                )
                self.batch_review_panel.record_measurement(record)
            self.statusBar().showMessage(f"Created {record.measurement_id}", 3000)
        except Exception as exc:
            logger.exception("Measurement creation failed")
            self.statusBar().showMessage(f"Measurement rejected: {exc}", 7000)

    def _edit_geometry(self, record_id: str, geometry: dict[str, Any]) -> None:
        try:
            self.session.update_geometry(record_id, geometry)
        except Exception as exc:
            logger.exception("Overlay edit failed")
            self.statusBar().showMessage(f"Overlay edit rejected: {exc}", 7000)

    def _set_status(self, record_ids: list[str], status: MeasurementStatus) -> None:
        if record_ids:
            self.session.update_metadata(record_ids, status=status)

    def _delete_selected(self) -> None:
        self._delete_ids(self.results_panel.selected_ids())

    def _delete_ids(self, record_ids: list[str]) -> None:
        if record_ids:
            self.session.delete_records(record_ids)

    def _run_analysis(self, method: str) -> None:
        if self.session.image is None or self.active_task is not None:
            return
        image = self.session.image
        roi = self.session.roi_bbox
        if method == "Fathom Assisted ROI":
            function = lambda: self.session.engine.run_fathom(
                image,
                roi_bbox=roi,
                options={"n_sections": self.analysis_panel.sections.value()},
            )
        elif method == "SIMPoly Source Compatible":
            function = lambda: self.session.engine.run_simpoly(
                image,
                profile=PROFILE_SOURCE_COMPAT_V1,
            )
        else:
            function = lambda: self.session.engine.run_simpoly(
                image,
                profile=PROFILE_CONTROLLED_INPUT_V1,
                roi_bbox=roi,
            )
        self._start_task(function, lambda result: self._analysis_done(method, result))

    def _analysis_done(self, method: str, result: Any) -> None:
        if isinstance(result, FathomAnalysisResult):
            records = self.session.apply_fathom_result(result)
            message = f"Fathom produced {len(records)} review proposals"
        else:
            simpoly_result, _intermediates = result
            profile = (
                PROFILE_SOURCE_COMPAT_V1
                if method == "SIMPoly Source Compatible"
                else PROFILE_CONTROLLED_INPUT_V1
            )
            record = self.session.apply_simpoly_result(
                simpoly_result,
                profile=profile,
                roi_bbox=self.session.roi_bbox if profile == PROFILE_CONTROLLED_INPUT_V1 else None,
            )
            message = f"SIMPoly result {record.measurement_id} awaits review"
        self.analysis_panel.set_running(False, message)
        self.statusBar().showMessage(message, 7000)

    def _run_comparison(self) -> None:
        if self.session.image is None or self.active_task is not None:
            return
        self._start_task(self.session.compare_methods, self._comparison_done)

    def _comparison_done(self, result: MethodComparisonResult) -> None:
        if self.session.image is not None:
            self.comparison_panel.set_result(result, self.session.image)
        self.analysis_panel.set_running(False, "Method comparison complete")

    def _run_batch_action(self, action: str) -> None:
        if action == "fathom":
            self._run_analysis("Fathom Assisted ROI")
        elif action == "python":
            self._run_analysis("SIMPoly Controlled Input")
        elif action == "compare":
            self._run_comparison()
        elif action == "matlab":
            self._run_matlab_current()

    def _run_matlab_current(self) -> None:
        if self.session.image is None or self.active_task is not None:
            return
        source = self.session.image.source_path
        if not source:
            self.statusBar().showMessage("MATLAB oracle requires a file-backed image", 7000)
            return

        def run_external_oracle() -> dict[str, Any]:
            from ..validation.matlab_oracle import MatlabOracle

            repo = Path.cwd().resolve()
            oracle = MatlabOracle.discover(repo)
            if oracle is None:
                raise RuntimeError("MATLAB executable is unavailable")
            case = self.batch_review_panel.current_case()
            case_id = case["case_id"] if case else Path(source).stem
            output = repo / ".validation/matlab-oracle/ui-runs" / case_id
            harness = oracle.harness_dir.as_posix().replace("'", "''")
            image = Path(source).resolve().as_posix().replace("'", "''")
            expression = (
                f"addpath('{harness}');run_simpoly_case('{image}','SOURCE_COMPAT',[],"
                f"'{output.as_posix()}',false);"
            )
            completed = oracle.batch(expression, timeout=1800)
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
            return {"case_id": case_id, "summary": str(output / "summary.json")}

        self._start_task(run_external_oracle, self._matlab_batch_done)

    def _matlab_batch_done(self, result: dict[str, Any]) -> None:
        message = f"MATLAB SIMPoly complete for {result['case_id']}"
        self.analysis_panel.set_running(False, message)
        self.statusBar().showMessage(f"{message}; {result['summary']}", 9000)

    def _start_task(self, function, on_result) -> None:
        task = AnalysisTask(function)
        self.active_task = task
        task.signals.result.connect(on_result)
        task.signals.error.connect(self._task_error)
        task.signals.finished.connect(self._task_finished)
        self.analysis_panel.set_running(True)
        self.thread_pool.start(task)

    def _cancel_task(self) -> None:
        if self.active_task:
            self.active_task.cancel()
            self.analysis_panel.set_running(True, "Cancellation requested…")

    def _task_error(self, error: Exception, traceback_text: str) -> None:
        logger.error("Background analysis failed: %s\n%s", error, traceback_text)
        self.analysis_panel.set_running(False, f"Failed: {error}")
        QMessageBox.critical(self, "Analysis failed", str(error))

    def _task_finished(self) -> None:
        self.active_task = None
        if self.analysis_panel.run_button.isEnabled() is False:
            self.analysis_panel.set_running(False, self.analysis_panel.status.text())

    def _start_autosave_timer(self) -> None:
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(30_000)
        self.autosave_timer.timeout.connect(self._autosave)
        self.autosave_timer.start()

    def _autosave(self) -> None:
        if not self.session.dirty or self.session.project is None:
            return
        path = perform_atomic_autosave(self.session.project)
        if path:
            self.statusBar().showMessage(f"Autosaved recovery copy: {path.name}", 3000)
        else:
            self.statusBar().showMessage("Autosave failed; see log", 5000)

    def _confirm_discard(self) -> bool:
        if not self.session.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "Discard unsaved changes?",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._smoke_test or self._confirm_discard():
            if self.active_task:
                self.active_task.cancel()
            event.accept()
        else:
            event.ignore()
