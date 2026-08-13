from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QRectF, Qt, QThreadPool, QTimer
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
from ..core.contracts import FathomAnalysisResult, MethodComparisonResult, ScientificImage
from ..core.methods import MethodId
from ..exporters import export_csv
from ..measurement_records import MeasurementRecord, MeasurementStatus
from ..oracles.simpoly_source import PROFILE_CONTROLLED_INPUT_V1, PROFILE_SOURCE_COMPAT_V1
from ..project_io import SourceVerificationStatus, verify_project_source
from ..unified_comparison import UnifiedMethodComparison
from .commands import HistoryBridge
from .overlays import OverlayLayers, build_overlay_payload
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
from .widgets.workspace_panels import (
    DatasetPanel,
    DistributionsPanel,
    Manual5x5Panel,
    MeasurementsPanel,
    MethodsPanel,
    OverlayPanel,
    QualityPanel,
    RunMethodsDialog,
    SummaryPanel,
    WorkspaceInspector,
)
from .workspace_controller import WorkspaceController

logger = logging.getLogger(__name__)

DEFAULT_DATASET_CANDIDATES = (
    Path("local_data/zeiss/30-07-26"),
    Path("/home/kegouro/HIBRIS/Workshop ⁄ Proyectos/fathom-fibers/local_data/zeiss/30-07-26"),
)

OVERLAY_DEFAULTS = {"manual", "edges", "refined_centerline", "refined_edges"}


class MainWindow(QMainWindow):
    """Scientific desktop shell; all domain mutations delegate to ProjectSession."""

    def __init__(
        self,
        session: ProjectSession | None = None,
        *,
        initial_path: str | None = None,
        smoke_test: bool = False,
        workspace_controller: WorkspaceController | None = None,
    ) -> None:
        super().__init__()
        self.session = session or ProjectSession(FathomEngine())
        self.session.subscribe(self._session_event)
        self.history_bridge = HistoryBridge(self.session)
        self.thread_pool = QThreadPool(self)
        self.active_task: AnalysisTask | None = None
        self._selection_sync = False
        self._smoke_test = smoke_test
        self._field_sample_index: int | None = None
        self._field_position: QRectF | None = None
        self._overlay_state: set[str] = set(OVERLAY_DEFAULTS)
        self.setWindowTitle("Fathom Fibers")
        self.resize(1500, 920)
        self.setMinimumSize(1050, 680)

        self.viewer = ScientificImageView()
        self.setCentralWidget(self.viewer)
        self.workspace = workspace_controller or WorkspaceController(self.session, self)
        self.overlay_layers = OverlayLayers(self.viewer)
        self.viewer.set_scientific_overlays(self.overlay_layers)

        self.project_panel = ProjectPanel(self.session)
        self.results_panel = ResultsPanel(self.session)
        self.inspector_panel = InspectorPanel()
        self.analysis_panel = AnalysisPanel()
        self.comparison_panel = ComparisonPanel()
        self.batch_review_panel = BatchReviewPanel()
        self.history_panel = HistoryPanel(self.session)

        self.dataset_panel = DatasetPanel()
        self.overlay_panel = OverlayPanel()
        self.workspace_inspector = WorkspaceInspector()
        self.summary_panel = SummaryPanel()
        self.distributions_panel = DistributionsPanel()
        self.methods_panel = MethodsPanel()
        self.measurements_panel = MeasurementsPanel(self.session)
        self.quality_panel = QualityPanel()
        self.manual_panel = Manual5x5Panel()

        self._build_docks()
        self._build_actions()
        self._build_menus_and_toolbars()
        self._build_status_bar()
        self._wire()
        self._wire_workspace()
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

    # ------------------------------------------------------------------ docks

    def _build_docks(self) -> None:
        project_dock = QDockWidget("PROJECT", self)
        project_dock.setObjectName("projectDock")
        project_dock.setWidget(self.project_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, project_dock)

        dataset_dock = QDockWidget("DATASET", self)
        dataset_dock.setObjectName("datasetDock")
        dataset_dock.setWidget(self.dataset_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dataset_dock)

        overlay_dock = QDockWidget("OVERLAYS", self)
        overlay_dock.setObjectName("overlayDock")
        overlay_dock.setWidget(self.overlay_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, overlay_dock)

        inspector_tabs = QTabWidget()
        inspector_tabs.addTab(self.inspector_panel, "Measurement")
        inspector_tabs.addTab(self.workspace_inspector, "Workspace")
        inspector_tabs.addTab(self._display_panel(), "Display")
        inspector_dock = QDockWidget("INSPECTOR", self)
        inspector_dock.setObjectName("inspectorDock")
        inspector_dock.setWidget(inspector_tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, inspector_dock)

        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.addTab(self.summary_panel, "SUMMARY")
        self.bottom_tabs.addTab(self.distributions_panel, "DISTRIBUTIONS")
        self.bottom_tabs.addTab(self.methods_panel, "METHODS")
        self.bottom_tabs.addTab(self.measurements_panel, "MEASUREMENTS")
        self.bottom_tabs.addTab(self.quality_panel, "QUALITY")
        self.bottom_tabs.addTab(self.manual_panel, "MANUAL 5×5")
        self.bottom_tabs.addTab(self.results_panel, "RESULTS / MEASUREMENTS")
        self.bottom_tabs.addTab(self.history_panel, "HISTORY")
        self.bottom_tabs.addTab(self.analysis_panel, "ANALYSIS")
        self.bottom_tabs.addTab(self.comparison_panel, "COMPARE METHODS")
        self.bottom_tabs.addTab(self.batch_review_panel, "BATCH MEASUREMENT REVIEW")
        bottom_dock = QDockWidget("RESULTS / HISTORY / ANALYSIS", self)
        bottom_dock.setObjectName("resultsDock")
        bottom_dock.setWidget(self.bottom_tabs)
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

    # ---------------------------------------------------------------- actions

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
        self.open_dataset_action = self._action("Open dataset…", self.open_dataset_dialog)
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
        self.zoom_in_action = self._action("Zoom in", self._zoom_in, "Ctrl++")
        self.zoom_out_action = self._action("Zoom out", self._zoom_out, "Ctrl+-")
        self.previous_image_action = self._action("Previous image", self._previous_image, "Left")
        self.next_image_action = self._action("Next image", self._next_image, "Right")
        self.run_action = self._action("Run methods…", self._run_methods_dialog, "R")
        self.run_all_action = self._action("Run all dataset methods", self._run_all_dataset)
        self.compare_action = self._action("Compare methods", self._show_comparison)
        self.manual_action = self._action("Manual measurement", self._start_manual_tool, "M")
        self.manual_5x5_action = self._action("Manual 5×5 workflow", self._show_manual_5x5)
        self.report_action = self._action("Generate scientific report", self._generate_report, "Ctrl+R")
        self.dataset_report_action = self._action(
            "Generate dataset scientific report", self._generate_dataset_report
        )
        self.export_results_action = self._action("Export current image results…", self._export_current)
        self.export_dataset_action = self._action("Export dataset results…", self._export_dataset)
        self.export_bundle_action = self._action("Export Analysis Bundle…", self._export_bundle)
        self.methods_help_action = self._action("About methods…", self._methods_help)

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

        self.view_mode_group = QActionGroup(self)
        self.view_mode_group.setExclusive(True)
        self.view_mode_group.triggered.connect(self._view_mode_changed)
        self.view_image_action = self._action("Image", self._noop, checkable=True)
        self.view_image_action.setChecked(True)
        self.view_measurements_action = self._action("Measurements", self._noop, checkable=True)
        self.view_comparison_action = self._action("Comparison", self._noop, checkable=True)
        self.view_manual_action = self._action("Manual Review", self._noop, checkable=True)
        for action in (
            self.view_image_action,
            self.view_measurements_action,
            self.view_comparison_action,
            self.view_manual_action,
        ):
            self.view_mode_group.addAction(action)

    def _build_menus_and_toolbars(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        for action in (
            self.open_image_action,
            self.open_dataset_action,
            self.open_project_action,
            self.save_action,
            self.save_as_action,
            self.export_action,
            self.quit_action,
        ):
            file_menu.addAction(action)
        image_menu = self.menuBar().addMenu("&Image")
        image_menu.addActions(
            (
                self.previous_image_action,
                self.next_image_action,
                self.fit_action,
                self.actual_action,
                self.reset_action,
                self.zoom_in_action,
                self.zoom_out_action,
            )
        )
        run_menu = self.menuBar().addMenu("&Run")
        run_menu.addActions((self.run_action, self.run_all_action, self.compare_action))
        report_menu = self.menuBar().addMenu("&Report")
        report_menu.addActions(
            (
                self.report_action,
                self.dataset_report_action,
                self.export_bundle_action,
                self.export_results_action,
                self.export_dataset_action,
            )
        )
        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addActions((self.undo_action, self.redo_action, self.delete_action))
        view_menu = self.menuBar().addMenu("&View")
        view_menu.addActions((self.fit_action, self.actual_action, self.reset_action))
        view_menu.addSeparator()
        view_menu.addActions(
            (
                self.view_image_action,
                self.view_measurements_action,
                self.view_comparison_action,
                self.view_manual_action,
            )
        )
        tools_menu = self.menuBar().addMenu("&Tools")
        tools_menu.addActions(self.tool_group.actions())
        tools_menu.addSeparator()
        tools_menu.addAction(self.cancel_action)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.methods_help_action)

        toolbar = QToolBar("Main", self)
        toolbar.setObjectName("mainToolbar")
        toolbar.setMovable(False)
        toolbar.addActions(
            (self.open_image_action, self.open_dataset_action, self.previous_image_action, self.next_image_action)
        )
        toolbar.addSeparator()
        toolbar.addActions((self.fit_action, self.actual_action, self.zoom_in_action, self.zoom_out_action))
        toolbar.addSeparator()
        toolbar.addActions((self.run_action, self.compare_action))
        toolbar.addSeparator()
        toolbar.addActions((self.manual_action, self.manual_5x5_action))
        toolbar.addSeparator()
        toolbar.addActions((self.report_action, self.dataset_report_action, self.export_results_action))
        self.addToolBar(toolbar)

    def _build_status_bar(self) -> None:
        self.scientific_notice = QLabel(
            "Measurements represent projected 2D geometry. Automatic results require review."
        )
        self.progress_label = QLabel("")
        self.coordinate_label = QLabel("x —  y —  value —")
        self.statusBar().addWidget(self.scientific_notice, 1)
        self.statusBar().addWidget(self.progress_label)
        self.statusBar().addPermanentWidget(self.coordinate_label)

    # ------------------------------------------------------------------ wire

    def _wire(self) -> None:
        self.viewer.measurementRequested.connect(self._create_measurement)
        self.viewer.roiDrawn.connect(self.session.set_roi)
        self.viewer.recordSelected.connect(self._select_record)
        self.viewer.geometryEdited.connect(self._edit_geometry)
        self.viewer.coordinateChanged.connect(self._show_coordinate)
        self.viewer.fieldSampleClicked.connect(self._select_field_sample)
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

    def _wire_workspace(self) -> None:
        controller = self.workspace
        self.dataset_panel.set_controller(controller)
        self.dataset_panel.imageRequested.connect(controller.select_image)
        self.dataset_panel.openRequested.connect(self.open_dataset_dialog)
        self.dataset_panel.runRequested.connect(self._dataset_run_requested)
        controller.datasetLoaded.connect(self._dataset_loaded)
        controller.imageChanged.connect(self._workspace_image_changed)
        controller.resultsChanged.connect(self._workspace_results_changed)
        controller.busyChanged.connect(self._workspace_busy)
        controller.methodProgress.connect(self._workspace_progress)
        controller.manualChanged.connect(self._workspace_manual_changed)
        controller.reportReady.connect(self._report_ready)
        controller.reportFailed.connect(self._report_failed)
        controller.errorRaised.connect(self._workspace_error)
        self.overlay_panel.overlayChanged.connect(self._overlay_toggled)
        self.overlay_panel.densityChanged.connect(self._overlay_density)
        self.measurements_panel.fieldSampleSelected.connect(self._select_field_sample)
        self.measurements_panel.recordSelected.connect(self._select_record)
        self.methods_panel.methodSelected.connect(self._method_selected)
        self.manual_panel.targetRequested.connect(self._manual_target)
        self.manual_panel.removeRequested.connect(self._manual_remove)
        self.manual_panel.skipRequested.connect(self._manual_skip)
        self.manual_panel.nextImageRequested.connect(self._manual_next_image)

    # ----------------------------------------------------------- workspace io

    def open_dataset_dialog(self) -> None:
        start = str(DEFAULT_DATASET_CANDIDATES[0]) if DEFAULT_DATASET_CANDIDATES[0].is_dir() else ""
        path = QFileDialog.getExistingDirectory(self, "Open scientific dataset", start)
        if path:
            self.open_dataset(path)

    def open_dataset(self, path: str | Path) -> None:
        try:
            dataset = self.workspace.open_dataset(path)
            self.statusBar().showMessage(
                f"Opened dataset {dataset.dataset_id} ({len(dataset.images)} images)", 7000
            )
        except Exception as exc:
            logger.exception("Failed to open dataset %s", path)
            QMessageBox.critical(self, "Open dataset failed", str(exc))

    def _dataset_loaded(self) -> None:
        self.dataset_panel.refresh()

    def _dataset_run_requested(self, action: str) -> None:
        if action == "all":
            self.workspace.run_all_dataset()
        else:
            self.workspace.run_missing()

    def _workspace_image_changed(self) -> None:
        image = self.session.image
        self.viewer.set_image(image)
        self._field_sample_index = None
        self._field_position = None
        self._refresh_workspace_panels()
        self._refresh_overlays()
        self._refresh_manual_panel()
        self._refresh_all()
        self._update_calibration()
        self._update_title()

    def _workspace_results_changed(self) -> None:
        self._refresh_workspace_panels()
        self._refresh_overlays()
        self._refresh_manual_panel()

    def _refresh_workspace_panels(self) -> None:
        comparison = self.workspace.comparison
        payload = self.workspace.summary_payload
        image = self.session.image
        self.summary_panel.set_image(image)
        if comparison is not None:
            self.summary_panel.set_comparison(comparison)
            self.distributions_panel.set_comparison(comparison)
            self.methods_panel.set_comparison(comparison)
            self.measurements_panel.set_comparison(comparison)
            self.quality_panel.set_comparison(comparison)
            self.comparison_panel.set_unified_result(comparison, image)
        elif payload is not None:

            self.summary_panel.set_comparison(None)
            self.distributions_panel.set_comparison(None)
            self.methods_panel.set_comparison(None)
            self.measurements_panel.set_comparison(None)
            self.quality_panel.set_comparison(None)
            self._summary_only_view(payload)
        else:
            self.summary_panel.set_comparison(None)
            self.distributions_panel.set_comparison(None)
            self.methods_panel.set_comparison(None)
            self.measurements_panel.set_comparison(None)
            self.quality_panel.set_comparison(None)
        self.dataset_panel.refresh()
        self._field_sample_index = None
        self.workspace_inspector.clear()

    def _summary_only_view(self, payload: dict[str, Any]) -> None:
        self.summary_panel.info.append(
            "<p style='color:#8a6d1a'>Summary cache only — run methods for full samples.</p>"
        )

    def _refresh_overlays(self) -> None:
        results = self.workspace.results
        image = self.session.image
        self.overlay_panel.set_availability(results)
        if results and image is not None:
            self.overlay_layers.set_payload(build_overlay_payload(results, image))
            self.overlay_layers.set_density(self.overlay_panel.density.currentText())
            for key in self.overlay_layers.LAYER_NAMES:
                check = self.overlay_panel.checks.get(key)
                if check is not None:
                    check.setChecked(key in self._overlay_state)
                self.overlay_layers.set_visible(key, key in self._overlay_state)
        else:
            self.overlay_layers.set_payload({})
        self._refresh_field_selection()

    def _overlay_toggled(self, key: str, visible: bool) -> None:
        if visible:
            self._overlay_state.add(key)
        else:
            self._overlay_state.discard(key)
        if key == "manual":
            self._refresh_manual_records_overlay()
            return
        self.overlay_layers.set_visible(key, visible)

    def _refresh_manual_records_overlay(self) -> None:
        project = self.session.project
        records = project.records if project and "manual" in self._overlay_state else []
        self.viewer.set_records(records, self.session.selected_record_id)

    def _overlay_density(self, density: str) -> None:
        self.overlay_layers.set_density(density)

    def _refresh_field_selection(self) -> None:
        if self._field_sample_index is None:
            self.viewer.set_field_selection_highlight(None)
            self.workspace_inspector.clear()
            return
        index = self._field_sample_index
        samples = self.workspace.results.get(MethodId.FATHOM_FIELD_GRAPH_V1)
        samples = samples.local_samples if samples else None
        rect = self.overlay_layers.selection_rect(index)
        self.viewer.set_field_selection_highlight(rect)
        self.workspace_inspector.set_field_sample(samples, index)

    def _select_field_sample(self, index: int) -> None:
        self._field_sample_index = int(index)
        self.measurements_panel.select_field_row(self._field_sample_index)
        self._refresh_field_selection()
        point = self.overlay_layers.sample_position_px(self._field_sample_index)
        if point is not None:
            self.viewer.centerOn(point)
            self.viewer.set_field_selection_highlight(self.overlay_layers.selection_rect(self._field_sample_index))
            self.workspace_inspector.set_field_sample(
                self.workspace.results.get(MethodId.FATHOM_FIELD_GRAPH_V1).local_samples
                if self.workspace.results.get(MethodId.FATHOM_FIELD_GRAPH_V1)
                else None,
                self._field_sample_index,
            )

    def _method_selected(self, method_id: str) -> None:
        result = self.workspace.results.get(MethodId(method_id))
        self.workspace_inspector.set_result_provenance(result)

    def _workspace_busy(self, busy: bool) -> None:
        self.run_action.setEnabled(not busy)
        self.run_all_action.setEnabled(not busy)
        self.dataset_panel.run_all_button.setEnabled(not busy)
        self.dataset_panel.run_missing_button.setEnabled(not busy)
        if not busy:
            self.progress_label.setText("")

    def _workspace_progress(self, message: str) -> None:
        self.progress_label.setText(message)
        self.statusBar().showMessage(message, 15000)

    def _workspace_error(self, message: str, details: str) -> None:
        logger.error("Workspace error: %s\n%s", message, details)
        self.progress_label.setText(f"Failed: {message}")
        if self._smoke_test:
            self.statusBar().showMessage(f"Failed: {message}", 15000)
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("Analysis failed")
        box.setText(message)
        box.setDetailedText(details[:4000])
        box.exec()

    def _report_ready(self, path: str) -> None:
        self.statusBar().showMessage(f"Report generated: {path}", 15000)
        if self._smoke_test:
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Scientific report")
        box.setText(f"Report generated:\n{path}")
        open_button = box.addButton("Open in browser", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_button:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices

            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _report_failed(self, message: str, details: str) -> None:
        self._workspace_error(f"Report failed: {message}", details)

    # ------------------------------------------------------------- navigation

    def _previous_image(self) -> None:
        self.workspace.previous_image()

    def _next_image(self) -> None:
        self.workspace.next_image()

    def _zoom_in(self) -> None:
        self.viewer.scale(1.25, 1.25)

    def _zoom_out(self) -> None:
        self.viewer.scale(0.8, 0.8)

    def _run_methods_dialog(self) -> None:
        if self.workspace.dataset is None:
            self.statusBar().showMessage("Open a dataset before running methods.", 5000)
            return
        dialog = RunMethodsDialog(self)
        if dialog.exec():
            if dialog.current_button == "missing":
                self.workspace.run_missing()
            elif dialog.current_button == "all":
                self.workspace.run_all_dataset()
            else:
                self.workspace.run_current_image()

    def _run_all_dataset(self) -> None:
        self.workspace.run_all_dataset()

    def _show_comparison(self) -> None:
        if self.workspace.comparison is None and self.workspace.current_image is not None:
            self.workspace.run_current_image()
        self.bottom_tabs.setCurrentWidget(self.comparison_panel)

    def _start_manual_tool(self) -> None:
        self.viewer.activate_tool("projected_width")
        self.tool_actions["projected_width"].setChecked(True)

    def _show_manual_5x5(self) -> None:
        self.bottom_tabs.setCurrentWidget(self.manual_panel)
        if self.workspace.current_image is not None:
            self.manual_panel.next_target()

    def _generate_report(self) -> None:
        self.workspace.generate_image_report()

    def _generate_dataset_report(self) -> None:
        self.workspace.generate_dataset_report()

    def _export_current(self) -> None:
        if self.workspace.current_image is None:
            self.statusBar().showMessage("Open an image before exporting.", 5000)
            return
        directory = QFileDialog.getExistingDirectory(self, "Export current image results")
        if not directory:
            return
        try:
            written = self.workspace.export_current_results(directory)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {len(written)} files to {directory}", 8000)

    def _export_bundle(self) -> None:
        if self.workspace.dataset is None:
            self.statusBar().showMessage("Open a dataset before exporting.", 5000)
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Export analysis bundle", str(Path.cwd() / "release")
        )
        if not directory:
            return
        self.workspace.export_analysis_bundle(directory)
        self.statusBar().showMessage("Analysis bundle export running…", 5000)

    def _export_dataset(self) -> None:
        if self.workspace.dataset is None:
            self.statusBar().showMessage("Open a dataset before exporting.", 5000)
            return
        directory = QFileDialog.getExistingDirectory(self, "Export dataset results")
        if not directory:
            return
        try:
            written = self.workspace.export_dataset_results(directory)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {len(written)} files to {directory}", 8000)

    def _methods_help(self) -> None:
        text = (
            "<h3>MATLAB SIMPoly</h3><p>Native MATLAB implementation of the SIMPoly pipeline. "
            "In this workspace it is consumed from the validated oracle cache; b1 is the "
            "reported native Gaussian center.</p>"
            "<h3>Python SIMPoly</h3><p>Python port of SIMPoly. Common estimand: calibrated "
            "length-weighted diameters on the skeleton. Known divergence: bwskel.</p>"
            "<h3>Fathom Local</h3><p>Local cross-section metrology on detected fiber "
            "candidates within the active ROI.</p>"
            "<h3>Fathom Field</h3>"
            "<p><b>EDT</b> — Twice the physical distance from the sampled centerline to the "
            "nearest background boundary.</p>"
            "<p><b>Paired Edge</b> — Distance between both local mask boundaries measured "
            "along the local fiber normal.</p>"
            "<p><b>Intensity Profile</b> — Paired-edge width refined against local subpixel "
            "gradient transitions in the raw SEM image.</p>"
            "<p>Fathom Field is an experimental field-measuring method. Graph reconstruction "
            "and fiber instances are not implemented.</p>"
            "<h3>Fathom Field / Oriented Ribbon V1</h3>"
            "<p><b>EXPERIMENTAL</b> — geometric centerline refinement from paired opposite "
            "boundaries: local midpoints, a confidence-weighted smooth centerline on "
            "non-branching runs, then re-measurement of EDT, paired-edge and profile along "
            "it. Validated on known-truth synthetic geometry; real SEM results represent "
            "method behavior/agreement, not known absolute accuracy.</p>"
            "<h3>Manual 5×5</h3><p>Operator reference grid: 25 positions per image, "
            "perpendicular width measurements.</p>"
            "<h3>Consensus</h3><p>Equal-method quantile pseudo-reference across participating "
            "methods. Not ground truth.</p>"
        )
        QMessageBox.information(self, "About methods", text)

    def _view_mode_changed(self, action: QAction) -> None:
        if action is self.view_measurements_action:
            self.bottom_tabs.setCurrentWidget(self.measurements_panel)
        elif action is self.view_comparison_action:
            self.bottom_tabs.setCurrentWidget(self.comparison_panel)
        elif action is self.view_manual_action:
            self.bottom_tabs.setCurrentWidget(self.manual_panel)

    @staticmethod
    def _noop() -> None:
        pass

    # ------------------------------------------------------- manual 5x5 flow

    def _refresh_manual_panel(self) -> None:
        image = self.workspace.current_image
        review = self.workspace.manual_review
        if image is None or review is None:
            self.manual_panel.set_review(None)
            return
        self.manual_panel.set_review(review, image.case_id)
        index = self.workspace.current_index + 1
        count = len(self.workspace.dataset.images)
        self.manual_panel.set_image_index(index, count, self.workspace.manual.total_measured)

    def _manual_target(self, row: int, column: int) -> None:
        image = self.session.image
        if image is None:
            return
        self._start_manual_tool()
        rect = _manual_cell_rect(row, column, image)
        self.viewer.set_manual_target(rect)
        self.viewer.fitInView(rect.adjusted(-rect.width() * 0.25, -rect.height() * 0.25, rect.width() * 0.25, rect.height() * 0.25), Qt.AspectRatioMode.KeepAspectRatio)

    def _manual_remove(self, row: int, column: int) -> None:
        self.workspace.remove_manual_measurement(row, column)
        record = self._record_for_cell(row, column)
        if record is not None:
            self.session.delete_records([record.measurement_id])
        self._refresh_manual_panel()
        self._refresh_all()

    def _record_for_cell(self, row: int, column: int) -> MeasurementRecord | None:
        position = f"R{row + 1}C{column + 1}"
        for record in self.session.project.records:
            if record.protocol_snapshot.get("protocol_id") == "MANUAL_5X5_REFERENCE" and (
                record.protocol_snapshot.get("grid_position") == position
            ):
                return record
        return None

    def _manual_skip(self, row: int, column: int) -> None:
        reason, accepted = QInputDialog.getText(
            self, "Skip grid position", "Scientific reason:"
        )
        if not accepted or not reason.strip():
            return
        self.workspace.skip_manual_measurement(row, column, reason)
        self._refresh_manual_panel()
        self._refresh_all()

    def _manual_next_image(self) -> None:
        self.workspace.next_image()

    def _workspace_manual_changed(self) -> None:
        self._refresh_manual_panel()
        self._refresh_workspace_panels()
        self.manual_panel.accept_and_advance()

    # ------------------------------------------------------------- session

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
        elif event == "records":
            self._refresh_all()

    def _refresh_all(self) -> None:
        self._selection_sync = True
        try:
            self._refresh_manual_records_overlay()
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
        if record_id is None:
            self._field_sample_index = None
            self.viewer.set_field_selection_highlight(None)
        if self._selection_sync or record_id == self.session.selected_record_id:
            return
        self.session.select(record_id)

    def _update_title(self) -> None:
        project = self.session.project
        if project is None:
            self.setWindowTitle("Fathom Fibers")
            return
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

    # ------------------------------------------------------------- open/save

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

    # ---------------------------------------------------------- measurements

    def _create_measurement(self, kind: str, geometry: dict[str, Any]) -> None:
        try:
            record = self.session.create_measurement(kind, geometry)
        except Exception as exc:
            logger.exception("Measurement creation failed")
            self.statusBar().showMessage(f"Measurement rejected: {exc}", 7000)
            return
        case = self.batch_review_panel.current_case()
        grid = self.batch_review_panel.active_grid_position()
        if kind == "PROJECTED_WIDTH":
            if self._try_record_manual_5x5(record):
                return
            if case is not None and grid is not None:
                position = f"R{grid[0] + 1}C{grid[1] + 1}"
                self.session.annotate_manual_grid(
                    record.measurement_id, case_id=case["case_id"], grid_position=position
                )
                self.batch_review_panel.record_measurement(record)
        self.statusBar().showMessage(f"Created {record.measurement_id}", 3000)

    def _try_record_manual_5x5(self, record: MeasurementRecord) -> bool:
        if self.workspace.current_image is None:
            return False
        position = self.manual_panel.active_grid_position()
        case_id = self.manual_panel.current_case_id()
        if position is None or case_id is None:
            return False
        row, column = position
        if record.kind.value != "PROJECTED_WIDTH":
            return False
        grid_position = f"R{row + 1}C{column + 1}"
        try:
            self.session.annotate_manual_grid(
                record.measurement_id, case_id=case_id, grid_position=grid_position
            )
        except Exception:
            logger.exception("Manual grid annotation failed for %s", record.measurement_id)
        self.workspace.accept_manual_measurement(record, row, column)
        self.statusBar().showMessage(f"Manual 5×5 {grid_position} recorded and autosaved", 4000)
        return True

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

    # -------------------------------------------------------------- analysis

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
        self._start_task(self.session.compare_all_methods, self._comparison_done)

    def _comparison_done(self, result: MethodComparisonResult | UnifiedMethodComparison) -> None:
        if self.session.image is not None:
            if isinstance(result, UnifiedMethodComparison):
                self.comparison_panel.set_unified_result(result, self.session.image)
            else:
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

    # --------------------------------------------------------------- autosave

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
            self.workspace._cancel_tasks()
            event.accept()
        else:
            event.ignore()


def _manual_cell_rect(row: int, column: int, image: ScientificImage) -> QRectF:
    height, width = image.shape
    body_height = image.footer_bounds[0] if image.footer_bounds else height
    cell_width = width / 5.0
    cell_height = body_height / 5.0
    return QRectF(
        column * cell_width,
        row * cell_height,
        cell_width,
        cell_height,
    )
