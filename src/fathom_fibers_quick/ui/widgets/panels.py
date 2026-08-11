from __future__ import annotations

import csv
import getpass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ...application import ProjectSession
from ...core.contracts import MethodComparisonResult, ScientificImage
from ...measurement_records import MeasurementKind, MeasurementRecord, MeasurementStatus
from ...validation.manual_review import GridCellStatus, Manual5x5Review
from ..models import MeasurementFilterModel, MeasurementTableModel, ProjectTreeModel
from ..models.project_tree import ID_ROLE, NODE_ROLE


class ProjectPanel(QWidget):
    recordSelected = Signal(object)
    focusRequested = Signal(str)

    def __init__(self, session: ProjectSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.model = ProjectTreeModel()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter project…")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.setHeaderHidden(False)
        self.tree.setAlternatingRowColors(True)
        self.focus_button = QPushButton("Focus selection")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.search)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.focus_button)
        self.tree.selectionModel().currentChanged.connect(self._current_changed)
        self.focus_button.clicked.connect(self._focus)

    def refresh(self) -> None:
        self.model.set_project(self.session.project)
        self.tree.expandToDepth(2)

    def _current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if current.data(NODE_ROLE) == "measurement":
            self.recordSelected.emit(current.data(ID_ROLE))

    def _focus(self) -> None:
        current = self.tree.currentIndex()
        if current.data(NODE_ROLE) == "measurement":
            self.focusRequested.emit(current.data(ID_ROLE))


class ResultsPanel(QWidget):
    recordSelected = Signal(object)
    acceptRequested = Signal(list)
    rejectRequested = Signal(list)
    deleteRequested = Signal(list)
    focusRequested = Signal(str)
    exportRequested = Signal(list)

    def __init__(self, session: ProjectSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.model = MeasurementTableModel(session)
        self.proxy = MeasurementFilterModel(self)
        self.proxy.setSourceModel(self.model)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search ID, name, source, flags or tags…")
        self.kind = QComboBox()
        self.kind.addItems(["All", *[kind.value for kind in MeasurementKind]])
        self.status = QComboBox()
        self.status.addItems(["All", *[status.value for status in MeasurementStatus]])
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(self.search, 1)
        filter_layout.addWidget(self.kind)
        filter_layout.addWidget(self.status)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._focus_current)

        buttons = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.reject_button = QPushButton("Reject")
        self.delete_button = QPushButton("Delete")
        self.focus_button = QPushButton("Focus")
        self.export_button = QPushButton("Export selected")
        for button in (
            self.accept_button,
            self.reject_button,
            self.delete_button,
            self.focus_button,
            self.export_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addLayout(filter_layout)
        layout.addWidget(self.table, 1)
        layout.addLayout(buttons)
        self.search.textChanged.connect(self._filter)
        self.kind.currentTextChanged.connect(self._filter)
        self.status.currentTextChanged.connect(self._filter)
        self.accept_button.clicked.connect(lambda: self.acceptRequested.emit(self.selected_ids()))
        self.reject_button.clicked.connect(lambda: self.rejectRequested.emit(self.selected_ids()))
        self.delete_button.clicked.connect(lambda: self.deleteRequested.emit(self.selected_ids()))
        self.export_button.clicked.connect(lambda: self.exportRequested.emit(self.selected_ids()))
        self.focus_button.clicked.connect(self._focus_current)

    def refresh(self) -> None:
        self.model.refresh()

    def _filter(self, *_args: Any) -> None:
        self.proxy.set_filters(
            self.search.text(), self.kind.currentText(), self.status.currentText()
        )

    def selected_ids(self) -> list[str]:
        rows = self.table.selectionModel().selectedRows()
        ids: list[str] = []
        for proxy_index in rows:
            source_index = self.proxy.mapToSource(proxy_index)
            record = self.model.record_at(source_index.row())
            if record:
                ids.append(record.measurement_id)
        return ids

    def select_record(self, record_id: str | None) -> None:
        if record_id is None:
            self.table.clearSelection()
            return
        for row, record in enumerate(self.model.records):
            if record.measurement_id == record_id:
                proxy_index = self.proxy.mapFromSource(self.model.index(row, 0))
                if proxy_index.isValid():
                    self.table.selectionModel().select(
                        proxy_index,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect
                        | QItemSelectionModel.SelectionFlag.Rows,
                    )
                    self.table.scrollTo(proxy_index)
                return

    def _selection_changed(self, *_args: Any) -> None:
        ids = self.selected_ids()
        self.recordSelected.emit(ids[0] if ids else None)

    def _focus_current(self, *_args: Any) -> None:
        ids = self.selected_ids()
        if ids:
            self.focusRequested.emit(ids[0])


class InspectorPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.title = QLabel("No measurement selected")
        self.title.setStyleSheet("font-weight: 600; font-size: 13px;")
        self.details = QTextBrowser()
        self.details.setOpenExternalLinks(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.title)
        layout.addWidget(self.details, 1)

    def set_record(self, record: MeasurementRecord | None) -> None:
        if record is None:
            self.title.setText("No measurement selected")
            self.details.setPlainText("")
            return
        self.title.setText(f"{record.measurement_id} — {record.name}")
        rows: list[tuple[str, Any]] = [
            ("Geometry", record.geometry),
            ("Value", record.primary_value),
            ("Unit", record.primary_unit),
            (
                "Pixel equivalent",
                record.values.get("length_px", record.values.get("gaussian_center_px")),
            ),
            ("Calibration", record.calibration_snapshot),
            ("Method / source", record.source.value),
            ("Status", record.status.value),
            ("Protocol", record.protocol_snapshot),
            ("Uncertainty", record.uncertainty),
            ("Flags", record.quality_flags),
            ("Tags", record.tags),
            ("Notes", record.notes),
            ("Created", record.created_at),
            ("Modified", record.updated_at),
        ]
        if record.kind == MeasurementKind.DIAMETER_DISTRIBUTION:
            rows.extend(
                [
                    ("Profile", record.values.get("profile")),
                    ("Estimand", record.values.get("estimand")),
                    ("Gaussian center", record.values.get("gaussian_center_px")),
                    ("Source-reported stdev", record.values.get("source_reported_stdev_px")),
                    ("Mathematical sigma", record.values.get("mathematical_gaussian_sigma_px")),
                    ("Arithmetic mean", record.values.get("arithmetic_mean_px")),
                    ("Median", record.values.get("median_px")),
                    ("Valid diameters", record.values.get("valid_diameter_count")),
                    ("Foreground fraction", record.values.get("foreground_fraction")),
                    ("Skeleton count", record.values.get("skeleton_count")),
                ]
            )
        html = (
            "<table>"
            + "".join(
                f"<tr><th align='left' valign='top'>{key}</th><td>{value!s}</td></tr>"
                for key, value in rows
            )
            + "</table>"
        )
        if record.kind == MeasurementKind.DIAMETER_DISTRIBUTION:
            html += (
                "<p><b>Interpretation:</b> SIMPoly's distribution fit and Fathom's "
                "section measurements do not necessarily share the same estimand.</p>"
            )
        self.details.setHtml(html)


class AnalysisPanel(QWidget):
    runRequested = Signal(str)
    cancelRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.method = QComboBox()
        self.method.addItems(
            [
                "Fathom Assisted ROI",
                "SIMPoly Source Compatible",
                "SIMPoly Controlled Input",
            ]
        )
        self.sections = QSpinBox()
        self.sections.setRange(1, 25)
        self.sections.setValue(3)
        self.parameters = QLabel(
            "Defaults are recorded with provenance. Results are review proposals and are not "
            "included in primary statistics until accepted."
        )
        self.parameters.setWordWrap(True)
        form = QFormLayout()
        form.addRow("Method", self.method)
        form.addRow("Fathom sections", self.sections)
        self.run_button = QPushButton("Run")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.run_button)
        row.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("Ready")
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.parameters)
        layout.addLayout(row)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addStretch(1)
        self.run_button.clicked.connect(lambda: self.runRequested.emit(self.method.currentText()))
        self.cancel_button.clicked.connect(self.cancelRequested)

    def set_running(self, running: bool, message: str = "") -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.progress.setRange(0, 0 if running else 100)
        if not running:
            self.progress.setValue(100 if message else 0)
        self.status.setText(message or ("Running…" if running else "Ready"))


def _pixmap_from_array(array: np.ndarray) -> QPixmap:
    data = np.asarray(array)
    if data.dtype == bool:
        display = data.astype(np.uint8) * 255
    else:
        finite = data[np.isfinite(data)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            display = np.clip((data - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
            display = np.round(display * 255).astype(np.uint8)
        else:
            display = np.zeros(data.shape, dtype=np.uint8)
    display = np.ascontiguousarray(display)
    height, width = display.shape[:2]
    image = QImage(
        display.data,
        width,
        height,
        display.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(image)


class ComparisonPanel(QWidget):
    runRequested = Signal()

    HEADERS = (
        "Method",
        "Estimand",
        "N",
        "Mean px",
        "Median px",
        "Main value px",
        "Difference px",
        "Relative %",
        "Flags",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.run_button = QPushButton("Compare methods on current ROI")
        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.previews = QTabWidget()
        self.preview_labels: dict[str, QLabel] = {}
        for name in ("Original ROI + Fathom", "SIMPoly mask", "SIMPoly skeleton", "Diameter map"):
            label = QLabel("No result")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(180)
            label.setScaledContents(False)
            self.previews.addTab(label, name)
            self.preview_labels[name] = label
        layout = QVBoxLayout(self)
        layout.addWidget(self.run_button)
        layout.addWidget(self.table)
        layout.addWidget(self.previews, 1)
        self.run_button.clicked.connect(self.runRequested)

    def set_result(self, result: MethodComparisonResult, image: ScientificImage) -> None:
        self.table.setRowCount(len(result.rows))
        for row_index, row in enumerate(result.rows):
            values = (
                row.method,
                row.estimand,
                row.n,
                row.mean_px,
                row.median_px,
                row.main_reported_px,
                row.difference_px,
                row.relative_difference_percent,
                ", ".join(row.flags),
            )
            for column, value in enumerate(values):
                if isinstance(value, float):
                    text = f"{value:.5g}"
                else:
                    text = "—" if value is None else str(value)
                self.table.setItem(row_index, column, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        x0, y0, x1, y1 = result.roi_bbox
        original = _pixmap_from_array(image.gray[y0:y1, x0:x1])
        painter = QPainter(original)
        painter.setPen(QPen(QColor("#ffd166"), 2.0))
        for candidate in result.fathom.candidates:
            for proposal in candidate.proposed_measurements:
                painter.drawLine(
                    round(proposal.p1[0] - x0),
                    round(proposal.p1[1] - y0),
                    round(proposal.p2[0] - x0),
                    round(proposal.p2[1] - y0),
                )
        painter.end()
        pixmaps = {
            "Original ROI + Fathom": original,
            "SIMPoly mask": _pixmap_from_array(result.simpoly_intermediates.thickened_mask),
            "SIMPoly skeleton": _pixmap_from_array(result.simpoly_intermediates.valid_skeleton),
            "Diameter map": _pixmap_from_array(result.simpoly_intermediates.distance_map),
        }
        for name, pixmap in pixmaps.items():
            self.preview_labels[name].setPixmap(
                pixmap.scaled(
                    500,
                    260,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )


class BatchReviewPanel(QWidget):
    """Manifest-driven 16-image review UI; external methods remain adapters."""

    imageRequested = Signal(str)
    runRequested = Signal(str)
    saveRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.manifest_path: Path | None = None
        self.queue_path: Path | None = None
        self.review_path: Path | None = None
        self.cases: list[dict[str, Any]] = []
        self.queue: list[dict[str, str]] = []
        self.index = 0
        self.reviews: dict[str, Manual5x5Review] = {}
        self.position = QLabel("No campaign loaded")
        self.filename = QLabel("—")
        self.filename.setWordWrap(True)
        self.global_progress = QLabel("Images reviewed: 0 / 16")
        self.manual_progress = QLabel("Manual measurements: 0 / 400")
        self.current_progress = QLabel("Current image measurements: 0 / 25")
        self.grid = QTableWidget(5, 5)
        self.grid.setHorizontalHeaderLabels([str(index) for index in range(1, 6)])
        self.grid.setVerticalHeaderLabels([str(index) for index in range(1, 6)])
        self.grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.grid.cellClicked.connect(self._cycle_cell)
        self.buttons: dict[str, QPushButton] = {}
        navigation = QHBoxLayout()
        for label, callback in (
            ("Previous", self.previous),
            ("Next", self.next),
            ("Mark Reviewed", lambda: self._set_manual_status("REVIEWED")),
            ("Skip", lambda: self._set_manual_status("SKIPPED")),
            ("Flag", lambda: self._set_manual_status("FLAGGED")),
        ):
            button = QPushButton(label)
            button.clicked.connect(callback)
            navigation.addWidget(button)
            self.buttons[label] = button
        actions = QHBoxLayout()
        for label, action in (
            ("Run Fathom", "fathom"),
            ("Run MATLAB SIMPoly", "matlab"),
            ("Run Python SIMPoly", "python"),
            ("Compare", "compare"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=action: self.runRequested.emit(value)
            )
            actions.addWidget(button)
            self.buttons[label] = button
        save = QPushButton("Save")
        save.clicked.connect(self.saveRequested)
        actions.addWidget(save)
        layout = QVBoxLayout(self)
        layout.addWidget(self.position)
        layout.addWidget(self.filename)
        layout.addLayout(navigation)
        layout.addLayout(actions)
        layout.addWidget(self.global_progress)
        layout.addWidget(self.manual_progress)
        layout.addWidget(self.current_progress)
        layout.addWidget(self.grid)

    def load_manifest(self, path: str | Path) -> None:
        self.manifest_path = Path(path)
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != "ZEISS_PVDF_2026-07-30" or payload.get("case_count") != 16:
            raise ValueError("Batch Measurement Review requires the frozen 16-image manifest")
        self.cases = list(payload["cases"])
        self.queue_path = self.manifest_path.with_name("review_queue.csv")
        self.review_path = self.manifest_path.with_name("manual_grid.json")
        if self.queue_path.exists():
            with self.queue_path.open(newline="", encoding="utf-8") as handle:
                self.queue = list(csv.DictReader(handle))
        else:
            self.queue = [
                {"case_id": case["case_id"], "manual_status": "NOT_MEASURED"} for case in self.cases
            ]
        self.reviews = {case["case_id"]: Manual5x5Review(case["case_id"]) for case in self.cases}
        if self.review_path.exists():
            stored = json.loads(self.review_path.read_text(encoding="utf-8"))
            self.reviews.update(
                {case_id: Manual5x5Review.from_dict(value) for case_id, value in stored.items()}
            )
        self.index = 0
        self._refresh()

    def current_case(self) -> dict[str, Any] | None:
        return self.cases[self.index] if self.cases else None

    def active_grid_position(self) -> tuple[int, int] | None:
        row, column = self.grid.currentRow(), self.grid.currentColumn()
        return (row, column) if row >= 0 and column >= 0 else None

    def record_measurement(self, record: MeasurementRecord) -> None:
        case = self.current_case()
        position = self.active_grid_position()
        if case is None or position is None:
            return
        cell = self.reviews[case["case_id"]].cell(*position)
        cell.status = GridCellStatus.MEASURED
        cell.fiber_id = record.fiber_id
        cell.measurement_id = record.measurement_id
        cell.geometry = dict(record.geometry)
        cell.diameter = record.primary_value
        cell.unit = record.primary_unit
        cell.calibration_snapshot = dict(record.calibration_snapshot)
        cell.operator = getpass.getuser()
        cell.timestamp = record.created_at
        cell.notes = record.notes
        self._save_reviews()
        self._refresh()

    def previous(self) -> None:
        if self.cases:
            self.index = max(0, self.index - 1)
            self._refresh()
            self._emit_image()

    def next(self) -> None:
        if self.cases:
            self.index = min(len(self.cases) - 1, self.index + 1)
            self._refresh()
            self._emit_image()

    def _emit_image(self) -> None:
        case = self.current_case()
        if case:
            self.imageRequested.emit(case["absolute_path"])

    def _cycle_cell(self, row: int, column: int) -> None:
        case = self.current_case()
        if case is None:
            return
        cell = self.reviews[case["case_id"]].cell(row, column)
        states = (
            GridCellStatus.NOT_REVIEWED,
            GridCellStatus.MEASURED,
            GridCellStatus.NO_VALID_FIBER,
            GridCellStatus.SKIPPED_WITH_REASON,
        )
        next_status = states[(states.index(cell.status) + 1) % len(states)]
        notes = ""
        if next_status == GridCellStatus.SKIPPED_WITH_REASON:
            notes, accepted = QInputDialog.getText(self, "Skip grid position", "Scientific reason:")
            if not accepted or not notes.strip():
                return
        cell.set_status(next_status, notes=notes)
        self._save_reviews()
        self._refresh()

    def _set_manual_status(self, status: str) -> None:
        case = self.current_case()
        if case is None:
            return
        for row in self.queue:
            if row.get("case_id") == case["case_id"]:
                row["manual_status"] = status
                break
        self._save_queue()
        self._refresh()

    def _save_queue(self) -> None:
        if self.queue_path is None or not self.queue:
            return
        with self.queue_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.queue[0]))
            writer.writeheader()
            writer.writerows(self.queue)

    def _save_reviews(self) -> None:
        if self.review_path is None:
            return
        payload = {case_id: review.to_dict() for case_id, review in self.reviews.items()}
        temporary = self.review_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.review_path)

    def _refresh(self) -> None:
        case = self.current_case()
        if case is None:
            return
        review = self.reviews[case["case_id"]]
        self.position.setText(f"Image {self.index + 1} / {len(self.cases)} — {case['case_id']}")
        self.filename.setText(case["filename"])
        reviewed = sum(row.get("manual_status") in {"REVIEWED", "SKIPPED"} for row in self.queue)
        total_measured = sum(item.measurement_count for item in self.reviews.values())
        self.global_progress.setText(f"Images reviewed: {reviewed} / 16")
        self.manual_progress.setText(f"Manual measurements: {total_measured} / 400")
        self.current_progress.setText(
            f"Current image measurements: {review.measurement_count} / 25"
        )
        for row in range(5):
            for column in range(5):
                status = review.cell(row, column).status.value
                self.grid.setItem(
                    row, column, QTableWidgetItem(status.replace("NOT_REVIEWED", "—"))
                )


class HistoryPanel(QWidget):
    def __init__(self, session: ProjectSession, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.list = QListWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.list)

    def refresh(self) -> None:
        self.list.clear()
        self.list.addItems(self.session.history.get_log_entries(200))
