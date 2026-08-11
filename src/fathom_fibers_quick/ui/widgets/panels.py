from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QItemSelectionModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
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
        self.proxy.set_filters(self.search.text(), self.kind.currentText(), self.status.currentText())

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
            ("Pixel equivalent", record.values.get("length_px", record.values.get("gaussian_center_px"))),
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
        html = "<table>" + "".join(
            f"<tr><th align='left' valign='top'>{key}</th><td>{value!s}</td></tr>"
            for key, value in rows
        ) + "</table>"
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
