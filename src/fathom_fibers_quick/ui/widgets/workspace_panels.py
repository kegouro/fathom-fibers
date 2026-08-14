"""Workspace panels: dataset, overlays, summary, methods, measurements,
quality and manual 5x5 review.  Panels only render controller state."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...core.contracts import ScientificImage
from ...core.distributions import summarize_distribution
from ...core.methods import MethodId, MethodResult, MethodStatus
from ...measurement_records import MeasurementRecord
from ...unified_comparison import UnifiedMethodComparison
from ...validation.manual_review import GridCellStatus, Manual5x5Review
from ..plots import DistributionCanvas, ECDFCanvas, distribution_quantile_table
from ..workspace_controller import WorkspaceController

DISPLAY_NAMES = {
    MethodId.MATLAB_SIMPOLY: "MATLAB SIMPoly",
    MethodId.PYTHON_SIMPOLY: "Python SIMPoly",
    MethodId.FATHOM_LOCAL: "Fathom Local",
    MethodId.FATHOM_FIELD_GRAPH_V1: "Fathom Field",
    MethodId.MANUAL_5X5_REFERENCE: "Manual 5×5",
    MethodId.CONSENSUS_PSEUDO_REFERENCE_V1: "Consensus",
}

FIELD_ESTIMATOR_LABELS = {
    "FATHOM_FIELD_PAIRED_EDGE_DIAMETER": "Field Paired Edge",
    "FATHOM_FIELD_PROFILE_DIAMETER": "Field Intensity Profile",
}


def method_display_name(method_id: MethodId) -> str:
    return DISPLAY_NAMES.get(method_id, method_id.value)


class DatasetPanel(QWidget):
    imageRequested = Signal(int)
    openRequested = Signal()
    runRequested = Signal(str)
    exploreRequested = Signal()

    STATUS_TEXT = (
        ("complete", "complete"),
        ("summary", "summary cache"),
        ("pending", "not analyzed"),
        ("running", "running"),
        ("failed", "failed"),
        ("warning", "warning"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.dataset = None
        self.controller: WorkspaceController | None = None
        self.header_title = QLabel("")
        self.header_title.setProperty("role", "title")
        self.header_subtitle = QLabel("")
        self.header_subtitle.setProperty("role", "caption")
        self.header_subtitle.setWordWrap(True)
        self.cta_button = QPushButton("")
        self.cta_button.setProperty("role", "primary")
        header = QVBoxLayout()
        header.setSpacing(4)
        header.addWidget(self.header_title)
        header.addWidget(self.header_subtitle)
        header.addWidget(self.cta_button)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Image", "Status"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.open_button = QPushButton("Open Dataset…")
        self.run_all_button = QPushButton("Run all dataset")
        self.run_all_button.setToolTip(
            "Recompute the analysis for every dataset image, including cached ones."
        )
        self.run_missing_button = QPushButton("Run missing")
        self.run_missing_button.setToolTip(
            "Analyze only dataset images without a valid cached result."
        )
        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.run_all_button)
        buttons.addWidget(self.run_missing_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(header)
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.open_button.clicked.connect(self.openRequested)
        self.run_all_button.clicked.connect(lambda: self.runRequested.emit("all"))
        self.run_missing_button.clicked.connect(lambda: self.runRequested.emit("missing"))
        self.cta_button.clicked.connect(self._cta_clicked)

    def set_controller(self, controller: WorkspaceController) -> None:
        self.controller = controller
        self.refresh()

    def _current_changed(self, current, _previous) -> None:
        if current is not None and current.data(0, Qt.ItemDataRole.UserRole) is not None:
            self.imageRequested.emit(int(current.data(0, Qt.ItemDataRole.UserRole)))

    def _cta_clicked(self) -> None:
        if self.controller is None or self.controller.dataset is None:
            self.openRequested.emit()
            return
        if self._all_complete():
            self.exploreRequested.emit()
        else:
            self.runRequested.emit("missing")

    def _all_complete(self) -> bool:
        if self.controller is None or self.controller.dataset is None:
            return False
        return all(
            self.controller.cache.has_full(image.stem) for image in self.controller.dataset.images
        )

    def refresh(self) -> None:
        self.tree.clear()
        if self.controller is None or self.controller.dataset is None:
            self.header_title.setText("")
            self.header_subtitle.setText("")
            self.cta_button.setText("Open Dataset…")
            return
        dataset = self.controller.dataset
        complete = sum(1 for image in dataset.images if self.controller.cache.has_full(image.stem))
        self.header_title.setText(dataset.dataset_id)
        self.header_subtitle.setText(
            f"{len(dataset.images)} images · analysis available {complete} / {len(dataset.images)}"
        )
        if complete == len(dataset.images):
            self.cta_button.setText("Explore Results")
        else:
            self.cta_button.setText("Analyze Dataset")
        root = QTreeWidgetItem([dataset.dataset_id])
        self.tree.addTopLevelItem(root)
        for index, image in enumerate(dataset.images):
            status, color = self._status_for(index)
            item = QTreeWidgetItem([f"{image.filename}", status])
            item.setData(0, Qt.ItemDataRole.UserRole, index)
            item.setForeground(1, QColor(color))
            root.addChild(item)
        root.setExpanded(True)

    def _status_for(self, index: int) -> tuple[str, str]:
        controller = self.controller
        image = controller.dataset.images[index]
        stem = image.stem
        summary = controller.cache.summary_payload(stem)
        if summary and any(
            entry.get("status") == MethodStatus.FAILED.value for entry in summary.get("results", ())
        ):
            return "failed", "#d56b6b"
        manual = controller.manual
        manual_pending = False
        if manual is not None:
            review = manual.reviews.get(image.case_id)
            manual_pending = review is not None and review.measurement_count < 25
        if controller.cache.has_full(stem):
            text = "✓ complete"
            color = "#33b67a"
            if manual_pending:
                text += f" · manual {review.measurement_count}/25"
                color = "#f0a83a"
            return text, color
        if summary is not None:
            text = "summary cache"
            color = "#40bfe8"
            if manual_pending:
                text += f" · manual {review.measurement_count}/25"
            return text, color
        return "not analyzed", "#8a8f98"


class OverlayPanel(QWidget):
    overlayChanged = Signal(str, bool)
    densityChanged = Signal(str)

    LAYERS = (
        ("manual", "Manual measurements"),
        ("local_sections", "Fathom Local measurements"),
        ("skeleton", "Python SIMPoly skeleton"),
        ("mask", "Python SIMPoly mask"),
        ("centerline", "Field centerline"),
        ("raw_centerline", "Raw centerline"),
        ("refined_centerline", "Refined centerline"),
        ("midpoints", "Midpoint observations"),
        ("orientation", "Field orientation"),
        ("edges", "Raw paired-edge segments"),
        ("refined_edges", "Refined paired-edge segments"),
        ("profile", "Field profile-refined edges"),
        ("rejected", "Rejected / flagged samples"),
        ("rejected_refined", "Rejected refinement samples"),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.checks: dict[str, QCheckBox] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        title = QLabel("OVERLAYS")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        for key, label in self.LAYERS:
            check = QCheckBox(label)
            check.toggled.connect(lambda checked, name=key: self.overlayChanged.emit(name, checked))
            self.checks[key] = check
            layout.addWidget(check)
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("Orientation density"))
        self.density = QComboBox()
        self.density.addItems(["Sparse", "Medium", "Dense"])
        self.density.setCurrentText("Medium")
        self.density.currentTextChanged.connect(self.densityChanged)
        density_row.addWidget(self.density)
        density_row.addStretch(1)
        layout.addLayout(density_row)
        self.available_note = QLabel("")
        self.available_note.setWordWrap(True)
        layout.addWidget(self.available_note)
        layout.addStretch(1)

    def set_availability(self, results: dict[MethodId, MethodResult] | None) -> None:
        results = results or {}
        field = results.get(MethodId.FATHOM_FIELD_GRAPH_V1)
        python = results.get(MethodId.PYTHON_SIMPOLY)
        local = results.get(MethodId.FATHOM_LOCAL)
        availability = {
            "skeleton": python is not None and python.centerline is not None,
            "mask": python is not None and python.mask is not None,
            "centerline": field is not None,
            "raw_centerline": field is not None,
            "orientation": field is not None,
            "edges": field is not None,
            "profile": field is not None,
            "rejected": field is not None,
            "refined_centerline": field is not None and "refined_mask" in field.local_samples,
            "midpoints": field is not None and "refined_mask" in field.local_samples,
            "refined_edges": field is not None and "refined_mask" in field.local_samples,
            "rejected_refined": field is not None and "refined_mask" in field.local_samples,
            "local_sections": local is not None and local.local_samples is not None,
            "manual": True,
        }
        for key, available in availability.items():
            check = self.checks.get(key)
            if check is None:
                continue
            check.setEnabled(available)
            check.setVisible(available)
        missing = []
        if python is None or python.centerline is None:
            missing.append("SIMPoly")
        if local is None or local.local_samples is None:
            missing.append("Fathom Local")
        if field is None:
            missing.append("Fathom Field")
        self.available_note.setText(
            "Overlays require full method results for this image (Run Methods)." if missing else ""
        )


class SummaryPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.info = QTextBrowser()
        self.info.setMinimumHeight(140)
        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            ["Method", "Status", "N", "Coverage", "Mean", "Median", "IQR", "P05", "P95"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.table, 1)

    def set_image(self, image: ScientificImage | None) -> None:
        if image is None:
            self.info.setPlainText("No image loaded.")
            return
        calibration = image.calibration
        metadata = dict(image.metadata or {})
        body_height = image.footer_bounds[0] if image.footer_bounds else image.shape[0]
        lines = [
            f"<b>Image</b> {image.image_id}",
            (
                f"<b>Calibration</b> {calibration.pixel_size_x_m * 1e9:.5g} × "
                f"{calibration.pixel_size_y_m * 1e9:.5g} nm/px ({calibration.source})"
            ),
            (
                f"<b>Valid ROI</b> 0, 0 → {image.shape[1]}, {body_height} px"
                + (
                    f" (footer excluded from row {image.footer_bounds[0]})"
                    if image.footer_bounds
                    else ""
                )
            ),
        ]
        if metadata.get("ap_mag") is not None:
            lines.append(f"<b>Magnification</b> {metadata['ap_mag']}")
        if metadata.get("ap_actualkv") is not None:
            lines.append(f"<b>EHT</b> {metadata['ap_actualkv']} kV")
        lines.append(f"<b>Source</b> {image.source_path or 'array-backed'}")
        self.info.setHtml("<br>".join(lines))

    def set_comparison(self, comparison: UnifiedMethodComparison | None) -> None:
        rows: list[tuple[str, ...]] = []
        if comparison is not None:
            for result in comparison.results:
                if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
                    rows.extend(self._field_rows(result))
                else:
                    rows.append(self._result_row(result))
            consensus = comparison.consensus
            if consensus.distribution is not None:
                summary = summarize_distribution(consensus.distribution)
                rows.append(
                    (
                        "Consensus",
                        "COMPLETE",
                        str(summary.n),
                        "—",
                        _fmt(summary.weighted_mean),
                        _fmt(summary.weighted_median),
                        _fmt_range(summary.p25, summary.p75),
                        _fmt(summary.p05),
                        _fmt(summary.p95),
                    )
                )
        self.table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(QColor(_status_color(value)))
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()

    @classmethod
    def _field_rows(cls, result: MethodResult) -> list[tuple[str, ...]]:
        """Fathom Field family rows: raw and Ribbon estimators, grouped."""
        statistics = result.native_statistics
        smooth_coverage = statistics.get("smooth_coverage_fraction")
        raw_rows = [
            ("Fathom Field / Raw EDT", result.common_distribution, "—", result.status.value),
            (
                "Fathom Field / Raw Edge",
                result.secondary_distributions.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER"),
                _frac(statistics.get("edge_acceptance_fraction")),
                result.status.value,
            ),
            (
                "Fathom Field / Raw Profile",
                result.secondary_distributions.get("FATHOM_FIELD_PROFILE_DIAMETER"),
                _frac(statistics.get("profile_acceptance_fraction")),
                result.status.value,
            ),
        ]
        ribbon_rows = [
            (
                "Fathom Field / Ribbon EDT",
                result.secondary_distributions.get("FATHOM_FIELD_REFINED_EDT_DIAMETER"),
                _frac(smooth_coverage),
                "EXPERIMENTAL",
            ),
            (
                "Fathom Field / Ribbon Edge",
                result.secondary_distributions.get("FATHOM_FIELD_REFINED_EDGE_DIAMETER"),
                _frac(statistics.get("refined_edge_acceptance_fraction")),
                "EXPERIMENTAL",
            ),
            (
                "Fathom Field / Ribbon Profile",
                result.secondary_distributions.get("FATHOM_FIELD_REFINED_PROFILE_DIAMETER"),
                _frac(statistics.get("refined_profile_acceptance_fraction")),
                "EXPERIMENTAL",
            ),
        ]
        rows: list[tuple[str, ...]] = []
        for name, distribution, coverage, status in raw_rows + ribbon_rows:
            if distribution is None:
                rows.append((name, status, "—", coverage, "—", "—", "—", "—", "—"))
                continue
            summary = summarize_distribution(distribution)
            rows.append(
                (
                    name,
                    status,
                    str(summary.n),
                    coverage,
                    _fmt(summary.weighted_mean),
                    _fmt(summary.weighted_median),
                    _fmt_range(summary.p25, summary.p75),
                    _fmt(summary.p05),
                    _fmt(summary.p95),
                )
            )
        return rows

    @staticmethod
    def _result_row(result: MethodResult) -> tuple[str, ...]:
        distribution = result.common_distribution
        if result.method_id == MethodId.MANUAL_5X5_REFERENCE:
            distribution = result.native_distribution
        if distribution is None:
            if result.method_id == MethodId.MATLAB_SIMPOLY:
                value = _fmt(result.native_result) if result.native_result is not None else "—"
                return (
                    method_display_name(result.method_id),
                    result.status.value,
                    "—",
                    "—",
                    "Native b1: " + value,
                    "—",
                    "—",
                    "—",
                    "—",
                )
            return (
                method_display_name(result.method_id),
                result.status.value,
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
                "—",
            )
        summary = summarize_distribution(distribution)
        return (
            method_display_name(result.method_id),
            result.status.value,
            str(summary.n),
            "—",
            _fmt(summary.weighted_mean),
            _fmt(summary.weighted_median),
            _fmt_range(summary.p25, summary.p75),
            _fmt(summary.p05),
            _fmt(summary.p95),
        )


class MethodsPanel(QWidget):
    methodSelected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Method", "Status", "Capabilities", "Native estimand", "Runtime s", "Flags"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.details = QTextBrowser()
        self.details.setMinimumHeight(180)
        split = QVBoxLayout(self)
        split.addWidget(self.table, 1)
        split.addWidget(self.details, 1)
        self.table.itemSelectionChanged.connect(self._selection_changed)

    def set_comparison(self, comparison: UnifiedMethodComparison | None) -> None:
        rows: list[tuple[str, ...]] = []
        self._results: dict[str, MethodResult] = {}
        if comparison is not None:
            for result in comparison.results:
                self._results[result.method_id.value] = result
                capabilities = (
                    ", ".join(
                        capability.value.replace("_", " ")
                        for capability, state in result.capabilities.states.items()
                        if state.value == "AVAILABLE"
                    )
                    or "—"
                )
                rows.append(
                    (
                        method_display_name(result.method_id),
                        result.status.value,
                        capabilities,
                        result.native_estimand.value.replace("_", " ")
                        if result.native_estimand
                        else "—",
                        _fmt(result.runtime_seconds),
                        ", ".join(result.quality_flags) or "—",
                    )
                )
        self.table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(QColor(_status_color(value)))
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()
        if rows:
            self.table.selectRow(0)

    def _selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        method_id = self._results_keys()[row]
        self.methodSelected.emit(method_id)
        self._show_details(self._results.get(method_id))

    def _results_keys(self) -> list[str]:
        return list(self._results)

    def _show_details(self, result: MethodResult | None) -> None:
        if result is None:
            self.details.setPlainText("Select a method to inspect its details.")
            return
        provenance = dict(result.provenance)
        lines = [
            f"<b>Method</b> {method_display_name(result.method_id)}",
            f"<b>Version</b> {result.method_version}",
            f"<b>Status</b> {result.status.value}",
            f"<b>Native estimand</b> {result.native_estimand.value if result.native_estimand else '—'}",
            f"<b>Native result</b> {_fmt(result.native_result)}",
            f"<b>Input ROI</b> {result.valid_roi}",
            f"<b>Unit</b> {result.unit}",
            f"<b>Runtime</b> {_fmt(result.runtime_seconds)} s",
            f"<b>Confidence</b> {_fmt(result.confidence, 4)}",
        ]
        if result.quality_flags:
            lines.append("<b>Flags</b> " + ", ".join(result.quality_flags))
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
            lines.append(
                "<p><b>Known limitations:</b> experimental field-measuring stage. "
                "Graph reconstruction, fiber instances and ML are not implemented.</p>"
            )
        if result.method_id == MethodId.PYTHON_SIMPOLY:
            lines.append("<p><b>Known limitation:</b> KNOWN_LIBRARY_DIVERGENCE — bwskel.</p>")
        if provenance:
            details = "<br>".join(f"{key}: {value}" for key, value in sorted(provenance.items()))
            lines.append(f"<p><b>Provenance</b><br><pre>{details}</pre></p>")
        self.details.setHtml("<br>".join(lines))


class FieldSamplesModel(QAbstractTableModel):
    HEADERS = (
        "ID",
        "x (µm)",
        "y (µm)",
        "d_EDT (nm)",
        "d_edge (nm)",
        "d_profile (nm)",
        "r− (nm)",
        "r+ (nm)",
        "asymmetry",
        "coherence",
        "profile conf.",
        "arc weight (µm)",
        "status",
        "flags",
        "refined EDT (nm)",
        "refined Edge (nm)",
        "refined Profile (nm)",
        "residual shift",
        "refinement",
        "segment",
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.samples: dict[str, np.ndarray] = {}
        self.n = 0

    def set_samples(self, samples: dict[str, np.ndarray] | None) -> None:
        self.beginResetModel()
        self.samples = dict(samples or {})
        first = next(iter(self.samples.values()), None)
        self.n = int(first.size) if first is not None else 0
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else self.n

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def get(self, key: str) -> np.ndarray:
        value = self.samples.get(key)
        if value is None:
            return np.array([], float)
        array = np.asarray(value)
        if array.size == self.n and array.ndim == 2 and array.shape[0] == self.n:
            return array.reshape(-1)
        return array

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not 0 <= index.row() < self.n:
            return None
        if role == Qt.ItemDataRole.UserRole:
            return index.row()
        if role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return None
        row = index.row()
        column = index.column()
        if column == 0:
            return str(row)
        if column == 1:
            return _fmt(self.get("x_m")[row] * 1e6)
        if column == 2:
            return _fmt(self.get("y_m")[row] * 1e6)
        if column == 3:
            return _fmt(self.get("diameter_um")[row] * 1000.0)
        if column == 4:
            return _fmt(self.get("edge_diameter_um")[row] * 1000.0)
        if column == 5:
            return _fmt(self.get("profile_diameter_um")[row] * 1000.0)
        if column == 6:
            return _fmt(self.get("radius_minus_um")[row] * 1000.0)
        if column == 7:
            return _fmt(self.get("radius_plus_um")[row] * 1000.0)
        if column == 8:
            return _fmt(self.get("edge_asymmetry")[row], 4)
        if column == 9:
            return _fmt(self.get("coherence")[row], 4)
        if column == 10:
            snr = self.get("profile_gradient_snr")
            return _fmt(snr[row], 4) if snr.size else "—"
        if column == 11:
            return _fmt(self.get("arc_length_weight_m")[row] * 1e6)
        if column == 12:
            accepted = self.get("edge_accepted")
            return "accepted" if accepted[row] else "rejected" if accepted.size else "—"
        if column == 13:
            flags = self.get("edge_flags")
            return flags[row] if flags.size else "—"
        if column == 14:
            return _fmt(self.get("refined_edt_um")[row] * 1000.0)
        if column == 15:
            return _fmt(self.get("refined_edge_um")[row] * 1000.0)
        if column == 16:
            return _fmt(self.get("refined_profile_um")[row] * 1000.0)
        if column == 17:
            return _fmt(self.get("residual_center_shift_um")[row])
        if column == 18:
            refined = self.get("refined_mask")
            return "refined" if refined.size and refined[row] else "—"
        if column == 19:
            segment = self.get("segment_id")
            return str(segment[row]) if segment.size and segment[row] >= 0 else "—"
        return None


class FieldSamplesFilter(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.mode = "all"
        self.min_coherence = 0.0
        self.setDynamicSortFilter(True)

    def set_filters(self, mode: str, min_coherence: float) -> None:
        self.mode = mode
        self.min_coherence = min_coherence
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model: FieldSamplesModel = self.sourceModel()
        if not model.n:
            return False
        accepted = model.get("edge_accepted")
        coherence = model.get("coherence")
        flags = model.get("edge_flags")
        if (
            self.min_coherence > 0
            and coherence.size
            and (
                not np.isfinite(coherence[source_row]) or coherence[source_row] < self.min_coherence
            )
        ):
            return False
        if self.mode == "accepted":
            return bool(accepted.size and accepted[source_row])
        if self.mode == "rejected":
            return bool(accepted.size and not accepted[source_row])
        if self.mode == "flagged":
            return bool(flags.size and str(flags[source_row]).strip())
        return True


class LocalSectionsModel(QAbstractTableModel):
    HEADERS = ("ID", "x (µm)", "y (µm)", "width (nm)", "flags")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.samples: dict[str, np.ndarray] = {}

    def set_samples(self, samples: dict[str, np.ndarray] | None) -> None:
        self.beginResetModel()
        self.samples = dict(samples or {})
        self.endResetModel()

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return int(self.samples.get("section_width_um", np.array([])).size)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None
        if role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return None
        row = index.row()
        if index.column() == 0:
            return str(row)
        if index.column() == 1:
            return _fmt(float(np.asarray(self.samples["section_x0_px"])[row]))
        if index.column() == 2:
            return _fmt(float(np.asarray(self.samples["section_y0_px"])[row]))
        if index.column() == 3:
            return _fmt(float(np.asarray(self.samples["section_width_um"])[row]) * 1000.0)
        if index.column() == 4:
            return str(np.asarray(self.samples["section_flags"])[row])
        return None


class MeasurementsPanel(QWidget):
    fieldSampleSelected = Signal(int)
    recordSelected = Signal(object)

    def __init__(self, session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self.source = QComboBox()
        self.source.addItems(["Fathom Field samples", "Fathom Local sections", "Manual records"])
        self.mode = QComboBox()
        self.mode.addItems(["All", "Accepted only", "Rejected only", "Flagged only"])
        self.min_coherence = QDoubleSpinBox()
        self.min_coherence.setRange(0.0, 1.0)
        self.min_coherence.setSingleStep(0.05)
        self.min_coherence.setValue(0.0)
        self.min_coherence.setDecimals(2)
        filters = QHBoxLayout()
        filters.addWidget(QLabel("Source"))
        filters.addWidget(self.source)
        filters.addWidget(QLabel("Filter"))
        filters.addWidget(self.mode)
        filters.addWidget(QLabel("Min coherence"))
        filters.addWidget(self.min_coherence)
        filters.addStretch(1)

        self.field_model = FieldSamplesModel(self)
        self.field_proxy = FieldSamplesFilter(self)
        self.field_proxy.setSourceModel(self.field_model)
        self.local_model = LocalSectionsModel(self)
        self.table = QTableView()
        self.table.setModel(self.field_proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.info = QLabel("")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addLayout(filters)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.info)
        self.source.currentTextChanged.connect(self._source_changed)
        self.mode.currentTextChanged.connect(self._filters_changed)
        self.min_coherence.valueChanged.connect(self._filters_changed)
        self.table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._focus_selected)

    def set_comparison(self, comparison: UnifiedMethodComparison | None) -> None:
        field = None
        local = None
        if comparison is not None:
            for result in comparison.results:
                if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
                    field = result.local_samples
                elif result.method_id == MethodId.FATHOM_LOCAL:
                    local = result.local_samples
        self.field_model.set_samples(field)
        self.local_model.set_samples(local)
        if self.source.currentText() == "Fathom Field samples":
            self.table.setModel(self.field_proxy)
            self.mode.setEnabled(True)
            self.min_coherence.setEnabled(True)
        elif self.source.currentText() == "Fathom Local sections":
            self.table.setModel(self.local_model)
            self.mode.setEnabled(False)
            self.min_coherence.setEnabled(False)
        else:
            from ..models import MeasurementTableModel

            self._manual_model = MeasurementTableModel(self.session)
            self.table.setModel(self._manual_model)
            self.mode.setEnabled(False)
            self.min_coherence.setEnabled(False)
        self._update_info()

    def _source_changed(self, *_args) -> None:
        if self.source.currentText() == "Fathom Field samples":
            self.table.setModel(self.field_proxy)
            self.mode.setEnabled(True)
            self.min_coherence.setEnabled(True)
        elif self.source.currentText() == "Fathom Local sections":
            self.table.setModel(self.local_model)
            self.mode.setEnabled(False)
            self.min_coherence.setEnabled(False)
        else:
            from ..models import MeasurementTableModel

            self._manual_model = MeasurementTableModel(self.session)
            self.table.setModel(self._manual_model)
            self.mode.setEnabled(False)
            self.min_coherence.setEnabled(False)
        self._update_info()

    def _filters_changed(self, *_args) -> None:
        self.field_proxy.set_filters(self.mode.currentText(), self.min_coherence.value())

    def _update_info(self) -> None:
        source = self.source.currentText()
        if source == "Fathom Field samples":
            n = self.field_model.n
            accepted = int(np.sum(self.field_model.get("edge_accepted"))) if n else 0
            self.info.setText(
                f"{n} local samples · {accepted} paired-edge accepted"
                if n
                else "Fathom Field not computed for this image (Run Methods)."
            )
        elif source == "Fathom Local sections":
            n = self.local_model.rowCount()
            self.info.setText(
                f"{n} Fathom Local cross-sections"
                if n
                else "Fathom Local not computed for this image (Run Methods)."
            )
        else:
            self.info.setText("Manual 5×5 measurements on the current image.")

    def _selection_changed(self, *_args) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        proxy_index = rows[0]
        if self.table.model() is self.field_proxy:
            source_index = self.field_proxy.mapToSource(proxy_index)
            self.fieldSampleSelected.emit(source_index.row())
        else:
            model = self.table.model()
            if hasattr(model, "record_at"):
                record = model.record_at(proxy_index.row())
                if record is not None:
                    self.recordSelected.emit(record.measurement_id)

    def select_field_row(self, index: int) -> None:
        if self.table.model() is not self.field_proxy:
            return
        source_index = self.field_model.index(index, 0)
        proxy_index = self.field_proxy.mapFromSource(source_index)
        if proxy_index.isValid():
            self.table.selectRow(proxy_index.row())
            self.table.scrollTo(proxy_index)

    def _focus_selected(self, *_args) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        if self.table.model() is self.field_proxy:
            source_index = self.field_proxy.mapToSource(rows[0])
            self.fieldSampleSelected.emit(source_index.row())


class QualityPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Method", "Raw N", "Accepted N", "Acceptance", "Coherence", "Flags"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ribbon_table = QTableWidget(0, 2)
        self.ribbon_table.setHorizontalHeaderLabels(["Oriented Ribbon V1", "Value"])
        self.ribbon_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.ribbon_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.ribbon_table.setVisible(False)
        self.flags = QTextBrowser()
        self.flags.setMinimumHeight(160)
        layout = QVBoxLayout(self)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.ribbon_table)
        layout.addWidget(self.flags, 1)

    def set_comparison(self, comparison: UnifiedMethodComparison | None) -> None:
        rows: list[tuple[str, ...]] = []
        flag_sections = ""
        if comparison is not None:
            for result in comparison.results:
                rows.append(self._result_row(result))
                flag_sections += self._flag_section(result)
        self.table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setForeground(QColor(_status_color(value)))
                self.table.setItem(row_index, column, item)
        self.table.resizeColumnsToContents()
        self._set_ribbon_rows(comparison)
        self.flags.setHtml(flag_sections or "<p>No flags recorded.</p>")

    def _set_ribbon_rows(self, comparison: UnifiedMethodComparison | None) -> None:
        field = None
        if comparison is not None:
            field = next(
                (
                    result
                    for result in comparison.results
                    if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
                ),
                None,
            )
        if field is None:
            self.ribbon_table.setVisible(False)
            self.ribbon_table.setRowCount(0)
            return
        statistics = field.native_statistics

        def row(label: str, value: str) -> None:
            rows.append((label, value))

        rows: list[tuple[str, str]] = []
        row("Observation coverage", _frac(statistics.get("refine_coverage_fraction")))
        row(
            "Supported centerline coverage",
            _frac(statistics.get("smooth_coverage_fraction")),
        )
        row(
            "Original center shift · median", _fmt(statistics.get("refine_median_shift_um")) + " µm"
        )
        row("Original center shift · P90", _fmt(statistics.get("refine_p90_shift_um")) + " µm")
        row(
            "Residual center shift · median",
            _fmt(statistics.get("refined_residual_shift_median_um")) + " µm",
        )
        row(
            "Residual center shift · P90",
            _fmt(statistics.get("refined_residual_shift_p90_um")) + " µm",
        )
        row(
            "Asymmetry raw → refined",
            _fmt(statistics.get("edge_median_asymmetry"), 4)
            + " → "
            + _fmt(statistics.get("refined_asymmetry_median"), 4),
        )
        row(
            "Edge acceptance raw → refined",
            _frac(statistics.get("edge_acceptance_fraction"))
            + " → "
            + _frac(statistics.get("refined_edge_acceptance_fraction")),
        )
        row(
            "Profile acceptance raw → refined",
            _frac(statistics.get("profile_acceptance_fraction"))
            + " → "
            + _frac(statistics.get("refined_profile_acceptance_fraction")),
        )
        row("Refinement segments", str(statistics.get("smooth_segment_count", "—")))
        ribbon_flags = [
            flag
            for flag in field.quality_flags
            if flag
            not in {
                "EXPERIMENTAL_FIELD_MEASURING",
                "FIELD_STAGE_IMPLEMENTED",
                "GRAPH_STAGE_NOT_IMPLEMENTED",
                "SMOOTH_CENTERLINE_V1",
                "REFINED_REMEASUREMENT",
            }
        ]
        row("Flags", ", ".join(ribbon_flags) or "—")
        self.ribbon_table.setRowCount(len(rows))
        for row_index, (label, value) in enumerate(rows):
            self.ribbon_table.setItem(row_index, 0, QTableWidgetItem(label))
            self.ribbon_table.setItem(row_index, 1, QTableWidgetItem(value))
        self.ribbon_table.setVisible(True)

    @staticmethod
    def _result_row(result: MethodResult) -> tuple[str, ...]:
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
            statistics = result.native_statistics
            samples = result.local_samples
            raw = statistics.get(
                "edge_raw_count", len(samples.get("edge_diameter_um", [])) if samples else 0
            )
            accepted = statistics.get("edge_accepted_count", 0)
            acceptance = statistics.get("edge_acceptance_fraction")
            coherence = statistics.get("mean_coherence")
            return (
                method_display_name(result.method_id),
                str(raw),
                str(accepted),
                "—" if acceptance is None else f"{acceptance:.2%}",
                "—" if coherence is None else f"{coherence:.4g}",
                ", ".join(
                    flag for flag in result.quality_flags if flag != "EXPERIMENTAL_FIELD_MEASURING"
                )
                or "—",
            )
        distribution = result.common_distribution
        if result.method_id == MethodId.MANUAL_5X5_REFERENCE:
            distribution = result.native_distribution
        n = distribution.diameter.size if distribution is not None else 0
        return (
            method_display_name(result.method_id),
            str(n),
            "—",
            "—",
            "—",
            ", ".join(result.quality_flags) or "—",
        )

    @staticmethod
    def _flag_section(result: MethodResult) -> str:
        counts: dict[str, int] = {}
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1 and result.local_samples:
            flags = result.local_samples.get("edge_flags")
            if flags is not None and flags.size:
                for value in flags:
                    for flag in str(value).split(";"):
                        if flag:
                            counts[flag] = counts.get(flag, 0) + 1
        for flag in result.quality_flags:
            counts[flag] = counts.get(flag, 0) + 1
        if not counts:
            return ""
        rows = "".join(
            f"<tr><td>{flag}</td><td>{count}</td><td>{count / max(1, sum(counts.values())):.1%}</td></tr>"
            for flag, count in sorted(counts.items())
        )
        return (
            f"<h3>{method_display_name(result.method_id)}</h3>"
            f"<table><tr><th>Flag</th><th>Count</th><th>Share</th></tr>{rows}</table>"
        )


class Manual5x5Panel(QWidget):
    targetRequested = Signal(int, int)
    removeRequested = Signal(int, int)
    skipRequested = Signal(int, int)
    nextImageRequested = Signal()
    acceptNextRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.review: Manual5x5Review | None = None
        self.case_id: str | None = None
        self.active_cell: tuple[int, int] | None = None
        self.position = QLabel("No dataset loaded")
        self.progress = QLabel("")
        self.feedback_label = QLabel("")
        self.feedback_label.setProperty("role", "success")
        self._feedback_timer = QTimer(self)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.setInterval(2200)
        self._feedback_timer.timeout.connect(lambda: self.feedback_label.setText(""))
        self.grid = QTableWidget(5, 5)
        self.grid.setHorizontalHeaderLabels([str(index) for index in range(1, 6)])
        self.grid.setVerticalHeaderLabels([str(index) for index in range(1, 6)])
        self.grid.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.grid.cellClicked.connect(self._cell_clicked)
        header = self.grid.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.grid.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.grid.setMinimumHeight(240)

        self.next_button = QPushButton("Next target (Enter)")
        self.previous_button = QPushButton("Previous target (Backspace)")
        self.remove_button = QPushButton("Remove measurement (Delete)")
        self.skip_button = QPushButton("Skip with reason…")
        self.finish_button = QPushButton("Next image")
        buttons = QHBoxLayout()
        for button in (
            self.next_button,
            self.previous_button,
            self.remove_button,
            self.skip_button,
            self.finish_button,
        ):
            buttons.addWidget(button)
        buttons.addStretch(1)

        help_text = QLabel(
            "Draw a perpendicular width line with the Projected width tool (M). "
            "The measurement is accepted and autosaved immediately. "
            "Enter: next target · Backspace: previous · Delete: remove · Esc: cancel."
        )
        help_text.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.position)
        layout.addWidget(self.progress)
        layout.addWidget(self.feedback_label)
        layout.addWidget(self.grid, 1)
        layout.addLayout(buttons)
        layout.addWidget(help_text)

        self.next_button.clicked.connect(self.next_target)
        self.previous_button.clicked.connect(self.previous_target)
        self.remove_button.clicked.connect(self._remove_active)
        self.skip_button.clicked.connect(self._skip_active)
        self.finish_button.clicked.connect(self.nextImageRequested)

    def set_review(self, review: Manual5x5Review | None, case_id: str | None = None) -> None:
        self.review = review
        self.case_id = case_id if review is not None else None
        self.active_cell = None
        self._refresh_grid()
        if review is not None:
            self.next_target()

    def set_progress(self, image_index: int, image_count: int, dataset_total: int) -> None:
        if self.review is None:
            return
        self.position.setText(f"Image {image_index} / {image_count} — {self.case_id or ''}")
        self.progress.setText(
            f"Point {self.review.measurement_count} / 25 · dataset {dataset_total} / 400"
        )

    def _refresh_grid(self) -> None:
        for row in range(5):
            for column in range(5):
                if self.review is None:
                    self.grid.setItem(row, column, QTableWidgetItem("—"))
                    continue
                cell = self.review.cell(row, column)
                status = cell.status.value.replace("NOT_REVIEWED", "—")
                item = QTableWidgetItem(status)
                color = {
                    GridCellStatus.MEASURED: "#33b67a",
                    GridCellStatus.NO_VALID_FIBER: "#8a8f98",
                    GridCellStatus.SKIPPED_WITH_REASON: "#f0a83a",
                    GridCellStatus.NOT_REVIEWED: "#d9dde2",
                }[cell.status]
                item.setForeground(QColor(color))
                self.grid.setItem(row, column, item)
                if (row, column) == self.active_cell:
                    item.setBackground(QColor("#2b3a4a"))
                else:
                    item.setBackground(QColor())

    def set_active(self, row: int, column: int) -> None:
        self.active_cell = (row, column)
        self.grid.setCurrentCell(row, column)
        self._refresh_grid()
        self.targetRequested.emit(row, column)

    def next_target(self) -> None:
        if self.review is None:
            return
        for row in range(5):
            for column in range(5):
                if self.review.cell(row, column).status == GridCellStatus.NOT_REVIEWED:
                    self.set_active(row, column)
                    return
        if self.review.measurement_count >= 25:
            self.position.setText(f"{self.position.text()} — grid complete")
            self.nextImageRequested.emit()

    def previous_target(self) -> None:
        if self.review is None or self.active_cell is None:
            return
        row, column = self.active_cell
        for delta in range(1, 26):
            index = (row * 5 + column - delta) % 25
            self.set_active(index // 5, index % 5)
            return

    def _remove_active(self) -> None:
        if self.active_cell is None:
            return
        self.removeRequested.emit(*self.active_cell)

    def _skip_active(self) -> None:
        if self.active_cell is None:
            return
        self.skipRequested.emit(*self.active_cell)

    def _cell_clicked(self, row: int, column: int) -> None:
        self.set_active(row, column)

    def active_grid_position(self) -> tuple[int, int] | None:
        return self.active_cell

    def current_case_id(self) -> str | None:
        return self.case_id

    def accept_and_advance(self) -> None:
        if self.review is None:
            return
        if self.review.measurement_count >= 25:
            self.nextImageRequested.emit()
        else:
            self.next_target()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.accept_and_advance()
            event.accept()
            return
        if key == Qt.Key.Key_Backspace:
            self.previous_target()
            event.accept()
            return
        if key == Qt.Key.Key_Delete:
            self._remove_active()
            event.accept()
            return
        super().keyPressEvent(event)

    def record_measurement(self, record: MeasurementRecord) -> None:
        if self.review is None or self.active_cell is None:
            return
        self._refresh_grid()
        self.set_progress(
            int(getattr(self, "_image_index", 1)),
            int(getattr(self, "_image_count", 1)),
            int(getattr(self, "_dataset_total", 0)),
        )

    def flash_feedback(self, text: str) -> None:
        self.feedback_label.setText(text)
        self._feedback_timer.start()

    def set_image_index(self, index: int, count: int, dataset_total: int) -> None:
        self._image_index = index
        self._image_count = count
        self._dataset_total = dataset_total
        if self.review is not None:
            self.set_progress(index, count, dataset_total)


class WorkspaceInspector(QWidget):
    """Right-dock inspector reacting to the current workspace selection."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.tabs = QTabWidget()
        self.selection = QTextBrowser()
        self.measurements = QTextBrowser()
        self.quality = QTextBrowser()
        self.provenance = QTextBrowser()
        self.tabs.addTab(self.selection, "Selection")
        self.tabs.addTab(self.measurements, "Measurements")
        self.tabs.addTab(self.quality, "Quality")
        self.tabs.addTab(self.provenance, "Provenance")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.tabs)
        self.clear()

    def clear(self) -> None:
        for browser in (self.selection, self.measurements, self.quality, self.provenance):
            browser.setPlainText("No selection.")
        self.tabs.setCurrentIndex(0)

    def set_field_sample(self, samples: dict[str, np.ndarray] | None, index: int | None) -> None:
        if (
            samples is None
            or index is None
            or not 0 <= index < int(next(iter(samples.values())).size if samples else 0)
        ):
            self.clear()
            return

        def value(key: str) -> Any:
            array = samples.get(key)
            if array is None or not np.asarray(array).size:
                return None
            return float(np.asarray(array).reshape(-1)[index])

        def value_str(key: str) -> str:
            array = samples.get(key)
            if array is None or not np.asarray(array).size:
                return "—"
            return str(np.asarray(array).reshape(-1)[index])

        x_m, y_m = value("x_m"), value("y_m")
        coherence = value("coherence")
        edge_accepted = value("edge_accepted")
        edge_flags = value_str("edge_flags")
        profile_flags = value_str("profile_flags")
        qx, qy = value("qx"), value("qy")
        theta = 0.5 * np.arctan2(qy, qx) if qx is not None and qy is not None else None
        refined_mask = value("refined_mask")
        refined_x, refined_y = value("refined_x_m"), value("refined_y_m")
        segment_id = value("segment_id")
        observed_shift = value("center_shift_um")
        residual_shift = value("residual_center_shift_um")
        residual_normal = value("residual_normal_shift_um")
        residual_tangent = value("residual_tangential_shift_um")
        refine_confidence = value("refine_confidence")
        axis_disagreement = value("refined_axis_disagreement_deg")
        refined_edge_accepted = value("refined_edge_accepted")
        refined_flags = value_str("refined_edge_flags")

        is_refined = bool(refined_mask) if refined_mask is not None else False
        refinement_rows = [
            ("Status", "Refined" if is_refined else "Not refined"),
            (
                "Segment ID",
                str(int(segment_id)) if segment_id is not None and segment_id >= 0 else "—",
            ),
            (
                "Original center",
                f"x {x_m * 1e6:.4g} µm · y {y_m * 1e6:.4g} µm" if x_m is not None else "—",
            ),
            (
                "Refined center",
                f"x {refined_x * 1e6:.4g} µm · y {refined_y * 1e6:.4g} µm"
                if is_refined and refined_x is not None
                else "—",
            ),
            ("Observed shift", f"{_fmt(observed_shift)} µm"),
            ("Residual shift", f"{_fmt(residual_shift)} µm"),
            ("Residual normal", f"{_fmt(residual_normal)} µm"),
            ("Residual tangential", f"{_fmt(residual_tangent)} µm"),
            (
                "Shift / local diameter",
                _fmt(observed_shift / value("edge_diameter_um"), 4)
                if observed_shift is not None and value("edge_diameter_um")
                else "—",
            ),
            ("Refinement confidence", _fmt(refine_confidence, 4)),
            ("Orientation disagreement", f"{_fmt(axis_disagreement, 4)}°"),
            (
                "Refined edge status",
                "accepted"
                if refined_edge_accepted
                else "rejected"
                if refined_edge_accepted is not None
                else "—",
            ),
            ("Refined flags", refined_flags.replace(";", " · ") or "—"),
        ]

        rows = [
            (
                "Position",
                f"x {x_m * 1e6:.4g} µm · y {y_m * 1e6:.4g} µm" if x_m is not None else "—",
            ),
            ("Orientation θ", f"{np.degrees(theta):.3g}°" if theta is not None else "—"),
            ("Coherence", _fmt(coherence, 4)),
            ("d_EDT", f"{_fmt(value('diameter_um'))} µm"),
            ("d_edge", f"{_fmt(value('edge_diameter_um'))} µm"),
            ("d_profile", f"{_fmt(value('profile_diameter_um'))} µm"),
            ("r−", f"{_fmt(value('radius_minus_um'))} µm"),
            ("r+", f"{_fmt(value('radius_plus_um'))} µm"),
            ("Asymmetry", _fmt(value("edge_asymmetry"), 4)),
            ("Arc-length weight", f"{_fmt(value('arc_length_weight_m') * 1e6)} µm"),
            (
                "Paired-edge status",
                "accepted" if edge_accepted else "rejected" if edge_accepted is not None else "—",
            ),
            ("Edge flags", edge_flags.replace(";", " · ") or "—"),
            ("Profile flags", profile_flags.replace(";", " · ") or "—"),
            ("Profile confidence", _fmt(value("profile_gradient_snr"), 4)),
            ("Edge shift minus", f"{_fmt(value('profile_minus_shift_um'))} µm"),
            ("Edge shift plus", f"{_fmt(value('profile_plus_shift_um'))} µm"),
            ("Suggested center shift", f"{_fmt(value('suggested_center_shift_um'))} µm"),
        ]
        self.selection.setHtml(
            _table(rows)
            + "<h3 style='font-size:12px;margin:.2em 0'>CENTERLINE REFINEMENT</h3>"
            + _table(refinement_rows)
            + "<h3 style='font-size:12px;margin:.2em 0'>RAW VS REFINED WIDTHS</h3>"
            + "<table><tr><th></th><th>Raw</th><th>Refined</th></tr>"
            + "<tr><td>EDT</td><td>"
            + _fmt(value("diameter_um"))
            + " µm</td><td>"
            + _fmt(value("refined_edt_um"))
            + " µm</td></tr>"
            + "<tr><td>Edge</td><td>"
            + _fmt(value("edge_diameter_um"))
            + " µm</td><td>"
            + _fmt(value("refined_edge_um"))
            + " µm</td></tr>"
            + "<tr><td>Profile</td><td>"
            + _fmt(value("profile_diameter_um"))
            + " µm</td><td>"
            + _fmt(value("refined_profile_um"))
            + " µm</td></tr>"
            + "<tr><td>r−</td><td>"
            + _fmt(value("radius_minus_um"))
            + " µm</td><td>"
            + _fmt(value("refined_r_minus_um"))
            + " µm</td></tr>"
            + "<tr><td>r+</td><td>"
            + _fmt(value("radius_plus_um"))
            + " µm</td><td>"
            + _fmt(value("refined_r_plus_um"))
            + " µm</td></tr>"
            + "<tr><td>Asymmetry</td><td>"
            + _fmt(value("edge_asymmetry"), 4)
            + "</td><td>"
            + _fmt(value("refined_asymmetry"), 4)
            + "</td></tr></table>"
        )
        measurement_rows = [
            ("d_EDT", _fmt(value("diameter_um")), "µm"),
            ("d_edge (paired)", _fmt(value("edge_diameter_um")), "µm"),
            ("d_profile (refined)", _fmt(value("profile_diameter_um")), "µm"),
            ("d_min from edges", _fmt(value("d_min_from_edges_um")), "µm"),
            ("edge − EDT", _fmt(value("edge_minus_edt_um")), "µm"),
            ("edt − dmin", _fmt(value("edt_minus_dmin_um")), "µm"),
        ]
        self.measurements.setHtml(
            "<table><tr><th>Quantity</th><th>Value</th><th>Unit</th></tr>"
            + "".join(
                f"<tr><td>{name}</td><td>{value}</td><td>{unit}</td></tr>"
                for name, value, unit in measurement_rows
            )
            + "</table>"
        )
        self.quality.setHtml(
            _table(
                [
                    ("Coherence", _fmt(coherence, 4)),
                    (
                        "Paired-edge accepted",
                        "yes" if edge_accepted else "no" if edge_accepted is not None else "—",
                    ),
                    ("Edge flags", edge_flags.replace(";", " · ") or "—"),
                    ("Profile flags", profile_flags.replace(";", " · ") or "—"),
                    ("Profile confidence", _fmt(value("profile_gradient_snr"), 4)),
                ]
            )
        )

    def set_result_provenance(self, result: MethodResult | None) -> None:
        if result is None:
            self.provenance.setPlainText("Select a method to inspect provenance.")
            return
        rows = [
            ("Method", method_display_name(result.method_id)),
            ("Version", result.method_version),
            ("Status", result.status.value),
            ("Native estimand", result.native_estimand.value if result.native_estimand else "—"),
            ("ROI", str(result.valid_roi)),
            ("Unit", result.unit),
            ("Runtime s", _fmt(result.runtime_seconds)),
        ]
        rows += [
            (key, _fmt(value, 6) if isinstance(value, float) else str(value))
            for key, value in sorted(result.provenance.items())
        ]
        self.provenance.setHtml(_table(rows))


def _table(rows: list[tuple[str, Any]]) -> str:
    return (
        "<table>"
        + "".join(
            f"<tr><th align='left' valign='top'>{key}</th><td>{value}</td></tr>"
            for key, value in rows
        )
        + "</table>"
    )


def _fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _frac(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"{low:.5g}–{high:.5g}"


def _status_color(status: str) -> str:
    if status in {"COMPLETE", "EXPERIMENTAL_FIELD_MEASURING"}:
        return "#2f9e63"
    if status == "FAILED":
        return "#d56b6b"
    if status == "NOT_MEASURED":
        return "#8a8f98"
    return "#b0a030"


class DistributionsPanel(QWidget):
    """Interactive histogram + ECDF + pairwise agreement view."""

    SERIES_CHOICES = (
        "All",
        "Python SIMPoly",
        "Fathom Local",
        "Fathom Field (EDT)",
        "Field Paired Edge",
        "Field Intensity Profile",
        "Ribbon Refined EDT",
        "Ribbon Refined Edge",
        "Ribbon Refined Profile",
        "Manual 5×5",
        "Consensus",
    )

    PRESETS: ClassVar[dict[str, frozenset[str]]] = {
        "ALL METHODS": frozenset(),
        "FIELD RAW": frozenset(
            {"Fathom Field (EDT)", "Field Paired Edge", "Field Intensity Profile"}
        ),
        "FIELD RIBBON": frozenset(
            {"Ribbon Refined EDT", "Ribbon Refined Edge", "Ribbon Refined Profile"}
        ),
        "RAW vs REFINED EDT": frozenset({"Fathom Field (EDT)", "Ribbon Refined EDT"}),
        "RAW vs REFINED EDGE": frozenset({"Field Paired Edge", "Ribbon Refined Edge"}),
        "RAW vs REFINED PROFILE": frozenset({"Field Intensity Profile", "Ribbon Refined Profile"}),
        "MANUAL COMPARISON": frozenset(
            {
                "Manual 5×5",
                "Python SIMPoly",
                "Fathom Local",
                "Ribbon Refined EDT",
                "Ribbon Refined Edge",
                "Ribbon Refined Profile",
            }
        ),
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.comparison: UnifiedMethodComparison | None = None
        self.series_combo = QComboBox()
        self.series_combo.addItems(self.SERIES_CHOICES)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Series"))
        controls.addWidget(self.series_combo)
        controls.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(self.PRESETS))
        self.preset_combo.setCurrentText("ALL METHODS")
        controls.addWidget(self.preset_combo)
        self.matlab_note = QLabel("")
        self.matlab_note.setStyleSheet("color: #8a6d1a;")
        self.matlab_note.setWordWrap(True)
        controls.addWidget(self.matlab_note, 1)

        plots = QHBoxLayout()
        self.histogram = DistributionCanvas(self)
        self.ecdf = ECDFCanvas(self)
        plots.addWidget(self.histogram, 1)
        plots.addWidget(self.ecdf, 1)

        self.summary_table = QTableWidget(0, 7)
        self.summary_table.setHorizontalHeaderLabels(
            ["Series", "N", "Mean", "Median", "IQR", "P05", "P95"]
        )
        self.summary_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.summary_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )

        self.agreement_table = QTableWidget(0, 5)
        self.agreement_table.setHorizontalHeaderLabels(
            ["A", "B", "Wasserstein-1", "KS", "Median Δ"]
        )
        self.agreement_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.agreement_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )

        tabs = QTabWidget()
        tabs.addTab(self.summary_table, "Distribution summary")
        tabs.addTab(self.agreement_table, "Pairwise distances")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.addLayout(controls)
        layout.addLayout(plots, 1)
        layout.addWidget(tabs, 1)
        self.series_combo.currentTextChanged.connect(self._refresh_plots)
        self.preset_combo.currentTextChanged.connect(self._preset_changed)

    def set_comparison(self, comparison: UnifiedMethodComparison | None) -> None:
        self.comparison = comparison
        matlab = None
        if comparison is not None:
            matlab = next(
                (
                    result
                    for result in comparison.results
                    if result.method_id == MethodId.MATLAB_SIMPOLY
                ),
                None,
            )
        if matlab is not None and matlab.common_distribution is None:
            self.matlab_note.setText(
                "MATLAB SIMPoly: native Gaussian center b1 only; common distribution unavailable "
                "from current cache."
            )
        else:
            self.matlab_note.setText("")
        self._refresh_plots()

    def _all_series(self) -> list[tuple[str, Any]]:
        if self.comparison is None:
            return []
        from ...reports import series_distributions

        return series_distributions(self.comparison)

    def _selected_series(self) -> list[tuple[str, Any]]:
        choice = self.series_combo.currentText()
        if choice == "All":
            return self._all_series()
        return [(name, distribution) for name, distribution in self._all_series() if name == choice]

    def _preset_changed(self, preset: str) -> None:
        names = self.PRESETS.get(preset)
        if not names:
            self.series_combo.setCurrentText("All")
            return
        self._refresh_plots(preset=set(names))

    def _refresh_plots(self, *_args, preset: set[str] | None = None) -> None:
        if preset is not None:
            series = [
                (name, distribution) for name, distribution in self._all_series() if name in preset
            ]
        else:
            series = self._selected_series()
        self.histogram.set_series(series)
        self.ecdf.set_series(series)
        self._refresh_tables(series)

    def _refresh_tables(self, series: list[tuple[str, Any]]) -> None:
        rows = distribution_quantile_table(series)
        self.summary_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                self.summary_table.setItem(row_index, column, QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()
        agreements: list[tuple[str, ...]] = []
        if self.comparison is not None:
            for item in self.comparison.agreements:
                if item.wasserstein_1 is None:
                    continue
                agreements.append(
                    (
                        method_display_name(item.left_method),
                        method_display_name(item.right_method),
                        _fmt(item.wasserstein_1),
                        _fmt(item.ks_statistic),
                        _fmt(item.median_difference),
                    )
                )
        self.agreement_table.setRowCount(len(agreements))
        for row_index, values in enumerate(agreements):
            for column, value in enumerate(values):
                self.agreement_table.setItem(row_index, column, QTableWidgetItem(value))
        self.agreement_table.resizeColumnsToContents()


class RunMethodsDialog(QDialog):
    """Simple method-run chooser; MATLAB is always consumed from cache only."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run methods")
        self.current_button = "current"
        self.python_check = QCheckBox("Python SIMPoly")
        self.local_check = QCheckBox("Fathom Local")
        self.field_check = QCheckBox("Fathom Field (EDT · Paired Edge · Intensity Profile)")
        for check in (self.python_check, self.local_check, self.field_check):
            check.setChecked(True)
        matlab_note = QLabel(
            "MATLAB SIMPoly is consumed from the validated cache whenever a matching "
            "cache entry exists; the workspace never launches MATLAB."
        )
        matlab_note.setWordWrap(True)
        matlab_note.setStyleSheet("color: #6b6b6b;")
        layout = QVBoxLayout(self)
        layout.addWidget(self.python_check)
        layout.addWidget(self.local_check)
        layout.addWidget(self.field_check)
        layout.addWidget(matlab_note)
        buttons = QDialogButtonBox(self)
        self.current_button_clicked = buttons.addButton(
            "Run current image", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.missing_button_clicked = buttons.addButton(
            "Run missing", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.all_button_clicked = buttons.addButton(
            "Run all dataset", QDialogButtonBox.ButtonRole.AcceptRole
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(buttons)
        self.current_button_clicked.clicked.connect(lambda: self._choose("current"))
        self.missing_button_clicked.clicked.connect(lambda: self._choose("missing"))
        self.all_button_clicked.clicked.connect(lambda: self._choose("all"))

    def _choose(self, button: str) -> None:
        self.current_button = button
        self.accept()


class ImageSummaryPanel(QWidget):
    """Compact current-image summary shown when nothing is selected."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.image_label = QLabel("No image")
        self.image_label.setProperty("role", "title")
        self.metadata_label = QLabel("—")
        self.metadata_label.setProperty("role", "caption")
        self.metadata_label.setWordWrap(True)
        self.median_label = QLabel("—")
        self.median_label.setProperty("role", "primary")
        self.median_caption = QLabel("median diameter")
        self.median_caption.setProperty("role", "caption")
        self.stat_table = QTableWidget(0, 2)
        self.stat_table.setHorizontalHeaderLabels(["Field estimator", "Median (µm)"])
        self.stat_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.stat_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.stat_table.setMaximumHeight(190)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setProperty("role", "muted")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self.image_label)
        layout.addWidget(self.metadata_label)
        layout.addWidget(self.median_caption)
        layout.addWidget(self.median_label)
        layout.addWidget(self.stat_table)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def set_image(self, image: Any, comparison: Any = None) -> None:
        if image is None:
            self.image_label.setText("No image")
            self.metadata_label.setText("Open a dataset to begin.")
            self.median_label.setText("—")
            self.status_label.setText("")
            self.stat_table.setRowCount(0)
            return
        calibration = image.calibration
        self.image_label.setText(image.image_id)
        self.metadata_label.setText(
            f"{calibration.pixel_size_x_m * 1e9:.5g} × {calibration.pixel_size_y_m * 1e9:.5g} nm/px · "
            f"{calibration.source}"
        )
        if comparison is None:
            self.median_label.setText("—")
            self.stat_table.setRowCount(0)
            self.status_label.setText("No cached results for this image yet.")
            return
        field = next(
            (
                result
                for result in comparison.results
                if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
            ),
            None,
        )
        rows: list[tuple[str, str]] = []
        if field is not None:
            for name, key in (
                ("Raw EDT", "diameter_um"),
                ("Raw Edge", "edge_diameter_um"),
                ("Ribbon EDT", "refined_edt_um"),
                ("Ribbon Edge", "refined_edge_um"),
                ("Ribbon Profile", "refined_profile_um"),
            ):
                values = field.local_samples.get(key)
                median = (
                    float(np.nanmedian(np.asarray(values)))
                    if values is not None and np.asarray(values).size
                    else None
                )
                rows.append((name, _fmt(median)))
            self.stat_table.setRowCount(len(rows))
            for row_index, (name, value) in enumerate(rows):
                self.stat_table.setItem(row_index, 0, QTableWidgetItem(name))
                self.stat_table.setItem(row_index, 1, QTableWidgetItem(value))
            statistics = field.native_statistics
            self.status_label.setText(
                f"Supported centerline coverage: {_frac(statistics.get('smooth_coverage_fraction'))} · "
                f"edge acceptance {_frac(statistics.get('edge_acceptance_fraction'))} → "
                f"{_frac(statistics.get('refined_edge_acceptance_fraction'))}"
            )
            median = (
                float(np.nanmedian(field.local_samples["refined_edge_um"]))
                if np.any(field.local_samples["refined_edge_accepted"])
                else None
            )
            self.median_label.setText(_fmt(median))
            self.median_caption.setText("median Ribbon edge diameter (µm)")
        else:
            self.stat_table.setRowCount(0)
            self.status_label.setText("Field results not available.")


class ReportHeaderPanel(QWidget):
    """Focused report/export actions for the Report workspace."""

    datasetReportRequested = Signal()
    bundleExportRequested = Signal()
    imageReportRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        title = QLabel("SCIENTIFIC REPORT")
        title.setProperty("role", "section")
        subtitle = QLabel(
            "Generate the deliverable HTML report and export results for the dataset."
        )
        subtitle.setProperty("role", "caption")
        subtitle.setWordWrap(True)
        self.dataset_button = QPushButton("Generate Dataset Scientific Report")
        self.dataset_button.setProperty("role", "primary")
        self.bundle_button = QPushButton("Export Analysis Bundle")
        self.image_button = QPushButton("Current Image Report")
        buttons = QHBoxLayout()
        buttons.addWidget(self.dataset_button)
        buttons.addWidget(self.bundle_button)
        buttons.addWidget(self.image_button)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(buttons)
        self.dataset_button.clicked.connect(self.datasetReportRequested)
        self.bundle_button.clicked.connect(self.bundleExportRequested)
        self.image_button.clicked.connect(self.imageReportRequested)


def method_entries() -> list[tuple[str, str, str, str]]:
    """Structured methods overview data shared by the in-app help dialogs."""
    return [
        (
            "MATLAB SIMPoly",
            "Native MATLAB SIMPoly consumed from the validated oracle cache.",
            "COMPLETE (cache)",
            (
                "<p>Native Gaussian center b1 reported. The common distribution is unavailable "
                "from the current cache; no histogram is fabricated.</p>"
            ),
        ),
        (
            "Python SIMPoly",
            "Python port of SIMPoly: calibrated length-weighted diameters on the skeleton.",
            "PARTIAL",
            ("<p>Known limitation: KNOWN_LIBRARY_DIVERGENCE — bwskel.</p>"),
        ),
        (
            "Fathom Local",
            "Assisted-ROI candidate cross-section metrology.",
            "COMPLETE",
            (
                "<p>Local fiber candidates with proposed cross-sections; automatic results "
                "require review.</p>"
            ),
        ),
        (
            "Fathom Field",
            "Structure-tensor orientation, anisotropic EDT and paired boundary metrology.",
            "EXPERIMENTAL",
            (
                "<p><b>EDT</b> — Twice the physical distance from the sampled centerline to the "
                "nearest background boundary.<br>"
                "<b>Paired Edge</b> — Distance between both local mask boundaries measured along "
                "the local fiber normal.<br>"
                "<b>Intensity Profile</b> — Paired-edge width refined against local subpixel "
                "gradient transitions in the raw SEM image.<br>"
                "No estimator is called best. Graph reconstruction and fiber instances are not "
                "implemented.</p>"
            ),
        ),
        (
            "Oriented Ribbon V1",
            "Experimental refined centerline from paired opposite boundaries.",
            "EXPERIMENTAL",
            (
                "<p><b>EXPERIMENTAL</b> — geometric centerline refinement from paired opposite "
                "boundaries: local midpoints, a confidence-weighted smooth centerline on "
                "non-branching runs, then re-measurement of EDT, paired-edge and profile along "
                "it. Validated on known-truth synthetic geometry; real SEM results represent "
                "method behavior/agreement, not known absolute accuracy.</p>"
            ),
        ),
        (
            "Manual 5×5",
            "Operator reference grid: 25 perpendicular width measurements per image.",
            "REFERENCE",
            (
                "<p>Sparse human reference; never ground truth. Missing measurements are never "
                "filled in.</p>"
            ),
        ),
        (
            "Consensus",
            "Equal-method quantile pseudo-reference across participating methods.",
            "REFERENCE",
            ("<p>Not ground truth. Field estimator variants do not add independent votes.</p>"),
        ),
    ]


class MethodsOverviewDialog(QDialog):
    """Structured methods overview: name, purpose, status, detailed info."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About methods")
        self.setMinimumSize(640, 520)
        self._details: dict[str, str] = {}
        layout = QVBoxLayout(self)
        title = QLabel("Fathom Fibers — measurement methods")
        title.setProperty("role", "title")
        layout.addWidget(title)
        self.list = QTableWidget(0, 3)
        self.list.setHorizontalHeaderLabels(["Method", "Purpose", "Status"])
        self.list.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.list.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.list.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.list.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.list.cellDoubleClicked.connect(self._show_details)
        layout.addWidget(self.list, 1)
        info = QLabel("Double-click a row for the full scientific description, including caveats.")
        info.setProperty("role", "caption")
        layout.addWidget(info)
        self._populate()

    def _populate(self) -> None:
        entries = method_entries()
        self.list.setRowCount(len(entries))
        for row_index, (name, purpose, status, details) in enumerate(entries):
            self.list.setItem(row_index, 0, QTableWidgetItem(name))
            self.list.setItem(row_index, 1, QTableWidgetItem(purpose))
            status_item = QTableWidgetItem(status)
            if status == "EXPERIMENTAL":
                status_item.setForeground(QColor("#d99a2b"))
            self.list.setItem(row_index, 2, status_item)
            self._details[name] = details

    def _show_details(self, row: int, _column: int) -> None:
        name = self.list.item(row, 0).text()
        details = self._details.get(name, "")
        QMessageBox.information(self, name, details)
