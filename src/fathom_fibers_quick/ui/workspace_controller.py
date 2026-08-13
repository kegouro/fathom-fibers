"""Qt controller coordinating dataset navigation, method runs and reports.

The controller owns no scientific logic: it orchestrates the Qt-free
``workspace`` layer, ``FathomEngine`` and ``ProjectSession`` behind Qt signals
so panels never call the scientific core directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThreadPool, Signal

from ..application import ProjectSession
from ..core.methods import MethodId, MethodResult
from ..measurement_records import MeasurementKind, MeasurementRecord
from ..unified_comparison import UnifiedMethodComparison
from ..validation.manual_review import GridCellStatus, Manual5x5Review
from ..workspace import (
    Manual5x5Store,
    WorkspaceCache,
    WorkspaceDataset,
    load_workspace_dataset,
    resolve_matlab_cache_root,
)
from .tasks import AnalysisTask

logger = logging.getLogger(__name__)

METHOD_ORDER = (
    MethodId.PYTHON_SIMPOLY,
    MethodId.FATHOM_LOCAL,
    MethodId.FATHOM_FIELD_GRAPH_V1,
)


class WorkspaceController(QObject):
    """Dataset-level state and background work for the scientific workspace."""

    datasetLoaded = Signal()
    imageChanged = Signal()
    resultsChanged = Signal()
    busyChanged = Signal(bool)
    methodProgress = Signal(str)
    manualChanged = Signal()
    reportReady = Signal(str)
    reportFailed = Signal(str, str)
    errorRaised = Signal(str, str)

    def __init__(
        self,
        session: ProjectSession,
        parent: QObject | None = None,
        *,
        pixel_size_m_fallback: float | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self._pixel_size_m_fallback = pixel_size_m_fallback
        self.cache = WorkspaceCache(cache_root)
        self.dataset: WorkspaceDataset | None = None
        self.current_index: int = -1
        self.matlab_cache_root: Path | None = None
        self.manual: Manual5x5Store | None = None
        self.comparison: UnifiedMethodComparison | None = None
        self.summary_payload: dict[str, Any] | None = None
        self._memory_cache: dict[str, UnifiedMethodComparison] = {}
        self._pool = QThreadPool(self)
        self._tasks: list[AnalysisTask] = []
        self._busy = False
        self._queue: list[tuple[int, str]] = []
        self._queue_index = 0

    # ------------------------------------------------------------------ state

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def current_image(self) -> Any | None:
        if self.dataset is None or not 0 <= self.current_index < len(self.dataset.images):
            return None
        return self.dataset.images[self.current_index]

    @property
    def current_stem(self) -> str | None:
        image = self.current_image
        return image.stem if image else None

    @property
    def results(self) -> dict[MethodId, MethodResult]:
        if self.comparison is None:
            return {}
        return {result.method_id: result for result in self.comparison.results}

    @property
    def manual_review(self) -> Manual5x5Review | None:
        image = self.current_image
        if self.manual is None or image is None:
            return None
        return self.manual.ensure_review(image.case_id)

    # ------------------------------------------------------------- dataset io

    def open_dataset(self, directory: str | Path) -> WorkspaceDataset:
        dataset = load_workspace_dataset(directory, repo=Path.cwd())
        self._adopt_dataset(dataset)
        return dataset

    def set_dataset(self, dataset: WorkspaceDataset) -> None:
        self._adopt_dataset(dataset)

    def _adopt_dataset(self, dataset: WorkspaceDataset) -> None:
        self._cancel_tasks()
        self.dataset = dataset
        self.matlab_cache_root = resolve_matlab_cache_root(
            dataset.source_dir, repo=Path.cwd()
        )
        self.manual = Manual5x5Store(self.cache.root, dataset.dataset_id)
        self.manual.load()
        self.comparison = None
        self.summary_payload = None
        self._memory_cache = {}
        self.datasetLoaded.emit()
        if dataset.images:
            self.select_image(0)

    def select_image(self, index: int) -> None:
        if self.dataset is None or not 0 <= index < len(self.dataset.images):
            return
        self._cancel_tasks()
        image = self.dataset.images[index]
        self.current_index = index
        try:
            self._open_in_session(image)
        except Exception as exc:
            self.errorRaised.emit(f"Open {image.filename} failed", str(exc))
            self.imageChanged.emit()
            return
        stem = image.stem
        self.comparison = self._memory_cache.get(stem)
        self.summary_payload = None
        if self.comparison is None and self.cache.has_full(stem):
            try:
                self.comparison = self.cache.load_comparison(stem)
            except Exception as exc:
                self.errorRaised.emit("Workspace cache read failed", str(exc))
                self.comparison = None
        if self.comparison is not None:
            self._memory_cache[stem] = self.comparison
        else:
            self.summary_payload = self.cache.summary_payload(stem)
        self._restore_manual_records()
        self.imageChanged.emit()

    def _open_in_session(self, image: Any) -> None:
        try:
            self.session.open_image(image.absolute_path)
            return
        except ValueError as exc:
            if "calibration" not in str(exc).lower() or self._pixel_size_m_fallback is None:
                raise
        self.session.open_image(image.absolute_path, manual_pixel_size_m=self._pixel_size_m_fallback)

    def next_image(self) -> None:
        if self.dataset:
            self.select_image(min(self.current_index + 1, len(self.dataset.images) - 1))

    def previous_image(self) -> None:
        if self.dataset:
            self.select_image(max(self.current_index - 1, 0))

    # -------------------------------------------------------------- analysis

    def run_current_image(self) -> None:
        if self.current_image is None:
            return
        self._launch(self._analysis_callable(self.current_index))

    def run_missing(self) -> None:
        if self.dataset is None:
            return
        missing = [
            index
            for index, image in enumerate(self.dataset.images)
            if not self.cache.has_full(image.stem)
        ]
        self._run_image_list(missing)

    def run_all_dataset(self) -> None:
        if self.dataset is None:
            return
        self._run_image_list(list(range(len(self.dataset.images))))

    def _run_image_list(self, indexes: list[int]) -> None:
        if not indexes:
            self.methodProgress.emit("Nothing to run: full results already cached.")
            return
        self._queue = [
            (index, self.dataset.images[index].filename) for index in indexes
        ]
        self._queue_index = 0
        self._launch(self._analysis_callable(self._queue[0][0]))

    def _analysis_callable(self, index: int) -> Callable[[], UnifiedMethodComparison]:
        image = self.dataset.images[index]

        def run() -> UnifiedMethodComparison:
            from ..workspace import compute_comparison_staged

            started = time.monotonic()
            scientific = self.session.engine.open_image(
                image.absolute_path,
                manual_pixel_size_m=self._pixel_size_m_fallback,
            )
            record_values = self._manual_records_for(scientific)

            def report(message: str) -> None:
                self.methodProgress.emit(f"Analyzing {image.filename} — {message}")

            comparison = compute_comparison_staged(
                self.session.engine,
                scientific,
                matlab_cache_root=self.matlab_cache_root,
                records=record_values,
                progress=report,
            )
            report("caching")
            self.cache.store_comparison(image.stem, comparison)
            comparison = self.cache.load_comparison(image.stem) or comparison
            self.methodProgress.emit(
                f"Analyzed {image.filename} in {time.monotonic() - started:.0f}s"
            )
            return comparison

        return run

    def _launch(self, function: Callable[[], Any]) -> None:
        if self._busy:
            return
        self._set_busy(True)
        task = AnalysisTask(function)
        task.signals.result.connect(self._analysis_done)
        task.signals.error.connect(self._analysis_error)
        task.signals.finished.connect(self._analysis_finished)
        self._tasks.append(task)
        self._pool.start(task)

    def _analysis_done(self, comparison: UnifiedMethodComparison) -> None:
        image = self.current_image
        if image is not None:
            self._memory_cache[image.stem] = comparison
        if image is not None and Path(comparison.image_id).stem == image.stem:
            self.comparison = comparison
            self.summary_payload = None
        self.resultsChanged.emit()

    def _analysis_error(self, error: Exception, traceback_text: str) -> None:
        self.errorRaised.emit(f"Analysis failed: {error}", traceback_text)

    def _analysis_finished(self) -> None:
        if self._queue and self._queue_index < len(self._queue) - 1:
            self._queue_index += 1
            self._launch(self._analysis_callable(self._queue[self._queue_index][0]))
            return
        self._queue = []
        self._set_busy(False)
        self.methodProgress.emit("Analysis queue complete")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busyChanged.emit(busy)

    def _cancel_tasks(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        self._queue = []
        self._queue_index = 0
        if self._busy:
            self._set_busy(False)

    # ---------------------------------------------------------------- manual

    def _manual_records_for(self, image: Any) -> list[MeasurementRecord]:
        return [
            record
            for record in self.session.project.records
            if record.kind == MeasurementKind.PROJECTED_WIDTH
            and record.is_included_in_statistics
        ]

    def _restore_manual_records(self) -> None:
        image = self.current_image
        if image is None or self.session.project is None or self.session.image is None:
            return
        review = self.manual.ensure_review(image.case_id)
        existing = {record.tags[-1] for record in self.session.project.records if len(record.tags) >= 2}
        for row in range(5):
            for column in range(5):
                cell = review.cell(row, column)
                if cell.status != GridCellStatus.MEASURED or not cell.geometry:
                    continue
                if cell.position in existing:
                    continue
                try:
                    record = self.session.create_measurement(
                        MeasurementKind.PROJECTED_WIDTH,
                        cell.geometry,
                        name=f"Manual 5×5 {cell.position}",
                        fiber_id=cell.fiber_id,
                    )
                except Exception as exc:
                    logger.warning("Manual 5x5 restore failed for %s: %s", cell.position, exc)
                    continue
                record.tags = [image.case_id, cell.position]
                record.notes = cell.notes
                record.protocol_snapshot = {
                    "protocol_id": "MANUAL_5X5_REFERENCE",
                    "case_id": image.case_id,
                    "grid_position": cell.position,
                }
                record.updated_at = cell.timestamp
        if self.session.project.records:
            self.session.dirty = True

    def accept_manual_measurement(self, record: MeasurementRecord, row: int, column: int) -> None:
        image = self.current_image
        if image is None or self.manual is None:
            return
        review = self.manual.ensure_review(image.case_id)
        cell = review.cell(row, column)
        cell.status = GridCellStatus.MEASURED
        cell.fiber_id = record.fiber_id
        cell.measurement_id = record.measurement_id
        cell.geometry = dict(record.geometry)
        cell.diameter = record.primary_value
        cell.unit = record.primary_unit
        cell.calibration_snapshot = dict(record.calibration_snapshot)
        cell.timestamp = record.created_at
        self.manual.save()
        self.manualChanged.emit()

    def remove_manual_measurement(self, row: int, column: int) -> None:
        image = self.current_image
        if image is None or self.manual is None:
            return
        review = self.manual.ensure_review(image.case_id)
        cell = review.cell(row, column)
        cell.status = GridCellStatus.NOT_REVIEWED
        cell.measurement_id = None
        cell.geometry = None
        cell.diameter = None
        cell.unit = None
        cell.calibration_snapshot = None
        cell.fiber_id = None
        cell.timestamp = None
        self.manual.save()
        self.manualChanged.emit()

    def skip_manual_measurement(self, row: int, column: int, reason: str) -> None:
        image = self.current_image
        if image is None or self.manual is None:
            return
        review = self.manual.ensure_review(image.case_id)
        cell = review.cell(row, column)
        cell.set_status(GridCellStatus.SKIPPED_WITH_REASON, notes=reason)
        self.manual.save()
        self.manualChanged.emit()

    # --------------------------------------------------------------- reports

    def generate_image_report(self) -> None:
        if self.comparison is None or self.session.image is None:
            self.errorRaised.emit("Image report unavailable", "Run the methods first.")
            return
        comparison = self.comparison
        image = self.session.image
        manual = self.manual_review
        manual_count = manual.measurement_count if manual else 0

        def build() -> str:
            from ..reports import build_image_report

            output = self.cache.root / "reports" / self.current_stem
            index = build_image_report(
                comparison,
                image,
                output_dir=output,
                manual_complete=None,
                manual_count=manual_count,
            )
            return str(index)

        self._launch_report_task(build)

    def generate_dataset_report(self) -> None:
        if self.dataset is None:
            self.errorRaised.emit("Dataset report unavailable", "Open a dataset first.")
            return
        dataset = self.dataset
        manual = self.manual

        def build() -> str:
            from ..reports import build_final_dataset_report

            output = self.cache.root.parent / "final-report"
            index = build_final_dataset_report(
                Path.cwd(),
                dataset=dataset,
                manual_store=manual,
                output_dir=output,
            )
            return str(index)

        self._launch_report_task(build)

    def export_analysis_bundle(self, directory: str | Path) -> None:
        """Export the full analysis bundle (report + CSV/JSON + figures)."""
        if self.dataset is None:
            self.errorRaised.emit("Export unavailable", "Open a dataset first.")
            return
        dataset = self.dataset
        manual = self.manual

        def build() -> str:
            from ..export_bundle import export_analysis_bundle

            root = export_analysis_bundle(
                Path.cwd(),
                dataset=dataset,
                manual_store=manual,
                output_dir=Path(directory),
            )
            return str(root)

        self._launch_report_task(build)

    def _launch_report_task(self, function: Callable[[], str]) -> None:
        if self._busy:
            self.errorRaised.emit(
                "Busy", "Finish or cancel the current analysis before generating a report."
            )
            return
        self._set_busy(True)
        task = AnalysisTask(function)
        task.signals.result.connect(self._report_done)
        task.signals.error.connect(self._report_error)
        task.signals.finished.connect(lambda: self._set_busy(False))
        self._tasks.append(task)
        self._pool.start(task)

    def _report_done(self, path: str) -> None:
        self.reportReady.emit(path)

    def _report_error(self, error: Exception, traceback_text: str) -> None:
        self.reportFailed.emit(str(error), traceback_text)

    # ---------------------------------------------------------------- export

    def export_current_results(self, directory: str | Path) -> list[Path]:
        """Export current-image measurements (CSV) and summaries (JSON)."""
        stem = self.current_stem or "image"
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        if self.session.project is not None:
            from ..exporters import export_csv

            written.append(export_csv(self.session.project, target / f"{stem}-measurements.csv"))
        if self.comparison is not None:
            payload = _comparison_payload(self.comparison)
            path = target / f"{stem}-method-results.json"
            path.write_text(json_dumps(payload), encoding="utf-8")
            written.append(path)
        return written

    def export_dataset_results(self, directory: str | Path) -> list[Path]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        summary_rows: list[dict[str, Any]] = []
        for image in self.dataset.images:
            payload = self.cache.summary_payload(image.stem)
            comparison = self._memory_cache.get(image.stem)
            if payload is None and comparison is None:
                continue
            row: dict[str, Any] = {"case_id": image.case_id, "filename": image.filename}
            if comparison is not None:
                row.update(_summary_row_from_comparison(comparison))
            if payload is not None:
                for entry in payload.get("results", ()):
                    row.setdefault(f"{entry['method_id']}_status", entry.get("status"))
            summary_rows.append(row)
        if summary_rows:
            import csv

            path = target / "dataset-summary.csv"
            fields: list[str] = []
            for row in summary_rows:
                for key in row:
                    if key not in fields:
                        fields.append(key)
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(summary_rows)
            written.append(path)
        payload_path = target / "dataset-method-results.json"
        payload_path.write_text(
            json_dumps(
                {
                    "dataset_id": self.dataset.dataset_id,
                    "images": [
                        {"case_id": image.case_id, "filename": image.filename}
                        for image in self.dataset.images
                    ],
                }
            ),
            encoding="utf-8",
        )
        written.append(payload_path)
        return written


def _comparison_payload(comparison: UnifiedMethodComparison) -> dict[str, Any]:
    from ..unified_comparison import build_image_report
    from ..validation.unified_methods import _comparison_payload as payload

    return payload(build_image_report(comparison))


def _summary_row_from_comparison(comparison: UnifiedMethodComparison) -> dict[str, Any]:
    from ..core.distributions import summarize_distribution

    row: dict[str, Any] = {}
    for result in comparison.results:
        key = result.method_id.value
        row[f"{key}_status"] = result.status.value
        distribution = result.common_distribution
        if distribution is None and result.method_id == MethodId.MANUAL_5X5_REFERENCE:
            distribution = result.native_distribution
        if distribution is not None:
            summary = summarize_distribution(distribution)
            row[f"{key}_median_um"] = summary.weighted_median
            row[f"{key}_n"] = summary.n
    return row


def json_dumps(payload: Any) -> str:
    import json

    return json.dumps(payload, indent=2, default=str)


__all__ = ["WorkspaceController"]
