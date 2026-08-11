from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..api import FathomEngine
from ..core.contracts import (
    FathomAnalysisResult,
    MethodComparisonResult,
    ScientificImage,
)
from ..history import Command, HistoryManager
from ..measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementSource,
    MeasurementStatus,
    normalize_tags,
)
from ..model import Project
from ..oracles.simpoly_source import (
    PROFILE_SOURCE_COMPAT_V1,
    SIMPolySourceResult,
)
from ..project_io import load_project, save_project
from ..protocols import BUILTIN_PROTOCOLS


class ProjectSession:
    """Application state and commands shared by any frontend.

    Widgets may request operations here, but never mutate ``MeasurementRecord``
    instances directly. All scientific geometry is recalculated by ``FathomEngine``.
    """

    def __init__(self, engine: FathomEngine | None = None) -> None:
        self.engine = engine or FathomEngine()
        self.project: Project | None = None
        self.image: ScientificImage | None = None
        self.history = HistoryManager()
        self.selected_record_id: str | None = None
        self.roi_bbox: tuple[int, int, int, int] | None = None
        self.dirty = False
        self._listeners: list[Callable[[str], None]] = []
        self.history.register_on_change(lambda: self._emit("history"))

    def subscribe(self, callback: Callable[[str], None]) -> None:
        self._listeners.append(callback)

    def _emit(self, event: str) -> None:
        for callback in tuple(self._listeners):
            callback(event)

    def _require(self) -> tuple[Project, ScientificImage]:
        if self.project is None or self.image is None:
            raise RuntimeError("No project image is open")
        return self.project, self.image

    def new_from_image(self, image: ScientificImage) -> Project:
        project = Project(
            schema_version=4,
            image=image.to_document(),
            protocols={key: protocol.to_dict() for key, protocol in BUILTIN_PROTOCOLS.items()},
        )
        self.project = project
        self.image = image
        self.selected_record_id = None
        self.roi_bbox = None
        self.history.clear()
        self.dirty = False
        self._emit("project")
        return project

    def open_image(
        self,
        path: str | Path,
        *,
        manual_pixel_size_m: float | None = None,
    ) -> Project:
        return self.new_from_image(
            self.engine.open_image(path, manual_pixel_size_m=manual_pixel_size_m)
        )

    def open_project(self, path: str | Path) -> Project:
        project = load_project(path)
        loaded = self.engine.open_image(
            project.image.path,
            manual_pixel_size_m=project.image.calibration.pixel_size_x_m,
            compute_hash=False,
        )
        image = ScientificImage(
            pixels=loaded.pixels,
            gray=loaded.gray,
            calibration=project.image.calibration,
            image_id=Path(project.image.path).name,
            metadata=dict(project.image.metadata),
            footer_bounds=project.image.footer_bounds,
            source_path=project.image.path,
            source_sha256=project.image.source_sha256,
        )
        self.project = project
        self.image = image
        self.selected_record_id = None
        self.roi_bbox = None
        self.history.clear()
        self.dirty = False
        self._emit("project")
        return project

    def save(self, path: str | Path | None = None) -> Path:
        project, _image = self._require()
        target = path or project.project_path
        if target is None:
            raise ValueError("A project path is required for the first save")
        saved = save_project(project, target)
        self.dirty = False
        self._emit("saved")
        return saved

    def select(self, record_id: str | None) -> None:
        self.selected_record_id = record_id
        self._emit("selection")

    def selected_record(self) -> MeasurementRecord | None:
        if self.project is None or self.selected_record_id is None:
            return None
        return next(
            (r for r in self.project.records if r.measurement_id == self.selected_record_id),
            None,
        )

    def set_roi(self, bbox: tuple[int, int, int, int] | None) -> None:
        self.roi_bbox = bbox
        self._emit("roi")

    def _push(self, command: Command) -> None:
        project, _image = self._require()
        self.history.push_and_execute(command)
        project.history_metadata.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "action": "EXECUTE",
                "description": command.description,
                "affected_ids": list(command.affected_ids),
            }
        )
        self.dirty = True
        self._emit("records")

    def undo(self) -> Command | None:
        command = self.history.undo()
        if command and self.project:
            self.project.history_metadata.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "action": "UNDO",
                    "description": command.description,
                    "affected_ids": list(command.affected_ids),
                }
            )
            self.dirty = True
            self._emit("records")
        return command

    def redo(self) -> Command | None:
        command = self.history.redo()
        if command and self.project:
            self.project.history_metadata.append(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "action": "REDO",
                    "description": command.description,
                    "affected_ids": list(command.affected_ids),
                }
            )
            self.dirty = True
            self._emit("records")
        return command

    def add_record(self, record: MeasurementRecord, description: str | None = None) -> None:
        project, _image = self._require()

        def execute() -> None:
            if not any(r.measurement_id == record.measurement_id for r in project.records):
                project.records.append(record)
            self.selected_record_id = record.measurement_id

        def undo() -> None:
            project.records = [r for r in project.records if r.measurement_id != record.measurement_id]
            if self.selected_record_id == record.measurement_id:
                self.selected_record_id = None

        self._push(
            Command(
                description or f"Add {record.measurement_id}",
                execute,
                undo,
                affected_ids=[record.measurement_id],
            )
        )

    def create_measurement(
        self,
        kind: MeasurementKind | str,
        geometry: Mapping[str, Any],
        *,
        name: str | None = None,
        fiber_id: str | None = None,
        status: MeasurementStatus = MeasurementStatus.ACCEPTED,
        source: MeasurementSource = MeasurementSource.MANUAL,
        profile_bandwidth_px: int = 3,
    ) -> MeasurementRecord:
        project, image = self._require()
        kind = MeasurementKind(kind)
        result = self.engine.measure(
            image,
            kind,
            geometry,
            profile_bandwidth_px=profile_bandwidth_px,
        )
        measurement_id = project.next_measurement_id()
        record = MeasurementRecord(
            measurement_id=measurement_id,
            kind=kind,
            name=name or f"{kind.value.replace('_', ' ').title()} {measurement_id}",
            status=status,
            source=source,
            image_id=image.image_id,
            sample_id=project.sample_id,
            fiber_id=fiber_id or (project.active_fiber_id if kind == MeasurementKind.PROJECTED_WIDTH else None),
            geometry=dict(result.geometry),
            values=dict(result.values),
            calibration_snapshot=asdict(image.calibration),
            quality_flags=list(result.flags),
            protocol_snapshot=dict(project.protocols.get(project.active_protocol_id, {})),
        )
        self.add_record(record, f"Create {kind.value} {measurement_id}")
        return record

    def update_geometry(self, record_id: str, geometry: Mapping[str, Any]) -> None:
        project, image = self._require()
        record = next(r for r in project.records if r.measurement_id == record_id)
        result = self.engine.measure(image, record.kind, geometry)
        old_geometry = copy.deepcopy(record.geometry)
        old_values = copy.deepcopy(record.values)
        old_status = record.status
        new_geometry = copy.deepcopy(result.geometry)
        new_values = copy.deepcopy(result.values)

        def execute() -> None:
            record.geometry = copy.deepcopy(new_geometry)
            record.values = copy.deepcopy(new_values)
            record.status = MeasurementStatus.MANUALLY_EDITED
            record.updated_at = datetime.now(UTC).isoformat()

        def undo() -> None:
            record.geometry = copy.deepcopy(old_geometry)
            record.values = copy.deepcopy(old_values)
            record.status = old_status

        self._push(Command(f"Edit geometry {record_id}", execute, undo, affected_ids=[record_id]))

    def update_metadata(self, record_ids: list[str], **changes: Any) -> None:
        project, _image = self._require()
        allowed = {"name", "notes", "tags", "status", "fiber_id", "group"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Derived or unsupported fields are read-only: {sorted(unknown)}")
        records = [r for r in project.records if r.measurement_id in set(record_ids)]
        old = {r.measurement_id: {key: copy.deepcopy(getattr(r, key)) for key in changes} for r in records}
        normalized = dict(changes)
        if "tags" in normalized:
            normalized["tags"] = normalize_tags(normalized["tags"])
        if "status" in normalized:
            normalized["status"] = MeasurementStatus(normalized["status"])

        def execute() -> None:
            for record in records:
                for key, value in normalized.items():
                    setattr(record, key, copy.deepcopy(value))
                record.updated_at = datetime.now(UTC).isoformat()

        def undo() -> None:
            for record in records:
                for key, value in old[record.measurement_id].items():
                    setattr(record, key, copy.deepcopy(value))

        self._push(Command(f"Edit metadata ({len(records)})", execute, undo, affected_ids=record_ids))

    def delete_records(self, record_ids: list[str]) -> None:
        project, _image = self._require()
        selected = set(record_ids)
        indexed = [(i, r) for i, r in enumerate(project.records) if r.measurement_id in selected]

        def execute() -> None:
            project.records = [r for r in project.records if r.measurement_id not in selected]
            if self.selected_record_id in selected:
                self.selected_record_id = None

        def undo() -> None:
            for index, record in indexed:
                project.records.insert(min(index, len(project.records)), record)

        self._push(Command(f"Delete {len(indexed)} records", execute, undo, affected_ids=record_ids))

    def apply_fathom_result(self, result: FathomAnalysisResult) -> list[MeasurementRecord]:
        project, image = self._require()
        records: list[MeasurementRecord] = []
        for candidate in result.candidates:
            fiber_id = project.get_next_fiber_id()
            for proposal in candidate.proposed_measurements:
                measurement_id = project.next_measurement_id()
                records.append(
                    MeasurementRecord(
                        measurement_id=measurement_id,
                        kind=MeasurementKind.PROJECTED_WIDTH,
                        name=f"Fathom proposal {measurement_id}",
                        status=MeasurementStatus.PROPOSED,
                        source=MeasurementSource.FATHOM,
                        image_id=image.image_id,
                        sample_id=project.sample_id,
                        fiber_id=fiber_id,
                        geometry={"p1": proposal.p1, "p2": proposal.p2},
                        values={"width_m": proposal.width_m, "length_m": proposal.width_m},
                        calibration_snapshot=asdict(image.calibration),
                        quality_flags=sorted(candidate.quality_flags | proposal.quality_flags),
                        confidence=candidate.confidence_score,
                        protocol_snapshot=dict(project.protocols.get(project.active_protocol_id, {})),
                    )
                )
        if not records:
            return []

        def execute() -> None:
            present = {r.measurement_id for r in project.records}
            project.records.extend(r for r in records if r.measurement_id not in present)
            project.analysis_runs.append(
                {
                    "method": result.method,
                    "roi_bbox": result.roi_bbox,
                    "status": "PROPOSED",
                    "record_ids": [r.measurement_id for r in records],
                }
            )

        def undo() -> None:
            ids = {r.measurement_id for r in records}
            project.records = [r for r in project.records if r.measurement_id not in ids]
            project.analysis_runs = [run for run in project.analysis_runs if run.get("record_ids") != list(ids)]

        self._push(
            Command(
                f"Apply Fathom proposals ({len(records)})",
                execute,
                undo,
                affected_ids=[r.measurement_id for r in records],
            )
        )
        return records

    def apply_simpoly_result(
        self,
        result: SIMPolySourceResult,
        *,
        profile: str,
        roi_bbox: tuple[int, int, int, int] | None,
    ) -> MeasurementRecord:
        project, image = self._require()
        measurement_id = project.next_measurement_id()
        scale = image.calibration.pixel_size_x_m
        source = (
            MeasurementSource.SIMPOLY_SOURCE_COMPAT
            if profile == PROFILE_SOURCE_COMPAT_V1
            else MeasurementSource.SIMPOLY_CONTROLLED_INPUT
        )
        center_m = result.gaussian_center_px * scale if result.gaussian_center_px is not None else None
        record = MeasurementRecord(
            measurement_id=measurement_id,
            kind=MeasurementKind.DIAMETER_DISTRIBUTION,
            name=f"SIMPoly {profile} {measurement_id}",
            status=MeasurementStatus.PROPOSED,
            source=source,
            image_id=image.image_id,
            sample_id=project.sample_id,
            roi_id="ACTIVE_ROI" if roi_bbox else None,
            geometry={"bbox": roi_bbox} if roi_bbox else {},
            values={
                "main_reported_value_m": center_m,
                "gaussian_center_px": result.gaussian_center_px,
                "source_reported_stdev_px": result.source_reported_stdev_px,
                "mathematical_gaussian_sigma_px": result.mathematical_gaussian_sigma_px,
                "arithmetic_mean_px": result.arithmetic_mean_px,
                "median_px": result.median_px,
                "valid_diameter_count": int(result.local_diameters_px.size),
                "histogram_counts": result.histogram_counts.tolist(),
                "histogram_edges": result.histogram_edges.tolist(),
                "profile": profile,
                "estimand": "SIMPOLY_GAUSSIAN_CENTER",
            },
            calibration_snapshot=asdict(image.calibration),
            quality_flags=list(result.flags),
        )
        self.add_record(record, f"Apply SIMPoly proposal {measurement_id}")
        return record

    def compare_methods(self) -> MethodComparisonResult:
        project, image = self._require()
        return self.engine.compare_methods(
            image,
            roi_bbox=self.roi_bbox,
            manual_measurements=project.records,
        )
