from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementStatus,
)


def compute_measurement_width(p1: tuple[float, float], p2: tuple[float, float], calibration: Calibration) -> float:
    calibration.validate()
    if not (math.isfinite(p1[0]) and math.isfinite(p1[1])):
        raise ValueError(f"Invalid p1 coordinates: {p1}")
    if not (math.isfinite(p2[0]) and math.isfinite(p2[1])):
        raise ValueError(f"Invalid p2 coordinates: {p2}")
    dx = (p2[0] - p1[0]) * calibration.pixel_size_x_m
    dy = (p2[1] - p1[1]) * calibration.pixel_size_y_m
    return math.hypot(dx, dy)


@dataclass(slots=True)
class Calibration:
    pixel_size_x_m: float
    pixel_size_y_m: float
    source: str
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not (math.isfinite(self.pixel_size_x_m) and self.pixel_size_x_m > 0):
            raise ValueError(f"Invalid pixel_size_x_m: {self.pixel_size_x_m}")
        if not (math.isfinite(self.pixel_size_y_m) and self.pixel_size_y_m > 0):
            raise ValueError(f"Invalid pixel_size_y_m: {self.pixel_size_y_m}")

    def distance_m(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        return compute_measurement_width(p1, p2, self)


@dataclass(slots=True)
class ImageDocument:
    path: str
    width_px: int
    height_px: int
    calibration: Calibration
    metadata: dict[str, Any] = field(default_factory=dict)
    footer_bounds: tuple[int, int] | None = None
    source_sha256: str | None = None


def Measurement(*args: Any, **kwargs: Any) -> MeasurementRecord:
    if len(args) >= 4 or "p1" in kwargs or "fiber_id" in kwargs or "accepted" in kwargs:
        return MeasurementRecord.create_legacy(*args, **kwargs)
    return MeasurementRecord(*args, **kwargs)


@dataclass
class Project:
    schema_version: int
    image: ImageDocument
    records: list[MeasurementRecord] = field(default_factory=list)
    project_path: str | None = None
    notes: str = ""
    target_sections: int = 5
    active_fiber_id: str = "F001"
    next_fiber_counter: int = 1
    next_record_counter: int = 1
    fiber_notes: dict[str, str] = field(default_factory=dict)
    group_names: dict[int, str] = field(default_factory=dict)
    manual_ranges: list[dict[str, Any]] = field(default_factory=list)

    @property
    def measurements(self) -> list[MeasurementRecord]:
        """Backward-compatible facade returning all records."""
        return self.records

    @measurements.setter
    def measurements(self, value: list[MeasurementRecord]) -> None:
        self.records = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "image": {
                "path": self.image.path,
                "width_px": self.image.width_px,
                "height_px": self.image.height_px,
                "calibration": asdict(self.image.calibration),
                "metadata": self.image.metadata,
                "footer_bounds": list(self.image.footer_bounds) if self.image.footer_bounds else None,
                "source_sha256": self.image.source_sha256,
            },
            "records": [r.to_dict() for r in self.records],
            "project_path": self.project_path,
            "notes": self.notes,
            "target_sections": self.target_sections,
            "active_fiber_id": self.active_fiber_id,
            "next_fiber_counter": self.next_fiber_counter,
            "next_record_counter": self.next_record_counter,
            "fiber_notes": self.fiber_notes,
            "group_names": {str(k): v for k, v in self.group_names.items()},
            "manual_ranges": self.manual_ranges,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        image_data = dict(data["image"])
        calibration = Calibration(**image_data.pop("calibration"))
        if image_data.get("footer_bounds") is not None:
            image_data["footer_bounds"] = tuple(image_data["footer_bounds"])
        image = ImageDocument(calibration=calibration, **image_data)

        records: list[MeasurementRecord] = []

        # 1. Check if new records schema is present
        if "records" in data:
            for item in data["records"]:
                records.append(MeasurementRecord.from_dict(item))
        # 2. Migration from legacy schema (measurements array)
        elif "measurements" in data:
            for item in data["measurements"]:
                d = dict(item)
                p1 = tuple(d.get("p1", (0.0, 0.0)))
                p2 = tuple(d.get("p2", (0.0, 0.0)))
                w_m = float(d.get("width_m", 0.0))
                m_id = str(d.get("measurement_id", "M001"))
                f_id = str(d.get("fiber_id", "F001"))
                accepted = bool(d.get("accepted", True))
                status = MeasurementStatus.ACCEPTED if accepted else MeasurementStatus.REJECTED
                source_str = str(d.get("method", "MANUAL"))

                rec = MeasurementRecord(
                    measurement_id=m_id,
                    kind=MeasurementKind.PROJECTED_WIDTH,
                    name=f"Ancho {m_id}",
                    status=status,
                    fiber_id=f_id,
                    group=d.get("group"),
                    defect=d.get("defect", "None"),
                    notes=d.get("note", ""),
                    confidence=d.get("confidence"),
                    geometry={"p1": p1, "p2": p2},
                    values={"width_m": w_m, "length_m": w_m},
                    calibration_snapshot={
                        "pixel_size_x_m": calibration.pixel_size_x_m,
                        "pixel_size_y_m": calibration.pixel_size_y_m,
                    },
                )
                rec.method = source_str
                records.append(rec)

        # Compute next counters
        used_records = []
        for r in records:
            if r.measurement_id.startswith("M") and r.measurement_id[1:].isdigit():
                used_records.append(int(r.measurement_id[1:]))
        next_rec = max(used_records, default=0) + 1
        stored_next_rec = int(data.get("next_record_counter", next_rec))

        used_fibers = []
        for r in records:
            if r.fiber_id and r.fiber_id.startswith("F") and r.fiber_id[1:].isdigit():
                used_fibers.append(int(r.fiber_id[1:]))
        next_fib = max(used_fibers, default=0) + 1
        stored_next_fib = int(data.get("next_fiber_counter", next_fib))

        return cls(
            schema_version=int(data.get("schema_version", 2)),
            image=image,
            records=records,
            project_path=data.get("project_path"),
            notes=data.get("notes", ""),
            target_sections=int(data.get("target_sections", 5)),
            active_fiber_id=str(data.get("active_fiber_id", "F001")),
            next_fiber_counter=max(stored_next_fib, next_fib),
            next_record_counter=max(stored_next_rec, next_rec),
            fiber_notes=dict(data.get("fiber_notes", {})),
            group_names={int(k): str(v) for k, v in data.get("group_names", {}).items()},
            manual_ranges=list(data.get("manual_ranges", [])),
        )

    def next_measurement_id(self) -> str:
        used = {r.measurement_id for r in self.records}
        while f"M{self.next_record_counter:04d}" in used:
            self.next_record_counter += 1
        m_id = f"M{self.next_record_counter:04d}"
        self.next_record_counter += 1
        return m_id

    def get_next_fiber_id(self) -> str:
        used_numbers = []
        for r in self.records:
            if r.fiber_id and r.fiber_id.startswith("F") and r.fiber_id[1:].isdigit():
                used_numbers.append(int(r.fiber_id[1:]))
        highest = max(used_numbers, default=0)
        self.next_fiber_counter = max(self.next_fiber_counter, highest + 1)
        fiber_id = f"F{self.next_fiber_counter:03d}"
        self.next_fiber_counter += 1
        return fiber_id

    def fiber_measurements(self, fiber_id: str) -> list[MeasurementRecord]:
        return [r for r in self.records if r.fiber_id == fiber_id and r.kind == MeasurementKind.PROJECTED_WIDTH]

    def accepted_fiber_measurements(self, fiber_id: str) -> list[MeasurementRecord]:
        return [r for r in self.records if r.fiber_id == fiber_id and r.kind == MeasurementKind.PROJECTED_WIDTH and r.is_included_in_statistics]

    def is_fiber_complete(self, fiber_id: str) -> bool:
        accepted_count = len(self.accepted_fiber_measurements(fiber_id))
        if self.target_sections <= 0:
            return accepted_count > 0
        return accepted_count >= self.target_sections

    def ensure_source_exists(self) -> Path:
        path = Path(self.image.path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
