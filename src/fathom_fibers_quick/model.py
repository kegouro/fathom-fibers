from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


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


@dataclass(slots=True)
class Measurement:
    measurement_id: str
    fiber_id: str
    p1: tuple[float, float]
    p2: tuple[float, float]
    width_m: float
    method: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    defect: str = "None"
    note: str = ""
    group: int | None = None
    confidence: float | None = None
    accepted: bool = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.p1[0]) and math.isfinite(self.p1[1])):
            raise ValueError(f"Invalid p1 coordinates: {self.p1}")
        if not (math.isfinite(self.p2[0]) and math.isfinite(self.p2[1])):
            raise ValueError(f"Invalid p2 coordinates: {self.p2}")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.p1[0] + self.p2[0]) / 2, (self.p1[1] + self.p2[1]) / 2)



@dataclass(slots=True)
class Project:
    schema_version: int
    image: ImageDocument
    measurements: list[Measurement] = field(default_factory=list)
    project_path: str | None = None
    notes: str = ""
    target_sections: int = 5
    active_fiber_id: str = "F001"
    next_fiber_counter: int = 1
    fiber_notes: dict[str, str] = field(default_factory=dict)
    group_names: dict[int, str] = field(default_factory=dict)
    manual_ranges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        image_data = dict(data["image"])
        calibration = Calibration(**image_data.pop("calibration"))
        if image_data.get("footer_bounds") is not None:
            image_data["footer_bounds"] = tuple(image_data["footer_bounds"])
        image = ImageDocument(calibration=calibration, **image_data)
        measurements = []
        for item in data.get("measurements", []):
            item = dict(item)
            item["p1"] = tuple(item["p1"])
            item["p2"] = tuple(item["p2"])
            measurements.append(Measurement(**item))

        # Recover or compute next_fiber_counter
        used_numbers = []
        for m in measurements:
            if m.fiber_id.startswith("F") and m.fiber_id[1:].isdigit():
                used_numbers.append(int(m.fiber_id[1:]))
        computed_next = max(used_numbers, default=0) + 1
        stored_next = int(data.get("next_fiber_counter", computed_next))

        return cls(
            schema_version=int(data.get("schema_version", 1)),
            image=image,
            measurements=measurements,
            project_path=data.get("project_path"),
            notes=data.get("notes", ""),
            target_sections=int(data.get("target_sections", 5)),
            active_fiber_id=str(data.get("active_fiber_id", "F001")),
            next_fiber_counter=max(stored_next, computed_next),
            fiber_notes=dict(data.get("fiber_notes", {})),
            group_names={int(k): str(v) for k, v in data.get("group_names", {}).items()},
            manual_ranges=list(data.get("manual_ranges", [])),
        )

    def next_measurement_id(self) -> str:
        used = {m.measurement_id for m in self.measurements}
        index = 1
        while f"M{index:04d}" in used:
            index += 1
        return f"M{index:04d}"

    def get_next_fiber_id(self) -> str:
        used_numbers = []
        for m in self.measurements:
            if m.fiber_id.startswith("F") and m.fiber_id[1:].isdigit():
                used_numbers.append(int(m.fiber_id[1:]))
        highest = max(used_numbers, default=0)
        self.next_fiber_counter = max(self.next_fiber_counter, highest + 1)
        fiber_id = f"F{self.next_fiber_counter:03d}"
        self.next_fiber_counter += 1
        return fiber_id

    def fiber_measurements(self, fiber_id: str) -> list[Measurement]:
        return [m for m in self.measurements if m.fiber_id == fiber_id]

    def accepted_fiber_measurements(self, fiber_id: str) -> list[Measurement]:
        return [m for m in self.measurements if m.fiber_id == fiber_id and m.accepted]

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
