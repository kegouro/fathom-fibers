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
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            image=image,
            measurements=measurements,
            project_path=data.get("project_path"),
            notes=data.get("notes", ""),
        )

    def next_measurement_id(self) -> str:
        used = {m.measurement_id for m in self.measurements}
        index = 1
        while f"M{index:04d}" in used:
            index += 1
        return f"M{index:04d}"

    def ensure_source_exists(self) -> Path:
        path = Path(self.image.path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
