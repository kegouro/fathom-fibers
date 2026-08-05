from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def get_utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class MeasurementKind(str, Enum):
    PROJECTED_WIDTH = "PROJECTED_WIDTH"
    DISTANCE = "DISTANCE"
    POLYLINE_LENGTH = "POLYLINE_LENGTH"
    ANGLE = "ANGLE"
    RECTANGLE_AREA = "RECTANGLE_AREA"
    POLYGON_AREA = "POLYGON_AREA"
    INTENSITY_PROFILE = "INTENSITY_PROFILE"


class MeasurementStatus(str, Enum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    MANUALLY_EDITED = "MANUALLY_EDITED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_MEASURABLE = "NOT_MEASURABLE"


class MeasurementSource(str, Enum):
    MANUAL = "MANUAL"
    ASSISTED = "ASSISTED"
    AUTO_ROI_COMPONENT = "AUTO_ROI_COMPONENT"
    IMPORTED = "IMPORTED"


@dataclass
class MeasurementRecord:
    measurement_id: str
    kind: MeasurementKind
    name: str
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    status: MeasurementStatus = MeasurementStatus.ACCEPTED
    source: MeasurementSource = MeasurementSource.MANUAL

    image_id: str = ""
    sample_id: str | None = None
    fiber_id: str | None = None
    roi_id: str | None = None
    group: int | None = None
    defect: str = "None"

    # Geometry dictionary: e.g. {"p1": (x,y), "p2": (x,y)}, {"points": [(x,y), ...]}, etc.
    geometry: dict[str, Any] = field(default_factory=dict)
    # Derived values dictionary: e.g. {"length_m": float, "area_m2": float, "angle_deg": float, ...}
    values: dict[str, Any] = field(default_factory=dict)
    display_units: dict[str, str] = field(default_factory=dict)

    calibration_snapshot: dict[str, Any] = field(default_factory=dict)
    quality_flags: list[str] = field(default_factory=list)

    confidence: float | None = None
    created_at: str = field(default_factory=get_utc_now_iso)
    updated_at: str = field(default_factory=get_utc_now_iso)

    @classmethod
    def create_legacy(
        cls,
        measurement_id: str,
        fiber_id: str,
        p1: tuple[float, float],
        p2: tuple[float, float],
        width_m: float,
        method: str,
        created_at: str | None = None,
        defect: str = "None",
        note: str = "",
        group: int | None = None,
        confidence: float | None = None,
        accepted: bool = True,
    ) -> MeasurementRecord:
        status = MeasurementStatus.ACCEPTED if accepted else MeasurementStatus.REJECTED
        source = MeasurementSource.MANUAL
        if "ASSISTED" in str(method).upper():
            source = MeasurementSource.ASSISTED
        elif "AUTO" in str(method).upper():
            source = MeasurementSource.AUTO_ROI_COMPONENT

        rec = cls(
            measurement_id=measurement_id,
            kind=MeasurementKind.PROJECTED_WIDTH,
            name=f"Ancho {measurement_id}",
            status=status,
            source=source,
            fiber_id=fiber_id,
            group=group,
            defect=defect,
            notes=note,
            confidence=confidence,
            geometry={"p1": tuple(p1), "p2": tuple(p2)},
            values={"width_m": float(width_m), "length_m": float(width_m)},
        )
        if created_at:
            rec.created_at = created_at
        return rec

    @property
    def primary_value(self) -> float | None:
        """Returns the primary physical scalar for display & summary."""
        if self.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
            val = self.values.get("length_m") or self.values.get("width_m")
            return float(val) if val is not None else None
        elif self.kind == MeasurementKind.POLYLINE_LENGTH:
            val = self.values.get("total_length_m")
            return float(val) if val is not None else None
        elif self.kind == MeasurementKind.ANGLE:
            val = self.values.get("interior_angle_deg")
            return float(val) if val is not None else None
        elif self.kind in {MeasurementKind.RECTANGLE_AREA, MeasurementKind.POLYGON_AREA}:
            val = self.values.get("area_m2")
            return float(val) if val is not None else None
        elif self.kind == MeasurementKind.INTENSITY_PROFILE:
            val = self.values.get("length_m")
            return float(val) if val is not None else None
        return None

    @property
    def primary_unit(self) -> str:
        if self.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE, MeasurementKind.POLYLINE_LENGTH, MeasurementKind.INTENSITY_PROFILE}:
            return "m"
        elif self.kind == MeasurementKind.ANGLE:
            return "deg"
        elif self.kind in {MeasurementKind.RECTANGLE_AREA, MeasurementKind.POLYGON_AREA}:
            return "m²"
        return ""

    @property
    def is_included_in_statistics(self) -> bool:
        """Rule 12: Only ACCEPTED or MANUALLY_EDITED records enter primary scientific summaries."""
        return self.status in {MeasurementStatus.ACCEPTED, MeasurementStatus.MANUALLY_EDITED}

    # Backward compatibility properties for legacy Measurement code
    @property
    def width_m(self) -> float:
        val = self.primary_value
        return val if val is not None else 0.0

    @width_m.setter
    def width_m(self, value: float) -> None:
        self.values["width_m"] = value
        self.values["length_m"] = value

    @property
    def p1(self) -> tuple[float, float]:
        if "p1" in self.geometry:
            pt = self.geometry["p1"]
            return (float(pt[0]), float(pt[1]))
        elif "points" in self.geometry and len(self.geometry["points"]) > 0:
            pt = self.geometry["points"][0]
            return (float(pt[0]), float(pt[1]))
        return (0.0, 0.0)

    @p1.setter
    def p1(self, pt: tuple[float, float]) -> None:
        self.geometry["p1"] = (float(pt[0]), float(pt[1]))

    @property
    def p2(self) -> tuple[float, float]:
        if "p2" in self.geometry:
            pt = self.geometry["p2"]
            return (float(pt[0]), float(pt[1]))
        elif "points" in self.geometry and len(self.geometry["points"]) > 1:
            pt = self.geometry["points"][-1]
            return (float(pt[0]), float(pt[1]))
        return (0.0, 0.0)

    @p2.setter
    def p2(self, pt: tuple[float, float]) -> None:
        self.geometry["p2"] = (float(pt[0]), float(pt[1]))

    @property
    def center(self) -> tuple[float, float]:
        if "center" in self.values:
            c = self.values["center"]
            return (float(c[0]), float(c[1]))
        p1, p2 = self.p1, self.p2
        return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    @property
    def accepted(self) -> bool:
        return self.status in {MeasurementStatus.ACCEPTED, MeasurementStatus.MANUALLY_EDITED}

    @accepted.setter
    def accepted(self, flag: bool) -> None:
        if flag:
            self.status = MeasurementStatus.ACCEPTED
        else:
            self.status = MeasurementStatus.REJECTED

    @property
    def method(self) -> str:
        return self.source.value if isinstance(self.source, MeasurementSource) else str(self.source)

    @method.setter
    def method(self, val: str) -> None:
        if "ASSISTED" in val.upper():
            self.source = MeasurementSource.ASSISTED
        elif "AUTO" in val.upper():
            self.source = MeasurementSource.AUTO_ROI_COMPONENT
        elif "IMPORT" in val.upper():
            self.source = MeasurementSource.IMPORTED
        else:
            self.source = MeasurementSource.MANUAL

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "kind": self.kind.value if isinstance(self.kind, MeasurementKind) else str(self.kind),
            "name": self.name,
            "tags": list(self.tags),
            "notes": self.notes,
            "status": self.status.value if isinstance(self.status, MeasurementStatus) else str(self.status),
            "source": self.source.value if isinstance(self.source, MeasurementSource) else str(self.source),
            "image_id": self.image_id,
            "sample_id": self.sample_id,
            "fiber_id": self.fiber_id,
            "roi_id": self.roi_id,
            "group": self.group,
            "defect": self.defect,
            "geometry": self.geometry,
            "values": self.values,
            "display_units": self.display_units,
            "calibration_snapshot": self.calibration_snapshot,
            "quality_flags": list(self.quality_flags),
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementRecord:
        kind_raw = data.get("kind", "PROJECTED_WIDTH")
        try:
            kind = MeasurementKind(kind_raw)
        except ValueError:
            kind = MeasurementKind.PROJECTED_WIDTH

        status_raw = data.get("status", "ACCEPTED" if data.get("accepted", True) else "REJECTED")
        try:
            status = MeasurementStatus(status_raw)
        except ValueError:
            status = MeasurementStatus.ACCEPTED if data.get("accepted", True) else MeasurementStatus.REJECTED

        source_raw = data.get("source", data.get("method", "MANUAL"))
        if "ASSISTED" in str(source_raw).upper():
            source = MeasurementSource.ASSISTED
        elif "AUTO" in str(source_raw).upper():
            source = MeasurementSource.AUTO_ROI_COMPONENT
        elif "IMPORT" in str(source_raw).upper():
            source = MeasurementSource.IMPORTED
        else:
            source = MeasurementSource.MANUAL

        geom = data.get("geometry", {})
        if "p1" not in geom and "p1" in data:
            geom["p1"] = data["p1"]
        if "p2" not in geom and "p2" in data:
            geom["p2"] = data["p2"]

        vals = data.get("values", {})
        if "width_m" not in vals and "width_m" in data:
            vals["width_m"] = data["width_m"]
            vals["length_m"] = data["width_m"]

        m_id = data.get("measurement_id", "M001")
        name = data.get("name", f"Medición {m_id}")

        return cls(
            measurement_id=m_id,
            kind=kind,
            name=name,
            tags=list(data.get("tags", [])),
            notes=data.get("notes", ""),
            status=status,
            source=source,
            image_id=data.get("image_id", ""),
            sample_id=data.get("sample_id"),
            fiber_id=data.get("fiber_id"),
            roi_id=data.get("roi_id"),
            group=data.get("group"),
            defect=data.get("defect", "None"),
            geometry=geom,
            values=vals,
            display_units=data.get("display_units", {}),
            calibration_snapshot=data.get("calibration_snapshot", {}),
            quality_flags=list(data.get("quality_flags", [])),
            confidence=data.get("confidence"),
            created_at=data.get("created_at", get_utc_now_iso()),
            updated_at=data.get("updated_at", get_utc_now_iso()),
        )


def normalize_tags(tags: Sequence[str]) -> list[str]:
    """Cleans, trims, and deduplicates tags case-insensitively while preserving order."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in tags:
        if not t:
            continue
        # Split comma-separated string if passed
        parts = str(t).split(",")
        for p in parts:
            item = p.strip()
            if item and item.lower() not in seen:
                seen.add(item.lower())
                cleaned.append(item)
    return cleaned
