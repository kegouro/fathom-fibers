from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .measurement_records import MeasurementKind, MeasurementStatus


def get_utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class MeasurementProtocol:
    protocol_id: str
    name: str
    description: str
    measurement_kind: MeasurementKind = MeasurementKind.PROJECTED_WIDTH
    sections_per_fiber: int = 5
    minimum_fibers_per_image: int = 10
    exclude_crossings: bool = True
    exclude_image_edges: bool = True
    exclude_invalid_mask: bool = True
    minimum_resolved_width_px: float = 2.5
    allowed_statuses: list[MeasurementStatus] = field(
        default_factory=lambda: [MeasurementStatus.ACCEPTED, MeasurementStatus.MANUALLY_EDITED]
    )
    required_tags: list[str] = field(default_factory=list)
    notes_template: str = ""
    created_at: str = field(default_factory=get_utc_now_iso)
    updated_at: str = field(default_factory=get_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "name": self.name,
            "description": self.description,
            "measurement_kind": self.measurement_kind.value if hasattr(self.measurement_kind, "value") else str(self.measurement_kind),
            "sections_per_fiber": self.sections_per_fiber,
            "minimum_fibers_per_image": self.minimum_fibers_per_image,
            "exclude_crossings": self.exclude_crossings,
            "exclude_image_edges": self.exclude_image_edges,
            "exclude_invalid_mask": self.exclude_invalid_mask,
            "minimum_resolved_width_px": self.minimum_resolved_width_px,
            "allowed_statuses": [st.value if hasattr(st, "value") else str(st) for st in self.allowed_statuses],
            "required_tags": list(self.required_tags),
            "notes_template": self.notes_template,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementProtocol:
        kind_raw = data.get("measurement_kind", "PROJECTED_WIDTH")
        try:
            kind = MeasurementKind(kind_raw)
        except ValueError:
            kind = MeasurementKind.PROJECTED_WIDTH

        statuses = []
        for s in data.get("allowed_statuses", ["ACCEPTED", "MANUALLY_EDITED"]):
            try:
                statuses.append(MeasurementStatus(s))
            except ValueError:
                pass
        if not statuses:
            statuses = [MeasurementStatus.ACCEPTED, MeasurementStatus.MANUALLY_EDITED]

        return cls(
            protocol_id=str(data.get("protocol_id", "PROTO_001")),
            name=str(data.get("name", "Protocolo Personalizado")),
            description=str(data.get("description", "")),
            measurement_kind=kind,
            sections_per_fiber=int(data.get("sections_per_fiber", 5)),
            minimum_fibers_per_image=int(data.get("minimum_fibers_per_image", 10)),
            exclude_crossings=bool(data.get("exclude_crossings", True)),
            exclude_image_edges=bool(data.get("exclude_image_edges", True)),
            exclude_invalid_mask=bool(data.get("exclude_invalid_mask", True)),
            minimum_resolved_width_px=float(data.get("minimum_resolved_width_px", 2.5)),
            allowed_statuses=statuses,
            required_tags=list(data.get("required_tags", [])),
            notes_template=str(data.get("notes_template", "")),
            created_at=str(data.get("created_at", get_utc_now_iso())),
            updated_at=str(data.get("updated_at", get_utc_now_iso())),
        )


PRESET_PVDF_3_SECTIONS = MeasurementProtocol(
    protocol_id="PVDF_3_SECTIONS",
    name="PVDF width — 3 sections",
    description="Protocolo estándar rápido para membranas electrohiladas de PVDF (3 secciones por fibra).",
    measurement_kind=MeasurementKind.PROJECTED_WIDTH,
    sections_per_fiber=3,
    minimum_fibers_per_image=10,
    exclude_crossings=True,
    exclude_image_edges=True,
    exclude_invalid_mask=True,
    minimum_resolved_width_px=2.5,
)

PRESET_PVDF_5_SECTIONS = MeasurementProtocol(
    protocol_id="PVDF_5_SECTIONS",
    name="PVDF width — 5 sections",
    description="Protocolo de alta precisión para caracterización de diámetro en fibras de PVDF (5 secciones por fibra).",
    measurement_kind=MeasurementKind.PROJECTED_WIDTH,
    sections_per_fiber=5,
    minimum_fibers_per_image=10,
    exclude_crossings=True,
    exclude_image_edges=True,
    exclude_invalid_mask=True,
    minimum_resolved_width_px=2.5,
)

PRESET_GENERAL_MICROSCOPY = MeasurementProtocol(
    protocol_id="GENERAL_MICROSCOPY",
    name="General microscopy measurement",
    description="Protocolo flexible para mediciones generales en micrografías SEM (distancia, áreas y ángulos).",
    measurement_kind=MeasurementKind.DISTANCE,
    sections_per_fiber=0,
    minimum_fibers_per_image=0,
    exclude_crossings=False,
    exclude_image_edges=False,
    exclude_invalid_mask=True,
    minimum_resolved_width_px=1.0,
)

PRESET_BLIND_REPEATABILITY = MeasurementProtocol(
    protocol_id="BLIND_REPEATABILITY",
    name="Blind repeatability",
    description="Protocolo para estudios ciegos de repetibilidad intra e inter-operador.",
    measurement_kind=MeasurementKind.PROJECTED_WIDTH,
    sections_per_fiber=3,
    minimum_fibers_per_image=5,
    exclude_crossings=True,
    exclude_image_edges=True,
    exclude_invalid_mask=True,
    minimum_resolved_width_px=2.5,
)

PRESET_SIMPOLY_MANUAL_5X5 = MeasurementProtocol(
    protocol_id="SIMPOLY_MANUAL_5X5",
    name="SIMPoly manual reference — 5×5",
    description="5×5 circular grid overlay (25 positions). Measure closest fully visible fiber perpendicular width.",
    sections_per_fiber=1,
    minimum_resolved_width_px=2.5,
    notes_template="SIMPoly 5x5 position {position_idx}",
)

BUILTIN_PROTOCOLS: dict[str, MeasurementProtocol] = {
    p.protocol_id: p
    for p in [
        PRESET_PVDF_3_SECTIONS,
        PRESET_PVDF_5_SECTIONS,
        PRESET_GENERAL_MICROSCOPY,
        PRESET_BLIND_REPEATABILITY,
        PRESET_SIMPOLY_MANUAL_5X5,
    ]
}
