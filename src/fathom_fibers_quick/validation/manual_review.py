from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class GridCellStatus(str, Enum):
    NOT_REVIEWED = "NOT_REVIEWED"
    MEASURED = "MEASURED"
    NO_VALID_FIBER = "NO_VALID_FIBER"
    SKIPPED_WITH_REASON = "SKIPPED_WITH_REASON"


@dataclass(slots=True)
class ManualGridCell:
    row: int
    column: int
    status: GridCellStatus = GridCellStatus.NOT_REVIEWED
    fiber_id: str | None = None
    measurement_id: str | None = None
    geometry: dict[str, Any] | None = None
    diameter: float | None = None
    unit: str | None = None
    calibration_snapshot: dict[str, Any] | None = None
    operator: str | None = None
    timestamp: str | None = None
    notes: str = ""

    @property
    def position(self) -> str:
        return f"R{self.row + 1}C{self.column + 1}"

    def set_status(self, status: GridCellStatus, *, notes: str = "") -> None:
        if status == GridCellStatus.SKIPPED_WITH_REASON and not notes.strip():
            raise ValueError("SKIPPED_WITH_REASON requires notes")
        self.status = status
        self.notes = notes
        self.timestamp = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["position"] = self.position
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ManualGridCell:
        fields = {key: value for key, value in payload.items() if key not in {"status", "position"}}
        return cls(**fields, status=GridCellStatus(payload.get("status", "NOT_REVIEWED")))


@dataclass(slots=True)
class Manual5x5Review:
    case_id: str
    protocol_id: str = "MANUAL_5X5_REFERENCE"
    cells: list[ManualGridCell] = field(
        default_factory=lambda: [
            ManualGridCell(row, column) for row in range(5) for column in range(5)
        ]
    )

    def __post_init__(self) -> None:
        if len(self.cells) != 25:
            raise ValueError("MANUAL_5X5_REFERENCE requires exactly 25 cells")

    @property
    def completed_count(self) -> int:
        return sum(cell.status != GridCellStatus.NOT_REVIEWED for cell in self.cells)

    @property
    def measurement_count(self) -> int:
        return sum(cell.status == GridCellStatus.MEASURED for cell in self.cells)

    def cell(self, row: int, column: int) -> ManualGridCell:
        if not (0 <= row < 5 and 0 <= column < 5):
            raise IndexError("grid position must be within the 5x5 reference grid")
        return self.cells[row * 5 + column]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "protocol_id": self.protocol_id,
            "cells": [cell.to_dict() for cell in self.cells],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Manual5x5Review:
        return cls(
            case_id=payload["case_id"],
            protocol_id=payload.get("protocol_id", "MANUAL_5X5_REFERENCE"),
            cells=[ManualGridCell.from_dict(cell) for cell in payload["cells"]],
        )
