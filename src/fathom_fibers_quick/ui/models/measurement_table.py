from __future__ import annotations

from datetime import datetime
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt

from ...application import ProjectSession
from ...measurement_records import MeasurementRecord, MeasurementStatus


def _format_value(record: MeasurementRecord) -> str:
    value = record.primary_value
    if value is None:
        return "—"
    if record.primary_unit == "m":
        if abs(value) < 1e-6:
            return f"{value * 1e9:.4g}"
        return f"{value * 1e6:.4g}"
    if record.primary_unit == "m²":
        return f"{value * 1e12:.4g}"
    return f"{value:.4g}"


class MeasurementTableModel(QAbstractTableModel):
    HEADERS = (
        "ID",
        "Name",
        "Kind",
        "Sample",
        "Image",
        "Fiber",
        "Primary value",
        "Unit",
        "Source",
        "Status",
        "Protocol",
        "Resolution",
        "Flags",
        "Tags",
        "Modified",
    )

    def __init__(self, session: ProjectSession) -> None:
        super().__init__()
        self.session = session

    @property
    def records(self) -> list[MeasurementRecord]:
        return self.session.project.records if self.session.project else []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.records)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def record_at(self, row: int) -> MeasurementRecord | None:
        return self.records[row] if 0 <= row < len(self.records) else None

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid() or not (record := self.record_at(index.row())):
            return None
        if role == Qt.ItemDataRole.UserRole:
            return record.measurement_id
        if role not in {Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole}:
            return None
        protocol = record.protocol_snapshot.get("protocol_id", "")
        resolution = next(
            (flag for flag in record.quality_flags if flag.startswith("RESOLUTION_")),
            "",
        )
        values = (
            record.measurement_id,
            record.name,
            record.kind.value,
            record.sample_id or "",
            record.image_id,
            record.fiber_id or "",
            _format_value(record),
            self._display_unit(record),
            record.source.value,
            record.status.value,
            protocol,
            resolution,
            ", ".join(record.quality_flags),
            ", ".join(record.tags),
            self._format_timestamp(record.updated_at),
        )
        return values[index.column()]

    @staticmethod
    def _display_unit(record: MeasurementRecord) -> str:
        if record.primary_unit == "m":
            value = record.primary_value
            return "nm" if value is not None and abs(value) < 1e-6 else "µm"
        if record.primary_unit == "m²":
            return "µm²"
        return record.primary_unit

    @staticmethod
    def _format_timestamp(value: str) -> str:
        try:
            return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return value

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        flags = super().flags(index)
        if index.column() in {1, 5, 9, 13}:
            flags |= Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value: Any, role=Qt.ItemDataRole.EditRole) -> bool:
        if role != Qt.ItemDataRole.EditRole or not (record := self.record_at(index.row())):
            return False
        changes: dict[str, Any]
        if index.column() == 1:
            changes = {"name": str(value).strip()}
        elif index.column() == 5:
            changes = {"fiber_id": str(value).strip() or None}
        elif index.column() == 9:
            try:
                changes = {"status": MeasurementStatus(str(value))}
            except ValueError:
                return False
        elif index.column() == 13:
            changes = {"tags": str(value).split(",")}
        else:
            return False
        self.session.update_metadata([record.measurement_id], **changes)
        self.refresh()
        return True

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class MeasurementFilterModel(QSortFilterProxyModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.search_text = ""
        self.kind = "All"
        self.status = "All"
        self.setDynamicSortFilter(True)

    def set_filters(self, search: str, kind: str, status: str) -> None:
        self.search_text = search.casefold().strip()
        self.kind = kind
        self.status = status
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self.sourceModel()
        record = model.record_at(source_row)
        if record is None:
            return False
        if self.kind != "All" and record.kind.value != self.kind:
            return False
        if self.status != "All" and record.status.value != self.status:
            return False
        if not self.search_text:
            return True
        haystack = " ".join(
            (
                record.measurement_id,
                record.name,
                record.kind.value,
                record.source.value,
                record.status.value,
                record.fiber_id or "",
                " ".join(record.tags),
                " ".join(record.quality_flags),
            )
        ).casefold()
        return self.search_text in haystack
