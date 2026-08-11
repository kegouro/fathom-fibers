from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel

from ...model import Project

NODE_ROLE = Qt.ItemDataRole.UserRole + 1
ID_ROLE = Qt.ItemDataRole.UserRole + 2


class ProjectTreeModel(QStandardItemModel):
    def __init__(self) -> None:
        super().__init__()
        self.setHorizontalHeaderLabels(["Project"])

    @staticmethod
    def _item(text: str, kind: str, object_id: str = "") -> QStandardItem:
        item = QStandardItem(text)
        item.setEditable(kind in {"sample", "fiber", "measurement", "roi"})
        item.setData(kind, NODE_ROLE)
        item.setData(object_id, ID_ROLE)
        return item

    def set_project(self, project: Project | None) -> None:
        self.clear()
        self.setHorizontalHeaderLabels(["Project"])
        if project is None:
            return
        root = self._item(
            Path(project.project_path).stem if project.project_path else "Unsaved Project",
            "project",
        )
        sample = self._item(project.sample_name, "sample", project.sample_id)
        image = self._item(Path(project.image.path).name, "image", project.image.path)
        fibers_root = self._item("Fibers", "folder")
        fiber_ids = sorted({r.fiber_id for r in project.records if r.fiber_id})
        for fiber_id in fiber_ids:
            fiber = self._item(str(fiber_id), "fiber", str(fiber_id))
            for record in project.records:
                if record.fiber_id == fiber_id:
                    fiber.appendRow(
                        self._item(
                            f"{record.measurement_id}  {record.name}",
                            "measurement",
                            record.measurement_id,
                        )
                    )
            fibers_root.appendRow(fiber)
        measurements_root = self._item("Unassigned measurements", "folder")
        for record in project.records:
            if not record.fiber_id:
                measurements_root.appendRow(
                    self._item(
                        f"{record.measurement_id}  {record.name}",
                        "measurement",
                        record.measurement_id,
                    )
                )
        rois_root = self._item("ROIs", "folder")
        roi_ids = sorted({r.roi_id for r in project.records if r.roi_id})
        for roi_id in roi_ids:
            rois_root.appendRow(self._item(str(roi_id), "roi", str(roi_id)))
        image.appendRow(fibers_root)
        image.appendRow(measurements_root)
        image.appendRow(rois_root)
        sample.appendRow(image)
        root.appendRow(sample)
        self.appendRow(root)

