"""Batched scientific overlay layers for the image viewer.

Overlays are display-only: no pixel data and no scientific results are
modified.  Large layers are rendered as single batched ``QPainterPath`` items
or subsampled so tens of thousands of samples never become individual Qt
graphics objects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsView

from ..core.contracts import ScientificImage
from ..core.methods import MethodId, MethodResult

DENSITY_STEP = {"Sparse": 9, "Medium": 5, "Dense": 1}
ORIENTATION_TICK_PX = 8.0


def build_overlay_payload(
    results: dict[MethodId, MethodResult],
    image: ScientificImage,
) -> dict[str, Any]:
    """Collect layer data from method results into viewer-friendly pixel data."""
    payload: dict[str, Any] = {}
    px = image.calibration.pixel_size_x_m
    py = image.calibration.pixel_size_y_m

    def roi_offset(result: MethodResult) -> tuple[float, float]:
        if result.valid_roi is not None:
            return float(result.valid_roi[0]), float(result.valid_roi[1])
        return 0.0, 0.0

    python = results.get(MethodId.PYTHON_SIMPOLY)
    if python is not None:
        x0, y0 = roi_offset(python)
        if python.mask is not None:
            payload["mask"] = {"array": np.asarray(python.mask, bool), "offset": (x0, y0)}
        if python.centerline is not None:
            payload["skeleton"] = {"array": np.asarray(python.centerline, bool), "offset": (x0, y0)}

    local = results.get(MethodId.FATHOM_LOCAL)
    if local is not None and local.local_samples is not None:
        samples = local.local_samples
        payload["local_sections"] = {
            "x0": samples.get("section_x0_px"),
            "y0": samples.get("section_y0_px"),
            "x1": samples.get("section_x1_px"),
            "y1": samples.get("section_y1_px"),
            "flags": samples.get("section_flags"),
        }

    field = results.get(MethodId.FATHOM_FIELD_GRAPH_V1)
    if field is not None:
        x0, y0 = roi_offset(field)
        if field.centerline is not None:
            payload["centerline"] = {"array": np.asarray(field.centerline, bool), "offset": (x0, y0)}
        if field.local_samples is not None:
            samples = field.local_samples
            x_px = np.asarray(samples["x_m"], float) / px
            y_px = np.asarray(samples["y_m"], float) / py
            accepted = np.asarray(samples["edge_accepted"], bool)
            flags = np.asarray(samples["edge_flags"], dtype=str) if "edge_flags" in samples else None
            rejected = ~accepted
            payload["field_positions"] = np.column_stack((x_px, y_px))
            payload["orientation"] = {
                "x": x_px,
                "y": y_px,
                "qx": np.asarray(samples["qx"], float),
                "qy": np.asarray(samples["qy"], float),
                "coherence": np.asarray(samples["coherence"], float),
            }
            if "minus_xy_m" in samples and "plus_xy_m" in samples:
                minus = np.asarray(samples["minus_xy_m"], float) / np.asarray((px, py))
                plus = np.asarray(samples["plus_xy_m"], float) / np.asarray((px, py))
                payload["edges"] = {
                    "x1": minus[:, 0],
                    "y1": minus[:, 1],
                    "x2": plus[:, 0],
                    "y2": plus[:, 1],
                    "accepted": accepted,
                    "flags": flags,
                }
                payload["rejected"] = {
                    "x": x_px[rejected],
                    "y": y_px[rejected],
                    "flags": flags[rejected] if flags is not None else None,
                }
            if "profile_minus_u_um" in samples and "profile_plus_u_um" in samples:
                normal = np.asarray(samples["normal_xy"], float)
                minus_u = np.asarray(samples["profile_minus_u_um"], float) * 1e-6
                plus_u = np.asarray(samples["profile_plus_u_um"], float) * 1e-6
                profile_accepted = np.asarray(samples["profile_accepted"], bool)
                minus_pos = np.column_stack((x_px, y_px)) + normal * (minus_u[:, None] / np.asarray((px, py)))
                plus_pos = np.column_stack((x_px, y_px)) + normal * (plus_u[:, None] / np.asarray((px, py)))
                payload["profile"] = {
                    "x1": minus_pos[:, 0],
                    "y1": minus_pos[:, 1],
                    "x2": plus_pos[:, 0],
                    "y2": plus_pos[:, 1],
                    "accepted": profile_accepted,
                }
    return payload


class OverlayLayers:
    """Owns all scientific overlay graphics items of one viewer scene."""

    LAYER_NAMES = (
        "mask",
        "skeleton",
        "centerline",
        "orientation",
        "edges",
        "profile",
        "rejected",
        "local_sections",
    )

    def __init__(self, view: QGraphicsView) -> None:
        self.view = view
        self.payload: dict[str, Any] = {}
        self.density = "Medium"
        self.visible: set[str] = set()
        self._items: dict[str, list[QGraphicsItem]] = {}
        self._z = {
            "mask": -10,
            "skeleton": 5,
            "centerline": 6,
            "local_sections": 7,
            "orientation": 8,
            "edges": 9,
            "profile": 10,
            "rejected": 11,
        }

    def set_payload(self, payload: dict[str, Any]) -> None:
        self.payload = payload or {}
        self.rebuild_all()

    def set_density(self, density: str) -> None:
        self.density = density if density in DENSITY_STEP else "Medium"
        self.rebuild_layer("orientation")

    def set_visible(self, layer: str, visible: bool) -> None:
        if visible:
            self.visible.add(layer)
        else:
            self.visible.discard(layer)
        self._apply_visibility(layer)

    def rebuild_all(self) -> None:
        for layer in self.LAYER_NAMES:
            self.rebuild_layer(layer)

    def clear(self) -> None:
        self.payload = {}
        self.rebuild_all()

    def rebuild_layer(self, layer: str) -> None:
        for item in self._items.pop(layer, []):
            self.view.scene().removeItem(item)
        if not self.visible or layer not in self.visible:
            return
        items = self._build_layer(layer)
        if items:
            self.view.scene().addItem(items)
        self._items[layer] = [items] if items else []

    def _apply_visibility(self, layer: str) -> None:
        self.rebuild_layer(layer)

    def _build_layer(self, layer: str) -> QGraphicsItem | None:
        data = self.payload.get(layer)
        if data is None:
            return None
        if layer == "mask":
            return self._build_mask(data)
        if layer in {"skeleton", "centerline"}:
            return self._build_bool_path(data, QColor(63, 193, 243, 200) if layer == "skeleton" else QColor(255, 216, 102, 210))
        if layer == "local_sections":
            return self._build_sections(data)
        if layer == "orientation":
            return self._build_orientation(data)
        if layer == "edges":
            return self._build_edges(data)
        if layer == "profile":
            return self._build_profile(data)
        if layer == "rejected":
            return self._build_rejected(data)
        return None

    @staticmethod
    def _offset(data: dict[str, Any]) -> tuple[float, float]:
        return tuple(float(value) for value in data.get("offset", (0.0, 0.0)))

    def _build_mask(self, data: dict[str, Any]) -> QGraphicsPixmapItem:
        binary = np.asarray(data["array"], bool)
        overlay = np.zeros((binary.shape[0], binary.shape[1], 4), dtype=np.uint8)
        overlay[binary, 0] = 255
        overlay[binary, 1] = 166
        overlay[binary, 2] = 0
        overlay[binary, 3] = 70
        height, width = binary.shape
        image = QImage(
            overlay.data, width, height, overlay.strides[0], QImage.Format.Format_RGBA8888
        ).copy()
        x0, y0 = self._offset(data)
        item = QGraphicsPixmapItem(QPixmap.fromImage(image))
        item.setPos(x0, y0)
        item.setZValue(self._z["mask"])
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        return item

    def _build_bool_path(self, data: dict[str, Any], color: QColor) -> QGraphicsPathItem:
        binary = np.asarray(data["array"], bool)
        x0, y0 = self._offset(data)
        rows, cols = np.nonzero(binary)
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        for row, col in zip(rows, cols, strict=False):
            if row > 0 and binary[row - 1, col]:
                continue
            path.moveTo(col + x0, row + y0)
            path.lineTo(col + 1.0 + x0, row + y0)
        pen = QPen(color, 1.0)
        pen.setCosmetic(True)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self._z["skeleton"])
        return item

    def _build_sections(self, data: dict[str, Any]) -> QGraphicsPathItem:
        x0 = np.asarray(data.get("x0"), float)
        path = QPainterPath()
        if x0.size == 0:
            return self._path_item(path, QColor("#f0a83a"))
        for index in range(x0.size):
            path.moveTo(float(data["x0"][index]), float(data["y0"][index]))
            path.lineTo(float(data["x1"][index]), float(data["y1"][index]))
        return self._path_item(path, QColor("#f0a83a"))

    def _path_item(self, path: QPainterPath, color: QColor, width: float = 1.4) -> QGraphicsPathItem:
        pen = QPen(color, width)
        pen.setCosmetic(True)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self._z["local_sections"])
        return item

    def _build_orientation(self, data: dict[str, Any]) -> QGraphicsPathItem:
        x = np.asarray(data["x"], float)
        y = np.asarray(data["y"], float)
        qx = np.asarray(data["qx"], float)
        qy = np.asarray(data["qy"], float)
        step = DENSITY_STEP.get(self.density, 5)
        selected = np.zeros(x.size, bool)
        selected[::step] = True
        ticks = ORIENTATION_TICK_PX * 0.5
        theta = 0.5 * np.arctan2(qy, qx)
        dx = np.cos(theta) * ticks
        dy = np.sin(theta) * ticks
        path = QPainterPath()
        for index in np.flatnonzero(selected):
            path.moveTo(x[index] - dx[index], y[index] - dy[index])
            path.lineTo(x[index] + dx[index], y[index] + dy[index])
        pen = QPen(QColor(120, 220, 160, 190), 1.0)
        pen.setCosmetic(True)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self._z["orientation"])
        return item

    def _build_edges(self, data: dict[str, Any]) -> QGraphicsPathItem:
        path = QPainterPath()
        for index in range(len(np.asarray(data["x1"]))):
            path.moveTo(float(data["x1"][index]), float(data["y1"][index]))
            path.lineTo(float(data["x2"][index]), float(data["y2"][index]))
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(84, 224, 160, 230), 1.1)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(self._z["edges"])
        return item

    def _build_profile(self, data: dict[str, Any]) -> QGraphicsPathItem:
        accepted = np.asarray(data.get("accepted", True), bool)
        path = QPainterPath()
        for index in range(len(np.asarray(data["x1"]))):
            if not accepted[index]:
                continue
            path.moveTo(float(data["x1"][index]), float(data["y1"][index]))
            path.lineTo(float(data["x2"][index]), float(data["y2"][index]))
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(214, 140, 255, 230), 1.1)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(self._z["profile"])
        return item

    def _build_rejected(self, data: dict[str, Any]) -> QGraphicsPathItem:
        x = np.asarray(data["x"], float)
        y = np.asarray(data["y"], float)
        step = DENSITY_STEP.get(self.density, 5)
        path = QPainterPath()
        for index in range(0, x.size, step):
            path.moveTo(x[index] - 2.0, y[index] - 2.0)
            path.lineTo(x[index] + 2.0, y[index] + 2.0)
            path.moveTo(x[index] + 2.0, y[index] - 2.0)
            path.lineTo(x[index] - 2.0, y[index] + 2.0)
        pen = QPen(QColor(213, 107, 107, 220), 1.0)
        pen.setCosmetic(True)
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        item.setZValue(self._z["rejected"])
        return item

    def hit_test_field_sample(self, scene_point: QPointF, radius: float = 30.0) -> int | None:
        positions = self.payload.get("field_positions")
        if positions is None or not positions.size:
            return None
        delta = positions - np.asarray((scene_point.x(), scene_point.y()))
        distances = np.hypot(delta[:, 0], delta[:, 1])
        index = int(np.argmin(distances))
        return index if distances[index] <= radius else None

    def sample_position_px(self, index: int) -> QPointF | None:
        positions = self.payload.get("field_positions")
        if positions is None or not 0 <= index < positions.shape[0]:
            return None
        return QPointF(float(positions[index, 0]), float(positions[index, 1]))

    def selection_rect(self, index: int) -> QRectF | None:
        point = self.sample_position_px(index)
        if point is None:
            return None
        return QRectF(point.x() - 8.0, point.y() - 8.0, 16.0, 16.0)
