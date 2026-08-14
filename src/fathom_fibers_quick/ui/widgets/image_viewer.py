from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
)

from ...core.contracts import ScientificImage
from ...measurement_records import MeasurementKind, MeasurementRecord, MeasurementStatus
from ..tools import ToolController


def _display_u8(
    gray: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
    inverted: bool,
) -> np.ndarray:
    array = np.asarray(gray, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        normalized = np.zeros_like(array)
    else:
        low, high = np.percentile(finite, (0.5, 99.5))
        normalized = np.clip((array - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
    normalized = np.clip((normalized - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)
    normalized = np.power(normalized, 1.0 / max(gamma, 0.05))
    if inverted:
        normalized = 1.0 - normalized
    return np.ascontiguousarray(np.round(normalized * 255.0).astype(np.uint8))


class EditableLineItem(QGraphicsLineItem):
    def __init__(
        self,
        record_id: str,
        p1: tuple[float, float],
        p2: tuple[float, float],
        callback: Callable[[str, dict[str, Any]], None],
    ) -> None:
        super().__init__(p1[0], p1[1], p2[0], p2[1])
        self.record_id = record_id
        self._callback = callback
        self.setData(0, record_id)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        delta = self.pos()
        if delta.manhattanLength() <= 0:
            return
        line = self.line()
        geometry = {
            "p1": (line.x1() + delta.x(), line.y1() + delta.y()),
            "p2": (line.x2() + delta.x(), line.y2() + delta.y()),
        }
        self.setPos(0.0, 0.0)
        self._callback(self.record_id, geometry)


class ScientificImageView(QGraphicsView):
    coordinateChanged = Signal(float, float, object, float, float)
    measurementRequested = Signal(str, dict)
    recordSelected = Signal(object)
    geometryEdited = Signal(str, dict)
    roiDrawn = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor("#17191c"))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.image: ScientificImage | None = None
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setZValue(-100)
        self.scene().addItem(self._pixmap_item)
        self._footer_item = QGraphicsRectItem()
        self._footer_item.setBrush(QColor(180, 40, 40, 45))
        footer_pen = QPen(QColor(220, 90, 90, 180))
        footer_pen.setCosmetic(True)
        footer_pen.setStyle(Qt.PenStyle.DashLine)
        self._footer_item.setPen(footer_pen)
        self._footer_item.setZValue(-20)
        self.scene().addItem(self._footer_item)
        self._footer_item.hide()

        self._overlay_items: list[QGraphicsItem] = []
        self._preview_items: list[QGraphicsItem] = []
        self._scale_items: list[QGraphicsItem] = []
        self._records: list[MeasurementRecord] = []
        self._selected_id: str | None = None
        self._footer_visible = True
        self._brightness = 0.0
        self._contrast = 1.0
        self._gamma = 1.0
        self._inverted = False
        self._space_pan = False
        self._last_pan: QPoint | None = None

        self.tools = ToolController()
        self.tools.committed.connect(self._tool_committed)
        self.tools.previewChanged.connect(self._show_preview)
        self.scene().selectionChanged.connect(self._scene_selection_changed)

    def set_image(self, image: ScientificImage | None) -> None:
        self.image = image
        self._records = []
        self._selected_id = None
        if image is None:
            self._pixmap_item.setPixmap(QPixmap())
            self.scene().setSceneRect(QRectF())
            return
        self._refresh_pixmap()
        height, width = image.shape
        self.scene().setSceneRect(0.0, 0.0, float(width), float(height))
        if image.footer_bounds:
            y0, y1 = image.footer_bounds
            self._footer_item.setRect(0.0, float(y0), float(width), float(y1 - y0))
            self._footer_item.show()
        else:
            self._footer_item.hide()
        self._update_scene_body()
        self._update_scale_bar()
        self.fit_to_window()

    def set_display_adjustments(
        self,
        *,
        brightness: float | None = None,
        contrast: float | None = None,
        gamma: float | None = None,
        inverted: bool | None = None,
    ) -> None:
        if brightness is not None:
            self._brightness = float(brightness)
        if contrast is not None:
            self._contrast = float(contrast)
        if gamma is not None:
            self._gamma = float(gamma)
        if inverted is not None:
            self._inverted = bool(inverted)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self.image is None:
            return
        display = _display_u8(
            self.image.gray,
            self._brightness,
            self._contrast,
            self._gamma,
            self._inverted,
        )
        height, width = display.shape
        qimage = QImage(
            display.data,
            width,
            height,
            display.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))

    def set_footer_visible(self, visible: bool) -> None:
        self._footer_visible = visible
        self._update_scene_body()

    def _update_scene_body(self) -> None:
        if self.image is None:
            return
        height, width = self.image.shape
        shown_height = height
        if not self._footer_visible and self.image.footer_bounds:
            shown_height = self.image.footer_bounds[0]
        self.scene().setSceneRect(0.0, 0.0, float(width), float(shown_height))
        self._footer_item.setVisible(self._footer_visible and self.image.footer_bounds is not None)

    def set_records(self, records: Sequence[MeasurementRecord], selected_id: str | None) -> None:
        self._records = list(records)
        self._selected_id = selected_id
        for item in self._overlay_items:
            self.scene().removeItem(item)
        self._overlay_items.clear()
        for record in self._records:
            self._add_record_overlay(record)

    def set_selected_record(self, record_id: str | None) -> None:
        self._selected_id = record_id
        for item in self._overlay_items:
            if item.data(0) is not None:
                item.setSelected(item.data(0) == record_id)
        if record_id:
            self.focus_record(record_id)

    def _record_pen(self, record: MeasurementRecord) -> QPen:
        colors = {
            MeasurementStatus.ACCEPTED: QColor("#33b67a"),
            MeasurementStatus.MANUALLY_EDITED: QColor("#f0a83a"),
            MeasurementStatus.PROPOSED: QColor("#40bfe8"),
            MeasurementStatus.REJECTED: QColor("#8a8f98"),
            MeasurementStatus.AMBIGUOUS: QColor("#d68cff"),
            MeasurementStatus.NOT_MEASURABLE: QColor("#d56b6b"),
        }
        pen = QPen(colors.get(record.status, QColor("white")), 2.0)
        pen.setCosmetic(True)
        if record.status == MeasurementStatus.PROPOSED:
            pen.setStyle(Qt.PenStyle.DashLine)
        elif record.status in {MeasurementStatus.REJECTED, MeasurementStatus.NOT_MEASURABLE}:
            pen.setStyle(Qt.PenStyle.DotLine)
        return pen

    def _configure_overlay(self, item: QGraphicsItem, record: MeasurementRecord) -> None:
        item.setData(0, record.measurement_id)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        item.setZValue(10)
        self.scene().addItem(item)
        self._overlay_items.append(item)
        item.setSelected(record.measurement_id == self._selected_id)

    def _add_record_overlay(self, record: MeasurementRecord) -> None:
        pen = self._record_pen(record)
        geometry = record.geometry
        if record.kind in {
            MeasurementKind.PROJECTED_WIDTH,
            MeasurementKind.DISTANCE,
            MeasurementKind.INTENSITY_PROFILE,
        } and "p1" in geometry and "p2" in geometry:
            item = EditableLineItem(
                record.measurement_id,
                tuple(geometry["p1"]),
                tuple(geometry["p2"]),
                lambda record_id, geom: self.geometryEdited.emit(record_id, geom),
            )
            item.setPen(pen)
            self._configure_overlay(item, record)
        elif record.kind == MeasurementKind.POLYLINE_LENGTH:
            self._add_path(record, geometry.get("points", []), pen, closed=False)
        elif record.kind == MeasurementKind.ANGLE:
            points = [geometry.get("pt_a"), geometry.get("pt_b"), geometry.get("pt_c")]
            self._add_path(record, [p for p in points if p is not None], pen, closed=False)
        elif record.kind == MeasurementKind.RECTANGLE_AREA and "bbox" in geometry:
            x0, y0, x1, y1 = geometry["bbox"]
            item = QGraphicsRectItem(float(x0), float(y0), float(x1 - x0), float(y1 - y0))
            item.setPen(pen)
            self._configure_overlay(item, record)
        elif record.kind == MeasurementKind.POLYGON_AREA:
            points = [QPointF(float(x), float(y)) for x, y in geometry.get("points", [])]
            item = QGraphicsPolygonItem(QPolygonF(points))
            item.setPen(pen)
            self._configure_overlay(item, record)

    def _add_path(
        self,
        record: MeasurementRecord,
        points: Sequence[tuple[float, float]],
        pen: QPen,
        *,
        closed: bool,
    ) -> None:
        if not points:
            return
        path = QPainterPath(QPointF(float(points[0][0]), float(points[0][1])))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        if closed:
            path.closeSubpath()
        item = QGraphicsPathItem(path)
        item.setPen(pen)
        self._configure_overlay(item, record)

    def _show_preview(self, payload: object) -> None:
        for item in self._preview_items:
            self.scene().removeItem(item)
        self._preview_items.clear()
        if not payload:
            return
        _name, geometry = payload
        pen = QPen(QColor("#ffd166"), 1.5, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        item: QGraphicsItem | None = None
        if "p1" in geometry and "p2" in geometry:
            p1, p2 = geometry["p1"], geometry["p2"]
            item = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
        elif "bbox" in geometry:
            x0, y0, x1, y1 = geometry["bbox"]
            item = QGraphicsRectItem(x0, y0, x1 - x0, y1 - y0)
        elif geometry.get("points"):
            path = QPainterPath(QPointF(*geometry["points"][0]))
            for point in geometry["points"][1:]:
                path.lineTo(*point)
            item = QGraphicsPathItem(path)
        if item is not None:
            item.setPen(pen)
            item.setZValue(50)
            self.scene().addItem(item)
            self._preview_items.append(item)

    def _tool_committed(self, kind: str, geometry: dict[str, Any]) -> None:
        if kind == MeasurementKind.RECTANGLE_AREA.value:
            self.roiDrawn.emit(tuple(geometry["bbox"]))
        self.measurementRequested.emit(kind, geometry)

    def _scene_selection_changed(self) -> None:
        selected = [item for item in self.scene().selectedItems() if item.data(0)]
        self.recordSelected.emit(selected[0].data(0) if selected else None)

    def focus_record(self, record_id: str) -> None:
        item = next((item for item in self._overlay_items if item.data(0) == record_id), None)
        if item is not None:
            self.centerOn(item)

    def fit_to_window(self) -> None:
        if self.image is not None:
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def actual_pixels(self) -> None:
        self.resetTransform()

    def reset_view(self) -> None:
        self.resetTransform()
        self.centerOn(self.sceneRect().center())

    def activate_tool(self, name: str) -> None:
        self.tools.activate(name)
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if name == "pan"
            else QGraphicsView.DragMode.NoDrag
        )

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.image is None:
            return
        factor = 1.18 if event.angleDelta().y() > 0 else 1.0 / 1.18
        current = self.transform().m11()
        if 0.02 <= current * factor <= 80.0:
            self.scale(factor, factor)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.MiddleButton or self._space_pan:
            self._last_pan = event.position().toPoint()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        scene_point = self.mapToScene(event.position().toPoint())
        self.tools.active_tool.mouse_press(scene_point, event.button(), event.modifiers())
        if self.tools.active_tool.name == "select":
            super().mousePressEvent(event)
        else:
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_pan is not None:
            delta = event.position().toPoint() - self._last_pan
            self._last_pan = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        self.tools.active_tool.mouse_move(point, event.buttons())
        self._emit_coordinate(point)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._last_pan is not None:
            self._last_pan = None
            self.viewport().unsetCursor()
            event.accept()
            return
        point = self.mapToScene(event.position().toPoint())
        self.tools.active_tool.mouse_release(point, event.button())
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = True
            event.accept()
            return
        if self.tools.active_tool.key_press(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pan = False
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _emit_coordinate(self, point: QPointF) -> None:
        if self.image is None:
            return
        x, y = point.x(), point.y()
        height, width = self.image.shape
        pixel: float | None = None
        if 0 <= x < width and 0 <= y < height:
            pixel = float(self.image.gray[int(y), int(x)])
        self.coordinateChanged.emit(
            x,
            y,
            pixel,
            x * self.image.calibration.pixel_size_x_m,
            y * self.image.calibration.pixel_size_y_m,
        )

    def _update_scale_bar(self) -> None:
        for item in self._scale_items:
            self.scene().removeItem(item)
        self._scale_items.clear()
        if self.image is None:
            return
        height, width = self.image.shape
        target_m = width * self.image.calibration.pixel_size_x_m * 0.18
        exponent = math.floor(math.log10(max(target_m, 1e-30)))
        base = target_m / 10**exponent
        nice = 1.0 if base < 1.5 else 2.0 if base < 3.5 else 5.0
        length_m = nice * 10**exponent
        length_px = length_m / self.image.calibration.pixel_size_x_m
        x0 = width * 0.05
        y0 = min(self.sceneRect().height() - 18.0, height - 18.0)
        line = QGraphicsLineItem(x0, y0, x0 + length_px, y0)
        pen = QPen(QColor("white"), 3.0)
        pen.setCosmetic(True)
        line.setPen(pen)
        line.setZValue(80)
        label = QGraphicsTextItem(self._format_length(length_m))
        label.setDefaultTextColor(QColor("white"))
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label.setPos(x0, y0 - 24.0)
        label.setZValue(80)
        self.scene().addItem(line)
        self.scene().addItem(label)
        self._scale_items.extend((line, label))

    @staticmethod
    def _format_length(value_m: float) -> str:
        if value_m >= 1e-3:
            return f"{value_m * 1e3:g} mm"
        if value_m >= 1e-6:
            return f"{value_m * 1e6:g} µm"
        return f"{value_m * 1e9:g} nm"
