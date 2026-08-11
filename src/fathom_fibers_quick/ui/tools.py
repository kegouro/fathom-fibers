from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, QPointF, Qt, Signal


class Tool(QObject):
    """State object for one interactive viewer tool."""

    committed = Signal(str, dict)
    previewChanged = Signal(object)
    cancelled = Signal()

    name = "tool"

    def __init__(self) -> None:
        super().__init__()
        self.active = False

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.cancel()
        self.active = False

    def mouse_press(
        self,
        point: QPointF,
        button: Qt.MouseButton,
        modifiers: Qt.KeyboardModifier,
    ) -> None:
        del point, button, modifiers

    def mouse_move(self, point: QPointF, buttons: Qt.MouseButton) -> None:
        del point, buttons

    def mouse_release(self, point: QPointF, button: Qt.MouseButton) -> None:
        del point, button

    def key_press(self, key: int) -> bool:
        if key == Qt.Key.Key_Escape:
            self.cancel()
            return True
        return False

    def cancel(self) -> None:
        self.previewChanged.emit(None)
        self.cancelled.emit()

    def commit(self) -> None:
        return


class PanTool(Tool):
    name = "pan"


class SelectTool(Tool):
    name = "select"
    pointSelected = Signal(QPointF)

    def mouse_press(self, point, button, modifiers) -> None:
        del modifiers
        if button == Qt.MouseButton.LeftButton:
            self.pointSelected.emit(point)


class TwoPointTool(Tool):
    measurement_kind = "DISTANCE"

    def __init__(self) -> None:
        super().__init__()
        self.start: QPointF | None = None
        self.current: QPointF | None = None

    def geometry(self, start: QPointF, end: QPointF) -> dict[str, Any]:
        return {"p1": (start.x(), start.y()), "p2": (end.x(), end.y())}

    def mouse_press(self, point, button, modifiers) -> None:
        del modifiers
        if button == Qt.MouseButton.LeftButton:
            self.start = QPointF(point)
            self.current = QPointF(point)
            self.previewChanged.emit((self.name, self.geometry(self.start, self.current)))

    def mouse_move(self, point, buttons) -> None:
        if self.start is not None and buttons & Qt.MouseButton.LeftButton:
            self.current = QPointF(point)
            self.previewChanged.emit((self.name, self.geometry(self.start, self.current)))

    def mouse_release(self, point, button) -> None:
        if self.start is None or button != Qt.MouseButton.LeftButton:
            return
        self.current = QPointF(point)
        if (self.current - self.start).manhattanLength() >= 1.0:
            self.committed.emit(self.measurement_kind, self.geometry(self.start, self.current))
        self.start = None
        self.current = None
        self.previewChanged.emit(None)

    def cancel(self) -> None:
        self.start = None
        self.current = None
        super().cancel()


class ProjectedWidthTool(TwoPointTool):
    name = "projected_width"
    measurement_kind = "PROJECTED_WIDTH"


class DistanceTool(TwoPointTool):
    name = "distance"
    measurement_kind = "DISTANCE"


class IntensityProfileTool(TwoPointTool):
    name = "intensity_profile"
    measurement_kind = "INTENSITY_PROFILE"


class RectangleROITool(TwoPointTool):
    name = "rectangle_roi"
    measurement_kind = "RECTANGLE_AREA"

    def geometry(self, start: QPointF, end: QPointF) -> dict[str, Any]:
        x0, x1 = sorted((round(start.x()), round(end.x())))
        y0, y1 = sorted((round(start.y()), round(end.y())))
        return {"bbox": (x0, y0, x1, y1)}


class PointSequenceTool(Tool):
    measurement_kind = "POLYLINE_LENGTH"
    minimum_points = 2
    automatic_count: int | None = None

    def __init__(self) -> None:
        super().__init__()
        self.points: list[QPointF] = []
        self.hover: QPointF | None = None

    def geometry(self) -> dict[str, Any]:
        return {"points": [(point.x(), point.y()) for point in self.points]}

    def mouse_press(self, point, button, modifiers) -> None:
        del modifiers
        if button != Qt.MouseButton.LeftButton:
            return
        self.points.append(QPointF(point))
        self.hover = QPointF(point)
        self.previewChanged.emit((self.name, self.preview_geometry()))
        if self.automatic_count is not None and len(self.points) == self.automatic_count:
            self.commit()

    def mouse_move(self, point, buttons) -> None:
        del buttons
        if self.points:
            self.hover = QPointF(point)
            self.previewChanged.emit((self.name, self.preview_geometry()))

    def preview_geometry(self) -> dict[str, Any]:
        points = [*self.points]
        if self.hover is not None and points and self.hover != points[-1]:
            points.append(self.hover)
        return {"points": [(point.x(), point.y()) for point in points]}

    def key_press(self, key: int) -> bool:
        if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            self.commit()
            return True
        return super().key_press(key)

    def commit(self) -> None:
        if len(self.points) >= self.minimum_points:
            self.committed.emit(self.measurement_kind, self.geometry())
        self.points.clear()
        self.hover = None
        self.previewChanged.emit(None)

    def cancel(self) -> None:
        self.points.clear()
        self.hover = None
        super().cancel()


class PolylineTool(PointSequenceTool):
    name = "polyline"
    measurement_kind = "POLYLINE_LENGTH"


class PolygonROITool(PointSequenceTool):
    name = "polygon_roi"
    measurement_kind = "POLYGON_AREA"
    minimum_points = 3


class AngleTool(PointSequenceTool):
    name = "angle"
    measurement_kind = "ANGLE"
    minimum_points = 3
    automatic_count = 3

    def geometry(self) -> dict[str, Any]:
        return {
            "pt_a": (self.points[0].x(), self.points[0].y()),
            "pt_b": (self.points[1].x(), self.points[1].y()),
            "pt_c": (self.points[2].x(), self.points[2].y()),
        }


class ToolController(QObject):
    committed = Signal(str, dict)
    previewChanged = Signal(object)
    activeChanged = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.tools: dict[str, Tool] = {
            tool.name: tool
            for tool in (
                PanTool(),
                SelectTool(),
                ProjectedWidthTool(),
                DistanceTool(),
                PolylineTool(),
                AngleTool(),
                RectangleROITool(),
                PolygonROITool(),
                IntensityProfileTool(),
            )
        }
        for tool in self.tools.values():
            tool.committed.connect(self.committed)
            tool.previewChanged.connect(self.previewChanged)
        self.active_tool: Tool = self.tools["select"]
        self.active_tool.activate()

    def activate(self, name: str) -> None:
        if name not in self.tools or self.active_tool.name == name:
            return
        self.active_tool.deactivate()
        self.active_tool = self.tools[name]
        self.active_tool.activate()
        self.activeChanged.emit(name)

    def cancel(self) -> None:
        self.active_tool.cancel()

