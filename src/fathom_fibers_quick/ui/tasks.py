from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class TaskCancelled(Exception):
    pass


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(Exception, str)
    finished = Signal()
    cancelled = Signal()


class AnalysisTask(QRunnable):
    """Run a pure computation away from Qt widgets and marshal its result."""

    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = TaskSignals()
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @Slot()
    def run(self) -> None:
        if self._cancelled:
            self.signals.cancelled.emit()
            self.signals.finished.emit()
            return
        try:
            result = self.function()
            if self._cancelled:
                self.signals.cancelled.emit()
            else:
                self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(exc, traceback.format_exc())
        finally:
            self.signals.finished.emit()

