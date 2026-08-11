from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from ..application import ProjectSession


class HistoryBridge(QObject):
    """Qt action-state bridge over the application-layer HistoryManager."""

    stateChanged = Signal(bool, bool)

    def __init__(self, session: ProjectSession) -> None:
        super().__init__()
        self.session = session
        session.history.register_on_change(self.refresh)

    def refresh(self) -> None:
        self.stateChanged.emit(self.session.history.can_undo(), self.session.history.can_redo())

    def undo(self) -> None:
        self.session.undo()
        self.refresh()

    def redo(self) -> None:
        self.session.redo()
        self.refresh()

