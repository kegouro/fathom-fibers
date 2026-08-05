from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime


def get_timestamp_str() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


@dataclass
class Command:
    description: str
    execute_fn: Callable[[], None]
    undo_fn: Callable[[], None]
    redo_fn: Callable[[], None] | None = None
    affected_ids: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=get_timestamp_str)

    def execute(self) -> None:
        self.execute_fn()

    def undo(self) -> None:
        self.undo_fn()

    def redo(self) -> None:
        if self.redo_fn:
            self.redo_fn()
        else:
            self.execute_fn()


class HistoryManager:
    def __init__(self, max_depth: int = 300) -> None:
        self.max_depth = max_depth
        self.undo_stack: list[Command] = []
        self.redo_stack: list[Command] = []
        self._on_change_callbacks: list[Callable[[], None]] = []

    def register_on_change(self, callback: Callable[[], None]) -> None:
        self._on_change_callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._on_change_callbacks:
            try:
                cb()
            except (ValueError, KeyError, RuntimeError, TypeError):
                pass

    def push_and_execute(self, cmd: Command) -> None:
        cmd.execute()
        self.undo_stack.append(cmd)
        if len(self.undo_stack) > self.max_depth:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self._notify()

    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def undo(self) -> Command | None:
        if not self.can_undo():
            return None
        cmd = self.undo_stack.pop()
        cmd.undo()
        self.redo_stack.append(cmd)
        self._notify()
        return cmd

    def redo(self) -> Command | None:
        if not self.can_redo():
            return None
        cmd = self.redo_stack.pop()
        cmd.redo()
        self.undo_stack.append(cmd)
        self._notify()
        return cmd

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._notify()

    def get_log_entries(self, limit: int = 50) -> list[str]:
        entries = []
        for cmd in reversed(self.undo_stack[-limit:]):
            entries.append(f"{cmd.timestamp}  {cmd.description}")
        return entries
