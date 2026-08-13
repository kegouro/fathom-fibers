"""Fathom Fibers desktop visual language.

A restrained dark scientific-instrument theme applied once at application
startup via QSS.  No scientific styling is applied here; only chrome,
navigation and data-display affordances.
"""

from __future__ import annotations

# Semantic palette (dark graphite + warm amber accent)
ACCENT = "#d99a2b"
ACCENT_DIM = "#b37d1f"
SUCCESS = "#33b67a"
WARNING = "#e0a83e"
ERROR = "#d56b6b"
INFO = "#5fc4d8"

BACKGROUND = "#1b1e24"
SURFACE = "#24282f"
SURFACE_RAISED = "#2b3038"
BORDER = "#3a404a"
BORDER_SOFT = "#333941"
TEXT = "#e8eaed"
TEXT_MUTED = "#9aa2ad"
TEXT_DIM = "#6c7480"

QSS = f"""
* {{
    font-family: "system-ui";
    outline: none;
}}
QMainWindow, QDialog {{
    background: {BACKGROUND};
}}
QWidget {{
    color: {TEXT};
    background: transparent;
}}
QLabel {{
    background: transparent;
}}
QLabel[role="title"] {{
    font-size: 15px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[role="section"] {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    color: {TEXT_MUTED};
}}
QLabel[role="primary"] {{
    font-size: 22px;
    font-weight: 600;
    color: {TEXT};
}}
QLabel[role="caption"] {{
    font-size: 11px;
    color: {TEXT_DIM};
}}
QLabel[role="muted"] {{
    color: {TEXT_MUTED};
}}
QLabel[role="success"] {{
    color: {SUCCESS};
}}
QFrame#panel, QWidget#panel {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 6px;
}}
QToolBar {{
    background: {SURFACE};
    border: none;
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 4px 10px;
    color: {TEXT};
}}
QToolBar QToolButton:hover {{
    background: {SURFACE_RAISED};
    border-color: {BORDER};
}}
QToolBar QToolButton:pressed, QToolBar QToolButton:checked {{
    background: {ACCENT_DIM};
    border-color: {ACCENT_DIM};
    color: #ffffff;
}}
QToolBar QToolButton:disabled {{
    color: {TEXT_DIM};
}}
QMenuBar {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    padding: 2px 6px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 8px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background: {SURFACE_RAISED};
}}
QMenu {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 22px 5px 10px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background: {ACCENT_DIM};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 6px;
}}
QDockWidget {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    border: none;
}}
QDockWidget::title {{
    background: {SURFACE};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
}}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background: {SURFACE};
    color: {TEXT_MUTED};
    padding: 6px 14px;
    border: 1px solid {BORDER_SOFT};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {SURFACE_RAISED};
    color: {TEXT};
    border-color: {BORDER};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}
QPushButton {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 5px 12px;
    color: {TEXT};
}}
QPushButton:hover {{
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background: {ACCENT_DIM};
    color: #ffffff;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
}}
QPushButton[role="primary"] {{
    background: {ACCENT_DIM};
    border-color: {ACCENT_DIM};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QComboBox {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 6px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT};
}}
QTableView, QTreeView, QListView {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 5px;
    gridline-color: {BORDER_SOFT};
    selection-background-color: {ACCENT_DIM};
    selection-color: #ffffff;
    alternate-background-color: #262a31;
}}
QHeaderView::section {{
    background: {SURFACE_RAISED};
    color: {TEXT_MUTED};
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER_SOFT};
    padding: 4px 6px;
    font-weight: 600;
}}
QTableWidget, QTreeWidget {{
    background: {SURFACE};
    border: 1px solid {BORDER_SOFT};
    border-radius: 5px;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QScrollBar:vertical {{
    background: {SURFACE};
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{
    background: {ACCENT_DIM};
}}
QScrollBar:horizontal {{
    background: {SURFACE};
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER};
    border-radius: 5px;
    min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
    font-size: 11px;
}}
QToolTip {{
    background: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
}}
QCheckBox {{
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {SURFACE_RAISED};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_DIM};
    border-color: {ACCENT_DIM};
}}
QSplitter::handle {{
    background: {BORDER_SOFT};
}}
QProgressBar {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    text-align: center;
    height: 14px;
}}
QProgressBar::chunk {{
    background: {ACCENT_DIM};
    border-radius: 4px;
}}
QDialogButtonBox QPushButton {{
    min-width: 72px;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
    background: {ACCENT_DIM};
}}
"""


def apply_theme(app) -> None:
    """Apply the dark scientific-instrument theme to a QApplication."""
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window, __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(BACKGROUND))
    palette.setColor(palette.ColorRole.WindowText, __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(TEXT))
    app.setPalette(palette)


__all__ = [
    "ACCENT",
    "ACCENT_DIM",
    "BACKGROUND",
    "BORDER",
    "ERROR",
    "INFO",
    "QSS",
    "SUCCESS",
    "SURFACE",
    "SURFACE_RAISED",
    "TEXT",
    "TEXT_DIM",
    "TEXT_MUTED",
    "WARNING",
    "apply_theme",
]
