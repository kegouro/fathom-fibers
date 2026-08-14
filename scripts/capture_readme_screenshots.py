"""Generate reproducible README screenshots from the real Qt application.

The screenshots use deterministic synthetic microscopy data so documentation
never depends on the private validation corpus.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from PySide6.QtWidgets import QApplication

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.ui.help import HelpDialog, PAGE_QUICK_START
from fathom_fibers_quick.ui.main_window import MainWindow
from fathom_fibers_quick.ui.theme import apply_theme


def synthetic_sem(height: int = 720, width: int = 1080) -> np.ndarray:
    """Return deterministic SEM-like fibrous pixels for documentation only."""
    yy, xx = np.mgrid[:height, :width]
    rng = np.random.default_rng(20260813)
    image = 34.0 + rng.normal(0.0, 7.0, size=(height, width))

    fibers = (
        (0.18, 90.0, 8.0, 185.0),
        (-0.24, 245.0, 11.0, 215.0),
        (0.31, 345.0, 9.0, 195.0),
        (-0.12, 470.0, 13.0, 205.0),
        (0.42, 555.0, 8.5, 175.0),
        (-0.36, 650.0, 10.0, 225.0),
    )
    for slope, intercept, sigma, amplitude in fibers:
        distance = np.abs(yy - (slope * xx + intercept)) / np.sqrt(1.0 + slope**2)
        image += amplitude * np.exp(-0.5 * (distance / sigma) ** 2)

    for slope, intercept in ((-0.62, 760.0), (0.58, 60.0), (0.04, 585.0)):
        distance = np.abs(yy - (slope * xx + intercept)) / np.sqrt(1.0 + slope**2)
        image += 150.0 * np.exp(-0.5 * (distance / 7.0) ** 2)

    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> None:
    out = Path("docs/assets")
    out.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Fathom Fibers")
    app.setOrganizationName("Fathom")
    apply_theme(app)

    engine = FathomEngine()
    image = engine.from_array(
        synthetic_sem(),
        calibration=Calibration(52.04e-9, 52.04e-9, "SYNTHETIC_DOCS"),
        image_id="synthetic-readme-sem",
    )
    session = ProjectSession(engine)
    window = MainWindow(session, smoke_test=True)
    session.new_from_image(image)
    window.viewer.set_image(image)

    session.create_measurement(
        "PROJECTED_WIDTH", {"p1": (410.0, 320.0), "p2": (428.0, 351.0)}
    )
    session.create_measurement(
        "PROJECTED_WIDTH", {"p1": (665.0, 424.0), "p2": (687.0, 457.0)}
    )
    window._refresh_all()
    window.resize(1500, 920)
    window.show()
    app.processEvents()
    window.viewer.fit_to_window()
    app.processEvents()
    window.grab().save(str(out / "workspace.png"))

    help_dialog = HelpDialog(window, page=PAGE_QUICK_START)
    help_dialog.resize(900, 620)
    help_dialog.show()
    app.processEvents()
    help_dialog.grab().save(str(out / "quick-start.png"))
    help_dialog.close()
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
