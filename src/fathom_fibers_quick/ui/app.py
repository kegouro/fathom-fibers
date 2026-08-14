from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ..api import FathomEngine
from ..application import ProjectSession
from ..model import Calibration
from .main_window import MainWindow


def _configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


def _run_smoke(window: MainWindow, app: QApplication) -> None:
    array = np.zeros((320, 480), dtype=np.uint8)
    array[110:150, 50:430] = 220
    image = FathomEngine().from_array(
        array,
        calibration=Calibration(5e-9, 5e-9, "SYNTHETIC_SMOKE"),
        image_id="synthetic-smoke",
    )
    window.session.new_from_image(image)
    window.viewer.set_image(image)
    window.session.create_measurement(
        "PROJECTED_WIDTH",
        {"p1": (180.0, 110.0), "p2": (180.0, 150.0)},
    )
    window._refresh_all()
    artifact_dir = Path("/tmp") / f"fathom-fibers-{os.environ.get('USER', 'user')}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(artifact_dir / "smoke-main-window.png"))
    QTimer.singleShot(150, app.quit)


def launch(
    initial_path: str | None = None,
    *,
    smoke_test: bool = False,
    argv: list[str] | None = None,
) -> int:
    _configure_logging()
    app = QApplication.instance() or QApplication(argv or sys.argv)
    app.setApplicationName("Fathom Fibers")
    app.setOrganizationName("Fathom")
    session = ProjectSession(FathomEngine())
    window = MainWindow(session, initial_path=initial_path, smoke_test=smoke_test)
    window.show()
    if smoke_test:
        QTimer.singleShot(0, lambda: _run_smoke(window, app))
    return app.exec()


def main() -> None:
    raise SystemExit(launch())


if __name__ == "__main__":
    main()
