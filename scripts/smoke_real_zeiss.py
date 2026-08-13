from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import QApplication

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.application import ProjectSession
from fathom_fibers_quick.exporters import export_csv
from fathom_fibers_quick.oracles.simpoly_source import PROFILE_CONTROLLED_INPUT_V1
from fathom_fibers_quick.project_io import verify_project_source
from fathom_fibers_quick.ui.main_window import MainWindow

PRIMARY_NAMES = tuple(n for n in os.environ.get("FATHOM_PRIMARY_NAMES", "").split(",") if n)


def _find_images(root: Path) -> list[Path]:
    available = {path.name: path for path in root.rglob("*.tif")}
    return [available[name] for name in PRIMARY_NAMES if name in available]


def run(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    reports: list[dict[str, Any]] = []
    for image_path in _find_images(root):
        engine = FathomEngine()
        session = ProjectSession(engine)
        image = engine.open_image(image_path)
        session.new_from_image(image)
        window = MainWindow(session, smoke_test=True)
        window.viewer.set_image(image)
        window.show()
        app.processEvents()

        # Viewer behavior without a physical monitor.
        window.viewer.fit_to_window()
        window.viewer.scale(1.2, 1.2)
        window.viewer.horizontalScrollBar().setValue(25)
        window.viewer.verticalScrollBar().setValue(15)
        window.viewer.actual_pixels()

        height, width = image.shape
        valid_height = image.footer_bounds[0] if image.footer_bounds else height
        center_x, center_y = width / 2.0, valid_height / 2.0
        manual = session.create_measurement(
            "PROJECTED_WIDTH",
            {"p1": (center_x, center_y - 10.0), "p2": (center_x, center_y + 10.0)},
            name="Real smoke projected width",
        )
        session.update_metadata([manual.measurement_id], notes="Real Zeiss smoke")
        session.undo()
        session.redo()

        project_path = output / f"{image_path.stem}.fiberquick.json"
        session.save(project_path)
        reopened = ProjectSession(engine)
        reopened.open_project(project_path)
        verification = verify_project_source(reopened.project)

        half = 192
        roi = (
            max(0, round(center_x) - half),
            max(0, round(center_y) - half),
            min(width, round(center_x) + half),
            min(valid_height, round(center_y) + half),
        )
        reopened.set_roi(roi)
        simpoly, _intermediates = engine.run_simpoly(
            reopened.image,
            profile=PROFILE_CONTROLLED_INPUT_V1,
            roi_bbox=roi,
        )
        sim_record = reopened.apply_simpoly_result(
            simpoly,
            profile=PROFILE_CONTROLLED_INPUT_V1,
            roi_bbox=roi,
        )
        fathom = engine.run_fathom(reopened.image, roi_bbox=roi, options={"n_sections": 3})
        fathom_records = reopened.apply_fathom_result(fathom)
        comparison = reopened.compare_methods()
        reopened.save(project_path)
        csv_path = export_csv(reopened.project, output / f"{image_path.stem}.csv")

        window.close()
        window = MainWindow(reopened, smoke_test=True)
        window.viewer.set_image(reopened.image)
        window.show()
        app.processEvents()
        screenshot = output / f"{image_path.stem}-workspace.png"
        window.grab().save(str(screenshot))
        window.close()

        reports.append(
            {
                "image": image_path.name,
                "shape": image.shape,
                "calibration_nm_per_px": (
                    image.calibration.pixel_size_x_m * 1e9,
                    image.calibration.pixel_size_y_m * 1e9,
                ),
                "footer_bounds": image.footer_bounds,
                "source_verification": verification.status.value,
                "manual_record": manual.measurement_id,
                "simpoly_record": sim_record.measurement_id,
                "simpoly_status": simpoly.status,
                "simpoly_valid_diameters": int(simpoly.local_diameters_px.size),
                "fathom_candidates": len(fathom.candidates),
                "fathom_proposals": len(fathom_records),
                "comparison_rows": len(comparison.rows),
                "project": str(project_path),
                "csv": str(csv_path),
                "screenshot": str(screenshot),
            }
        )
    payload = {"requested": list(PRIMARY_NAMES), "processed": reports}
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(os.environ.get("FATHOM_ZEISS_DATASET", "data/zeiss")))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp") / f"fathom-fibers-{os.environ.get('USER', 'user')}" / "real-smoke",
    )
    args = parser.parse_args()
    report = run(args.input, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
