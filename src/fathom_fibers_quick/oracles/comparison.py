from __future__ import annotations

from typing import Any

import numpy as np

from ..measurement_records import MeasurementKind
from ..model import Project
from ..simpoly_compat import run_simpoly_pipeline
from .contracts import EstimandType


def compare_fathom_and_simpoly(
    project: Project,
    gray: np.ndarray,
) -> dict[str, Any]:
    """Compares Fathom's LOCAL_SECTION_WEIGHTED estimator with SIMPoly literature reimplementation."""
    sim_res = run_simpoly_pipeline(gray, project.image.calibration, project.image.footer_bounds)

    fathom_sections = [
        r.primary_value
        for r in project.records
        if r.kind == MeasurementKind.PROJECTED_WIDTH and r.is_included_in_statistics and r.primary_value is not None
    ]

    fathom_mean_px = float(np.mean(fathom_sections) / project.image.calibration.pixel_size_x_m) if fathom_sections else None
    fathom_median_px = float(np.median(fathom_sections) / project.image.calibration.pixel_size_x_m) if fathom_sections else None

    sim_center_px = sim_res["gaussian_center_px"]
    sim_mean_px = sim_res["arithmetic_mean_px"]

    diff_center_px = (fathom_median_px - sim_center_px) if (fathom_median_px and sim_center_px) else None
    rel_diff_percent = (abs(diff_center_px) / max(sim_center_px, 1e-6)) * 100.0 if diff_center_px is not None else None

    return {
        "estimand_fathom": EstimandType.LOCAL_SECTION_WEIGHTED.value,
        "estimand_simpoly": EstimandType.SIMPOLY_GAUSSIAN_CENTER.value,
        "fathom_median_px": fathom_median_px,
        "fathom_mean_px": fathom_mean_px,
        "simpoly_gaussian_center_px": sim_center_px,
        "simpoly_skeleton_mean_px": sim_mean_px,
        "difference_px": diff_center_px,
        "relative_difference_percent": rel_diff_percent,
        "n_fathom_sections": len(fathom_sections),
    }
