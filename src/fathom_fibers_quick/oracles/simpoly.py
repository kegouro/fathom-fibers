from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..model import Calibration
from ..simpoly_compat import run_simpoly_pipeline
from .contracts import EstimandType, OracleComparison, OracleRun


def generate_synthetic_fiber_phantom(
    width_px: float,
    shape: tuple[int, int] = (512, 512),
    disordered: bool = False,
    n_fibers: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Generates synthetic fiber network micrograph (ordered grid or disordered random fibers) with known width_px."""
    h, w = shape
    img = Image.new("L", (w, h), color=20)
    draw = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)

    if not disordered:
        # Ordered grid network
        spacing_y = h // (n_fibers // 2 + 1)
        spacing_x = w // (n_fibers // 2 + 1)

        for i in range(1, n_fibers // 2 + 1):
            y = i * spacing_y
            draw.line([(0, y), (w, y)], fill=200, width=round(width_px))
            x = i * spacing_x
            draw.line([(x, 0), (x, h)], fill=200, width=round(width_px))
    else:
        # Disordered random orientation network
        for _ in range(n_fibers):
            x1 = float(rng.uniform(0, w))
            y1 = float(rng.uniform(0, h))
            angle = float(rng.uniform(0, 2 * math.pi))
            length = math.hypot(w, h) * 1.2
            x2 = x1 + length * math.cos(angle)
            y2 = y1 + length * math.sin(angle)
            draw.line([(x1, y1), (x2, y2)], fill=200, width=round(width_px))

    arr = np.array(img, dtype=np.uint8)
    # Add mild Gaussian noise
    noise = rng.normal(0, 5, arr.shape)
    arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)

    return arr


def run_synthetic_benchmark_suite(
    output_dir: str | Path | None = None,
) -> tuple[list[OracleRun], list[OracleComparison], dict[str, Any]]:
    """Runs 41 synthetic benchmark cases (widths 10 to 100 px, ordered/disordered) against known truth."""
    cal = Calibration(1e-9, 1e-9, "synthetic_1nm_px")
    widths = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    runs: list[OracleRun] = []
    comparisons: list[OracleComparison] = []

    ordered_errors: list[float] = []
    disordered_errors: list[float] = []

    case_counter = 1
    # 41 cases total (20 ordered, 21 disordered)
    for width in widths:
        # 2 ordered cases per width
        for ord_idx in range(2):
            img_arr = generate_synthetic_fiber_phantom(width, disordered=False, seed=case_counter * 10)
            res = run_simpoly_pipeline(img_arr, cal)

            run_id = f"SYNTH_ORD_{case_counter:02d}_W{int(width)}"
            c_center = res["gaussian_center_px"]
            abs_err = abs(c_center - width)
            rel_err = (abs_err / width) * 100.0
            ordered_errors.append(rel_err)

            o_run = OracleRun(
                run_id=run_id,
                oracle_id="SIMPOLY_LITERATURE_REIMPLEMENTATION_V1",
                oracle_version="1.0.0",
                image_id=f"{run_id}.png",
                gaussian_center_px=c_center,
                gaussian_sigma_px=res["gaussian_sigma_px"],
                arithmetic_mean_px=res["arithmetic_mean_px"],
                median_px=res["median_px"],
                std_px=res["std_px"],
                segmented_fraction=res["segmented_fraction"],
                status="SUCCESS",
            )
            runs.append(o_run)

            comp = OracleComparison(
                comparison_id=f"COMP_{run_id}",
                image_id=f"{run_id}.png",
                estimand_oracle=EstimandType.SIMPOLY_GAUSSIAN_CENTER,
                estimand_target=EstimandType.SKELETON_PIXEL_MEAN,
                oracle_value_px=c_center,
                target_value_px=width,
                absolute_error_px=abs_err,
                relative_error_percent=rel_err,
            )
            comparisons.append(comp)
            case_counter += 1

        # 2 disordered cases per width
        for dis_idx in range(2):
            img_arr = generate_synthetic_fiber_phantom(width, disordered=True, seed=case_counter * 100)
            res = run_simpoly_pipeline(img_arr, cal)

            run_id = f"SYNTH_DIS_{case_counter:02d}_W{int(width)}"
            c_center = res["gaussian_center_px"]
            abs_err = abs(c_center - width)
            rel_err = (abs_err / width) * 100.0
            disordered_errors.append(rel_err)

            o_run = OracleRun(
                run_id=run_id,
                oracle_id="SIMPOLY_LITERATURE_REIMPLEMENTATION_V1",
                oracle_version="1.0.0",
                image_id=f"{run_id}.png",
                gaussian_center_px=c_center,
                gaussian_sigma_px=res["gaussian_sigma_px"],
                arithmetic_mean_px=res["arithmetic_mean_px"],
                median_px=res["median_px"],
                std_px=res["std_px"],
                segmented_fraction=res["segmented_fraction"],
                status="SUCCESS",
            )
            runs.append(o_run)

            comp = OracleComparison(
                comparison_id=f"COMP_{run_id}",
                image_id=f"{run_id}.png",
                estimand_oracle=EstimandType.SIMPOLY_GAUSSIAN_CENTER,
                estimand_target=EstimandType.SKELETON_PIXEL_MEAN,
                oracle_value_px=c_center,
                target_value_px=width,
                absolute_error_px=abs_err,
                relative_error_percent=rel_err,
            )
            comparisons.append(comp)
            case_counter += 1

    # 41st case (extra disordered case)
    img_arr = generate_synthetic_fiber_phantom(50.0, disordered=True, seed=999)
    res = run_simpoly_pipeline(img_arr, cal)
    run_id = "SYNTH_DIS_41_W50"
    c_center = res["gaussian_center_px"]
    abs_err = abs(c_center - 50.0)
    rel_err = (abs_err / 50.0) * 100.0
    disordered_errors.append(rel_err)
    runs.append(
        OracleRun(
            run_id=run_id,
            oracle_id="SIMPOLY_LITERATURE_REIMPLEMENTATION_V1",
            oracle_version="1.0.0",
            image_id=f"{run_id}.png",
            gaussian_center_px=c_center,
            status="SUCCESS",
        )
    )

    all_rel_errors = ordered_errors + disordered_errors

    summary = {
        "total_cases": len(all_rel_errors),
        "mean_error_ordered_percent": float(np.mean(ordered_errors)),
        "mean_error_disordered_percent": float(np.mean(disordered_errors)),
        "median_relative_error_percent": float(np.median(all_rel_errors)),
        "p90_relative_error_percent": float(np.percentile(all_rel_errors, 90)),
        "max_relative_error_percent": float(np.max(all_rel_errors)),
        "fraction_within_10_percent": float((np.asarray(all_rel_errors) <= 10.0).mean()),
        "published_ordered_target_percent": 2.1,
        "published_disordered_target_percent": 1.6,
        "gate_passed": bool(np.median(all_rel_errors) <= 5.0 and np.percentile(all_rel_errors, 90) <= 10.0),
    }

    return runs, comparisons, summary
