from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..api import FathomEngine
from ..oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    SIMPolySourceConfig,
    run_simpoly_source_pipeline,
)
from .matlab_oracle import load_intermediates
from .parity_metrics import (
    boolean_parity,
    diameter_parity,
    first_divergence_record,
    fit_parity,
    float_parity,
    histogram_parity,
    skeleton_parity,
)

MASK_STAGES = {
    "CANNY": ("E_canny", "canny_raw"),
    "AREA_FILTER": ("E_area_filtered", "canny_area_filtered"),
    "EDGE_THICKEN": ("E_thickened", "canny_edges"),
    "THRESHOLD": ("BW_threshold", "bw_threshold"),
    "CLOSING": ("BW_closed", "bw_closed"),
    "CLEAN": ("BW_clean", "bw_clean"),
    "FILL": ("BW_fill", "bw_fill"),
    "MAJORITY": ("BW_majority", "bw_majority"),
    "THIN": ("BW_thin", "bw_thin"),
    "MEDIAN_LOOP": ("BW_median", "median_mask"),
    "THICKEN": ("BW_thickened", "thickened_mask"),
    "BRANCHPOINTS": ("branchpoints", "branchpoints"),
    "BRANCH_GUARD": ("branch_guard", "branch_guard"),
    "SPUR": ("SK_after_spur", "skeleton_after_spur"),
    "EDGE_DISTANCE_FILTER": ("SK_valid", "valid_skeleton"),
}


def run_detailed_case(case: dict[str, Any], matlab_path: Path, output_path: Path) -> None:
    matlab = load_intermediates(matlab_path)
    image = FathomEngine().open_image(case["absolute_path"])
    body = np.asarray(image.pixels)[: image.valid_body.shape[0], :]
    result, inter = run_simpoly_source_pipeline(
        body,
        SIMPolySourceConfig(
            profile=PROFILE_CONTROLLED_INPUT_V1,
            conversion_um_per_px=case["conversion_um_per_px"],
        ),
    )
    metrics: dict[str, Any] = {}
    matches: dict[str, bool] = {}

    metrics["CROP"] = float_parity(matlab["I"], inter.cropped, rtol=0, atol=0)
    matches["CROP"] = bool(metrics["CROP"]["allclose"])
    for stage, matlab_name, python_value in (
        ("CLAHE", "Ihist_after_clahe", inter.clahe),
        ("HISTEQ", "Ihist_after_histeq", inter.equalized),
        ("EROSION", "marker", inter.marker),
        ("RECONSTRUCTION", "Iobr", inter.reconstruction),
    ):
        matlab_value = matlab[matlab_name]
        if np.issubdtype(matlab_value.dtype, np.integer):
            matlab_value = matlab_value.astype(float) / np.iinfo(matlab_value.dtype).max
        metrics[stage] = float_parity(matlab_value, python_value, rtol=0, atol=1 / 255)
        matches[stage] = bool(metrics[stage]["allclose"])

    matlab_level = float(matlab["threshold_level"].item())
    metrics["OTSU"] = {
        "matlab": matlab_level,
        "python": inter.threshold_level,
        "absolute_difference": abs(matlab_level - inter.threshold_level),
    }
    matches["OTSU"] = metrics["OTSU"]["absolute_difference"] <= 1e-12
    for stage, (matlab_name, attribute) in MASK_STAGES.items():
        metrics[stage] = boolean_parity(matlab[matlab_name], getattr(inter, attribute))
        matches[stage] = bool(metrics[stage].get("exact_equal", False))
    metrics["SKELETON"] = skeleton_parity(matlab["SK_bwskel"], inter.raw_skeleton)
    matches["SKELETON"] = bool(metrics["SKELETON"].get("exact_equal", False))
    metrics["EDGE_DISTANCE_FILTER"] = skeleton_parity(matlab["SK_valid"], inter.valid_skeleton)
    matches["EDGE_DISTANCE_FILTER"] = bool(
        metrics["EDGE_DISTANCE_FILTER"].get("exact_equal", False)
    )
    for stage, matlab_name, python_value in (
        ("EDGE_DISTANCE_FILTER", "F_edge_distance", inter.edge_distance_map),
        ("EDT", "Dist", inter.distance_map),
    ):
        distance_metrics = float_parity(matlab[matlab_name], python_value, rtol=1e-6, atol=2e-5)
        metrics[f"{stage}_DISTANCE"] = distance_metrics
        if stage == "EDT":
            matches[stage] = bool(distance_metrics["allclose"])
    metrics["DIAMETERS"] = diameter_parity(matlab["diameters"], result.local_diameters_px)
    matches["DIAMETERS"] = bool(
        metrics["DIAMETERS"]["max_aligned_difference"] == 0
        and metrics["DIAMETERS"]["matlab"]["n"] == metrics["DIAMETERS"]["python"]["n"]
    )
    metrics["HISTOGRAM"] = histogram_parity(
        matlab["hist_values"], matlab["hist_edges"], result.histogram_counts, result.histogram_edges
    )
    matches["HISTOGRAM"] = bool(
        metrics["HISTOGRAM"]["counts_equal"] and metrics["HISTOGRAM"]["edges_equal"]
    )
    python_c1 = result.gaussian_c1_px * case["conversion_um_per_px"]
    metrics["GAUSSIAN_FIT"] = fit_parity(
        {
            "a1": matlab["gauss_a1"].item(),
            "b1": matlab["gauss_b1"].item(),
            "c1": matlab["gauss_c1"].item(),
        },
        {"a1": result.gaussian_amplitude, "b1": result.reported_center, "c1": python_c1},
    )
    matches["GAUSSIAN_FIT"] = all(
        metrics["GAUSSIAN_FIT"][name]["absolute_difference"] == 0 for name in ("a1", "b1", "c1")
    )
    divergence = first_divergence_record(
        matches,
        metrics,
        {"CLAHE": "Inference: MATLAB and scikit-image CLAHE interpolation/clip semantics differ."},
    )
    payload = {
        "case_id": case["case_id"],
        "profile": PROFILE_CONTROLLED_INPUT_V1,
        "status": "COMPLETE",
        **divergence,
        "stage_matches": matches,
        "stage_metrics": metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    case_path, matlab_path, output_path = map(Path, sys.argv[1:4])
    run_detailed_case(json.loads(case_path.read_text(encoding="utf-8")), matlab_path, output_path)


if __name__ == "__main__":
    main()
