from __future__ import annotations

import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from ..api import FathomEngine
from ..oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    PROFILE_SOURCE_COMPAT_V1,
    SIMPolySourceConfig,
    run_simpoly_source_pipeline,
)


def array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    normalized = array.astype(np.uint8, copy=False) if array.dtype == bool else array
    if normalized.dtype.byteorder == ">":
        normalized = normalized.byteswap().view(normalized.dtype.newbyteorder("<"))
    contiguous = np.ascontiguousarray(normalized)
    finite = array.astype(float, copy=False).ravel()
    return {
        "shape": list(array.shape),
        "class": str(array.dtype),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
        "mean": float(np.mean(finite)) if finite.size else None,
        "foreground_count": int(np.count_nonzero(array)),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _result_payload(result: Any, inter: Any) -> dict[str, Any]:
    diameters = np.asarray(result.local_diameters_px, dtype=float)
    stages = {
        "I": inter.cropped,
        "Ihist_after_clahe": inter.clahe,
        "Ihist_after_histeq": inter.equalized,
        "marker": inter.marker,
        "Iobr": inter.reconstruction,
        "E_canny": inter.canny_raw,
        "E_area_filtered": inter.canny_area_filtered,
        "E_thickened": inter.canny_edges,
        "BW_threshold": inter.bw_threshold,
        "BW_closed": inter.bw_closed,
        "BW_clean": inter.bw_clean,
        "BW_fill": inter.bw_fill,
        "BW_majority": inter.bw_majority,
        "BW_thin": inter.bw_thin,
        "BW_median": inter.median_mask,
        "BW_thickened": inter.thickened_mask,
        "SK_bwskel": inter.raw_skeleton,
        "branchpoints": inter.branchpoints,
        "branch_guard": inter.branch_guard,
        "SK_without_branches": inter.skeleton_without_branches,
        "SK_after_spur": inter.skeleton_after_spur,
        "F_edge_distance": inter.edge_distance_map,
        "SK_valid": inter.valid_skeleton,
        "Dist": inter.distance_map,
        "diameters": result.local_diameters_px,
        "hist_values": result.histogram_counts,
        "hist_edges": result.histogram_edges,
    }
    return {
        "status": result.status,
        "threshold_level": inter.threshold_level,
        "median_iterations": inter.median_iterations,
        "gauss_a1": result.gaussian_amplitude,
        "gauss_b1": result.reported_center,
        "gauss_b1_px": result.gaussian_center_px,
        "gauss_c1": (
            result.gaussian_c1_px
            if result.reported_unit == "px"
            else result.gaussian_c1_px * result.reported_center / result.gaussian_center_px
            if result.gaussian_center_px not in {None, 0} and result.reported_center is not None
            else None
        ),
        "gauss_c1_px": result.gaussian_c1_px,
        "source_reported_stdev_px": result.source_reported_stdev_px,
        "mathematical_sigma_px": result.mathematical_gaussian_sigma_px,
        "diameter_n": int(result.local_diameters_px.size),
        "diameter_mean_px": result.arithmetic_mean_px,
        "diameter_median_px": result.median_px,
        "diameter_std_px": float(np.std(diameters, ddof=1)) if diameters.size > 1 else 0.0,
        "diameter_quantiles_px": (
            [float(value) for value in np.quantile(diameters, [0.05, 0.25, 0.5, 0.75, 0.95])]
            if diameters.size
            else []
        ),
        "hist_values": result.histogram_counts.astype(float).tolist(),
        "hist_edges": result.histogram_edges.astype(float).tolist(),
        "flags": list(result.flags),
        "stages": {
            name: array_summary(value) for name, value in stages.items() if value is not None
        },
    }


def run_case(case: dict[str, Any], output_path: Path) -> None:
    payload: dict[str, Any] = {"case_id": case["case_id"], "status": "FAILED"}
    try:
        raw = tifffile.imread(case["absolute_path"])
        conversion = float(case["conversion_um_per_px"])
        source_result, source_inter = run_simpoly_source_pipeline(
            raw,
            SIMPolySourceConfig(
                profile=PROFILE_SOURCE_COMPAT_V1,
                conversion_um_per_px=conversion,
            ),
        )
        engine = FathomEngine()
        image = engine.open_image(case["absolute_path"])
        body = np.asarray(image.pixels)[: image.valid_body.shape[0], :]
        controlled_result, controlled_inter = run_simpoly_source_pipeline(
            body,
            SIMPolySourceConfig(
                profile=PROFILE_CONTROLLED_INPUT_V1,
                conversion_um_per_px=conversion,
            ),
        )
        fathom = engine.run_fathom(image)
        widths_px = [
            proposal.width_m / image.calibration.pixel_size_x_m
            for candidate in fathom.candidates
            for proposal in candidate.proposed_measurements
        ]
        flags = sorted(
            set(fathom.flags)
            | {flag for candidate in fathom.candidates for flag in candidate.quality_flags}
        )
        payload.update(
            status="COMPLETE",
            source_compat=_result_payload(source_result, source_inter),
            controlled_input=_result_payload(controlled_result, controlled_inter),
            fathom={
                "status": "COMPLETE",
                "candidate_count": len(fathom.candidates),
                "measurement_count": len(widths_px),
                "section_median_px": float(np.median(widths_px)) if widths_px else None,
                "section_median_um": (
                    float(np.median(widths_px)) * conversion if widths_px else None
                ),
                "resolution_status": fathom.summary.resolution_status,
                "flags": flags,
            },
        )
    except Exception as exc:
        payload.update(error=str(exc), traceback=traceback.format_exc())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    case_path, output_path = map(Path, sys.argv[1:3])
    run_case(json.loads(case_path.read_text(encoding="utf-8")), output_path)


if __name__ == "__main__":
    main()
