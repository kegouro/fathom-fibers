from __future__ import annotations

import json
import os
from pathlib import Path

from fathom_fibers_quick.oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    PROFILE_SOURCE_COMPAT_V1,
    SIMPolySourceConfig,
    run_simpoly_source_pipeline,
)
from fathom_fibers_quick.zeiss import load_image_document


def compare_simpoly_and_fathom_on_zeiss():
    tiff_dir = Path(os.environ.get("FATHOM_ZEISS_DATASET", "data/zeiss"))
    if not tiff_dir.exists():
        print("Zeiss dataset directory not found. Skipping Zeiss comparison.")
        return

    all_tiffs = list(tiff_dir.rglob("*.tif")) + list(tiff_dir.rglob("*.tiff"))
    if not all_tiffs:
        print("No TIFF images found in the Zeiss dataset directory.")
        return

    results = []
    for p in sorted(all_tiffs):
        try:
            doc, _img, gray = load_image_document(p)

            # Pre-cropped image body
            usable_h = doc.footer_bounds[0] if doc.footer_bounds else gray.shape[0]
            body_crop = gray[:usable_h, :].copy()

            # 1. Run SIMPoly CONTROLLED_INPUT_V1
            cfg_controlled = SIMPolySourceConfig(profile=PROFILE_CONTROLLED_INPUT_V1)
            res_ctrl, _inter = run_simpoly_source_pipeline(body_crop, cfg_controlled)

            # 2. Run SIMPoly SOURCE_COMPAT_V1 (fixed 90-row crop)
            cfg_source = SIMPolySourceConfig(profile=PROFILE_SOURCE_COMPAT_V1)
            res_source, _inter_src = run_simpoly_source_pipeline(gray, cfg_source)

            # Estimands
            sp_center_px = res_ctrl.gaussian_center_px
            sp_mean_px = res_ctrl.arithmetic_mean_px
            sp_median_px = res_ctrl.median_px

            sp_center_nm = sp_center_px * doc.calibration.pixel_size_x_m * 1e9 if sp_center_px else None
            sp_mean_nm = sp_mean_px * doc.calibration.pixel_size_x_m * 1e9 if sp_mean_px else None
            sp_median_nm = sp_median_px * doc.calibration.pixel_size_x_m * 1e9 if sp_median_px else None

            row_data = {
                "image_name": p.name,
                "calibration_nm_per_px": doc.calibration.pixel_size_x_m * 1e9,
                "simpoly_controlled_gaussian_center_px": sp_center_px,
                "simpoly_controlled_gaussian_center_nm": sp_center_nm,
                "simpoly_controlled_skeleton_mean_px": sp_mean_px,
                "simpoly_controlled_skeleton_mean_nm": sp_mean_nm,
                "simpoly_controlled_skeleton_median_px": sp_median_px,
                "simpoly_controlled_skeleton_median_nm": sp_median_nm,
                "simpoly_source_compat_status": res_source.status,
                "simpoly_source_compat_gaussian_center_px": res_source.gaussian_center_px,
                "fathom_section_mean_nm": None,
                "fathom_section_median_nm": None,
                "fathom_fiber_median_nm": None,
                "manual_reviewed_mean_nm": None,
                "manual_reviewed_median_nm": None,
                "status": res_ctrl.status,
            }
            results.append(row_data)

            print(
                f"Image: {p.name:25s} | SIMPoly Ctrl Peak: {sp_center_nm if sp_center_nm else 'N/A'} nm | Mean: {sp_mean_nm if sp_mean_nm else 'N/A'} nm"
            )
        except Exception as exc:
            print(f"Error processing {p.name}: {exc}")

    out_json = Path(".validation/simpoly-runs/zeiss_comparison_exploratory.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Comparison report saved to {out_json.resolve()}")


if __name__ == "__main__":
    compare_simpoly_and_fathom_on_zeiss()
