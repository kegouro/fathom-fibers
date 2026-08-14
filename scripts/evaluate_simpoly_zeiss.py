from __future__ import annotations

from pathlib import Path

from fathom_fibers_quick.oracles.report import generate_simpoly_validation_report
from fathom_fibers_quick.oracles.simpoly import run_synthetic_benchmark_suite
from fathom_fibers_quick.simpoly_compat import run_simpoly_pipeline
from fathom_fibers_quick.zeiss import load_image_document


def evaluate_zeiss_campaign():
    tiff_dir = Path("local_data/zeiss")
    if not tiff_dir.exists():
        print("local_data/zeiss directory not found. Skipping Zeiss evaluation.")
        return

    primary_names = ["PVDF Jose_02.tif", "PVDF Jose_09.tif", "PVDF Jose_13.tif"]
    borderline_names = ["PVDF Jose_01.tif", "PVDF Jose_15.tif", "PVDF Jose_16.tif"]
    unsupported_names = ["PVDF Jose_03.tif", "PVDF Jose_04.tif", "PVDF Jose_05.tif", "PVDF Jose_10.tif", "PVDF Jose_14.tif"]

    all_tiffs = list(tiff_dir.rglob("*.tif")) + list(tiff_dir.rglob("*.tiff"))

    zeiss_results = []
    for p in all_tiffs:
        name = p.name
        if name in primary_names:
            domain = "SUPPORTED"
        elif name in borderline_names:
            domain = "BORDERLINE"
        elif name in unsupported_names:
            domain = "UNSUPPORTED"
        else:
            domain = "BORDERLINE"

        try:
            doc, _img, gray = load_image_document(p)
            res = run_simpoly_pipeline(gray, doc.calibration, doc.footer_bounds)

            sim_center_px = res["gaussian_center_px"]
            sim_center_nm = sim_center_px * (doc.calibration.pixel_size_x_m * 1e9)

            fathom_val_str = "NOT_MEASURED"
            manual_val_str = "NOT_MEASURED"

            zeiss_results.append({
                "image_name": name,
                "domain": domain,
                "simpoly_px": f"{sim_center_px:.1f} px ({sim_center_nm:.1f} nm)",
                "fathom_px": fathom_val_str,
                "manual_5x5_px": manual_val_str,
                "notes": f"Cal: {doc.calibration.pixel_size_x_m*1e9:.2f} nm/px. Segmented frac: {res['segmented_fraction']*100:.1f}%",
            })
        except Exception as exc:
            zeiss_results.append({
                "image_name": name,
                "domain": "ANALYSIS_FAILED",
                "simpoly_px": "ANALYSIS_FAILED",
                "fathom_px": "NOT_MEASURED",
                "manual_5x5_px": "NOT_MEASURED",
                "notes": f"Error during processing: {exc}",
            })

    # Run synthetic benchmark suite
    _runs, _comps, synth_summary = run_synthetic_benchmark_suite()

    # Generate HTML validation report at .validation/simpoly-runs/latest/index.html
    report_path = Path(".validation/simpoly-runs/latest/index.html")
    generate_simpoly_validation_report(synth_summary, zeiss_results, report_path)
    print(f"Validation report successfully written to {report_path.resolve()}")


if __name__ == "__main__":
    evaluate_zeiss_campaign()
