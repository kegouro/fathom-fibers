from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fathom_fibers_quick.auto_roi import (
    analyze_roi,
    generate_diagnostic_panel,
    generate_diagnostic_rois,
    get_preset_for_calibration,
)
from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.zeiss import detect_footer, load_image_document, load_pixels


def run_campaign(
    search_dirs: list[Path],
    output_dir: Path,
) -> Path:
    """Executes diagnostic campaign on available TIFFs or synthetic SEM images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Locate valid Zeiss TIFF files
    valid_tiff_files: list[Path] = []
    for s_dir in search_dirs:
        if s_dir.exists():
            for p in list(s_dir.glob("*.tif")) + list(s_dir.glob("*.tiff")) + list(s_dir.rglob("*.tif")) + list(s_dir.rglob("*.tiff")):
                if p.name.startswith("._") or "synthetic" in str(p):
                    continue
                try:
                    load_image_document(p)
                    valid_tiff_files.append(p)
                except (ValueError, KeyError, OSError, RuntimeError):
                    pass

    valid_tiff_files = sorted(set(valid_tiff_files))

    is_synthetic = False
    if not valid_tiff_files:
        is_synthetic = True
        synth_dir = output_dir / "synthetic_inputs"
        synth_dir.mkdir(parents=True, exist_ok=True)

        for _i, (name, width_px, nm_px) in enumerate([
            ("synth_high_mag.tif", 600, 2.0),
            ("synth_mid_mag.tif", 600, 10.0),
            ("synth_low_mag.tif", 600, 40.0),
        ]):
            img_arr = np.full((width_px, width_px), 25.0, dtype=np.float32)
            img_arr[50:550, 150:175] = 210.0
            img_arr[50:550, 380:400] = 210.0
            img_p = synth_dir / name
            Image.fromarray(img_arr.astype(np.uint8)).save(img_p)
            valid_tiff_files.append(img_p)

    campaign_data = []
    csv_rows = []

    for tiff_path in valid_tiff_files:
        try:
            if is_synthetic:
                nm_px = 2.0 if "high" in tiff_path.name else (10.0 if "mid" in tiff_path.name else 40.0)
                cal = Calibration(nm_px * 1e-9, nm_px * 1e-9, "synthetic")
                _source_image, gray = load_pixels(tiff_path)
                footer_b = None
            else:
                doc, _source_image, gray = load_image_document(tiff_path)
                cal = doc.calibration
                footer_b = doc.footer_bounds or detect_footer(gray)

            preset = get_preset_for_calibration(cal)
            rois = generate_diagnostic_rois(gray.shape[:2], footer_bounds=footer_b, n_rois=4)

            img_summary = {
                "file": tiff_path.name,
                "preset": preset.name,
                "nm_per_px": cal.pixel_size_x_m * 1e9,
                "rois_evaluated": len(rois),
                "candidates": [],
            }

            for r_idx, roi_box in enumerate(rois, start=1):
                candidates, summary = analyze_roi(
                    gray, roi_box, cal, footer_bounds=footer_b, preset=preset
                )

                # Generate diagnostic panel
                panel_img = generate_diagnostic_panel(
                    gray, roi_box, candidates, summary, title_info=f"{tiff_path.name} | ROI #{r_idx}"
                )
                panel_name = f"{tiff_path.stem}_roi_{r_idx:02d}.png"
                panel_img.save(output_dir / panel_name)

                for cand in candidates:
                    med_um = cand.median_width_m * 1e6 if cand.median_width_m else None
                    csv_rows.append({
                        "file": tiff_path.name,
                        "roi_index": r_idx,
                        "candidate_id": cand.candidate_id,
                        "confidence_score": cand.confidence_score,
                        "confidence_level": cand.confidence_level,
                        "median_width_um": med_um,
                        "quality_flags": ";".join(sorted(cand.quality_flags)),
                        "threshold_method": cand.threshold_method,
                        "status": cand.status,
                    })

                img_summary["candidates"].append({
                    "roi_index": r_idx,
                    "roi_bbox": list(roi_box),
                    "total_components": summary.total_components,
                    "measurable": summary.measurable_candidates,
                    "high_confidence": summary.high_confidence,
                    "excluded": summary.excluded,
                })

            campaign_data.append(img_summary)
        except Exception as exc:
            print(f"Error processing {tiff_path.name}: {exc}")

    # Save CSV
    csv_path = output_dir / "candidates_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as h:
        if csv_rows:
            writer = csv.DictWriter(h, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Save JSON
    json_path = output_dir / "campaign_results.json"
    json_path.write_text(json.dumps(campaign_data, indent=2), encoding="utf-8")

    # Save HTML Contact Sheet
    html_path = output_dir / "index.html"
    panels_list = sorted(output_dir.glob("*.png"))
    panels_html = "\n".join(
        f'<div class="card"><h3>{p.name}</h3><img src="{p.name}" alt="{p.name}"></div>'
        for p in panels_list
    )

    verif_badge = "INSPECCIONADO SINTÉTICAMENTE" if is_synthetic else "INSPECCIONADO EN ZEISS REAL"

    html_content = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Fathom Auto-ROI Campaign Report</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; background: #1a1a1a; color: #eee; }}
.badge {{ background: #0072B2; color: #fff; padding: 0.4rem 0.8rem; border-radius: 4px; font-weight: bold; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
.card {{ background: #2a2a2a; border: 1px solid #444; border-radius: 8px; padding: 1rem; }}
img {{ max-width: 100%; border-radius: 4px; border: 1px solid #555; }}
</style></head>
<body>
<h1>Fathom Fibers Quick — Informe de Campaña Auto-ROI</h1>
<p><span class="badge">{verif_badge}</span> <strong>Imágenes evaluadas:</strong> {len(valid_tiff_files)}</p>
<div class="grid">{panels_html}</div>
</body></html>"""
    html_path.write_text(html_content, encoding="utf-8")

    return html_path


def main() -> None:
    search_dirs = [
        Path("local_data/zeiss"),
        Path("/home/kegouro/HIBRIS/Workshop ⁄ Proyectos"),
    ]
    output_dir = Path("local_results/auto_roi_campaign")
    res_path = run_campaign(search_dirs, output_dir)
    print(f"Campaña completada. Informe disponible en: {res_path}")


if __name__ == "__main__":
    main()
