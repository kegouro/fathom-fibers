from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from fathom_fibers_quick.auto_roi import (
    analyze_roi,
    generate_diagnostic_panel,
    generate_diagnostic_rois,
    get_preset_for_calibration,
)
from fathom_fibers_quick.zeiss import detect_footer, load_image_document


def run_real_campaign(
    tiff_dir: Path,
    output_dir: Path,
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    """Executes diagnostic campaign on real Zeiss SEM images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tiff_paths = sorted([
        p for p in tiff_dir.rglob("*")
        if p.suffix.lower() in {".tif", ".tiff"} and not p.name.startswith("._")
    ])

    if not tiff_paths:
        raise RuntimeError(f"No real TIFF files found in {tiff_dir}")

    inventory_rows = []
    roi_rows = []
    campaign_summary = []

    for tiff_path in tiff_paths:
        doc, _source_img, gray = load_image_document(tiff_path)
        cal = doc.calibration
        nm_px = cal.pixel_size_x_m * 1e9
        preset = get_preset_for_calibration(cal)
        footer_b = doc.footer_bounds or detect_footer(gray)
        footer_h = (footer_b[1] - footer_b[0]) if footer_b else 0

        meta = doc.metadata or {}
        std_val = float(gray.std())
        p2, p98 = float(np.percentile(gray, 2)), float(np.percentile(gray, 98))

        inv_row = {
            "file": tiff_path.name,
            "width_px": gray.shape[1],
            "height_px": gray.shape[0],
            "nm_per_px": round(nm_px, 3),
            "magnification": meta.get("ap_mag", "N/A"),
            "kv": meta.get("ap_actualkv", "N/A"),
            "working_distance_mm": meta.get("ap_wd", "N/A"),
            "detector": meta.get("dp_detector_channel", "N/A"),
            "footer_height_px": footer_h,
            "intensity_min": int(gray.min()),
            "intensity_max": int(gray.max()),
            "contrast_std": round(std_val, 2),
            "contrast_p2_p98": round(p98 - p2, 2),
            "proposed_preset": preset.name,
        }
        inventory_rows.append(inv_row)

        rois = generate_diagnostic_rois(gray.shape[:2], footer_bounds=footer_b, n_rois=4)

        img_record: dict[str, Any] = {
            "file": tiff_path.name,
            "nm_per_px": nm_px,
            "preset": preset.name,
            "rois": [],
        }

        for r_idx, roi_box in enumerate(rois, start=1):
            t0 = time.perf_counter()
            candidates, summary = analyze_roi(
                gray, roi_box, cal, footer_bounds=footer_b, preset=preset
            )
            elapsed_sec = time.perf_counter() - t0

            # Classification of ROI result
            if summary.resolution_status == "RESOLUTION_INSUFFICIENT":
                roi_class = "UNSUITABLE_FOR_AUTOMATIC_WIDTH"
            elif summary.measurable_candidates == 0:
                roi_class = "NO_CANDIDATES"
            elif summary.total_components > 40:
                roi_class = "MANUAL_REVIEW_REQUIRED"
            elif summary.high_confidence > 0:
                roi_class = "USABLE"
            else:
                roi_class = "PARTIALLY_USABLE"

            panel_img = generate_diagnostic_panel(
                gray, roi_box, candidates, summary, title_info=f"{tiff_path.name} | ROI #{r_idx} | {roi_class}"
            )
            panel_name = f"{tiff_path.stem}_roi_{r_idx:02d}.png"
            panel_img.save(output_dir / panel_name)

            roi_row = {
                "file": tiff_path.name,
                "roi_index": r_idx,
                "roi_bbox": f"{roi_box[0]},{roi_box[1]},{roi_box[2]},{roi_box[3]}",
                "preset": preset.name,
                "threshold_method": summary.threshold_method_used,
                "resolution_status": summary.resolution_status,
                "classification": roi_class,
                "total_components": summary.total_components,
                "measurable_candidates": summary.measurable_candidates,
                "high_confidence": summary.high_confidence,
                "needs_review": summary.needs_review,
                "excluded": summary.excluded,
                "execution_time_sec": round(elapsed_sec, 4),
            }
            roi_rows.append(roi_row)

            img_record["rois"].append({
                "roi_index": r_idx,
                "roi_bbox": list(roi_box),
                "classification": roi_class,
                "summary": {
                    "total_components": summary.total_components,
                    "measurable": summary.measurable_candidates,
                    "high_confidence": summary.high_confidence,
                    "excluded": summary.excluded,
                },
                "candidates_count": len(candidates),
                "elapsed_sec": round(elapsed_sec, 4),
            })

        campaign_summary.append(img_record)

    # 1. Write inventory.csv
    inv_csv_path = output_dir / "inventory.csv"
    with inv_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(inventory_rows)

    # 2. Write roi_results.csv
    roi_csv_path = output_dir / "roi_results.csv"
    with roi_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(roi_rows[0].keys()))
        writer.writeheader()
        writer.writerows(roi_rows)

    # 3. Write summary.json
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(json.dumps(campaign_summary, indent=2), encoding="utf-8")

    # 4. Write index.html
    html_path = output_dir / "index.html"
    panels_list = sorted(output_dir.glob("*.png"))
    cards_html = []
    for p in panels_list:
        cards_html.append(
            f'<div class="card"><h3>{p.name}</h3><img src="{p.name}" alt="{p.name}"></div>'
        )

    html_content = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Fathom Fibers — Campaña Real Zeiss PVDF</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1600px; margin: 2rem auto; padding: 0 1rem; background: #121212; color: #eee; }}
.badge {{ background: #0080FF; color: #fff; padding: 0.4rem 0.8rem; border-radius: 4px; font-weight: bold; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 1.5rem; margin-top: 1.5rem; }}
.card {{ background: #222; border: 1px solid #444; border-radius: 8px; padding: 1rem; }}
img {{ max-width: 100%; border-radius: 4px; border: 1px solid #555; }}
</style></head>
<body>
<h1>Fathom Fibers Quick — Evaluación en Micrografías Reales Zeiss SEM (PVDF Electrospinning)</h1>
<p><span class="badge">INSPECCIONADO VISUALMENTE EN TIFF ZEISS</span> <strong>Imágenes evaluadas:</strong> {len(tiff_paths)} | <strong>ROIs totales:</strong> {len(roi_rows)}</p>
<div class="grid">
{''.join(cards_html)}
</div>
</body></html>"""

    html_path.write_text(html_content, encoding="utf-8")
    return html_path, inventory_rows, roi_rows


def main() -> None:
    tiff_dir = Path("local_data/zeiss")
    output_dir = Path("local_results/real_zeiss_campaign")
    res_path, inv, rois = run_real_campaign(tiff_dir, output_dir)
    print(f"Campaña real completada. Reporte generado en: {res_path}")
    print(f"Imágenes procesadas: {len(inv)}, ROIs analizadas: {len(rois)}")


if __name__ == "__main__":
    main()
