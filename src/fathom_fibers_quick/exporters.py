from __future__ import annotations

import csv
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .analysis import (
    fiber_level_summary,
    fiber_statistics,
    format_length_m,
    get_fiber_extrema,
    section_level_summary,
)
from .measurement_records import MeasurementRecord
from .model import Project

GROUP_COLORS = [
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
]


def export_csv(
    project: Project,
    path: str | Path,
    records: Sequence[MeasurementRecord] | None = None,
) -> Path:
    """Exports unified measurement records to CSV according to Section 16."""
    path = Path(path)
    target_records = records if records is not None else project.records

    rows = []
    for r in target_records:
        val_m = r.primary_value
        unit = r.primary_unit

        length_m = r.values.get("length_m")
        proj_w_m = r.values.get("width_m") if r.kind == "PROJECTED_WIDTH" else None
        angle_deg = r.values.get("interior_angle_deg")
        area_m2 = r.values.get("area_m2")
        perimeter_m = r.values.get("perimeter_m")
        tortuosity = r.values.get("tortuosity")
        mean_int = r.values.get("mean_intensity_au") or r.values.get("mean_intensity")
        std_int = r.values.get("std_intensity_au")

        rows.append({
            "measurement_id": r.measurement_id,
            "name": r.name,
            "kind": r.kind.value if hasattr(r.kind, "value") else str(r.kind),
            "status": r.status.value if hasattr(r.status, "value") else str(r.status),
            "source": r.source.value if hasattr(r.source, "value") else str(r.source),
            "image_id": r.image_id or project.image.path,
            "sample_id": r.sample_id or "",
            "fiber_id": r.fiber_id or "",
            "roi_id": r.roi_id or "",
            "primary_value": val_m if val_m is not None else "",
            "primary_unit": unit,
            "tags": ", ".join(r.tags),
            "notes": r.notes,
            "quality_flags": ";".join(r.quality_flags),
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "length_m": length_m if length_m is not None else "",
            "projected_width_m": proj_w_m if proj_w_m is not None else "",
            "angle_deg": angle_deg if angle_deg is not None else "",
            "area_m2": area_m2 if area_m2 is not None else "",
            "perimeter_m": perimeter_m if perimeter_m is not None else "",
            "tortuosity": tortuosity if tortuosity is not None else "",
            "mean_intensity": mean_int if mean_int is not None else "",
            "std_intensity": std_int if std_int is not None else "",
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "measurement_id",
        "name",
        "kind",
        "status",
        "source",
        "image_id",
        "sample_id",
        "fiber_id",
        "roi_id",
        "primary_value",
        "primary_unit",
        "tags",
        "notes",
        "quality_flags",
        "created_at",
        "updated_at",
        "length_m",
        "projected_width_m",
        "angle_deg",
        "area_m2",
        "perimeter_m",
        "tortuosity",
        "mean_intensity",
        "std_intensity",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def export_profile_csv(record: MeasurementRecord, path: str | Path) -> Path:
    """Exports raw and smoothed intensity profile data points to CSV."""
    path = Path(path)
    dists = record.values.get("distance_m", [])
    raws = record.values.get("profile_raw", [])
    smooths = record.values.get("profile_smoothed", [])

    rows = []
    for i in range(len(raws)):
        d_m = dists[i] if i < len(dists) else i
        r_v = raws[i]
        s_v = smooths[i] if i < len(smooths) else r_v
        rows.append({
            "distance_m": d_m,
            "raw_intensity": r_v,
            "averaged_intensity": r_v,
            "smoothed_intensity": s_v,
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["distance_m", "raw_intensity", "averaged_intensity", "smoothed_intensity"])
        writer.writeheader()
        writer.writerows(rows)

    return path


def export_annotated(
    project: Project,
    source_image: Image.Image,
    path: str | Path,
    show_ids: bool = True,
    show_values: bool = True,
    show_extrema: bool = False,
    show_defects: bool = True,
    show_legend: bool = True,
    show_candidates: bool = False,
    candidates: list[Any] | None = None,
) -> Path:
    path = Path(path)
    image = source_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=14)

    if project.image.footer_bounds:
        y0, y1 = project.image.footer_bounds
        draw.rectangle([0, y0, image.width, y1], fill=(40, 40, 40, 180), outline=(255, 100, 100), width=2)
        draw.text((10, y0 + 10), "FOOTER EXCLUIDO", fill=(255, 200, 200), font=font)

    extrema_by_m: dict[str, list[str]] = {}
    if show_extrema:
        fibers = {m.fiber_id for m in project.measurements if m.fiber_id}
        for fid in fibers:
            f_ext = get_fiber_extrema(project.measurements, fid)
            for mid, labels in f_ext.items():
                extrema_by_m.setdefault(mid, []).extend(labels)

    for measurement in project.measurements:
        if measurement.accepted:
            color = GROUP_COLORS[(measurement.group or 0) % len(GROUP_COLORS)]
            width = 4
        else:
            color = (130, 130, 130)
            width = 2

        draw.line([measurement.p1, measurement.p2], fill=color, width=width)
        radius = 5
        for x, y in (measurement.p1, measurement.p2):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=(255, 255, 255), width=2)

        parts = []
        if show_ids:
            parts.append(measurement.fiber_id or measurement.measurement_id)
        if show_values:
            parts.append(f"{measurement.width_m * 1e6:.3f} µm")
        if show_defects and measurement.defect != "None":
            parts.append(f"[{measurement.defect}]")
        if show_extrema and measurement.measurement_id in extrema_by_m:
            ext_label = "/".join(extrema_by_m[measurement.measurement_id])
            parts.append(f"({ext_label})")

        if parts:
            center = measurement.center
            label = " ".join(parts)
            draw.text(
                (center[0] + 8, center[1] + 8),
                label,
                fill=(255, 255, 0),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )

    f_stats = fiber_level_summary(project.measurements)
    if f_stats["n_fibers"] > 0:
        med_str = format_length_m(float(f_stats["median_m"])) if f_stats["median_m"] else "—"
        p05_str = format_length_m(float(f_stats["p05_m"])) if f_stats["p05_m"] else "—"
        p95_str = format_length_m(float(f_stats["p95_m"])) if f_stats["p95_m"] else "—"
        summary_text = (
            f"Fathom Fibers | Sample: {Path(project.image.path).name}\n"
            f"Fibras: {f_stats['n_fibers']} | Mediana por fibra: {med_str} | P05-P95: {p05_str} - {p95_str}"
        )
        draw.rectangle([10, 10, 520, 50], fill=(0, 0, 0, 180), outline=(255, 255, 255))
        draw.text((16, 14), summary_text, fill=(255, 255, 255), font=font)

    if show_legend:
        group_names = project.group_names
        legend_y = 60
        draw.rectangle([10, legend_y, 220, legend_y + 25 + max(1, len(group_names)) * 20], fill=(0, 0, 0, 180), outline=(200, 200, 200))
        draw.text((16, legend_y + 4), "Grupos de tamaño:", fill=(255, 255, 255), font=font)
        for g_idx in range(max(1, len(group_names))):
            g_color = GROUP_COLORS[g_idx % len(GROUP_COLORS)]
            g_name = group_names.get(g_idx, f"Grupo {g_idx + 1}")
            item_y = legend_y + 24 + g_idx * 20
            draw.rectangle([16, item_y + 2, 28, item_y + 14], fill=g_color, outline=(255, 255, 255))
            draw.text((34, item_y), g_name, fill=(255, 255, 255), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def export_html_report(project: Project, annotated_name: str, path: str | Path) -> Path:
    path = Path(path)
    f_stats = fiber_level_summary(project.measurements)
    s_stats = section_level_summary(project.measurements)
    per_fiber = fiber_statistics(project.measurements)

    def show(stats: dict[str, float | int | None], key: str) -> str:
        value = stats[key]
        return "—" if value is None else format_length_m(float(value))

    rows = "\n".join(
        f"<tr><td>{html.escape(fid)}</td><td>{values['n']}</td>"
        f"<td>{values['mean_m'] * 1e6:.3f}</td><td>{values['median_m'] * 1e6:.3f}</td>"
        f"<td>{values['min_m'] * 1e6:.3f}</td><td>{values['max_m'] * 1e6:.3f}</td></tr>"
        for fid, values in sorted(per_fiber.items())
    )
    metadata = json.dumps(project.image.metadata, indent=2, ensure_ascii=False)
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Fathom Fibers Report</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;border:1px solid #aaa}}code,pre{{background:#f4f4f4;padding:.5rem;overflow:auto}}</style></head>
<body><h1>Fathom Fibers Quick — Informe de Ancho Proyectado</h1>
<p><strong>Imagen:</strong> {html.escape(project.image.path)}</p>
<p><strong>Calibración:</strong> {project.image.calibration.pixel_size_x_m * 1e9:.4f} nm/px ({html.escape(project.image.calibration.source)})</p>
<h2>1. Resumen por Fibra (medianas por fibra)</h2><ul>
<li>Fibras identificadas: {f_stats['n_fibers']}</li>
<li>Mediciones válidas totales: {f_stats['n_measurements']}</li>
<li>Media de medianas: {show(f_stats, 'mean_m')}</li>
<li>Mediana global: {show(f_stats, 'median_m')}</li>
<li>Mínimo crudo: {show(f_stats, 'min_m')}</li>
<li>Máximo crudo: {show(f_stats, 'max_m')}</li>
<li>P05–P95: {show(f_stats, 'p05_m')} – {show(f_stats, 'p95_m')}</li></ul>
<h2>2. Distribución de Secciones Locales</h2><ul>
<li>Secciones totales aceptadas: {s_stats['n_measurements']}</li>
<li>Media por sección: {show(s_stats, 'mean_m')}</li>
<li>Mediana por sección: {show(s_stats, 'median_m')}</li>
<li>Mínimo crudo: {show(s_stats, 'min_m')}</li>
<li>Máximo crudo: {show(s_stats, 'max_m')}</li>
<li>P05–P95: {show(s_stats, 'p05_m')} – {show(s_stats, 'p95_m')}</li></ul>
<h2>Imagen anotada</h2><img src="{html.escape(annotated_name)}" alt="Imagen anotada">
<h2>Estadísticas por fibra individual (µm)</h2><table><thead><tr><th>Fibra</th><th>N</th><th>Media</th><th>Mediana</th><th>Mínimo crudo</th><th>Máximo crudo</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Metadata instrumental</h2><pre>{html.escape(metadata)}</pre>
<p><small>Advertencia: la herramienta mide <strong>ancho proyectado</strong> en la micrografía 2D. Interpretarlo como diámetro verdadero asume fibras cilíndricas, aisladas y paralelas al plano de imagen. Cruces, inclinación, cintas y fusiones requieren revisión humana.</small></p>
</body></html>"""
    path.write_text(document, encoding="utf-8")
    return path
