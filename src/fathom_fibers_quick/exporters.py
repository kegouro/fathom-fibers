from __future__ import annotations

import csv
import html
import json
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
from .model import Project

GROUP_COLORS = [
    (0, 114, 178),
    (230, 159, 0),
    (0, 158, 115),
    (204, 121, 167),
]


def export_csv(project: Project, path: str | Path) -> Path:
    path = Path(path)
    rows = []
    for m in project.measurements:
        rows.append({
            "measurement_id": m.measurement_id,
            "fiber_id": m.fiber_id,
            "method": m.method,
            "accepted": m.accepted,
            "width_m": m.width_m,
            "width_um": m.width_m * 1e6,
            "width_nm": m.width_m * 1e9,
            "p1_x_px": m.p1[0],
            "p1_y_px": m.p1[1],
            "p2_x_px": m.p2[0],
            "p2_y_px": m.p2[1],
            "group": m.group,
            "confidence": m.confidence,
            "defect": m.defect,
            "note": m.note,
            "created_at": m.created_at,
            "source_image": project.image.path,
            "source_sha256": project.image.source_sha256,
            "pixel_size_x_m": project.image.calibration.pixel_size_x_m,
            "pixel_size_y_m": project.image.calibration.pixel_size_y_m,
            "calibration_source": project.image.calibration.source,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["measurement_id"])
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

    # Shaded footer if present
    if project.image.footer_bounds:
        y0, y1 = project.image.footer_bounds
        draw.rectangle([0, y0, image.width, y1], fill=(40, 40, 40, 180), outline=(255, 100, 100), width=2)
        draw.text((10, y0 + 10), "FOOTER EXCLUIDO", fill=(255, 200, 200), font=font)

    extrema_by_m: dict[str, list[str]] = {}
    if show_extrema:
        fibers = {m.fiber_id for m in project.measurements}
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
            parts.append(measurement.fiber_id)
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

    # Header summary overlay
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

    # Legend if requested
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
