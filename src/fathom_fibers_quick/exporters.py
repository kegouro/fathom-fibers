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
)
from .hierarchy import compute_hierarchical_statistics, hierarchical_bootstrap
from .measurement_records import MeasurementRecord
from .model import Project
from .provenance import SoftwareProvenance
from .repeatability import compare_automatic_and_manual

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
    """Generates complete scientific HTML report according to Section 13."""
    path = Path(path)
    hier_stats = compute_hierarchical_statistics(project)
    boot_stats = hierarchical_bootstrap(project, n_bootstraps=500, seed=42)
    auto_manual_pairs = compare_automatic_and_manual(project.records)
    provenance = SoftwareProvenance.from_project(project).to_dict()

    per_fiber = fiber_statistics(project.measurements)

    def format_val(val: float | None) -> str:
        return "—" if val is None else format_length_m(float(val))

    fiber_rows = "\n".join(
        f"<tr><td>{html.escape(fid)}</td><td>{values['n']}</td>"
        f"<td>{values['mean_m'] * 1e6:.3f} µm</td><td>{values['median_m'] * 1e6:.3f} µm</td>"
        f"<td>{values['min_m'] * 1e6:.3f} µm</td><td>{values['max_m'] * 1e6:.3f} µm</td></tr>"
        for fid, values in sorted(per_fiber.items())
    )

    auto_manual_rows = "\n".join(
        f"<tr><td>{html.escape(p['auto_id'])}</td><td>{html.escape(p['manual_id'])}</td>"
        f"<td>{p['auto_value_m'] * 1e6:.3f} µm</td><td>{p['manual_value_m'] * 1e6:.3f} µm</td>"
        f"<td>{p['absolute_difference_m'] * 1e6:.3f} µm</td><td>{p['relative_difference'] * 100:.1f}%</td></tr>"
        for p in auto_manual_pairs
    ) if auto_manual_pairs else "<tr><td colspan='6'>No hay pares automático-manual comparados.</td></tr>"

    prov_json = json.dumps(provenance, indent=2, ensure_ascii=False)
    meta_json = json.dumps(project.image.metadata, indent=2, ensure_ascii=False)

    boot_mean_str = format_val(boot_stats["bootstrap_mean_m"])
    boot_ci_low_str = format_val(boot_stats["ci_lower_m"])
    boot_ci_high_str = format_val(boot_stats["ci_upper_m"])

    sec_st = hier_stats["section_level"]
    fib_st = hier_stats["fiber_level"]
    img_st = hier_stats["image_level"]
    smp_st = hier_stats["sample_level"]

    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Fathom Fibers Scientific Report</title>
<style>body{{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.5}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccc;padding:.5rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#f0f4f8}}img{{max-width:100%;border:1px solid #aaa;margin:1rem 0}}code,pre{{background:#f4f4f4;padding:.6rem;border-radius:4px;overflow:auto}}.alert{{background:#e8f4f8;border-left:4px solid #0072b2;padding:0.8rem;margin:1rem 0}}.disclaimer{{background:#fff8e6;border-left:4px solid #e69f00;padding:0.8rem;margin:1rem 0;font-size:0.9rem}}</style></head>
<body>
<h1>Fathom Fibers Quick 0.3 — Informe de Medición Científica Reproducible</h1>
<div class="alert">
<p><strong>Muestra:</strong> {html.escape(project.sample_name)} ({html.escape(project.sample_id)})</p>
<p><strong>Imagen fuente:</strong> {html.escape(project.image.path)}</p>
<p><strong>Calibración:</strong> {project.image.calibration.pixel_size_x_m * 1e9:.4f} nm/px ({html.escape(project.image.calibration.source)})</p>
<p><strong>Protocolo activo:</strong> {html.escape(project.active_protocol_id)}</p>
</div>

<h2>1. Resumen Estadístico Jerárquico (4 Niveles)</h2>
<table>
<thead><tr><th>Nivel</th><th>N Unidades</th><th>Media</th><th>Mediana</th><th>SD</th><th>IQR</th><th>Mín. Crudo</th><th>Máx. Crudo</th><th>P05</th><th>P95</th></tr></thead>
<tbody>
<tr><td><strong>Sección</strong></td><td>{sec_st['n']}</td><td>{format_val(sec_st['mean'])}</td><td>{format_val(sec_st['median'])}</td><td>{format_val(sec_st['sd'])}</td><td>{format_val(sec_st['iqr'])}</td><td>{format_val(sec_st['min'])}</td><td>{format_val(sec_st['max'])}</td><td>{format_val(sec_st['p05'])}</td><td>{format_val(sec_st['p95'])}</td></tr>
<tr><td><strong>Fibra</strong></td><td>{fib_st['n']}</td><td>{format_val(fib_st['mean'])}</td><td>{format_val(fib_st['median'])}</td><td>{format_val(fib_st['sd'])}</td><td>{format_val(fib_st['iqr'])}</td><td>{format_val(fib_st['min'])}</td><td>{format_val(fib_st['max'])}</td><td>{format_val(fib_st['p05'])}</td><td>{format_val(fib_st['p95'])}</td></tr>
<tr><td><strong>Imagen</strong></td><td>{img_st['n']}</td><td>{format_val(img_st['mean'])}</td><td>{format_val(img_st['median'])}</td><td>{format_val(img_st['sd'])}</td><td>{format_val(img_st['iqr'])}</td><td>{format_val(img_st['min'])}</td><td>{format_val(img_st['max'])}</td><td>{format_val(img_st['p05'])}</td><td>{format_val(img_st['p95'])}</td></tr>
<tr><td><strong>Muestra</strong></td><td>{smp_st['n_fibers']} fibras ({smp_st['n_sections']} sec)</td><td>{format_val(smp_st['mean'])}</td><td>{format_val(smp_st['median'])}</td><td>{format_val(smp_st['sd'])}</td><td>{format_val(smp_st['iqr'])}</td><td>{format_val(smp_st['min'])}</td><td>{format_val(smp_st['max'])}</td><td>{format_val(smp_st['p05'])}</td><td>{format_val(smp_st['p95'])}</td></tr>
</tbody>
</table>

<h3>Bootstrap Jerárquico Determinista (500 réplicas)</h3>
<ul>
<li>Media Bootstrap: {boot_mean_str}</li>
<li>IC 95%: {boot_ci_low_str} – {boot_ci_high_str}</li>
</ul>

<h2>2. Imagen Aotada de Micrografía</h2>
<img src="{html.escape(annotated_name)}" alt="Imagen anotada">

<h2>3. Comparación Automático vs Referencia Manual Revisada</h2>
<table>
<thead><tr><th>ID Auto</th><th>ID Manual</th><th>Valor Auto</th><th>Valor Manual</th><th>Diferencia Absoluta</th><th>Diferencia Relativa</th></tr></thead>
<tbody>{auto_manual_rows}</tbody>
</table>

<h2>4. Estadísticas por Fibra Individual (µm)</h2>
<table><thead><tr><th>Fibra</th><th>N Secciones</th><th>Media</th><th>Mediana</th><th>Mínimo Crudo</th><th>Máximo Crudo</th></tr></thead>
<tbody>{fiber_rows}</tbody></table>

<h2>5. Software Provenance & Metadata</h2>
<pre>{html.escape(prov_json)}</pre>
<pre>{html.escape(meta_json)}</pre>

<div class="disclaimer">
<h3>Declaraciones de Limitaciones Científicas Obligatorias</h3>
<ul>
<li>ⓘ <strong>Las mediciones representan geometría proyectada 2D.</strong> La interpretación física depende de calibración, resolución y geometría de la muestra.</li>
<li>ⓘ <strong>Los resultados automáticos son propuestas revisadas.</strong> Las referencias manuales representan la 'Referencia manual revisada', no un ground truth absoluto.</li>
<li>ⓘ <strong>La ausencia de uncertainty no equivale a uncertainty cero.</strong> Los componentes de incertidumbre se expresan explícitamente cuando existe evidencia empírica.</li>
</ul>
</div>
</body></html>"""

    path.write_text(document, encoding="utf-8")
    return path
