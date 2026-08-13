"""ORIENTED_RIBBON_V1 headless validation: synthetic truth + 16-TIFF report.

Writes ``.validation/oriented-ribbon-v1/latest/`` with an HTML summary, JSON/CSV
payloads and minimal diagnostic figures.  The 16-image numbers come from the
workspace full caches (regenerated with the remeasurement arrays); synthetic
truth is recomputed from the shared ribbon phantoms.
"""

from __future__ import annotations

import csv
import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..core.methods import MethodId, MethodResult
from ..workspace import WorkspaceCache, load_workspace_dataset
from .ribbon_phantoms import (
    arc_phantom,
    rotated_phantom,
    run_case,
    straight_phantom,
)

REPORT_ROOT = ".validation/oriented-ribbon-v1"
LATEST_DIR = "latest"
METHODS = (
    "FATHOM_FIELD_PAIRED_EDGE_DIAMETER",
    "FATHOM_FIELD_PROFILE_DIAMETER",
    "FATHOM_FIELD_REFINED_EDT_DIAMETER",
    "FATHOM_FIELD_REFINED_EDGE_DIAMETER",
    "FATHOM_FIELD_REFINED_PROFILE_DIAMETER",
)


def synthetic_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    builders = [
        ("straight", straight_phantom, {}),
        ("straight_noisy", straight_phantom, {"noise_seed": 42}),
        ("curved", arc_phantom, {}),
        ("curved_noisy", arc_phantom, {"noise_seed": 11}),
        ("variable_radius", straight_phantom, {"variable_radius": True}),
    ]
    for name, builder, kwargs in builders:
        mask, body, skeleton, samples, true_xy = builder(**kwargs)
        case = run_case(mask, body, skeleton, samples, true_xy)
        cases.append(
            {
                "case": name,
                "center_seed_mae_um": case["center_seed_mae"],
                "center_refined_mae_um": case["center_refined_mae"],
                "coverage": case["coverage"],
                "edt_raw_mae_um": case["edt_raw_mae"],
                "edt_refined_mae_um": case["edt_refined_mae"],
                "edge_raw_mae_um": case["edge_raw_mae"],
                "edge_refined_mae_um": case["edge_refined_mae"],
                "profile_raw_mae_um": case["profile_raw_mae"],
                "profile_refined_mae_um": case["profile_refined_mae"],
                "asymmetry_raw": case["asymmetry_raw"],
                "asymmetry_refined": case["asymmetry_refined"],
            }
        )
    for angle in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        mask, body, skeleton, samples, true_xy = rotated_phantom(angle)
        case = run_case(mask, body, skeleton, samples, true_xy)
        cases.append(
            {
                "case": f"rotation_{int(angle)}",
                "center_seed_mae_um": case["center_seed_mae"],
                "center_refined_mae_um": case["center_refined_mae"],
                "coverage": case["coverage"],
                "edt_raw_mae_um": case["edt_raw_mae"],
                "edt_refined_mae_um": case["edt_refined_mae"],
            }
        )
    return cases


def _field_result(comparison: Any) -> MethodResult:
    return next(
        result for result in comparison.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )


def _w1(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import wasserstein_distance

    if a.size == 0 or b.size == 0:
        return None
    return float(wasserstein_distance(a, b))


def image_metrics(comparison: Any) -> dict[str, Any]:
    field = _field_result(comparison)
    ls = field.local_samples
    statistics = field.native_statistics

    def median(key: str, mask: np.ndarray | None = None) -> float | None:
        values = ls.get(key)
        if values is None:
            return None
        values = np.asarray(values, float)
        if mask is not None:
            values = values[mask]
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None

    edge_raw = np.asarray(ls["edge_accepted"], bool)
    edge_refined = np.asarray(ls["refined_edge_accepted"], bool)
    profile_raw = np.asarray(ls["profile_accepted"], bool)
    profile_refined = np.asarray(ls["refined_profile_accepted"], bool)

    w1 = {
        "edt_edge_raw": _w1(
            ls["diameter_um"][edge_raw], ls["edge_diameter_um"][edge_raw]
        ),
        "edt_edge_refined": _w1(
            ls["refined_edt_um"][edge_refined], ls["refined_edge_um"][edge_refined]
        ),
        "edt_profile_raw": _w1(
            ls["diameter_um"][profile_raw], ls["profile_diameter_um"][profile_raw]
        ),
        "edt_profile_refined": _w1(
            ls["refined_edt_um"][profile_refined], ls["refined_profile_um"][profile_refined]
        ),
        "edge_raw_vs_refined": _w1(
            ls["edge_diameter_um"][edge_raw], ls["refined_edge_um"][edge_refined]
        ),
        "profile_raw_vs_refined": _w1(
            ls["profile_diameter_um"][profile_raw], ls["refined_profile_um"][profile_refined]
        ),
        "edge_profile_refined": _w1(
            ls["refined_edge_um"][edge_refined], ls["refined_profile_um"][profile_refined]
        ),
    }
    return {
        "case_id": statistics.get("case_id") or comparison.image_id,
        "image": comparison.image_id,
        "smooth_coverage": statistics.get("smooth_coverage_fraction"),
        "refine_coverage": statistics.get("refine_coverage_fraction"),
        "smooth_segment_count": statistics.get("smooth_segment_count"),
        "center_shift_median_um": statistics.get("refine_median_shift_um"),
        "center_shift_p90_um": statistics.get("refine_p90_shift_um"),
        "residual_shift_median_um": statistics.get("refined_residual_shift_median_um"),
        "residual_shift_p90_um": statistics.get("refined_residual_shift_p90_um"),
        "asymmetry_raw_median": statistics.get("edge_median_asymmetry"),
        "asymmetry_refined_median": statistics.get("refined_asymmetry_median"),
        "edt_raw_median_um": median("diameter_um"),
        "edt_refined_median_um": statistics.get("refined_edt_median_um"),
        "edge_raw_median_um": median("edge_diameter_um", edge_raw),
        "edge_refined_median_um": statistics.get("refined_edge_median_um"),
        "profile_raw_median_um": median("profile_diameter_um", profile_raw),
        "profile_refined_median_um": statistics.get("refined_profile_median_um"),
        "edge_raw_acceptance": statistics.get("edge_acceptance_fraction"),
        "edge_refined_acceptance": statistics.get("refined_edge_acceptance_fraction"),
        "profile_raw_acceptance": statistics.get("profile_acceptance_fraction"),
        "profile_refined_acceptance": statistics.get("refined_profile_acceptance_fraction"),
        "w1": w1,
    }


def build_oriented_ribbon_report(
    repo: str | Path,
    *,
    dataset_dir: str | Path,
) -> Path:
    """Rebuild the headless validation report and return the index path."""
    repo = Path(repo)
    output = repo / REPORT_ROOT / LATEST_DIR
    output.mkdir(parents=True, exist_ok=True)
    synthetic = synthetic_cases()
    dataset = load_workspace_dataset(dataset_dir, repo=repo)
    cache = WorkspaceCache(repo)
    missing = [image for image in dataset.images if not cache.has_full(image.stem)]
    if missing:
        raise RuntimeError(
            f"full caches missing for {len(missing)} images; run "
            "scripts/cache_workspace_results.py first"
        )
    rows = []
    for image in dataset.images:
        comparison = cache.load_comparison(image.stem)
        rows.append(image_metrics(comparison))

    (output / "synthetic_summary.json").write_text(
        json.dumps(synthetic, indent=2, default=str), encoding="utf-8"
    )
    (output / "real_16_summary.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )
    _write_csv(output / "real_16_summary.csv", rows)
    _figures(repo, output, synthetic, rows)
    _diagnostic_crops(cache, dataset, output)
    index = output / "index.html"
    index.write_text(_html(synthetic, rows, dataset), encoding="utf-8")
    return index


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for row in rows:
        flat_row = {key: value for key, value in row.items() if key != "w1"}
        flat_row.update({f"w1_{key}": value for key, value in row["w1"].items()})
        flat.append(flat_row)
    fields: list[str] = []
    for flat_row in flat:
        for key in flat_row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _figures(repo: Path, output: Path, synthetic: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def save(figure: Figure, name: str) -> None:
        FigureCanvasAgg(figure).print_figure(output / name, dpi=130)
        figure.clear()

    # 1. synthetic raw vs refined error per estimator
    figure = Figure(figsize=(7, 3.8))
    axis = figure.add_subplot(111)
    names = [item["case"] for item in synthetic]
    x = np.arange(len(synthetic))
    width = 0.22
    for offset, key in enumerate(("edt", "edge", "profile")):
        raw = [item.get(f"{key}_raw_mae_um", np.nan) for item in synthetic]
        ref = [item.get(f"{key}_refined_mae_um", np.nan) for item in synthetic]
        axis.bar(x + (offset - 1.5) * width, raw, width, label=f"{key} raw", color="#c9a227")
        axis.bar(x + (offset - 0.5) * width, ref, width, label=f"{key} refined", color="#2f9e63")
    axis.set_xticks(x)
    axis.set_xticklabels(names, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("MAE (µm)")
    axis.set_title("Synthetic: raw vs refined estimator error")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "synthetic-raw-vs-refined.png")

    # 2. per-image estimator medians raw vs refined
    figure = Figure(figsize=(8, 3.8))
    axis = figure.add_subplot(111)
    labels = [Path(item["image"]).stem.replace("PVDF Jose_", "J") for item in rows]
    x = np.arange(len(rows))
    series = {
        "EDT": ("edt_raw_median_um", "edt_refined_median_um"),
        "Edge": ("edge_raw_median_um", "edge_refined_median_um"),
        "Profile": ("profile_raw_median_um", "profile_refined_median_um"),
    }
    for offset, (label, (raw_key, ref_key)) in enumerate(series.items()):
        raw = [item.get(raw_key) for item in rows]
        ref = [item.get(ref_key) for item in rows]
        axis.plot(x + offset * 0.25, raw, ".", color="#8a6d1a", label=f"{label} raw")
        axis.plot(x + offset * 0.25, ref, "x", color="#2f9e63", label=f"{label} refined")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median diameter (µm)")
    axis.set_title("Per-image raw/refined estimator medians")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "per-image-medians.png")

    # 3. per-image W1 EDT<->Edge raw vs refined
    figure = Figure(figsize=(8, 3.8))
    axis = figure.add_subplot(111)
    raw_w1 = [item["w1"]["edt_edge_raw"] for item in rows]
    ref_w1 = [item["w1"]["edt_edge_refined"] for item in rows]
    axis.plot(x, raw_w1, "o-", color="#c9a227", label="W1 EDT\u2194Edge raw")
    axis.plot(x, ref_w1, "s-", color="#2f9e63", label="W1 EDT\u2194Edge refined")
    axis.axhline(0.0, color="k", lw=0.6)
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Wasserstein-1 (µm)")
    axis.set_title("Per-image W1 EDT\u2194Edge: raw vs refined")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "per-image-w1.png")

    # 4. per-image center shift raw vs residual
    figure = Figure(figsize=(8, 3.8))
    axis = figure.add_subplot(111)
    raw_shift = [item.get("center_shift_median_um") for item in rows]
    residual = [item.get("residual_shift_median_um") for item in rows]
    axis.plot(x, raw_shift, "o-", color="#c9a227", label="original center shift median")
    axis.plot(x, residual, "s-", color="#2f9e63", label="residual shift median")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Shift (µm)")
    axis.set_title("Per-image original vs residual center shift")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "per-image-shifts.png")


def _diagnostic_crops(
    cache: WorkspaceCache,
    dataset: Any,
    output: Path,
) -> None:
    """Small raw/refined centerline + boundary crops for priority cases."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    crops_dir = output / "crops"
    crops_dir.mkdir(exist_ok=True)
    for case_id in ("ZEISS_001", "ZEISS_002", "ZEISS_003", "ZEISS_004", "ZEISS_016"):
        image = dataset.image_by_case(case_id)
        if image is None or not cache.has_full(image.stem):
            continue
        comparison = cache.load_comparison(image.stem)
        field = _field_result(comparison)
        ls = field.local_samples
        from ..api import FathomEngine

        engine = FathomEngine()
        scientific = engine.open_image(image.absolute_path)
        px, py = scientific.calibration.pixel_size_x_m, scientific.calibration.pixel_size_y_m
        roi = field.valid_roi or (0, 0, scientific.shape[1], scientific.shape[0])
        x0, y0 = roi[0], roi[1]
        body = np.asarray(scientific.gray[y0:roi[3], x0:roi[2]], float)
        height, width = body.shape
        if min(height, width) < 400:
            continue
        # pick a supported segment in the middle of the image
        refined_mask = np.asarray(ls["refined_mask"], bool)
        indices = np.flatnonzero(refined_mask)
        if indices.size == 0:
            continue
        center = indices[indices.size // 2]
        cx_px = float(ls["x_m"][center]) / px - x0
        cy_px = float(ls["y_m"][center]) / py - y0
        half = 120
        row0, row1 = max(0, int(cy_px) - half), min(height, int(cy_px) + half)
        col0, col1 = max(0, int(cx_px) - half), min(width, int(cx_px) + half)
        inside = (
            (ls["x_m"] / px - x0 >= col0)
            & (ls["x_m"] / px - x0 <= col1)
            & (ls["y_m"] / py - y0 >= row0)
            & (ls["y_m"] / py - y0 <= row1)
        )
        figure = Figure(figsize=(6.4, 5.4))
        axis = figure.add_subplot(111)
        axis.imshow(body[row0:row1, col0:col1], cmap="gray", vmin=0, vmax=255)
        axis.plot(
            ls["x_m"][inside] / px - x0 - col0,
            ls["y_m"][inside] / py - y0 - row0,
            "o", ms=1.5, color="#f0a83a", label="seed centerline",
        )
        refined = np.column_stack((ls["refined_x_m"], ls["refined_y_m"]))
        axis.plot(
            refined[inside, 0] / px - x0 - col0,
            refined[inside, 1] / py - y0 - row0,
            ".", ms=2, color="#40e0ff", label="refined centerline",
        )
        for start, end in (
            (ls["minus_xy_m"], ls["plus_xy_m"]),
        ):
            axis.plot(
                start[inside, 0] / px - x0 - col0,
                start[inside, 1] / py - y0 - row0,
                ".", ms=1, color="#2f9e63", label="paired edges",
            )
            axis.plot(
                end[inside, 0] / px - x0 - col0,
                end[inside, 1] / py - y0 - row0,
                ".", ms=1, color="#2f9e63",
            )
        axis.set_title(f"{case_id} — raw vs refined centerline")
        axis.legend(fontsize="small", loc="upper right")
        figure.tight_layout()
        FigureCanvasAgg(figure).print_figure(crops_dir / f"{case_id}.png", dpi=120)
        figure.clear()


def _fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _html(
    synthetic: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    dataset: Any,
) -> str:
    synthetic_rows = ""
    for item in synthetic:
        synthetic_rows += (
            "<tr><td>" + html.escape(item["case"])
            + "</td><td>" + _fmt(item.get("center_seed_mae_um"))
            + "</td><td>" + _fmt(item.get("center_refined_mae_um"))
            + "</td><td>" + _fmt(item.get("edt_raw_mae_um"))
            + "</td><td>" + _fmt(item.get("edt_refined_mae_um"))
            + "</td><td>" + _fmt(item.get("edge_raw_mae_um"))
            + "</td><td>" + _fmt(item.get("edge_refined_mae_um"))
            + "</td><td>" + _fmt(item.get("profile_raw_mae_um"))
            + "</td><td>" + _fmt(item.get("profile_refined_mae_um"))
            + "</td><td>" + _fmt(item.get("coverage"), 4) + "</td></tr>"
        )
    real_rows = ""
    for item in rows:
        real_rows += (
            "<tr><td>" + html.escape(Path(item["image"]).stem)
            + "</td><td>" + _fmt(item.get("smooth_coverage"), 4)
            + "</td><td>" + _fmt(item.get("center_shift_median_um"))
            + "</td><td>" + _fmt(item.get("residual_shift_median_um"))
            + "</td><td>" + _fmt(item.get("edt_raw_median_um"))
            + "</td><td>" + _fmt(item.get("edt_refined_median_um"))
            + "</td><td>" + _fmt(item.get("edge_raw_median_um"))
            + "</td><td>" + _fmt(item.get("edge_refined_median_um"))
            + "</td><td>" + _fmt(item.get("profile_raw_median_um"))
            + "</td><td>" + _fmt(item.get("profile_refined_median_um"))
            + "</td><td>" + _fmt(item["w1"].get("edt_edge_raw"))
            + "</td><td>" + _fmt(item["w1"].get("edt_edge_refined")) + "</td></tr>"
        )
    improved = sum(
        1
        for item in rows
        if item["w1"].get("edt_edge_raw") is not None
        and item["w1"].get("edt_edge_refined") is not None
        and item["w1"]["edt_edge_refined"] < item["w1"]["edt_edge_raw"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Fathom Fibers — Oriented Ribbon V1 validation</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a;max-width:1200px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.3rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eef1f4}}img{{max-width:100%;border:1px solid #c8cdd4}}
h2{{border-bottom:1px solid #d4d9df;padding-bottom:.2rem}}</style></head><body>
<h1>Fathom Fibers — Oriented Ribbon V1 validation</h1>
<p>Generated {datetime.now(UTC).isoformat()}. Synthetic accuracy uses known-truth phantoms; the 16-image section reports agreement and behavior only, never accuracy on SEM data.</p>
<h2>Synthetic truth (µm)</h2>
<table><tr><th>case</th><th>seed MAE</th><th>refined center MAE</th><th>EDT raw</th><th>EDT ref</th><th>Edge raw</th><th>Edge ref</th><th>Prof raw</th><th>Prof ref</th><th>coverage</th></tr>{synthetic_rows}</table>
<h2>16-image campaign</h2>
<p>Images where W1(EDT\u2194Edge) refined &lt; raw: <b>{improved} / {len(rows)}</b>.</p>
<table><tr><th>image</th><th>smooth cov</th><th>shift med</th><th>residual med</th><th>EDT raw</th><th>EDT ref</th><th>Edge raw</th><th>Edge ref</th><th>Prof raw</th><th>Prof ref</th><th>W1 raw</th><th>W1 ref</th></tr>{real_rows}</table>
<h2>Figures</h2>
<img src="synthetic-raw-vs-refined.png" alt="synthetic raw vs refined"><br>
<img src="per-image-medians.png" alt="per-image medians"><br>
<img src="per-image-w1.png" alt="per-image W1"><br>
<img src="per-image-shifts.png" alt="per-image shifts">
<h2>Diagnostic crops (raw vs refined centerline, paired edges)</h2>
<img src="crops/ZEISS_001.png" alt="ZEISS_001"><img src="crops/ZEISS_002.png" alt="ZEISS_002"><br>
<img src="crops/ZEISS_003.png" alt="ZEISS_003"><img src="crops/ZEISS_004.png" alt="ZEISS_004"><br>
<img src="crops/ZEISS_016.png" alt="ZEISS_016">
</body></html>"""


__all__ = ["build_oriented_ribbon_report", "image_metrics", "synthetic_cases"]
