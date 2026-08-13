"""Qt-free scientific HTML reporting for images and datasets.

All figures are rendered headless with matplotlib Agg; nothing in this module
imports Qt or launches external runtimes.  Reports describe method
differences, distances and agreement; they never claim accuracy or truth.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .core.distributions import (
    DistributionAgreement,
    DistributionSummary,
    common_histogram_edges,
    summarize_distribution,
)
from .core.methods import DiameterDistribution, MethodId, MethodResult, MethodStatus
from .unified_comparison import UnifiedMethodComparison
from .validation.unified_methods import _load_cached_payloads, _root
from .workspace import WorkspaceCache

DISPLAY_NAMES = {
    MethodId.MATLAB_SIMPOLY.value: "MATLAB SIMPoly",
    MethodId.PYTHON_SIMPOLY.value: "Python SIMPoly",
    MethodId.FATHOM_LOCAL.value: "Fathom Local",
    MethodId.FATHOM_FIELD_GRAPH_V1.value: "Fathom Field (EDT)",
    MethodId.MANUAL_5X5_REFERENCE.value: "Manual 5×5",
    MethodId.CONSENSUS_PSEUDO_REFERENCE_V1.value: "Consensus",
    "FATHOM_FIELD_PAIRED_EDGE_DIAMETER": "Field Paired Edge",
    "FATHOM_FIELD_PROFILE_DIAMETER": "Field Intensity Profile",
}

FIELD_ESTIMATORS = (
    ("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "Field Paired Edge"),
    ("FATHOM_FIELD_PROFILE_DIAMETER", "Field Intensity Profile"),
)

RIBBON_ESTIMATORS = (
    ("FATHOM_FIELD_REFINED_EDT_DIAMETER", "Ribbon Refined EDT"),
    ("FATHOM_FIELD_REFINED_EDGE_DIAMETER", "Ribbon Refined Edge"),
    ("FATHOM_FIELD_REFINED_PROFILE_DIAMETER", "Ribbon Refined Profile"),
)

SERIES_COLORS = {
    "Python SIMPoly": "#fdb462",
    "Fathom Local": "#7fc97f",
    "Fathom Field (EDT)": "#8c6bb1",
    "Field Paired Edge": "#d95f02",
    "Field Intensity Profile": "#1b9e77",
    "Ribbon Refined EDT": "#386cb0",
    "Ribbon Refined Edge": "#4daf4a",
    "Ribbon Refined Profile": "#984ea3",
    "Manual 5×5": "#e31a1c",
    "Consensus": "#252525",
}


def display_name(method_id: str | MethodId) -> str:
    value = method_id.value if isinstance(method_id, MethodId) else str(method_id)
    return DISPLAY_NAMES.get(value, value)


def series_distributions(
    comparison: UnifiedMethodComparison,
) -> list[tuple[str, DiameterDistribution]]:
    """Named display series with physical units, one per method family.

    Field EDT, Paired Edge and Intensity Profile are estimator variants of the
    same experimental method and are therefore emitted as a single method with
    its secondary estimators.  The Oriented Ribbon V1 re-measurements (Refined
    EDT / Edge / Profile) belong to the same Field family and are labelled
    accordingly, never as independent methods.
    """
    series: list[tuple[str, DiameterDistribution]] = []
    for result in comparison.results:
        if result.method_id == MethodId.MATLAB_SIMPOLY:
            continue
        if result.method_id == MethodId.CONSENSUS_PSEUDO_REFERENCE_V1:
            continue
        if result.method_id == MethodId.MANUAL_5X5_REFERENCE:
            distribution = result.native_distribution
            if distribution is not None:
                series.append((display_name(result.method_id), distribution))
            continue
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
            if result.common_distribution is not None:
                series.append((display_name(result.method_id), result.common_distribution))
            for name, label in FIELD_ESTIMATORS:
                distribution = result.secondary_distributions.get(name)
                if distribution is not None:
                    series.append((label, distribution))
            for name, label in RIBBON_ESTIMATORS:
                distribution = result.secondary_distributions.get(name)
                if distribution is not None:
                    series.append((label, distribution))
            continue
        if result.common_distribution is not None:
            series.append((display_name(result.method_id), result.common_distribution))
    if comparison.consensus.distribution is not None:
        series.append(("Consensus", comparison.consensus.distribution))
    return series


def _weighted_density(distribution: DiameterDistribution, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(distribution.diameter, bins=edges, weights=distribution.weight)
    widths = np.diff(edges)
    total = distribution.weight.sum()
    return hist / (total * widths) if total > 0 else np.zeros_like(hist)


def _new_figure():
    from matplotlib.figure import Figure

    return Figure(figsize=(8, 4.6), dpi=150)


def _save(figure, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.print_figure(path, dpi=150)
    figure.clear()
    del figure


def figure_common_histogram(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    distributions = [item[1] for item in series]
    edges = common_histogram_edges(distributions)
    if not edges.size:
        axis.text(0.5, 0.5, "No common distributions available", ha="center")
    else:
        centers = edges[:-1] + np.diff(edges) / 2.0
        for name, distribution in series:
            axis.plot(
                centers,
                _weighted_density(distribution, edges),
                label=f"{name}  (N={distribution.diameter.size})",
                color=SERIES_COLORS.get(name),
                drawstyle="steps-mid",
            )
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title("Common diameter histogram")
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = output_dir / "figure-histogram.png"
    _save(figure, path)
    return path


def figure_ecdf(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    for name, distribution in series:
        order = np.argsort(distribution.diameter, kind="stable")
        x = distribution.diameter[order]
        y = np.cumsum(distribution.weight[order]) / distribution.weight.sum()
        axis.step(x, y, where="post", label=f"{name}  (N={distribution.diameter.size})", color=SERIES_COLORS.get(name))
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Cumulative weight")
    axis.set_title("Diameter ECDF comparison")
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = output_dir / "figure-ecdf.png"
    _save(figure, path)
    return path


def figure_field_estimators(comparison: UnifiedMethodComparison, output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    series = series_distributions(comparison)
    field = [(name, dist) for name, dist in series if name in {label for _, label in FIELD_ESTIMATORS}]
    edt = next(
        (
            (name, dist)
            for name, dist in series
            if name == display_name(MethodId.FATHOM_FIELD_GRAPH_V1)
        ),
        None,
    )
    if edt is not None:
        field.insert(0, edt)
    if not field:
        axis.text(0.5, 0.5, "Field estimators unavailable for this image", ha="center")
    else:
        distributions = [item[1] for item in field]
        edges = common_histogram_edges(distributions)
        if edges.size:
            centers = edges[:-1] + np.diff(edges) / 2.0
            for name, distribution in field:
                axis.plot(
                    centers,
                    _weighted_density(distribution, edges),
                    label=f"{name}  (N={distribution.diameter.size})",
                    color=SERIES_COLORS.get(name),
                    drawstyle="steps-mid",
                )
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title("Field estimator comparison: EDT / Paired Edge / Intensity Profile")
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small")
    figure.tight_layout()
    path = output_dir / "figure-field-estimators.png"
    _save(figure, path)
    return path


def figure_weighted_boxplot(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    boxes = []
    labels = []
    for name, distribution in series:
        if not distribution.diameter.size:
            continue
        summary = summarize_distribution(distribution)
        if summary.p25 is None or summary.p75 is None:
            continue
        boxes.append(
            {
                "med": summary.weighted_median,
                "q1": summary.p25,
                "q3": summary.p75,
                "whislo": summary.p05 if summary.p05 is not None else summary.p25,
                "whishi": summary.p95 if summary.p95 is not None else summary.p75,
            }
        )
        labels.append(f"{name} (N={distribution.diameter.size})")
    if not boxes:
        axis.text(0.5, 0.5, "No comparable distributions", ha="center")
    else:
        axis.bxp(boxes, showfliers=False)
        axis.set_xticklabels(labels, rotation=15, fontsize="small")
    axis.set_ylabel("Diameter (µm)")
    axis.set_title("Method summary — weighted P05/P25/P50/P75/P95")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output_dir / "figure-method-summary.png"
    _save(figure, path)
    return path


def _fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def summary_row(result: MethodResult) -> tuple[tuple[str, ...], DistributionSummary | None]:
    distribution = result.common_distribution
    if result.method_id == MethodId.MANUAL_5X5_REFERENCE:
        distribution = result.native_distribution
    if distribution is None:
        return (result.method_id.value, result.status.value, "—", "—", "—", "—", "—", "—"), None
    summary = summarize_distribution(distribution)
    iqr = (
        f"{summary.p25:.5g}–{summary.p75:.5g}"
        if summary.p25 is not None and summary.p75 is not None
        else "—"
    )
    return (
        result.method_id.value,
        result.status.value,
        str(summary.n),
        _fmt(summary.weighted_mean),
        _fmt(summary.weighted_median),
        iqr,
        _fmt(summary.p05),
        _fmt(summary.p95),
    ), summary


def valid_agreements(comparison: UnifiedMethodComparison) -> list[DistributionAgreement]:
    return [item for item in comparison.agreements if item.wasserstein_1 is not None]


def _matlab_section(result: MethodResult | None) -> str:
    if result is None:
        return "<tr><td colspan='3'>MATLAB SIMPoly not available for this image.</td></tr>"
    if result.status != MethodStatus.COMPLETE:
        return (
            "<tr><td>MATLAB SIMPoly</td><td>"
            + html.escape(result.status.value)
            + "</td><td>—</td></tr>"
        )
    stats_payload = result.native_statistics
    provenance = result.provenance
    b1 = stats_payload.get("gauss_b1", result.native_result)
    return (
        "<tr><td>MATLAB SIMPoly — native Gaussian center b1</td><td>"
        + _fmt(b1, 6)
        + " µm</td><td>"
        + html.escape(str(provenance.get("matlab_version", "R2026a")))
        + "</td></tr>"
        "<tr><td>MATLAB source hash</td><td colspan='2'><code>"
        + html.escape(str(provenance.get("source_matlab_sha256", "—")))
        + "</code></td></tr>"
        "<tr><td>Cache representation</td><td colspan='2'>"
        + html.escape(str(provenance.get("cache_representation", "CONTROLLED_CACHE")))
        + "</td></tr>"
        "<tr><td>Common distribution</td><td colspan='2'>"
        + (
            "Unavailable from current cache."
            if any(flag in result.quality_flags for flag in ("MATLAB_RAW_DIAMETERS_UNAVAILABLE", "COMMON_LENGTH_WEIGHT_UNAVAILABLE"))
            else "Available."
        )
        + "</td></tr>"
    )


def build_image_report(
    comparison: UnifiedMethodComparison,
    image: Any,
    *,
    output_dir: str | Path,
    manual_complete: bool | None = None,
    manual_count: int = 0,
) -> Path:
    """Render one image's scientific report and return its index.html path."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series = series_distributions(comparison)
    figure_common_histogram(series, output)
    figure_ecdf(series, output)
    figure_field_estimators(comparison, output)
    figure_weighted_boxplot(series, output)

    summary_rows = ""
    for result in comparison.results:
        values, _summary = summary_row(result)
        method = values[0]
        if method == MethodId.FATHOM_FIELD_GRAPH_V1.value:
            method = "Fathom Field"
        summary_rows += (
            "<tr><td>" + html.escape(display_name(method))
            + "</td><td>" + html.escape(values[1])
            + "</td><td>" + html.escape(values[2])
            + "</td><td>" + html.escape(values[3])
            + "</td><td>" + html.escape(values[4])
            + "</td><td>" + html.escape(values[5])
            + "</td><td>" + html.escape(values[6])
            + "</td><td>" + html.escape(values[7]) + "</td></tr>"
        )

    agreement_rows = ""
    for item in valid_agreements(comparison):
        agreement_rows += (
            "<tr><td>" + html.escape(display_name(item.left_method))
            + "</td><td>" + html.escape(display_name(item.right_method))
            + "</td><td>" + _fmt(item.wasserstein_1)
            + "</td><td>" + _fmt(item.ks_statistic)
            + "</td><td>" + _fmt(item.median_difference) + "</td></tr>"
        )
    agreement_rows = agreement_rows or "<tr><td colspan='5'>No comparable distributions.</td></tr>"

    field = next(
        (result for result in comparison.results if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1),
        None,
    )
    field_rows, field_notes = _field_diagnostics(field)

    quality_rows = ""
    for result in comparison.results:
        quality_rows += (
            "<tr><td>" + html.escape(display_name(result.method_id))
            + "</td><td>" + html.escape(result.status.value)
            + "</td><td>" + html.escape(", ".join(result.quality_flags) or "—") + "</td></tr>"
        )
    flag_breakdown = _flag_breakdown(comparison)

    matlab = next(
        (result for result in comparison.results if result.method_id == MethodId.MATLAB_SIMPOLY),
        None,
    )
    manual = next(
        (result for result in comparison.results if result.method_id == MethodId.MANUAL_5X5_REFERENCE),
        None,
    )
    if manual_complete is None and manual is not None:
        manual_complete = manual.status == MethodStatus.COMPLETE and (
            manual.native_statistics.get("measurement_count", 0) >= 25
        )
        manual_count = int(manual.native_statistics.get("measurement_count", 0))
    manual_rows = ""
    if manual is None or manual.status == MethodStatus.NOT_MEASURED:
        manual_rows = "<tr><td>Manual 5×5</td><td>0 / 25 measurements complete</td><td>INCOMPLETE REFERENCE</td></tr>"
    elif manual_complete:
        manual_rows = (
            "<tr><td>Manual 5×5</td><td>25 / 25 measurements complete</td>"
            "<td>Distribution included above; reference remains operator-defined.</td></tr>"
        )
    else:
        manual_rows = (
            f"<tr><td>Manual 5×5</td><td>{manual_count} / 25 measurements complete</td>"
            "<td>INCOMPLETE REFERENCE</td></tr>"
        )

    calibration = image.calibration
    metadata = dict(image.metadata or {})
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Fathom Fibers — {html.escape(comparison.image_id)}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a;max-width:1100px}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.4rem;text-align:left}}th{{background:#eef1f4}}img{{max-width:100%;border:1px solid #c8cdd4}}code{{background:#f4f4f4;padding:.1rem .3rem}}h2{{border-bottom:1px solid #d4d9df;padding-bottom:.2rem}}.note{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>Fathom Fibers Scientific Analysis</h1>
<p class="note">Agreement and consensus pseudo-reference are not ground truth. Measurements represent projected 2-D geometry.</p>
<h2>Image</h2>
<table><tr><th>Identity</th><td>{html.escape(comparison.image_id)}</td></tr>
<tr><th>Calibration</th><td>{calibration.pixel_size_x_m * 1e9:.5g} × {calibration.pixel_size_y_m * 1e9:.5g} nm/px ({html.escape(calibration.source)})</td></tr>
<tr><th>ROI</th><td>{html.escape(str(next((result.valid_roi for result in comparison.results if result.valid_roi), None)))}</td></tr>
<tr><th>Magnification</th><td>{html.escape(str(metadata.get("ap_mag", "—")))}</td></tr>
<tr><th>EHT</th><td>{html.escape(str(metadata.get("ap_actualkv", "—")))}</td></tr>
<tr><th>Generated</th><td>{datetime.now(UTC).isoformat()}</td></tr></table>
<h2>Method summary</h2>
<table><tr><th>Method</th><th>Status</th><th>N</th><th>Mean</th><th>Median</th><th>IQR</th><th>P05</th><th>P95</th></tr>{summary_rows}</table>
<h2>MATLAB SIMPoly cache</h2>
<table><tr><th>Quantity</th><th>Value</th><th>Source</th></tr>{_matlab_section(matlab)}</table>
<h2>Distributions</h2>
<img src="figure-histogram.png" alt="Common diameter histogram">
<img src="figure-ecdf.png" alt="Diameter ECDF">
<h2>Field estimator comparison</h2>
{field_notes}
<img src="figure-field-estimators.png" alt="Field estimator comparison">
<h2>Field diagnostics</h2>
<table><tr><th>Field estimator</th><th>N</th><th>Median (µm)</th><th>IQR</th><th>P05</th><th>P95</th></tr>{field_rows}</table>
<h2>Method summary box</h2>
<img src="figure-method-summary.png" alt="Weighted method summary">
<h2>Method differences (common estimand)</h2>
<table><tr><th>A</th><th>B</th><th>Wasserstein-1 (µm)</th><th>KS</th><th>Median difference (µm)</th></tr>{agreement_rows}</table>
<h2>Quality</h2>
<table><tr><th>Method</th><th>Status</th><th>Flags</th></tr>{quality_rows}</table>
{flag_breakdown}
<h2>Manual reference</h2>
<table><tr><th>Method</th><th>Progress</th><th>Status</th></tr>{manual_rows}</table>
</body></html>"""
    index = output / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def _field_diagnostics(field: MethodResult | None) -> tuple[str, str]:
    if field is None:
        return (
            "<tr><td colspan='6'>Fathom Field not available.</td></tr>",
            "<p>Fathom Field is not available for this image.</p>",
        )
    stats_payload = field.native_statistics
    rows = ""
    for name, label in (
        ("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "Paired Edge"),
        ("FATHOM_FIELD_PROFILE_DIAMETER", "Intensity Profile"),
    ):
        distribution = field.secondary_distributions.get(name)
        if distribution is None:
            rows += f"<tr><td>{label}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>"
            continue
        summary = summarize_distribution(distribution)
        iqr = (
            f"{summary.p25:.5g}–{summary.p75:.5g}"
            if summary.p25 is not None and summary.p75 is not None
            else "—"
        )
        rows += (
            f"<tr><td>{label}</td><td>{summary.n}</td><td>{_fmt(summary.weighted_median)}</td>"
            f"<td>{iqr}</td><td>{_fmt(summary.p05)}</td><td>{_fmt(summary.p95)}</td></tr>"
        )
    if field.common_distribution is not None:
        summary = summarize_distribution(field.common_distribution)
        rows += (
            f"<tr><td>EDT</td><td>{summary.n}</td><td>{_fmt(summary.weighted_median)}</td>"
            f"<td>{_fmt(summary.p25)}–{_fmt(summary.p75)}</td><td>{_fmt(summary.p05)}</td>"
            f"<td>{_fmt(summary.p95)}</td></tr>"
        )
    coherence = stats_payload.get("mean_coherence")
    edge_acceptance = stats_payload.get("edge_acceptance_fraction")
    profile_acceptance = stats_payload.get("profile_acceptance_fraction")
    asymmetry = stats_payload.get("edge_median_asymmetry")
    notes = (
        "<p>"
        f"Mean coherence: <b>{_fmt(coherence, 4)}</b>. "
        f"Paired-edge acceptance: <b>{'—' if edge_acceptance is None else f'{edge_acceptance:.2%}'}</b>. "
        f"Intensity-profile acceptance: <b>{'—' if profile_acceptance is None else f'{profile_acceptance:.2%}'}</b>. "
        f"Median edge asymmetry: <b>{_fmt(asymmetry, 4)}</b>.</p>"
    )
    return rows, notes


def _flag_breakdown(comparison: UnifiedMethodComparison) -> str:
    sections = ""
    for result in comparison.results:
        if not result.quality_flags:
            continue
        counts: dict[str, int] = {}
        for flag in result.quality_flags:
            counts[flag] = counts.get(flag, 0) + 1
        rows = "".join(
            f"<tr><td>{html.escape(flag)}</td><td>{count}</td></tr>"
            for flag, count in sorted(counts.items())
        )
        sections += (
            f"<h3>Flag breakdown — {html.escape(display_name(result.method_id))}</h3>"
            f"<table><tr><th>Flag</th><th>Count</th></tr>{rows}</table>"
        )
    if not sections:
        return ""
    return f"<h2>Field flag breakdown</h2>{sections}"


def build_dataset_report(
    repo: str | Path,
    *,
    dataset: Any,
    manual_store: Any = None,
) -> Path:
    """Render the 16-image dataset scientific report into ``latest/index.html``."""
    repo = Path(repo)
    payloads = _load_cached_payloads(repo)
    output = _root(repo) / "latest"
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    method_order = [
        MethodId.MATLAB_SIMPOLY.value,
        MethodId.PYTHON_SIMPOLY.value,
        MethodId.FATHOM_LOCAL.value,
        MethodId.FATHOM_FIELD_GRAPH_V1.value,
        MethodId.MANUAL_5X5_REFERENCE.value,
        MethodId.CONSENSUS_PSEUDO_REFERENCE_V1.value,
    ]
    status_matrix_rows = ""
    per_image_rows = ""
    per_image_links = ""
    for payload in payloads:
        image_id = payload.get("image_id", "unknown")
        stem = Path(image_id).stem
        statuses: dict[str, str] = {}
        medians: dict[str, str] = {}
        for entry in payload.get("results", ()):
            statuses[entry["method_id"]] = entry["status"]
            if entry["method_id"] == MethodId.MATLAB_SIMPOLY.value:
                medians[entry["method_id"]] = _fmt(entry.get("native_result"))
            elif entry.get("common_distribution"):
                medians[entry["method_id"]] = _fmt(entry["common_distribution"].get("median"))
            elif entry.get("secondary_distributions"):
                medians[entry["method_id"]] = _fmt(
                    next(iter(entry["secondary_distributions"].values())).get("median")
                )
        cells = "".join(
            f"<td>{html.escape(statuses.get(method, 'NOT_RUN'))}</td>"
            for method in method_order
        )
        median_cells = "".join(
            f"<td>{html.escape(medians.get(method, '—'))}</td>" for method in method_order
        )
        link = images_dir / stem / "index.html"
        _render_summary_image_page(payload, link.parent)
        per_image_links += (
            f"<li><a href='images/{stem}/index.html'>{html.escape(image_id)}</a>"
            f" — {html.escape(', '.join(statuses.values()) or '—')}</li>"
        )
        status_matrix_rows += (
            f"<tr><td>{html.escape(image_id)}</td>{cells}</tr>"
        )
        per_image_rows += f"<tr><td>{html.escape(image_id)}</td>{median_cells}</tr>"

    _dataset_figures(payloads, output)

    method_completion_rows = ""
    for method in method_order:
        counts = {status: 0 for status in ("COMPLETE", "NOT_RUN", "EXPERIMENTAL_FIELD_MEASURING", "FAILED", "NOT_MEASURED", "UNAVAILABLE")}
        for payload in payloads:
            for entry in payload.get("results", ()):
                if entry["method_id"] == method:
                    counts[entry["status"]] = counts.get(entry["status"], 0) + 1
        status_text = ", ".join(f"{key} {value}" for key, value in sorted(counts.items()) if value)
        method_completion_rows += (
            f"<tr><td>{html.escape(display_name(method))}</td><td>{html.escape(status_text)}</td></tr>"
        )

    manual_progress = "Manual 5×5 store not loaded."
    if manual_store is not None:
        manual_progress = (
            f"Images with stored status: {manual_store.reviewed_images} / {len(payloads)}. "
            f"Measurements recorded: {manual_store.total_measured} / 400."
        )

    wasserstein_rows, difference_rows = _dataset_agreement_rows(payloads)

    observations = _factual_observations(payloads)

    index = output / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Fathom Fibers Dataset Report — {html.escape(str(dataset.dataset_id))}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a;max-width:1200px}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.4rem;text-align:left}}th{{background:#eef1f4}}img{{max-width:100%;border:1px solid #c8cdd4}}h2{{border-bottom:1px solid #d4d9df;padding-bottom:.2rem}}.note{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>Fathom Fibers Dataset Scientific Report</h1>
<p class="note">Agreement is not truth. Consensus, where available, is an equal-method quantile pseudo-reference only. Dataset display balances images; it does not pool all local samples.</p>
<p>Dataset: <code>{html.escape(str(dataset.dataset_id))}</code> — {len(payloads)} / {len(dataset.images)} images represented. MATLAB is consumed from a validated cache; Python SIMPoly retains its documented <code>bwskel</code> divergence.</p>
<h2>Observations</h2>
<ul>{observations}</ul>
<h2>Method completion</h2>
<table><tr><th>Method</th><th>Status counts</th></tr>{method_completion_rows}</table>
<h2>Manual progress</h2><p>{html.escape(manual_progress)}</p>
<h2>16-image processing matrix</h2>
<table><tr><th>Image</th>{''.join(f"<th>{html.escape(display_name(m))}</th>" for m in method_order)}</tr>{status_matrix_rows}</table>
<h2>Diameter summary by image (µm)</h2>
<table><tr><th>Image</th>{''.join(f"<th>{html.escape(display_name(m))}</th>" for m in method_order)}</tr>{per_image_rows}</table>
<h2>Per-image reports</h2><ul>{per_image_links}</ul>
<h2>Dataset figures</h2>
<p>Per-image weighted medians (image-level summaries; images are statistical units, not pooled samples).</p>
<img src="dataset-medians.png" alt="Per-image medians"><br>
<img src="dataset-median-ecdf.png" alt="Distribution of per-image medians"><br>
<img src="dataset-wasserstein-matrix.png" alt="Pairwise Wasserstein-1 matrix">
<h2>Pairwise distribution distance</h2>
<table><tr><th>Left</th><th>Right</th><th>Median Wasserstein-1 (µm)</th><th>Images</th></tr>{wasserstein_rows}</table>
<h2>Pairwise median-method difference</h2>
<table><tr><th>Left</th><th>Right</th><th>Median difference (µm)</th><th>Images</th></tr>{difference_rows}</table>
<h2>Provenance</h2>
<p>Generated {datetime.now(UTC).isoformat()} from cached MethodResult payloads; no algorithm was rerun for this report and MATLAB was never launched.</p>
</body></html>""",
        encoding="utf-8",
    )
    return index


def _render_summary_image_page(payload: dict[str, Any], output: Path) -> Path:
    """Render a compact per-image page from cached summaries alone."""
    output.mkdir(parents=True, exist_ok=True)
    rows = ""
    for entry in payload.get("results", ()):
        method = entry["method_id"]
        distribution = entry.get("common_distribution")
        secondary = entry.get("secondary_distributions") or {}
        native = entry.get("native_result")
        if method == MethodId.MATLAB_SIMPOLY.value:
            value = f"b1 {_fmt(native, 6)} µm"
        elif distribution and distribution.get("median") is not None:
            value = (
                f"median {_fmt(distribution['median'], 5)} µm · N={distribution.get('n', '—')}"
            )
        elif secondary:
            first = next(iter(secondary.values()))
            value = f"median {_fmt(first.get('median'), 5)} µm" if first.get("median") else "—"
        else:
            value = "—"
        rows += (
            "<tr><td>" + html.escape(display_name(method))
            + "</td><td>" + html.escape(entry.get("status", "NOT_RUN"))
            + "</td><td>" + html.escape(str(value))
            + "</td><td>" + html.escape(", ".join(entry.get("quality_flags", ())) or "—")
            + "</td></tr>"
        )
    consensus = payload.get("consensus", {})
    index = output / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(str(payload.get('image_id', 'image')))}</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.4rem;text-align:left}}th{{background:#eef1f4}}.note{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>{html.escape(str(payload.get('image_id', 'image')))}</h1>
<p class="note">Summary-cache page. Run the methods for this image in the workspace to obtain full samples, overlays and distribution figures.</p>
<table><tr><th>Method</th><th>Status</th><th>Summary value</th><th>Flags</th></tr>{rows}</table>
<h2>Consensus pseudo-reference</h2>
<p>Participants: {html.escape(', '.join(consensus.get('participating_methods', ())) or '—')}.
Excluded: {html.escape(', '.join(f"{k}: {v}" for k, v in consensus.get('excluded_methods', {}).items()) or '—')}.</p>
</body></html>""",
        encoding="utf-8",
    )
    return index


def _dataset_figures(payloads: list[dict[str, Any]], output: Path) -> list[Path]:
    method_ids = [
        MethodId.PYTHON_SIMPOLY.value,
        MethodId.FATHOM_LOCAL.value,
        MethodId.FATHOM_FIELD_GRAPH_V1.value,
    ]
    labels = [display_name(item) for item in method_ids]
    medians = {method: [] for method in method_ids}
    for payload in payloads:
        for method in method_ids:
            entry = next(
                (item for item in payload.get("results", ()) if item["method_id"] == method),
                None,
            )
            distribution = entry.get("common_distribution") if entry else None
            if distribution and distribution.get("median") is not None:
                medians[method].append(float(distribution["median"]))
    paths: list[Path] = []

    figure = _new_figure()
    axis = figure.add_subplot(111)
    x = np.arange(len(payloads))
    width = 0.26
    for offset, method in enumerate(method_ids):
        values = medians[method]
        axis.bar(x + (offset - 1) * width, values, width, label=labels[offset], color=SERIES_COLORS.get(labels[offset]))
    axis.set_xticks(x)
    axis.set_xticklabels(
        [html.escape(Path(p.get("image_id", "?")).stem.replace("PVDF Jose_", "J")) for p in payloads],
        rotation=45,
        ha="right",
        fontsize="small",
    )
    axis.set_ylabel("Weighted median diameter (µm)")
    axis.set_title("Per-image median comparison")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    path = output / "dataset-medians.png"
    _save(figure, path)
    paths.append(path)

    figure = _new_figure()
    axis = figure.add_subplot(111)
    for method in method_ids:
        values = np.asarray(medians[method], float)
        if values.size:
            order = np.argsort(values)
            axis.step(values[order], np.arange(1, values.size + 1) / values.size, where="post", label=labels[method_ids.index(method)])
    axis.set_xlabel("Weighted median diameter (µm)")
    axis.set_ylabel("Fraction of images")
    axis.set_title("Distribution of per-image medians")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    path = output / "dataset-median-ecdf.png"
    _save(figure, path)
    paths.append(path)

    figure = _new_figure()
    axis = figure.add_subplot(111)
    matrix: dict[tuple[str, str], list[float]] = {}
    for payload in payloads:
        for agreement in payload.get("agreements", ()):
            if agreement.get("wasserstein_1") is None:
                continue
            key = (agreement["left_method"], agreement["right_method"])
            matrix.setdefault(key, []).append(float(agreement["wasserstein_1"]))
    matrix_keys = sorted(matrix)
    if matrix_keys:
        x = np.arange(len(matrix_keys))
        axis.bar(x, [np.median(matrix[key]) for key in matrix_keys])
        axis.set_xticks(x)
        axis.set_xticklabels([f"{display_name(a)} ↔ {display_name(b)}" for a, b in matrix_keys], rotation=40, ha="right", fontsize="small")
    axis.set_ylabel("Median Wasserstein-1 (µm)")
    axis.set_title("Pairwise distribution distance matrix")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    path = output / "dataset-wasserstein-matrix.png"
    _save(figure, path)
    paths.append(path)
    return paths


def _dataset_agreement_rows(payloads: list[dict[str, Any]]) -> tuple[str, str]:
    wasserstein: dict[tuple[str, str], list[float]] = {}
    differences: dict[tuple[str, str], list[float]] = {}
    for payload in payloads:
        for agreement in payload.get("agreements", ()):
            key = (agreement["left_method"], agreement["right_method"])
            if agreement.get("wasserstein_1") is not None:
                wasserstein.setdefault(key, []).append(float(agreement["wasserstein_1"]))
            if agreement.get("median_difference") is not None:
                differences.setdefault(key, []).append(float(agreement["median_difference"]))
    wasserstein_rows = "".join(
        f"<tr><td>{html.escape(display_name(left))}</td><td>{html.escape(display_name(right))}</td>"
        f"<td>{np.median(values):.6g}</td><td>{len(values)}</td></tr>"
        for (left, right), values in sorted(wasserstein.items())
    ) or "<tr><td colspan='4'>No comparable pairwise distributions.</td></tr>"
    difference_rows = "".join(
        f"<tr><td>{html.escape(display_name(left))}</td><td>{html.escape(display_name(right))}</td>"
        f"<td>{np.median(values):.6g}</td><td>{len(values)}</td></tr>"
        for (left, right), values in sorted(differences.items())
    ) or "<tr><td colspan='4'>No comparable pairwise distributions.</td></tr>"
    return wasserstein_rows, difference_rows


def _factual_observations(payloads: list[dict[str, Any]]) -> str:
    processed = sum(
        1
        for payload in payloads
        if any(entry.get("status") not in {"NOT_RUN", "FAILED"} for entry in payload.get("results", ()))
    )
    edge_acceptances: list[float] = []
    profile_acceptances: list[float] = []
    edge_minus_edt: list[float] = []
    for payload in payloads:
        field = next(
            (entry for entry in payload.get("results", ()) if entry["method_id"] == MethodId.FATHOM_FIELD_GRAPH_V1.value),
            None,
        )
        if not field:
            continue
        statistics = field.get("native_statistics", {})
        if statistics.get("edge_acceptance_fraction") is not None:
            edge_acceptances.append(float(statistics["edge_acceptance_fraction"]))
        if statistics.get("profile_acceptance_fraction") is not None:
            profile_acceptances.append(float(statistics["profile_acceptance_fraction"]))
        if statistics.get("centering_median_edge_minus_edt_um") is not None:
            edge_minus_edt.append(float(statistics["centering_median_edge_minus_edt_um"]))
    items = [
        f"{processed}/{len(payloads)} images processed.",
    ]
    if edge_acceptances:
        items.append(
            f"Fathom Field paired-edge median acceptance across images: {np.median(edge_acceptances):.2%}."
        )
    if profile_acceptances:
        items.append(
            f"Intensity-profile refinement median acceptance across images: {np.median(profile_acceptances):.2%}."
        )
    if edge_minus_edt:
        median = np.median(edge_minus_edt)
        items.append(
            f"Median of per-image paired-edge minus EDT centering offset: {median:.4g} µm."
        )
    items.append(
        "Graph reconstruction, fiber instances and ML remain outside this batch; Fathom Field is an experimental field-measuring method."
    )
    return "".join(f"<li>{html.escape(item)}</li>" for item in items)


__all__ = [
    "build_dataset_report",
    "build_image_report",
    "display_name",
    "series_distributions",
    "valid_agreements",
]


# ---------------------------------------------------------------------------
# Final deliverable report (reads the v2 full caches; used by the workspace
# "Generate Dataset Report" action and the export bundle).


def _per_image_estimator_medians(field: MethodResult) -> dict[str, float | None]:
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
    profile_raw = np.asarray(ls["profile_accepted"], bool)
    return {
        "edt_raw": median("diameter_um"),
        "edt_refined": statistics.get("refined_edt_median_um"),
        "edge_raw": median("edge_diameter_um", edge_raw),
        "edge_refined": statistics.get("refined_edge_median_um"),
        "profile_raw": median("profile_diameter_um", profile_raw),
        "profile_refined": statistics.get("refined_profile_median_um"),
        "smooth_coverage": statistics.get("smooth_coverage_fraction"),
        "center_shift_median_um": statistics.get("refine_median_shift_um"),
        "residual_shift_median_um": statistics.get("refined_residual_shift_median_um"),
        "asymmetry_raw": statistics.get("edge_median_asymmetry"),
        "asymmetry_refined": statistics.get("refined_asymmetry_median"),
        "edge_acceptance_raw": statistics.get("edge_acceptance_fraction"),
        "edge_acceptance_refined": statistics.get("refined_edge_acceptance_fraction"),
        "profile_acceptance_raw": statistics.get("profile_acceptance_fraction"),
        "profile_acceptance_refined": statistics.get("refined_profile_acceptance_fraction"),
        "mean_coherence": statistics.get("mean_coherence"),
        "segment_count": statistics.get("smooth_segment_count"),
    }


def _w1_pair(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import wasserstein_distance

    if a.size == 0 or b.size == 0:
        return None
    return float(wasserstein_distance(a, b))


def _dataset_ribbon_metrics(comparisons: list[Any]) -> dict[str, Any]:
    w1_raw: list[float] = []
    w1_refined: list[float] = []
    improved = 0
    for comparison in comparisons:
        field = next(
            r for r in comparison.results
            if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
        )
        ls = field.local_samples
        edge_raw = np.asarray(ls["edge_accepted"], bool)
        edge_refined = np.asarray(ls["refined_edge_accepted"], bool)
        a = _w1_pair(ls["diameter_um"][edge_raw], ls["edge_diameter_um"][edge_raw])
        b = _w1_pair(ls["refined_edt_um"][edge_refined], ls["refined_edge_um"][edge_refined])
        if a is not None and b is not None:
            w1_raw.append(a)
            w1_refined.append(b)
            if b < a:
                improved += 1
    return {
        "w1_edt_edge_raw": w1_raw,
        "w1_edt_edge_refined": w1_refined,
        "improved_count": improved,
        "total": len(comparisons),
    }


def _final_figures(output: Path, comparisons: list[Any], metrics: dict[str, Any]) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def save(figure: Figure, name: str) -> None:
        FigureCanvasAgg(figure).print_figure(output / name, dpi=130)
        figure.clear()

    stems = [Path(c.image_id).stem.replace("PVDF Jose_", "J") for c in comparisons]
    x = np.arange(len(comparisons))
    medians = {key: [] for key in ("edt_raw", "edt_refined", "edge_raw", "edge_refined", "profile_raw", "profile_refined")}
    shifts = {"observed": [], "residual": []}
    asymmetry = {"raw": [], "refined": []}
    coverage: list[float | None] = []
    for comparison in comparisons:
        field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
        values = _per_image_estimator_medians(field)
        for key, bucket in medians.items():
            bucket.append(values[key])

        shifts["observed"].append(values["center_shift_median_um"])
        shifts["residual"].append(values["residual_shift_median_um"])
        asymmetry["raw"].append(values["asymmetry_raw"])
        asymmetry["refined"].append(values["asymmetry_refined"])
        coverage.append(values["smooth_coverage"])

    figure = Figure(figsize=(9, 4))
    axis = figure.add_subplot(111)
    for offset, (label, key) in enumerate(
        (("EDT raw", "edt_raw"), ("EDT refined", "edt_refined"),
         ("Edge raw", "edge_raw"), ("Edge refined", "edge_refined"))
    ):
        axis.plot(x + offset * 0.2, medians[key], ".", color=SERIES_COLORS.get(label, "#333333"), label=label)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median diameter (µm)")
    axis.set_title("Raw vs refined EDT and Edge medians by image")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "dataset-figure-A.png")

    figure = Figure(figsize=(9, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, metrics["w1_edt_edge_raw"], "o-", color="#c9a227", label="W1 raw EDT\u2194Edge")
    axis.plot(x, metrics["w1_edt_edge_refined"], "s-", color="#2f9e63", label="W1 refined EDT\u2194Edge")
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Wasserstein-1 (µm)")
    axis.set_title("Pairwise EDT\u2194Edge agreement, raw vs refined")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "dataset-figure-B.png")

    figure = Figure(figsize=(9, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, shifts["observed"], "o-", color="#c9a227", label="observed center shift")
    axis.plot(x, shifts["residual"], "s-", color="#2f9e63", label="residual center shift")
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median shift (µm)")
    axis.set_title("Observed vs residual center shift by image")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "dataset-figure-C.png")

    figure = Figure(figsize=(9, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, asymmetry["raw"], "o-", color="#c9a227", label="asymmetry raw")
    axis.plot(x, asymmetry["refined"], "s-", color="#2f9e63", label="asymmetry refined")
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median asymmetry")
    axis.set_title("Paired-edge asymmetry, raw vs refined")
    axis.legend(fontsize="small")
    figure.tight_layout()
    save(figure, "dataset-figure-D.png")

    figure = Figure(figsize=(9, 4))
    axis = figure.add_subplot(111)
    axis.bar(x, coverage, color="#386cb0")
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1)
    axis.set_title("Supported refined-centerline coverage by image")
    figure.tight_layout()
    save(figure, "dataset-figure-E.png")


def build_final_dataset_report(
    repo: str | Path,
    *,
    dataset: Any,
    manual_store: Any = None,
    output_dir: str | Path,
    comparisons: list[Any] | None = None,
) -> Path:
    """Render the deliverable 16-image scientific report from v2 full caches.

    ``comparisons`` may carry already-loaded caches so callers that build a
    bundle avoid decompressing the full NPZ stores twice.
    """
    repo = Path(repo)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = WorkspaceCache(repo)
    if comparisons is None:
        comparisons = []
        for image in dataset.images:
            comparison = cache.load_comparison(image.stem)
            if comparison is not None:
                comparisons.append(comparison)
    comparisons = sorted(comparisons, key=lambda item: item.image_id)
    metrics = _dataset_ribbon_metrics(comparisons)
    _final_figures(output, comparisons, metrics)

    per_image_rows = ""
    for comparison in comparisons:
        field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
        values = _per_image_estimator_medians(field)
        matlab = next(
            (r for r in comparison.results if r.method_id == MethodId.MATLAB_SIMPOLY), None
        )
        python = next(
            (r for r in comparison.results if r.method_id == MethodId.PYTHON_SIMPOLY), None
        )
        local = next(
            (r for r in comparison.results if r.method_id == MethodId.FATHOM_LOCAL), None
        )
        manual = next(
            (r for r in comparison.results if r.method_id == MethodId.MANUAL_5X5_REFERENCE), None
        )
        stem = Path(comparison.image_id).stem
        manual_median = None
        if manual is not None and manual.native_distribution is not None:
            manual_median = float(np.median(manual.native_distribution.diameter))
        per_image_rows += (
            "<tr><td>" + html.escape(stem) + "</td>"
            "<td>" + _fmt(matlab.native_result if matlab is not None else None) + "</td>"
            "<td>" + _fmt(_distribution_median(python.common_distribution) if python is not None else None) + "</td>"
            "<td>" + _fmt(_distribution_median(local.common_distribution) if local is not None else None) + "</td>"
            "<td>" + _fmt(values["edt_raw"]) + "</td>"
            "<td>" + _fmt(values["edt_refined"]) + "</td>"
            "<td>" + _fmt(values["edge_raw"]) + "</td>"
            "<td>" + _fmt(values["edge_refined"]) + "</td>"
            "<td>" + _fmt(values["profile_raw"]) + "</td>"
            "<td>" + _fmt(values["profile_refined"]) + "</td>"
            "<td>" + _fmt(manual_median) + "</td>"
            "<td>" + _fmt(values["smooth_coverage"], 4) + "</td></tr>"
        )
        _final_per_image_figures(output, comparison, stem)

    per_image_figure_links = ""
    for comparison in comparisons:
        stem = Path(comparison.image_id).stem
        escaped = html.escape(stem)
        per_image_figure_links += (
            f"<h3>{escaped}</h3>"
            f"<img src='images/{escaped}/figure-A-histogram.png' alt='{escaped} histogram'><br>"
            f"<img src='images/{escaped}/figure-B-ecdf.png' alt='{escaped} ECDF'><br>"
            f"<img src='images/{escaped}/figure-C-field-ribbon.png' alt='{escaped} field raw vs ribbon'><br>"
        )

    manual_section = _final_manual_section(manual_store, dataset)
    ribbon_box = _final_ribbon_box(metrics)
    provenance = _final_provenance(dataset, repo, cache)

    index = output / "index.html"
    index.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Fathom Fibers — Scientific Morphological Fiber Analysis</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a;max-width:1200px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.35rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#eef1f4}}img{{max-width:100%;border:1px solid #c8cdd4}}
h1{{font-size:1.6rem}}h2{{border-bottom:1px solid #d4d9df;padding-bottom:.2rem;margin-top:2rem}}
.resultbox{{border-left:4px solid #2f9e63;background:#f0faf4;padding:1rem}}
.notebox{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>Fathom Fibers — Scientific Morphological Fiber Analysis</h1>
<p>Generated {datetime.now(UTC).isoformat()} from cached MethodResult payloads (schema {WorkspaceCache.FULL_SCHEMA}). No MATLAB execution; no algorithm was rerun for this report.</p>
{ribbon_box}
<h2>1. Dataset</h2>
<p>Dataset <code>{html.escape(str(dataset.dataset_id))}</code> — {len(dataset.images)} images, {len(comparisons)} represented in the current caches.</p>
<h2>2. Calibration / acquisition</h2>
<p>Calibration is read from the embedded Zeiss SEM metadata; no uniform assumption is made. See provenance below.</p>
<h2>3. Methods</h2>
<table><tr><th>Method</th><th>Description</th><th>Status</th></tr>
<tr><td>MATLAB SIMPoly</td><td>Native MATLAB implementation consumed from the validated oracle cache; native Gaussian center b1 reported.</td><td>COMPLETE (cache)</td></tr>
<tr><td>Python SIMPoly</td><td>Python port of SIMPoly; calibrated length-weighted diameters on the skeleton. Documented divergence: bwskel.</td><td>COMPLETE</td></tr>
<tr><td>Fathom Local</td><td>Assisted-ROI candidate cross-section metrology.</td><td>COMPLETE</td></tr>
<tr><td>Fathom Field (Raw)</td><td>Structure-tensor orientation, anisotropic EDT and paired-edge / intensity-profile widths on the skeleton centerline.</td><td>EXPERIMENTAL_FIELD_MEASURING</td></tr>
<tr><td>Fathom Oriented Ribbon V1</td><td>Pairs opposite mask boundaries, forms geometric midpoints, fits a confidence-weighted smooth centerline on non-branching runs and re-measures EDT / paired-edge / profile along it. Real SEM results represent method behavior and agreement, not known absolute accuracy.</td><td>EXPERIMENTAL</td></tr>
<tr><td>Manual 5×5</td><td>Operator reference grid (25 positions per image). Sparse human reference; not ground truth.</td><td>{'INCOMPLETE' if (manual_store is None or manual_store.total_measured < 25) else 'COMPLETE'}</td></tr>
</table>
<h2>4. Per-image results (median µm)</h2>
<table><tr><th>Image</th><th>MATLAB b1</th><th>Python</th><th>Local</th><th>Raw EDT</th><th>Ribbon EDT</th><th>Raw Edge</th><th>Ribbon Edge</th><th>Raw Profile</th><th>Ribbon Profile</th><th>Manual</th><th>Ribbon cov</th></tr>{per_image_rows}</table>
<h2>5. Diameter distributions</h2>
<p>Per-image Figure A (weighted density, shared physical bins), Figure B (ECDF) and Figure C (Fathom Field raw vs Ribbon) are rendered below.</p>
<h2>6. Manual 5×5</h2>
{manual_section}
<h2>7. Fathom Field diagnostics</h2>
<p>Orientation coherence, paired-edge acceptance, profile acceptance and asymmetry are reported per image in the quality section.</p>
<h2>8. Oriented Ribbon refinement</h2>
<p>Dataset figures A–E below show raw vs refined EDT/Edge medians, EDT\u2194Edge agreement, center shifts and coverage.</p>
<h2>9. Method comparisons</h2>
<p class="notebox">MATLAB SIMPoly native Gaussian center b1 is reported; a common distribution is unavailable from the current MATLAB cache and no histogram is fabricated. Python SIMPoly supplies compatible samples for distribution comparisons.</p>
<h2>10. Quality / abstention</h2>
<p>The Ribbon deliberately abstains on crossings, junctions, gaps and low-confidence regions; unsupported samples are not failures.</p>
<h2>11. Provenance</h2>
{provenance}
<h2>12. Limitations</h2>
<ul>
<li>Measurements represent projected 2-D geometry; no 3-D claim.</li>
<li>Fathom Field and Oriented Ribbon V1 are experimental; SEM results describe agreement, not known accuracy.</li>
<li>Graph reconstruction, fiber instances and ML are not implemented.</li>
<li>Manual 5×5 remains a sparse human reference.</li>
</ul>
<h2>Dataset figures</h2>
<img src="dataset-figure-A.png" alt="raw vs refined medians"><br>
<img src="dataset-figure-B.png" alt="W1 raw vs refined"><br>
<img src="dataset-figure-C.png" alt="center shifts"><br>
<img src="dataset-figure-D.png" alt="asymmetry"><br>
<img src="dataset-figure-E.png" alt="coverage">
<h2>Per-image figures</h2>
{per_image_figure_links}
</body></html>""",
        encoding="utf-8",
    )
    return index


def _distribution_median(distribution: Any) -> float | None:
    if distribution is None:
        return None
    return float(np.median(distribution.diameter))


def _final_per_image_figures(output: Path, comparison: Any, stem: str) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    image_dir = output / "images" / stem
    image_dir.mkdir(parents=True, exist_ok=True)
    series = series_distributions(comparison)
    if not series:
        return
    distributions = [item[1] for item in series]
    edges = common_histogram_edges(distributions)
    if not edges.size:
        return

    figure = Figure(figsize=(7, 4))
    axis = figure.add_subplot(111)
    centers = edges[:-1] + np.diff(edges) / 2.0
    for name, distribution in series:
        axis.plot(
            centers,
            _weighted_density(distribution, edges),
            label=f"{name} (N={distribution.diameter.size})",
            color=SERIES_COLORS.get(name),
            drawstyle="steps-mid",
        )
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title(f"{stem} — common diameter histogram")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    FigureCanvasAgg(figure).print_figure(image_dir / "figure-A-histogram.png", dpi=120)
    figure.clear()

    figure = Figure(figsize=(7, 4))
    axis = figure.add_subplot(111)
    for name, distribution in series:
        order = np.argsort(distribution.diameter, kind="stable")
        x = distribution.diameter[order]
        y = np.cumsum(distribution.weight[order]) / distribution.weight.sum()
        axis.step(x, y, where="post", label=f"{name} (N={distribution.diameter.size})", color=SERIES_COLORS.get(name))
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Cumulative weight")
    axis.set_title(f"{stem} — ECDF")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    FigureCanvasAgg(figure).print_figure(image_dir / "figure-B-ecdf.png", dpi=120)
    figure.clear()

    ribbon = [item for item in series if item[0].startswith("Ribbon")]
    raw_field = [
        item for item in series
        if item[0] in {"Fathom Field (EDT)", "Field Paired Edge", "Field Intensity Profile"}
    ]
    if ribbon:
        figure = Figure(figsize=(7, 4))
        axis = figure.add_subplot(111)
        all_field = raw_field + ribbon
        field_distributions = [item[1] for item in all_field]
        field_edges = common_histogram_edges(field_distributions)
        if field_edges.size:
            centers = field_edges[:-1] + np.diff(field_edges) / 2.0
            for name, distribution in all_field:
                axis.plot(
                    centers,
                    _weighted_density(distribution, field_edges),
                    label=f"{name} (N={distribution.diameter.size})",
                    color=SERIES_COLORS.get(name),
                    drawstyle="steps-mid",
                )
        axis.set_xlabel("Diameter (µm)")
        axis.set_ylabel("Weighted density (1/µm)")
        axis.set_title(f"{stem} — Fathom Field raw vs Ribbon")
        axis.legend(fontsize="small")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        FigureCanvasAgg(figure).print_figure(image_dir / "figure-C-field-ribbon.png", dpi=120)
        figure.clear()


def _final_manual_section(manual_store: Any, dataset: Any) -> str:
    if manual_store is None:
        return "<p>Manual 5×5 store not loaded.</p>"
    total = manual_store.total_measured
    rows = ""
    for image in dataset.images:
        review = manual_store.reviews.get(image.case_id)
        count = review.measurement_count if review is not None else 0
        rows += (
            f"<tr><td>{html.escape(image.filename)}</td><td>{count} / 25</td></tr>"
        )
    status = "COMPLETE REFERENCE" if total >= 400 else "INCOMPLETE REFERENCE"
    return (
        f"<p>Measurements recorded: <b>{total} / 400</b> — {status}.</p>"
        f"<table><tr><th>Image</th><th>Progress</th></tr>{rows}</table>"
        f"<p class='notebox'>The report is generated regardless of manual completeness; "
        f"missing manual values are never filled in.</p>"
    )


def _final_ribbon_box(metrics: dict[str, Any]) -> str:
    import statistics

    w1_raw = metrics["w1_edt_edge_raw"]
    w1_refined = metrics["w1_edt_edge_refined"]
    if not w1_raw:
        return "<div class='resultbox'><b>Oriented Ribbon V1:</b> no comparable distributions.</div>"
    return (
        "<div class='resultbox'><b>Oriented Ribbon V1 (experimental):</b> "
        f"{metrics['improved_count']}/{metrics['total']} images show "
        "W1(refined EDT ↔ refined Edge) &lt; W1(raw EDT ↔ raw Edge); "
        f"dataset median W1 {statistics.median(w1_raw):.3f} → {statistics.median(w1_refined):.3f} µm.</div>"
    )


def _final_provenance(dataset: Any, repo: Path, cache: WorkspaceCache) -> str:
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()[:12]
    except Exception:
        commit = "unknown"
    hashes = {
        image.case_id: image.sha256[:12] if image.sha256 else "—"
        for image in dataset.images
    }
    sample_hashes = ", ".join(f"{k}: {v}" for k, v in list(hashes.items())[:4])
    return (
        "<table><tr><th>Application</th><td>Fathom Fibers</td></tr>"
        f"<tr><th>Commit</th><td>{html.escape(commit)}</td></tr>"
        f"<tr><th>Cache schema</th><td>{cache.FULL_SCHEMA}</td></tr>"
        f"<tr><th>Input image hashes (first four)</th><td>{html.escape(sample_hashes)}</td></tr>"
        f"<tr><th>Dataset</th><td>{html.escape(str(dataset.dataset_id))}</td></tr>"
        "<tr><th>MATLAB cache</th><td>validated controlled-origin cache; native b1 only</td></tr>"
        "</table>"
    )
