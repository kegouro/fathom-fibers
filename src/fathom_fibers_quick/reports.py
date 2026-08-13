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
from .core.methods import DiameterDistribution, Estimand, MethodId, MethodResult, MethodStatus
from .report_style import short_name
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


DATASET_FIGURE_A_TITLE = "Python SIMPoly vs Fathom Local — per-image medians"


def _series_style(name: str) -> tuple[str, str, str]:

    """Color-blind-safe color, line style and short label per series family."""
    styles = {
        "Python SIMPoly": ("#e69f00", "solid"),
        "Fathom Local": ("#009e73", "dashed"),
        "Fathom Field (EDT)": ("#9aa7c9", "dashed"),
        "Field Paired Edge": ("#9cc4c4", "dashed"),
        "Field Intensity Profile": ("#c2a8cc", "dashed"),
        "Ribbon Refined EDT": ("#4d648d", "solid"),
        "Ribbon Refined Edge": ("#3b7f7f", "solid"),
        "Ribbon Refined Profile": ("#76538a", "solid"),
        "Manual 5×5": ("#c44e52", "dashdot"),
        "Consensus": ("#555555", "solid"),
    }
    color, style = styles.get(name, ("#777777", "solid"))
    return color, style, short_name(name)


def _primary_x_max(series: list[tuple[str, DiameterDistribution]]) -> float | None:
    """Data-driven right edge of the primary comparison range.

    Uses the weighted P99 of the primary/common estimators only (Python
    SIMPoly, Field raw and Ribbon family, Consensus, Manual when present).
    Fathom Local, a deliberately alternative estimator, is excluded from
    scale selection only; its full distribution stays visible in the
    full-range views.
    """
    from .core.distributions import weighted_quantile

    excluded = {"Fathom Local"}
    p99s: list[float] = []
    for name, distribution in series:
        if name in excluded or distribution.diameter.size == 0:
            continue
        value = float(weighted_quantile(distribution.diameter, distribution.weight, np.array([0.99]))[0])
        if np.isfinite(value) and value > 0:
            p99s.append(value)
    if not p99s:
        return None
    return float(np.max(p99s)) * 1.12


def _long_tail_ratio(series: list[tuple[str, DiameterDistribution]], name: str) -> float | None:
    distribution = next((item[1] for item in series if item[0] == name), None)
    if distribution is None or distribution.diameter.size == 0:
        return None
    from .core.distributions import summarize_distribution, weighted_quantile

    summary = summarize_distribution(distribution)
    if summary.weighted_median is None or summary.weighted_median <= 0:
        return None
    return float(weighted_quantile(distribution.diameter, distribution.weight, np.array([0.95]))[0]) / summary.weighted_median


def _common_series(series: list[tuple[str, DiameterDistribution]]) -> list[tuple[str, DiameterDistribution]]:
    return [(name, dist) for name, dist in series if name != "Fathom Local"]


def _legend(axis, series: list[tuple[str, DiameterDistribution]]) -> None:
    handles = []
    labels = []
    for name, distribution in series:
        color, style, short = _series_style(name)
        handle = axis.plot([], [], color=color, linestyle=style, linewidth=2)
        handles.append(handle[0])
        labels.append(short)
    axis.legend(handles, labels, fontsize="small", ncol=2, loc="upper right", framealpha=0.9)


def figure_primary_histogram(series: list[tuple[str, DiameterDistribution]], output_dir: Path, x_max: float | None) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    distributions = [item[1] for item in series]
    edges = common_histogram_edges(distributions)
    if not edges.size:
        axis.text(0.5, 0.5, "No common distributions available", ha="center")
    else:
        centers = edges[:-1] + np.diff(edges) / 2.0
        for name, distribution in series:
            color, style, label = _series_style(name)
            axis.plot(
                centers, _weighted_density(distribution, edges),
                label=label, color=color, linestyle=style, linewidth=1.9, drawstyle="steps-mid",
            )
    if x_max is not None and edges.size:
        axis.set_xlim(0, x_max)
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title("Common diameter histogram — primary range")
    axis.grid(alpha=0.2)
    _legend(axis, series)
    figure.tight_layout()
    path = output_dir / "figure-primary-histogram.png"
    _save(figure, path)
    return path


def figure_full_histogram(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    distributions = [item[1] for item in series]
    edges = common_histogram_edges(distributions)
    if not edges.size:
        axis.text(0.5, 0.5, "No common distributions available", ha="center")
    else:
        centers = edges[:-1] + np.diff(edges) / 2.0
        for name, distribution in series:
            color, style, label = _series_style(name)
            axis.plot(
                centers, _weighted_density(distribution, edges),
                label=label, color=color, linestyle=style, linewidth=1.9, drawstyle="steps-mid",
            )
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title("Common diameter histogram — full observed range")
    axis.grid(alpha=0.2)
    _legend(axis, series)
    figure.tight_layout()
    path = output_dir / "figure-full-histogram.png"
    _save(figure, path)
    return path


def figure_ecdf_primary(series: list[tuple[str, DiameterDistribution]], output_dir: Path, x_max: float | None) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    for name, distribution in series:
        if not distribution.diameter.size:
            continue
        order = np.argsort(distribution.diameter, kind="stable")
        x = distribution.diameter[order]
        y = np.cumsum(distribution.weight[order]) / distribution.weight.sum()
        color, style, label = _series_style(name)
        axis.step(x, y, where="post", label=label, color=color, linestyle=style, linewidth=1.9)
    if x_max is not None:
        axis.set_xlim(0, x_max)
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Cumulative weight")
    axis.set_title("Diameter ECDF — primary range")
    axis.grid(alpha=0.2)
    _legend(axis, series)
    figure.tight_layout()
    path = output_dir / "figure-primary-ecdf.png"
    _save(figure, path)
    return path


def figure_ecdf_full(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    for name, distribution in series:
        if not distribution.diameter.size:
            continue
        order = np.argsort(distribution.diameter, kind="stable")
        x = distribution.diameter[order]
        y = np.cumsum(distribution.weight[order]) / distribution.weight.sum()
        color, style, label = _series_style(name)
        axis.step(x, y, where="post", label=label, color=color, linestyle=style, linewidth=1.9)
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Cumulative weight")
    axis.set_title("Diameter ECDF — full observed range")
    axis.grid(alpha=0.2)
    _legend(axis, series)
    figure.tight_layout()
    path = output_dir / "figure-full-ecdf.png"
    _save(figure, path)
    return path


def figure_field_raw_vs_ribbon(comparison: UnifiedMethodComparison, output_dir: Path) -> Path:
    """Two-panel comparison: raw estimators left, Ribbon re-measurements right."""
    from matplotlib.gridspec import GridSpec

    series = series_distributions(comparison)
    raw = [item for item in series if item[0] in {"Fathom Field (EDT)", "Field Paired Edge", "Field Intensity Profile"}]
    ribbon = [item for item in series if item[0] in {"Ribbon Refined EDT", "Ribbon Refined Edge", "Ribbon Refined Profile"}]
    figure = _new_figure()
    figure.set_size_inches(9.6, 4.2)
    grid = GridSpec(1, 2, wspace=0.3, figure=figure)
    for panel_index, (panel_series, title) in enumerate(
        ((raw, "Field estimators — raw"), (ribbon, "Field estimators — Ribbon V1"))
    ):
        axis = figure.add_subplot(grid[panel_index])
        distributions = [item[1] for item in panel_series]
        edges = common_histogram_edges(distributions)
        if not edges.size:
            axis.text(0.5, 0.5, "No distributions", ha="center")
        else:
            centers = edges[:-1] + np.diff(edges) / 2.0
            for name, distribution in panel_series:
                color, style, label = _series_style(name)
                axis.plot(
                    centers, _weighted_density(distribution, edges),
                    label=label, color=color, linestyle=style, linewidth=2.0, drawstyle="steps-mid",
                )
        axis.set_xlabel("Diameter (µm)")
        axis.set_ylabel("Weighted density (1/µm)")
        axis.set_title(title, fontsize="small")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    figure.suptitle("Field estimators — raw vs Oriented Ribbon V1", fontsize="medium")
    figure.tight_layout()
    path = output_dir / "figure-field-raw-vs-ribbon.png"
    _save(figure, path)
    return path


def _weighted_box_bxp(axis, series: list[tuple[str, DiameterDistribution]], title: str) -> None:
    from .core.distributions import weighted_quantile

    boxes = []
    labels = []
    for name, distribution in series:
        if distribution.diameter.size == 0:
            continue
        q = weighted_quantile(distribution.diameter, distribution.weight, np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
        if not np.isfinite(q).all():
            continue
        boxes.append(
            {
                "med": float(q[2]), "q1": float(q[1]), "q3": float(q[3]),
                "whislo": float(q[0]), "whishi": float(q[4]),
            }
        )
        labels.append(short_name(name))
    if not boxes:
        axis.text(0.5, 0.5, "No comparable distributions", ha="center")
        return
    axis.bxp(boxes, showfliers=False)
    axis.set_xticklabels(labels, rotation=20, fontsize="small")
    axis.set_ylabel("Diameter (µm)")
    axis.set_title(title, fontsize="small")
    axis.grid(alpha=0.2, axis="y")


def figure_method_summary_primary(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    _weighted_box_bxp(axis, _common_series(series), "Method summary — primary methods (weighted P05/P25/P50/P75/P95)")
    figure.tight_layout()
    path = output_dir / "figure-method-summary-primary.png"
    _save(figure, path)
    return path


def figure_method_summary_full(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    _weighted_box_bxp(axis, series, "Method summary — full range incl. Fathom Local (weighted P05/P25/P50/P75/P95)")
    figure.tight_layout()
    path = output_dir / "figure-method-summary-full.png"
    _save(figure, path)
    return path


def figure_local_tail(series: list[tuple[str, DiameterDistribution]], output_dir: Path) -> Path:
    figure = _new_figure()
    axis = figure.add_subplot(111)
    distributions = [item[1] for item in series]
    edges = common_histogram_edges(distributions)
    centers = edges[:-1] + np.diff(edges) / 2.0
    for name, distribution in series:
        color, style, label = _series_style(name)
        axis.plot(
            centers, _weighted_density(distribution, edges),
            label=label, color=color, linestyle=style, linewidth=1.9, drawstyle="steps-mid",
        )
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Weighted density (1/µm)")
    axis.set_title("Fathom Local — distribution tail in context")
    axis.grid(alpha=0.2)
    _legend(axis, series)
    figure.tight_layout()
    path = output_dir / "figure-local-tail.png"
    _save(figure, path)
    return path


def _fmt(value: float | None, digits: int = 5) -> str:
    return "—" if value is None else f"{value:.{digits}g}"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"{low:.5g}–{high:.5g}"


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


def _estimate_table_rows(comparison: UnifiedMethodComparison) -> list[tuple[str, str, str, str, str, str, str, str]]:
    """(method, status, N, coverage, mean, median, IQR, P05-P95) rows."""
    rows: list[tuple[str, str, str, str, str, str, str, str]] = []
    for result in comparison.results:
        values, _summary = summary_row(result)
        rows.append(
            (
                display_name(result.method_id),
                values[1],
                values[2],
                "—",
                values[3],
                values[4],
                values[5],
                f"{values[6]}–{values[7]}",
            )
        )
    return rows


def _grouped_method_table(comparison: UnifiedMethodComparison) -> str:
    from .report_style import grouped_table

    def field_rows(estimator_names: list[tuple[str, str]]) -> list[list[Any]]:
        field = next(
            (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
        )
        rows: list[list[Any]] = []
        if field is None:
            return rows
        statistics = field.native_statistics
        coverage_by_key = {
            "FATHOM_FIELD_REFINED_EDT_DIAMETER": statistics.get("smooth_coverage_fraction"),
            "FATHOM_FIELD_REFINED_EDGE_DIAMETER": statistics.get("refined_edge_acceptance_fraction"),
            "FATHOM_FIELD_REFINED_PROFILE_DIAMETER": statistics.get("refined_profile_acceptance_fraction"),
        }
        for key, label in estimator_names:
            distribution = field.secondary_distributions.get(key)
            status = "EXPERIMENTAL" if key.startswith("FATHOM_FIELD_REFINED_") else field.status.value
            coverage = coverage_by_key.get(key)
            if distribution is None:
                rows.append([label, status, "—", "—", "—", "—", "—", "—"])
                continue
            summary = summarize_distribution(distribution)
            rows.append(
                [
                    label, status, str(summary.n),
                    "—" if coverage is None else f"{coverage:.1%}",
                    _fmt(summary.weighted_mean), _fmt(summary.weighted_median),
                    _fmt_range(summary.p25, summary.p75),
                    f"{_fmt(summary.p05)}–{_fmt(summary.p95)}",
                ]
            )
        return rows

    headers = ["Method / estimator", "Status", "N", "Coverage", "Mean", "Median", "IQR", "P05–P95"]
    groups: list[tuple[str, list[str], list[list[Any]]]] = []

    matlab = next((r for r in comparison.results if r.method_id == MethodId.MATLAB_SIMPOLY), None)
    python = next((r for r in comparison.results if r.method_id == MethodId.PYTHON_SIMPOLY), None)
    local = next((r for r in comparison.results if r.method_id == MethodId.FATHOM_LOCAL), None)
    manual = next((r for r in comparison.results if r.method_id == MethodId.MANUAL_5X5_REFERENCE), None)

    reference_rows: list[list[Any]] = []
    if matlab is not None:
        reference_rows.append(
            [
                "MATLAB SIMPoly — native Gaussian center b1",
                matlab.status.value, "—", "—", "—",
                _fmt(matlab.native_result) + " µm", "—",
                "Common sample distribution unavailable from current cache",
            ]
        )
    if python is not None:
        values, _summary = summary_row(python)
        reference_rows.append([values[0], values[1], values[2], "—", values[3], values[4], values[5], f"{values[6]}–{values[7]}"])
    groups.append(("REFERENCE / COMPATIBILITY", headers, reference_rows))

    independent_rows: list[list[Any]] = []
    if local is not None:
        values, _summary = summary_row(local)
        independent_rows.append([values[0], values[1], values[2], "—", values[3], values[4], values[5], f"{values[6]}–{values[7]}"])
    groups.append(("INDEPENDENT ESTIMATOR", headers, independent_rows))

    raw_field_rows = field_rows(
        [
            ("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "Raw Paired Edge"),
            ("FATHOM_FIELD_PROFILE_DIAMETER", "Raw Intensity Profile"),
        ]
    )
    field = next((r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None)
    if field is not None and field.common_distribution is not None:
        summary = summarize_distribution(field.common_distribution)
        raw_field_rows.insert(
            0,
            [
                "Raw EDT", field.status.value, str(summary.n), "—",
                _fmt(summary.weighted_mean), _fmt(summary.weighted_median),
                _fmt_range(summary.p25, summary.p75),
                f"{_fmt(summary.p05)}–{_fmt(summary.p95)}",
            ],
        )
    groups.append(("FATHOM FIELD — RAW", headers, raw_field_rows))

    ribbon_rows = field_rows(
        [
            ("FATHOM_FIELD_REFINED_EDT_DIAMETER", "Ribbon EDT"),
            ("FATHOM_FIELD_REFINED_EDGE_DIAMETER", "Ribbon Edge"),
            ("FATHOM_FIELD_REFINED_PROFILE_DIAMETER", "Ribbon Profile"),
        ]
    )
    groups.append(("FATHOM FIELD — ORIENTED RIBBON V1 (EXPERIMENTAL)", headers, ribbon_rows))

    human_rows: list[list[Any]] = []
    if manual is not None:
        values, _summary = summary_row(manual)
        human_rows.append(
            [values[0], values[1], values[2], "—", values[3], values[4], values[5], f"{values[6]}–{values[7]}"]
        )
    groups.append(("HUMAN REFERENCE", headers, human_rows))

    consensus = comparison.consensus
    if consensus.distribution is not None:
        summary = summarize_distribution(consensus.distribution)
        groups.append(
            (
                "PSEUDO-REFERENCE",
                headers,
                [
                    [
                        "Consensus", "COMPLETE", str(summary.n), "—",
                        _fmt(summary.weighted_mean), _fmt(summary.weighted_median),
                        _fmt_range(summary.p25, summary.p75),
                        f"{_fmt(summary.p05)}–{_fmt(summary.p95)}",
                    ]
                ],
            )
        )
    return grouped_table(groups)


def _scientific_summary_cards(comparison: UnifiedMethodComparison) -> str:
    from .report_style import cards

    series = {name: distribution for name, distribution in series_distributions(comparison)}
    items: list[tuple[str, str, str]] = []

    def median_of(name: str) -> str:
        distribution = series.get(name)
        if distribution is None:
            return "—"
        return _fmt(float(np.median(distribution.diameter))) + " µm"

    for name in (
        "Python SIMPoly",
        "Fathom Field (EDT)",
        "Field Paired Edge",
        "Field Intensity Profile",
        "Ribbon Refined EDT",
        "Ribbon Refined Edge",
        "Ribbon Refined Profile",
        "Manual 5×5",
    ):
        items.append((median_of(name), "median " + short_name(name), _n_of(series.get(name))))

    field = next(
        (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
    )
    if field is not None:
        statistics = field.native_statistics
        items.append((_fmt(statistics.get("smooth_coverage_fraction"), 4), "Ribbon supported coverage", "fraction of samples with a refined centerline"))
        items.append(
            (
                _fmt(statistics.get("edge_acceptance_fraction"), 4) + " → " + _fmt(statistics.get("refined_edge_acceptance_fraction"), 4),
                "Edge acceptance raw → refined", "paired-edge acceptance",
            )
        )
        items.append(
            (
                _fmt(statistics.get("refine_median_shift_um"), 4) + " → " + _fmt(statistics.get("refined_residual_shift_median_um"), 4) + " µm",
                "Center shift observed → residual", "median values",
            )
        )
        items.append(
            (
                _fmt(statistics.get("edge_median_asymmetry"), 4) + " → " + _fmt(statistics.get("refined_asymmetry_median"), 4),
                "Asymmetry raw → refined", "median values",
            )
        )
    return cards(items)


def _n_of(distribution: DiameterDistribution | None) -> str:
    return f"N = {distribution.diameter.size:,}" if distribution is not None else "N = —"


def _refinement_effect_table(comparison: UnifiedMethodComparison) -> str:
    from .report_style import table

    field = next(
        (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
    )
    if field is None:
        return table([], [])
    rows: list[list[Any]] = []
    estimators = (
        ("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "Paired Edge"),
        ("FATHOM_FIELD_PROFILE_DIAMETER", "Intensity Profile"),
    )
    raw_common = field.common_distribution
    if raw_common is not None:
        summary = summarize_distribution(raw_common)
        ribbon = field.secondary_distributions.get("FATHOM_FIELD_REFINED_EDT_DIAMETER")
        rsummary = summarize_distribution(ribbon) if ribbon is not None else None
        rows.append(
            [
                "EDT",
                str(summary.n), _fmt(summary.weighted_median),
                str(rsummary.n) if rsummary else "—", _fmt(rsummary.weighted_median) if rsummary else "—",
                _fmt(rsummary.weighted_median - summary.weighted_median) if rsummary else "—",
                _fmt(summary.p05), _fmt(summary.p95),
                _fmt(rsummary.p05) if rsummary else "—", _fmt(rsummary.p95) if rsummary else "—",
            ]
        )
    for key, label in estimators:
        raw = field.secondary_distributions.get(key)
        ribbon = field.secondary_distributions.get(key.replace("FATHOM_FIELD_", "FATHOM_FIELD_REFINED_"))
        rsummary = summarize_distribution(ribbon) if ribbon is not None else None
        if raw is None:
            continue
        summary = summarize_distribution(raw)
        rows.append(
            [
                label,
                str(summary.n), _fmt(summary.weighted_median),
                str(rsummary.n) if rsummary else "—", _fmt(rsummary.weighted_median) if rsummary else "—",
                _fmt(rsummary.weighted_median - summary.weighted_median) if rsummary else "—",
                _fmt(summary.p05), _fmt(summary.p95),
                _fmt(rsummary.p05) if rsummary else "—", _fmt(rsummary.p95) if rsummary else "—",
            ]
        )
    return table(
        ["Estimator", "Raw N", "Raw median", "Ribbon N", "Ribbon median",
         "Median change", "Raw P05", "Raw P95", "Ribbon P05", "Ribbon P95"],
        rows,
    )


def _agreement_section(comparison: UnifiedMethodComparison) -> str:
    from .report_style import details, table

    definitions = (
        "<p>Wasserstein-1 reports the typical transport distance between two diameter "
        "distributions, in µm. KS is the maximum separation between their cumulative "
        "distributions. Median difference is the signed difference A − B in µm. "
        "These quantify difference/agreement, not accuracy.</p>"
    )
    agreement_map = {
        (item.left_method, item.right_method): item
        for item in comparison.agreements
        if item.wasserstein_1 is not None
    }

    def pair_rows(pairs) -> list[list[Any]]:
        rows: list[list[Any]] = []
        for left, right in pairs:
            item = agreement_map.get((left, right)) or agreement_map.get((right, left))
            if item is None:
                continue
            rows.append(
                [
                    display_name(left) + " vs " + display_name(right),
                    _fmt(item.wasserstein_1), _fmt(item.ks_statistic), _fmt(item.median_difference),
                ]
            )
        return rows

    primary = pair_rows([(MethodId.PYTHON_SIMPOLY, MethodId.FATHOM_LOCAL), (MethodId.PYTHON_SIMPOLY, MethodId.FATHOM_FIELD_GRAPH_V1)])
    field = next((r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None)
    internal: list[list[Any]] = []
    refinement: list[list[Any]] = []
    if field is not None:
        for name, label in (
            ("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "Raw Edge vs Raw EDT"),
            ("FATHOM_FIELD_PROFILE_DIAMETER", "Raw Profile vs Raw EDT"),
        ):
            distribution = field.secondary_distributions.get(name)
            if distribution is not None and field.common_distribution is not None:
                internal.append(
                    [
                        f"Field {label}",
                        _fmt(_w1_pair(field.common_distribution.diameter, distribution.diameter)),
                        _fmt(_ks_pair(field.common_distribution.diameter, distribution.diameter)),
                        "—",
                    ]
                )
        raw_edge = field.secondary_distributions.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER")
        ribbon_edge = field.secondary_distributions.get("FATHOM_FIELD_REFINED_EDGE_DIAMETER")
        if raw_edge is not None and ribbon_edge is not None:
            refinement.append(
                [
                    "Raw Edge vs Ribbon Edge",
                    _fmt(_w1_pair(raw_edge.diameter, ribbon_edge.diameter)),
                    _fmt(_ks_pair(raw_edge.diameter, ribbon_edge.diameter)),
                    "—",
                ]
            )
        raw_edt = field.common_distribution
        ribbon_edt = field.secondary_distributions.get("FATHOM_FIELD_REFINED_EDT_DIAMETER")
        if raw_edt is not None and ribbon_edt is not None:
            refinement.append(
                [
                    "Raw EDT vs Ribbon EDT",
                    _fmt(_w1_pair(raw_edt.diameter, ribbon_edt.diameter)),
                    _fmt(_ks_pair(raw_edt.diameter, ribbon_edt.diameter)),
                    "—",
                ]
            )
        raw_profile = field.secondary_distributions.get("FATHOM_FIELD_PROFILE_DIAMETER")
        ribbon_profile = field.secondary_distributions.get("FATHOM_FIELD_REFINED_PROFILE_DIAMETER")
        if raw_profile is not None and ribbon_profile is not None:
            refinement.append(
                [
                    "Raw Profile vs Ribbon Profile",
                    _fmt(_w1_pair(raw_profile.diameter, ribbon_profile.diameter)),
                    _fmt(_ks_pair(raw_profile.diameter, ribbon_profile.diameter)),
                    "—",
                ]
            )
    headers = ["Comparison", "W1 (µm)", "KS", "Median Δ (µm)"]
    body = definitions
    body += "<h3>Primary cross-method comparisons</h3>" + (table(headers, primary) if primary else "<p>No comparable pairs.</p>")
    body += "<h3>Field internal comparisons</h3>" + (table(headers, internal) if internal else "<p>No comparable pairs.</p>")
    body += "<h3>Refinement effect</h3>" + (table(headers, refinement) if refinement else "<p>No comparable pairs.</p>")
    all_rows = [
        [display_name(item.left_method), display_name(item.right_method), _fmt(item.wasserstein_1), _fmt(item.ks_statistic), _fmt(item.median_difference)]
        for item in comparison.agreements
        if item.wasserstein_1 is not None
    ]
    body += details("Full pairwise comparison table (appendix)", table(["A", "B", "W1 (µm)", "KS", "Median Δ (µm)"], all_rows) if all_rows else "<p>No comparable pairs.</p>")
    return body


def _ks_pair(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import ks_2samp

    if a.size == 0 or b.size == 0:
        return None
    return float(ks_2samp(a, b).statistic)


def build_image_report(
    comparison: UnifiedMethodComparison,
    image: Any,
    *,
    output_dir: str | Path,
    manual_complete: bool | None = None,
    manual_count: int = 0,
) -> Path:
    """Render one image's scientific report and return its index.html path."""
    from .report_style import (
        cards,
        info,
        note,
        page,
        report_header,
        section,
        table,
    )
    from .report_style import (
        figure as figure_block,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    series = series_distributions(comparison)
    x_max = _primary_x_max(series)
    figure_primary_histogram(series, output, x_max)
    figure_full_histogram(series, output)
    figure_ecdf_primary(series, output, x_max)
    figure_ecdf_full(series, output)
    figure_field_raw_vs_ribbon(comparison, output)
    figure_method_summary_primary(series, output)
    figure_method_summary_full(series, output)

    local_ratio = _long_tail_ratio(series, "Fathom Local")
    common_ratio = _long_tail_ratio(series, "Python SIMPoly")
    tail_triggered = bool(
        local_ratio is not None and (common_ratio is None or local_ratio > max(2.5, 2.0 * (common_ratio or 1.0)))
    )
    if tail_triggered:
        figure_local_tail(series, output)

    calibration = image.calibration
    metadata = dict(image.metadata or {})
    field = next(
        (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
    )
    matlab = next((r for r in comparison.results if r.method_id == MethodId.MATLAB_SIMPOLY), None)
    manual = next((r for r in comparison.results if r.method_id == MethodId.MANUAL_5X5_REFERENCE), None)

    if manual_complete is None and manual is not None:
        manual_complete = manual.status == MethodStatus.COMPLETE and (
            manual.native_statistics.get("measurement_count", 0) >= 25
        )
        manual_count = int(manual.native_statistics.get("measurement_count", 0))

    chips = [
        ("Calibration", f"{calibration.pixel_size_x_m * 1e9:.5g} × {calibration.pixel_size_y_m * 1e9:.5g} nm/px ({calibration.source})"),
        ("ROI", str(next((result.valid_roi for result in comparison.results if result.valid_roi), None))),
    ]
    if metadata.get("ap_mag") is not None:
        chips.append(("Magnification", str(metadata["ap_mag"])))
    if metadata.get("ap_actualkv") is not None:
        chips.append(("EHT", f"{metadata['ap_actualkv']} kV"))
    chips.append(("Generated", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")))
    badges = [("AUTOMATIC ANALYSIS COMPLETE", "ok")]
    if manual is not None and manual.status != MethodStatus.NOT_MEASURED:
        badges.append(("MANUAL REFERENCE INCOMPLETE" if not manual_complete else "MANUAL REFERENCE COMPLETE", "warn" if not manual_complete else "ok"))
    else:
        badges.append(("MANUAL REFERENCE INCOMPLETE", "warn"))
    badges.append(("RIBBON EXPERIMENTAL", "exp"))

    head = report_header(
        "Fathom Fibers — Scientific Fiber Morphology Analysis",
        comparison.image_id,
        chips,
        badges,
    )

    summary_cards = _scientific_summary_cards(comparison)
    body = section(
        "Scientific Summary",
        summary_cards
        + "<p>Median diameters per method (µm). Coverage and acceptance describe the "
        "fraction of supported samples; unmeasured quantities show —.</p>",
    )

    body += section(
        "Key diameter results",
        _grouped_method_table(comparison)
        + note(
            "Consensus is a pseudo-reference across participating methods, not ground truth. "
            "Measurements represent projected 2-D geometry."
        ),
    )

    refinement_metrics = ""
    if field is not None:
        statistics = field.native_statistics
        refinement_metrics = cards(
            [
                (str(statistics.get("smooth_segment_count", "—")), "Refined segments", "supported non-branching runs"),
                (_fmt(statistics.get("refine_median_shift_um"), 4) + " µm", "Median observed center shift", "before refinement"),
                (_fmt(statistics.get("refined_residual_shift_median_um"), 4) + " µm", "Median residual center shift", "after refinement"),
                (_fmt(statistics.get("smooth_coverage_fraction"), 4), "Supported centerline coverage", "abstentions are intentional"),
            ]
        )
        body += section(
            "Centerline refinement effect",
            _refinement_effect_table(comparison)
            + refinement_metrics
            + info(
                "Oriented Ribbon V1 estimates a local geometric centerline from paired opposite "
                "boundaries and re-measures the Field estimators on supported, non-ambiguous "
                "centerline segments. <b>EXPERIMENTAL</b> — known-truth synthetic validation "
                "supports the centerline mechanism; real SEM comparisons characterize behavior "
                "and agreement, not known absolute accuracy."
            ),
        )

    body += section(
        "Distributions",
        "<h3>Primary range</h3>"
        + figure_block(
            "figure-primary-histogram.png",
            "Common diameter histogram, primary range",
            "Length-weighted diameter distributions for the common estimators. The primary "
            "view is restricted to the central comparison range for readability; the "
            "full-range view preserves long-tailed observations. The data are never filtered.",
        )
        + figure_block(
            "figure-primary-ecdf.png",
            "Diameter ECDF, primary range",
            "Fraction of weighted measurements below a given diameter (primary range).",
        )
        + "<h3>Full observed range</h3>"
        + figure_block(
            "figure-full-histogram.png",
            "Common diameter histogram, full range",
            "The entire supported observed range, including alternative estimators with "
            "longer tails.",
        )
        + figure_block(
            "figure-full-ecdf.png",
            "Diameter ECDF, full range",
            "Fraction of weighted measurements below a given diameter (full range).",
        ),
    )

    if tail_triggered:
        local_distribution = next((d for name, d in series if name == "Fathom Local"), None)
        local_rows = ""
        if local_distribution is not None:
            local_summary = summarize_distribution(local_distribution)
            local_rows = table(
                ["N", "Median", "IQR", "P95", "Max"],
                [
                    [
                        str(local_summary.n),
                        _fmt(local_summary.weighted_median),
                        _fmt_range(local_summary.p25, local_summary.p75),
                        _fmt(local_summary.p95),
                        _fmt(float(np.nanmax(local_distribution.diameter))),
                    ]
                ],
            )
        body += section(
            "Fathom Local — distribution tail",
            local_rows
            + figure_block(
                "figure-local-tail.png",
                "Fathom Local distribution tail in context",
                "Fathom Local shows a broader/right-tailed distribution in this image; its full "
                "observed range remains visible and no observations are removed.",
            )
            + note(
                "A broad/right-tailed distribution indicates that the local cross-section "
                "estimator samples a wider range of candidate geometries in this image. "
                "Diagnostic flags below provide context where available."
            ),
        )

    body += section(
        "Field estimators — raw vs Ribbon",
        figure_block(
            "figure-field-raw-vs-ribbon.png",
            "Field estimators raw vs Ribbon V1",
            "Raw and Ribbon re-measured estimators share the same bin edges and x-range; "
            "Ribbon uses the refined centerline on supported segments.",
        ),
    )

    body += section(
        "Method summary",
        figure_block(
            "figure-method-summary-primary.png",
            "Primary method summary",
            "Weighted percentiles: box = P25–P75, center = P50, whiskers = P05–P95. "
            "Fathom Local is shown separately in the full-range summary.",
        )
        + figure_block(
            "figure-method-summary-full.png",
            "Full-range method summary",
            "Weighted percentiles over the full observed range, including Fathom Local. "
            "Box = P25–P75, center = P50, whiskers = P05–P95.",
        ),
    )

    body += section("Method agreement", _agreement_section(comparison))

    body += _quality_section_v2(comparison)

    if manual is None or manual.status == MethodStatus.NOT_MEASURED:
        body += section(
            "Manual reference",
            note("Manual 5×5: 0 / 25 measurements — <b>INCOMPLETE REFERENCE</b>. "
                 "Missing manual values are never filled in."),
        )
    elif manual_complete:
        body += section(
            "Manual reference",
            note("Manual 5×5: 25 / 25 measurements — reference complete. "
                 "Manual is a sparse human reference, not ground truth."),
        )
    else:
        body += section(
            "Manual reference",
            note(
                f"Manual 5×5: {manual_count} / 25 measurements — <b>INCOMPLETE REFERENCE</b>. "
                "Missing manual values are never filled in."
            ),
        )

    body += _methods_cards(comparison)
    body += _matlab_card_v2(matlab)
    body += section("Provenance", _provenance_v2(comparison, image, matlab))
    body += section("Limitations", _limitations_v2(comparison))

    document = page(f"Fathom Fibers — {comparison.image_id}", head_html=head, body_html=body)
    index = output / "index.html"
    index.write_text(document, encoding="utf-8")
    return index


def _quality_section_v2(comparison: UnifiedMethodComparison) -> str:
    from .report_style import cards, details, info, section, table

    field = next(
        (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
    )
    summary_cards: list[tuple[str, str, str]] = []
    important_rows: list[list[Any]] = []
    for result in comparison.results:
        summary_cards.append(
            (result.status.value, display_name(result.method_id), f"{len(result.quality_flags)} flags")
        )
    if field is not None:
        statistics = field.native_statistics
        summary_cards.append(("—", "Ribbon supported coverage", _fmt(statistics.get("smooth_coverage_fraction"), 4)))
        summary_cards.append(("—", "Raw → refined Edge acceptance", _fmt(statistics.get("edge_acceptance_fraction"), 4) + " → " + _fmt(statistics.get("refined_edge_acceptance_fraction"), 4)))

    important_flags = {
        "POSSIBLE_CROSSING",
        "AMBIGUOUS_LOCAL_WIDTH",
        "PROFILE_AMBIGUOUS_EDGE",
        "REFINED_ORIENTATION_DISAGREEMENT",
    }
    if field is not None:
        counts: dict[str, int] = {}
        for flags in field.local_samples["edge_flags"]:
            for flag in str(flags).split(";"):
                if flag:
                    counts[flag] = counts.get(flag, 0) + 1
        for flags in field.local_samples["profile_flags"]:
            for flag in str(flags).split(";"):
                if flag:
                    counts[flag] = counts.get(flag, 0) + 1
        for flag in sorted(counts):
            if flag in important_flags:
                important_rows.append([flag, counts[flag], _flag_text(flag)])

    flag_sections = ""
    for result in comparison.results:
        if not result.quality_flags:
            continue
        rows = "".join(
            f"<tr><td>{flag}</td><td>{_flag_text(flag)}</td></tr>"
            for flag in sorted(result.quality_flags)
        )
        flag_sections += f"<h3>{display_name(result.method_id)}</h3><table><tr><th>Flag</th><th>Meaning</th></tr>{rows}</table>"
    body = cards(summary_cards)
    body += "<h3>Important diagnostics</h3>"
    body += table(["Flag", "Count", "Meaning"], important_rows) if important_rows else "<p>No important diagnostics flagged.</p>"
    body += info(
        "Ribbon coverage below 100% is not automatically a failure: unsupported samples "
        "include intentionally abstained regions (crossings, junctions, gaps, "
        "low-confidence geometry). Abstention is preferable to inventing a measurement."
    )
    body += details("Full flag breakdown (technical)", flag_sections or "<p>No flags recorded.</p>")
    return section("Quality / abstention", body)


def _flag_text(flag: str) -> str:
    from .report_style import flag_definition

    return flag_definition(flag) or flag


def _methods_cards(comparison: UnifiedMethodComparison) -> str:
    from .report_style import cards, section

    rows = [
        ("MATLAB SIMPoly", "Reference/native implementation from a validated cache.", "COMPLETE (cache)"),
        ("Python SIMPoly", "Python source-compatible approximation; calibrated length-weighted diameters on the skeleton.", "COMPLETE"),
        ("Fathom Local", "Independent local cross-section estimator; not truth and may show a broader distribution.", "COMPLETE"),
        ("Fathom Field", "Structure-tensor orientation plus local boundary metrology (EDT / paired edge / intensity profile).", "EXPERIMENTAL"),
        ("Oriented Ribbon V1", "Experimental refined centerline and re-measurement of the Field estimators.", "EXPERIMENTAL"),
        ("Manual 5×5", "Sparse human reference grid; not ground truth.", "REFERENCE"),
        ("Consensus", "Equal-method quantile pseudo-reference across participating methods; not ground truth.", "REFERENCE"),
    ]
    return section("Methods", cards([("", label, purpose + " " + status) for label, purpose, status in rows]))


def _matlab_card_v2(matlab: MethodResult | None) -> str:
    from .report_style import cards, info, section

    if matlab is None:
        return section("MATLAB SIMPoly", info("MATLAB SIMPoly not available for this image."))
    if matlab.status != MethodStatus.COMPLETE:
        return section("MATLAB SIMPoly", info(f"Status: {matlab.status.value}."))
    provenance = matlab.provenance
    b1 = matlab.native_statistics.get("gauss_b1", matlab.native_result)
    version = provenance.get("matlab_version", "R2026a")
    return section(
        "MATLAB SIMPoly",
        cards(
            [
                (f"{_fmt(b1, 6)} µm", "Native Gaussian center b1", f"reference {version}"),
                ("unavailable", "Common local distribution", "no raw diameter array in the current cache"),
            ]
        )
        + info(
            "The MATLAB cache reports the native Gaussian center b1. A common sample "
            "distribution is unavailable from the current cache, so no histogram or ECDF "
            "is fabricated for MATLAB."
        ),
    )


def _provenance_v2(comparison: UnifiedMethodComparison, image: Any, matlab: MethodResult | None) -> str:
    from .report_style import details, table

    rows: list[list[Any]] = [
        ["Image", comparison.image_id],
        ["Calibration", f"{image.calibration.pixel_size_x_m * 1e9:.5g} × {image.calibration.pixel_size_y_m * 1e9:.5g} nm/px ({image.calibration.source})"],
        ["Source", str(image.source_path)],
    ]
    for result in comparison.results:
        rows.append([f"Method {display_name(result.method_id)}", f"version {result.method_version} · status {result.status.value}"])
    if matlas := matlab:
        for key in ("source_matlab_sha256", "cache_representation", "matlab_version"):
            if matlas.provenance.get(key):
                rows.append([f"MATLAB {key}", str(matlas.provenance[key])])
    return details("Provenance / Reproducibility", table(["Item", "Value"], rows))


def _limitations_v2(comparison: UnifiedMethodComparison) -> str:
    from .report_style import info

    items = [
        "Measurements represent projected 2-D geometry; no 3-D claim is made.",
        "Agreement metrics and the consensus pseudo-reference are not ground truth.",
        "Manual 5×5 is a sparse human reference, not ground truth.",
        "MATLAB SIMPoly reports the native b1 statistic; a common raw distribution is unavailable from the current cache.",
        "Oriented Ribbon V1 is EXPERIMENTAL; real SEM comparisons characterize behavior and agreement, not known absolute accuracy.",
        "Unsupported or crossing regions may be intentionally abstained rather than measured.",
        "Fathom Local may show broad or right-tailed candidates depending on image geometry; quality flags provide context.",
    ]
    return info("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")


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

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    for method_id, marker, color in (
        (MethodId.PYTHON_SIMPOLY, "o", "#e69f00"),
        (MethodId.FATHOM_LOCAL, "s", "#009e73"),
    ):
        values = []
        for comparison in comparisons:
            result = next((r for r in comparison.results if r.method_id == method_id), None)
            distribution = result.common_distribution if result is not None else None
            values.append(float(np.median(distribution.diameter)) if distribution is not None else None)
        label = "Python SIMPoly" if method_id == MethodId.PYTHON_SIMPOLY else "Fathom Local"
        axis.plot(x, values, marker, color=color, label=label, ms=5)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median diameter (µm)")
    axis.set_title(DATASET_FIGURE_A_TITLE)
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    save(figure, "dataset-figure-A.png")

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, medians["edt_raw"], "o--", color="#9aa7c9", label="Raw EDT", ms=5)
    axis.plot(x, medians["edt_refined"], "o-", color="#4d648d", label="Ribbon EDT", ms=5)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median diameter (µm)")
    axis.set_title("Raw EDT vs Ribbon EDT by image")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    save(figure, "dataset-figure-B.png")

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, metrics["w1_edt_edge_raw"], "o--", color="#9aa7c9", label="W1 raw EDT↔Edge", ms=5)
    axis.plot(x, metrics["w1_edt_edge_refined"], "o-", color="#4d648d", label="W1 Ribbon EDT↔Edge", ms=5)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Wasserstein-1 (µm)")
    axis.set_title("Pairwise EDT↔Edge agreement, raw vs Ribbon")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    save(figure, "dataset-figure-C.png")

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, shifts["observed"], "o--", color="#b7791f", label="observed center shift", ms=5)
    axis.plot(x, shifts["residual"], "o-", color="#3b7f7f", label="residual center shift", ms=5)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median shift (µm)")
    axis.set_title("Observed vs residual center shift by image")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    save(figure, "dataset-figure-D.png")

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    axis.plot(x, asymmetry["raw"], "o--", color="#9cc4c4", label="asymmetry raw", ms=5)
    axis.plot(x, asymmetry["refined"], "o-", color="#3b7f7f", label="asymmetry refined", ms=5)
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Median asymmetry")
    axis.set_title("Paired-edge asymmetry, raw vs refined")
    axis.legend(fontsize="small")
    axis.grid(alpha=0.2, axis="y")
    figure.tight_layout()
    save(figure, "dataset-figure-E.png")

    figure = Figure(figsize=(10, 4))
    axis = figure.add_subplot(111)
    axis.bar(x, coverage, color="#4d648d")
    axis.set_xticks(x)
    axis.set_xticklabels(stems, rotation=45, ha="right", fontsize="small")
    axis.set_ylabel("Coverage")
    axis.set_ylim(0, 1)
    axis.set_title("Supported refined-centerline coverage by image")
    figure.tight_layout()
    save(figure, "dataset-figure-F.png")


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
    from .report_style import (
        cards,
        note,
        page,
        report_header,
        section,
    )
    from .report_style import (
        figure as figure_block,
    )

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
    calibration_audit = _calibration_audit(dataset, comparisons)
    _final_figures(output, comparisons, metrics)

    head = report_header(
        "Fathom Fibers — Scientific Morphological Fiber Analysis",
        "Dataset-level report generated from cached analysis results.",
        [
            ("Dataset", str(dataset.dataset_id)),
            ("Generated", datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")),
            ("Cache schema", cache.FULL_SCHEMA),
        ],
        [("ORIENTED RIBBON V1 EXPERIMENTAL", "exp")],
    )

    # executive overview
    manual_total = manual_store.total_measured if manual_store is not None else 0
    overview_cards = cards(
        [
            (str(len(dataset.images)), "Images", "dataset images"),
            (f"{len(comparisons)} / {len(dataset.images)}", "Automatic analysis", "images with cached full results"),
            (f"{manual_total} / 400", "Manual 5×5", "operator measurements recorded"),
            (_calibration_short(calibration_audit), "Calibration", "per-image pixel size"),
        ]
    )
    body = section(
        "1. Dataset overview",
        overview_cards + _calibration_audit_summary(calibration_audit) + note(
            "Dataset conclusions aggregate image-level metrics; local samples are not pooled "
            "across images into a single inferential claim."
        ),
        id_="dataset-overview",
    )

    # TOC + per-image navigation
    image_links = "".join(
        f"<a href='#image-{index + 1:02d}'>{index + 1:02d}</a>"
        for index in range(len(comparisons))
    )
    toc_sections = (
        ("dataset-overview", "Dataset overview"),
        ("dataset-method-summary", "Dataset method summary"),
        ("dataset-distribution", "Dataset-wide diameter distribution"),
        ("ribbon-dataset-behavior", "Oriented Ribbon dataset behavior"),
        ("dataset-figures", "Dataset figures"),
        ("quality-overview", "Quality overview"),
        ("manual-reference", "Manual 5×5"),
        ("methods", "Methods"),
        ("images", "Images"),
        ("provenance", "Provenance"),
        ("limitations", "Limitations"),
    )
    toc_items = "".join(f"<li><a href='#{anchor}'>{heading}</a></li>" for anchor, heading in toc_sections)
    body += f"<section id='dataset-toc'><h2>Contents</h2><div class='toc'><ol>{toc_items}</ol></div></section>"
    body += f"<section id='images-nav'><h3>Per-image navigation</h3><div class='image-nav'>{image_links}</div></section>"

    body += section("2. Dataset method summary", _dataset_method_summary(comparisons), id_="dataset-method-summary")
    body += _dataset_distribution_section(comparisons, output)
    body += section(
        "4. Oriented Ribbon dataset behavior", _dataset_ribbon_section(metrics, comparisons),
        id_="ribbon-dataset-behavior",
    )
    body += section(
        "5. Dataset figures",
        figure_block(
            "dataset-figure-A.png",
            DATASET_FIGURE_A_TITLE,
            "Median diameter per image for the two independent automatic estimators.",
        )
        + figure_block("dataset-figure-B.png", "Raw EDT vs Ribbon EDT by image", "Refined-centerline effect on EDT medians.")
        + figure_block("dataset-figure-C.png", "EDT↔Edge agreement", "Pairwise Wasserstein-1 raw vs Ribbon by image.")
        + figure_block("dataset-figure-D.png", "Center shifts", "Observed vs residual median center shift by image.")
        + figure_block("dataset-figure-E.png", "Asymmetry", "Median paired-edge asymmetry raw vs refined by image.")
        + figure_block("dataset-figure-F.png", "Ribbon coverage", "Supported refined-centerline coverage per image."),
        id_="dataset-figures",
    )
    body += section("6. Quality overview", _dataset_quality(comparisons), id_="quality-overview")
    body += section("7. Manual 5×5", _final_manual_section(manual_store, dataset), id_="manual-reference")
    body += section("8. Methods", _methods_cards_dataset(), id_="methods")
    body += section("9. Images", _dataset_image_sections(output, comparisons, dataset, manual_store), id_="images")
    body += section("10. Provenance", _final_provenance(dataset, repo, cache, calibration_audit), id_="provenance")
    body += section("11. Limitations", _dataset_limitations(), id_="limitations")

    index = output / "index.html"
    index.write_text(page("Fathom Fibers — Dataset Scientific Report", head_html=head, body_html=body), encoding="utf-8")
    return index


def _dataset_method_summary(comparisons: list[Any]) -> str:
    from .report_style import table

    estimators = (
        ("PYTHON_SIMPOLY", "Python SIMPoly", "common"),
        ("FATHOM_LOCAL", "Fathom Local", "common"),
        ("FATHOM_FIELD_GRAPH_V1", "Raw EDT", "common"),
        ("FATHOM_FIELD_GRAPH_V1", "Raw Edge", "secondary__FATHOM_FIELD_PAIRED_EDGE_DIAMETER"),
        ("FATHOM_FIELD_GRAPH_V1", "Raw Profile", "secondary__FATHOM_FIELD_PROFILE_DIAMETER"),
        ("FATHOM_FIELD_GRAPH_V1", "Ribbon EDT", "secondary__FATHOM_FIELD_REFINED_EDT_DIAMETER"),
        ("FATHOM_FIELD_GRAPH_V1", "Ribbon Edge", "secondary__FATHOM_FIELD_REFINED_EDGE_DIAMETER"),
        ("FATHOM_FIELD_GRAPH_V1", "Ribbon Profile", "secondary__FATHOM_FIELD_REFINED_PROFILE_DIAMETER"),
    )
    rows: list[list[Any]] = []
    for method_id, label, source in estimators:
        medians: list[float] = []
        coverage_values: list[float] = []
        available = 0
        for comparison in comparisons:
            result = next((r for r in comparison.results if r.method_id.value == method_id), None)
            if result is None:
                continue
            if source == "common":
                distribution = result.common_distribution
            elif source.startswith("secondary__"):
                distribution = result.secondary_distributions.get(source[len("secondary__"):])
            else:
                distribution = None
            if distribution is None or distribution.diameter.size == 0:
                continue
            available += 1
            medians.append(float(np.median(distribution.diameter)))
            if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
                field_coverage = result.native_statistics.get("smooth_coverage_fraction") if "REFINED" in source else None
                if field_coverage is not None:
                    coverage_values.append(float(field_coverage))
        if not medians:
            rows.append([label, "0", "—", "—", "—", "—"])
            continue
        median = float(np.median(medians))
        iqr = float(np.quantile(medians, 0.75) - np.quantile(medians, 0.25))
        rows.append(
            [
                label,
                str(available),
                _fmt(median),
                _fmt(iqr),
                _fmt(float(np.median(coverage_values))) if coverage_values else "—",
                "COMPLETE" if available else "NOT_RUN",
            ]
        )
    return table(
        ["Estimator", "Images available", "Median of image medians (µm)", "IQR across image medians", "Median coverage", "Status"],
        rows,
    ) + "<p>Summary across images — not a pooled fiber distribution.</p>"


def _dataset_ribbon_section(metrics: dict[str, Any], comparisons: list[Any]) -> str:
    from .report_style import cards, note

    w1_raw = metrics["w1_edt_edge_raw"]
    w1_refined = metrics["w1_edt_edge_refined"]
    shifts_observed: list[float] = []
    shifts_residual: list[float] = []
    asymmetry_raw: list[float] = []
    asymmetry_refined: list[float] = []
    coverage: list[float] = []
    for comparison in comparisons:
        field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
        values = _per_image_estimator_medians(field)
        for key, bucket in (
            ("center_shift_median_um", shifts_observed),
            ("residual_shift_median_um", shifts_residual),
            ("asymmetry_raw", asymmetry_raw),
            ("asymmetry_refined", asymmetry_refined),
            ("smooth_coverage", coverage),
        ):
            value = values[key]
            if value is not None:
                bucket.append(float(value))
    median = lambda values: float(np.median(values)) if values else None
    improved = metrics["improved_count"]
    cards_html = cards(
        [
            (f"{improved} / {len(comparisons)}", "Images with reduced EDT↔Edge W1", "after Ribbon refinement"),
            (_fmt(median(w1_raw)) + " → " + _fmt(median(w1_refined)) + " µm", "Median W1 EDT↔Edge", "raw → Ribbon"),
            (_fmt(median(shifts_observed)) + " → " + _fmt(median(shifts_residual)) + " µm", "Median center shift", "observed → residual"),
            (_fmt(median(asymmetry_raw), 4) + " → " + _fmt(median(asymmetry_refined), 4), "Median asymmetry", "raw → refined"),
            (_fmt(median(coverage), 4), "Median Ribbon coverage", "supported centerline"),
        ]
    )
    return cards_html + note(
        "Agreement changed/improved across images; this characterizes method agreement, "
        "not known absolute accuracy on real SEM. <b>Oriented Ribbon V1 is EXPERIMENTAL</b>: "
        "known-truth synthetic validation supports the centerline mechanism."
    )


def _dataset_quality(comparisons: list[Any]) -> str:
    from .report_style import table

    rows: list[list[Any]] = []
    for comparison in comparisons:
        field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
        statistics = field.native_statistics
        stem = Path(comparison.image_id).stem
        rows.append(
            [
                stem,
                _fmt(statistics.get("edge_acceptance_fraction"), 4),
                _fmt(statistics.get("refined_edge_acceptance_fraction"), 4),
                _fmt(statistics.get("profile_acceptance_fraction"), 4),
                _fmt(statistics.get("refined_profile_acceptance_fraction"), 4),
                _fmt(statistics.get("smooth_coverage_fraction"), 4),
                ", ".join(field.quality_flags[:2]) or "—",
            ]
        )
    return table(
        ["Image", "Edge acc. raw", "Edge acc. refined", "Profile acc. raw", "Profile acc. refined", "Ribbon coverage", "Notable flags"],
        rows,
    ) + "<p>Ribbon coverage below 100% reflects intentional abstention (crossings, junctions, gaps, low-confidence regions).</p>"


def _methods_cards_dataset() -> str:
    from .report_style import cards

    entries = [
        ("MATLAB SIMPoly", "Reference/native implementation consumed from a validated cache; native Gaussian center b1 reported. Common sample distribution unavailable from the current cache.", "COMPLETE (cache)"),
        ("Python SIMPoly", "Python source-compatible approximation; calibrated length-weighted diameters on the skeleton.", "COMPLETE"),
        ("Fathom Local", "Independent local cross-section estimator; not truth and may show a broader distribution.", "COMPLETE"),
        ("Fathom Field", "Structure-tensor orientation plus local boundary metrology (EDT / paired edge / intensity profile) on the sampled centerline.", "EXPERIMENTAL"),
        ("Oriented Ribbon V1", "Experimental refined centerline from paired opposite boundaries and re-measurement of the Field estimators on supported segments.", "EXPERIMENTAL"),
        ("Manual 5×5", "Sparse human reference grid; not ground truth.", "REFERENCE"),
        ("Consensus", "Equal-method quantile pseudo-reference across participating methods; not ground truth. Field Raw/Ribbon are variants of one method and do not add independent votes.", "REFERENCE"),
    ]
    return cards([("", label, purpose + " · " + status) for label, purpose, status in entries])


def _dataset_image_sections(output: Path, comparisons: list[Any], dataset: Any, manual_store: Any) -> str:
    from .report_style import cards
    from .report_style import figure as figure_block

    body = ""
    for index, comparison in enumerate(comparisons):
        stem = Path(comparison.image_id).stem
        image_dir = output / "images" / stem
        image_dir.mkdir(parents=True, exist_ok=True)
        _final_per_image_figures(image_dir, comparison, stem)
        field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
        values = _per_image_estimator_medians(field)
        manual = manual_store.reviews.get(dataset.image_by_stem(stem).case_id) if manual_store is not None and dataset.image_by_stem(stem) is not None else None
        manual_count = manual.measurement_count if manual is not None else 0
        card_items = [
            ("Ribbon EDT", _fmt(values["edt_refined"]) + " µm", "median"),
            ("Ribbon Edge", _fmt(values["edge_refined"]) + " µm", "median"),
            ("Ribbon Profile", _fmt(values["profile_refined"]) + " µm", "median"),
            ("Coverage", _fmt(values["smooth_coverage"], 4), "supported centerline"),
            ("Manual", f"{manual_count} / 25", "measurements"),
        ]
        body += f"<section id='image-{index + 1:02d}'>"
        body += f"<h3>{stem}</h3>"
        body += cards(card_items)
        body += figure_block(f"images/{stem}/figure-A-primary-histogram.png", f"{stem} primary histogram", "Common diameter histogram, primary range.")
        body += figure_block(f"images/{stem}/figure-B-ecdf.png", f"{stem} ECDF", "Diameter ECDF, full observed range.")
        body += figure_block(f"images/{stem}/figure-C-field-ribbon.png", f"{stem} field raw vs Ribbon", "Field estimators raw vs Oriented Ribbon V1.")
        body += "</section>"
    return body


def _calibration_audit(dataset: Any, comparisons: list[Any]) -> list[dict[str, Any]]:
    """Read-only per-image calibration/ROI audit from TIFF metadata + caches."""
    from .api import FathomEngine

    engine = FathomEngine()
    audit: list[dict[str, Any]] = []
    for image in dataset.images:
        entry: dict[str, Any] = {
            "image": image.stem,
            "case_id": image.case_id,
            "shape": None,
            "footer": None,
            "body_rows": None,
            "roi": None,
            "pixel_size_x_nm": None,
            "pixel_size_y_nm": None,
            "source": None,
            "isotropic": None,
            "calibrated": False,
        }
        try:
            doc = engine.open_image(image.absolute_path)
            entry["shape"] = doc.shape
            entry["footer"] = doc.footer_bounds
            entry["body_rows"] = doc.footer_bounds[0] if doc.footer_bounds else doc.shape[0]
            entry["pixel_size_x_nm"] = float(doc.calibration.pixel_size_x_m) * 1e9
            entry["pixel_size_y_nm"] = float(doc.calibration.pixel_size_y_m) * 1e9
            entry["source"] = doc.calibration.source
            entry["calibrated"] = True
            entry["isotropic"] = (
                abs(entry["pixel_size_x_nm"] - entry["pixel_size_y_nm"]) < 1e-6
            )
        except Exception:
            entry["calibrated"] = False
        comparison = next((c for c in comparisons if Path(c.image_id).stem == image.stem), None)
        if comparison is not None:
            field = next(
                (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1),
                None,
            )
            if field is not None:
                entry["roi"] = field.valid_roi
        audit.append(entry)
    return audit


def _calibration_short(audit: list[dict[str, Any]]) -> str:
    sizes = [entry["pixel_size_x_nm"] for entry in audit if entry["calibrated"]]
    if not sizes:
        return "unavailable"
    low, high = min(sizes), max(sizes)
    if high - low < 1e-6:
        return f"{low:.4g} nm/px · {len(sizes)} / {len(audit)}"
    return f"mixed · {low:.4g}–{high:.4g} nm/px · {len(sizes)} / {len(audit)}"


def _calibration_audit_summary(audit: list[dict[str, Any]]) -> str:
    from .report_style import details, table

    calibrated = [entry for entry in audit if entry["calibrated"]]
    sizes = [entry["pixel_size_x_nm"] for entry in calibrated]
    if not sizes:
        return "<h3>Calibration</h3><p>Calibration metadata not available for any image.</p>"
    low, high = min(sizes), max(sizes)
    if high - low < 1e-6:
        summary = f"Calibration: {low:.4g} nm/px · {len(calibrated)} / {len(audit)} images"
    else:
        summary = (
            f"Calibration: mixed · {low:.4g}–{high:.4g} nm/px · "
            f"{len(calibrated)} / {len(audit)} calibrated"
        )
    rows = [
        [
            entry["image"],
            _fmt(entry["pixel_size_x_nm"]),
            _fmt(entry["pixel_size_y_nm"]),
            str(entry["roi"]),
            entry["source"] or "—",
            "ok" if entry["calibrated"] else "missing",
            "isotropic" if entry["isotropic"] else ("anisotropic" if entry["calibrated"] else "—"),
        ]
        for entry in audit
    ]
    return (
        f"<h3>Calibration</h3><p>{summary}</p>"
        + details(
            "Per-image calibration and ROI audit (technical)",
            table(["Image", "px (nm)", "py (nm)", "Science ROI", "Source", "Status", "Symmetry"], rows),
        )
    )


def _dataset_mixture(comparisons: list[Any], estimator_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Equal-image-weight mixture: per image normalize weights to unit mass, concat.

    Images with the estimator missing contribute nothing; available images
    contribute exactly equal total weight.
    """
    diameters: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for comparison in comparisons:
        values = _mixture_values(comparison, estimator_key)
        if values is None:
            continue
        diameter, weight = values
        total = float(weight.sum())
        if total <= 0 or diameter.size == 0:
            continue
        diameters.append(np.asarray(diameter, float))
        weights.append(np.asarray(weight, float) / total)
    if not diameters:
        return np.array([]), np.array([])
    return np.concatenate(diameters), np.concatenate(weights)


def _mixture_values(comparison: Any, estimator_key: str) -> tuple[np.ndarray, np.ndarray] | None:
    field = next(
        (r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None
    )
    if estimator_key == "PYTHON_SIMPOLY":
        result = next((r for r in comparison.results if r.method_id == MethodId.PYTHON_SIMPOLY), None)
        distribution = result.common_distribution if result is not None else None
        if distribution is None:
            return None
        return distribution.diameter, distribution.weight
    if estimator_key == "FATHOM_LOCAL":
        result = next((r for r in comparison.results if r.method_id == MethodId.FATHOM_LOCAL), None)
        distribution = result.common_distribution if result is not None else None
        if distribution is None:
            return None
        return distribution.diameter, distribution.weight
    if field is None:
        return None
    if estimator_key == "RAW_EDT":
        distribution = field.common_distribution
        return (distribution.diameter, distribution.weight) if distribution is not None else None
    if estimator_key == "RIBBON_EDT":
        distribution = field.secondary_distributions.get("FATHOM_FIELD_REFINED_EDT_DIAMETER")
        return (distribution.diameter, distribution.weight) if distribution is not None else None
    if estimator_key == "RAW_EDGE":
        distribution = field.secondary_distributions.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER")
        return (distribution.diameter, distribution.weight) if distribution is not None else None
    if estimator_key == "RIBBON_EDGE":
        distribution = field.secondary_distributions.get("FATHOM_FIELD_REFINED_EDGE_DIAMETER")
        return (distribution.diameter, distribution.weight) if distribution is not None else None
    return None


def _mixture_figure(
    comparisons: list[Any],
    output: Path,
    *,
    primary_keys: list[str],
    primary_labels: list[str],
    include_local: bool,
    filename: str,
    title: str,
    x_max: float | None,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    keys = list(primary_keys) + (["FATHOM_LOCAL"] if include_local else [])
    labels = list(primary_labels) + (["Fathom Local"] if include_local else [])
    distributions = []
    for key in keys:
        diameter, weight = _dataset_mixture(comparisons, key)
        if diameter.size:
            distributions.append(
                DiameterDistribution(diameter, weight, "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.CONSENSUS_PSEUDO_REFERENCE_V1)
            )
    edges = common_histogram_edges(distributions) if distributions else np.array([])
    figure = Figure(figsize=(9.6, 4.4), dpi=130)
    axis = figure.add_subplot(111)
    if not edges.size:
        axis.text(0.5, 0.5, "No dataset distributions available", ha="center")
    else:
        centers = edges[:-1] + np.diff(edges) / 2.0
        for key, label in zip(keys, labels, strict=False):
            diameter, weight = _dataset_mixture(comparisons, key)
            if diameter.size == 0:
                continue
            distribution = DiameterDistribution(diameter, weight, "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.CONSENSUS_PSEUDO_REFERENCE_V1)
            color, style, _short = _mixture_style(key)
            axis.plot(
                centers, _weighted_density(distribution, edges),
                label=label, color=color, linestyle=style, linewidth=1.9, drawstyle="steps-mid",
            )
    if x_max is not None and edges.size:
        axis.set_xlim(0, x_max)
    axis.set_xlabel("Diameter (µm)")
    axis.set_ylabel("Equal-image weighted density (1/µm)")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    axis.legend(fontsize="small", ncol=2, loc="upper right")
    figure.tight_layout()
    FigureCanvasAgg(figure).print_figure(output / filename, dpi=130)
    figure.clear()


def _mixture_style(key: str) -> tuple[str, str, str]:
    styles = {
        "PYTHON_SIMPOLY": ("#e69f00", "solid"),
        "FATHOM_LOCAL": ("#009e73", "dashed"),
        "RAW_EDT": ("#9aa7c9", "dashed"),
        "RIBBON_EDT": ("#4d648d", "solid"),
        "RAW_EDGE": ("#9cc4c4", "dashed"),
        "RIBBON_EDGE": ("#3b7f7f", "solid"),
        "RAW_PROFILE": ("#c2a8cc", "dashed"),
        "RIBBON_PROFILE": ("#76538a", "solid"),
    }
    color, style = styles.get(key, ("#777777", "solid"))
    return color, style, key


def _dataset_distribution_section(comparisons: list[Any], output: Path) -> str:
    from .report_style import figure as figure_block
    from .report_style import section, table

    primary = [
        ("PYTHON_SIMPOLY", "Python SIMPoly"),
        ("RAW_EDT", "Fathom Field — Raw EDT"),
        ("RIBBON_EDT", "Oriented Ribbon V1 — EDT"),
        ("RAW_EDGE", "Fathom Field — Raw Edge"),
        ("RIBBON_EDGE", "Oriented Ribbon V1 — Edge"),
    ]
    primary_names = [label for _key, label in primary]

    # data-driven primary x-range from the mixture P99 of the primary estimators
    p99s = []
    for key, _label in primary:
        diameter, weight = _dataset_mixture(comparisons, key)
        if diameter.size:
            from .core.distributions import weighted_quantile

            p99s.append(float(weighted_quantile(diameter, weight, np.array([0.99]))[0]))
    x_max = float(np.max(p99s)) * 1.12 if p99s else None

    _mixture_figure(comparisons, output, primary_keys=[k for k, _ in primary], primary_labels=primary_names,
                    include_local=False, filename="dataset-distribution-primary.png",
                    title="Dataset-wide diameter distribution — primary range (equal image weight)",
                    x_max=x_max)
    _mixture_figure(comparisons, output, primary_keys=[k for k, _ in primary], primary_labels=primary_names,
                    include_local=True, filename="dataset-distribution-full.png",
                    title="Dataset-wide diameter distribution — full observed range (equal image weight)",
                    x_max=None)

    rows = []
    for key, label in primary + [("FATHOM_LOCAL", "Fathom Local")]:
        diameter, weight = _dataset_mixture(comparisons, key)
        if diameter.size == 0:
            rows.append([label, "0", "—", "—", "—", "—", "—"])
            continue
        from .core.distributions import weighted_quantile

        q = weighted_quantile(diameter, weight, np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
        images = sum(1 for c in comparisons if _mixture_values(c, key) is not None)
        rows.append(
            [
                label,
                str(images),
                _fmt(float(q[2])),
                _fmt_range(float(q[1]), float(q[3])),
                _fmt(float(q[0])),
                _fmt(float(q[4])),
                "long tail shown in full range" if key == "FATHOM_LOCAL" else "—",
            ]
        )
    body = (
        "<p>Each image contributes equal total weight: per-image weighted "
        "distributions are normalized to unit mass and then averaged. The local "
        "sample count of an image does not determine its influence.</p>"
        "<p>This is a descriptive dataset-level mixture across SEM fields, not an "
        "inferential pooled population model.</p>"
        + figure_block(
            "dataset-distribution-primary.png",
            "Dataset-wide diameter distribution, primary range",
            "Equal-image-weight mixture of the comparable automatic estimators. "
            "Calibration is mixed across the dataset (see Dataset overview); values are physical µm.",
        )
        + figure_block(
            "dataset-distribution-full.png",
            "Dataset-wide diameter distribution, full range",
            "Full observed range including Fathom Local; long tails remain available and "
            "no observations are removed.",
        )
        + table(
            ["Estimator", "Images", "Mixture median", "IQR", "P05", "P95", "Notes"],
            rows,
        )
        + "<p>If multiple modes appear, they may reflect real morphological populations, "
        "bundles/merged structures, spatial heterogeneity, acquisition differences, or "
        "measurement behavior. The current report does not assign a cause.</p>"
    )
    return section("3. Dataset-wide diameter distribution", body, id_="dataset-distribution")


def _dataset_limitations() -> str:
    from .report_style import info

    items = [
        "Measurements represent projected 2-D geometry; no 3-D claim is made.",
        "Agreement metrics and the consensus pseudo-reference are not ground truth.",
        "Manual 5×5 is a sparse human reference; missing measurements are never filled in.",
        "MATLAB SIMPoly reports the native b1 statistic; a common raw distribution is unavailable from the current cache.",
        "Oriented Ribbon V1 is EXPERIMENTAL; real SEM comparisons characterize behavior and agreement, not known absolute accuracy.",
        "Unsupported or crossing regions may be intentionally abstained rather than measured.",
        "Fathom Local may show broad or right-tailed candidates depending on image geometry; quality flags provide context.",
    ]
    return info("<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>")


def _distribution_median(distribution: Any) -> float | None:
    if distribution is None:
        return None
    return float(np.median(distribution.diameter))


def _final_per_image_figures(output: Path, comparison: Any, stem: str) -> None:
    """Per-image figures: primary histogram, ECDF and raw-vs-Ribbon panels."""
    series = series_distributions(comparison)
    if not series:
        return
    x_max = _primary_x_max(series)
    figure_primary_histogram(series, output, x_max)
    figure_ecdf_full(series, output)
    figure_field_raw_vs_ribbon(comparison, output)
    # normalize file names for the image sections
    for old, new in (
        ("figure-primary-histogram.png", "figure-A-primary-histogram.png"),
        ("figure-full-ecdf.png", "figure-B-ecdf.png"),
        ("figure-field-raw-vs-ribbon.png", "figure-C-field-ribbon.png"),
    ):
        source = output / old
        if source.exists():
            source.replace(output / new)


def _final_manual_section(manual_store: Any, dataset: Any) -> str:
    from .report_style import note

    if manual_store is None:
        return "<p>Manual 5×5 store not loaded.</p>"
    total = manual_store.total_measured
    rows = ""
    for image in dataset.images:
        review = manual_store.reviews.get(image.case_id)
        count = review.measurement_count if review is not None else 0
        rows += f"<tr><td>{html.escape(image.filename)}</td><td>{count} / 25</td></tr>"
    status = "COMPLETE REFERENCE" if total >= 400 else "INCOMPLETE REFERENCE"
    return (
        f"<p>Measurements recorded: <b>{total} / 400</b> — {status}.</p>"
        "<table><tr><th>Image</th><th>Progress</th></tr>" + rows + "</table>"
    ) + note(
        "The report is generated regardless of manual completeness; missing manual "
        "values are never filled in. Manual 5×5 is a sparse human reference, not ground truth."
    )


def _final_ribbon_box(metrics: dict[str, Any]) -> str:
    import statistics

    w1_raw = metrics["w1_edt_edge_raw"]
    w1_refined = metrics["w1_edt_edge_refined"]
    if not w1_raw:
        return "<div class='info'><b>Oriented Ribbon V1:</b> no comparable distributions.</div>"
    return (
        f"<div class='info'><b>Oriented Ribbon V1 (experimental):</b> "
        f"{metrics['improved_count']}/{metrics['total']} images show "
        "W1(refined EDT ↔ refined Edge) &lt; W1(raw EDT ↔ raw Edge); "
        f"dataset median W1 {statistics.median(w1_raw):.3f} → {statistics.median(w1_refined):.3f} µm.</div>"
    )


def _final_provenance(
    dataset: Any,
    repo: Path,
    cache: WorkspaceCache,
    calibration_audit: list[dict[str, Any]] | None = None,
) -> str:
    from .report_style import details, table

    try:
        import subprocess

        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip()[:12]
    except Exception:
        commit = "unknown"
    rows = [
        ["Application", "Fathom Fibers"],
        ["Commit", commit],
        ["Cache schema", cache.FULL_SCHEMA],
        ["Dataset", str(dataset.dataset_id)],
        ["MATLAB cache", "validated controlled-origin cache; native b1 only"],
    ]
    hashes = {
        image.case_id: image.sha256[:12] if image.sha256 else "—"
        for image in dataset.images
    }
    for case_id, short_hash in list(hashes.items())[:4]:
        rows.append([f"Image hash {case_id}", short_hash])
    calibration_rows = ""
    if calibration_audit:
        calibration_rows = "<h3>Calibration and ROI audit</h3>" + table(
            ["Image", "TIFF shape", "Science ROI", "px (nm)", "py (nm)", "Source", "Symmetry"],
            [
                [
                    entry["image"],
                    str(entry["shape"]),
                    str(entry["roi"]),
                    _fmt(entry["pixel_size_x_nm"]),
                    _fmt(entry["pixel_size_y_nm"]),
                    entry["source"] or "—",
                    "isotropic" if entry["isotropic"] else ("anisotropic" if entry["calibrated"] else "missing"),
                ]
                for entry in calibration_audit
            ],
        )
    return details("Provenance / Reproducibility", table(["Item", "Value"], rows) + calibration_rows)
