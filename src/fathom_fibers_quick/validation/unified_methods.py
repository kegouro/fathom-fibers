"""Cached 16-image unified-method campaign and headless HTML reporting."""

from __future__ import annotations

import html
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import stats

from ..api import FathomEngine
from ..core.distributions import summarize_distribution
from ..core.methods import MethodId, MethodResult
from ..unified_comparison import (
    DatasetMorphologyReport,
    ImageMorphologyReport,
    build_image_report,
)
from .real_campaign import DATASET_ID, EXPECTED_CASES


def _root(repo: Path) -> Path:
    return repo / ".validation/unified-method-comparison"


def _jsonable_result(result: MethodResult) -> dict[str, Any]:
    distribution = result.common_distribution
    summary = summarize_distribution(distribution) if distribution else None
    return {
        "method_id": result.method_id.value,
        "method_version": result.method_version,
        "status": result.status.value,
        "native_estimand": result.native_estimand.value if result.native_estimand else None,
        "native_result": result.native_result,
        "native_statistics": dict(result.native_statistics),
        "capabilities": result.capabilities.to_dict(),
        "quality_flags": list(result.quality_flags),
        "confidence": result.confidence,
        "runtime_seconds": result.runtime_seconds,
        "provenance": dict(result.provenance),
        "common_distribution": None if summary is None else {
            "unit": distribution.unit,
            "estimand": distribution.estimand.value,
            "n": summary.n,
            "weight_sum": summary.weight_sum,
            "mean": summary.weighted_mean,
            "median": summary.weighted_median,
            "p05": summary.p05,
            "p25": summary.p25,
            "p50": summary.p50,
            "p75": summary.p75,
            "p95": summary.p95,
        },
        "secondary_distributions": {
            name: {
                "unit": distribution.unit,
                "estimand": distribution.estimand.value,
                "n": item_summary.n,
                "weight_sum": item_summary.weight_sum,
                "mean": item_summary.weighted_mean,
                "median": item_summary.weighted_median,
                "p05": item_summary.p05,
                "p25": item_summary.p25,
                "p50": item_summary.p50,
                "p75": item_summary.p75,
                "p95": item_summary.p95,
            }
            for name, distribution in result.secondary_distributions.items()
            for item_summary in (summarize_distribution(distribution),)
        },
    }


def _comparison_payload(report: ImageMorphologyReport) -> dict[str, Any]:
    comparison = report.comparison
    field = next((result for result in comparison.results if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1), None)
    edge = field.secondary_distributions.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER") if field else None
    profile = field.secondary_distributions.get("FATHOM_FIELD_PROFILE_DIAMETER") if field else None
    edge_agreements: dict[str, dict[str, float | None]] = {}
    profile_agreements: dict[str, dict[str, float | None]] = {}
    if edge is not None:
        for other in comparison.results:
            distribution = other.common_distribution
            if distribution is None or not distribution.diameter.size:
                continue
            edge_summary = summarize_distribution(edge)
            other_summary = summarize_distribution(distribution)
            edge_agreements[other.method_id.value] = {
                "wasserstein_1": float(stats.wasserstein_distance(edge.diameter, distribution.diameter, edge.weight, distribution.weight)),
                "median_difference": edge_summary.weighted_median - other_summary.weighted_median,
            }
    if profile is not None:
        for other in comparison.results:
            distribution = other.common_distribution
            if distribution is None or not distribution.diameter.size:
                continue
            profile_summary = summarize_distribution(profile)
            other_summary = summarize_distribution(distribution)
            profile_agreements[other.method_id.value] = {
                "wasserstein_1": float(stats.wasserstein_distance(profile.diameter, distribution.diameter, profile.weight, distribution.weight)),
                "median_difference": profile_summary.weighted_median - other_summary.weighted_median,
            }
        if edge is not None:
            edge_summary = summarize_distribution(edge)
            profile_summary = summarize_distribution(profile)
            profile_agreements["FATHOM_FIELD_PAIRED_EDGE_DIAMETER"] = {
                "wasserstein_1": float(stats.wasserstein_distance(profile.diameter, edge.diameter, profile.weight, edge.weight)),
                "median_difference": profile_summary.weighted_median - edge_summary.weighted_median,
            }
    return {
        "image_id": report.image_id,
        "results": [_jsonable_result(result) for result in comparison.results],
        "agreements": [asdict(item) | {"left_method": item.left_method.value, "right_method": item.right_method.value, "estimand": item.estimand.value} for item in comparison.agreements],
        "consensus": {
            "participating_methods": [item.value for item in comparison.consensus.participating_methods],
            "excluded_methods": comparison.consensus.excluded_methods,
            "quantile_grid": comparison.consensus.quantile_grid.tolist(),
            "quantiles": comparison.consensus.quantiles.tolist(),
            "disagreement_mad": comparison.consensus.disagreement_mad.tolist(),
        },
        "limitations": list(report.limitations),
        "field_edge_agreements": edge_agreements,
        "field_profile_agreements": profile_agreements,
    }


def run_unified_campaign(
    repo: Path,
    *,
    dataset: Path,
    matlab_cache_root: Path | None = None,
    resume: bool = False,
    case: str | None = None,
) -> DatasetMorphologyReport:
    paths = sorted([*dataset.glob("*.tif"), *dataset.glob("*.tiff")], key=lambda item: item.name)
    if len(paths) != EXPECTED_CASES:
        raise RuntimeError(f"Expected exactly {EXPECTED_CASES} TIFF files, found {len(paths)}")
    if case is not None:
        paths = [path for path in paths if path.name == case or path.stem == case]
        if len(paths) != 1:
            raise RuntimeError(f"Expected one canonical TIFF matching {case!r}, found {len(paths)}")
    output = _root(repo)
    runs = output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    engine = FathomEngine()
    reports: list[ImageMorphologyReport] = []
    failures: dict[str, str] = {}
    for path in paths:
        cached = runs / f"{path.stem}.json"
        if resume and cached.exists():
            # The cache intentionally skips recomputation but remains represented
            # in the HTML as a cache record, not silently dropped.
            continue
        try:
            image = engine.open_image(path)
            comparison = engine.compare_all_methods(image, matlab_cache_root=matlab_cache_root)
            report = build_image_report(comparison)
            cached.write_text(json.dumps(_comparison_payload(report), indent=2, default=str), encoding="utf-8")
            reports.append(report)
        except Exception as exc:  # campaign reports failures rather than dropping images
            failures[path.name] = str(exc)
            cached.write_text(json.dumps({"image_id": path.name, "status": "FAILED", "error": str(exc)}, indent=2), encoding="utf-8")
    dataset_report = DatasetMorphologyReport(
        DATASET_ID,
        tuple(reports),
        failures,
        (
            "Each image remains a statistical unit in dataset summaries.",
            "No pooled skeleton-pixel estimate is reported as a dataset result.",
            "Agreement and consensus pseudo-reference are not ground truth.",
        ),
    )
    (output / "dataset_report.json").write_text(
        json.dumps({
            "dataset_id": DATASET_ID,
            "processed": len(reports),
            "failed": failures,
            "matlab_cache_root": str(matlab_cache_root) if matlab_cache_root else None,
            "image_reports": [_comparison_payload(item) for item in reports],
            "limitations": list(dataset_report.limitations),
        }, indent=2, default=str),
        encoding="utf-8",
    )
    return dataset_report


def _load_cached_payloads(repo: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((_root(repo) / "runs").glob("*.json"))]


def _image_balanced_samples(
    payloads: list[dict[str, Any]], method_id: str, secondary_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    # V1 plot aggregate: each image contributes total weight one, avoiding
    # domination by a skeleton-rich image. This is display-only, not a new estimand.
    values: list[float] = []
    weights: list[float] = []
    for payload in payloads:
        result = next((entry for entry in payload.get("results", []) if entry["method_id"] == method_id), None)
        distribution = (
            result and result.get("common_distribution") if secondary_name is None
            else result and result.get("secondary_distributions", {}).get(secondary_name)
        )
        if not distribution or not distribution.get("n"):
            continue
        # Per-image quantile summaries are all that the on-disk report retains.
        # Use five reported quantiles with equal total image weight for an honest
        # cross-image display rather than reconstructing all local samples.
        quantiles = [distribution[key] for key in ("p05", "p25", "p50", "p75", "p95")]
        if all(value is not None for value in quantiles):
            values.extend(quantiles)
            weights.extend([1 / len(quantiles)] * len(quantiles))
    return np.asarray(values), np.asarray(weights)


def generate_unified_report(repo: Path) -> Path:
    payloads = _load_cached_payloads(repo)
    output = _root(repo) / "latest"
    output.mkdir(parents=True, exist_ok=True)
    method_ids = [
        MethodId.MATLAB_SIMPOLY.value,
        MethodId.PYTHON_SIMPOLY.value,
        MethodId.FATHOM_LOCAL.value,
        MethodId.FATHOM_FIELD_GRAPH_V1.value,
    ]
    colors = {
        MethodId.MATLAB_SIMPOLY.value: "#386cb0",
        MethodId.PYTHON_SIMPOLY.value: "#fdb462",
        MethodId.FATHOM_LOCAL.value: "#7fc97f",
        MethodId.FATHOM_FIELD_GRAPH_V1.value: "#8c6bb1",
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    method_rows = []
    for method_id in method_ids:
        x, w = _image_balanced_samples(payloads, method_id)
        if x.size:
            order = np.argsort(x)
            axis.step(x[order], np.cumsum(w[order]) / w.sum(), where="post", label=method_id, color=colors[method_id])
            method_rows.append((method_id, len(x) // 5, float(np.median(x)), float(np.quantile(x, .25)), float(np.quantile(x, .75)), float(np.quantile(x, .05)), float(np.quantile(x, .95))))
        else:
            method_rows.append((method_id, 0, None, None, None, None, None))
    axis.set(xlabel="Diameter (µm)", ylabel="Image-balanced ECDF", title="Common length-weighted diameter display")
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    plot = output / "ecdf.png"
    figure.savefig(plot, dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, color in (("FATHOM_FIELD_PAIRED_EDGE_DIAMETER", "#d95f02"), ("FATHOM_FIELD_PROFILE_DIAMETER", "#1b9e77")):
        x, w = _image_balanced_samples(payloads, MethodId.FATHOM_FIELD_GRAPH_V1.value, name)
        if x.size:
            order = np.argsort(x)
            axis.step(x[order], np.cumsum(w[order]) / w.sum(), where="post", label=name, color=color)
    axis.set(xlabel="Diameter (µm)", ylabel="Image-balanced ECDF", title="Field secondary estimators")
    axis.legend()
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output / "field-secondary-ecdf.png", dpi=150)
    plt.close(figure)

    wasserstein_matrix: dict[tuple[str, str], list[float]] = {}
    median_difference_matrix: dict[tuple[str, str], list[float]] = {}
    for payload in payloads:
        for agreement in payload.get("agreements", []):
            if agreement.get("wasserstein_1") is not None:
                key = (agreement["left_method"], agreement["right_method"])
                wasserstein_matrix.setdefault(key, []).append(agreement["wasserstein_1"])
            if agreement.get("median_difference") is not None:
                key = (agreement["left_method"], agreement["right_method"])
                median_difference_matrix.setdefault(key, []).append(agreement["median_difference"])
    wasserstein_rows = "".join(
        f"<tr><td>{html.escape(left)}</td><td>{html.escape(right)}</td><td>{np.median(values):.6g}</td><td>{len(values)}</td></tr>"
        for (left, right), values in sorted(wasserstein_matrix.items())
    ) or "<tr><td colspan='4'>No comparable pairwise distributions.</td></tr>"
    median_difference_rows = "".join(
        f"<tr><td>{html.escape(left)}</td><td>{html.escape(right)}</td><td>{np.median(values):.6g}</td><td>{len(values)}</td></tr>"
        for (left, right), values in sorted(median_difference_matrix.items())
    ) or "<tr><td colspan='4'>No comparable pairwise distributions.</td></tr>"
    summary_rows = "".join(
        f"<tr><td>{html.escape(method)}</td><td>{count}/16</td><td>{'—' if median is None else f'{median:.6g}'}</td><td>{'—' if q25 is None else f'{q25:.6g}–{q75:.6g}'}</td><td>{'—' if p05 is None else f'{p05:.6g} / {p95:.6g}'}</td></tr>"
        for method, count, median, q25, q75, p05, p95 in method_rows
    )
    image_rows = []
    field_rows = []
    edge_rows = []
    profile_rows = []
    for payload in payloads:
        statuses = ", ".join(f"{entry['method_id']}: {entry['status']}" for entry in payload.get("results", []))
        consensus = payload.get("consensus", {})
        image_rows.append(f"<tr><td>{html.escape(payload.get('image_id', 'unknown'))}</td><td>{html.escape(statuses)}</td><td>{html.escape(', '.join(consensus.get('participating_methods', [])) or '—')}</td></tr>")
        field = next((entry for entry in payload.get("results", []) if entry["method_id"] == MethodId.FATHOM_FIELD_GRAPH_V1.value), None)
        stats = field.get("native_statistics", {}) if field else {}
        edge = (field.get("secondary_distributions") or {}).get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER") if field else None
        profile = (field.get("secondary_distributions") or {}).get("FATHOM_FIELD_PROFILE_DIAMETER") if field else None
        coherence = stats.get("mean_coherence")
        nematic = stats.get("nematic_order_parameter")
        coherence_text = "—" if coherence is None else f"{coherence:.4g}"
        nematic_text = "—" if nematic is None else f"{nematic:.4g}"
        acceptance = stats.get("edge_acceptance_fraction")
        asymmetry = stats.get("edge_median_asymmetry")
        acceptance_text = "—" if acceptance is None else f"{acceptance:.2%}"
        asymmetry_text = "—" if asymmetry is None else f"{asymmetry:.4g}"
        edge_median = "—" if edge is None else f"{edge['median']:.6g}"
        edge_edt = payload.get("field_edge_agreements", {}).get(MethodId.FATHOM_FIELD_GRAPH_V1.value)
        edge_edt_text = "—" if edge_edt is None else f"{edge_edt['wasserstein_1']:.6g}"
        field_rows.append(
            f"<tr><td>{html.escape(payload.get('image_id', 'unknown'))}</td><td>{html.escape(field.get('status', 'NOT_RUN') if field else 'NOT_RUN')}</td>"
            f"<td>{stats.get('sample_count', '—')}</td><td>{coherence_text}</td>"
            f"<td>{nematic_text}</td>"
            f"<td>{html.escape(str(stats.get('centerline_source', '—')))}</td></tr>"
        )
        edge_rows.append(
            f"<tr><td>{html.escape(payload.get('image_id', 'unknown'))}</td><td>{stats.get('edge_raw_count', '—')}</td>"
            f"<td>{stats.get('edge_accepted_count', '—')}</td><td>{acceptance_text}</td>"
            f"<td>{asymmetry_text}</td><td>{edge_median}</td><td>{edge_edt_text}</td></tr>"
        )
        profile_agreement = payload.get("field_profile_agreements", {})
        profile_acceptance = stats.get("profile_acceptance_fraction")
        profile_edge_shift = stats.get("profile_median_abs_edge_shift_um")
        profile_center_shift = stats.get("profile_median_center_shift_um")
        profile_acceptance_text = "—" if profile_acceptance is None else f"{profile_acceptance:.2%}"
        profile_edge_shift_text = "—" if profile_edge_shift is None else f"{profile_edge_shift:.4g}"
        profile_center_shift_text = "—" if profile_center_shift is None else f"{profile_center_shift:.4g}"
        profile_median_text = "—" if profile is None else f"{profile['median']:.6g}"
        profile_edge = profile_agreement.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER")
        profile_edge_text = "—" if profile_edge is None else f"{profile_edge['wasserstein_1']:.6g}"
        profile_rows.append(
            f"<tr><td>{html.escape(payload.get('image_id', 'unknown'))}</td><td>{stats.get('profile_accepted_count', '—')}</td>"
            f"<td>{profile_acceptance_text}</td><td>{profile_median_text}</td>"
            f"<td>{profile_edge_shift_text}</td><td>{profile_center_shift_text}</td><td>{profile_edge_text}</td></tr>"
        )
    index = output / "index.html"
    index.write_text(f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Unified Method Comparison</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.4rem;text-align:left}}th{{background:#eef1f4}}.note{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>Unified Method Comparison — {DATASET_ID}</h1><p class='note'>Agreement is not truth. Consensus, where available, is an equal-method quantile pseudo-reference only.</p>
<p>Images represented: {len(payloads)} / {EXPECTED_CASES}. MATLAB is consumed from a validated cache; Python SIMPoly retains its documented <code>bwskel</code> divergence.</p>
<h2>Image-balanced common distribution summary (µm)</h2><table><tr><th>Method</th><th>Images</th><th>Median</th><th>IQR</th><th>P05 / P95</th></tr>{summary_rows}</table>
<img src='ecdf.png' alt='ECDF comparison'><h2>Pairwise distribution distance</h2><table><tr><th>Left</th><th>Right</th><th>Median Wasserstein-1 (µm)</th><th>Images</th></tr>{wasserstein_rows}</table>
<h2>Pairwise median-method difference</h2><table><tr><th>Left</th><th>Right</th><th>Median difference (µm)</th><th>Images</th></tr>{median_difference_rows}</table>
<h2>16-image processing matrix</h2><table><tr><th>Image</th><th>Method states</th><th>Consensus participants</th></tr>{''.join(image_rows)}</table>
<h2>Field diagnostics</h2><table><tr><th>Image</th><th>Status</th><th>N samples</th><th>Mean coherence</th><th>Nematic S</th><th>Centerline source</th></tr>{''.join(field_rows)}</table>
<h2>Paired-edge diagnostics (experimental)</h2><table><tr><th>Image</th><th>N raw</th><th>N accepted</th><th>Acceptance</th><th>Median asymmetry</th><th>Edge median µm</th><th>W1 Edge ↔ EDT µm</th></tr>{''.join(edge_rows)}</table>
<h2>Intensity-profile diagnostics (experimental)</h2><table><tr><th>Image</th><th>N accepted</th><th>Acceptance</th><th>Profile median µm</th><th>Median |edge shift| µm</th><th>Median suggested center shift µm</th><th>W1 Profile ↔ Edge µm</th></tr>{''.join(profile_rows)}</table>
<img src='field-secondary-ecdf.png' alt='Edge and profile ECDF comparison'>
<h2>Method limitations</h2><ul><li>Measurements represent projected 2-D geometry.</li><li>FATHOM_FIELD_GRAPH_V1 implements a field stage only: orientation and mask-derived EDT diameters are sampled on an explicit skeleton baseline. Graph, topology, crossing resolution and fibre instances are unavailable.</li><li>Manual 5×5 remains <code>NOT_MEASURED</code> until actual accepted records exist.</li><li>Dataset display balances images; it does not blindly pool skeleton samples.</li></ul>
</body></html>""", encoding="utf-8")
    return index
