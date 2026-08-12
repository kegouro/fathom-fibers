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
    }


def _comparison_payload(report: ImageMorphologyReport) -> dict[str, Any]:
    comparison = report.comparison
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
    }


def run_unified_campaign(
    repo: Path,
    *,
    dataset: Path,
    matlab_cache_root: Path | None = None,
    resume: bool = False,
) -> DatasetMorphologyReport:
    paths = sorted([*dataset.glob("*.tif"), *dataset.glob("*.tiff")], key=lambda item: item.name)
    if len(paths) != EXPECTED_CASES:
        raise RuntimeError(f"Expected exactly {EXPECTED_CASES} TIFF files, found {len(paths)}")
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


def _image_balanced_samples(payloads: list[dict[str, Any]], method_id: str) -> tuple[np.ndarray, np.ndarray]:
    # V1 plot aggregate: each image contributes total weight one, avoiding
    # domination by a skeleton-rich image. This is display-only, not a new estimand.
    values: list[float] = []
    weights: list[float] = []
    for payload in payloads:
        result = next((entry for entry in payload.get("results", []) if entry["method_id"] == method_id), None)
        distribution = result and result.get("common_distribution")
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
    method_ids = [MethodId.MATLAB_SIMPOLY.value, MethodId.PYTHON_SIMPOLY.value, MethodId.FATHOM_LOCAL.value]
    colors = {MethodId.MATLAB_SIMPOLY.value: "#386cb0", MethodId.PYTHON_SIMPOLY.value: "#fdb462", MethodId.FATHOM_LOCAL.value: "#7fc97f"}
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
    for payload in payloads:
        statuses = ", ".join(f"{entry['method_id']}: {entry['status']}" for entry in payload.get("results", []))
        consensus = payload.get("consensus", {})
        image_rows.append(f"<tr><td>{html.escape(payload.get('image_id', 'unknown'))}</td><td>{html.escape(statuses)}</td><td>{html.escape(', '.join(consensus.get('participating_methods', [])) or '—')}</td></tr>")
    index = output / "index.html"
    index.write_text(f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Unified Method Comparison</title>
<style>body{{font:14px system-ui,sans-serif;margin:2rem;color:#20242a}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd2d8;padding:.4rem;text-align:left}}th{{background:#eef1f4}}.note{{border-left:4px solid #b7791f;background:#fff9e6;padding:.7rem}}</style></head><body>
<h1>Unified Method Comparison — {DATASET_ID}</h1><p class='note'>Agreement is not truth. Consensus, where available, is an equal-method quantile pseudo-reference only.</p>
<p>Images represented: {len(payloads)} / {EXPECTED_CASES}. MATLAB is consumed from a validated cache; Python SIMPoly retains its documented <code>bwskel</code> divergence.</p>
<h2>Image-balanced common distribution summary (µm)</h2><table><tr><th>Method</th><th>Images</th><th>Median</th><th>IQR</th><th>P05 / P95</th></tr>{summary_rows}</table>
<img src='ecdf.png' alt='ECDF comparison'><h2>Pairwise distribution distance</h2><table><tr><th>Left</th><th>Right</th><th>Median Wasserstein-1 (µm)</th><th>Images</th></tr>{wasserstein_rows}</table>
<h2>Pairwise median-method difference</h2><table><tr><th>Left</th><th>Right</th><th>Median difference (µm)</th><th>Images</th></tr>{median_difference_rows}</table>
<h2>16-image processing matrix</h2><table><tr><th>Image</th><th>Method states</th><th>Consensus participants</th></tr>{''.join(image_rows)}</table>
<h2>Method limitations</h2><ul><li>Measurements represent projected 2-D geometry.</li><li>FATHOM_FIELD_GRAPH_V1 is registered as <code>EXPERIMENTAL_NOT_YET_MEASURING</code>; no graph, topology or fiber-instance values are fabricated.</li><li>Manual 5×5 remains <code>NOT_MEASURED</code> until actual accepted records exist.</li><li>Dataset display balances images; it does not blindly pool skeleton samples.</li></ul>
</body></html>""", encoding="utf-8")
    return index
