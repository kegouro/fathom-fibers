"""Qt-free analysis bundle export for the deliverable release.

Produces an organized folder with tidy measurements, an image x estimator
summary, manual 5x5 records, method results JSON, provenance, the final
scientific report and its figures.  The private SEM TIFFs are never copied.
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .core.methods import MethodId, MethodResult
from .reports import build_final_dataset_report
from .workspace import WorkspaceCache

BUNDLE_VERSION = "1.0.0"

_METHOD_NAMES = {
    "MATLAB_SIMPOLY": "MATLAB SIMPoly",
    "PYTHON_SIMPOLY": "Python SIMPoly",
    "FATHOM_LOCAL": "Fathom Local",
    "FATHOM_FIELD_GRAPH_V1": "Fathom Field",
    "MANUAL_5X5_REFERENCE": "Manual 5x5",
    "CONSENSUS_PSEUDO_REFERENCE_V1": "Consensus",
}

FIELD_ESTIMATORS = (
    ("diameter_um", "Fathom Field", "Raw EDT"),
    ("edge_diameter_um", "Fathom Field", "Raw Edge"),
    ("profile_diameter_um", "Fathom Field", "Raw Profile"),
    ("refined_edt_um", "Fathom Oriented Ribbon V1", "Refined EDT"),
    ("refined_edge_um", "Fathom Oriented Ribbon V1", "Refined Edge"),
    ("refined_profile_um", "Fathom Oriented Ribbon V1", "Refined Profile"),
)


def _git_commit(repo: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip()[:12]
    except Exception:
        return "unknown"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def export_analysis_bundle(
    repo: str | Path,
    *,
    dataset: Any,
    manual_store: Any = None,
    output_dir: str | Path,
) -> Path:
    """Write the complete analysis bundle and return its root path."""
    repo = Path(repo)
    output = Path(output_dir)
    results_dir = output / "results"
    figures_dir = output / "figures"
    report_dir = output / "report"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    cache = WorkspaceCache(repo)
    comparisons = []
    for image in dataset.images:
        comparison = cache.load_comparison(image.stem)
        if comparison is not None:
            comparisons.append(comparison)
    comparisons.sort(key=lambda item: item.image_id)

    _write_dataset_summary(results_dir, comparisons)
    _write_measurements(results_dir, comparisons)
    _write_manual_csv(results_dir, manual_store, dataset)
    _write_method_results(results_dir, comparisons)
    _write_provenance(results_dir, repo, dataset, cache)

    build_final_dataset_report(
        repo,
        dataset=dataset,
        manual_store=manual_store,
        output_dir=report_dir,
        comparisons=comparisons,
    )
    _copy_figures(repo, figures_dir)

    _write_readme(output, dataset, manual_store)
    return output


def _write_dataset_summary(directory: Path, comparisons: list[Any]) -> Path:
    fields = [
        "image", "method", "estimator", "N", "coverage", "mean", "median",
        "std", "IQR", "P05", "P95", "status",
    ]
    path = directory / "dataset_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for comparison in comparisons:
            image_id = Path(comparison.image_id).stem
            for result in comparison.results:
                rows = _distribution_summaries(result)
                for estimator, summary in rows:
                    writer.writerow(
                        {
                            "image": image_id,
                            "method": _METHOD_NAMES.get(result.method_id.value, result.method_id.value),
                            "estimator": estimator,
                            **{key: _fmt_num(summary.get(key)) for key in ("N", "coverage", "mean", "median", "std", "IQR", "P05", "P95")},
                            "status": result.status.value,
                        }
                    )
    return path


def _distribution_summaries(result: MethodResult) -> list[tuple[str, dict[str, Any]]]:
    from .core.distributions import summarize_distribution

    candidates: list[tuple[str, Any]] = [("common", result.common_distribution)]
    if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1:
        candidates = [
            ("Raw EDT", result.common_distribution),
            ("Raw Edge", result.secondary_distributions.get("FATHOM_FIELD_PAIRED_EDGE_DIAMETER")),
            ("Raw Profile", result.secondary_distributions.get("FATHOM_FIELD_PROFILE_DIAMETER")),
            ("Refined EDT", result.secondary_distributions.get("FATHOM_FIELD_REFINED_EDT_DIAMETER")),
            ("Refined Edge", result.secondary_distributions.get("FATHOM_FIELD_REFINED_EDGE_DIAMETER")),
            ("Refined Profile", result.secondary_distributions.get("FATHOM_FIELD_REFINED_PROFILE_DIAMETER")),
        ]
    if result.method_id == MethodId.MANUAL_5X5_REFERENCE:
        candidates = [("Manual", result.native_distribution)]
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, distribution in candidates:
        if distribution is None or distribution.diameter.size == 0:
            rows.append((name, {"N": 0, "coverage": None, "mean": None, "median": None, "std": None, "IQR": None, "P05": None, "P95": None}))
            continue
        summary = summarize_distribution(distribution)
        iqr = (summary.p75 - summary.p25) if summary.p25 is not None and summary.p75 is not None else None
        rows.append(
            (
                name,
                {
                    "N": summary.n,
                    "coverage": None,
                    "mean": summary.weighted_mean,
                    "median": summary.weighted_median,
                    "std": None,
                    "IQR": iqr,
                    "P05": summary.p05,
                    "P95": summary.p95,
                },
            )
        )
    return rows


def _write_measurements(directory: Path, comparisons: list[Any]) -> Path:
    path = directory / "measurements.csv"
    header = _csv_line(
        ["image_id", "method", "estimator", "sample_id",
         "x_um", "y_um", "diameter_um", "weight_m", "accepted", "confidence",
         "flags", "refinement_status", "segment_id"]
    )
    with path.open("w", newline="", encoding="utf-8", buffering=1 << 20) as handle:
        handle.write(header)
        chunk: list[str] = []
        for comparison in comparisons:
            image_id = Path(comparison.image_id).stem
            field = next(
                (
                    result
                    for result in comparison.results
                    if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
                ),
                None,
            )
            if field is not None and field.local_samples:
                _write_field_rows(chunk, handle, image_id, field)
            for method_id, method_name, estimator_name in (
                (MethodId.PYTHON_SIMPOLY, "Python SIMPoly", "common"),
                (MethodId.FATHOM_LOCAL, "Fathom Local", "common"),
            ):
                result = next(
                    (item for item in comparison.results if item.method_id == method_id), None
                )
                if result is not None and result.common_distribution is not None:
                    _write_distribution_rows(chunk, handle, image_id, method_name, estimator_name, result.common_distribution)
            manual = next(
                (
                    result
                    for result in comparison.results
                    if result.method_id == MethodId.MANUAL_5X5_REFERENCE
                ),
                None,
            )
            if manual is not None and manual.native_distribution is not None:
                _write_distribution_rows(chunk, handle, image_id, "Manual 5x5", "Manual", manual.native_distribution)
        if chunk:
            handle.write("".join(chunk))
    return path


def _write_field_rows(chunk: list[str], handle: Any, image_id: str, field: MethodResult) -> None:
    ls = field.local_samples
    n = int(np.asarray(ls["x_m"]).size)

    def column(key: str) -> np.ndarray:
        values = np.asarray(ls.get(key), float) if key in ls else np.full(n, np.nan)
        if values.size != n:
            values = np.full(n, np.nan)
        return values

    x_um = column("x_m") * 1e6
    y_um = column("y_m") * 1e6
    accepted_raw = column("edge_accepted")
    accepted_refined = column("refined_edge_accepted")
    refined_mask = column("refined_mask")
    segment = np.asarray(ls.get("segment_id"), int) if "segment_id" in ls else np.full(n, -1)
    flags = np.asarray(ls.get("edge_flags"), dtype=str) if "edge_flags" in ls else np.full(n, "", dtype=str)
    refined_flags = np.asarray(ls.get("refined_edge_flags"), dtype=str) if "refined_edge_flags" in ls else np.full(n, "", dtype=str)
    confidence = column("refine_confidence")
    weight_m = column("arc_length_weight_m")
    for estimator_key, method, estimator in FIELD_ESTIMATORS:
        diameters = column(estimator_key)
        is_refined = estimator_key.startswith("refined_")
        accepted = accepted_refined if is_refined else accepted_raw
        flag_values = refined_flags if is_refined else flags
        for index in range(n):
            if not np.isfinite(diameters[index]):
                continue
            chunk.append(
                _csv_line(
                    [
                        image_id,
                        method,
                        estimator,
                        index,
                        _fmt_num(x_um[index]),
                        _fmt_num(y_um[index]),
                        _fmt_num(diameters[index]),
                        _fmt_num(weight_m[index]),
                        bool(accepted[index]),
                        _fmt_num(confidence[index]),
                        str(flag_values[index]),
                        "refined" if refined_mask[index] else "not_refined",
                        int(segment[index]) if segment[index] >= 0 else "",
                    ]
                )
            )
            if len(chunk) >= 20000:
                handle.write("".join(chunk))
                chunk.clear()


def _write_distribution_rows(chunk: list[str], handle: Any, image_id: str, method: str, estimator: str, distribution: Any) -> None:
    for index, (diameter, weight) in enumerate(zip(distribution.diameter, distribution.weight, strict=False)):
        chunk.append(
            _csv_line(
                [
                    image_id,
                    method,
                    estimator,
                    index,
                    "",
                    "",
                    _fmt_num(diameter),
                    _fmt_num(weight),
                    True,
                    "",
                    "",
                    "",
                    "",
                ]
            )
        )
        if len(chunk) >= 20000:
            handle.write("".join(chunk))
            chunk.clear()


def _csv_line(values: list[Any]) -> str:
    parts = []
    for value in values:
        if value is None or value == "":
            parts.append('""')
            continue
        text = str(value)
        if "," in text or '"' in text or "\n" in text:
            parts.append('"' + text.replace('"', '""') + '"')
        else:
            parts.append(text)
    return ",".join(parts) + "\n"


def _write_manual_csv(directory: Path, manual_store: Any, dataset: Any) -> Path | None:
    if manual_store is None or not manual_store.reviews:
        return None
    path = directory / "manual_5x5.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case_id", "image", "row", "column", "position", "status", "diameter_um", "notes", "timestamp"])
        for image in dataset.images:
            review = manual_store.reviews.get(image.case_id)
            if review is None:
                continue
            for row in range(5):
                for column in range(5):
                    cell = review.cell(row, column)
                    writer.writerow(
                        [
                            image.case_id,
                            image.filename,
                            row + 1,
                            column + 1,
                            cell.position,
                            cell.status.value,
                            _fmt_num(cell.diameter),
                            cell.notes,
                            cell.timestamp,
                        ]
                    )
    return path


def _write_method_results(directory: Path, comparisons: list[Any]) -> Path:

    payload = {
        "bundle_version": BUNDLE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "images": [
            {
                "image_id": comparison.image_id,
                "methods": {
                    result.method_id.value: {
                        "status": result.status.value,
                        "native_result": result.native_result,
                        "native_statistics": _json_safe(result.native_statistics),
                        "quality_flags": list(result.quality_flags),
                        "provenance": _json_safe(result.provenance),
                        "common_distribution": _distribution_payload(result.common_distribution),
                        "secondary_distributions": {
                            name: _distribution_payload(distribution)
                            for name, distribution in result.secondary_distributions.items()
                        },
                    }
                    for result in comparison.results
                },
                "consensus": {
                    "participating_methods": [item.value for item in comparison.consensus.participating_methods],
                    "excluded_methods": dict(comparison.consensus.excluded_methods),
                },
            }
            for comparison in comparisons
        ],
    }
    path = directory / "method_results.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _distribution_payload(distribution: Any) -> dict[str, Any] | None:
    if distribution is None:
        return None
    return {
        "unit": distribution.unit,
        "estimand": distribution.estimand.value,
        "n": int(distribution.diameter.size),
        "mean": _json_safe(float(np.mean(distribution.diameter))),
        "median": _json_safe(float(np.median(distribution.diameter))),
        "p05": _json_safe(float(np.quantile(distribution.diameter, 0.05))),
        "p95": _json_safe(float(np.quantile(distribution.diameter, 0.95))),
    }


def _write_provenance(directory: Path, repo: Path, dataset: Any, cache: WorkspaceCache) -> Path:
    commit = _git_commit(repo)
    payload = {
        "application": "Fathom Fibers",
        "version": _application_version(),
        "git_commit": commit,
        "bundle_version": BUNDLE_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "dataset_id": dataset.dataset_id,
        "cache_schema": cache.FULL_SCHEMA,
        "images": [
            {
                "case_id": image.case_id,
                "filename": image.filename,
                "sha256": image.sha256,
                "resolution_class": image.resolution_class,
            }
            for image in dataset.images
        ],
        "matlab": "validated controlled-origin cache; native b1 only; never launched",
        "simpoly_source": "canonical SIMPoly source hash verified in validation profile",
        "ribbon": {
            "algorithm": "FATHOM_ORIENTED_RIBBON_V1",
            "status": "EXPERIMENTAL",
        },
    }
    path = directory / "provenance.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _application_version() -> str:
    from . import __version__

    return __version__


def _copy_figures(repo: Path, figures_dir: Path) -> None:
    from shutil import copytree

    final = repo / ".validation/final-report"
    if final.exists():
        copytree(final / "images", figures_dir / "images", dirs_exist_ok=True)
        for name in final.glob("dataset-figure-*.png"):
            name.replace(figures_dir / name.name)
    ribbon = repo / ".validation/oriented-ribbon-v1/latest"
    if ribbon.exists():
        (figures_dir / "validation").mkdir(exist_ok=True)
        for name in ribbon.glob("*.png"):
            name.replace(figures_dir / "validation" / name.name)


def _write_readme(output: Path, dataset: Any, manual_store: Any) -> None:
    manual_line = "manual 5x5: no measurements recorded" if manual_store is None or manual_store.total_measured == 0 else f"manual 5x5: {manual_store.total_measured} / 400 recorded"
    (output / "README.md").write_text(
        f"""# Fathom Fibers — Analysis Bundle

Dataset: `{dataset.dataset_id}` — {len(dataset.images)} images.
{manual_line}

## Contents
- `results/dataset_summary.csv` — one row per image x method x estimator
- `results/measurements.csv` — tidy per-sample measurements
- `results/manual_5x5.csv` — manual reference grid (when present)
- `results/method_results.json` — full method results and distributions
- `results/provenance.json` — versions, hashes and cache provenance
- `report/index.html` — final scientific report
- `figures/` — dataset and per-image figures

Raw SEM images are intentionally not included.
""",
        encoding="utf-8",
    )


def _fmt_num(value: Any) -> Any:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not np.isfinite(number):
        return ""
    return round(number, 6)


__all__ = ["BUNDLE_VERSION", "export_analysis_bundle"]
