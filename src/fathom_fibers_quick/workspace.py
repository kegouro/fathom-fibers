"""Qt-free scientific workspace state: datasets, result caches, manual store.

This module is the application layer between frontends and the scientific
core.  It imports no Qt, launches no external runtimes and never mutates
scientific parameters.  A frontend uses :class:`WorkspaceCache`,
:class:`Manual5x5Store` and :func:`compute_image_comparison` to assemble the
per-image scientific evidence the workspace displays.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .core.methods import (
    Capability,
    CapabilityState,
    DiameterDistribution,
    Estimand,
    MethodCapabilities,
    MethodId,
    MethodResult,
    MethodStatus,
)
from .unified_comparison import UnifiedMethodComparison, compare_method_results
from .validation.manual_review import Manual5x5Review

if TYPE_CHECKING:
    from .api import FathomEngine

VALIDATION_ROOT = ".validation/unified-method-comparison"
MANUAL_SUBDIR = "manual5x5"
FULL_CACHE_SUBDIR = "full"
RUNS_SUBDIR = "runs"
FROZEN_MANIFEST_NAME = "dataset_manifest.json"

@dataclass(frozen=True, slots=True)
class WorkspaceImage:
    case_id: str
    filename: str
    absolute_path: Path
    sha256: str | None = None
    resolution_class: str | None = None

    @property
    def stem(self) -> str:
        return self.absolute_path.stem


@dataclass(frozen=True, slots=True)
class WorkspaceDataset:
    dataset_id: str
    images: tuple[WorkspaceImage, ...]
    manifest_path: Path | None = None
    source_dir: Path | None = None

    def image_by_stem(self, stem: str) -> WorkspaceImage | None:
        return next((image for image in self.images if image.stem == stem), None)

    def image_by_case(self, case_id: str) -> WorkspaceImage | None:
        return next((image for image in self.images if image.case_id == case_id), None)


def _read_frozen_manifest(manifest_path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(payload.get("dataset_id", "UNKNOWN_DATASET")), list(payload.get("cases", []))


def load_workspace_dataset(
    dataset_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    repo: str | Path | None = None,
) -> WorkspaceDataset:
    """Build a workspace dataset from a directory of TIFF images.

    When the frozen campaign manifest is available it supplies case ids,
    absolute paths and source hashes; otherwise the directory is scanned and
    stable ``ZEISS_NNN`` identifiers are assigned by sorted filename.
    """
    directory = Path(dataset_dir).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    manifest: Path | None = None
    if manifest_path is not None:
        candidate = Path(manifest_path).resolve()
        if candidate.exists():
            manifest = candidate
    if manifest is None:
        for candidate in _manifest_candidates(directory, repo):
            if candidate.exists():
                manifest = candidate
                break
    dataset_id = f"DATASET_{directory.name}"
    images: list[WorkspaceImage] = []
    if manifest is not None:
        dataset_id, cases = _read_frozen_manifest(manifest)
        for index, case in enumerate(cases, 1):
            path = Path(case.get("absolute_path") or "").resolve()
            if not path.exists():
                path = directory / str(case.get("filename", ""))
            if not path.exists() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            images.append(
                WorkspaceImage(
                    case_id=str(case.get("case_id", f"ZEISS_{index:03d}")),
                    filename=path.name,
                    absolute_path=path,
                    sha256=case.get("sha256"),
                    resolution_class=case.get("resolution_class"),
                )
            )
    if not images:
        paths = sorted(directory.glob("*.tif"), key=lambda item: item.name)
        paths += sorted(directory.glob("*.tiff"), key=lambda item: item.name)
        paths.sort(key=lambda item: item.name)
        images = [
            WorkspaceImage(
                case_id=f"ZEISS_{index:03d}",
                filename=path.name,
                absolute_path=path.resolve(),
            )
            for index, path in enumerate(paths, 1)
        ]
    return WorkspaceDataset(dataset_id, tuple(images), manifest, directory)


def _manifest_candidates(directory: Path, repo: str | Path | None) -> list[Path]:
    candidates: list[Path] = []
    for root in (directory.parents[2] / ".validation", Path(repo or ".") / ".validation"):
        candidates.append(root / "real-tiff-campaign" / FROZEN_MANIFEST_NAME)
    candidates.append(directory / FROZEN_MANIFEST_NAME)
    return candidates


def resolve_matlab_cache_root(
    dataset_dir: str | Path | None = None,
    *,
    repo: str | Path | None = None,
    env_value: str | None = None,
) -> Path | None:
    """Locate the validated MATLAB SIMPoly cache without ever starting MATLAB."""
    import os

    candidates: list[Path] = []
    env = env_value if env_value is not None else os.environ.get("FATHOM_MATLAB_CACHE_ROOT")
    if env:
        candidates.append(Path(env).resolve())
    if repo is not None:
        candidates.append(Path(repo).resolve() / ".validation/real-tiff-campaign")
    if dataset_dir is not None:
        candidates.append(Path(dataset_dir).resolve().parents[2] / ".validation/real-tiff-campaign")
    candidates.append(Path.cwd().resolve() / ".validation/real-tiff-campaign")
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / FROZEN_MANIFEST_NAME).exists():
            return candidate
    return None


class WorkspaceCache:
    """Disk cache of full per-image ``MethodResult`` arrays and summaries.

    The cache lives under ``<repo>/.validation/unified-method-comparison/``.
    Arrays are stored in a per-image NPZ; scalars and provenance in a sibling
    JSON.  Older summary-only campaign runs are still readable and are
    reported as summary-level evidence, never as full samples.
    """

    def __init__(self, repo: str | Path | None = None) -> None:
        root = Path(repo or Path.cwd()).resolve() / VALIDATION_ROOT
        self.root = root
        self.full_dir = root / FULL_CACHE_SUBDIR
        self.runs_dir = root / RUNS_SUBDIR

    @property
    def latest_dir(self) -> Path:
        return self.root / "latest"

    def full_json_path(self, stem: str) -> Path:
        return self.full_dir / f"{stem}.json"

    def full_npz_path(self, stem: str) -> Path:
        return self.full_dir / f"{stem}.npz"

    def has_full(self, stem: str) -> bool:
        return self.full_json_path(stem).exists() and self.full_npz_path(stem).exists()

    def summary_payload(self, stem: str) -> dict[str, Any] | None:
        path = self.runs_dir / f"{stem}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def store_comparison(self, stem: str, comparison: UnifiedMethodComparison) -> Path:
        self.full_dir.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        results: list[dict[str, Any]] = []
        for result in comparison.results:
            entry, entry_arrays = _serialize_result(result)
            arrays.update(entry_arrays)
            results.append(entry)
        payload = {
            "schema": "fathom-workspace-full-v1",
            "image_id": comparison.image_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "results": results,
            "consensus": _serialize_consensus(comparison.consensus),
        }
        json_path = self.full_json_path(stem)
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
        temporary.replace(json_path)
        npz_path = self.full_npz_path(stem)
        if arrays:
            npz_temporary = npz_path.with_suffix(".npz.tmp")
            with npz_temporary.open("wb") as handle:
                np.savez(handle, **arrays)
            npz_temporary.replace(npz_path)
        return json_path

    def load_comparison(self, stem: str) -> UnifiedMethodComparison | None:
        json_path = self.full_json_path(stem)
        npz_path = self.full_npz_path(stem)
        if not json_path.exists() or not npz_path.exists():
            return None
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        arrays = dict(np.load(npz_path))
        results = tuple(_deserialize_result(entry, arrays) for entry in payload["results"])
        comparison = compare_method_results(results)
        consensus = _deserialize_consensus(payload.get("consensus", {}))
        if consensus is not None:

            comparison = UnifiedMethodComparison(
                payload.get("image_id", comparison.image_id),
                comparison.results,
                comparison.summaries,
                comparison.agreements,
                consensus,
            )
        return comparison


def _serialize_result(result: MethodResult) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    prefix = result.method_id.value
    distributions: dict[str, Any] = {}

    def store_distribution(name: str, distribution: DiameterDistribution | None) -> None:
        if distribution is None:
            distributions[name] = None
            return
        diameter_key = f"{prefix}_{name}_diameter"
        weight_key = f"{prefix}_{name}_weight"
        arrays[diameter_key] = distribution.diameter
        arrays[weight_key] = distribution.weight
        distributions[name] = {
            "unit": distribution.unit,
            "estimand": distribution.estimand.value,
            "source_method": distribution.source_method.value,
            "diameter_array": diameter_key,
            "weight_array": weight_key,
        }

    store_distribution("native", result.native_distribution)
    store_distribution("common", result.common_distribution)
    store_distribution("fiber_balanced", result.fiber_balanced_distribution)
    secondary: dict[str, Any] = {}
    for name, distribution in result.secondary_distributions.items():
        key = f"secondary__{name}"
        store_distribution(key, distribution)
        secondary[name] = distributions.pop(key)
    distributions["secondary"] = secondary

    def store_array(name: str, array: np.ndarray | None) -> str | None:
        if array is None:
            return None
        key = f"{prefix}_{name}"
        arrays[key] = np.asarray(array)
        return key

    local_samples: dict[str, str] = {}
    if result.local_samples:
        for name, array in result.local_samples.items():
            local_samples[name] = store_array(f"local_{name}", array) or ""
    entry: dict[str, Any] = {
        "method_id": result.method_id.value,
        "method_version": result.method_version,
        "image_id": result.image_id,
        "valid_roi": result.valid_roi,
        "unit": result.unit,
        "status": result.status.value,
        "native_estimand": result.native_estimand.value if result.native_estimand else None,
        "native_result": result.native_result,
        "native_statistics": dict(result.native_statistics),
        "capabilities": result.capabilities.to_dict(),
        "quality_flags": list(result.quality_flags),
        "confidence": result.confidence,
        "runtime_seconds": result.runtime_seconds,
        "provenance": dict(result.provenance),
        "distributions": distributions,
        "mask": store_array("mask", result.mask),
        "centerline": store_array("centerline", result.centerline),
        "radius_map": store_array("radius_map", result.radius_map),
        "local_samples": {name: key for name, key in local_samples.items() if key},
    }
    orientation = result.orientation_field
    if orientation is not None:
        entry["orientation_qx"] = store_array("orientation_qx", orientation[0])
        entry["orientation_qy"] = store_array("orientation_qy", orientation[1])
    else:
        entry["orientation_qx"] = None
        entry["orientation_qy"] = None
    return entry, arrays


def distribution_from_spec(
    spec: dict[str, Any] | None,
    arrays: dict[str, np.ndarray],
) -> DiameterDistribution | None:
    """Rebuild a stored distribution spec against a loaded NPZ array store."""
    if not spec:
        return None
    diameter = arrays.get(spec["diameter_array"])
    weight = arrays.get(spec["weight_array"])
    if diameter is None or weight is None:
        return None
    return DiameterDistribution(
        diameter,
        weight,
        str(spec["unit"]),
        Estimand(spec["estimand"]),
        MethodId(spec["source_method"]),
    )


def _deserialize_result(entry: dict[str, Any], arrays: dict[str, np.ndarray]) -> MethodResult:
    def array(name: str | None) -> np.ndarray | None:
        return arrays[name] if name else None

    distributions: dict[str, Any] = entry.get("distributions", {})

    def distribution(name: str) -> DiameterDistribution | None:
        return distribution_from_spec(distributions.get(name), arrays)

    secondary_specs = distributions.get("secondary") or {}
    secondary = {
        name: distribution_from_spec(spec, arrays)
        for name, spec in secondary_specs.items()
    }
    secondary = {name: value for name, value in secondary.items() if value is not None}
    orientation = None
    if entry.get("orientation_qx") and entry.get("orientation_qy"):
        qx, qy = array(entry["orientation_qx"]), array(entry["orientation_qy"])
        if qx is not None and qy is not None:
            orientation = (qx, qy)
    local_samples: dict[str, np.ndarray] = {}
    for name, key in (entry.get("local_samples") or {}).items():
        value = array(key)
        if value is not None:
            local_samples[name] = value
    return MethodResult(
        MethodId(entry["method_id"]),
        str(entry["method_version"]),
        str(entry["image_id"]),
        dict(entry.get("native_statistics", {})) or {},
        tuple(entry["valid_roi"]) if entry.get("valid_roi") else None,
        str(entry["unit"]),
        MethodCapabilities(
            {Capability(key): CapabilityState(value) for key, value in (entry.get("capabilities") or {}).items()}
        ),
        MethodStatus(entry["status"]),
        Estimand(entry["native_estimand"]) if entry.get("native_estimand") else None,
        entry.get("native_result"),
        dict(entry.get("native_statistics", {})),
        distribution("native"),
        distribution("common"),
        secondary,
        distribution("fiber_balanced"),
        mask=array(entry.get("mask")),
        centerline=array(entry.get("centerline")),
        orientation_field=orientation,
        radius_map=array(entry.get("radius_map")),
        local_samples=local_samples,
        quality_flags=tuple(entry.get("quality_flags", ())),
        confidence=entry.get("confidence"),
        runtime_seconds=entry.get("runtime_seconds"),
        provenance=dict(entry.get("provenance", {})),
    )


def _serialize_consensus(consensus: Any) -> dict[str, Any]:
    if consensus.distribution is None:
        return {
            "participating_methods": [item.value for item in consensus.participating_methods],
            "excluded_methods": dict(consensus.excluded_methods),
            "quantile_grid": consensus.quantile_grid.tolist(),
            "quantiles": consensus.quantiles.tolist(),
            "disagreement_mad": consensus.disagreement_mad.tolist(),
        }
    return {
        "participating_methods": [item.value for item in consensus.participating_methods],
        "excluded_methods": dict(consensus.excluded_methods),
        "quantile_grid": consensus.quantile_grid.tolist(),
        "quantiles": consensus.quantiles.tolist(),
        "disagreement_mad": consensus.disagreement_mad.tolist(),
        "distribution": {
            "diameter": consensus.distribution.diameter.tolist(),
            "weight": consensus.distribution.weight.tolist(),
            "unit": consensus.distribution.unit,
            "estimand": consensus.distribution.estimand.value,
        },
    }


def _deserialize_consensus(payload: dict[str, Any]) -> Any:
    from .core.distributions import ConsensusPseudoReference, DiameterDistribution

    distribution = None
    if payload.get("distribution"):
        spec = payload["distribution"]
        distribution = DiameterDistribution(
            np.asarray(spec["diameter"], float),
            np.asarray(spec["weight"], float),
            str(spec["unit"]),
            Estimand(spec["estimand"]),
            MethodId.CONSENSUS_PSEUDO_REFERENCE_V1,
        )
    return ConsensusPseudoReference(
        distribution,
        np.asarray(payload.get("quantile_grid", []), float),
        np.asarray(payload.get("quantiles", []), float),
        np.asarray(payload.get("disagreement_mad", []), float),
        tuple(MethodId(item) for item in payload.get("participating_methods", ())),
        dict(payload.get("excluded_methods", {})),
    )


def compute_image_comparison(
    engine: FathomEngine,
    image: Any,
    *,
    matlab_cache_root: str | Path | None = None,
) -> UnifiedMethodComparison:
    """Run the unified comparison for one calibrated image (no MATLAB launch)."""
    return engine.compare_all_methods(image, matlab_cache_root=matlab_cache_root)


def compute_comparison_staged(
    engine: FathomEngine,
    image: Any,
    *,
    matlab_cache_root: str | Path | None = None,
    records: Iterable[Any] = (),
    progress: Callable[[str], None] | None = None,
) -> UnifiedMethodComparison:
    """Stage-by-stage unified comparison with per-method progress reports.

    This mirrors :meth:`FathomEngine.compare_all_methods` so frontends and the
    headless precompute script share one orchestration.  MATLAB is read from
    the validated cache only and is never launched.
    """
    from .methods import (
        classical_field_adapter,
        fathom_local_adapter,
        manual_adapter,
        matlab_simpoly_cached_adapter,
        python_simpoly_adapter_with_intermediates,
    )

    def report(message: str) -> None:
        if progress is not None:
            progress(message)

    report("Python SIMPoly")
    python_result, intermediates = python_simpoly_adapter_with_intermediates(
        engine, image, roi_bbox=None
    )
    report("Fathom Local")
    local_result = fathom_local_adapter(engine, image, roi_bbox=None)
    report("Fathom Field")
    field_result = classical_field_adapter(
        image,
        roi_bbox=None,
        mask=intermediates.thickened_mask,
    )
    report("MATLAB cache")
    matlab_result = matlab_simpoly_cached_adapter(
        image, roi_bbox=None, cache_root=matlab_cache_root
    )
    manual_result = manual_adapter(image, records, roi_bbox=None)
    return compare_method_results(
        (matlab_result, python_result, local_result, field_result, manual_result)
    )


def summary_method_rows(payload: dict[str, Any]) -> dict[MethodId, dict[str, Any]]:
    """Index a summary-only campaign payload by method id for partial displays."""
    rows: dict[MethodId, dict[str, Any]] = {}
    for entry in payload.get("results", ()):
        try:
            method = MethodId(entry["method_id"])
        except ValueError:
            continue
        rows[method] = entry
    return rows


def manual_store_path(root: Path, dataset_id: str) -> Path:
    """Path of one dataset's manual 5x5 store under a validation root."""
    return root / MANUAL_SUBDIR / f"{dataset_id}.json"


class Manual5x5Store:
    """Dataset-level 5x5 reference persistence with atomic writes.

    Each accepted measurement is written immediately so a crash never loses
    the 400-measurement campaign.  ``root`` is the validation root used by
    :class:`WorkspaceCache` (``<repo>/.validation/unified-method-comparison``).
    """

    def __init__(self, root: str | Path | None = None, dataset_id: str = "") -> None:
        validation_root = Path(root or Path.cwd())
        if not validation_root.name:
            validation_root = Path.cwd()
        if validation_root.name == ".validation":
            validation_root = validation_root / "unified-method-comparison"
        self.root = validation_root
        self.path = manual_store_path(validation_root, dataset_id)
        self.dataset_id = dataset_id
        self.reviews: dict[str, Manual5x5Review] = {}
        self.status: dict[str, str] = {}

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("dataset_id") != self.dataset_id:
            return
        for case_id, value in payload.get("reviews", {}).items():
            self.reviews[case_id] = Manual5x5Review.from_dict(value)
        self.status = dict(payload.get("status", {}))

    def ensure_review(self, case_id: str) -> Manual5x5Review:
        review = self.reviews.get(case_id)
        if review is None:
            review = Manual5x5Review(case_id)
            self.reviews[case_id] = review
        return review

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset_id": self.dataset_id,
            "updated_utc": datetime.now(UTC).isoformat(),
            "reviews": {case_id: review.to_dict() for case_id, review in self.reviews.items()},
            "status": dict(self.status),
        }
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)

    def set_image_status(self, case_id: str, status: str) -> None:
        self.status[case_id] = status
        self.save()

    @property
    def total_measured(self) -> int:
        return sum(review.measurement_count for review in self.reviews.values())

    @property
    def reviewed_images(self) -> int:
        return sum(
            value in {"REVIEWED", "SKIPPED"} for value in self.status.values()
        )


__all__ = [
    "FULL_CACHE_SUBDIR",
    "MANUAL_SUBDIR",
    "RUNS_SUBDIR",
    "VALIDATION_ROOT",
    "Manual5x5Store",
    "WorkspaceCache",
    "WorkspaceDataset",
    "WorkspaceImage",
    "compute_comparison_staged",
    "compute_image_comparison",
    "load_workspace_dataset",
    "resolve_matlab_cache_root",
    "summary_method_rows",
]
