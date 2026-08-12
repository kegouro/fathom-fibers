"""Adapters that expose independent Fathom methods through ``MethodResult``."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import loadmat

from .core.contracts import ScientificImage
from .core.methods import (
    Capability,
    CapabilityState,
    DiameterDistribution,
    Estimand,
    MethodCapabilities,
    MethodId,
    MethodResult,
    MethodStatus,
    method_cache_key,
)
from .measurement_records import MeasurementKind, MeasurementRecord
from .oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    SIMPolyIntermediates,
    SIMPolySourceResult,
)

if TYPE_CHECKING:
    from .api import FathomEngine


def _calibration(image: ScientificImage) -> dict[str, Any]:
    return asdict(image.calibration)


def _roi(image: ScientificImage, roi_bbox: tuple[int, int, int, int] | None) -> tuple[int, int, int, int]:
    height, width = image.shape
    valid_height = image.footer_bounds[0] if image.footer_bounds else height
    return roi_bbox or (0, 0, width, valid_height)


def _length_weights(mask: np.ndarray, calibration_x_m: float, calibration_y_m: float) -> np.ndarray:
    """Local physical arclength proxy for 8-connected skeleton samples.

    Every edge contributes half its calibrated length to each endpoint.  Isolated
    pixels receive the smaller pixel dimension as an explicit conservative proxy.
    """
    source = np.asarray(mask, bool)
    points = np.argwhere(source)
    weights = np.empty(points.shape[0], dtype=float)
    lookup = {tuple(point): index for index, point in enumerate(points)}
    for index, (row, col) in enumerate(points):
        total = 0.0
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            if (int(row + dy), int(col + dx)) in lookup:
                total += 0.5 * float(np.hypot(dx * calibration_x_m, dy * calibration_y_m))
        weights[index] = total or min(calibration_x_m, calibration_y_m)
    return weights


def _simply_capabilities(*, matlab: bool) -> MethodCapabilities:
    states = {
        Capability.GLOBAL_DIAMETER_DISTRIBUTION: CapabilityState.AVAILABLE,
        Capability.LOCAL_EDT_DIAMETERS: CapabilityState.AVAILABLE,
        Capability.MASK: CapabilityState.AVAILABLE,
        Capability.SKELETON: CapabilityState.AVAILABLE,
        Capability.MATLAB_COMPATIBILITY_EVIDENCE: CapabilityState.AVAILABLE if not matlab else CapabilityState.UNAVAILABLE,
    }
    return MethodCapabilities(states)


def _simpoly_result(
    *,
    method_id: MethodId,
    image: ScientificImage,
    roi_bbox: tuple[int, int, int, int] | None,
    result: SIMPolySourceResult,
    inter: SIMPolyIntermediates,
    method_version: str,
    provenance: dict[str, Any],
    matlab: bool = False,
) -> MethodResult:
    unit = result.reported_unit
    skeleton = np.asarray(inter.valid_skeleton, bool)
    diameters = np.asarray(result.local_diameters_px, float)
    # SIMPoly's values are stored in px, then converted to its reported unit.
    scale = image.calibration.pixel_size_x_m * 1e6 if unit == "um" else 1.0
    reported = diameters * scale
    weights = _length_weights(skeleton, image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    if weights.shape != reported.shape:
        # This only occurs for incomplete external cache artifacts; do not invent a geometry claim.
        common = None
        flags = (*result.flags, "COMMON_LENGTH_WEIGHT_UNAVAILABLE")
    else:
        common = DiameterDistribution(reported, weights, unit, Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, method_id)
        flags = result.flags
    native = DiameterDistribution(reported, np.ones(reported.size), unit, Estimand.SIMPOLY_NATIVE_GAUSS1, method_id)
    stats = {
        "gauss_a1": result.gaussian_amplitude,
        "gauss_b1": result.reported_center,
        "gauss_c1": result.gaussian_c1_px * scale if result.gaussian_c1_px is not None else None,
        "legacy_source_reported_stdev": result.source_reported_stdev_px * scale if result.source_reported_stdev_px is not None else None,
        "mathematical_sigma": result.mathematical_gaussian_sigma_px * scale if result.mathematical_gaussian_sigma_px is not None else None,
        "arithmetic_mean": result.arithmetic_mean_px * scale if result.arithmetic_mean_px is not None else None,
        "median": result.median_px * scale if result.median_px is not None else None,
    }
    return MethodResult(
        method_id, method_version, image.image_id, _calibration(image), roi_bbox, unit,
        _simply_capabilities(matlab=matlab), MethodStatus.COMPLETE if result.status == "OK" else MethodStatus.FAILED,
        Estimand.SIMPOLY_NATIVE_GAUSS1, result.reported_center, stats, native, common,
        mask=np.asarray(inter.thickened_mask, bool), centerline=skeleton,
        quality_flags=tuple(flags), runtime_seconds=provenance.get("runtime_seconds"), provenance=provenance,
    )


def python_simpoly_adapter(
    engine: FathomEngine,
    image: ScientificImage,
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> MethodResult:
    started = time.monotonic()
    roi = _roi(image, roi_bbox)
    result, inter = engine.run_simpoly(image, profile=PROFILE_CONTROLLED_INPUT_V1, roi_bbox=roi)
    return _simpoly_result(
        method_id=MethodId.PYTHON_SIMPOLY, image=image, roi_bbox=roi, result=result, inter=inter,
        method_version=PROFILE_CONTROLLED_INPUT_V1,
        provenance={
            "profile": PROFILE_CONTROLLED_INPUT_V1,
            "matlab_compatibility": "PARTIAL",
            "known_library_divergence": "bwskel",
            "runtime_seconds": time.monotonic() - started,
            "cache_key": method_cache_key(image_sha256=image.source_sha256, valid_roi=roi, calibration=_calibration(image), method_id=MethodId.PYTHON_SIMPOLY, method_version=PROFILE_CONTROLLED_INPUT_V1, parameters={}),
        },
    )


def _fathom_distribution(image: ScientificImage, candidates: Iterable[Any]) -> DiameterDistribution | None:
    values: list[float] = []
    weights: list[float] = []
    for candidate in candidates:
        sections = list(candidate.proposed_measurements)
        if not sections:
            continue
        line = np.asarray(candidate.centerline_points, float)
        if len(line) > 1:
            delta = np.diff(line, axis=0)
            length_m = float(np.sum(np.hypot(delta[:, 0] * image.calibration.pixel_size_x_m, delta[:, 1] * image.calibration.pixel_size_y_m)))
        else:
            length_m = 0.0
        if length_m <= 0:
            continue
        values.extend(section.width_m * 1e6 for section in sections)
        weights.extend([length_m / len(sections)] * len(sections))
    if not values:
        return None
    return DiameterDistribution(np.asarray(values), np.asarray(weights), "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.FATHOM_LOCAL)


def fathom_local_adapter(
    engine: FathomEngine,
    image: ScientificImage,
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> MethodResult:
    started = time.monotonic()
    roi = _roi(image, roi_bbox)
    analysis = engine.run_fathom(image, roi_bbox=roi)
    common = _fathom_distribution(image, analysis.candidates)
    values = common.diameter if common else np.array([])
    native = DiameterDistribution(values, np.ones(values.size), "um", Estimand.FATHOM_NATIVE_LOCAL, MethodId.FATHOM_LOCAL) if values.size else None
    flags = tuple(sorted(set(analysis.flags) | {flag for candidate in analysis.candidates for flag in candidate.quality_flags}))
    return MethodResult(
        MethodId.FATHOM_LOCAL, "FATHOM_ASSISTED_ROI_V1", image.image_id, _calibration(image), roi, "um",
        MethodCapabilities({Capability.LOCAL_METROLOGY: CapabilityState.AVAILABLE, Capability.CROSS_SECTIONS: CapabilityState.AVAILABLE, Capability.QUALITY_FLAGS: CapabilityState.AVAILABLE, Capability.MANUAL_REVIEW: CapabilityState.AVAILABLE}),
        MethodStatus.COMPLETE, Estimand.FATHOM_NATIVE_LOCAL, float(np.median(values)) if values.size else None,
        {"candidate_count": len(analysis.candidates), "resolution_status": analysis.summary.resolution_status, "section_count": int(values.size)},
        native, common, quality_flags=flags, runtime_seconds=time.monotonic() - started,
        provenance={"roi": roi, "sampling_weights": "candidate_centerline_length / proposed_section_count", "cache_key": method_cache_key(image_sha256=image.source_sha256, valid_roi=roi, calibration=_calibration(image), method_id=MethodId.FATHOM_LOCAL, method_version="FATHOM_ASSISTED_ROI_V1", parameters={})},
    )


def field_graph_placeholder(image: ScientificImage, *, roi_bbox: tuple[int, int, int, int] | None = None) -> MethodResult:
    return MethodResult(
        MethodId.FATHOM_FIELD_GRAPH_V1, "FATHOM_FIELD_GRAPH_V1", image.image_id, _calibration(image), _roi(image, roi_bbox), "um",
        MethodCapabilities({
            Capability.ORIENTATION_FIELD: CapabilityState.EXPERIMENTAL,
            Capability.LOCAL_RADIUS: CapabilityState.EXPERIMENTAL,
            Capability.LOCAL_DIAMETER: CapabilityState.EXPERIMENTAL,
            Capability.GRAPH: CapabilityState.UNAVAILABLE,
            Capability.CROSSINGS: CapabilityState.UNAVAILABLE,
            Capability.FIBER_INSTANCES: CapabilityState.UNAVAILABLE,
            Capability.TOPOLOGY: CapabilityState.UNAVAILABLE,
        }),
        MethodStatus.EXPERIMENTAL_NOT_YET_MEASURING,
        quality_flags=("EXPERIMENTAL_NOT_YET_MEASURING",),
        provenance={"reason": "Contract registered; no field/graph measurement is emitted in V1."},
    )


def manual_adapter(image: ScientificImage, records: Iterable[MeasurementRecord], *, roi_bbox: tuple[int, int, int, int] | None = None) -> MethodResult:
    values = np.asarray([
        record.primary_value * 1e6
        for record in records
        if record.kind == MeasurementKind.PROJECTED_WIDTH and record.primary_value is not None and record.is_included_in_statistics
    ])
    status = MethodStatus.COMPLETE if values.size else MethodStatus.NOT_MEASURED
    distribution = DiameterDistribution(values, np.ones(values.size), "um", Estimand.MANUAL_5X5_REFERENCE, MethodId.MANUAL_5X5_REFERENCE) if values.size else None
    return MethodResult(MethodId.MANUAL_5X5_REFERENCE, "MANUAL_5X5_REFERENCE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", MethodCapabilities({Capability.MANUAL_REVIEW: CapabilityState.AVAILABLE, Capability.LOCAL_METROLOGY: CapabilityState.PARTIAL}), status, Estimand.MANUAL_5X5_REFERENCE, float(np.median(values)) if values.size else None, {"measurement_count": int(values.size)}, distribution, quality_flags=("NOT_MEASURED",) if not values.size else (), provenance={"protocol": "MANUAL_5X5_REFERENCE"})


def matlab_simpoly_cached_adapter(
    image: ScientificImage,
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
    cache_root: str | Path | None = None,
) -> MethodResult:
    """Read MATLAB output only from a caller-supplied or environment cache.

    This adapter deliberately never starts MATLAB. Campaign code is responsible
    for creating/revalidating the cache under its separate oracle workflow.
    """
    root = Path(cache_root or os.environ.get("FATHOM_MATLAB_CACHE_ROOT", ""))
    requested_roi = _roi(image, roi_bbox)
    canonical_roi = _roi(image, None)
    if requested_roi != canonical_roi:
        return MethodResult(
            MethodId.MATLAB_SIMPOLY,
            "MATLAB_R2026A_CACHE",
            image.image_id,
            _calibration(image),
            requested_roi,
            "um",
            _simply_capabilities(matlab=True),
            MethodStatus.NOT_RUN,
            quality_flags=("MATLAB_CACHE_ROI_NOT_AVAILABLE",),
            provenance={"requested_roi": requested_roi, "cached_controlled_roi": canonical_roi},
        )
    if not str(root) or not root.exists():
        return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_CACHE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.NOT_RUN, quality_flags=("MATLAB_CACHE_NOT_AVAILABLE",), provenance={"cache_root": str(root)})
    manifest_path = root / "dataset_manifest.json"
    if not manifest_path.exists():
        return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_CACHE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.NOT_RUN, quality_flags=("MATLAB_CACHE_MANIFEST_MISSING",), provenance={"cache_root": str(root)})
    manifest = json.loads(manifest_path.read_text())
    case = next((entry for entry in manifest.get("cases", ()) if entry.get("sha256") == image.source_sha256), None)
    if case is None:
        return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_CACHE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.NOT_RUN, quality_flags=("MATLAB_CACHE_NO_MATCHING_IMAGE",), provenance={"cache_root": str(root)})
    case_id = case["case_id"]
    run_root = root.parent / "matlab-oracle/runs/r2026a-latest" / case_id / "controlled"
    summary_path, inter_path = run_root / "summary.json", run_root / "intermediates.mat"
    if not summary_path.exists() or not inter_path.exists():
        return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_CACHE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.NOT_RUN, quality_flags=("MATLAB_CACHE_CASE_INCOMPLETE",), provenance={"case_id": case_id})
    summary = json.loads(summary_path.read_text())
    arrays = loadmat(inter_path)
    skeleton = np.asarray(arrays["SK_valid"], bool)
    px = np.asarray(arrays["diameters"], float).ravel()
    scale = image.calibration.pixel_size_x_m * 1e6
    values = px * scale
    weights = _length_weights(skeleton, image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    common = DiameterDistribution(values, weights, "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.MATLAB_SIMPOLY) if weights.shape == values.shape else None
    native = DiameterDistribution(values, np.ones(values.size), "um", Estimand.SIMPOLY_NATIVE_GAUSS1, MethodId.MATLAB_SIMPOLY)
    return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_SOURCE_COMPAT", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.COMPLETE, Estimand.SIMPOLY_NATIVE_GAUSS1, summary.get("gauss_b1"), {"gauss_a1": summary.get("gauss_a1"), "gauss_b1": summary.get("gauss_b1"), "gauss_c1": summary.get("gauss_c1"), "legacy_source_reported_stdev": summary.get("source_reported_stdev"), "mathematical_sigma": summary.get("mathematical_sigma")}, native, common, mask=np.asarray(arrays["BW_thickened"], bool), centerline=skeleton, quality_flags=(() if common else ("COMMON_LENGTH_WEIGHT_UNAVAILABLE",)), provenance={"case_id": case_id, "matlab_version": "R2026a", "source_matlab_sha256": manifest.get("source_matlab_sha256"), "cache_root": str(root)})


__all__ = ["fathom_local_adapter", "field_graph_placeholder", "manual_adapter", "matlab_simpoly_cached_adapter", "python_simpoly_adapter"]
