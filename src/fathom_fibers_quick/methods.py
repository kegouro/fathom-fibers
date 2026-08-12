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
from scipy.stats import spearmanr

from .core.contracts import ScientificImage
from .core.fiber_field import ClassicalFiberField, IntensityProfileSampler, OrientedBoundaryEngine
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


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 3:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return value if np.isfinite(value) else None


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
    return python_simpoly_adapter_with_intermediates(engine, image, roi_bbox=roi_bbox)[0]


def python_simpoly_adapter_with_intermediates(
    engine: FathomEngine,
    image: ScientificImage,
    *,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> tuple[MethodResult, SIMPolyIntermediates]:
    """Run Python SIMPoly once and expose its mask to sibling adapters.

    The field backend receives the resulting mask as an explicit, provenance
    tracked input; it does not alter SIMPoly or invoke MATLAB at runtime.
    """
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
    ), inter


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
    sections_xy = np.empty((0, 4), float)
    sections_weights = np.empty(0, float)
    sections_flags: list[str] = []
    for candidate in analysis.candidates:
        for proposal in candidate.proposed_measurements:
            sections_xy = np.vstack(
                (
                    sections_xy,
                    np.asarray(
                        [
                            (
                                proposal.p1[0],
                                proposal.p1[1],
                                proposal.p2[0],
                                proposal.p2[1],
                            )
                        ],
                        float,
                    ),
                )
            )
            sections_weights = np.append(sections_weights, proposal.width_m * 1e6)
            sections_flags.append(";".join(sorted(proposal.quality_flags)))
    local_samples = None
    if sections_xy.size:
        local_samples = {
            "section_x0_px": sections_xy[:, 0],
            "section_y0_px": sections_xy[:, 1],
            "section_x1_px": sections_xy[:, 2],
            "section_y1_px": sections_xy[:, 3],
            "section_width_um": sections_weights,
            "section_flags": np.asarray(sections_flags, dtype="<U80"),
        }
    return MethodResult(
        MethodId.FATHOM_LOCAL, "FATHOM_ASSISTED_ROI_V1", image.image_id, _calibration(image), roi, "um",
        MethodCapabilities({Capability.LOCAL_METROLOGY: CapabilityState.AVAILABLE, Capability.CROSS_SECTIONS: CapabilityState.AVAILABLE, Capability.QUALITY_FLAGS: CapabilityState.AVAILABLE, Capability.MANUAL_REVIEW: CapabilityState.AVAILABLE}),
        MethodStatus.COMPLETE, Estimand.FATHOM_NATIVE_LOCAL, float(np.median(values)) if values.size else None,
        {"candidate_count": len(analysis.candidates), "resolution_status": analysis.summary.resolution_status, "section_count": int(values.size)},
        native, common, local_samples=local_samples, quality_flags=flags, runtime_seconds=time.monotonic() - started,
        provenance={"roi": roi, "sampling_weights": "candidate_centerline_length / proposed_section_count", "cache_key": method_cache_key(image_sha256=image.source_sha256, valid_roi=roi, calibration=_calibration(image), method_id=MethodId.FATHOM_LOCAL, method_version="FATHOM_ASSISTED_ROI_V1", parameters={})},
    )


def classical_field_adapter(
    image: ScientificImage,
    *,
    mask: np.ndarray,
    roi_bbox: tuple[int, int, int, int] | None = None,
    field: ClassicalFiberField | None = None,
) -> MethodResult:
    """Adapt an explicit mask and scientific image to Field V1 metrology.

    V1 intentionally uses a documented ``skimage.skeletonize`` sampling
    baseline.  It provides neither fibre identities nor graph topology.
    """
    started = time.monotonic()
    roi = _roi(image, roi_bbox)
    x0, y0, x1, y1 = roi
    body = np.asarray(image.gray[y0:y1, x0:x1], dtype=float)
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != body.shape:
        raise ValueError("field mask must match the selected image body")
    backend = field or ClassicalFiberField()
    output = backend.infer_field(
        body,
        mask=binary,
        pixel_size_xy_m=(image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m),
    )
    boundary_engine = OrientedBoundaryEngine()
    contours = boundary_engine.extract_contours(
        binary, pixel_size_xy_m=(image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    )
    paired = boundary_engine.pair_centerline(
        mask=binary,
        centerline=np.asarray(output.centerline, bool),
        orientation_qx=np.asarray(output.orientation_qx, float),
        orientation_qy=np.asarray(output.orientation_qy, float),
        coherence=np.asarray(output.coherence, float),
        edt_diameter_m=np.asarray(output.diameter_m, float),
        pixel_size_xy_m=(image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m),
        contours=contours,
    )
    profile = IntensityProfileSampler().refine(
        body, paired, pixel_size_xy_m=(image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    )
    centerline = np.asarray(output.centerline, dtype=bool)
    rows, cols = np.nonzero(centerline)
    weights_m = _length_weights(centerline, image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    diameter_um = np.asarray(output.diameter_m, dtype=float)[centerline] * 1e6
    radius_um = np.asarray(output.radius_m, dtype=float)[centerline] * 1e6
    coherence = np.asarray(output.coherence, dtype=float)[centerline]
    qx = np.asarray(output.orientation_qx, dtype=float)[centerline]
    qy = np.asarray(output.orientation_qy, dtype=float)[centerline]
    valid = np.isfinite(diameter_um) & np.isfinite(weights_m) & (weights_m > 0) & (diameter_um > 0)
    diameter_um, radius_um, weights_m = diameter_um[valid], radius_um[valid], weights_m[valid]
    coherence, qx, qy, rows, cols = coherence[valid], qx[valid], qy[valid], rows[valid], cols[valid]
    edge_offset_m = np.asarray((x0 * image.calibration.pixel_size_x_m, y0 * image.calibration.pixel_size_y_m), float)
    common = (
        DiameterDistribution(diameter_um, weights_m, "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.FATHOM_FIELD_GRAPH_V1)
        if diameter_um.size else None
    )
    edge_accepted = np.asarray(paired.accepted, bool)[valid]
    edge_diameter_um = np.asarray(paired.diameter_m, float)[valid] * 1e6
    edge_asymmetry = np.asarray(paired.asymmetry, float)[valid]
    edge_distribution = (
        DiameterDistribution(
            edge_diameter_um[edge_accepted], weights_m[edge_accepted], "um",
            Estimand.FATHOM_FIELD_PAIRED_EDGE_DIAMETER, MethodId.FATHOM_FIELD_GRAPH_V1,
        ) if np.any(edge_accepted) else None
    )
    profile_accepted = np.asarray(profile.accepted, bool)[valid]
    profile_diameter_um = np.asarray(profile.diameter_m, float)[valid] * 1e6
    profile_distribution = (
        DiameterDistribution(
            profile_diameter_um[profile_accepted], weights_m[profile_accepted], "um",
            Estimand.FATHOM_FIELD_PROFILE_DIAMETER, MethodId.FATHOM_FIELD_GRAPH_V1,
        ) if np.any(profile_accepted) else None
    )
    edge_flags = np.asarray(
        [";".join(flags) for flags in paired.flags], dtype="<U80"
    )[valid]
    profile_flags = np.asarray(
        [";".join(flags) for flags in profile.flags], dtype="<U80"
    )[valid]
    d_min = np.asarray(paired.d_min_from_edges_m, float)[valid]
    imbalance = np.asarray(paired.absolute_side_imbalance_m, float)[valid]
    edge_minus_edt = edge_diameter_um * 1e-6 - np.asarray(output.diameter_m, float)[centerline][valid]
    edt_minus_dmin = np.asarray(output.diameter_m, float)[centerline][valid] - d_min
    profile_minus_edge = profile.diameter_m[valid] - edge_diameter_um * 1e-6
    native = common
    vector_x = float(np.average(qx, weights=weights_m)) if weights_m.size else 0.0
    vector_y = float(np.average(qy, weights=weights_m)) if weights_m.size else 0.0
    nematic = float(np.hypot(vector_x, vector_y)) if weights_m.size else None
    mean_coherence = float(np.average(coherence, weights=weights_m)) if weights_m.size else None
    status = MethodStatus.EXPERIMENTAL_FIELD_MEASURING if common is not None else MethodStatus.FAILED
    flags = ["EXPERIMENTAL_FIELD_MEASURING", "FIELD_STAGE_IMPLEMENTED", "GRAPH_STAGE_NOT_IMPLEMENTED"]
    if not diameter_um.size:
        flags.append("NO_CENTERLINE_DIAMETER_SAMPLES")
    local_samples = {
        "x_m": (cols + x0) * image.calibration.pixel_size_x_m,
        "y_m": (rows + y0) * image.calibration.pixel_size_y_m,
        "qx": qx,
        "qy": qy,
        "coherence": coherence,
        "radius_um": radius_um,
        "diameter_um": diameter_um,
        "arc_length_weight_m": weights_m,
        "normal_xy": paired.normal_xy[valid],
        "minus_xy_m": paired.minus_xy_m[valid] + edge_offset_m,
        "plus_xy_m": paired.plus_xy_m[valid] + edge_offset_m,
        "radius_minus_um": paired.radius_minus_m[valid] * 1e6,
        "radius_plus_um": paired.radius_plus_m[valid] * 1e6,
        "edge_diameter_um": edge_diameter_um,
        "edge_asymmetry": edge_asymmetry,
        "edge_accepted": edge_accepted,
        "edge_tangent_alignment": paired.tangent_alignment[valid],
        "edge_normal_consistency": paired.boundary_normal_consistency[valid],
        "d_min_from_edges_um": d_min * 1e6,
        "edge_minus_dmin_um": imbalance * 1e6,
        "edt_minus_dmin_um": edt_minus_dmin * 1e6,
        "edge_minus_edt_um": edge_minus_edt * 1e6,
        "profile_diameter_um": profile_diameter_um,
        "profile_accepted": profile_accepted,
        "profile_minus_edge_um": profile_minus_edge * 1e6,
        "profile_minus_shift_um": profile.minus_shift_m[valid] * 1e6,
        "profile_plus_shift_um": profile.plus_shift_m[valid] * 1e6,
        "profile_minus_u_um": profile.minus_u_m[valid] * 1e6,
        "profile_plus_u_um": profile.plus_u_m[valid] * 1e6,
        "profile_gradient_snr": profile.gradient_snr[valid],
        "suggested_center_shift_um": profile.suggested_center_shift_m[valid] * 1e6,
        "edge_flags": edge_flags,
        "profile_flags": profile_flags,
        "seed_row": rows + y0,
        "seed_col": cols + x0,
    }
    if local_samples:
        from .core.centerline_refinement import refine_centerline
        from .core.oriented_ribbon import compute_midpoint_observations

        ribbon = compute_midpoint_observations(
            local_samples, include_observations=False
        )
        local_samples.update(
            {
                "refine_accepted": ribbon.accepted_mask,
                "refine_confidence": ribbon.confidence,
                "midpoint_mask_x_m": ribbon.mask_midpoint_xy_m[:, 0],
                "midpoint_mask_y_m": ribbon.mask_midpoint_xy_m[:, 1],
                "midpoint_profile_x_m": ribbon.profile_midpoint_xy_m[:, 0],
                "midpoint_profile_y_m": ribbon.profile_midpoint_xy_m[:, 1],
                "midpoint_preferred_x_m": ribbon.preferred_midpoint_xy_m[:, 0],
                "midpoint_preferred_y_m": ribbon.preferred_midpoint_xy_m[:, 1],
                "center_shift_um": ribbon.shift_um,
                "center_shift_signed_um": ribbon.signed_normal_shift_um,
                "center_shift_tangent_um": ribbon.tangential_shift_um,
                "midpoint_source": ribbon.midpoint_source,
            }
        )
        ribbon_flags = tuple(flag for flag in ribbon.flags if flag != "MIDPOINT_OBSERVATIONS_ONLY")
        flags.extend(ribbon_flags)
        smooth = refine_centerline(
            ribbon,
            local_samples,
            np.asarray(centerline, dtype=bool),
            pixel_size_xy_m=(image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m),
        )
        refined_xy = smooth.refined_xy_m
        if refined_xy is not None:
            local_samples.update(
                {
                    "refined_xy_m": refined_xy,
                    "refined_mask": smooth.refined_mask,
                    "segment_id": smooth.segment_ids,
                    "smooth_shift_um": smooth.smooth_shift_um,
                    "smooth_shift_signed_um": smooth.smooth_normal_shift_um,
                    "smooth_shift_tangent_um": smooth.smooth_tangential_shift_um,
                }
            )
            smooth_flags = tuple(flag for flag in smooth.flags if flag not in {"SMOOTH_CENTERLINE_V1"})
            flags.extend(smooth_flags)
        native_statistics_payload = {
            "refine_accepted_count": ribbon.summary["accepted_count"],
            "refine_coverage_fraction": ribbon.coverage_fraction,
            "refine_median_shift_um": ribbon.summary["median_shift_um"],
            "refine_p90_shift_um": ribbon.summary["p90_shift_um"],
            "smooth_coverage_fraction": smooth.summary["smooth_coverage"],
            "smooth_segment_count": smooth.summary["segment_count"],
            "smooth_median_shift_um": smooth.summary["median_smooth_shift_um"],
            "smooth_p90_shift_um": smooth.summary["p90_smooth_shift_um"],
        }
    else:
        native_statistics_payload = {}
    return MethodResult(
        MethodId.FATHOM_FIELD_GRAPH_V1, "CLASSICAL_FIBER_FIELD_V1", image.image_id, _calibration(image), roi, "um",
        MethodCapabilities({
            Capability.MASK: CapabilityState.AVAILABLE,
            Capability.SKELETON: CapabilityState.AVAILABLE,
            Capability.ORIENTATION_FIELD: CapabilityState.AVAILABLE,
            Capability.ORIENTED_BOUNDARIES: CapabilityState.AVAILABLE,
            Capability.PAIRED_EDGE_LOCAL_WIDTH: CapabilityState.EXPERIMENTAL,
            Capability.LOCAL_RADIUS: CapabilityState.AVAILABLE,
            Capability.LOCAL_DIAMETER: CapabilityState.AVAILABLE,
            Capability.GLOBAL_DIAMETER_DISTRIBUTION: CapabilityState.AVAILABLE,
            Capability.GRAPH: CapabilityState.UNAVAILABLE,
            Capability.CROSSINGS: CapabilityState.UNAVAILABLE,
            Capability.FIBER_INSTANCES: CapabilityState.UNAVAILABLE,
            Capability.TOPOLOGY: CapabilityState.UNAVAILABLE,
        }),
        status,
        Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER,
        float(np.median(diameter_um)) if diameter_um.size else None,
        {
            "sample_count": int(diameter_um.size),
            "mean_coherence": mean_coherence,
            "nematic_order_parameter": nematic,
            "radius_sample_median_um": float(np.median(radius_um)) if radius_um.size else None,
            "centerline_source": output.metadata["centerline_algorithm"],
            "radius_estimator": output.metadata["radius_method"],
            "edge_raw_count": int(edge_diameter_um.size),
            "edge_accepted_count": int(np.sum(edge_accepted)),
            "edge_acceptance_fraction": float(np.mean(edge_accepted)) if edge_accepted.size else None,
            "edge_mean_asymmetry": float(np.nanmean(edge_asymmetry)) if np.any(np.isfinite(edge_asymmetry)) else None,
            "edge_median_asymmetry": float(np.nanmedian(edge_asymmetry)) if np.any(np.isfinite(edge_asymmetry)) else None,
            "edge_mean_tangent_alignment": float(np.nanmean(paired.tangent_alignment[valid])) if np.any(np.isfinite(paired.tangent_alignment[valid])) else None,
            "edge_mean_normal_consistency": float(np.nanmean(paired.boundary_normal_consistency[valid])) if np.any(np.isfinite(paired.boundary_normal_consistency[valid])) else None,
            "edge_flag_counts": {flag: sum(flag in sample for sample in paired.flags) for flag in sorted({flag for sample in paired.flags for flag in sample})},
            "centering_median_absolute_imbalance_um": float(np.nanmedian(imbalance) * 1e6),
            "centering_p90_absolute_imbalance_um": float(np.nanquantile(imbalance, .9) * 1e6),
            "centering_median_edge_minus_edt_um": float(np.nanmedian(edge_minus_edt) * 1e6),
            "centering_median_edt_minus_dmin_um": float(np.nanmedian(edt_minus_dmin) * 1e6),
            "exploratory_spearman_imbalance_vs_edge_minus_edt": _spearman(imbalance, edge_minus_edt),
            "exploratory_spearman_coherence_vs_abs_edge_minus_edt": _spearman(coherence, np.abs(edge_minus_edt)),
            "profile_accepted_count": int(np.sum(profile_accepted)),
            "profile_acceptance_fraction": float(np.mean(profile_accepted)) if profile_accepted.size else None,
            "profile_median_abs_edge_shift_um": float(np.nanmedian(np.abs(np.concatenate((profile.minus_shift_m[valid], profile.plus_shift_m[valid])))) * 1e6),
            "profile_median_center_shift_um": float(np.nanmedian(np.abs(profile.suggested_center_shift_m[valid])) * 1e6),
            "profile_flag_counts": {flag: sum(flag in sample for sample in profile.flags) for flag in sorted({flag for sample in profile.flags for flag in sample})},
            "exploratory_spearman_abs_profile_shift_vs_profile_minus_edge": _spearman(
                np.maximum(np.abs(profile.minus_shift_m[valid]), np.abs(profile.plus_shift_m[valid])), profile_minus_edge
            ),
            "boundary_contour_count": len(contours),
            **native_statistics_payload,
        },
        native,
        common,
        secondary_distributions={
            name: distribution for name, distribution in {
                "FATHOM_FIELD_PAIRED_EDGE_DIAMETER": edge_distribution,
                "FATHOM_FIELD_PROFILE_DIAMETER": profile_distribution,
            }.items() if distribution is not None
        },
        mask=binary,
        centerline=centerline,
        orientation_field=(np.asarray(output.orientation_qx), np.asarray(output.orientation_qy)),
        radius_map=np.asarray(output.radius_m),
        local_samples=local_samples,
        quality_flags=tuple(flags), confidence=mean_coherence,
        runtime_seconds=time.monotonic() - started,
        provenance={
            "field_stage": "FIELD_STAGE_IMPLEMENTED",
            "graph_stage": "GRAPH_STAGE_NOT_IMPLEMENTED",
            "mask_source": "PYTHON_SIMPOLY_CONTROLLED_INPUT_THICKENED_MASK",
            "mask_profile": PROFILE_CONTROLLED_INPUT_V1,
            "centerline_source": output.metadata["centerline_algorithm"],
            "orientation_method": output.metadata["orientation_method"],
            "radius_method": output.metadata["radius_method"],
            "contour_algorithm": "SKIMAGE_FIND_CONTOURS_0.5",
            "tangent_estimator": "CENTRAL_CONTOUR_DIFFERENCE_PHYSICAL_V1",
            "pairing_strategy": "CENTERLINE_NORMAL_BINARY_RAY_INTERPOLATION_V1",
            "pairing_config": {
                "tangent_window": boundary_engine.tangent_window,
                "ray_samples": boundary_engine.ray_samples,
                "low_coherence": boundary_engine.low_coherence,
                "high_asymmetry": boundary_engine.high_asymmetry,
                "minimum_tangent_alignment": boundary_engine.minimum_tangent_alignment,
            },
            "profile_input": "RAW_SCIENTIFIC_SEM_BODY",
            "profile_sampler": "LINEAR_SUBPIXEL_NORMAL_PROFILE_PARABOLIC_GRADIENT_PEAK_V1",
            "sampling_weights": "half_edge_physical_length_on_8_connected_centerline",
            "cache_key": method_cache_key(image_sha256=image.source_sha256, valid_roi=roi, calibration=_calibration(image), method_id=MethodId.FATHOM_FIELD_GRAPH_V1, method_version="CLASSICAL_FIBER_FIELD_V1", parameters=dict(output.metadata)),
        },
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
    if not summary_path.exists():
        return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_CACHE", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.NOT_RUN, quality_flags=("MATLAB_CACHE_CASE_INCOMPLETE",), provenance={"case_id": case_id})
    summary = json.loads(summary_path.read_text())
    if not inter_path.exists():
        # The 16-case oracle cache deliberately retained compact summaries but
        # not private full intermediates.  Keep its real native MATLAB result
        # visible without fabricating a length-weighted common distribution.
        return MethodResult(
            MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_SOURCE_COMPAT", image.image_id,
            _calibration(image), requested_roi, "um", _simply_capabilities(matlab=True),
            MethodStatus.COMPLETE, Estimand.SIMPOLY_NATIVE_GAUSS1,
            summary.get("gauss_b1"),
            {
                "gauss_a1": summary.get("gauss_a1"),
                "gauss_b1": summary.get("gauss_b1"),
                "gauss_c1": summary.get("gauss_c1"),
                "legacy_source_reported_stdev": summary.get("source_reported_stdev"),
                "mathematical_sigma": summary.get("mathematical_sigma"),
                "diameter_n": summary.get("diameter_n"),
                "diameter_median": summary.get("diameter_median"),
            },
            quality_flags=("MATLAB_RAW_DIAMETERS_UNAVAILABLE", "COMMON_LENGTH_WEIGHT_UNAVAILABLE"),
            provenance={
                "case_id": case_id,
                "matlab_version": "R2026a",
                "source_matlab_sha256": manifest.get("source_matlab_sha256"),
                "cache_root": str(root),
                "cache_representation": "SUMMARY_ONLY_NO_COMMON_DISTRIBUTION",
            },
        )
    arrays = loadmat(inter_path)
    skeleton = np.asarray(arrays["SK_valid"], bool)
    px = np.asarray(arrays["diameters"], float).ravel()
    scale = image.calibration.pixel_size_x_m * 1e6
    values = px * scale
    weights = _length_weights(skeleton, image.calibration.pixel_size_x_m, image.calibration.pixel_size_y_m)
    common = DiameterDistribution(values, weights, "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.MATLAB_SIMPOLY) if weights.shape == values.shape else None
    native = DiameterDistribution(values, np.ones(values.size), "um", Estimand.SIMPOLY_NATIVE_GAUSS1, MethodId.MATLAB_SIMPOLY)
    return MethodResult(MethodId.MATLAB_SIMPOLY, "MATLAB_R2026A_SOURCE_COMPAT", image.image_id, _calibration(image), _roi(image, roi_bbox), "um", _simply_capabilities(matlab=True), MethodStatus.COMPLETE, Estimand.SIMPOLY_NATIVE_GAUSS1, summary.get("gauss_b1"), {"gauss_a1": summary.get("gauss_a1"), "gauss_b1": summary.get("gauss_b1"), "gauss_c1": summary.get("gauss_c1"), "legacy_source_reported_stdev": summary.get("source_reported_stdev"), "mathematical_sigma": summary.get("mathematical_sigma")}, native, common, mask=np.asarray(arrays["BW_thickened"], bool), centerline=skeleton, quality_flags=(() if common else ("COMMON_LENGTH_WEIGHT_UNAVAILABLE",)), provenance={"case_id": case_id, "matlab_version": "R2026a", "source_matlab_sha256": manifest.get("source_matlab_sha256"), "cache_root": str(root)})


__all__ = ["classical_field_adapter", "fathom_local_adapter", "manual_adapter", "matlab_simpoly_cached_adapter", "python_simpoly_adapter", "python_simpoly_adapter_with_intermediates"]
