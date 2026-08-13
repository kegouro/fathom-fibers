"""FATHOM_ORIENTED_RIBBON_V1 — stage 3: refined-centerline remeasurement.

Re-measures EDT, paired-edge and raw-intensity profile diameters along the
smooth refined centerline from stage 2.  The mask, segmentation, EDT,
paired-edge and profile algorithms are reused unchanged; only the sampling
center and normal change.  Raw results are never overwritten; refined values
are reported additively as a separate family.

The pipeline stops after this stage: no second refinement iteration is
performed even if a residual center shift remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .centerline_refinement import CenterlineRefinementResult
from .fiber_field import (
    BoundaryContour,
    IntensityProfileSampler,
    OrientedBoundaryEngine,
)
from .methods import (
    DiameterDistribution,
    Estimand,
    MethodId,
)

STAGE = "REFINED_REMEASUREMENT"
FLAG_ORIENTATION_DISAGREEMENT = "REFINED_ORIENTATION_DISAGREEMENT"
ORIENTATION_DISAGREEMENT_DEG = 45.0

DISTRIBUTIONS = {
    "FATHOM_FIELD_REFINED_EDT_DIAMETER": Estimand.FATHOM_FIELD_REFINED_EDT_DIAMETER,
    "FATHOM_FIELD_REFINED_EDGE_DIAMETER": Estimand.FATHOM_FIELD_REFINED_EDGE_DIAMETER,
    "FATHOM_FIELD_REFINED_PROFILE_DIAMETER": Estimand.FATHOM_FIELD_REFINED_PROFILE_DIAMETER,
}


@dataclass(frozen=True, slots=True)
class RefinedRemeasurement:
    """Per-sample refined measurements; NaN/False where not supported.

    All positions are physical metres; diameters and shifts are microns.
    Arrays are indexed exactly like the original ``local_samples``.
    """

    refined_mask: np.ndarray
    refined_tangent_xy: np.ndarray
    refined_normal_xy: np.ndarray
    refined_edt_um: np.ndarray
    refined_r_minus_um: np.ndarray
    refined_r_plus_um: np.ndarray
    refined_edge_um: np.ndarray
    refined_edge_accepted: np.ndarray
    refined_edge_flags: np.ndarray
    refined_profile_um: np.ndarray
    refined_profile_accepted: np.ndarray
    refined_profile_flags: np.ndarray
    refined_asymmetry: np.ndarray
    residual_center_shift_um: np.ndarray
    residual_normal_shift_um: np.ndarray
    residual_tangential_shift_um: np.ndarray
    refined_arc_weight_m: np.ndarray
    axis_disagreement_deg: np.ndarray
    profile_center_shift_um: np.ndarray
    distributions: dict[str, DiameterDistribution]
    summary: dict[str, float | int | None]
    flags: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def supported_mask(self) -> np.ndarray:
        return self.refined_mask


def remeasure_refined_centerline(
    refinement: CenterlineRefinementResult,
    *,
    mask: np.ndarray,
    body: np.ndarray,
    edt_radius_m: np.ndarray,
    pixel_size_xy_m: tuple[float, float],
    roi_origin_px: tuple[int, int],
    raw_coherence: np.ndarray,
    raw_normal_xy: np.ndarray,
    raw_qx: np.ndarray,
    raw_qy: np.ndarray,
    boundary_engine: OrientedBoundaryEngine | None = None,
    profile_sampler: IntensityProfileSampler | None = None,
    contours: tuple[BoundaryContour, ...] = (),
) -> RefinedRemeasurement:
    """Re-measure geometry along the refined centerline (one pass, then stop).

    Only samples with ``refined_mask`` on a valid segment are measured;
    junctions, gaps, crossings and rejected observations stay unavailable.
    The refined normal is the geometric tangent's perpendicular with its sign
    aligned to the raw normal (``dot >= 0``) so plus/minus never flip.
    """
    binary = np.asarray(mask, dtype=bool)
    px, py = (float(value) for value in pixel_size_xy_m)
    x0, y0 = roi_origin_px
    n = int(refinement.original_xy_m.shape[0])
    refined = refinement.refined_xy_m
    refined_mask = np.asarray(refinement.refined_mask, bool) if refined is not None else np.zeros(n, bool)
    segment_ids = refinement.segment_ids if refinement.segment_ids is not None else np.full(n, -1)

    tangent = np.full((n, 2), np.nan)
    normal = np.full((n, 2), np.nan)
    if refined is not None:
        tangent, normal = _refined_tangents_normals(
            refined, refined_mask, segment_ids, raw_normal_xy
        )

    # --- refined EDT: bilinear interpolation of the physical EDT -----------
    edt_refined_um = np.full(n, np.nan)
    if refined is not None:
        rows_frac = (refined[refined_mask, 1] / py) - y0
        cols_frac = (refined[refined_mask, 0] / px) - x0
        radius = ndimage.map_coordinates(
            np.asarray(edt_radius_m, float),
            [rows_frac, cols_frac],
            order=1,
            mode="nearest",
        )
        edt_refined_um[refined_mask] = 2.0 * radius * 1e6

    # --- refined paired edge (same ray engine, refined centers) -------------
    engine = boundary_engine or OrientedBoundaryEngine()
    paired = None
    edge_um = np.full(n, np.nan)
    r_minus_um = np.full(n, np.nan)
    r_plus_um = np.full(n, np.nan)
    edge_accepted = np.zeros(n, bool)
    edge_flags = np.full(n, "", dtype="<U80")
    asymmetry = np.full(n, np.nan)
    if refined is not None and np.any(refined_mask):
        paired = engine.pair_centers(
            refined[refined_mask],
            normal[refined_mask],
            edt_refined_um[refined_mask] * 1e-6,
            raw_coherence[refined_mask],
            mask=binary,
            pixel_size_xy_m=pixel_size_xy_m,
            contours=contours,
        )
        edge_um[refined_mask] = np.asarray(paired.diameter_m) * 1e6
        r_minus_um[refined_mask] = np.asarray(paired.radius_minus_m) * 1e6
        r_plus_um[refined_mask] = np.asarray(paired.radius_plus_m) * 1e6
        edge_accepted[refined_mask] = np.asarray(paired.accepted, bool)
        asymmetry[refined_mask] = np.asarray(paired.asymmetry)
        edge_flags[refined_mask] = np.asarray(
            [";".join(flags) for flags in paired.flags], dtype="<U80"
        )

    # --- refined profile (same sampler, refined paired priors) --------------
    sampler = profile_sampler or IntensityProfileSampler()
    profile_um = np.full(n, np.nan)
    profile_accepted = np.zeros(n, bool)
    profile_flags = np.full(n, "", dtype="<U80")
    residual_shift = np.full(n, np.nan)
    residual_normal = np.full(n, np.nan)
    residual_tangent = np.full(n, np.nan)
    if paired is not None and refined is not None:
        # residual center shift: midpoint of the re-measured mask edges minus
        # the refined centerline, projected on the refined normal/tangent
        midpoint = 0.5 * (np.asarray(paired.minus_xy_m) + np.asarray(paired.plus_xy_m))
        centers = refined[refined_mask]
        shift_vector = midpoint - centers
        target = np.flatnonzero(refined_mask)
        normal_m = normal[target]
        tangent_m = tangent[target]
        residual_shift[target] = np.linalg.norm(shift_vector, axis=1) * 1e6
        residual_normal[target] = (
            shift_vector[:, 0] * normal_m[:, 0] + shift_vector[:, 1] * normal_m[:, 1]
        ) * 1e6
        residual_tangent[target] = (
            shift_vector[:, 0] * tangent_m[:, 0] + shift_vector[:, 1] * tangent_m[:, 1]
        ) * 1e6
    profile_center_shift_um = np.full(n, np.nan)
    if paired is not None:
        profile = sampler.refine(
            np.asarray(body, dtype=float),
            paired,
            pixel_size_xy_m=pixel_size_xy_m,
        )
        profile_um[refined_mask] = np.asarray(profile.diameter_m) * 1e6
        profile_accepted[refined_mask] = np.asarray(profile.accepted, bool)
        profile_flags[refined_mask] = np.asarray(
            [";".join(flags) for flags in profile.flags], dtype="<U80"
        )
        if refined is not None:
            u_center = 0.5 * (np.asarray(profile.minus_u_m) + np.asarray(profile.plus_u_m))
            valid_profile = np.isfinite(u_center)
            if np.any(valid_profile):
                profile_center_shift_um[np.flatnonzero(refined_mask)[valid_profile]] = (
                    np.abs(u_center[valid_profile]) * 1e6
                )

    # --- arc-length weights (Voronoi half-neighbour on refined segments) ----
    arc_weight = np.zeros(n)
    if refined is not None:
        arc_weight = _refined_arc_weights(n, segment_ids, refinement)

    # --- orientation consistency diagnostic --------------------------------
    axis_disagreement = np.full(n, np.nan)
    if refined is not None:
        field_theta = 0.5 * np.arctan2(raw_qy, raw_qx)
        field_axis = np.column_stack((np.cos(field_theta), np.sin(field_theta)))
        dot_axis = np.abs(
            tangent[refined_mask, 0] * field_axis[refined_mask, 0]
            + tangent[refined_mask, 1] * field_axis[refined_mask, 1]
        )
        axis_disagreement[refined_mask] = np.degrees(np.arccos(np.clip(dot_axis, 0.0, 1.0)))

    distributions = {}
    weights = arc_weight
    if np.any(edge_accepted & np.isfinite(edge_um) & (arc_weight > 0)):
        distributions["FATHOM_FIELD_REFINED_EDGE_DIAMETER"] = DiameterDistribution(
            edge_um[edge_accepted & (arc_weight > 0)],
            weights[edge_accepted & (arc_weight > 0)],
            "um",
            Estimand.FATHOM_FIELD_REFINED_EDGE_DIAMETER,
            MethodId.FATHOM_FIELD_GRAPH_V1,
        )
    if np.any(profile_accepted & np.isfinite(profile_um) & (arc_weight > 0)):
        distributions["FATHOM_FIELD_REFINED_PROFILE_DIAMETER"] = DiameterDistribution(
            profile_um[profile_accepted & (arc_weight > 0)],
            weights[profile_accepted & (arc_weight > 0)],
            "um",
            Estimand.FATHOM_FIELD_REFINED_PROFILE_DIAMETER,
            MethodId.FATHOM_FIELD_GRAPH_V1,
        )
    if np.any(refined_mask & np.isfinite(edt_refined_um) & (arc_weight > 0)):
        distributions["FATHOM_FIELD_REFINED_EDT_DIAMETER"] = DiameterDistribution(
            edt_refined_um[refined_mask & (arc_weight > 0)],
            weights[refined_mask & (arc_weight > 0)],
            "um",
            Estimand.FATHOM_FIELD_REFINED_EDT_DIAMETER,
            MethodId.FATHOM_FIELD_GRAPH_V1,
        )

    summary = _remeasurement_summary(
        refined_mask,
        edt_refined_um,
        edge_um,
        edge_accepted,
        profile_um,
        profile_accepted,
        asymmetry,
        residual_shift,
        arc_weight,
        axis_disagreement,
    )
    severe = np.any(np.isfinite(axis_disagreement) & (axis_disagreement > ORIENTATION_DISAGREEMENT_DEG))
    flags = (STAGE, FLAG_ORIENTATION_DISAGREEMENT) if severe else (STAGE,)
    metadata = {
        "algorithm": "FATHOM_ORIENTED_RIBBON_V1",
        "stage": STAGE,
        "seed_source": "FATHOM_FIELD_GRAPH_V1_CENTERLINE",
        "midpoint_preference": "PROFILE_ELSE_MASK",
        "smoothing": "SCIPY_CUBIC_SMOOTHING_SPLINE",
        "field_mask_version": "PYTHON_SIMPOLY_CONTROLLED_INPUT_THICKENED_MASK",
        "paired_edge_version": "CENTERLINE_NORMAL_BINARY_RAY_INTERPOLATION_V1",
        "profile_version": "SUBPIXEL_GRADIENT_REFINEMENT_V1",
        "edt_interpolation": "SCIPY_MAP_COORDINATES_BILINEAR",
        "normal_sign_rule": "DOT_WITH_RAW_NORMAL_GE_0",
        "orientation_disagreement_threshold_deg": ORIENTATION_DISAGREEMENT_DEG,
    }
    return RefinedRemeasurement(
        refined_mask=refined_mask,
        refined_tangent_xy=tangent,
        refined_normal_xy=normal,
        refined_edt_um=edt_refined_um,
        refined_r_minus_um=r_minus_um,
        refined_r_plus_um=r_plus_um,
        refined_edge_um=edge_um,
        refined_edge_accepted=edge_accepted,
        refined_edge_flags=edge_flags,
        refined_profile_um=profile_um,
        refined_profile_accepted=profile_accepted,
        refined_profile_flags=profile_flags,
        refined_asymmetry=asymmetry,
        residual_center_shift_um=residual_shift,
        residual_normal_shift_um=residual_normal,
        residual_tangential_shift_um=residual_tangent,
        refined_arc_weight_m=arc_weight,
        axis_disagreement_deg=axis_disagreement,
        profile_center_shift_um=profile_center_shift_um,
        distributions=distributions,
        summary=summary,
        flags=flags,
        metadata=metadata,
    )


def _refined_tangents_normals(
    refined_xy: np.ndarray,
    refined_mask: np.ndarray,
    segment_ids: np.ndarray,
    raw_normal_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference tangents per segment; normal sign aligned to raw."""
    n = refined_xy.shape[0]
    tangent = np.full((n, 2), np.nan)
    for segment_id in np.unique(segment_ids[segment_ids >= 0]):
        indices = np.flatnonzero((segment_ids == segment_id) & refined_mask)
        if indices.size < 2:
            continue
        positions = refined_xy[indices]
        # central differences; one-sided at segment ends
        step = np.zeros((indices.size, 2))
        step[1:-1] = positions[2:] - positions[:-2]
        step[0] = positions[1] - positions[0]
        step[-1] = positions[-1] - positions[-2]
        norms = np.linalg.norm(step, axis=1)
        tangent[indices] = step / norms[:, None]
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    # sign alignment: keep plus/minus consistent with the raw normal
    flipped = normal.copy()
    if np.any(refined_mask):
        dot = (
            normal[refined_mask, 0] * raw_normal_xy[refined_mask, 0]
            + normal[refined_mask, 1] * raw_normal_xy[refined_mask, 1]
        )
        flip = dot < 0.0
        to_flip = refined_mask.copy()
        to_flip[refined_mask] = flip
        flipped[to_flip] = -flipped[to_flip]
    return tangent, flipped


def _refined_arc_weights(
    n: int,
    segment_ids: np.ndarray,
    refinement: CenterlineRefinementResult,
) -> np.ndarray:
    """Voronoi half-neighbour physical arc-length weights from segment ``s``."""
    weights = np.zeros(n)
    for segment in refinement.segments:
        s = np.asarray(segment.s_m, float)
        indices = np.asarray(segment.source_indices, int)
        if s.size < 2:
            continue
        half_steps = 0.5 * np.diff(s)
        weight = np.empty(s.size)
        weight[0] = half_steps[0]
        weight[-1] = half_steps[-1]
        weight[1:-1] = half_steps[:-1] + half_steps[1:]
        weights[indices] = weight
    return weights


def _remeasurement_summary(
    refined_mask: np.ndarray,
    edt_um: np.ndarray,
    edge_um: np.ndarray,
    edge_accepted: np.ndarray,
    profile_um: np.ndarray,
    profile_accepted: np.ndarray,
    asymmetry: np.ndarray,
    residual_shift: np.ndarray,
    arc_weight: np.ndarray,
    axis_disagreement: np.ndarray,
) -> dict[str, float | int | None]:
    supported = refined_mask
    edge_ok = edge_accepted & np.isfinite(edge_um)
    profile_ok = profile_accepted & np.isfinite(profile_um)
    return {
        "refined_supported_count": int(np.sum(supported)),
        "refined_edge_accepted_count": int(np.sum(edge_ok)),
        "refined_edge_acceptance_fraction": float(np.mean(edge_ok[supported])) if np.any(supported) else None,
        "refined_profile_accepted_count": int(np.sum(profile_ok)),
        "refined_profile_acceptance_fraction": float(np.mean(profile_ok[supported])) if np.any(supported) else None,
        "refined_edt_median_um": float(np.nanmedian(edt_um[supported])) if np.any(supported) else None,
        "refined_edge_median_um": float(np.nanmedian(edge_um[edge_ok])) if np.any(edge_ok) else None,
        "refined_profile_median_um": float(np.nanmedian(profile_um[profile_ok])) if np.any(profile_ok) else None,
        "refined_asymmetry_median": float(np.nanmedian(asymmetry[edge_ok])) if np.any(edge_ok) else None,
        "refined_residual_shift_median_um": float(np.nanmedian(residual_shift[edge_ok])) if np.any(edge_ok) else None,
        "refined_residual_shift_p90_um": float(np.nanquantile(residual_shift[edge_ok], 0.9)) if np.any(edge_ok) else None,
        "refined_axis_disagreement_median_deg": float(np.nanmedian(axis_disagreement[supported])) if np.any(supported) else None,
        "refined_arc_length_m": float(np.sum(arc_weight)),
    }


__all__ = [
    "FLAG_ORIENTATION_DISAGREEMENT",
    "ORIENTATION_DISAGREEMENT_DEG",
    "STAGE",
    "RefinedRemeasurement",
    "remeasure_refined_centerline",
]
