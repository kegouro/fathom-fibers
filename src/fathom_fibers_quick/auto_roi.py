from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .analysis import _best_edge_pair, sample_profile, validate_measurement_geometry
from .model import Calibration


@dataclass(slots=True)
class ProposedMeasurement:
    p1: tuple[float, float]
    p2: tuple[float, float]
    center: tuple[float, float]
    width_m: float
    quality_flags: set[str] = field(default_factory=set)


@dataclass(slots=True)
class AutoFiberCandidate:
    candidate_id: str
    roi_bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    component_label: int
    centerline_points: list[tuple[float, float]]
    proposed_measurements: list[ProposedMeasurement]
    median_width_m: float | None
    confidence_score: float
    quality_flags: set[str] = field(default_factory=set)
    status: str = "PENDING"  # "PENDING", "ACCEPTED", "REJECTED", "EDITED"

    @property
    def confidence_level(self) -> str:
        if "TOUCHES_ROI_EDGE" in self.quality_flags or "TOUCHES_INVALID_MASK" in self.quality_flags:
            return "Baja"
        if "TOO_SMALL" in self.quality_flags or "LOW_ELONGATION" in self.quality_flags:
            return "Baja"
        if self.confidence_score >= 0.70 and not self.quality_flags:
            return "Alta"
        if self.confidence_score >= 0.40:
            return "Media"
        return "Baja"


@dataclass(slots=True)
class AutoROISummary:
    total_components: int
    measurable_candidates: int
    high_confidence: int
    needs_review: int
    excluded: int
    exclusion_reasons: dict[str, int] = field(default_factory=dict)


def otsu_threshold(gray_patch: np.ndarray) -> float:
    """Computes global Otsu threshold for a 2D intensity array."""
    flat = gray_patch.ravel()
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 128.0
    hist, bin_edges = np.histogram(flat, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    cum_hist = np.cumsum(hist)
    cum_sum = np.cumsum(hist * bin_centers)

    total_weight = cum_hist[-1]
    total_sum = cum_sum[-1]

    w1 = cum_hist[:-1]
    w2 = total_weight - w1

    m1 = cum_sum[:-1] / np.maximum(w1, 1e-12)
    m2 = (total_sum - cum_sum[:-1]) / np.maximum(w2, 1e-12)

    variance_between = w1 * w2 * (m1 - m2) ** 2
    max_var = float(np.max(variance_between))
    max_indices = np.where(variance_between >= max_var - 1e-5)[0]
    idx = int(np.mean(max_indices))
    return float(bin_centers[idx])


def analyze_roi(
    gray: np.ndarray,
    roi_bbox: tuple[int, int, int, int],  # (x0, y0, x1, y1)
    calibration: Calibration,
    footer_bounds: tuple[int, int] | None = None,
    polarity: str = "auto",  # "auto", "bright", "dark"
    min_area_px: int = 40,
    min_elongation: float = 2.2,
    min_width_px: float = 2.0,
    n_sections: int = 3,
) -> tuple[list[AutoFiberCandidate], AutoROISummary]:
    """Extracts simple fiber candidates from an image ROI using connected component analysis."""
    x0, y0, x1, y1 = roi_bbox
    height_img, width_img = gray.shape[:2]

    # Clamp bbox within image bounds
    x0 = max(0, min(x0, width_img - 1))
    x1 = max(x0 + 1, min(x1, width_img))
    y0 = max(0, min(y0, height_img - 1))
    y1 = max(y0 + 1, min(y1, height_img))

    roi_patch = gray[y0:y1, x0:x1]
    if roi_patch.size < 20:
        return [], AutoROISummary(0, 0, 0, 0, 0, {})

    # Robust normalization
    p2, p98 = np.percentile(roi_patch, (2, 98))
    norm_patch = np.clip((roi_patch - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)

    # Determine polarity
    if polarity == "auto":
        # Compare mean intensity of center vs border
        border_mask = np.ones_like(roi_patch, dtype=bool)
        border_mask[1:-1, 1:-1] = False
        is_bright = norm_patch[~border_mask].mean() > norm_patch[border_mask].mean()
    else:
        is_bright = (polarity == "bright")

    # Otsu segmentation
    thresh = otsu_threshold(norm_patch * 255.0) / 255.0
    binary = norm_patch > thresh if is_bright else norm_patch < thresh

    # Morphological cleaning
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((2, 2)))

    # Label components
    labeled, num_features = ndimage.label(binary)
    slices = ndimage.find_objects(labeled)

    candidates: list[AutoFiberCandidate] = []
    exclusion_reasons: dict[str, int] = defaultdict(int)

    total_components = num_features
    measurable_cnt = 0
    high_conf_cnt = 0
    needs_review_cnt = 0
    excluded_cnt = 0

    px_w_m = calibration.pixel_size_x_m
    px_h_m = calibration.pixel_size_y_m

    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue

        comp_mask = (labeled[sl] == idx)
        area_px = int(comp_mask.sum())

        ys, xs = np.where(comp_mask)
        # Global image coordinates
        abs_xs = x0 + sl[1].start + xs
        abs_ys = y0 + sl[0].start + ys

        flags: set[str] = set()

        if area_px < min_area_px:
            flags.add("TOO_SMALL")

        # Check contact with ROI edge
        if (
            sl[1].start == 0
            or sl[0].start == 0
            or sl[1].stop == (x1 - x0)
            or sl[0].stop == (y1 - y0)
        ):
            flags.add("TOUCHES_ROI_EDGE")

        # Check contact with footer
        if footer_bounds is not None:
            fy0, fy1 = footer_bounds
            if np.any((abs_ys >= fy0) & (abs_ys <= fy1)):
                flags.add("TOUCHES_INVALID_MASK")

        # PCA in physical coordinates for orientation & elongation
        X_m = abs_xs * px_w_m
        Y_m = abs_ys * px_h_m
        coords_m = np.column_stack((X_m, Y_m))

        if coords_m.shape[0] < 5:
            flags.add("TOO_SMALL")
            cov = np.eye(2)
        else:
            cov = np.cov(coords_m, rowvar=False)

        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        lambda1, lambda2 = evals[0], max(evals[1], 1e-24)
        elongation = float(math.sqrt(lambda1 / lambda2))

        if elongation < min_elongation:
            flags.add("LOW_ELONGATION")

        # Principal orientation direction (in image space)
        # Note: evecs are in physical coordinates (dx_m, dy_m)
        v_phys = evecs[:, 0]
        v_img = np.array([v_phys[0] / px_w_m, v_phys[1] / px_h_m], dtype=float)
        v_norm = np.linalg.norm(v_img)
        if v_norm == 0:
            v_dir = np.array([1.0, 0.0])
        else:
            v_dir = v_img / v_norm

        v_perp = np.array([-v_dir[1], v_dir[0]], dtype=float)

        # Project component points onto principal axis
        center_abs = (float(abs_xs.mean()), float(abs_ys.mean()))
        pts_rel = np.column_stack((abs_xs - center_abs[0], abs_ys - center_abs[1]))
        projections = pts_rel @ v_dir

        t_min, t_max = float(projections.min()), float(projections.max())
        span = t_max - t_min

        if span < 6.0:
            flags.add("TOO_SMALL")

        # Place equispaced section centers avoiding ends (15% trim)
        t_start = t_min + 0.15 * span
        t_end = t_max - 0.15 * span
        t_centers = np.linspace(t_start, t_end, n_sections)

        proposed_m: list[ProposedMeasurement] = []
        centerline_pts: list[tuple[float, float]] = []

        for tc in t_centers:
            c_pt = (center_abs[0] + tc * v_dir[0], center_abs[1] + tc * v_dir[1])
            centerline_pts.append(c_pt)

            # Sample profile along perpendicular
            try:
                offsets, profile = sample_profile(gray, c_pt, (v_perp[0], v_perp[1]), half_length=max(20.0, span * 0.6))
                _score, li, ri, _smooth, _gradient = _best_edge_pair(offsets, profile)
                p1 = (c_pt[0] + offsets[li] * v_perp[0], c_pt[1] + offsets[li] * v_perp[1])
                p2 = (c_pt[0] + offsets[ri] * v_perp[0], c_pt[1] + offsets[ri] * v_perp[1])

                val_res = validate_measurement_geometry(
                    p1, p2, width_px=width_img, height_px=height_img, footer_bounds=footer_bounds, min_length_px=min_width_px
                )
                if val_res.valid:
                    w_m = calibration.distance_m(p1, p2)
                    proposed_m.append(ProposedMeasurement(p1=p1, p2=p2, center=c_pt, width_m=w_m))
            except (ValueError, RuntimeError, IndexError):
                pass

        if len(proposed_m) < min(2, n_sections):
            flags.add("PROFILE_FAILED")

        # Compute width stability
        if proposed_m:
            widths = [pm.width_m for pm in proposed_m]
            med_width_m = float(np.median(widths))
            mean_width = float(np.mean(widths))
            std_width = float(np.std(widths)) if len(widths) > 1 else 0.0
            if mean_width > 0 and (std_width / mean_width) > 0.40:
                flags.add("WIDTH_TOO_VARIABLE")
        else:
            med_width_m = None

        # Heuristic confidence score calculation
        if "TOO_SMALL" in flags or "PROFILE_FAILED" in flags:
            confidence = 0.10
        else:
            base_score = min(1.0, elongation / 6.0) * 0.4
            profile_score = (len(proposed_m) / float(n_sections)) * 0.4
            flags_penalty = 0.3 if (flags & {"TOUCHES_ROI_EDGE", "TOUCHES_INVALID_MASK", "WIDTH_TOO_VARIABLE"}) else 1.0
            confidence = max(0.0, min(1.0, (base_score + profile_score + 0.2) * flags_penalty))

        cand_id = f"C{len(candidates) + 1:03d}"
        cand = AutoFiberCandidate(
            candidate_id=cand_id,
            roi_bbox=(x0 + sl[1].start, y0 + sl[0].start, x0 + sl[1].stop, y0 + sl[0].stop),
            component_label=idx,
            centerline_points=centerline_pts,
            proposed_measurements=proposed_m,
            median_width_m=med_width_m,
            confidence_score=confidence,
            quality_flags=flags,
            status="PENDING",
        )
        candidates.append(cand)

        # Update summary counters
        critical_flags = {"TOO_SMALL", "PROFILE_FAILED", "LOW_ELONGATION"}
        if flags & critical_flags:
            excluded_cnt += 1
            for f in flags & critical_flags:
                exclusion_reasons[f] += 1
        else:
            measurable_cnt += 1
            if cand.confidence_level == "Alta":
                high_conf_cnt += 1
            else:
                needs_review_cnt += 1

    summary = AutoROISummary(
        total_components=total_components,
        measurable_candidates=measurable_cnt,
        high_confidence=high_conf_cnt,
        needs_review=needs_review_cnt,
        excluded=excluded_cnt,
        exclusion_reasons=dict(exclusion_reasons),
    )

    return candidates, summary
