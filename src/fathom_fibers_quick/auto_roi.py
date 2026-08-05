from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .analysis import _best_edge_pair, sample_profile, validate_measurement_geometry
from .model import Calibration, Measurement


@dataclass(slots=True)
class ResolutionPreset:
    name: str
    nm_per_px_min: float
    nm_per_px_max: float
    expected_width_px: float
    min_area_px: int
    min_elongation: float
    min_width_px: float
    n_sections: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nm_per_px_min": self.nm_per_px_min,
            "nm_per_px_max": self.nm_per_px_max,
            "expected_width_px": self.expected_width_px,
            "min_area_px": self.min_area_px,
            "min_elongation": self.min_elongation,
            "min_width_px": self.min_width_px,
            "n_sections": self.n_sections,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolutionPreset:
        return cls(
            name=data["name"],
            nm_per_px_min=float(data["nm_per_px_min"]),
            nm_per_px_max=float(data["nm_per_px_max"]),
            expected_width_px=float(data["expected_width_px"]),
            min_area_px=int(data["min_area_px"]),
            min_elongation=float(data["min_elongation"]),
            min_width_px=float(data["min_width_px"]),
            n_sections=int(data["n_sections"]),
            description=data.get("description", ""),
        )


PRESET_HIGH_MAG_FINE = ResolutionPreset(
    name="HIGH_MAG_FINE",
    nm_per_px_min=0.0,
    nm_per_px_max=10.0,
    expected_width_px=30.0,
    min_area_px=80,
    min_elongation=2.5,
    min_width_px=5.0,
    n_sections=5,
    description="Alta magnificación (≤10 nm/px). Medición precisa de fibras individuales y bordes.",
)

PRESET_MID_MAG_GENERAL = ResolutionPreset(
    name="MID_MAG_GENERAL",
    nm_per_px_min=10.0,
    nm_per_px_max=80.0,
    expected_width_px=12.0,
    min_area_px=40,
    min_elongation=2.2,
    min_width_px=3.0,
    n_sections=3,
    description="Magnificación media (10-80 nm/px). Segmentos visibles y candidatos revisables.",
)

PRESET_LOW_MAG_NETWORK = ResolutionPreset(
    name="LOW_MAG_NETWORK",
    nm_per_px_min=80.0,
    nm_per_px_max=1e9,
    expected_width_px=3.0,
    min_area_px=20,
    min_elongation=2.0,
    min_width_px=1.5,
    n_sections=3,
    description="Baja magnificación (>80 nm/px). Redes densas / fibras subresueltas. Solo medición manual.",
)


def get_preset_for_calibration(calibration: Calibration) -> ResolutionPreset:
    nm_px = calibration.pixel_size_x_m * 1e9
    if nm_px <= 10.0:
        return PRESET_HIGH_MAG_FINE
    elif nm_px <= 80.0:
        return PRESET_MID_MAG_GENERAL
    else:
        return PRESET_LOW_MAG_NETWORK


def check_resolution_resolvability(
    roi_patch: np.ndarray,
    calibration: Calibration,
    expected_width_m: float = 300e-9,  # Default ~300 nm PVDF electrospun fibers
) -> tuple[str, str]:
    """Estimates resolvability before ROI analysis based on physical scale and edge gradients."""
    nm_per_px = calibration.pixel_size_x_m * 1e9
    expected_px = expected_width_m / calibration.pixel_size_x_m

    gy, gx = np.gradient(roi_patch.astype(float))
    grad_mag = float(np.max(np.hypot(gx, gy))) if roi_patch.size > 0 else 0.0

    if expected_px < 3.5 or nm_per_px > 85.0:
        return (
            "RESOLUTION_INSUFFICIENT",
            f"El ancho físico esperado (~{expected_width_m * 1e9:.0f} nm) equivale a solo {expected_px:.1f} px a {nm_per_px:.2f} nm/px. Resolución insuficiente para diámetros automáticos confiables. Usar magnificación mayor o caliper manual.",
        )
    elif expected_px < 7.0 or grad_mag < 1.0:
        return (
            "RESOLUTION_MARGINAL",
            f"El ancho esperado equivale a {expected_px:.1f} px ({nm_per_px:.2f} nm/px). Resolución marginal; requiere revisión manual estricta de candidatos.",
        )
    else:
        return (
            "RESOLUTION_OK",
            f"Resolución adecuada ({expected_px:.1f} px por fibra a {nm_per_px:.2f} nm/px).",
        )


@dataclass(slots=True)
class ProposedMeasurement:
    p1: tuple[float, float]
    p2: tuple[float, float]
    center: tuple[float, float]
    width_m: float
    mask_width_m: float | None = None
    profile_width_m: float | None = None
    discrepancy: float | None = None
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
    threshold_method: str = "Otsu"
    curved_trace_used: bool = False
    preset_name: str = "MID_MAG_GENERAL"

    @property
    def confidence_level(self) -> str:
        """Strict confidence classification enforcing Requirements 9 & 10."""
        disqualifying_flags = {
            "TOUCHES_ROI_EDGE",
            "TOUCHES_INVALID_MASK",
            "TOO_SMALL",
            "LOW_ELONGATION",
            "LIKELY_MERGED",
            "WIDTH_TOO_VARIABLE",
            "WIDTH_ESTIMATORS_DISAGREE",
            "PROFILE_FAILED",
            "RESOLUTION_MARGINAL",
            "RESOLUTION_INSUFFICIENT",
            "CURVED_TRACE_UNSTABLE",
        }
        if self.quality_flags & disqualifying_flags:
            return "Baja"

        if self.preset_name == "LOW_MAG_NETWORK":
            return "Baja"

        if self.confidence_score >= 0.70 and len(self.quality_flags) == 0:
            return "Alta"
        elif self.confidence_score >= 0.40:
            return "Media"
        return "Baja"

    @property
    def cv_width(self) -> float | None:
        if not self.proposed_measurements:
            return None
        widths = [pm.width_m for pm in self.proposed_measurements]
        mean = float(np.mean(widths))
        return float(np.std(widths)) / mean if mean > 0 else 0.0

    @property
    def mean_discrepancy(self) -> float | None:
        discs = [pm.discrepancy for pm in self.proposed_measurements if pm.discrepancy is not None]
        return float(np.mean(discs)) if discs else None


@dataclass(slots=True)
class AutoROISummary:
    total_components: int
    measurable_candidates: int
    high_confidence: int
    needs_review: int
    excluded: int
    threshold_method_used: str = "Otsu"
    resolution_status: str = "RESOLUTION_OK"
    exclusion_reasons: dict[str, int] = field(default_factory=dict)


# ---------- Thresholding Strategies with Diagnostic Scoring ----------

def otsu_threshold(gray_patch: np.ndarray) -> float:
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


def percentile_threshold(norm_patch: np.ndarray, percentile: float = 65.0) -> float:
    return float(np.percentile(norm_patch, percentile))


def local_adaptive_threshold(norm_patch: np.ndarray, window_size: int = 35, offset: float = 0.03) -> np.ndarray:
    local_mean = ndimage.uniform_filter(norm_patch, size=window_size)
    return norm_patch > (local_mean + offset)


def evaluate_segmentation_quality(binary_mask: np.ndarray) -> float:
    """Computes diagnostic quality score for a segmentation mask on SEM image."""
    frac = float(binary_mask.mean())
    if frac < 0.01 or frac > 0.60:
        return 0.05

    labeled, num_features = ndimage.label(binary_mask)
    if num_features == 0:
        return 0.0

    counts = np.bincount(labeled.ravel())[1:]  # Exclude background
    max_comp_frac = float(counts.max()) / float(binary_mask.sum())

    # Heavy penalty if one giant merged component covers > 40% of foreground
    penalty = 1.0
    if max_comp_frac > 0.40:
        penalty *= 0.20
    if num_features > 150:
        penalty *= 0.30

    base_score = (1.0 - abs(frac - 0.25) * 2.0) * penalty
    return max(0.01, float(base_score))


def auto_threshold(norm_patch: np.ndarray, is_bright: bool) -> tuple[np.ndarray, str]:
    """Evaluates multiple segmentation methods and picks best strategy via quality score."""
    # Method 1: Global Otsu
    t_otsu = otsu_threshold(norm_patch * 255.0) / 255.0
    bin_otsu = norm_patch > t_otsu if is_bright else norm_patch < t_otsu
    score_otsu = evaluate_segmentation_quality(bin_otsu)

    # Method 2: Robust Percentile
    t_perc = percentile_threshold(norm_patch, 65.0)
    bin_perc = norm_patch > t_perc if is_bright else norm_patch < t_perc
    score_perc = evaluate_segmentation_quality(bin_perc)

    # Method 3: Local Adaptive
    if is_bright:
        bin_local = local_adaptive_threshold(norm_patch, window_size=35, offset=0.03)
    else:
        bin_local = norm_patch < (ndimage.uniform_filter(norm_patch, size=35) - 0.03)
    score_local = evaluate_segmentation_quality(bin_local)

    candidates = [
        (score_local, bin_local, "Local Adaptativo"),
        (score_otsu, bin_otsu, "Otsu Global"),
        (score_perc, bin_perc, "Percentil Robusto"),
    ]
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


# ---------- Curved Tracing Helper ----------

def trace_curved_centerline(
    abs_xs: np.ndarray,
    abs_ys: np.ndarray,
    v_dir: np.ndarray,
    center_abs: tuple[float, float],
    n_slices: int = 9,
) -> tuple[list[tuple[float, float]], list[np.ndarray], bool]:
    pts_rel = np.column_stack((abs_xs - center_abs[0], abs_ys - center_abs[1]))
    projections = pts_rel @ v_dir
    t_min, t_max = float(projections.min()), float(projections.max())
    span = t_max - t_min

    if span < 18.0 or len(abs_xs) < 40:
        return [], [], False

    slice_edges = np.linspace(t_min + 0.10 * span, t_max - 0.10 * span, n_slices + 1)
    centroids: list[tuple[float, float]] = []

    for i in range(n_slices):
        mask = (projections >= slice_edges[i]) & (projections < slice_edges[i + 1])
        if mask.sum() >= 3:
            centroids.append((float(abs_xs[mask].mean()), float(abs_ys[mask].mean())))

    if len(centroids) < 3:
        return [], [], False

    c_arr = np.array(centroids)
    smoothed_x = ndimage.gaussian_filter1d(c_arr[:, 0], sigma=1.0)
    smoothed_y = ndimage.gaussian_filter1d(c_arr[:, 1], sigma=1.0)

    centerline_pts = [(float(x), float(y)) for x, y in zip(smoothed_x, smoothed_y, strict=True)]

    tangents: list[np.ndarray] = []
    for i in range(len(centerline_pts)):
        if i == 0:
            dx = centerline_pts[1][0] - centerline_pts[0][0]
            dy = centerline_pts[1][1] - centerline_pts[0][1]
        elif i == len(centerline_pts) - 1:
            dx = centerline_pts[-1][0] - centerline_pts[-2][0]
            dy = centerline_pts[-1][1] - centerline_pts[-2][1]
        else:
            dx = centerline_pts[i + 1][0] - centerline_pts[i - 1][0]
            dy = centerline_pts[i + 1][1] - centerline_pts[i - 1][1]

        norm = math.hypot(dx, dy)
        if norm > 0:
            tangents.append(np.array([dx / norm, dy / norm]))
        else:
            tangents.append(v_dir)

    return centerline_pts, tangents, True


# ---------- Core Analysis Function ----------

def analyze_roi(
    gray: np.ndarray,
    roi_bbox: tuple[int, int, int, int],  # (x0, y0, x1, y1)
    calibration: Calibration,
    footer_bounds: tuple[int, int] | None = None,
    polarity: str = "auto",  # "auto", "bright", "dark"
    threshold_method: str = "Otsu",  # "Otsu", "Percentil", "Local Adaptativo", "Automático"
    preset: ResolutionPreset | None = None,
    min_area_px: int | None = None,
    min_elongation: float | None = None,
    min_width_px: float | None = None,
    n_sections: int | None = None,
    allow_curved_trace: bool = True,
) -> tuple[list[AutoFiberCandidate], AutoROISummary]:
    """Extracts simple fiber candidates from an image ROI using connected component analysis."""
    if preset is None:
        preset = get_preset_for_calibration(calibration)

    area_limit = min_area_px if min_area_px is not None else preset.min_area_px
    elong_limit = min_elongation if min_elongation is not None else preset.min_elongation
    width_limit = min_width_px if min_width_px is not None else preset.min_width_px
    sections_count = n_sections if n_sections is not None else preset.n_sections

    x0, y0, x1, y1 = roi_bbox
    height_img, width_img = gray.shape[:2]

    # Strict footer exclusion on y1
    if footer_bounds is not None:
        y1 = min(y1, footer_bounds[0])

    # Clamp bbox within image bounds
    x0 = max(0, min(x0, width_img - 1))
    x1 = max(x0 + 1, min(x1, width_img))
    y0 = max(0, min(y0, height_img - 1))
    y1 = max(y0 + 1, min(y1, height_img))

    roi_patch = gray[y0:y1, x0:x1]
    if roi_patch.size < 20:
        return [], AutoROISummary(0, 0, 0, 0, 0, threshold_method_used=threshold_method)

    res_status, _res_msg = check_resolution_resolvability(roi_patch, calibration)

    # Robust normalization
    p2, p98 = np.percentile(roi_patch, (2, 98))
    norm_patch = np.clip((roi_patch - p2) / max(p98 - p2, 1e-5), 0.0, 1.0)

    # Determine polarity
    if polarity == "auto":
        border_mask = np.ones_like(roi_patch, dtype=bool)
        border_mask[1:-1, 1:-1] = False
        is_bright = norm_patch[~border_mask].mean() > norm_patch[border_mask].mean()
    else:
        is_bright = (polarity == "bright")

    # Thresholding
    chosen_method_name = threshold_method
    if threshold_method == "Otsu":
        t_val = otsu_threshold(norm_patch * 255.0) / 255.0
        binary = norm_patch > t_val if is_bright else norm_patch < t_val
    elif threshold_method == "Percentil":
        t_val = percentile_threshold(norm_patch, 65.0)
        binary = norm_patch > t_val if is_bright else norm_patch < t_val
    elif threshold_method == "Local Adaptativo":
        binary = local_adaptive_threshold(norm_patch, window_size=35, offset=0.03)
        if not is_bright:
            binary = ~binary
    else:
        binary, chosen_method_name = auto_threshold(norm_patch, is_bright)

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
        abs_xs = x0 + sl[1].start + xs
        abs_ys = y0 + sl[0].start + ys

        flags: set[str] = set()

        if res_status != "RESOLUTION_OK":
            flags.add(res_status)

        if area_px < area_limit:
            flags.add("TOO_SMALL")

        if (
            sl[1].start == 0
            or sl[0].start == 0
            or sl[1].stop == (x1 - x0)
            or sl[0].stop == (y1 - y0)
        ):
            flags.add("TOUCHES_ROI_EDGE")

        if footer_bounds is not None:
            fy0, fy1 = footer_bounds
            if np.any((abs_ys >= fy0) & (abs_ys <= fy1)):
                flags.add("TOUCHES_INVALID_MASK")

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

        if elongation < elong_limit:
            flags.add("LOW_ELONGATION")

        v_phys = evecs[:, 0]
        v_img = np.array([v_phys[0] / px_w_m, v_phys[1] / px_h_m], dtype=float)
        v_norm = np.linalg.norm(v_img)
        v_dir = v_img / v_norm if v_norm > 0 else np.array([1.0, 0.0])

        center_abs = (float(abs_xs.mean()), float(abs_ys.mean()))
        pts_rel = np.column_stack((abs_xs - center_abs[0], abs_ys - center_abs[1]))
        projections = pts_rel @ v_dir

        t_min, t_max = float(projections.min()), float(projections.max())
        span = t_max - t_min

        # LIKELY_MERGED Check
        expected_width_px = preset.expected_width_px
        global_width_est_px = (area_px / max(span, 1.0))
        if global_width_est_px > 2.8 * expected_width_px or (area_px > 450 and elongation < 2.5):
            flags.add("LIKELY_MERGED")

        curved_used = False
        centerline_pts: list[tuple[float, float]] = []
        tangents: list[np.ndarray] = []

        if allow_curved_trace and "LIKELY_MERGED" not in flags and span >= 20.0:
            c_pts, t_vecs, ok = trace_curved_centerline(abs_xs, abs_ys, v_dir, center_abs, n_slices=sections_count)
            if ok:
                centerline_pts = c_pts
                tangents = t_vecs
                curved_used = True
                flags.add("CURVED_TRACE_USED")

        if not curved_used:
            t_start = t_min + 0.15 * span
            t_end = t_max - 0.15 * span
            t_centers = np.linspace(t_start, t_end, sections_count)
            for tc in t_centers:
                c_pt = (center_abs[0] + tc * v_dir[0], center_abs[1] + tc * v_dir[1])
                centerline_pts.append(c_pt)
                tangents.append(v_dir)

        proposed_m: list[ProposedMeasurement] = []

        for c_pt, t_dir in zip(centerline_pts, tangents, strict=True):
            p_dir = np.array([-t_dir[1], t_dir[0]], dtype=float)
            try:
                offsets, profile = sample_profile(gray, c_pt, (p_dir[0], p_dir[1]), half_length=max(20.0, span * 0.6))
                _score, li, ri, _smooth, _gradient = _best_edge_pair(offsets, profile)
                p1 = (c_pt[0] + offsets[li] * p_dir[0], c_pt[1] + offsets[li] * p_dir[1])
                p2 = (c_pt[0] + offsets[ri] * p_dir[0], c_pt[1] + offsets[ri] * p_dir[1])

                val_res = validate_measurement_geometry(
                    p1, p2, width_px=width_img, height_px=height_img, footer_bounds=footer_bounds, min_length_px=width_limit
                )
                if val_res.valid:
                    profile_w_m = calibration.distance_m(p1, p2)

                    mask_w_px = float(offsets[ri] - offsets[li])
                    mask_w_m = mask_w_px * calibration.pixel_size_x_m

                    discrepancy = abs(mask_w_m - profile_w_m) / max(profile_w_m, 1e-12)
                    sec_flags: set[str] = set()
                    if discrepancy >= 0.20:
                        sec_flags.add("WIDTH_ESTIMATORS_DISAGREE")
                        flags.add("WIDTH_ESTIMATORS_DISAGREE")
                    else:
                        sec_flags.add("WIDTH_ESTIMATORS_AGREE")

                    proposed_m.append(
                        ProposedMeasurement(
                            p1=p1,
                            p2=p2,
                            center=c_pt,
                            width_m=profile_w_m,
                            mask_width_m=mask_w_m,
                            profile_width_m=profile_w_m,
                            discrepancy=discrepancy,
                            quality_flags=sec_flags,
                        )
                    )
            except (ValueError, RuntimeError, IndexError):
                pass

        if len(proposed_m) < min(2, sections_count):
            flags.add("PROFILE_FAILED")

        if proposed_m:
            widths = [pm.width_m for pm in proposed_m]
            med_width_m = float(np.median(widths))
            mean_width = float(np.mean(widths))
            std_width = float(np.std(widths)) if len(widths) > 1 else 0.0
            if mean_width > 0 and (std_width / mean_width) > 0.40:
                flags.add("WIDTH_TOO_VARIABLE")
        else:
            med_width_m = None

        if "TOO_SMALL" in flags or "PROFILE_FAILED" in flags or "LIKELY_MERGED" in flags:
            confidence = 0.10
        else:
            base_score = min(1.0, elongation / 6.0) * 0.4
            profile_score = (len(proposed_m) / float(sections_count)) * 0.4
            flags_penalty = 0.3 if (flags & {"TOUCHES_ROI_EDGE", "TOUCHES_INVALID_MASK", "WIDTH_TOO_VARIABLE", "WIDTH_ESTIMATORS_DISAGREE"}) else 1.0
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
            threshold_method=chosen_method_name,
            curved_trace_used=curved_used,
            preset_name=preset.name,
        )
        candidates.append(cand)

        critical_flags = {"TOO_SMALL", "PROFILE_FAILED", "LOW_ELONGATION", "LIKELY_MERGED"}
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
        threshold_method_used=chosen_method_name,
        resolution_status=res_status,
        exclusion_reasons=dict(exclusion_reasons),
    )

    return candidates, summary


# ---------- Diagnostic ROIs & Comparisons ----------

def generate_diagnostic_rois(
    image_shape: tuple[int, int],  # (height, width)
    footer_bounds: tuple[int, int] | None = None,
    n_rois: int = 4,
    roi_size: int = 500,
) -> list[tuple[int, int, int, int]]:
    """Generates 4 diagnostic ROIs strictly avoiding footers and edges."""
    height, width = image_shape
    usable_h = footer_bounds[0] if footer_bounds else height
    margin = 50

    if width < roi_size + 2 * margin or usable_h < roi_size + 2 * margin:
        cx, cy = width // 2, usable_h // 2
        half = min(width, usable_h) // 3
        return [(max(0, cx - half), max(0, cy - half), min(width, cx + half), min(usable_h, cy + half))]

    rois: list[tuple[int, int, int, int]] = []
    # Center ROI
    cx, cy = width // 2, usable_h // 2
    r_half = roi_size // 2
    rois.append((cx - r_half, cy - r_half, cx + r_half, cy + r_half))

    # Top-Left ROI
    q_size = min(roi_size, (width - 3 * margin) // 2, (usable_h - 3 * margin) // 2)
    if q_size > 100:
        rois.append((margin, margin, margin + q_size, margin + q_size))
        # Top-Right ROI
        rois.append((width - margin - q_size, margin, width - margin, margin + q_size))
        # Mid-Bottom ROI (strictly above footer)
        b_y1 = usable_h - margin
        b_y0 = b_y1 - q_size
        rois.append((cx - q_size // 2, b_y0, cx + q_size // 2, b_y1))

    return rois[:n_rois]


def compare_candidate_to_manual(candidate: AutoFiberCandidate, manual: Measurement) -> dict[str, float]:
    cand_w = candidate.median_width_m or 0.0
    man_w = manual.width_m
    abs_diff = abs(cand_w - man_w)
    rel_diff = (abs_diff / max(man_w, 1e-12)) * 100.0

    c_center = candidate.proposed_measurements[0].center if candidate.proposed_measurements else (0.0, 0.0)
    m_center = manual.center
    center_dist = math.hypot(c_center[0] - m_center[0], c_center[1] - m_center[1])

    return {
        "candidate_width_um": cand_w * 1e6,
        "manual_width_um": man_w * 1e6,
        "abs_diff_um": abs_diff * 1e6,
        "rel_diff_pct": rel_diff,
        "center_dist_px": center_dist,
    }


def generate_diagnostic_panel(
    gray: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    candidates: list[AutoFiberCandidate],
    summary: AutoROISummary,
    title_info: str = "",
) -> Image.Image:
    x0, y0, x1, y1 = roi_bbox
    crop = gray[y0:y1, x0:x1]
    h, w = crop.shape[:2]

    panel_w, panel_h = max(250, w), max(250, h)

    orig_img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((panel_w, panel_h))
    p2, p98 = np.percentile(crop, (2, 98))
    norm_crop = np.clip((crop - p2) / max(p98 - p2, 1e-5) * 255.0, 0, 255).astype(np.uint8)
    norm_img = Image.fromarray(norm_crop).resize((panel_w, panel_h))

    overlay = orig_img.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=12)

    scale_x = panel_w / float(w)
    scale_y = panel_h / float(h)

    for cand in candidates:
        color = (0, 229, 255) if cand.confidence_level != "Baja" else (150, 150, 150)
        if cand.status == "ACCEPTED":
            color = (0, 255, 127)

        for pm in cand.proposed_measurements:
            p1c = ((pm.p1[0] - x0) * scale_x, (pm.p1[1] - y0) * scale_y)
            p2c = ((pm.p2[0] - x0) * scale_x, (pm.p2[1] - y0) * scale_y)
            draw.line([p1c, p2c], fill=color, width=2)

    combined = Image.new("RGB", (panel_w * 3, panel_h + 60), color=(30, 30, 30))
    combined.paste(orig_img.convert("RGB"), (0, 60))
    combined.paste(norm_img.convert("RGB"), (panel_w, 60))
    combined.paste(overlay, (panel_w * 2, 60))

    header_draw = ImageDraw.Draw(combined)
    hdr_text = (
        f"Fathom Auto-ROI Diagnostic | {title_info}\n"
        f"Threshold: {summary.threshold_method_used} | Res: {summary.resolution_status} | "
        f"Comp: {summary.total_components} | Medibles: {summary.measurable_candidates} | Alta Conf: {summary.high_confidence}"
    )
    header_draw.text((10, 8), hdr_text, fill=(255, 255, 255), font=font)

    return combined
