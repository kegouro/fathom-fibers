from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage

from .analysis import _best_edge_pair
from .model import Calibration


def compute_line_geometry(
    p1: tuple[float, float],
    p2: tuple[float, float],
    calibration: Calibration,
) -> dict[str, Any]:
    """Calculates physical line length, px length, dx/dy, orientation, and center."""
    dx_px = p2[0] - p1[0]
    dy_px = p2[1] - p1[1]

    dx_m = dx_px * calibration.pixel_size_x_m
    dy_m = dy_px * calibration.pixel_size_y_m

    length_m = calibration.distance_m(p1, p2)
    length_px = math.hypot(dx_px, dy_px)

    angle_rad = math.atan2(dy_m, dx_m)
    angle_deg = math.degrees(angle_rad) % 360.0

    center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

    return {
        "p1": p1,
        "p2": p2,
        "center": center,
        "length_m": length_m,
        "width_m": length_m,
        "length_px": length_px,
        "delta_x_m": dx_m,
        "delta_y_m": dy_m,
        "delta_x_px": dx_px,
        "delta_y_px": dy_px,
        "orientation_deg": angle_deg,
    }


def compute_polyline_geometry(
    points: list[tuple[float, float]],
    calibration: Calibration,
) -> dict[str, Any]:
    """Calculates total physical polyline length, direct endpoint distance, and tortuosity."""
    if len(points) < 2:
        return {
            "points": points,
            "segment_count": 0,
            "total_length_m": 0.0,
            "direct_distance_m": 0.0,
            "tortuosity": None,
            "global_orientation_deg": 0.0,
        }

    total_len_m = 0.0
    for i in range(len(points) - 1):
        total_len_m += calibration.distance_m(points[i], points[i + 1])

    direct_m = calibration.distance_m(points[0], points[-1])

    # Rule 5: Tortuosity = total_length / direct_distance if direct > 0 else None
    tortuosity = (total_len_m / direct_m) if direct_m > 1e-12 else None

    dx_m = (points[-1][0] - points[0][0]) * calibration.pixel_size_x_m
    dy_m = (points[-1][1] - points[0][1]) * calibration.pixel_size_y_m
    global_orientation = math.degrees(math.atan2(dy_m, dx_m)) % 360.0

    return {
        "points": points,
        "segment_count": len(points) - 1,
        "total_length_m": total_len_m,
        "direct_distance_m": direct_m,
        "tortuosity": tortuosity,
        "global_orientation_deg": global_orientation,
    }


def compute_angle_geometry(
    pt_a: tuple[float, float],
    pt_b: tuple[float, float],  # Vertex B
    pt_c: tuple[float, float],
    calibration: Calibration,
) -> dict[str, Any]:
    """Calculates 3-point angle A -> B (vertex) -> C with interior and acute angles."""
    # Vectors in physical space from vertex B
    v_ba_m = (
        (pt_a[0] - pt_b[0]) * calibration.pixel_size_x_m,
        (pt_a[1] - pt_b[1]) * calibration.pixel_size_y_m,
    )
    v_bc_m = (
        (pt_c[0] - pt_b[0]) * calibration.pixel_size_x_m,
        (pt_c[1] - pt_b[1]) * calibration.pixel_size_y_m,
    )

    len_ba = math.hypot(v_ba_m[0], v_ba_m[1])
    len_bc = math.hypot(v_bc_m[0], v_bc_m[1])

    # Rule 6: No angle if any arm length is zero
    if len_ba < 1e-12 or len_bc < 1e-12:
        return {
            "pt_a": pt_a,
            "pt_b": pt_b,
            "pt_c": pt_c,
            "interior_angle_deg": None,
            "acute_angle_deg": None,
            "orientation_ab_deg": None,
            "orientation_bc_deg": None,
        }

    dot = v_ba_m[0] * v_bc_m[0] + v_ba_m[1] * v_bc_m[1]
    cos_theta = max(-1.0, min(1.0, dot / (len_ba * len_bc)))
    interior_deg = math.degrees(math.acos(cos_theta))
    acute_deg = interior_deg if interior_deg <= 90.0 else (180.0 - interior_deg)

    angle_ba = math.degrees(math.atan2(v_ba_m[1], v_ba_m[0])) % 360.0
    angle_bc = math.degrees(math.atan2(v_bc_m[1], v_bc_m[0])) % 360.0

    return {
        "pt_a": pt_a,
        "pt_b": pt_b,
        "pt_c": pt_c,
        "interior_angle_deg": interior_deg,
        "acute_angle_deg": acute_deg,
        "orientation_ab_deg": angle_ba,
        "orientation_bc_deg": angle_bc,
    }


def _roi_mask_from_bbox_or_polygon(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int] | None = None,
    polygon: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    h, w = shape

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        x0_c, x1_c = max(0, min(x0, w)), max(0, min(x1, w))
        y0_c, y1_c = max(0, min(y0, h)), max(0, min(y1, h))
        mask[y0_c:y1_c, x0_c:x1_c] = True

    elif polygon is not None and len(polygon) >= 3:
        # Rasterize polygon into binary mask
        poly_arr = np.array(polygon)
        px_min = np.floor(poly_arr.min(axis=0)).astype(int)
        px_max = np.ceil(poly_arr.max(axis=0)).astype(int)

        x0, y0 = max(0, px_min[0]), max(0, px_min[1])
        x1, y1 = min(w, px_max[0] + 1), min(h, px_max[1] + 1)

        if x1 > x0 and y1 > y0:
            grid_y, grid_x = np.mgrid[y0:y1, x0:x1]
            pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))

            # Point in polygon ray-casting test
            n = len(polygon)
            inside = np.zeros(len(pts), dtype=bool)
            p1x, p1y = polygon[0]
            for i in range(n + 1):
                p2x, p2y = polygon[i % n]
                cond = (pts[:, 1] > min(p1y, p2y)) & (pts[:, 1] <= max(p1y, p2y)) & (pts[:, 0] <= max(p1x, p2x))
                if p1y != p2y:
                    xinters = (pts[:, 1] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                cond &= (p1x == p2x) | (pts[:, 0] <= xinters)
                inside ^= cond
                p1x, p1y = p2x, p2y

            mask[y0:y1, x0:x1] = inside.reshape((y1 - y0, x1 - x0))

    return mask


def compute_area_roi_geometry(
    gray: np.ndarray,
    calibration: Calibration,
    bbox: tuple[int, int, int, int] | None = None,
    polygon: list[tuple[float, float]] | None = None,
    footer_bounds: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Calculates physical area (m²), perimeter, centroid, and intensity statistics on valid pixels."""
    h, w = gray.shape[:2]
    raw_mask = _roi_mask_from_bbox_or_polygon((h, w), bbox=bbox, polygon=polygon)

    # Footer exclusion
    valid_mask = raw_mask.copy()
    if footer_bounds is not None:
        fy0, fy1 = footer_bounds
        valid_mask[fy0:fy1, :] = False

    raw_pixel_count = int(raw_mask.sum())
    valid_pixel_count = int(valid_mask.sum())
    excluded_pixel_count = raw_pixel_count - valid_pixel_count
    excluded_fraction = (excluded_pixel_count / max(raw_pixel_count, 1)) if raw_pixel_count > 0 else 0.0

    px_w_m = calibration.pixel_size_x_m
    px_h_m = calibration.pixel_size_y_m
    pixel_area_m2 = px_w_m * px_h_m

    area_m2 = valid_pixel_count * pixel_area_m2

    if valid_pixel_count > 0:
        ys, xs = np.where(valid_mask)
        cx_px = float(xs.mean())
        cy_px = float(ys.mean())
        cx_m = cx_px * px_w_m
        cy_m = cy_px * px_h_m
        centroid = (cx_px, cy_px)

        vals = gray[valid_mask].astype(float)
        mean_intensity = float(vals.mean())
        std_intensity = float(vals.std())
        min_intensity = float(vals.min())
        max_intensity = float(vals.max())
    else:
        centroid = (0.0, 0.0)
        cx_m, cy_m = 0.0, 0.0
        mean_intensity, std_intensity, min_intensity, max_intensity = 0.0, 0.0, 0.0, 0.0

    # Perimeter estimation in physical units
    if valid_pixel_count > 0:
        eroded = ndimage.binary_erosion(valid_mask)
        boundary = valid_mask & (~eroded)
        _bys, bxs = np.where(boundary)
        perimeter_m = len(bxs) * math.sqrt(px_w_m**2 + px_h_m**2) / 1.414
    else:
        perimeter_m = 0.0

    return {
        "bbox": bbox,
        "polygon": polygon,
        "area_m2": area_m2,
        "perimeter_m": perimeter_m,
        "centroid_px": centroid,
        "centroid_m": (cx_m, cy_m),
        "mean_intensity_au": mean_intensity,
        "std_intensity_au": std_intensity,
        "min_intensity_au": min_intensity,
        "max_intensity_au": max_intensity,
        "valid_pixel_count": valid_pixel_count,
        "excluded_pixel_count": excluded_pixel_count,
        "excluded_fraction": excluded_fraction,
    }


def compute_profile_geometry(
    gray: np.ndarray,
    p1: tuple[float, float],
    p2: tuple[float, float],
    calibration: Calibration,
    bandwidth_px: int = 1,
) -> dict[str, Any]:
    """Extracts intensity profile along line p1->p2 with optional perpendicular bandwidth averaging."""
    length_m = calibration.distance_m(p1, p2)
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dist_px = math.hypot(dx, dy)

    if dist_px < 1e-6:
        return {
            "p1": p1,
            "p2": p2,
            "length_m": 0.0,
            "samples_count": 0,
            "bandwidth_px": bandwidth_px,
            "profile_raw": [],
            "profile_smoothed": [],
            "distance_m": [],
            "suggested_width_m": None,
        }

    n_samples = max(2, round(dist_px))
    t = np.linspace(0.0, 1.0, n_samples)
    xs = p1[0] + t * dx
    ys = p1[1] + t * dy

    v_dir = np.array([dx / dist_px, dy / dist_px])
    v_perp = np.array([-v_dir[1], v_dir[0]])

    # Sample perpendicular band
    half_bw = bandwidth_px // 2
    bw_offsets = range(-half_bw, half_bw + 1) if bandwidth_px > 1 else [0]

    profile_matrix = []
    for off in bw_offsets:
        xs_off = xs + off * v_perp[0]
        ys_off = ys + off * v_perp[1]
        prof_off = ndimage.map_coordinates(gray, [ys_off, xs_off], order=1, mode="nearest")
        profile_matrix.append(prof_off)

    prof_matrix_arr = np.array(profile_matrix)
    profile_raw = prof_matrix_arr.mean(axis=0)

    # Sub-pixel distance vector in physical meters
    dist_m_vec = [float(ti * length_m) for ti in t]

    # Smooth profile
    profile_smoothed = ndimage.gaussian_filter1d(profile_raw, sigma=1.5)

    # Suggested edge pair detection along profile
    suggested_width_m = None
    try:
        offsets_px = np.linspace(-dist_px / 2.0, dist_px / 2.0, n_samples)
        _score, li, ri, _s, _g = _best_edge_pair(offsets_px, profile_raw)
        edge_px = abs(offsets_px[ri] - offsets_px[li])
        suggested_width_m = edge_px * calibration.pixel_size_x_m
    except (ValueError, KeyError, IndexError, RuntimeError):
        pass

    return {
        "p1": p1,
        "p2": p2,
        "length_m": length_m,
        "samples_count": n_samples,
        "bandwidth_px": bandwidth_px,
        "profile_raw": [float(v) for v in profile_raw],
        "profile_smoothed": [float(v) for v in profile_smoothed],
        "distance_m": dist_m_vec,
        "min_intensity": float(profile_raw.min()),
        "max_intensity": float(profile_raw.max()),
        "mean_intensity": float(profile_raw.mean()),
        "suggested_width_m": suggested_width_m,
    }
