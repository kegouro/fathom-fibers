"""Deterministic classical image field used by ``FATHOM_FIELD_GRAPH_V1``.

This module deliberately stops before instance or graph reconstruction.  It
observes a supplied fibre mask, estimates a local *axis* field from the raw
image, and measures mask-derived EDT radii at a documented skeleton baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import skeletonize


def double_angle_orientation(theta: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Return the pi-periodic representation of an unoriented axis.

    Rounding defines a stable field serialization convention and removes the
    otherwise unavoidable last-bit difference between ``theta`` and
    ``theta + pi`` in libm trigonometry.
    """
    angle = np.asarray(theta, dtype=float)
    return np.round(np.cos(2.0 * angle), 12), np.round(np.sin(2.0 * angle), 12)


@dataclass(frozen=True, slots=True)
class FiberFieldResult:
    """Field output in pixel and calibrated physical coordinates.

    ``centerline`` is only a sampling support in V1.  It is not a claim that
    fibres, crossings, or topology have been reconstructed.
    """

    fiber_probability: np.ndarray | None = None
    orientation_qx: np.ndarray | None = None
    orientation_qy: np.ndarray | None = None
    coherence: np.ndarray | None = None
    radius_px: np.ndarray | None = None
    diameter_px: np.ndarray | None = None
    radius_m: np.ndarray | None = None
    diameter_m: np.ndarray | None = None
    centerline: np.ndarray | None = None
    crossing_probability: np.ndarray | None = None
    confidence: np.ndarray | None = None
    binary_mask: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def radius_proposal_px(self) -> np.ndarray | None:
        """Compatibility alias for the initial future-backend contract."""
        return self.radius_px


@dataclass(frozen=True, slots=True)
class BoundaryContour:
    """A subpixel boundary representation in row/column and physical x/y."""

    row_col_px: np.ndarray
    xy_m: np.ndarray
    tangent_xy: np.ndarray
    inward_normal_xy: np.ndarray
    inward_ambiguous: np.ndarray


@dataclass(frozen=True, slots=True)
class PairedEdgeSamples:
    """Centerline-anchored local paired-edge observations in physical units."""

    center_xy_m: np.ndarray
    normal_xy: np.ndarray
    minus_xy_m: np.ndarray
    plus_xy_m: np.ndarray
    radius_minus_m: np.ndarray
    radius_plus_m: np.ndarray
    diameter_m: np.ndarray
    asymmetry: np.ndarray
    coherence: np.ndarray
    edt_diameter_m: np.ndarray
    tangent_alignment: np.ndarray
    boundary_normal_consistency: np.ndarray
    accepted: np.ndarray
    flags: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class OrientedBoundaryEngine:
    """Subpixel contours plus centerline-normal paired-edge measurements.

    Pairing is deliberately local.  It neither labels sides globally nor
    reconstructs fibre instances or a graph.
    """

    tangent_window: int = 2
    ray_samples: int = 64
    low_coherence: float = 0.15
    high_asymmetry: float = 0.5
    minimum_tangent_alignment: float = 0.5

    def extract_contours(
        self, mask: np.ndarray, *, pixel_size_xy_m: tuple[float, float]
    ) -> tuple[BoundaryContour, ...]:
        binary = np.asarray(mask, dtype=bool)
        px, py = pixel_size_xy_m
        contours: list[BoundaryContour] = []
        for row_col in find_contours(binary.astype(float), 0.5):
            if len(row_col) < 3:
                continue
            step = min(self.tangent_window, max(1, len(row_col) // 4))
            before = np.roll(row_col, step, axis=0)
            after = np.roll(row_col, -step, axis=0)
            tangent = np.column_stack(((after[:, 1] - before[:, 1]) * px, (after[:, 0] - before[:, 0]) * py))
            length = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = np.divide(tangent, length, out=np.zeros_like(tangent), where=length > 0)
            normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
            eps_m = 0.35 * min(px, py)
            # Probe both normal directions in mask pixel coordinates.
            x_plus = row_col[:, 1] + normal[:, 0] * eps_m / px
            y_plus = row_col[:, 0] + normal[:, 1] * eps_m / py
            x_minus = row_col[:, 1] - normal[:, 0] * eps_m / px
            y_minus = row_col[:, 0] - normal[:, 1] * eps_m / py
            interpolated = binary.astype(float)
            plus = ndimage.map_coordinates(interpolated, [y_plus, x_plus], order=1, mode="constant", cval=0.0) > 0.5
            minus = ndimage.map_coordinates(interpolated, [y_minus, x_minus], order=1, mode="constant", cval=0.0) > 0.5
            inward = np.where(plus[:, None], normal, -normal)
            ambiguous = plus == minus
            inward[ambiguous] = 0.0
            contours.append(BoundaryContour(
                row_col, np.column_stack((row_col[:, 1] * px, row_col[:, 0] * py)), tangent, inward, ambiguous
            ))
        return tuple(contours)

    def pair_centerline(
        self,
        *,
        mask: np.ndarray,
        centerline: np.ndarray,
        orientation_qx: np.ndarray,
        orientation_qy: np.ndarray,
        coherence: np.ndarray,
        edt_diameter_m: np.ndarray,
        pixel_size_xy_m: tuple[float, float],
        contours: tuple[BoundaryContour, ...] = (),
    ) -> PairedEdgeSamples:
        binary = np.asarray(mask, dtype=bool)
        line = np.asarray(centerline, dtype=bool)
        px, py = pixel_size_xy_m
        rows, cols = np.nonzero(line)
        count = len(rows)
        center = np.column_stack((cols * px, rows * py)).astype(float)
        qx, qy = orientation_qx[line], orientation_qy[line]
        theta = 0.5 * np.arctan2(qy, qx)
        normal = np.column_stack((-np.sin(theta), np.cos(theta)))
        edt = np.asarray(edt_diameter_m, float)[line]
        coh = np.asarray(coherence, float)[line]
        minus = np.full((count, 2), np.nan)
        plus = np.full((count, 2), np.nan)
        r_minus = np.full(count, np.nan)
        r_plus = np.full(count, np.nan)
        tangent_alignment = np.full(count, np.nan)
        normal_consistency = np.full(count, np.nan)
        all_flags: list[tuple[str, ...]] = []
        # A fixed normalized ray grid avoids orientation-dependent pixel counts;
        # threshold crossings are linearly interpolated in physical space.
        u = np.linspace(0.0, 1.0, self.ray_samples)
        for start in range(0, count, 2048):
            stop = min(start + 2048, count)
            local_center, local_normal = center[start:stop], normal[start:stop]
            max_radius = np.maximum(1.5 * edt[start:stop], 2.0 * min(px, py))
            distance = max_radius[:, None] * u[None, :]
            for sign, target in ((-1.0, "minus"), (1.0, "plus")):
                xy = local_center[:, None, :] + sign * distance[:, :, None] * local_normal[:, None, :]
                values = ndimage.map_coordinates(
                    binary.astype(float), [xy[..., 1].ravel() / py, xy[..., 0].ravel() / px],
                    order=1, mode="constant", cval=0.0,
                ).reshape(stop - start, self.ray_samples)
                outside = values < 0.5
                has = outside.any(axis=1)
                index = np.argmax(outside, axis=1)
                previous = np.maximum(index - 1, 0)
                row_index = np.arange(stop - start)
                inner, outer = values[row_index, previous], values[row_index, index]
                du = distance[row_index, index] - distance[row_index, previous]
                fraction = np.divide(inner - 0.5, inner - outer, out=np.zeros_like(inner), where=inner != outer)
                hit_distance = distance[row_index, previous] + np.clip(fraction, 0.0, 1.0) * du
                hit_distance[~has] = np.nan
                hit = local_center + sign * hit_distance[:, None] * local_normal
                if target == "minus":
                    r_minus[start:stop], minus[start:stop] = hit_distance, hit
                else:
                    r_plus[start:stop], plus[start:stop] = hit_distance, hit
        if contours:
            boundary_xy = np.concatenate([item.xy_m for item in contours])
            boundary_tangent = np.concatenate([item.tangent_xy for item in contours])
            boundary_inward = np.concatenate([item.inward_normal_xy for item in contours])
            tree = cKDTree(boundary_xy)
            center_tangent = np.column_stack((np.cos(theta), np.sin(theta)))
            for hit, radius in ((minus, r_minus), (plus, r_plus)):
                valid_hit = np.isfinite(radius)
                _, nearest = tree.query(hit[valid_hit])
                alignment = np.abs(np.sum(boundary_tangent[nearest] * center_tangent[valid_hit], axis=1))
                direction_to_center = (center[valid_hit] - hit[valid_hit]) / radius[valid_hit, None]
                inward = np.sum(boundary_inward[nearest] * direction_to_center, axis=1)
                target = np.flatnonzero(valid_hit)
                tangent_alignment[target] = np.fmin(
                    np.nan_to_num(tangent_alignment[target], nan=alignment), alignment
                )
                normal_consistency[target] = np.fmin(
                    np.nan_to_num(normal_consistency[target], nan=inward), inward
                )
        diameter = r_minus + r_plus
        asymmetry = np.abs(r_plus - r_minus) / diameter
        accepted = np.isfinite(diameter) & (diameter > 0)
        for index in range(count):
            flags: list[str] = []
            if not np.isfinite(r_plus[index]): flags.append("MISSING_POSITIVE_EDGE")
            if not np.isfinite(r_minus[index]): flags.append("MISSING_NEGATIVE_EDGE")
            if coh[index] < self.low_coherence: flags.append("LOW_ORIENTATION_COHERENCE")
            if np.isfinite(asymmetry[index]) and asymmetry[index] > self.high_asymmetry: flags.append("HIGH_ASYMMETRY")
            if np.isfinite(tangent_alignment[index]) and tangent_alignment[index] < self.minimum_tangent_alignment:
                flags.append("EDGE_TANGENT_MISMATCH")
            if np.isfinite(normal_consistency[index]) and normal_consistency[index] < 0.0:
                flags.append("EDGE_NORMAL_MISMATCH")
            ratio = diameter[index] / edt[index] if edt[index] > 0 and np.isfinite(diameter[index]) else np.nan
            if np.isfinite(ratio) and (ratio < 0.5 or ratio > 2.0): flags.append("POSSIBLE_CROSSING")
            if any(flag in flags for flag in ("LOW_ORIENTATION_COHERENCE", "HIGH_ASYMMETRY", "POSSIBLE_CROSSING", "EDGE_NORMAL_MISMATCH")):
                flags.append("AMBIGUOUS_LOCAL_WIDTH")
                accepted[index] = False
            all_flags.append(tuple(flags))
        return PairedEdgeSamples(
            center, normal, minus, plus, r_minus, r_plus, diameter, asymmetry,
            coh, edt, tangent_alignment, normal_consistency, accepted, tuple(all_flags),
        )


@dataclass(frozen=True, slots=True)
class ClassicalFiberField:
    """Structure-tensor orientation plus anisotropic EDT metrology.

    The tensor estimates the local *longitudinal* fibre axis (perpendicular to
    the dominant image-gradient axis).  The mask is intentionally supplied by
    the caller: segmentation remains an independently provenance-tracked step.
    """

    sigma_derivative: float = 1.0
    sigma_tensor: float = 3.0
    epsilon: float = 1e-12

    def infer_field(
        self,
        image: np.ndarray,
        *,
        mask: np.ndarray,
        pixel_size_xy_m: tuple[float, float],
    ) -> FiberFieldResult:
        source = np.asarray(image, dtype=float)
        binary = np.asarray(mask, dtype=bool)
        if source.ndim != 2 or binary.shape != source.shape:
            raise ValueError("image and mask must be equal-shape 2D arrays")
        pixel_x_m, pixel_y_m = (float(value) for value in pixel_size_xy_m)
        if pixel_x_m <= 0 or pixel_y_m <= 0:
            raise ValueError("pixel spacing must be positive")

        finite = source[np.isfinite(source)]
        if not finite.size:
            raise ValueError("image must contain finite values")
        low, high = float(finite.min()), float(finite.max())
        normalized = np.zeros_like(source) if high == low else (source - low) / (high - low)

        # Axis 0 is rows/y and axis 1 is columns/x.
        iy = ndimage.gaussian_filter(normalized, self.sigma_derivative, order=(1, 0), mode="reflect")
        ix = ndimage.gaussian_filter(normalized, self.sigma_derivative, order=(0, 1), mode="reflect")
        jxx = ndimage.gaussian_filter(ix * ix, self.sigma_tensor, mode="reflect")
        jxy = ndimage.gaussian_filter(ix * iy, self.sigma_tensor, mode="reflect")
        jyy = ndimage.gaussian_filter(iy * iy, self.sigma_tensor, mode="reflect")
        discriminant = np.hypot(jxx - jyy, 2.0 * jxy)
        trace = jxx + jyy
        theta_gradient = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
        qx, qy = double_angle_orientation(theta_gradient + np.pi / 2.0)
        coherence = discriminant / (trace + self.epsilon)

        radius_px = ndimage.distance_transform_edt(binary)
        radius_m = ndimage.distance_transform_edt(binary, sampling=(pixel_y_m, pixel_x_m))
        centerline = skeletonize(binary)
        return FiberFieldResult(
            fiber_probability=binary.astype(float),
            orientation_qx=qx,
            orientation_qy=qy,
            coherence=coherence,
            radius_px=radius_px,
            diameter_px=2.0 * radius_px,
            radius_m=radius_m,
            diameter_m=2.0 * radius_m,
            centerline=centerline,
            confidence=coherence,
            binary_mask=binary,
            metadata={
                "orientation_method": "STRUCTURE_TENSOR_DOUBLE_ANGLE_V1",
                "radius_method": "ANISOTROPIC_EDT_MASK_V1",
                "centerline_algorithm": "SKIMAGE_SKELETONIZE_MASK_BASELINE",
                "sigma_derivative": str(self.sigma_derivative),
                "sigma_tensor": str(self.sigma_tensor),
            },
        )


class FiberPerceptionBackend(Protocol):
    """Contract for classical, Omnipose, embedding, or future ML perception."""

    method_id: str

    def infer_field(
        self,
        image: np.ndarray,
        *,
        mask: np.ndarray,
        pixel_size_xy_m: tuple[float, float],
    ) -> FiberFieldResult: ...


class FiberGraphBuilder(Protocol):
    """Future deterministic topology reconstruction from an observed field."""

    def build_graph(self, field: FiberFieldResult, *, pixel_size_xy_m: tuple[float, float]) -> dict[str, object]: ...
