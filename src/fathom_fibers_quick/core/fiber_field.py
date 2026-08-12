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
