"""Future-facing perception and graph contracts; no ML runtime is required."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class FiberFieldResult:
    """Perception output in pixel coordinates before calibrated metrology.

    A backend may provide only a subset.  The V1 registry intentionally has no
    implementation that emits measurements from this result yet.
    """

    fiber_probability: np.ndarray | None = None
    orientation_qx: np.ndarray | None = None
    orientation_qy: np.ndarray | None = None
    radius_proposal_px: np.ndarray | None = None
    crossing_probability: np.ndarray | None = None
    confidence: np.ndarray | None = None
    binary_mask: np.ndarray | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class FiberPerceptionBackend(Protocol):
    """Contract for classical, Omnipose, embedding, or future ML perception."""

    method_id: str

    def infer_field(
        self,
        image: np.ndarray,
        *,
        pixel_size_xy_m: tuple[float, float],
    ) -> FiberFieldResult: ...


class FiberGraphBuilder(Protocol):
    """Future deterministic topology reconstruction from an observed field."""

    def build_graph(self, field: FiberFieldResult, *, pixel_size_xy_m: tuple[float, float]) -> dict[str, object]: ...
