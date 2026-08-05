"""Stable extension contracts reserved for future classical and ML backends.

The MVP does not load models. These contracts keep future additions from coupling
model inference to project state or physical measurement semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class BackendManifest:
    backend_id: str
    version: str
    tasks: tuple[str, ...]
    description: str
    deterministic: bool
    dependencies: tuple[str, ...] = ()
    model_license: str | None = None
    weights_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProposalBundle:
    """Model output only. It is not a final scientific result."""

    mask: np.ndarray | None = None
    score_map: np.ndarray | None = None
    centerlines: tuple[np.ndarray, ...] = ()
    defect_regions: tuple[dict[str, Any], ...] = ()
    uncertainty_map: np.ndarray | None = None
    backend_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ProposalBackend(Protocol):
    def manifest(self) -> BackendManifest: ...

    def propose(
        self,
        image: np.ndarray,
        *,
        pixel_size_x_m: float,
        pixel_size_y_m: float,
        valid_mask: np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> ProposalBundle: ...


ENTRY_POINT_GROUP_CLASSICAL = "fathom_fibers.plugins.v1"
ENTRY_POINT_GROUP_MODELS = "fathom_fibers.models.v1"
