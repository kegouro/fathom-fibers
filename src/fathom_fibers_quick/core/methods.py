"""Qt-free contracts for comparable, but not interchangeable, scientific methods."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MethodId(str, Enum):
    MATLAB_SIMPOLY = "MATLAB_SIMPOLY"
    PYTHON_SIMPOLY = "PYTHON_SIMPOLY"
    FATHOM_LOCAL = "FATHOM_LOCAL"
    FATHOM_FIELD_GRAPH_V1 = "FATHOM_FIELD_GRAPH_V1"
    MANUAL_5X5_REFERENCE = "MANUAL_5X5_REFERENCE"
    CONSENSUS_PSEUDO_REFERENCE_V1 = "CONSENSUS_PSEUDO_REFERENCE_V1"


class MethodStatus(str, Enum):
    COMPLETE = "COMPLETE"
    NOT_RUN = "NOT_RUN"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_MEASURED = "NOT_MEASURED"
    EXPERIMENTAL_NOT_YET_MEASURING = "EXPERIMENTAL_NOT_YET_MEASURING"
    EXPERIMENTAL_FIELD_MEASURING = "EXPERIMENTAL_FIELD_MEASURING"
    FAILED = "FAILED"


class Capability(str, Enum):
    GLOBAL_DIAMETER_DISTRIBUTION = "GLOBAL_DIAMETER_DISTRIBUTION"
    LOCAL_EDT_DIAMETERS = "LOCAL_EDT_DIAMETERS"
    LOCAL_METROLOGY = "LOCAL_METROLOGY"
    CROSS_SECTIONS = "CROSS_SECTIONS"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    QUALITY_FLAGS = "QUALITY_FLAGS"
    MASK = "MASK"
    SKELETON = "SKELETON"
    ORIENTATION_FIELD = "ORIENTATION_FIELD"
    ORIENTED_BOUNDARIES = "ORIENTED_BOUNDARIES"
    PAIRED_EDGE_LOCAL_WIDTH = "PAIRED_EDGE_LOCAL_WIDTH"
    INTENSITY_PROFILE_LOCAL_WIDTH = "INTENSITY_PROFILE_LOCAL_WIDTH"
    LOCAL_RADIUS = "LOCAL_RADIUS"
    LOCAL_DIAMETER = "LOCAL_DIAMETER"
    REFINED_CENTERLINE_REMEASUREMENT = "REFINED_CENTERLINE_REMEASUREMENT"
    GRAPH = "GRAPH"
    CROSSINGS = "CROSSINGS"
    FIBER_INSTANCES = "FIBER_INSTANCES"
    TOPOLOGY = "TOPOLOGY"
    MATLAB_COMPATIBILITY_EVIDENCE = "MATLAB_COMPATIBILITY_EVIDENCE"


class CapabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    EXPERIMENTAL = "EXPERIMENTAL"
    UNAVAILABLE = "UNAVAILABLE"


class Estimand(str, Enum):
    COMMON_LENGTH_WEIGHTED_DIAMETER = "COMMON_LENGTH_WEIGHTED_DIAMETER"
    FIBER_BALANCED_DIAMETER = "FIBER_BALANCED_DIAMETER"
    SIMPOLY_NATIVE_GAUSS1 = "SIMPOLY_NATIVE_GAUSS1"
    FATHOM_NATIVE_LOCAL = "FATHOM_NATIVE_LOCAL"
    FATHOM_FIELD_PAIRED_EDGE_DIAMETER = "FATHOM_FIELD_PAIRED_EDGE_DIAMETER"
    FATHOM_FIELD_PROFILE_DIAMETER = "FATHOM_FIELD_PROFILE_DIAMETER"
    FATHOM_FIELD_REFINED_EDT_DIAMETER = "FATHOM_FIELD_REFINED_EDT_DIAMETER"
    FATHOM_FIELD_REFINED_EDGE_DIAMETER = "FATHOM_FIELD_REFINED_EDGE_DIAMETER"
    FATHOM_FIELD_REFINED_PROFILE_DIAMETER = "FATHOM_FIELD_REFINED_PROFILE_DIAMETER"
    MANUAL_5X5_REFERENCE = "MANUAL_5X5_REFERENCE"


@dataclass(frozen=True, slots=True)
class MethodCapabilities:
    states: Mapping[Capability, CapabilityState] = field(default_factory=dict)

    def state(self, capability: Capability) -> CapabilityState:
        return self.states.get(capability, CapabilityState.UNAVAILABLE)

    def supports(self, capability: Capability) -> bool:
        return self.state(capability) == CapabilityState.AVAILABLE

    def to_dict(self) -> dict[str, str]:
        return {capability.value: state.value for capability, state in self.states.items()}


@dataclass(frozen=True, slots=True)
class DiameterDistribution:
    diameter: np.ndarray
    weight: np.ndarray
    unit: str
    estimand: Estimand
    source_method: MethodId

    def __post_init__(self) -> None:
        diameter = np.asarray(self.diameter, dtype=float).ravel()
        weight = np.asarray(self.weight, dtype=float).ravel()
        if diameter.shape != weight.shape:
            raise ValueError("diameter and weight must have the same shape")
        if not self.unit:
            raise ValueError("unit is required")
        if np.any(~np.isfinite(diameter)) or np.any(~np.isfinite(weight)):
            raise ValueError("diameter and weight must be finite")
        if np.any(diameter < 0) or np.any(weight < 0):
            raise ValueError("diameter and weight must be non-negative")
        if diameter.size and not np.any(weight > 0):
            raise ValueError("at least one sample must have positive weight")
        object.__setattr__(self, "diameter", diameter)
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class MethodResult:
    """One method's observation of one calibrated image/ROI.

    Optional fields are intentionally absent when a backend cannot support them;
    consumers must consult ``capabilities`` rather than infer data.
    """

    method_id: MethodId
    method_version: str
    image_id: str
    calibration: Mapping[str, Any]
    valid_roi: tuple[int, int, int, int] | None
    unit: str
    capabilities: MethodCapabilities
    status: MethodStatus = MethodStatus.NOT_RUN
    native_estimand: Estimand | None = None
    native_result: float | None = None
    native_statistics: Mapping[str, Any] = field(default_factory=dict)
    native_distribution: DiameterDistribution | None = None
    common_distribution: DiameterDistribution | None = None
    secondary_distributions: Mapping[str, DiameterDistribution] = field(default_factory=dict)
    fiber_balanced_distribution: DiameterDistribution | None = None
    mask: np.ndarray | None = None
    centerline: np.ndarray | None = None
    orientation_field: tuple[np.ndarray, np.ndarray] | None = None
    radius_map: np.ndarray | None = None
    local_samples: Mapping[str, np.ndarray] | None = None
    fiber_instances: Mapping[str, Any] | None = None
    fiber_graph: Mapping[str, Any] | None = None
    quality_flags: tuple[str, ...] = ()
    confidence: float | None = None
    runtime_seconds: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_version or not self.image_id or not self.unit:
            raise ValueError("method_version, image_id and unit are required")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if self.runtime_seconds is not None and self.runtime_seconds < 0:
            raise ValueError("runtime_seconds cannot be negative")
        for distribution in (
            self.native_distribution,
            self.common_distribution,
            self.fiber_balanced_distribution,
            *self.secondary_distributions.values(),
        ):
            if distribution is not None and distribution.unit != self.unit:
                raise ValueError("all distributions must use the result unit")


def method_cache_key(
    *,
    image_sha256: str | None,
    valid_roi: tuple[int, int, int, int] | None,
    calibration: Mapping[str, Any],
    method_id: MethodId,
    method_version: str,
    parameters: Mapping[str, Any],
    code_version: str | None = None,
    external_dependency_version: str | None = None,
) -> str:
    """Stable cache identity; no image bytes or private path enter the key."""
    payload = {
        "image_sha256": image_sha256,
        "valid_roi": valid_roi,
        "calibration": dict(calibration),
        "method_id": method_id.value,
        "method_version": method_version,
        "parameters": dict(parameters),
        "code_version": code_version,
        "external_dependency_version": external_dependency_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
