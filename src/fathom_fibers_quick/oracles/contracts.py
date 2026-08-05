from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EstimandType(str, Enum):
    SIMPOLY_GAUSSIAN_CENTER = "SIMPOLY_GAUSSIAN_CENTER"
    SIMPOLY_GAUSSIAN_SIGMA = "SIMPOLY_GAUSSIAN_SIGMA"
    SKELETON_PIXEL_MEAN = "SKELETON_PIXEL_MEAN"
    SKELETON_PIXEL_MEDIAN = "SKELETON_PIXEL_MEDIAN"
    LOCAL_SECTION_WEIGHTED = "LOCAL_SECTION_WEIGHTED"
    FIBER_MEDIAN_WEIGHTED = "FIBER_MEDIAN_WEIGHTED"
    MANUAL_GRID_MEAN = "MANUAL_GRID_MEAN"


@dataclass
class OracleManifest:
    oracle_id: str
    name: str
    version: str
    source_doi: str
    license_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_id": self.oracle_id,
            "name": self.name,
            "version": self.version,
            "source_doi": self.source_doi,
            "license_status": self.license_status,
        }


@dataclass
class OracleRun:
    run_id: str
    oracle_id: str
    oracle_version: str
    image_id: str
    input_sha256: str | None = None
    calibration: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    local_diameters_px: list[float] = field(default_factory=list)
    gaussian_center_px: float | None = None
    gaussian_sigma_px: float | None = None
    arithmetic_mean_px: float | None = None
    median_px: float | None = None
    std_px: float | None = None
    segmented_fraction: float | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    status: str = "SUCCESS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "oracle_id": self.oracle_id,
            "oracle_version": self.oracle_version,
            "image_id": self.image_id,
            "input_sha256": self.input_sha256,
            "calibration": self.calibration,
            "parameters": self.parameters,
            "local_diameters_px": self.local_diameters_px,
            "gaussian_center_px": self.gaussian_center_px,
            "gaussian_sigma_px": self.gaussian_sigma_px,
            "arithmetic_mean_px": self.arithmetic_mean_px,
            "median_px": self.median_px,
            "std_px": self.std_px,
            "segmented_fraction": self.segmented_fraction,
            "artifacts": self.artifacts,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OracleRun:
        return cls(
            run_id=data.get("run_id", "RUN_001"),
            oracle_id=data.get("oracle_id", "SIMPOLY_MATLAB_ORIGINAL"),
            oracle_version=data.get("oracle_version", "1.0.0"),
            image_id=data.get("image_id", "image.tif"),
            input_sha256=data.get("input_sha256"),
            calibration=data.get("calibration", {}),
            parameters=data.get("parameters", {}),
            local_diameters_px=list(data.get("local_diameters_px", [])),
            gaussian_center_px=data.get("gaussian_center_px"),
            gaussian_sigma_px=data.get("gaussian_sigma_px"),
            arithmetic_mean_px=data.get("arithmetic_mean_px"),
            median_px=data.get("median_px"),
            std_px=data.get("std_px"),
            segmented_fraction=data.get("segmented_fraction"),
            artifacts=dict(data.get("artifacts", {})),
            status=data.get("status", "SUCCESS"),
        )


@dataclass
class OracleComparison:
    comparison_id: str
    image_id: str
    estimand_oracle: EstimandType
    estimand_target: EstimandType
    oracle_value_px: float
    target_value_px: float
    absolute_error_px: float
    relative_error_percent: float
    status: str = "COMPARED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "image_id": self.image_id,
            "estimand_oracle": self.estimand_oracle.value,
            "estimand_target": self.estimand_target.value,
            "oracle_value_px": self.oracle_value_px,
            "target_value_px": self.target_value_px,
            "absolute_error_px": self.absolute_error_px,
            "relative_error_percent": self.relative_error_percent,
            "status": self.status,
        }
