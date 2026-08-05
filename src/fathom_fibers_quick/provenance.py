from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy

from .model import Project


@dataclass
class SoftwareProvenance:
    application_version: str = "0.3.0"
    schema_version: int = 3
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    numpy_version: str = field(default_factory=lambda: np.__version__)
    scipy_version: str = field(default_factory=lambda: scipy.__version__)
    platform_info: str = field(default_factory=platform.platform)
    image_sha256: str | None = None
    reader_id: str = "zeiss_tiff_reader"
    calibration_source: str = "ZEISS_HEADER"
    measurement_method: str = "MANUAL_AND_ASSISTED"
    method_parameters: dict[str, Any] = field(default_factory=dict)
    protocol_id: str = "PVDF_5_SECTIONS"
    protocol_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "application_version": self.application_version,
            "schema_version": self.schema_version,
            "python_version": self.python_version,
            "numpy_version": self.numpy_version,
            "scipy_version": self.scipy_version,
            "platform_info": self.platform_info,
            "image_sha256": self.image_sha256,
            "reader_id": self.reader_id,
            "calibration_source": self.calibration_source,
            "measurement_method": self.measurement_method,
            "method_parameters": self.method_parameters,
            "protocol_id": self.protocol_id,
            "protocol_version": self.protocol_version,
        }

    @classmethod
    def from_project(cls, project: Project) -> SoftwareProvenance:
        return cls(
            image_sha256=project.image.source_sha256,
            calibration_source=project.image.calibration.source,
            protocol_id=project.active_protocol_id or "PVDF_5_SECTIONS",
        )
