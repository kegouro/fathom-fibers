from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..auto_roi import AutoFiberCandidate, AutoROISummary
from ..measurement_records import MeasurementKind
from ..model import Calibration, ImageDocument
from ..oracles.simpoly_source import SIMPolyIntermediates, SIMPolySourceResult


@dataclass(frozen=True, slots=True)
class ScientificImage:
    """Image pixels and physical context independent of any UI or filesystem."""

    pixels: np.ndarray
    gray: np.ndarray
    calibration: Calibration
    image_id: str = "memory-image"
    metadata: dict[str, Any] = field(default_factory=dict)
    footer_bounds: tuple[int, int] | None = None
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.gray.ndim != 2 or not self.gray.size:
            raise ValueError("gray must be a non-empty 2D array")
        if self.pixels.shape[:2] != self.gray.shape:
            raise ValueError("pixels and gray must have identical spatial shape")
        self.calibration.validate()

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.gray.shape[0]), int(self.gray.shape[1])

    @property
    def valid_body(self) -> np.ndarray:
        if self.footer_bounds is None:
            return self.gray
        return self.gray[: self.footer_bounds[0], :]

    def to_document(self) -> ImageDocument:
        height, width = self.shape
        return ImageDocument(
            path=self.source_path or self.image_id,
            width_px=width,
            height_px=height,
            calibration=self.calibration,
            metadata=dict(self.metadata),
            footer_bounds=self.footer_bounds,
            source_sha256=self.source_sha256,
        )


@dataclass(frozen=True, slots=True)
class ScientificMeasurement:
    kind: MeasurementKind
    geometry: dict[str, Any]
    values: dict[str, Any]
    primary_value: float | None
    primary_unit: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FathomAnalysisResult:
    method: str
    roi_bbox: tuple[int, int, int, int]
    candidates: tuple[AutoFiberCandidate, ...]
    summary: AutoROISummary
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MethodComparisonRow:
    method: str
    estimand: str
    n: int
    mean_px: float | None
    median_px: float | None
    main_reported_px: float | None
    difference_px: float | None
    relative_difference_percent: float | None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MethodComparisonResult:
    roi_bbox: tuple[int, int, int, int]
    rows: tuple[MethodComparisonRow, ...]
    simpoly: SIMPolySourceResult
    simpoly_intermediates: SIMPolyIntermediates
    fathom: FathomAnalysisResult

