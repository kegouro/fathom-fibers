from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .measurement_records import (
    MeasurementKind,
    MeasurementRecord,
    MeasurementSource,
    MeasurementStatus,
)
from .model import Calibration


@dataclass
class MeasurementUncertainty:
    calibration_m: float | None = None
    edge_localization_m: float | None = None
    repeatability_m: float | None = None
    combined_standard_m: float | None = None
    method: str = "INDEPENDENT_QUADRATURE"
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_m": self.calibration_m,
            "edge_localization_m": self.edge_localization_m,
            "repeatability_m": self.repeatability_m,
            "combined_standard_m": self.combined_standard_m,
            "method": self.method,
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MeasurementUncertainty:
        return cls(
            calibration_m=data.get("calibration_m"),
            edge_localization_m=data.get("edge_localization_m"),
            repeatability_m=data.get("repeatability_m"),
            combined_standard_m=data.get("combined_standard_m"),
            method=data.get("method", "INDEPENDENT_QUADRATURE"),
            assumptions=list(data.get("assumptions", [])),
        )


def validate_resolution(
    width_m: float,
    calibration: Calibration,
    resolved_px_threshold: float = 5.0,
    marginal_px_threshold: float = 2.5,
) -> tuple[str, float, list[str]]:
    """Calculates width_px_equivalent and classifies resolution status."""
    px_size = calibration.pixel_size_x_m
    width_px_equivalent = width_m / px_size if px_size > 0 else 0.0

    flags: list[str] = []
    if width_px_equivalent >= resolved_px_threshold:
        status = "RESOLVED"
    elif width_px_equivalent >= marginal_px_threshold:
        status = "MARGINAL"
        flags.append("RESOLUTION_MARGINAL")
    else:
        status = "UNRESOLVED"
        flags.append("RESOLUTION_INSUFFICIENT")

    return status, width_px_equivalent, flags


def derive_quality_flags(
    record: MeasurementRecord,
    calibration: Calibration,
    image_shape: tuple[int, int],
    footer_bounds: tuple[int, int] | None = None,
    edge_margin_px: float = 5.0,
) -> list[str]:
    """Pure function deriving standardized quality flags from geometry, calibration, resolution, and source."""
    flags: set[str] = set(record.quality_flags)

    # 1. Calibration flag
    if calibration.source == "UNVERIFIED" or calibration.confidence < 0.8:
        flags.add("CALIBRATION_UNVERIFIED")

    # 2. Resolution flags (for width & distance measurements)
    primary_val = record.primary_value
    if primary_val is not None and record.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
        _res_status, _px_eq, res_flags = validate_resolution(primary_val, calibration)
        flags.update(res_flags)

    # 3. Geometry proximity flags (Image edges & Footer bounds)
    h, w = image_shape
    pts: list[tuple[float, float]] = []
    if "p1" in record.geometry and "p2" in record.geometry:
        pts = [record.geometry["p1"], record.geometry["p2"]]
    elif "points" in record.geometry:
        pts = record.geometry["points"]
    elif "polygon" in record.geometry:
        pts = record.geometry["polygon"]

    touches_edge = False
    touches_footer = False

    for x, y in pts:
        if x <= edge_margin_px or x >= (w - edge_margin_px) or y <= edge_margin_px or y >= (h - edge_margin_px):
            touches_edge = True

        if footer_bounds is not None:
            fy0, fy1 = footer_bounds
            if fy0 <= y <= fy1:
                touches_footer = True

    if touches_edge:
        flags.add("TOUCHES_IMAGE_EDGE")
    if touches_footer:
        flags.add("TOUCHES_INVALID_MASK")

    # 4. Source & review flags
    if record.source == MeasurementSource.AUTO_ROI_COMPONENT and record.status == MeasurementStatus.PROPOSED:
        flags.add("AUTOMATIC_NOT_REVIEWED")
    if record.source == MeasurementSource.MANUAL:
        flags.add("MANUAL_REFERENCE")

    return sorted(flags)


def compute_measurement_uncertainty(
    record: MeasurementRecord,
    calibration: Calibration,
    repeatability_std_m: float | None = None,
) -> MeasurementUncertainty:
    """Computes uncertainty components without inventing values."""
    assumptions: list[str] = ["Se asumen componentes de incertidumbre independientes."]
    val_m = record.primary_value

    if val_m is None or val_m <= 0:
        return MeasurementUncertainty(method="NONE", assumptions=["Medición sin valor físico evaluable."])

    # 1. Calibration component (standard uncertainty estimated at 2% if unspecified)
    cal_u_m: float | None = None
    if calibration.confidence > 0:
        cal_rel = max(0.01, 1.0 - calibration.confidence)
        cal_u_m = val_m * cal_rel
        assumptions.append(f"Incertidumbre de calibración ({cal_rel * 100:.1f}%) derivada de confianza ({calibration.confidence}).")

    # 2. Geometric edge localization component (sensitivity ±0.5 px)
    # u_edge = 0.5 * pixel_size / sqrt(3) (rectangular distribution)
    edge_u_m: float | None = (0.5 * calibration.pixel_size_x_m) / math.sqrt(3)
    assumptions.append("Incertidumbre de localización de bordes estimada por sensibilidad geométrica de ±0.5 px (distribución rectangular).")

    # 3. Repeatability component
    rep_u_m = repeatability_std_m

    # Combined standard uncertainty u_c = sqrt(u_cal^2 + u_edge^2 + u_rep^2)
    terms = [u for u in (cal_u_m, edge_u_m, rep_u_m) if u is not None]
    if terms:
        combined_m = math.sqrt(sum(u**2 for u in terms))
    else:
        combined_m = None

    return MeasurementUncertainty(
        calibration_m=cal_u_m,
        edge_localization_m=edge_u_m,
        repeatability_m=rep_u_m,
        combined_standard_m=combined_m,
        method="INDEPENDENT_QUADRATURE",
        assumptions=assumptions,
    )
