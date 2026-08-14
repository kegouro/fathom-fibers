"""Stable headless API shared by the desktop application and integrations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .auto_roi import analyze_roi
from .core.contracts import (
    FathomAnalysisResult,
    MethodComparisonResult,
    MethodComparisonRow,
    ScientificImage,
    ScientificMeasurement,
)
from .measurement_geometry import (
    compute_angle_geometry,
    compute_area_roi_geometry,
    compute_line_geometry,
    compute_polyline_geometry,
    compute_profile_geometry,
)
from .measurement_records import MeasurementKind, MeasurementRecord
from .model import Calibration
from .oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    PROFILE_SOURCE_COMPAT_V1,
    SIMPolySourceConfig,
    SIMPolySourceResult,
    run_simpoly_source_pipeline,
)
from .zeiss import detect_footer, load_image_document


def _gray_from_array(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels = np.asarray(array)
    if pixels.ndim == 2:
        return pixels, pixels.astype(np.float64, copy=False)
    if pixels.ndim == 3 and pixels.shape[2] >= 3:
        rgb = pixels[..., :3]
        gray = (
            0.2126 * rgb[..., 0].astype(np.float64)
            + 0.7152 * rgb[..., 1].astype(np.float64)
            + 0.0722 * rgb[..., 2].astype(np.float64)
        )
        return pixels, gray
    raise ValueError("image must be a 2D grayscale or HxWx3+ array")


def _normalized_roi(
    image: ScientificImage,
    roi_bbox: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int]:
    height, width = image.shape
    valid_height = image.footer_bounds[0] if image.footer_bounds else height
    x0, y0, x1, y1 = roi_bbox or (0, 0, width, valid_height)
    x0 = max(0, min(int(x0), width - 1))
    y0 = max(0, min(int(y0), valid_height - 1))
    x1 = max(x0 + 1, min(int(x1), width))
    y1 = max(y0 + 1, min(int(y1), valid_height))
    return x0, y0, x1, y1


class FathomEngine:
    """Deterministic, Qt-free façade over Fathom scientific operations."""

    def open_image(
        self,
        path: str | Path,
        *,
        manual_pixel_size_m: float | None = None,
        compute_hash: bool = True,
    ) -> ScientificImage:
        document, rgb, gray = load_image_document(
            path,
            manual_pixel_size_m=manual_pixel_size_m,
            compute_hash=compute_hash,
        )
        return ScientificImage(
            pixels=np.asarray(rgb),
            gray=gray,
            calibration=document.calibration,
            image_id=Path(document.path).name,
            metadata=dict(document.metadata),
            footer_bounds=document.footer_bounds,
            source_path=document.path,
            source_sha256=document.source_sha256,
        )

    def from_array(
        self,
        array: np.ndarray,
        *,
        calibration: Calibration,
        image_id: str = "memory-image",
        metadata: Mapping[str, Any] | None = None,
        footer_bounds: tuple[int, int] | None = None,
        detect_footer_band: bool = False,
    ) -> ScientificImage:
        pixels, gray = _gray_from_array(np.asarray(array))
        detected = detect_footer(gray) if detect_footer_band and footer_bounds is None else footer_bounds
        return ScientificImage(
            pixels=pixels,
            gray=gray,
            calibration=calibration,
            image_id=image_id,
            metadata=dict(metadata or {}),
            footer_bounds=detected,
        )

    def measure(
        self,
        image: ScientificImage,
        kind: MeasurementKind | str,
        geometry: Mapping[str, Any],
        *,
        profile_bandwidth_px: int = 3,
    ) -> ScientificMeasurement:
        kind = MeasurementKind(kind)
        geom = dict(geometry)
        values: dict[str, Any]
        flags: tuple[str, ...] = ()
        if kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
            values = compute_line_geometry(tuple(geom["p1"]), tuple(geom["p2"]), image.calibration)
        elif kind == MeasurementKind.POLYLINE_LENGTH:
            values = compute_polyline_geometry(geom["points"], image.calibration)
        elif kind == MeasurementKind.ANGLE:
            values = compute_angle_geometry(
                tuple(geom["pt_a"]), tuple(geom["pt_b"]), tuple(geom["pt_c"]), image.calibration
            )
        elif kind in {MeasurementKind.RECTANGLE_AREA, MeasurementKind.POLYGON_AREA}:
            values = compute_area_roi_geometry(
                image.gray,
                image.calibration,
                bbox=tuple(geom["bbox"]) if "bbox" in geom else None,
                polygon=geom.get("points"),
                footer_bounds=image.footer_bounds,
            )
        elif kind == MeasurementKind.INTENSITY_PROFILE:
            values = compute_profile_geometry(
                image.gray,
                tuple(geom["p1"]),
                tuple(geom["p2"]),
                image.calibration,
                bandwidth_px=profile_bandwidth_px,
            )
        else:
            raise ValueError(f"Interactive measurement kind is unsupported: {kind.value}")
        record = MeasurementRecord("preview", kind, "preview", geometry=geom, values=values)
        return ScientificMeasurement(kind, geom, values, record.primary_value, record.primary_unit, flags)

    def run_fathom(
        self,
        image: ScientificImage,
        *,
        roi_bbox: tuple[int, int, int, int] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> FathomAnalysisResult:
        roi = _normalized_roi(image, roi_bbox)
        candidates, summary = analyze_roi(
            image.gray,
            roi,
            image.calibration,
            footer_bounds=image.footer_bounds,
            **dict(options or {}),
        )
        return FathomAnalysisResult("FATHOM_ASSISTED_ROI", roi, tuple(candidates), summary)

    def run_simpoly(
        self,
        image: ScientificImage,
        *,
        profile: str = PROFILE_SOURCE_COMPAT_V1,
        roi_bbox: tuple[int, int, int, int] | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[SIMPolySourceResult, Any]:
        if profile not in {PROFILE_SOURCE_COMPAT_V1, PROFILE_CONTROLLED_INPUT_V1}:
            raise ValueError(f"Unknown SIMPoly profile: {profile}")
        flags: list[str] = []
        if profile == PROFILE_SOURCE_COMPAT_V1:
            body = image.pixels
        else:
            x0, y0, x1, y1 = _normalized_roi(image, roi_bbox)
            body = image.pixels[y0:y1, x0:x1].copy()
            if valid_mask is not None:
                mask = np.asarray(valid_mask, dtype=bool)
                if mask.shape != body.shape[:2]:
                    raise ValueError("valid_mask must match the controlled ROI shape")
                if body.ndim == 3:
                    body[~mask, :] = 0
                else:
                    body[~mask] = 0
                flags.append("CALLER_VALID_MASK_APPLIED")
        um_per_px = image.calibration.pixel_size_x_m * 1e6
        if not np.isclose(
            image.calibration.pixel_size_x_m,
            image.calibration.pixel_size_y_m,
            rtol=1e-6,
        ):
            flags.append("ANISOTROPIC_PIXEL_X_CALIBRATION_USED")
        result, intermediates = run_simpoly_source_pipeline(
            body,
            SIMPolySourceConfig(profile=profile, conversion_um_per_px=um_per_px),
        )
        if flags:
            result = SIMPolySourceResult(**{**asdict(result), "flags": (*result.flags, *flags)})
        return result, intermediates

    def compare_methods(
        self,
        image: ScientificImage,
        *,
        roi_bbox: tuple[int, int, int, int] | None = None,
        manual_measurements: Sequence[MeasurementRecord] = (),
    ) -> MethodComparisonResult:
        roi = _normalized_roi(image, roi_bbox)
        fathom = self.run_fathom(image, roi_bbox=roi)
        simpoly, intermediates = self.run_simpoly(
            image,
            profile=PROFILE_CONTROLLED_INPUT_V1,
            roi_bbox=roi,
        )
        candidate_px = [
            p.width_m / image.calibration.pixel_size_x_m
            for candidate in fathom.candidates
            for p in candidate.proposed_measurements
        ]
        manual_px = [
            record.primary_value / image.calibration.pixel_size_x_m
            for record in manual_measurements
            if record.primary_value is not None and record.is_included_in_statistics
        ]
        baseline = simpoly.gaussian_center_px

        def row(method: str, estimand: str, values: Sequence[float], main: float | None, flags=()):
            mean = float(np.mean(values)) if len(values) else None
            median = float(np.median(values)) if len(values) else None
            difference = main - baseline if main is not None and baseline is not None else None
            relative = (
                100.0 * difference / baseline if difference is not None and baseline not in {None, 0.0} else None
            )
            return MethodComparisonRow(
                method,
                estimand,
                len(values),
                mean,
                median,
                main,
                difference,
                relative,
                tuple(flags),
            )

        rows = (
            row(
                "SIMPOLY_CONTROLLED_INPUT",
                "SIMPOLY_GAUSSIAN_CENTER",
                simpoly.local_diameters_px,
                simpoly.gaussian_center_px,
                simpoly.flags,
            ),
            row("FATHOM", "LOCAL_SECTION_WEIGHTED", candidate_px, float(np.median(candidate_px)) if candidate_px else None),
            row("MANUAL", "MANUAL_ACCEPTED_SECTIONS", manual_px, float(np.mean(manual_px)) if manual_px else None),
        )
        return MethodComparisonResult(roi, rows, simpoly, intermediates, fathom)


__all__ = ["FathomEngine", "ScientificImage", "ScientificMeasurement"]
