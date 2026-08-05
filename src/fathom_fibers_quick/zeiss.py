from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image

from .model import Calibration, ImageDocument


def _unwrap(value: Any) -> tuple[Any, str | None, str | None]:
    if isinstance(value, tuple):
        if len(value) >= 3 and isinstance(value[0], str):
            return value[1], str(value[2]), value[0]
        if len(value) >= 2 and isinstance(value[0], str):
            return value[1], None, value[0]
    return value, None, None


def _to_meters(value: float, unit: str | None) -> float:
    unit_key = (unit or "m").strip().lower().replace("μ", "µ")
    factors = {
        "m": 1.0,
        "mm": 1e-3,
        "µm": 1e-6,
        "um": 1e-6,
        "nm": 1e-9,
        "pm": 1e-12,
    }
    if unit_key not in factors:
        raise ValueError(f"Unsupported length unit: {unit}")
    return float(value) * factors[unit_key]


def _flatten_cz_sem(data: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, raw in data.items():
        if not key:
            continue
        value, unit, label = _unwrap(raw)
        result[key] = value
        if unit:
            result[f"{key}__unit"] = unit
        if label:
            result[f"{key}__label"] = label
    return result


def detect_footer(gray: np.ndarray) -> tuple[int, int] | None:
    """Detect the bright Zeiss information band while preserving image rows below it."""
    if gray.ndim != 2 or gray.shape[0] < 100:
        return None
    start_search = int(gray.shape[0] * 0.65)
    bright_fraction = (gray > 240).mean(axis=1)
    candidate = bright_fraction[start_search:] > 0.60
    indices = np.flatnonzero(candidate) + start_search
    if indices.size < 20:
        return None
    # Find the longest near-contiguous run, allowing tiny text-induced gaps.
    runs: list[list[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) - runs[-1][-1] <= 3:
            runs[-1].append(int(idx))
        else:
            runs.append([int(idx)])
    run = max(runs, key=lambda values: values[-1] - values[0])
    if run[-1] - run[0] < 30:
        return None
    return max(0, run[0] - 4), min(gray.shape[0], run[-1] + 5)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: str | Path, compute_hash: bool = False) -> dict[str, Any]:
    path = Path(path)
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}
    calibration: Calibration | None = None
    format_id = "raster"

    if suffix in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            shape = page.shape
            width_px = int(shape[1])
            height_px = int(shape[0])
            tag = page.tags.get("CZ_SEM") or page.tags.get(34118)
            if tag is not None and isinstance(tag.value, dict):
                format_id = "zeiss_cz_sem_tiff"
                metadata = _flatten_cz_sem(tag.value)
                pixel_raw = metadata.get("ap_image_pixel_size")
                pixel_unit = metadata.get("ap_image_pixel_size__unit")
                if pixel_raw is not None:
                    pixel_m = _to_meters(float(pixel_raw), pixel_unit)
                    calibration = Calibration(pixel_m, pixel_m, "ZEISS_CZ_SEM", 1.0)
            elif page.tags.get("XResolution") and page.tags.get("ResolutionUnit"):
                format_id = "generic_tiff"
    else:
        with Image.open(path) as im:
            width_px, height_px = im.size

    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "format_id": format_id,
        "width_px": width_px,
        "height_px": height_px,
        "metadata": metadata,
        "calibration": asdict(calibration) if calibration else None,
    }
    if calibration:
        result["field_width_m"] = calibration.pixel_size_x_m * width_px
        result["field_height_m"] = calibration.pixel_size_y_m * height_px
        meta_width = metadata.get("ap_width")
        meta_height = metadata.get("ap_height")
        if meta_width is not None:
            meta_width_m = _to_meters(meta_width, metadata.get("ap_width__unit"))
            result["width_crosscheck_relative"] = abs(result["field_width_m"] - meta_width_m) / meta_width_m
        if meta_height is not None:
            meta_height_m = _to_meters(meta_height, metadata.get("ap_height__unit"))
            result["height_crosscheck_relative"] = abs(result["field_height_m"] - meta_height_m) / meta_height_m
    if compute_hash:
        result["sha256"] = file_sha256(path)
    return result


def load_pixels(path: str | Path) -> tuple[Image.Image, np.ndarray]:
    path = Path(path)
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        rgb.load()
    array = np.asarray(rgb, dtype=np.float32)
    gray = 0.2126 * array[..., 0] + 0.7152 * array[..., 1] + 0.0722 * array[..., 2]
    return rgb, gray.astype(np.float32)


def load_image_document(
    path: str | Path,
    manual_pixel_size_m: float | None = None,
    compute_hash: bool = True,
) -> tuple[ImageDocument, Image.Image, np.ndarray]:
    info = inspect_image(path, compute_hash=compute_hash)
    calibration_data = info["calibration"]
    if calibration_data is None:
        if manual_pixel_size_m is None or manual_pixel_size_m <= 0:
            raise ValueError("This image has no supported physical calibration.")
        calibration = Calibration(
            manual_pixel_size_m,
            manual_pixel_size_m,
            "USER_ENTERED",
            0.8,
        )
    else:
        calibration = Calibration(**calibration_data)
    rgb, gray = load_pixels(path)
    footer = detect_footer(gray)
    document = ImageDocument(
        path=info["path"],
        width_px=info["width_px"],
        height_px=info["height_px"],
        calibration=calibration,
        metadata=info["metadata"],
        footer_bounds=footer,
        source_sha256=info.get("sha256"),
    )
    return document, rgb, gray
