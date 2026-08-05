from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .measurement_records import MeasurementKind
from .model import Calibration, Project


def get_utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class SampleRecord:
    sample_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=get_utc_now_iso)


@dataclass
class ImageRecord:
    image_id: str
    sample_id: str | None
    path: str
    width_px: int
    height_px: int
    calibration: Calibration
    created_at: str = field(default_factory=get_utc_now_iso)


@dataclass
class FiberRecord:
    fiber_id: str
    image_id: str
    sample_id: str | None = None
    group: int | None = None
    defect: str = "None"
    notes: str = ""


def _compute_level_metrics(values: Sequence[float]) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "iqr": None,
            "min": None,
            "max": None,
            "p05": None,
            "p95": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    sd_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = float(q75 - q25)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    p05_val = float(np.percentile(arr, 5))
    p95_val = float(np.percentile(arr, 95))

    return {
        "n": n,
        "mean": mean_val,
        "median": median_val,
        "sd": sd_val,
        "iqr": iqr_val,
        "min": min_val,
        "max": max_val,
        "p05": p05_val,
        "p95": p95_val,
    }


def compute_hierarchical_statistics(project: Project) -> dict[str, Any]:
    """Calculates statistics at 4 distinct levels: Section, Fiber, Image, Sample."""
    # 1. Section level: All accepted PROJECTED_WIDTH records
    sec_records = [r for r in project.records if r.kind == MeasurementKind.PROJECTED_WIDTH and r.is_included_in_statistics]
    sec_values = [float(r.primary_value) for r in sec_records if r.primary_value is not None]
    sec_stats = _compute_level_metrics(sec_values)

    # 2. Fiber level: Median per fiber
    fiber_groups: dict[str, list[float]] = {}
    for r in sec_records:
        fid = r.fiber_id or "UNASSIGNED"
        if r.primary_value is not None:
            fiber_groups.setdefault(fid, []).append(float(r.primary_value))

    fiber_medians = [float(np.median(vals)) for vals in fiber_groups.values() if len(vals) > 0]
    fiber_stats = _compute_level_metrics(fiber_medians)

    # 3. Image level: Summary of fiber medians per image
    image_groups: dict[str, list[float]] = {}
    for r in sec_records:
        img_id = r.image_id or project.image.path
        fid = r.fiber_id or "UNASSIGNED"
        if r.primary_value is not None:
            image_groups.setdefault(img_id, []).append(float(r.primary_value))

    image_medians = [float(np.median(vals)) for vals in image_groups.values() if len(vals) > 0]
    image_stats = _compute_level_metrics(image_medians)

    # 4. Sample level: Summary across sample
    sample_stats = dict(image_stats)
    sample_stats["n_images"] = len(image_groups)
    sample_stats["n_fibers"] = len(fiber_groups)
    sample_stats["n_sections"] = len(sec_values)

    return {
        "section_level": sec_stats,
        "fiber_level": fiber_stats,
        "image_level": image_stats,
        "sample_level": sample_stats,
    }


def hierarchical_bootstrap(
    project: Project,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict[str, float | None]:
    """Performs deterministic two-stage hierarchical bootstrap (Resample images -> Resample fibers)."""
    sec_records = [r for r in project.records if r.kind == MeasurementKind.PROJECTED_WIDTH and r.is_included_in_statistics]

    # Organize data: image -> fiber -> [widths]
    data_hierarchy: dict[str, dict[str, list[float]]] = {}
    for r in sec_records:
        img_id = r.image_id or project.image.path
        fid = r.fiber_id or "UNASSIGNED"
        if r.primary_value is not None:
            data_hierarchy.setdefault(img_id, {}).setdefault(fid, []).append(float(r.primary_value))

    images = list(data_hierarchy.keys())
    if not images:
        return {"bootstrap_mean_m": None, "ci_lower_m": None, "ci_upper_m": None, "seed": seed, "n_bootstraps": n_bootstraps}

    rng = np.random.default_rng(seed)
    bootstrap_medians = []

    for _ in range(n_bootstraps):
        sampled_img_keys = rng.choice(images, size=len(images), replace=True)
        resampled_fiber_medians = []

        for img_key in sampled_img_keys:
            fibers_dict = data_hierarchy[img_key]
            fiber_keys = list(fibers_dict.keys())
            if not fiber_keys:
                continue
            sampled_fiber_keys = rng.choice(fiber_keys, size=len(fiber_keys), replace=True)

            for f_key in sampled_fiber_keys:
                widths = fibers_dict[f_key]
                sampled_widths = rng.choice(widths, size=len(widths), replace=True)
                resampled_fiber_medians.append(float(np.median(sampled_widths)))

        if resampled_fiber_medians:
            bootstrap_medians.append(float(np.median(resampled_fiber_medians)))

    if not bootstrap_medians:
        return {"bootstrap_mean_m": None, "ci_lower_m": None, "ci_upper_m": None, "seed": seed, "n_bootstraps": n_bootstraps}

    arr = np.array(bootstrap_medians)
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])

    return {
        "bootstrap_mean_m": float(np.mean(arr)),
        "ci_lower_m": float(ci_low),
        "ci_upper_m": float(ci_high),
        "seed": seed,
        "n_bootstraps": n_bootstraps,
    }
