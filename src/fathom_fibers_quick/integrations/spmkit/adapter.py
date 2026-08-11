from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ...api import FathomEngine
from ...core.contracts import FathomAnalysisResult, MethodComparisonResult, ScientificImage
from ...model import Calibration
from ...oracles.simpoly_source import PROFILE_CONTROLLED_INPUT_V1


class SPMChannelLike(Protocol):
    """Structural subset of public ``spmkit.core.models.SPMChannel`` used here."""

    name: str
    data: np.ndarray
    unit: str
    x_range: float
    y_range: float
    metadata: dict[str, Any]


def from_spm_channel(
    channel: SPMChannelLike,
    *,
    image_id: str | None = None,
    normalize_signal: bool = True,
) -> ScientificImage:
    """Translate a public SPMKit channel to Fathom's image contract.

    Spatial axis ranges determine anisotropic pixel calibration. Signal values
    are optionally mapped to [0, 1] for segmentation; this never changes the
    source channel and the original signal unit/range are recorded as metadata.
    """
    data = np.asarray(channel.data)
    if data.ndim != 2 or not data.size:
        raise ValueError("SPMKit channel must contain a non-empty 2D image")
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise ValueError("SPMKit channel data must be finite and numeric")
    height, width = data.shape
    if channel.x_range <= 0 or channel.y_range <= 0:
        raise ValueError("SPMKit channel spatial ranges must be positive")
    signal = data.astype(np.float64, copy=True)
    if normalize_signal:
        low, high = np.percentile(signal, (0.5, 99.5))
        signal = np.clip((signal - low) / max(high - low, np.finfo(float).eps), 0.0, 1.0)
    calibration = Calibration(
        pixel_size_x_m=float(channel.x_range) / width,
        pixel_size_y_m=float(channel.y_range) / height,
        source="SPMKIT_SPMCHANNEL_RANGES",
    )
    metadata = dict(channel.metadata)
    metadata.update(
        {
            "spmkit_channel_name": channel.name,
            "spmkit_signal_unit": channel.unit,
            "spmkit_signal_min": float(data.min()),
            "spmkit_signal_max": float(data.max()),
            "spmkit_signal_normalized_for_segmentation": normalize_signal,
        }
    )
    return FathomEngine().from_array(
        signal,
        calibration=calibration,
        image_id=image_id or channel.name,
        metadata=metadata,
    )


@dataclass(frozen=True)
class FathomAnalysisProvider:
    """SPMKit v1 ``Analysis`` plus an explicit callable operation."""

    name: str = "fathom-fibers"
    kinds: tuple[str, ...] = ("image",)

    def run(
        self,
        channel: SPMChannelLike,
        *,
        method: str = "fathom",
        roi_bbox: tuple[int, int, int, int] | None = None,
    ) -> FathomAnalysisResult | MethodComparisonResult | tuple[Any, Any]:
        image = from_spm_channel(channel)
        engine = FathomEngine()
        if method == "fathom":
            return engine.run_fathom(image, roi_bbox=roi_bbox)
        if method == "simpoly-controlled":
            return engine.run_simpoly(
                image,
                profile=PROFILE_CONTROLLED_INPUT_V1,
                roi_bbox=roi_bbox,
            )
        if method == "compare":
            return engine.compare_methods(image, roi_bbox=roi_bbox)
        raise ValueError(f"Unknown Fathom SPMKit method: {method}")


@dataclass(frozen=True)
class FathomDomain:
    """Object compatible with the public ``spmkit.plugins.v1`` Domain contract."""

    name: str = "Fathom Fibers"
    readers: tuple[Any, ...] = ()
    perspectives: tuple[str, ...] = ("fibers",)
    analyses: tuple[FathomAnalysisProvider, ...] = (FathomAnalysisProvider(),)


FATHOM_DOMAIN = FathomDomain()

