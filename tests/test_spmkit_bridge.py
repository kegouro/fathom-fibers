from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from fathom_fibers_quick.integrations.spmkit import (
    FATHOM_DOMAIN,
    FathomAnalysisProvider,
    from_spm_channel,
)


@dataclass
class DummySPMChannel:
    name: str = "Height"
    data: np.ndarray = field(
        default_factory=lambda: np.arange(80 * 100, dtype=float).reshape(80, 100)
    )
    unit: str = "m"
    x_range: float = 10e-6
    y_range: float = 4e-6
    metadata: dict = field(default_factory=lambda: {"direction": "forward"})


def test_spmkit_channel_translation_preserves_axes_and_context():
    channel = DummySPMChannel()
    image = from_spm_channel(channel)

    assert image.calibration.pixel_size_x_m == pytest.approx(100e-9)
    assert image.calibration.pixel_size_y_m == pytest.approx(50e-9)
    assert image.metadata["spmkit_signal_unit"] == "m"
    assert image.gray.min() == 0.0
    assert image.gray.max() == 1.0
    assert not np.shares_memory(image.gray, channel.data)


def test_domain_matches_public_v1_shape_without_spmkit_runtime_dependency():
    assert FATHOM_DOMAIN.name == "Fathom Fibers"
    assert FATHOM_DOMAIN.readers == ()
    assert FATHOM_DOMAIN.perspectives == ("fibers",)
    assert isinstance(FATHOM_DOMAIN.analyses[0], FathomAnalysisProvider)
