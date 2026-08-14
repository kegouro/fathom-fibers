from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear


def _matlab_derivative_gaussian(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    extent = int(np.ceil(4 * sigma))
    coordinates = np.arange(-extent, extent + 1, dtype=np.float64)
    gaussian = np.exp(-(coordinates**2) / (2 * sigma**2))
    gaussian /= gaussian.sum()
    derivative = np.gradient(gaussian)
    positive = derivative > 0
    negative = derivative < 0
    derivative[positive] /= derivative[positive].sum()
    derivative[negative] /= abs(derivative[negative].sum())
    # imfilter returns single for the uint8->im2single canonical path.
    return gaussian.astype(np.float32), derivative.astype(np.float32)


def matlab_canny_compat(
    image: np.ndarray,
    *,
    low_threshold: float = 0.2,
    high_threshold: float = 0.4,
    sigma: float = np.sqrt(2.0),
) -> np.ndarray:
    """Behavioral compatibility for R2026a ``edge(I,'Canny',[low high])``."""
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("image must be a nonempty 2-D array")
    if not 0 <= low_threshold < high_threshold < 1:
        raise ValueError("thresholds must satisfy 0 <= low < high < 1")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        normalized = (source.astype(np.float32) - info.min) / np.float32(
            info.max - info.min
        )
    else:
        normalized = source.astype(np.float32)

    gaussian, derivative = _matlab_derivative_gaussian(float(sigma))
    dx = ndimage.convolve1d(normalized, gaussian, axis=0, mode="nearest")
    dx = ndimage.convolve1d(dx, derivative, axis=1, mode="nearest")
    dy = ndimage.convolve1d(normalized, gaussian, axis=1, mode="nearest")
    dy = ndimage.convolve1d(dy, derivative, axis=0, mode="nearest")
    magnitude = np.hypot(dx, dy)
    maximum = float(magnitude.max())
    if maximum > 0:
        magnitude /= maximum

    valid = np.ones(source.shape, dtype=bool)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    local_maxima = _nonmaximum_suppression_bilinear(
        dy, dx, magnitude, valid, low_threshold
    )
    weak = local_maxima > 0
    labels, count = ndimage.label(weak, np.ones((3, 3), dtype=bool))
    if count == 0:
        return weak
    strong = weak & (magnitude > high_threshold)
    selected = np.zeros(count + 1, dtype=bool)
    selected[np.unique(labels[strong])] = True
    return selected[labels]
