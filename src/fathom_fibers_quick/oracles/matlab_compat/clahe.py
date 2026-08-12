from __future__ import annotations

import numpy as np


def _round_nonnegative(values: np.ndarray | float) -> np.ndarray:
    """Round nonnegative values as MATLAB integer conversion does."""
    return np.floor(np.asarray(values) + 0.5)


def _clip_and_redistribute(histogram: np.ndarray, limit: int) -> np.ndarray:
    """Redistribute clipped counts evenly without changing the tile population."""
    hist = np.asarray(histogram, dtype=np.int64).copy()
    bins = hist.size
    excess = int(np.maximum(hist - limit, 0).sum())
    increment = excess // bins
    upper = limit - increment

    above = hist > limit
    middle = (~above) & (hist > upper)
    low = ~(above | middle)
    hist[above] = limit
    excess -= int((limit - hist[middle]).sum())
    hist[middle] = limit
    hist[low] += increment
    excess -= int(low.sum()) * increment

    start = 0
    while excess:
        step = max(bins // excess, 1)
        for index in range(start, bins, step):
            if hist[index] < limit:
                hist[index] += 1
                excess -= 1
                if not excess:
                    break
        start = (start + 1) % bins
    return hist


def _integer_bin_indices(image: np.ndarray, bins: int, maximum: int) -> np.ndarray:
    values = np.asarray(image, dtype=np.uint64)
    indices = (values * bins) // (maximum + 1)
    return np.minimum(indices, bins - 1).astype(np.int64)


def _padded_tile_geometry(shape: tuple[int, int], tiles: tuple[int, int]) -> tuple[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    tile_shape: list[int] = []
    pads: list[tuple[int, int]] = []
    for size, count in zip(shape, tiles, strict=True):
        tile = (size + count - 1) // count
        if tile % 2:
            tile += 1
        total = tile * count - size
        pads.append((total // 2, total - total // 2))
        tile_shape.append(tile)
    return (tile_shape[0], tile_shape[1]), (pads[0], pads[1])


def matlab_adapthisteq_compat(
    image: np.ndarray,
    *,
    num_tiles: tuple[int, int] = (8, 8),
    clip_limit: float = 0.01,
    nbins: int = 256,
) -> np.ndarray:
    """Reproduce R2026a ``adapthisteq`` defaults for unsigned integer images.

    This is a behavioral compatibility primitive, not a universal CLAHE API.
    Its tile geometry, clipping, count redistribution, integer quantization,
    and interpolation are independently exercised against MATLAB probes.
    """
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("image must be a nonempty 2-D array")
    if source.dtype not in (np.dtype(np.uint8), np.dtype(np.uint16)):
        raise TypeError("MATLAB compatibility is verified for uint8 and uint16")
    if len(num_tiles) != 2 or min(num_tiles) < 2:
        raise ValueError("num_tiles must contain two integers >= 2")
    if not 0 <= clip_limit <= 1:
        raise ValueError("clip_limit must lie in [0, 1]")
    if nbins < 1:
        raise ValueError("nbins must be positive")

    tile_shape, pads = _padded_tile_geometry(source.shape, num_tiles)
    padded = np.pad(source, pads, mode="symmetric")
    tile_pixels = tile_shape[0] * tile_shape[1]
    minimum_limit = (tile_pixels + nbins - 1) // nbins
    actual_limit = minimum_limit + int(
        _round_nonnegative(clip_limit * (tile_pixels - minimum_limit))
    )
    maximum = int(np.iinfo(source.dtype).max)
    bin_image = _integer_bin_indices(padded, nbins, maximum)

    mappings = np.empty((num_tiles[0], num_tiles[1], nbins), dtype=np.int64)
    for tile_row in range(num_tiles[0]):
        row0 = tile_row * tile_shape[0]
        for tile_col in range(num_tiles[1]):
            col0 = tile_col * tile_shape[1]
            tile = bin_image[
                row0 : row0 + tile_shape[0], col0 : col0 + tile_shape[1]
            ]
            histogram = np.bincount(tile.ravel(), minlength=nbins)
            clipped = _clip_and_redistribute(histogram, actual_limit)
            cumulative = np.cumsum(clipped, dtype=np.int64)
            mappings[tile_row, tile_col] = _round_nonnegative(
                cumulative * maximum / tile_pixels
            ).astype(np.int64)

    result = np.empty(padded.shape, dtype=np.int64)
    row_start = 0
    half_rows, half_cols = tile_shape[0] // 2, tile_shape[1] // 2
    for block_row in range(num_tiles[0] + 1):
        if block_row == 0:
            block_rows, upper, lower = half_rows, 0, 0
        elif block_row == num_tiles[0]:
            block_rows, upper, lower = half_rows, num_tiles[0] - 1, num_tiles[0] - 1
        else:
            block_rows, upper, lower = tile_shape[0], block_row - 1, block_row
        row_weight = np.arange(block_rows, dtype=np.int64)[:, None]
        reverse_row = block_rows - row_weight
        col_start = 0
        for block_col in range(num_tiles[1] + 1):
            if block_col == 0:
                block_cols, left, right = half_cols, 0, 0
            elif block_col == num_tiles[1]:
                block_cols, left, right = half_cols, num_tiles[1] - 1, num_tiles[1] - 1
            else:
                block_cols, left, right = tile_shape[1], block_col - 1, block_col
            col_weight = np.arange(block_cols, dtype=np.int64)[None, :]
            reverse_col = block_cols - col_weight
            region = bin_image[
                row_start : row_start + block_rows,
                col_start : col_start + block_cols,
            ]
            upper_left = mappings[upper, left][region]
            upper_right = mappings[upper, right][region]
            lower_left = mappings[lower, left][region]
            lower_right = mappings[lower, right][region]
            numerator = reverse_row * (
                reverse_col * upper_left + col_weight * upper_right
            ) + row_weight * (reverse_col * lower_left + col_weight * lower_right)
            result[
                row_start : row_start + block_rows,
                col_start : col_start + block_cols,
            ] = _round_nonnegative(numerator / (block_rows * block_cols)).astype(np.int64)
            col_start += block_cols
        row_start += block_rows

    row_pad, col_pad = pads
    unpadded = result[
        row_pad[0] : result.shape[0] - row_pad[1] if row_pad[1] else None,
        col_pad[0] : result.shape[1] - col_pad[1] if col_pad[1] else None,
    ]
    return np.clip(unpadded, 0, maximum).astype(source.dtype)
