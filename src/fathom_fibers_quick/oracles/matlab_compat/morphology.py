from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import morphology


def _boolean_lut(encoded: str) -> np.ndarray:
    packed = np.frombuffer(bytes.fromhex(encoded), dtype=np.uint8)
    return np.unpackbits(packed, bitorder="little").astype(bool)


_THIN_1 = _boolean_lut(
    "00007f33000033330000ff310000ff330000ffff0000ffff0000ffff0000ffff"
    "0000ffff000023330000ffff0000ffff0000ffff000022ff0000ffff0000ffff"
)
_THIN_2 = _boolean_lut(
    "0000fff70000ffff0000fff50000ffff0000fff40000eeec0000fff40000eeec"
    "0000ffff0000efff0000ffff0000ffff0000fff40000eeec0000fef40000eeec"
)
_DIAG = _boolean_lut(
    "0044ffff0c4cffff0044ffff0c4cffff00ffffffffffffff0044ffffffffffff"
    "0044ffff0c4cffff0044ffff0c4cffff00ffffff0cffffff0044ffff0c4cffff"
)
_ISOLATED = _boolean_lut(
    "0000010000000000000000000000000000000000000000000000000000000000"
    "0000000000000000000000000000000000000000000000000000000000000000"
)
_DILATE = _boolean_lut(
    "feffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
)
_SINGLE = _boolean_lut(
    "1601010001000000010000000000000001000000000000000000000000000000"
    "0100000000000000000000000000000000000000000000000000000000000000"
)
_SPUR = _boolean_lut(
    "0000000000800000000000000080000000000000000000000088000000800000"
    "0000000000800000000000000080000000000000c080000000880000c07b0000"
)
_BRANCH_CANDIDATE = _boolean_lut(
    "000080e80000e8fe0000e8fe0000feff0000e8fe0000feff0000feff0000ffff"
    "0000e8fe0000feff0000feff0000ffff0000feff0000ffff0000ffff0000ffff"
)
_BACKGROUND_4_COUNT = np.frombuffer(
    bytes.fromhex(
        "0000000000000000000000000000000001010101010201010101020102020201"
        "0000000000000000000000000000000001020202010201010202030202020201"
        "0000000000000000000000000000000001020202020302020101020102020201"
        "0000000000000000000000000000000002030303020302020202030202020201"
        "0000000000000000000000000000000001020202020302020202030203030302"
        "0000000000000000000000000000000002030303020302020303040303030302"
        "0000000000000000000000000000000001020202020302020101020102020201"
        "0000000000000000000000000000000002030303020302020202030202020201"
        "0000000000000000000000000000000001020202020302020202030203030302"
        "0000000000000000000000000000000001020202010201010202030202020201"
        "0000000000000000000000000000000002030303030403030202030203030302"
        "0000000000000000000000000000000002030303020302020202030202020201"
        "0000000000000000000000000000000001020202020302020202030203030302"
        "0000000000000000000000000000000001020202010201010202030202020201"
        "0000000000000000000000000000000001020202020302020101020102020201"
        "0000000000000000000000000000000001020202010201010101020101010100"
    ),
    dtype=np.uint8,
)
_CODE_WEIGHTS = np.asarray(
    [[1 << (row + 3 * column) for column in range(3)] for row in range(3)],
    dtype=np.uint16,
)


def _apply_lut(mask: np.ndarray, table: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    codes = ndimage.correlate(
        source.astype(np.uint16), _CODE_WEIGHTS, mode="constant", cval=0
    )
    return table[codes]


def matlab_closing_compat(mask: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Match ``imclose`` background padding and same-size cropping."""
    source = np.asarray(mask, dtype=bool)
    neighborhood = np.asarray(footprint, dtype=bool)
    padding = tuple((int(np.ceil(size / 2)),) * 2 for size in neighborhood.shape)
    padded = np.pad(source, padding, mode="constant")
    closed = morphology.erosion(morphology.dilation(padded, neighborhood), neighborhood)
    slices = tuple(slice(before, -after) for before, after in padding)
    return closed[slices]


def matlab_thin_compat(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(iterations, 0)):
        updated = _apply_lut(_apply_lut(result, _THIN_1), _THIN_2)
        if np.array_equal(updated, result):
            break
        result = updated
    return result


def _thicken_once(mask: np.ndarray) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    isolated = _apply_lut(result, _ISOLATED)
    if isolated.any():
        result |= _apply_lut(result, _SINGLE) & _apply_lut(isolated, _DILATE)
    complement = np.ones((result.shape[0] + 4, result.shape[1] + 4), dtype=bool)
    complement[2:-2, 2:-2] = ~result
    thinned = _apply_lut(_apply_lut(complement, _THIN_1), _THIN_2)
    diagonal = _apply_lut(thinned, _DIAG)
    complement = (complement & ~thinned & diagonal) | thinned
    complement[:2, :] = True
    complement[-2:, :] = True
    complement[:, :2] = True
    complement[:, -2:] = True
    return ~complement[2:-2, 2:-2]


def matlab_thicken_compat(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(iterations, 0)):
        updated = _thicken_once(result)
        if np.array_equal(updated, result):
            break
        result = updated
    return result


def matlab_branchpoints_compat(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    candidates = _apply_lut(source, _BRANCH_CANDIDATE)
    background_components = _apply_lut(source, _BACKGROUND_4_COUNT)
    endpoints = background_components == 1
    final_candidates = candidates & ~endpoints
    vp_two = (background_components == 2) & ~endpoints
    vp_above_two = (background_components > 2) & ~endpoints
    neighboring_complex = _apply_lut(vp_above_two, _DILATE)
    return final_candidates & ~(final_candidates & vp_two & neighboring_complex)


def matlab_spur_compat(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    fields = (
        (slice(0, None, 2), slice(0, None, 2)),
        (slice(0, None, 2), slice(1, None, 2)),
        (slice(1, None, 2), slice(0, None, 2)),
        (slice(1, None, 2), slice(1, None, 2)),
    )
    for _ in range(max(iterations, 0)):
        complemented = ~result
        endpoints = _apply_lut(complemented, _SPUR)
        before = complemented.copy()
        for field_index, field in enumerate(fields):
            candidates = endpoints if field_index == 0 else endpoints & _apply_lut(
                complemented, _SPUR
            )
            complemented[field] ^= candidates[field]
        result = ~complemented
        if np.array_equal(complemented, before):
            break
    return result
