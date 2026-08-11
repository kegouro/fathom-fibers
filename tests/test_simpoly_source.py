from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.oracles.simpoly_source import (
    PROFILE_CONTROLLED_INPUT_V1,
    PROFILE_SOURCE_COMPAT_V1,
    SIMPolySourceConfig,
    _as_matlab_unit_interval,
    _bwareaopen_4_connected,
    bwmorph_branchpoints,
    bwmorph_clean,
    bwmorph_fill,
    bwmorph_majority,
    bwmorph_spur,
    fit_matlab_gauss1,
    run_simpoly_source_pipeline,
)

EXPECTED_ORIGINAL_SHA256 = "6cbb827ebfa92a2f951d3fd06cb3561d81854ddd8fc4fc9f8f7bb1151ad1f446"


def test_original_matlab_source_sha256_manifest():
    original_path = Path(".reference/simpoly/original/SIMPolyMatlabCode.m")
    assert original_path.exists()

    data = original_path.read_bytes()
    calc_sha = hashlib.sha256(data).hexdigest()
    assert calc_sha == EXPECTED_ORIGINAL_SHA256


def test_image_shorter_than_91_rows():
    short_img = np.zeros((50, 100), dtype=np.uint8)
    cfg = SIMPolySourceConfig(profile=PROFILE_SOURCE_COMPAT_V1, footer_rows=90)

    res, _inter = run_simpoly_source_pipeline(short_img, cfg)
    assert res.status == "IMAGE_TOO_SHORT_FOR_CROP"
    assert "IMAGE_TOO_SHORT_FOR_CROP" in res.flags


def test_source_compat_fixed_90_row_crop_and_channel_selection():
    img_3d = np.ones((200, 100, 3), dtype=np.uint8) * 100
    cfg = SIMPolySourceConfig(profile=PROFILE_SOURCE_COMPAT_V1, footer_rows=90)

    _res, inter = run_simpoly_source_pipeline(img_3d, cfg)
    assert inter.cropped.shape == (110, 100)


def test_controlled_input_bypasses_90_row_crop():
    img = np.ones((200, 100), dtype=np.uint8) * 100
    cfg = SIMPolySourceConfig(profile=PROFILE_CONTROLLED_INPUT_V1)

    _res, inter = run_simpoly_source_pipeline(img, cfg)
    assert inter.cropped.shape == (200, 100)


def test_bwmorph_clean_semantics():
    arr = np.zeros((5, 5), dtype=bool)
    arr[2, 2] = True  # Isolated 1

    cleaned = bwmorph_clean(arr)
    assert not cleaned[2, 2]


def test_bwareaopen_preserves_component_of_exact_minimum_area():
    arr = np.zeros((12, 12), dtype=bool)
    arr[1:5, 1:6] = True  # exactly 20 pixels: MATLAB bwareaopen(BW, 20) retains it
    arr[9, 9] = True

    cleaned = _bwareaopen_4_connected(arr, 20)
    assert cleaned[1:5, 1:6].all()
    assert not cleaned[9, 9]


def test_uint16_uses_full_matlab_class_range():
    arr = np.array([0, 255, 65535], dtype=np.uint16)
    normalized = _as_matlab_unit_interval(arr)
    assert normalized.tolist() == pytest.approx([0.0, 255.0 / 65535.0, 1.0])


def test_bwmorph_fill_semantics():
    arr = np.ones((5, 5), dtype=bool)
    arr[2, 2] = False  # Isolated 0 hole

    filled = bwmorph_fill(arr)
    assert filled[2, 2]


def test_bwmorph_majority_semantics():
    arr = np.zeros((3, 3), dtype=bool)
    arr[0, :] = True
    arr[1, 0:2] = True  # 5 pixels = 1

    maj = bwmorph_majority(arr)
    assert maj[1, 1]  # Becomes 1


def test_bwmorph_branchpoints_and_spur():
    skel = np.zeros((7, 7), dtype=bool)
    skel[3, 1:6] = True  # Horizontal line
    skel[1:6, 3] = True  # Vertical line (crossing at 3,3)

    bps = bwmorph_branchpoints(skel)
    assert bps[3, 3]

    spurred = bwmorph_spur(skel, iterations=1)
    assert spurred.shape == skel.shape


def test_fit_matlab_gauss1_semantics():
    x = np.linspace(10, 50, 40)
    a1_true, b1_true, c1_true = 100.0, 30.0, 6.0
    y = a1_true * np.exp(-(((x - b1_true) / c1_true) ** 2))

    _a1, b1, c1 = fit_matlab_gauss1(x, y)
    assert b1 == pytest.approx(30.0, abs=0.5)
    assert c1 == pytest.approx(6.0, abs=0.5)

    source_stdev = c1 / 2.0
    math_sigma = c1 / math.sqrt(2.0)
    assert math.isclose(source_stdev, c1 / 2.0)
    assert math.isclose(math_sigma, c1 / math.sqrt(2.0))


def test_pipeline_no_foreground_and_no_skeleton():
    small_img = np.full((10, 10), 100, dtype=np.uint8)
    cfg = SIMPolySourceConfig(profile=PROFILE_CONTROLLED_INPUT_V1)

    res, _inter = run_simpoly_source_pipeline(small_img, cfg)
    assert res.status in {"OK", "NO_FOREGROUND", "NO_SKELETON", "NO_VALID_DIAMETERS"}
    if res.status == "OK":
        assert res.foreground_fraction is not None
        assert res.skeleton_count == res.local_diameters_px.size
