from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.oracles.matlab_compat import (
    matlab_adapthisteq_compat,
    matlab_canny_compat,
)
from fathom_fibers_quick.oracles.simpoly_source import (
    _bwareaopen_4_connected,
    _matlab_disk5_footprint,
    _matlab_histeq_default,
    _matlab_histogram_auto,
    _matlab_imbinarize,
    bwmorph_branchpoints,
    bwmorph_spur,
    bwmorph_thicken,
)
from fathom_fibers_quick.validation.campaign_worker import run_case
from fathom_fibers_quick.validation.manual_review import GridCellStatus, Manual5x5Review
from fathom_fibers_quick.validation.matlab_oracle import (
    MatlabOracle,
    oracle_cache_key,
    read_environment_report,
)
from fathom_fibers_quick.validation.parity_metrics import (
    boolean_parity,
    diameter_parity,
    first_divergence,
    float_parity,
    skeleton_parity,
)
from fathom_fibers_quick.validation.real_campaign import (
    build_review_queue,
    inventory_dataset,
    run_python_campaign,
)

pytestmark = pytest.mark.unit


def test_boolean_float_and_skeleton_metrics() -> None:
    matlab = np.array([[1, 0], [1, 0]], dtype=bool)
    python = np.array([[1, 1], [0, 0]], dtype=bool)
    masks = boolean_parity(matlab, python)
    assert masks["different_pixels"] == 2
    assert masks["dice"] == pytest.approx(0.5)
    floats = float_parity(matlab.astype(float), python.astype(float))
    assert floats["rmse"] == pytest.approx(np.sqrt(0.5))
    skeleton = skeleton_parity(matlab, python)
    assert skeleton["median_skeleton_displacement"] >= 0


def test_diameter_metrics_and_first_divergence() -> None:
    metrics = diameter_parity(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert metrics["matlab"]["n"] == 2
    assert metrics["wasserstein_distance"] == pytest.approx(0.5)
    assert first_divergence({"CROP": True, "CLAHE": False, "HISTEQ": False}) == "CLAHE"
    assert first_divergence({"CROP": True}) == "MATCHED"


def test_oracle_cache_key_is_complete_and_deterministic() -> None:
    arguments = {
        "source_tiff_sha256": "a" * 64,
        "matlab_release": "2026a",
        "matlab_source_sha256": "b" * 64,
        "profile": "CONTROLLED_INPUT",
        "conversion_ratio": 0.05204,
        "pipeline_version": "v1",
    }
    first = oracle_cache_key(**arguments)
    assert first == oracle_cache_key(**arguments)
    assert first != oracle_cache_key(**{**arguments, "conversion_ratio": 0.1})


def test_environment_report_parser(tmp_path: Path) -> None:
    path = tmp_path / "environment.json"
    path.write_text(json.dumps({"matlab_release": "2026a", "toolboxes": []}), encoding="utf-8")
    assert read_environment_report(path)["matlab_release"] == "2026a"


def test_matlab_morphology_rules_cross_validated_by_r2026a_probes() -> None:
    disk5 = _matlab_disk5_footprint()
    assert disk5.shape == (9, 9)
    assert np.count_nonzero(disk5) == 69

    diagonal_component = np.eye(20, dtype=bool)
    assert np.count_nonzero(_bwareaopen_4_connected(diagonal_component, 20)) == 20

    plus = np.zeros((15, 15), dtype=bool)
    plus[7, 2:13] = True
    plus[2:13, 7] = True
    branchpoints = bwmorph_branchpoints(plus)
    assert np.argwhere(branchpoints).tolist() == [[7, 7]]

    line = np.zeros((15, 15), dtype=bool)
    line[7, 2:13] = True
    line[6, 7] = True
    assert np.count_nonzero(bwmorph_spur(line, 1)) == 9
    assert np.count_nonzero(bwmorph_thicken(line, 4)) == 121


def test_matlab_histeq_mapping_has_64_target_levels() -> None:
    source = np.arange(256, dtype=np.uint8).reshape(16, 16) / 255.0
    result = _matlab_histeq_default(source, np.dtype("uint8"))
    assert len(np.unique(result)) == 64
    assert result.min() == 0
    assert result.max() == 1


def test_matlab_adapthisteq_uint8_behavioral_fixtures() -> None:
    constant = np.full((64, 64), 73, dtype=np.uint8)
    assert np.array_equal(matlab_adapthisteq_compat(constant), np.full_like(constant, 84))

    gradient_row = np.floor(np.linspace(0, 255, 64) + 0.5).astype(np.uint8)
    gradient = np.tile(gradient_row, (64, 1))
    result = matlab_adapthisteq_compat(gradient)
    assert hashlib.sha256(result.tobytes(order="C")).hexdigest() == (
        "f18509a992fe6256c1914cf14f15116263beb44f5f9989139945d0e369238511"
    )


def test_matlab_canny_behavioral_fixtures() -> None:
    vertical = np.zeros((128, 128), dtype=np.uint8)
    vertical[:, 64:] = 255
    result = matlab_canny_compat(vertical)
    assert np.count_nonzero(result) == 252
    assert hashlib.sha256(result.astype(np.uint8).tobytes(order="C")).hexdigest() == (
        "f84e1ca7b02c6cd3d2355b5fa6068a8be4cec2303401cf8f33462b3657d14218"
    )

    rectangle = np.zeros((128, 128), dtype=np.uint8)
    rectangle[31:96, 31:96] = 255
    result = matlab_canny_compat(rectangle)
    assert hashlib.sha256(result.astype(np.uint8).tobytes(order="C")).hexdigest() == (
        "64696f586e230e2e513915224bc05e47347cb8b6836ab7dac4c1e194a0df9056"
    )


def test_matlab_histogram_auto_scott_nice_edges() -> None:
    values = np.linspace(-3, 8, 10_000)
    counts, edges = _matlab_histogram_auto(values)
    assert np.array_equal(edges, np.arange(-3, 8.5, 0.5))
    assert counts.sum() == values.size


def test_matlab_imbinarize_excludes_values_equal_to_threshold() -> None:
    values = np.asarray([0.4, 0.5, 0.6])
    assert np.array_equal(_matlab_imbinarize(values, 0.5), [False, False, True])


def test_manual_5x5_reference_does_not_force_numeric_values() -> None:
    review = Manual5x5Review("ZEISS_001")
    assert len(review.cells) == 25
    review.cell(0, 0).set_status(GridCellStatus.MEASURED)
    review.cell(0, 1).set_status(GridCellStatus.NO_VALID_FIBER)
    review.cell(0, 2).set_status(GridCellStatus.SKIPPED_WITH_REASON, notes="crossing only")
    assert review.completed_count == 3
    assert review.measurement_count == 1
    with pytest.raises(ValueError, match="requires notes"):
        review.cell(0, 3).set_status(GridCellStatus.SKIPPED_WITH_REASON)


def test_campaign_inventory_refuses_any_count_other_than_16(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "only-one.tif").write_bytes(b"not opened because count gate runs first")
    with pytest.raises(RuntimeError, match="Expected exactly 16 TIFF files, found 1"):
        inventory_dataset(tmp_path, dataset)


def test_worker_records_failure_instead_of_dropping_case(tmp_path: Path) -> None:
    output = tmp_path / "summary.json"
    run_case(
        {
            "case_id": "ZEISS_001",
            "absolute_path": str(tmp_path / "missing.tif"),
            "conversion_um_per_px": 1.0,
        },
        output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert "missing.tif" in payload["error"]


def test_campaign_resume_preserves_completed_case(tmp_path: Path) -> None:
    root = tmp_path / ".validation/real-tiff-campaign"
    root.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps({"cases": [{"case_id": "ZEISS_001"}]}), encoding="utf-8"
    )
    output = root / "runs/python-latest/ZEISS_001/python_summary.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"status":"COMPLETE","sentinel":true}', encoding="utf-8")
    run_python_campaign(tmp_path, resume=True)
    assert json.loads(output.read_text(encoding="utf-8"))["sentinel"] is True


def test_review_queue_has_exactly_16_not_measured_rows(tmp_path: Path) -> None:
    root = tmp_path / ".validation/real-tiff-campaign"
    root.mkdir(parents=True)
    cases = [
        {
            "case_id": f"ZEISS_{index:03d}",
            "filename": f"image-{index}.tif",
            "conversion_um_per_px": 0.1,
            "resolution_class": "MID_MAG_GENERAL",
        }
        for index in range(1, 17)
    ]
    (root / "dataset_manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    rows = build_review_queue(tmp_path)
    assert len(rows) == 16
    assert {row["manual_status"] for row in rows} == {"NOT_MEASURED"}


@pytest.mark.matlab
@pytest.mark.external
def test_live_matlab_batch_when_explicitly_enabled() -> None:
    executable = os.environ.get("FATHOM_MATLAB_EXECUTABLE")
    if not executable:
        pytest.skip("FATHOM_MATLAB_EXECUTABLE is not set")
    repo = Path(__file__).resolve().parents[1]
    oracle = MatlabOracle.discover(repo)
    assert oracle is not None
    assert oracle.check(timeout=180)["available"] is True
