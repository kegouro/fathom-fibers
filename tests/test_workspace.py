from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.core.methods import MethodId
from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.validation.manual_review import GridCellStatus
from fathom_fibers_quick.workspace import (
    VALIDATION_ROOT,
    Manual5x5Store,
    WorkspaceCache,
    load_workspace_dataset,
    resolve_matlab_cache_root,
)


def synthetic_image(engine: FathomEngine):
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    return engine.from_array(
        pixels,
        calibration=Calibration(5e-9, 5e-9, "test"),
        image_id="synthetic",
    )


def write_tiff(directory: Path, name: str) -> Path:
    import tifffile

    pixels = np.zeros((48, 64), dtype=np.uint8)
    pixels[18:30, 8:56] = 200
    path = directory / name
    tifffile.imwrite(path, pixels)
    return path


def test_core_workspace_imports_do_not_import_qt(tmp_path):
    import subprocess
    import sys

    env = {"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    code = (
        "import sys; import fathom_fibers_quick.workspace; "
        "import fathom_fibers_quick.reports; "
        "assert 'PySide6' not in sys.modules, sys.modules.keys(); "
        "print('qt-free ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-free ok" in result.stdout


def test_load_workspace_dataset_scans_tiff_directory(tmp_path):
    for index in range(1, 5):
        write_tiff(tmp_path, f"PVDF Jose_{index:02d}.tif")
    dataset = load_workspace_dataset(tmp_path, repo=tmp_path)
    assert dataset.dataset_id == f"DATASET_{tmp_path.name}"
    assert len(dataset.images) == 4
    assert dataset.images[0].case_id == "ZEISS_001"
    assert dataset.images[0].stem == "PVDF Jose_01"
    assert dataset.images[-1].case_id == "ZEISS_004"


def test_load_workspace_dataset_prefers_frozen_manifest(tmp_path):
    images = {}
    for index in range(1, 4):
        images[f"image-{index}.tif"] = write_tiff(tmp_path, f"image-{index}.tif")
    cases = [
        {
            "case_id": f"CANON_{index:03d}",
            "filename": f"image-{index}.tif",
            "absolute_path": str(images[f"image-{index}.tif"]),
            "sha256": f"abc{index}",
        }
        for index in range(1, 4)
    ]
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps({"dataset_id": "CANONICAL_DATASET", "case_count": 3, "cases": cases})
    )
    dataset = load_workspace_dataset(tmp_path, manifest_path=manifest, repo=tmp_path)
    assert dataset.dataset_id == "CANONICAL_DATASET"
    assert [image.case_id for image in dataset.images] == ["CANON_001", "CANON_002", "CANON_003"]
    assert dataset.images[0].sha256 == "abc1"


def test_matlab_cache_root_resolution(tmp_path):
    cache_dir = tmp_path / ".validation/real-tiff-campaign"
    cache_dir.mkdir(parents=True)
    (cache_dir / "dataset_manifest.json").write_text("{}")
    resolved = resolve_matlab_cache_root(dataset_dir=tmp_path / "local_data/zeiss/d", repo=tmp_path)
    assert resolved == cache_dir
    env_resolved = resolve_matlab_cache_root(repo=tmp_path, env_value=str(cache_dir))
    assert env_resolved == cache_dir


def test_full_cache_round_trip_preserves_arrays(tmp_path):
    engine = FathomEngine()
    image = synthetic_image(engine)
    comparison = engine.compare_all_methods(image)
    cache = WorkspaceCache(tmp_path)
    cache.store_comparison("synthetic", comparison)
    assert cache.has_full("synthetic")
    loaded = cache.load_comparison("synthetic")
    assert loaded is not None
    left = {result.method_id: result for result in comparison.results}
    right = {result.method_id: result for result in loaded.results}
    assert set(left) == set(right)
    for method, result in left.items():
        assert result.status == right[method].status
        left_common = result.common_distribution
        right_common = right[method].common_distribution
        assert (left_common is None) == (right_common is None)
        if left_common is not None:
            assert np.allclose(left_common.diameter, right_common.diameter)
            assert np.allclose(left_common.weight, right_common.weight)
        if result.local_samples:
            assert set(result.local_samples) == set(right[method].local_samples)
            for key, value in result.local_samples.items():
                left_array = np.asarray(value)
                right_array = np.asarray(right[method].local_samples[key])
                if left_array.dtype.kind in {"U", "S"}:
                    assert np.array_equal(left_array, right_array), key
                else:
                    np.testing.assert_allclose(left_array, right_array)
    assert len(loaded.agreements) == len(comparison.agreements)


def test_full_cache_round_trip_preserves_secondary_distributions(tmp_path):
    from fathom_fibers_quick.core.distributions import DiameterDistribution
    from fathom_fibers_quick.core.methods import (
        Capability,
        CapabilityState,
        Estimand,
        MethodCapabilities,
    )
    from fathom_fibers_quick.unified_comparison import compare_method_results

    engine = FathomEngine()
    image = synthetic_image(engine)
    comparison = engine.compare_all_methods(image)
    original = next(
        result for result in comparison.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )
    if not original.secondary_distributions:
        edge = DiameterDistribution(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0]), "um",
            Estimand.FATHOM_FIELD_PAIRED_EDGE_DIAMETER, MethodId.FATHOM_FIELD_GRAPH_V1,
        )
        from fathom_fibers_quick.core.methods import MethodResult

        original = MethodResult(
            original.method_id, original.method_version, original.image_id,
            original.calibration, original.valid_roi, original.unit,
            MethodCapabilities({Capability.MASK: CapabilityState.AVAILABLE}),
            original.status, original.native_estimand, original.native_result,
            original.native_statistics, original.native_distribution,
            original.common_distribution,
            {"FATHOM_FIELD_PAIRED_EDGE_DIAMETER": edge},
            original.fiber_balanced_distribution, mask=original.mask,
            centerline=original.centerline,
            local_samples=original.local_samples,
            quality_flags=original.quality_flags,
        )
        comparison = compare_method_results(
            tuple(
                original if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1 else result
                for result in comparison.results
            )
        )
    cache = WorkspaceCache(tmp_path)
    cache.store_comparison("synthetic-secondary", comparison)
    loaded = cache.load_comparison("synthetic-secondary")
    assert loaded is not None
    loaded_field = next(
        result for result in loaded.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )
    assert loaded_field.secondary_distributions.keys() == original.secondary_distributions.keys()
    for name, distribution in original.secondary_distributions.items():
        restored = loaded_field.secondary_distributions[name]
        assert np.allclose(distribution.diameter, restored.diameter)
        assert np.allclose(distribution.weight, restored.weight)


def test_field_local_samples_expose_edges_and_flags(tmp_path):
    engine = FathomEngine()
    image = synthetic_image(engine)
    comparison = engine.compare_all_methods(image)
    field = next(
        result for result in comparison.results
        if result.method_id == MethodId.FATHOM_FIELD_GRAPH_V1
    )
    samples = field.local_samples
    n = int(samples["x_m"].size)
    assert samples["minus_xy_m"].shape == (n, 2)
    assert samples["radius_minus_um"].shape == (n,)
    assert samples["edge_flags"].shape == (n,)
    assert samples["profile_flags"].shape == (n,)
    assert samples["profile_minus_u_um"].shape == (n,)


def test_fathom_local_sections_exposed(tmp_path):
    engine = FathomEngine()
    image = synthetic_image(engine)
    comparison = engine.compare_all_methods(image)
    local = next(
        result for result in comparison.results if result.method_id == MethodId.FATHOM_LOCAL
    )
    assert local.local_samples is not None
    assert local.local_samples["section_width_um"].size > 0
    assert "section_flags" in local.local_samples


def test_manual_store_round_trip_and_atomic_save(tmp_path):
    store = Manual5x5Store(tmp_path, "ZEISS_PVDF_2026-07-30")
    review = store.ensure_review("ZEISS_001")
    cell = review.cell(1, 2)
    cell.status = GridCellStatus.MEASURED
    cell.diameter = 0.812
    cell.calibration_snapshot = {"pixel_size_x_m": 5e-9}
    cell.notes = "ok"
    store.save()
    assert (tmp_path / "manual5x5" / "ZEISS_PVDF_2026-07-30.json").exists()

    reloaded = Manual5x5Store(tmp_path, "ZEISS_PVDF_2026-07-30")
    reloaded.load()
    restored = reloaded.reviews["ZEISS_001"].cell(1, 2)
    assert restored.status == GridCellStatus.MEASURED
    assert restored.diameter == 0.812
    assert restored.position == "R2C3"
    assert reloaded.total_measured == 1
    reloaded.set_image_status("ZEISS_001", "REVIEWED")
    assert reloaded.reviewed_images == 1


def test_summary_payload_indexing(tmp_path):
    from fathom_fibers_quick.workspace import summary_method_rows

    payload = {
        "results": [
            {"method_id": "PYTHON_SIMPOLY", "status": "COMPLETE"},
            {"method_id": "NOT_A_METHOD", "status": "FAILED"},
        ]
    }
    rows = summary_method_rows(payload)
    assert MethodId.PYTHON_SIMPOLY in rows
    assert len(rows) == 1


def test_reports_build_headless(tmp_path):
    from fathom_fibers_quick.reports import build_dataset_report, build_image_report
    from fathom_fibers_quick.unified_comparison import build_image_report as wrap
    from fathom_fibers_quick.validation.unified_methods import _comparison_payload

    engine = FathomEngine()
    image = synthetic_image(engine)
    comparison = engine.compare_all_methods(image)
    image_report = build_image_report(comparison, image, output_dir=tmp_path / "report-image")
    assert image_report.exists()
    assert (image_report.parent / "figure-histogram.png").exists()
    assert (image_report.parent / "figure-ecdf.png").exists()
    assert (image_report.parent / "figure-field-estimators.png").exists()
    assert (image_report.parent / "figure-method-summary.png").exists()
    text = image_report.read_text()
    assert "Method summary" in text
    assert "INCOMPLETE REFERENCE" in text
    assert "Agreement and consensus pseudo-reference are not ground truth" in text

    runs = tmp_path / VALIDATION_ROOT / "runs"
    runs.mkdir(parents=True)
    payload = _comparison_payload(wrap(comparison))
    payload["image_id"] = "PVDF Jose_01.tif"
    (runs / "PVDF Jose_01.json").write_text(json.dumps(payload, default=str))
    from fathom_fibers_quick.workspace import WorkspaceDataset, WorkspaceImage

    dataset = WorkspaceDataset(
        "ZEISS_PVDF_2026-07-30",
        (WorkspaceImage("ZEISS_001", "PVDF Jose_01.tif", Path("/tmp/x.tif")),),
    )
    dataset_report = build_dataset_report(
        tmp_path, dataset=dataset, manual_store=Manual5x5Store(tmp_path, dataset.dataset_id)
    )
    assert dataset_report.exists()
    assert (dataset_report.parent / "images/PVDF Jose_01/index.html").exists()
    assert (dataset_report.parent / "dataset-medians.png").exists()
    assert "16-image processing matrix" in dataset_report.read_text()
