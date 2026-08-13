from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fathom_fibers_quick.export_bundle import export_analysis_bundle
from fathom_fibers_quick.reports import (
    build_final_dataset_report,
    build_image_report,
    series_distributions,
)
from fathom_fibers_quick.workspace import WorkspaceCache

pytestmark = pytest.mark.qt


def synthetic_workspace(tmp_path: Path) -> tuple[Path, WorkspaceCache, Any]:
    from fathom_fibers_quick.api import FathomEngine

    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels,
        calibration=__import__("fathom_fibers_quick.model", fromlist=["Calibration"]).Calibration(
            5e-9, 5e-9, "test"
        ),
        image_id="synthetic",
    )
    comparison = engine.compare_all_methods(image)
    cache = WorkspaceCache(tmp_path)
    cache.store_comparison("synthetic", comparison)
    return tmp_path, cache, image


def test_series_distributions_include_ribbon_within_field_family():
    from fathom_fibers_quick.workspace import WorkspaceCache

    cache = WorkspaceCache()
    comparison = cache.load_comparison("PVDF Jose_01")
    assert comparison is not None
    series = series_distributions(comparison)
    names = {name for name, _dist in series}
    assert {"Ribbon Refined EDT", "Ribbon Refined Edge", "Ribbon Refined Profile"} <= names
    assert "Fathom Field (EDT)" in names
    # the ribbon series belong to the field family, not to independent methods
    assert "Ribbon Refined EDT" in names


def test_final_dataset_report_sections(tmp_path):
    from fathom_fibers_quick.workspace import WorkspaceDataset, WorkspaceImage

    _repo, cache, _image = synthetic_workspace(tmp_path)
    from fathom_fibers_quick.workspace import Manual5x5Store

    dataset = WorkspaceDataset(
        "SYNTHETIC",
        (WorkspaceImage("SYN_001", "synthetic.tif", tmp_path / "synthetic.tif"),),
    )
    out = build_final_dataset_report(
        _repo,
        dataset=dataset,
        manual_store=Manual5x5Store(tmp_path / "cache-root", "SYNTHETIC"),
        output_dir=tmp_path / "final-report",
        comparisons=[cache.load_comparison("synthetic")],
    )
    text = out.read_text()
    for section in (
        "Scientific Morphological Fiber Analysis",
        "Oriented Ribbon V1",
        "EXPERIMENTAL",
        "Method comparisons",
        "Limitations",
        "Provenance",
    ):
        assert section in text, section
    assert (out.parent / "dataset-figure-A.png").exists()
    assert "INCOMPLETE REFERENCE" in text


def test_final_report_manual_incomplete_does_not_block(tmp_path):
    from fathom_fibers_quick.workspace import WorkspaceDataset, WorkspaceImage

    _repo, cache, _image = synthetic_workspace(tmp_path)
    from fathom_fibers_quick.workspace import Manual5x5Store

    dataset = WorkspaceDataset(
        "SYNTHETIC",
        (WorkspaceImage("SYN_001", "synthetic.tif", tmp_path / "synthetic.tif"),),
    )
    out = build_final_dataset_report(
        _repo,
        dataset=dataset,
        manual_store=Manual5x5Store(tmp_path / "cache-root", "SYNTHETIC"),
        output_dir=tmp_path / "final-report-2",
        comparisons=[cache.load_comparison("synthetic")],
    )
    assert out.exists()
    text = out.read_text()
    assert "0 / 400" in text
    assert "INCOMPLETE REFERENCE" in text


def test_image_report_includes_ribbon_series(tmp_path):
    _repo, cache, image = synthetic_workspace(tmp_path)
    comparison = cache.load_comparison("synthetic")
    out = build_image_report(comparison, image, output_dir=tmp_path / "image-report")
    assert out.exists()
    assert (out.parent / "figure-histogram.png").exists()


def test_export_bundle_schemas(tmp_path):
    from fathom_fibers_quick.workspace import WorkspaceDataset, WorkspaceImage

    _repo, _cache, _image = synthetic_workspace(tmp_path)
    dataset = WorkspaceDataset(
        "SYNTHETIC",
        (WorkspaceImage("SYN_001", "synthetic.tif", tmp_path / "synthetic.tif"),),
    )
    root = export_analysis_bundle(
        _repo,
        dataset=dataset,
        manual_store=None,
        output_dir=tmp_path / "bundle",
    )
    with open(root / "results/dataset_summary.csv") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows, "dataset summary must not be empty"
    methods = {row["method"] for row in summary_rows}
    assert "Fathom Field" in methods
    assert {"Refined EDT", "Refined Edge", "Refined Profile"} <= {row["estimator"] for row in summary_rows}
    with open(root / "results/measurements.csv") as handle:
        measurement_rows = list(csv.DictReader(handle))
    assert len(measurement_rows) > 10
    measurement_methods = {row["method"] for row in measurement_rows}
    assert "Fathom Oriented Ribbon V1" in measurement_methods
    for row in measurement_rows[:50]:
        assert row["image_id"] == "synthetic"
        assert row["method"] in {"Fathom Field", "Fathom Oriented Ribbon V1", "Python SIMPoly", "Fathom Local"}
    method_results = json.loads((root / "results/method_results.json").read_text())
    assert method_results["bundle_version"]
    assert "images" in method_results
    provenance = json.loads((root / "results/provenance.json").read_text())
    assert provenance["application"] == "Fathom Fibers"
    assert "matlab" in provenance
    assert provenance["ribbon"]["status"] == "EXPERIMENTAL"


def test_measurements_csv_has_no_invalid_json_nan(tmp_path):
    from fathom_fibers_quick.workspace import WorkspaceDataset, WorkspaceImage

    _repo, _cache, _image = synthetic_workspace(tmp_path)
    dataset = WorkspaceDataset(
        "SYNTHETIC",
        (WorkspaceImage("SYN_001", "synthetic.tif", tmp_path / "synthetic.tif"),),
    )
    root = export_analysis_bundle(
        _repo,
        dataset=dataset,
        manual_store=None,
        output_dir=tmp_path / "bundle2",
    )
    text = (root / "results/method_results.json").read_text()
    json.loads(text)  # must parse
    assert "NaN" not in text
    assert "Infinity" not in text


def test_release_package_smoke():
    """The frozen app exists and its smoke test passes (skips without build)."""
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    binary = repo / "dist" / "FathomFibers" / "FathomFibers"
    if not binary.exists():
        pytest.skip("PyInstaller build not present; run the packaging step first")
    env = dict(__import__("os").environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [str(binary), "gui", "--smoke-test"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    help_result = subprocess.run(
        [str(binary), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "validation" in help_result.stdout


def test_release_spec_and_launcher_exist():
    repo = Path(__file__).resolve().parents[1]
    assert (repo / "packaging/fathom-fibers.spec").exists()
    assert (repo / "packaging/launcher.py").exists()
    spec = (repo / "packaging/fathom-fibers.spec").read_text()
    assert "PySide6" in spec or "collect_data_files" in spec
