from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def test_source_commit_falls_back_to_git_head():
    from fathom_fibers_quick import _build_info

    commit = _build_info.source_commit()
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", commit)
    assert _build_info.embedded_commit() is None  # placeholder in source


def test_embedded_metadata_takes_priority():
    from fathom_fibers_quick import _build_info

    _build_info.BUILD_COMMIT = "a" * 40
    try:
        assert _build_info.source_commit() == "a" * 40
        assert _build_info.build_info()["embedded"] is True
    finally:
        _build_info.BUILD_COMMIT = "0" * 40


def test_version_consistent():
    from fathom_fibers_quick import __version__

    assert __version__ == "0.2.0rc2"
    pyproject = (REPO / "pyproject.toml").read_text()
    assert 'version = "0.2.0rc2"' in pyproject


def test_version_cli_output():
    result = subprocess.run(
        [sys.executable, "-m", "fathom_fibers_quick", "--version"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Fathom Fibers 0.2.0rc2" in result.stdout
    assert "commit " in result.stdout
    assert "platform " in result.stdout


def test_release_filename_scheme():
    import fathom_fibers_quick.release_scheme as scheme

    assert scheme.archive_name("linux", "x86_64").startswith("FathomFibers-0.2.0-rc2-linux-x86_64")
    assert scheme.archive_name("windows", "x86_64").startswith(
        "FathomFibers-0.2.0-rc2-windows-x86_64"
    )
    assert scheme.archive_name("macos", "arm64").startswith("FathomFibers-0.2.0-rc2-macos-arm64")


def test_readme_first_exists_and_mentions_core_flow():
    readme = REPO / "packaging/README_FIRST.md"
    assert readme.exists()
    text = readme.read_text()
    for needle in (
        "5-minute start",
        "Open Dataset",
        "Run missing",
        "Run all dataset",
        "Generate Dataset Scientific Report",
        "Export Analysis Bundle",
        "projected 2-D diameters",
        "not ground truth",
        "experimental",
    ):
        assert needle in text, needle
    # no private dataset markers in the packaged tutorial
    for marker in ("30-07-26", "PVDF Jose_", "/home/kegouro", "HIBRIS"):
        assert marker not in text


def test_archive_tree_windows_produces_real_zip(tmp_path):
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "build_release", _Path(__file__).resolve().parents[1] / "packaging/build_release.py"
    )
    build_release = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(build_release)
    archive_tree = build_release.archive_tree

    staging = tmp_path / "staging"
    (staging / "FathomFibers").mkdir(parents=True)
    (staging / "FathomFibers" / "FathomFibers.exe").write_bytes(b"\x4d\x5a fake")
    (staging / "README_FIRST.md").write_text("hello\n")
    (staging / "VERSION").write_text("0.2.0-rc1\ncommit deadbeef\n")
    destination = tmp_path / "FathomFibers-0.2.0-rc1-windows-amd64.zip"
    archive_tree(staging, destination, platform_tag="windows-amd64")
    assert destination.read_bytes()[:4] == b"PK\x03\x04"
    import zipfile

    with zipfile.ZipFile(destination) as handle:
        assert handle.testzip() is None
        names = handle.namelist()
    assert any("README_FIRST.md" in name for name in names)


def test_verify_release_rejects_private_content(tmp_path):
    import sys as _sys

    staged = tmp_path / "fake-release"
    (staged / "app").mkdir(parents=True)
    (staged / "app/FathomFibers").write_text("#!/bin/sh\nexit 0\n")
    (staged / "README_FIRST.md").write_text("# ok\n")
    (staged / "VERSION").write_text("0.2.0-rc1\ncommit abc\n")
    (staged / "private.txt").write_text("path: /home/kegouro/HIBRIS/private\n")

    script = REPO / "packaging/verify_release.py"
    result = subprocess.run(
        [_sys.executable, str(script), str(staged)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "private" in result.stdout.lower()


def test_verify_release_accepts_clean_staging(tmp_path):
    import sys as _sys

    staged = tmp_path / "clean-release"
    (staged / "app").mkdir(parents=True)
    (staged / "app/FathomFibers").write_text("#!/bin/sh\nexit 0\n")
    (staged / "README_FIRST.md").write_text("# Fathom Fibers — 5-minute start\n")
    (staged / "VERSION").write_text("0.2.0-rc1\ncommit abc\n")
    (staged / "LICENSE").write_text("MIT\n")

    script = REPO / "packaging/verify_release.py"
    result = subprocess.run(
        [_sys.executable, str(script), str(staged)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "release verification PASSED" in result.stdout


def test_obsolete_report_figures_removed_on_regeneration(tmp_path):
    from fathom_fibers_quick.api import FathomEngine
    from fathom_fibers_quick.model import Calibration
    from fathom_fibers_quick.reports import build_image_report
    from fathom_fibers_quick.workspace import WorkspaceCache

    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels, calibration=Calibration(5e-9, 5e-9, "test"), image_id="synthetic"
    )
    cache = WorkspaceCache(tmp_path)
    comparison = engine.compare_all_methods(image)
    cache.store_comparison("synthetic", comparison)
    loaded = cache.load_comparison("synthetic")

    output = tmp_path / "report"
    build_image_report(loaded, image, output_dir=output)
    # plant a legacy figure and regenerate
    (output / "figure-A-histogram.png").write_bytes(b"stale")
    build_image_report(loaded, image, output_dir=output)
    assert not (output / "figure-A-histogram.png").exists()
    text = (output / "index.html").read_text()
    for png in output.glob("*.png"):
        assert png.name in text, f"unreferenced figure {png.name}"
    assert "figure-A-histogram.png" not in text


def test_report_provenance_uses_build_commit(tmp_path):
    from fathom_fibers_quick import _build_info
    from fathom_fibers_quick.api import FathomEngine
    from fathom_fibers_quick.model import Calibration
    from fathom_fibers_quick.reports import build_image_report
    from fathom_fibers_quick.workspace import WorkspaceCache

    engine = FathomEngine()
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    image = engine.from_array(
        pixels, calibration=Calibration(5e-9, 5e-9, "test"), image_id="synthetic"
    )
    cache = WorkspaceCache(tmp_path)
    cache.store_comparison("synthetic", engine.compare_all_methods(image))
    loaded = cache.load_comparison("synthetic")

    _build_info.BUILD_COMMIT = "b" * 40
    try:
        out = build_image_report(loaded, image, output_dir=tmp_path / "report2")
    finally:
        _build_info.BUILD_COMMIT = "0" * 40
    assert "b" * 12 in out.read_text()


def test_changelog_and_ci_workflow_exist():
    assert (REPO / "CHANGELOG.md").exists()
    assert "0.2.0-rc1" in (REPO / "CHANGELOG.md").read_text()
    workflow = REPO / ".github/workflows/release-build.yml"
    assert workflow.exists()
    text = workflow.read_text()
    assert "ubuntu-latest" in text
    assert "windows-latest" in text
    assert "macos-latest" in text
    assert "workflow_dispatch" in text
    assert "build_release.py" in text


def test_private_dataset_names_absent_from_public_docs():
    for path in (REPO / "README.md", REPO / "packaging/README_FIRST.md"):
        text = path.read_text()
        assert "30-07-26" not in text
        assert "PVDF Jose_" not in text
