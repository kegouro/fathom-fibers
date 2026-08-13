#!/usr/bin/env python3
"""Verify a staged or archived Fathom Fibers release.

Checks: executable/app present, README_FIRST present, no private SEM data,
no private absolute source paths in packaged text, expected build metadata,
no legacy report artifacts, and runs the frozen smoke test for Linux.

Exit code 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

PRIVATE_PATTERNS = (
    r"/home/kegouro",
    r"HIBRIS",
    r"30-07-26",
    r"PVDF Jose_.*\.tif",
    r"local_data",
    r"\.validation/",
    r"\.reference",
)
LEGACY_REPORT_FIGURES = (
    "figure-A-histogram.png",
    "figure-histogram.png",
    "figure-ecdf.png",
    "figure-field-estimators.png",
    "figure-method-summary.png",
)


def extract_archive(path: Path) -> Path:
    target = Path(tempfile.mkdtemp(prefix="fathom-verify-"))
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as handle:
            handle.extractall(target)
    else:
        with tarfile.open(path) as handle:
            handle.extractall(target, filter="data")
    # archives contain a single top-level directory; use it as the root
    children = [child for child in target.iterdir() if child.is_dir()]
    if len(children) == 1:
        return children[0]
    return target


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path, help="Release archive or staging directory")
    parser.add_argument("--smoke", action="store_true", help="Run the frozen GUI smoke test (Linux)")
    args = parser.parse_args()

    source = args.release
    temporary = None
    if source.is_file():
        temporary = extract_archive(source)
        root = temporary
    else:
        root = source

    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)
        print(f"FAIL: {message}")

    # locate the app directory
    app_candidates = [root / "app", root]
    app_dir = next((candidate for candidate in app_candidates if candidate.is_dir()), None)
    if app_dir is None:
        fail("no app directory found")
        app_dir = root

    binary = app_dir / "FathomFibers" / "FathomFibers"
    if not binary.exists():
        binary = app_dir / "FathomFibers"
    if not binary.exists():
        fail("main executable not found")
    else:
        print(f"OK: executable {binary}")

    readme = next(iter(root.glob("README_FIRST.*")), None)
    if readme is None:
        fail("README_FIRST missing")
    else:
        print(f"OK: {readme.name}")

    version_file = root / "VERSION"
    if version_file.exists():
        print(f"OK: VERSION -> {version_file.read_text().strip()}")
    else:
        fail("VERSION / build info missing")

    private_hits: list[str] = []
    for path in iter_files(root):
        relative = path.relative_to(root).as_posix()
        if any(re.search(pattern, relative, re.IGNORECASE) for pattern in PRIVATE_PATTERNS):
            private_hits.append(relative)
    if private_hits:
        fail(f"private path markers in release contents: {private_hits[:5]}")
    else:
        print("OK: no private path markers in release contents")

    for path in iter_files(root):
        if path.suffix in {".md", ".txt", ".json", ".html", ".py", ".toml", ".cfg", ".ini", ".yml", ".yaml"}:
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            for pattern in PRIVATE_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    fail(f"private pattern {pattern!r} in {path.relative_to(root)}")
                    break

    legacy_hits = [
        path.relative_to(root).as_posix()
        for path in iter_files(root)
        if path.name in LEGACY_REPORT_FIGURES
    ]
    if legacy_hits:
        fail(f"legacy report figures shipped: {legacy_hits[:5]}")
    else:
        print("OK: no legacy report figures shipped")

    tiff_count = sum(1 for path in iter_files(root) if path.suffix.lower() in {".tif", ".tiff"})
    print(f"INFO: {tiff_count} TIFF files in release (skimage public fixtures are expected)")

    if args.smoke and binary.exists():
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [str(binary), "gui", "--smoke-test"],
            capture_output=True, text=True, env=env, timeout=240, check=False,
        )
        if result.returncode == 0:
            print("OK: frozen smoke test passed")
        else:
            fail(f"frozen smoke test failed: {result.stderr[-500:]}")
        version_result = subprocess.run(
            [str(binary), "--version"], capture_output=True, text=True, timeout=60, check=False
        )
        if version_result.returncode == 0 and "Fathom Fibers" in version_result.stdout:
            print(f"OK: --version -> {version_result.stdout.strip()}")
        else:
            fail("--version did not report application identity")

    if temporary is not None:
        import shutil

        shutil.rmtree(temporary)

    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nrelease verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
