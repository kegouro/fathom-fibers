#!/usr/bin/env python3
"""Deterministic cross-platform Fathom Fibers release build.

Steps:
1. determine platform + architecture;
2. determine version and source commit;
3. write build metadata into src/fathom_fibers_quick/_build_info.py;
4. clean only packaging output directories;
5. run PyInstaller (native, no cross-compilation);
6. run the frozen smoke test where supported;
7. restore the committed build-info placeholder;
8. assemble the portable release directory;
9. archive it;
10. calculate SHA256;
11. print the exact artifact path.

Windows/macOS must run natively on their own OS (CI runners); this script
does not cross-compile.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPEC = Path(__file__).resolve().parent / "fathom-fibers.spec"
BUILD_INFO = REPO / "src/fathom_fibers_quick/_build_info.py"
DIST_DIR = REPO / "dist"
BUILD_DIR = REPO / "build"
RELEASE_DIR = REPO / "release"
STAGING_DIR = RELEASE_DIR / "staging"

BUILD_INFO_TEMPLATE = '''"""Frozen-build metadata (version, source commit, platform).

The build pipeline regenerates this module with the actual source commit and
timestamp before packaging; the committed placeholder values exist so source
execution and tests fall back to ``git HEAD`` at runtime.  Frozen builds read
the embedded values and never depend on a ``.git`` directory existing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .release_scheme import VERSION

# Replaced by packaging/build_release.py before PyInstaller runs.
BUILD_COMMIT: str = "{commit}"
BUILD_TIMESTAMP: str = "{timestamp}"
BUILD_PLATFORM: str = "{platform}"

_PLACEHOLDER = "0" * 40


def embedded_commit() -> str | None:
    commit = BUILD_COMMIT.strip()
    return commit if commit and commit != _PLACEHOLDER else None


def source_commit() -> str:
    """Best available source commit: embedded build metadata first, git HEAD
    as a development fallback, then ``unknown`` outside a git checkout."""
    embedded = embedded_commit()
    if embedded:
        return embedded
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and commit:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def application_version() -> str:
    """Package version (PEP 440) for provenance and diagnostics."""
    return VERSION


def build_info() -> dict[str, str]:
    """Version, source commit, platform and build timestamp (where available)."""
    return {{
        "version": application_version(),
        "commit": source_commit(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "build_timestamp": BUILD_TIMESTAMP,
        "embedded": bool(embedded_commit()),
    }}


def describe() -> str:
    info = build_info()
    lines = [
        f"Fathom Fibers {{info['version']}}",
        f"commit {{info['commit']}}",
        f"platform {{info['platform']}} ({{info['python']}})",
    ]
    if info["build_timestamp"]:
        lines.append(f"built {{info['build_timestamp']}}")
    return "\\n".join(lines)
'''


def git_commit() -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_clean() -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout.strip()


def write_build_info(commit: str, platform_tag: str) -> None:
    from datetime import UTC, datetime

    content = BUILD_INFO_TEMPLATE.format(
        commit=commit,
        timestamp=datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        platform=platform_tag,
    )
    BUILD_INFO.write_text(content, encoding="utf-8")


def restore_build_info() -> None:
    subprocess.run(
        ["git", "-C", str(REPO), "checkout", "--", str(BUILD_INFO.relative_to(REPO))],
        capture_output=True,
        check=False,
    )


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def frozen_smoke(app_path: Path) -> bool:
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [str(app_path), "gui", "--smoke-test"],
        capture_output=True,
        text=True,
        env=env,
        timeout=240,
        check=False,
    )
    return result.returncode == 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_tree(source: Path, destination: Path, *, platform_tag: str) -> Path:
    if platform_tag.startswith("windows"):
        with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as handle:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    handle.write(path, path.relative_to(source.parent))
        return destination
    with tarfile.open(destination, "w:gz") as handle:
        handle.add(source, arcname=source.name)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the frozen smoke test")
    args = parser.parse_args()

    if not git_clean():
        print("WARNING: worktree is not clean; build metadata may not reflect a released state.")
    system = sys.platform
    if system == "linux":
        arch = platform.machine() or "x86_64"
        platform_tag = f"linux-{arch}"
    elif system == "darwin":
        arch = platform.machine() or "arm64"
        platform_tag = f"macos-{arch}"
    elif system == "win32":
        arch = os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64").lower()
        platform_tag = f"windows-{arch}"
    else:
        print(f"Unsupported platform: {system}", file=sys.stderr)
        return 2

    commit = git_commit()
    version = "0.2.0-rc2"
    artifact_name = f"FathomFibers-{version}-{platform_tag}"

    for directory in (DIST_DIR, BUILD_DIR):
        if directory.exists():
            shutil.rmtree(directory)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] platform={platform_tag} version={version} commit={commit[:12]}")
    write_build_info(commit, platform_tag)
    try:
        print("[2/8] running PyInstaller (native)")
        build_env = dict(os.environ)
        # pin the package to this checkout so an editable install in the
        # interpreter's environment cannot shadow the release source
        build_env["PYTHONPATH"] = str(REPO / "src")
        run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--clean",
                "--distpath",
                str(DIST_DIR),
                "--workpath",
                str(BUILD_DIR),
                str(SPEC),
            ],
            env=build_env,
        )

        app_dir = DIST_DIR / "FathomFibers"
        if not app_dir.exists():
            print("PyInstaller output not found", file=sys.stderr)
            return 3

        print("[3/8] frozen smoke test")
        if not args.skip_smoke and not frozen_smoke(app_dir / "FathomFibers"):
            print("frozen smoke test FAILED", file=sys.stderr)
            return 4

        print("[4/8] removing stale developer distribution metadata")
        internal = app_dir / "_internal"
        for dist_info in internal.glob("fathom_fibers_quick-*.dist-info"):
            shutil.rmtree(dist_info)
        print("[4b/8] restoring committed build-info placeholder")
    finally:
        restore_build_info()

    print("[5/8] assembling portable release directory")
    staging = STAGING_DIR / artifact_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(app_dir, staging / "app")
    readme_first = Path(__file__).resolve().parent / "README_FIRST.md"
    shutil.copy(readme_first, staging / "README_FIRST.md")
    license_file = REPO / "LICENSE"
    if license_file.exists():
        shutil.copy(license_file, staging / "LICENSE")
    changelog = REPO / "CHANGELOG.md"
    if changelog.exists():
        shutil.copy(changelog, staging / "CHANGELOG.md")
    (staging / "VERSION").write_text(f"{version}\ncommit {commit}\n", encoding="utf-8")

    print("[6/8] archiving")
    artifact = (
        RELEASE_DIR / f"{artifact_name}.tar.gz"
        if platform_tag.startswith(("linux", "macos"))
        else RELEASE_DIR / f"{artifact_name}.zip"
    )
    if artifact.exists():
        artifact.unlink()
    archive_tree(staging, artifact, platform_tag=platform_tag)

    print("[7/8] checksum")
    digest = sha256(artifact)
    (RELEASE_DIR / f"{artifact.name}.sha256").write_text(
        f"{digest}  {artifact.name}\n", encoding="utf-8"
    )

    print(f"[8/8] artifact: {artifact}")
    print(f"size: {artifact.stat().st_size} bytes")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
