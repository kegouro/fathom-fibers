"""Frozen-build metadata (version, source commit, platform).

The build pipeline regenerates this module with the actual source commit and
timestamp before packaging; the committed placeholder values exist so source
execution and tests fall back to ``git HEAD`` at runtime.  Frozen builds read
the embedded values and never depend on a ``.git`` directory existing.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Replaced by packaging/build_release.py before PyInstaller runs.
BUILD_COMMIT: str = "0000000000000000000000000000000000000000"
BUILD_TIMESTAMP: str = ""
BUILD_PLATFORM: str = ""

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
    """Application version (PEP 440), kept in sync with pyproject.toml."""
    from . import __version__

    return __version__


def build_info() -> dict[str, str]:
    """Version, source commit, platform and build timestamp (where available)."""
    return {
        "version": application_version(),
        "commit": source_commit(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "build_timestamp": BUILD_TIMESTAMP,
        "embedded": bool(embedded_commit()),
    }


def describe() -> str:
    info = build_info()
    lines = [
        f"Fathom Fibers {info['version']}",
        f"commit {info['commit']}",
        f"platform {info['platform']} ({info['python']})",
    ]
    if info["build_timestamp"]:
        lines.append(f"built {info['build_timestamp']}")
    return "\n".join(lines)
