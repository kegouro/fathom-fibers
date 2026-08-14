"""Release artifact naming and versioning scheme (single source of truth)."""

from __future__ import annotations

VERSION = "0.2.0rc2"
HUMAN_VERSION = "0.2.0-rc2"
PRODUCT = "Fathom Fibers"


def archive_name(platform_tag: str, arch: str) -> str:
    return f"FathomFibers-{HUMAN_VERSION}-{platform_tag}-{arch}"


def artifact_extension(platform_tag: str) -> str:
    return ".zip" if platform_tag == "windows" else ".tar.gz"
