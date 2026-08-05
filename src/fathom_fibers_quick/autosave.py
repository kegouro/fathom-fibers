from __future__ import annotations

import tempfile
from pathlib import Path

from .model import Project
from .project_io import load_project, save_project


def get_autosave_dir() -> Path:
    """Returns application user data directory for autosaves (~/.local/share/fathom-fibers/autosaves/)."""
    try:
        from platformdirs import user_data_dir

        base_dir = Path(user_data_dir("fathom-fibers", "pharos"))
    except ImportError:
        base_dir = Path.home() / ".local" / "share" / "fathom-fibers"

    autosave_dir = base_dir / "autosaves"
    autosave_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_gemini_autosaves(autosave_dir)
    return autosave_dir


def _migrate_legacy_gemini_autosaves(target_dir: Path) -> None:
    """Detects and moves old autosaves from ~/.gemini/antigravity-cli/autosaves/ if present."""
    legacy_dir = Path.home() / ".gemini" / "antigravity-cli" / "autosaves"
    if legacy_dir.exists() and legacy_dir.is_dir():
        for old_file in legacy_dir.glob("*.fiberquick.autosave.json"):
            new_file = target_dir / old_file.name
            if not new_file.exists():
                try:
                    old_file.replace(new_file)
                except OSError:
                    pass


def get_autosave_file_path(project: Project) -> Path:
    source_name = Path(project.image.path).stem
    sha = project.image.source_sha256[:12] if project.image.source_sha256 else "nosha"
    return get_autosave_dir() / f"{source_name}_{sha}.fiberquick.autosave.json"


def perform_atomic_autosave(project: Project) -> Path | None:
    """Saves project atomically to sidecar autosave file. Returns None on error without crashing."""
    try:
        path = get_autosave_file_path(project)
        temp_dir = path.parent
        temp_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
            temp_path = Path(tf.name)
            save_project(project, temp_path)

        temp_path.replace(path)
        return path
    except Exception as exc:
        print(f"Non-fatal autosave warning: {exc}")
        return None


def check_has_autosave(project: Project) -> tuple[bool, Path | None, float]:
    """Checks if a valid, newer autosave exists for the given project."""
    try:
        path = get_autosave_file_path(project)
        if not path.exists():
            return False, None, 0.0

        autosave_mtime = path.stat().st_mtime
        main_mtime = 0.0

        if project.project_path and Path(project.project_path).exists():
            main_mtime = Path(project.project_path).stat().st_mtime

        if autosave_mtime > main_mtime + 2.0:
            return True, path, autosave_mtime
        return False, path, autosave_mtime
    except Exception:
        return False, None, 0.0


def load_autosave(path: Path) -> Project:
    return load_project(path)


def clear_autosave(project: Project) -> None:
    try:
        path = get_autosave_file_path(project)
        if path.exists():
            path.unlink()
    except OSError:
        pass
