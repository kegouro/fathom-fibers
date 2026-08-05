from __future__ import annotations

import tempfile
from pathlib import Path

from .model import Project
from .project_io import load_project, save_project


def get_autosave_dir() -> Path:
    app_dir = Path.home() / ".gemini" / "antigravity-cli" / "autosaves"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_autosave_file_path(project: Project) -> Path:
    source_name = Path(project.image.path).stem
    sha = project.image.source_sha256[:12] if project.image.source_sha256 else "nosha"
    return get_autosave_dir() / f"{source_name}_{sha}.fiberquick.autosave.json"


def perform_atomic_autosave(project: Project) -> Path:
    """Saves project atomically to sidecar autosave file."""
    path = get_autosave_file_path(project)
    temp_dir = path.parent
    temp_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", dir=temp_dir, delete=False, encoding="utf-8") as tf:
        temp_path = Path(tf.name)
        save_project(project, temp_path)

    temp_path.replace(path)
    return path


def check_has_autosave(project: Project) -> tuple[bool, Path | None, float]:
    """Checks if a valid, newer autosave exists for the given project."""
    path = get_autosave_file_path(project)
    if not path.exists():
        return False, None, 0.0

    autosave_mtime = path.stat().st_mtime
    main_mtime = 0.0

    if project.project_path and Path(project.project_path).exists():
        main_mtime = Path(project.project_path).stat().st_mtime

    # Valid if autosave exists and is newer than main project file
    if autosave_mtime > main_mtime + 2.0:
        return True, path, autosave_mtime
    return False, path, autosave_mtime


def load_autosave(path: Path) -> Project:
    return load_project(path)


def clear_autosave(project: Project) -> None:
    path = get_autosave_file_path(project)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
