from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .measurement_geometry import (
    compute_angle_geometry,
    compute_polyline_geometry,
)
from .measurement_records import MeasurementKind
from .model import Project, compute_measurement_width
from .zeiss import file_sha256

CURRENT_PROJECT_SCHEMA = 4


class SourceVerificationStatus(str, Enum):
    MATCH = "MATCH"
    MISSING = "MISSING"
    MISMATCH = "MISMATCH"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True, slots=True)
class SourceVerificationResult:
    status: SourceVerificationStatus
    message: str
    image_path: Path
    expected_sha256: str | None = None
    actual_sha256: str | None = None


def verify_project_source(project: Project) -> SourceVerificationResult:
    image_path = Path(project.image.path)
    if not image_path.exists():
        return SourceVerificationResult(
            status=SourceVerificationStatus.MISSING,
            message=f"La imagen fuente no existe en la ruta: {image_path}",
            image_path=image_path,
            expected_sha256=project.image.source_sha256,
        )

    expected = project.image.source_sha256
    if not expected:
        return SourceVerificationResult(
            status=SourceVerificationStatus.UNVERIFIED,
            message="El proyecto no contiene un SHA-256 de referencia para verificar la fuente.",
            image_path=image_path,
        )

    actual = file_sha256(image_path)
    if actual == expected:
        return SourceVerificationResult(
            status=SourceVerificationStatus.MATCH,
            message="Verificación exitosa: el SHA-256 coincide con el registrado.",
            image_path=image_path,
            expected_sha256=expected,
            actual_sha256=actual,
        )

    return SourceVerificationResult(
        status=SourceVerificationStatus.MISMATCH,
        message=f"El SHA-256 de la imagen ({actual}) difiere del registrado en el proyecto ({expected}).",
        image_path=image_path,
        expected_sha256=expected,
        actual_sha256=actual,
    )


def recalculate_and_validate_project(project: Project, tolerance: float = 1e-12) -> int:
    """Recalculate derived values from geometry & calibration for all records in project.

    Returns the number of records whose derived values differed beyond tolerance.
    """
    corrections_count = 0
    calibration = project.image.calibration
    calibration.validate()

    for r in project.records:
        if r.kind in {MeasurementKind.PROJECTED_WIDTH, MeasurementKind.DISTANCE}:
            p1, p2 = r.p1, r.p2
            recalc_m = compute_measurement_width(p1, p2, calibration)
            old_val = r.primary_value or 0.0
            if not math.isclose(old_val, recalc_m, abs_tol=tolerance, rel_tol=1e-5):
                r.values["length_m"] = recalc_m
                r.values["width_m"] = recalc_m
                corrections_count += 1

        elif r.kind == MeasurementKind.POLYLINE_LENGTH:
            pts = r.geometry.get("points", [])
            poly_info = compute_polyline_geometry(pts, calibration)
            r.values.update(poly_info)

        elif r.kind == MeasurementKind.ANGLE:
            pt_a = r.geometry.get("pt_a", (0.0, 0.0))
            pt_b = r.geometry.get("pt_b", (0.0, 0.0))
            pt_c = r.geometry.get("pt_c", (0.0, 0.0))
            ang_info = compute_angle_geometry(pt_a, pt_b, pt_c, calibration)
            r.values.update(ang_info)

    return corrections_count


def _legacy_backup_if_needed(path: Path) -> Path | None:
    if not path.exists():
        return None
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        version = int(existing.get("schema_version", 1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if version >= CURRENT_PROJECT_SCHEMA:
        return None
    backup = path.with_name(f"{path.name}.schema-v{version}.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def save_project(
    project: Project,
    path: str | Path,
    *,
    create_legacy_backup: bool = True,
) -> Path:
    path = Path(path)
    if not path.name.lower().endswith(".fiberquick.json"):
        if path.suffix:
            path = path.with_suffix(path.suffix + ".fiberquick.json")
        else:
            path = path.with_suffix(".fiberquick.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    if create_legacy_backup:
        _legacy_backup_if_needed(path)
    payload = project.to_dict()
    payload["project_path"] = str(path.resolve())
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        "w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        encoding="utf-8",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if temporary.exists():
            temporary.unlink()
    project.project_path = str(path.resolve())
    project.schema_version = CURRENT_PROJECT_SCHEMA
    return path


def load_project(path: str | Path) -> Project:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    project = Project.from_dict(data)
    project.project_path = str(path.resolve())
    recalculate_and_validate_project(project)
    return project
