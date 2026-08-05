from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .model import Project, compute_measurement_width
from .zeiss import file_sha256


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
    """Recalculate width_m from geometry & calibration for all measurements in project.

    Returns the number of measurements whose stored width_m differed beyond tolerance.
    """
    corrections_count = 0
    calibration = project.image.calibration
    calibration.validate()

    for measurement in project.measurements:
        recalculated_width = compute_measurement_width(measurement.p1, measurement.p2, calibration)
        if not math.isclose(measurement.width_m, recalculated_width, abs_tol=tolerance, rel_tol=1e-5):
            measurement.width_m = recalculated_width
            corrections_count += 1
    return corrections_count


def save_project(project: Project, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".fiberquick.json":
        if path.suffix:
            path = path.with_suffix(path.suffix + ".fiberquick.json")
        else:
            path = path.with_suffix(".fiberquick.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = project.to_dict()
    payload["project_path"] = str(path.resolve())
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    project.project_path = str(path.resolve())
    return path


def load_project(path: str | Path) -> Project:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    project = Project.from_dict(data)
    project.project_path = str(path.resolve())
    recalculate_and_validate_project(project)
    return project
