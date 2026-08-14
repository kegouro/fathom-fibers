from __future__ import annotations

import concurrent.futures
import csv
import html
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from ..api import FathomEngine
from ..auto_roi import get_preset_for_calibration
from ..zeiss import inspect_image
from .matlab_oracle import CANONICAL_SOURCE_SHA256, MatlabOracle, oracle_cache_key
from .parity_metrics import first_divergence

DATASET_ID = "ZEISS_PVDF_2026-07-30"
EXPECTED_CASES = 16
DEFAULT_DATASET = Path("data/zeiss")
PIPELINE_VERSION = "MATLAB_ORACLE_R2026A_V1"

INVENTORY_FIELDS = (
    "case_id",
    "filename",
    "absolute_path",
    "sha256",
    "file_size_bytes",
    "width_px",
    "height_px",
    "channels",
    "dtype",
    "zeiss_metadata_status",
    "footer_start_row",
    "footer_height_px",
    "pixel_size_x",
    "pixel_size_y",
    "physical_unit",
    "magnification",
    "accelerating_voltage",
    "working_distance",
    "detector",
    "source_reader",
    "reader_flags",
)


def campaign_root(repo: Path) -> Path:
    return repo / ".validation/real-tiff-campaign"


def inventory_dataset(repo: Path, dataset: Path | None = None) -> dict[str, Any]:
    root = (dataset or repo / DEFAULT_DATASET).resolve()
    paths = sorted((*root.glob("*.tif"), *root.glob("*.tiff")), key=lambda path: path.name)
    if len(paths) != EXPECTED_CASES:
        names = "\n".join(path.name for path in paths)
        raise RuntimeError(f"Expected exactly 16 TIFF files, found {len(paths)}:\n{names}")
    output_root = campaign_root(repo)
    input_root = output_root / "controlled-inputs"
    output_root.mkdir(parents=True, exist_ok=True)
    input_root.mkdir(parents=True, exist_ok=True)
    engine = FathomEngine()
    rows: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for index, path in enumerate(paths, 1):
        case_id = f"ZEISS_{index:03d}"
        info = inspect_image(path, compute_hash=True)
        image = engine.open_image(path)
        pixels = np.asarray(image.pixels)
        controlled = pixels[: image.valid_body.shape[0], :]
        if controlled.ndim == 3:
            controlled = controlled[:, :, 0]
        controlled_path = input_root / f"{case_id}.mat"
        savemat(controlled_path, {"controlled_input": controlled}, do_compression=True)
        metadata = info.get("metadata", {})
        footer_start = image.footer_bounds[0] if image.footer_bounds else image.shape[0]
        footer_height = image.footer_bounds[1] - footer_start if image.footer_bounds else 0
        row = {
            "case_id": case_id,
            "filename": path.name,
            "absolute_path": str(path),
            "sha256": info["sha256"],
            "file_size_bytes": path.stat().st_size,
            "width_px": info["width_px"],
            "height_px": info["height_px"],
            "channels": pixels.shape[2] if pixels.ndim == 3 else 1,
            "dtype": str(pixels.dtype),
            "zeiss_metadata_status": "PRESENT"
            if info["format_id"] == "zeiss_cz_sem_tiff"
            else "ABSENT",
            "footer_start_row": footer_start,
            "footer_height_px": footer_height,
            "pixel_size_x": image.calibration.pixel_size_x_m,
            "pixel_size_y": image.calibration.pixel_size_y_m,
            "physical_unit": "m",
            "magnification": metadata.get("ap_mag", ""),
            "accelerating_voltage": metadata.get("ap_actualkv", ""),
            "working_distance": metadata.get("ap_wd", ""),
            "detector": metadata.get("dp_detector_channel", ""),
            "source_reader": info["format_id"],
            "reader_flags": "" if image.footer_bounds else "FOOTER_NOT_DETECTED",
        }
        rows.append(row)
        cases.append(
            {
                **row,
                "controlled_input_path": str(controlled_path.resolve()),
                "conversion_um_per_px": image.calibration.pixel_size_x_m * 1e6,
                "resolution_class": get_preset_for_calibration(image.calibration).name,
                "matlab_cache_key_source": oracle_cache_key(
                    source_tiff_sha256=info["sha256"],
                    matlab_release="2026a",
                    matlab_source_sha256=CANONICAL_SOURCE_SHA256,
                    profile="SIMPOLY_SOURCE_COMPAT_V1",
                    conversion_ratio=image.calibration.pixel_size_x_m * 1e6,
                    pipeline_version=PIPELINE_VERSION,
                ),
                "matlab_cache_key_controlled": oracle_cache_key(
                    source_tiff_sha256=info["sha256"],
                    matlab_release="2026a",
                    matlab_source_sha256=CANONICAL_SOURCE_SHA256,
                    profile="SIMPOLY_CONTROLLED_INPUT_V1",
                    conversion_ratio=image.calibration.pixel_size_x_m * 1e6,
                    pipeline_version=PIPELINE_VERSION,
                ),
            }
        )
    with (output_root / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "dataset_id": DATASET_ID,
        "case_count": len(cases),
        "root": str(root),
        "inventory_generation_timestamp": datetime.now(UTC).isoformat(),
        "fathom_commit": head,
        "source_matlab_sha256": CANONICAL_SOURCE_SHA256,
        "pipeline_version": PIPELINE_VERSION,
        "cases": cases,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _selected_cases(
    manifest: dict[str, Any], case: str | None, limit: int | None
) -> list[dict[str, Any]]:
    cases = list(manifest["cases"])
    if case:
        cases = [item for item in cases if item["case_id"] == case or item["filename"] == case]
        if not cases:
            raise ValueError(f"Unknown campaign case: {case}")
    return cases[:limit] if limit else cases


def run_python_campaign(
    repo: Path,
    *,
    case: str | None = None,
    limit: int | None = None,
    resume: bool = False,
    force: bool = False,
    timeout: float = 900,
    workers: int = 1,
) -> Path:
    root = campaign_root(repo)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_dir = root / "runs" / "python-latest"
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict[str, Any], Path, Path]] = []
    for item in _selected_cases(manifest, case, limit):
        case_dir = run_dir / item["case_id"]
        output = case_dir / "python_summary.json"
        if output.exists() and resume and not force:
            continue
        case_dir.mkdir(parents=True, exist_ok=True)
        case_json = case_dir / "case.json"
        case_json.write_text(json.dumps(item, indent=2), encoding="utf-8")
        jobs.append((item, case_json, output))

    def execute(job: tuple[dict[str, Any], Path, Path]) -> None:
        item, case_json, output = job
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fathom_fibers_quick.validation.campaign_worker",
                    str(case_json),
                    str(output),
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output.write_text(
                json.dumps(
                    {
                        "case_id": item["case_id"],
                        "status": "FAILED",
                        "error": f"Worker timeout after {timeout} seconds",
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    }
                ),
                encoding="utf-8",
            )
            return
        if completed.returncode != 0:
            output.write_text(
                json.dumps(
                    {
                        "case_id": item["case_id"],
                        "status": "FAILED",
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    }
                ),
                encoding="utf-8",
            )

    if workers == 1:
        for job in jobs:
            execute(job)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(execute, jobs))
    return run_dir


def run_matlab_campaign(repo: Path, *, timeout: float = 7200, force: bool = False) -> Path:
    oracle = MatlabOracle.discover(repo)
    if oracle is None:
        raise RuntimeError("MATLAB executable is unavailable")
    root = campaign_root(repo)
    run_dir = repo / ".validation/matlab-oracle/runs/r2026a-latest"
    expression = (
        f"addpath('{oracle.harness_dir.as_posix()}');"
        f"run_simpoly_campaign('{(root / 'dataset_manifest.json').as_posix()}',"
        f"'{run_dir.as_posix()}',{str(force).lower()});"
    )
    completed = oracle.batch(expression, timeout=timeout)
    (run_dir / "matlab.log").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "matlab.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"MATLAB campaign failed ({completed.returncode}): {completed.stderr}")
    return run_dir


def run_detailed_parity_campaign(
    repo: Path, *, timeout: float = 900, cleanup_intermediates: bool = True
) -> Path:
    """Compare controlled-input arrays one case at a time and release temporary MAT files."""
    root = campaign_root(repo)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    output_root = root / "parity-details"
    for item in manifest["cases"]:
        case_dir = root / "runs/python-latest" / item["case_id"]
        case_path = case_dir / "case.json"
        case_path.write_text(json.dumps(item, indent=2), encoding="utf-8")
        matlab_path = (
            repo
            / ".validation/matlab-oracle/runs/r2026a-latest"
            / item["case_id"]
            / "controlled/intermediates.mat"
        )
        output = output_root / f"{item['case_id']}.json"
        if not matlab_path.exists():
            output.write_text(
                json.dumps(
                    {
                        "case_id": item["case_id"],
                        "status": "FAILED",
                        "error": f"Missing MATLAB intermediates: {matlab_path}",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "fathom_fibers_quick.validation.parity_worker",
                    str(case_path),
                    str(matlab_path),
                    str(output),
                ],
                cwd=repo,
                text=True,
                capture_output=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "PYTHONPATH": f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}",
                },
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            output.write_text(
                json.dumps(
                    {
                        "case_id": item["case_id"],
                        "status": "FAILED",
                        "error": f"Detailed parity timeout after {timeout} seconds",
                        "stdout": exc.stdout,
                        "stderr": exc.stderr,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        if completed.returncode:
            output.write_text(
                json.dumps(
                    {
                        "case_id": item["case_id"],
                        "status": "FAILED",
                        "stdout": completed.stdout,
                        "stderr": completed.stderr,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif cleanup_intermediates:
            matlab_path.unlink()
    return output_root


STAGE_NAMES = {
    "CROP": "I",
    "CLAHE": "Ihist_after_clahe",
    "HISTEQ": "Ihist_after_histeq",
    "EROSION": "marker",
    "RECONSTRUCTION": "Iobr",
    "CANNY": "E_canny",
    "AREA_FILTER": "E_area_filtered",
    "EDGE_THICKEN": "E_thickened",
    "THRESHOLD": "BW_threshold",
    "CLOSING": "BW_closed",
    "CLEAN": "BW_clean",
    "FILL": "BW_fill",
    "MAJORITY": "BW_majority",
    "THIN": "BW_thin",
    "MEDIAN_LOOP": "BW_median",
    "THICKEN": "BW_thickened",
    "SKELETON": "SK_bwskel",
    "BRANCHPOINTS": "branchpoints",
    "BRANCH_GUARD": "branch_guard",
    "SPUR": "SK_after_spur",
    "EDGE_DISTANCE_FILTER": "SK_valid",
    "EDT": "Dist",
    "DIAMETERS": "diameters",
    "HISTOGRAM": "hist_values",
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _profile_parity(matlab: dict[str, Any], python: dict[str, Any]) -> dict[str, Any]:
    stage_matches: dict[str, bool] = {}
    stage_metrics: dict[str, Any] = {}
    for stage, name in STAGE_NAMES.items():
        ms = matlab.get("stages", {}).get(name, {})
        ps = python.get("stages", {}).get(name, {})
        matched = bool(ms and ps and ms.get("sha256") == ps.get("sha256"))
        stage_matches[stage] = matched
        stage_metrics[stage] = {
            "matched_hash": matched,
            "matlab_foreground": ms.get("foreground_count"),
            "python_foreground": ps.get("foreground_count"),
            "matlab_mean": ms.get("mean"),
            "python_mean": ps.get("mean"),
        }
    threshold_delta = (
        abs(float(matlab["threshold_level"]) - float(python["threshold_level"]))
        if matlab.get("threshold_level") is not None and python.get("threshold_level") is not None
        else None
    )
    stage_matches["OTSU"] = threshold_delta is not None and threshold_delta <= 1e-12
    stage_metrics["OTSU"] = {"absolute_difference": threshold_delta}
    matlab_b1, python_b1 = matlab.get("gauss_b1"), python.get("gauss_b1")
    matlab_c1, python_c1 = matlab.get("gauss_c1"), python.get("gauss_c1")
    b1_delta = (
        abs(matlab_b1 - python_b1) if matlab_b1 is not None and python_b1 is not None else None
    )
    c1_delta = (
        abs(matlab_c1 - python_c1) if matlab_c1 is not None and python_c1 is not None else None
    )
    stage_matches["GAUSSIAN_FIT"] = b1_delta == 0 and c1_delta == 0
    stage_metrics["GAUSSIAN_FIT"] = {
        "matlab_b1": matlab_b1,
        "python_b1": python_b1,
        "b1_absolute_difference": b1_delta,
        "b1_relative_difference_percent": (
            100 * b1_delta / abs(matlab_b1) if b1_delta is not None and matlab_b1 else None
        ),
        "matlab_c1": matlab_c1,
        "python_c1": python_c1,
        "c1_absolute_difference": c1_delta,
        "c1_relative_difference_percent": (
            100 * c1_delta / abs(matlab_c1) if c1_delta is not None and matlab_c1 else None
        ),
    }
    matlab_hist = matlab.get("hist_values", [])
    python_hist = python.get("hist_values", [])
    matlab_edges = matlab.get("hist_edges", [])
    python_edges = python.get("hist_edges", [])
    stage_metrics["HISTOGRAM"].update(
        matlab_bin_count=len(matlab_hist),
        python_bin_count=len(python_hist),
        counts_equal=matlab_hist == python_hist,
        edges_equal=matlab_edges == python_edges,
    )
    return {
        "status": "COMPARED" if matlab.get("status") == "COMPLETE" and python else "FAILED",
        "first_divergence": first_divergence(stage_matches),
        "stage_matches": stage_matches,
        "stage_metrics": stage_metrics,
        "diameters": {
            "matlab": {
                "n": matlab.get("diameter_n"),
                "mean": matlab.get("diameter_mean"),
                "median": matlab.get("diameter_median"),
                "std": matlab.get("diameter_std"),
                "quantiles": matlab.get("diameter_quantiles", []),
            },
            "python": {
                "n": python.get("diameter_n"),
                "mean": python.get("diameter_mean_px"),
                "median": python.get("diameter_median_px"),
                "std": python.get("diameter_std_px"),
                "quantiles": python.get("diameter_quantiles_px", []),
            },
        },
    }


def compile_parity(repo: Path) -> dict[str, Any]:
    root = campaign_root(repo)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    cases: list[dict[str, Any]] = []
    for item in manifest["cases"]:
        py = _read_json(root / "runs/python-latest" / item["case_id"] / "python_summary.json") or {}
        matlab_root = repo / ".validation/matlab-oracle/runs/r2026a-latest" / item["case_id"]
        matlab_source = _read_json(matlab_root / "source_compat/summary.json") or {}
        matlab_controlled = _read_json(matlab_root / "controlled/summary.json") or {}
        controlled = _profile_parity(matlab_controlled, py.get("controlled_input", {}))
        detail = _read_json(root / "parity-details" / f"{item['case_id']}.json") or {}
        if detail.get("status") == "COMPLETE":
            controlled["first_divergence"] = detail["first_divergence"]
            controlled["detailed_metrics"] = detail
        cases.append(
            {
                "case_id": item["case_id"],
                "filename": item["filename"],
                "source_compat": _profile_parity(matlab_source, py.get("source_compat", {})),
                "controlled_input": controlled,
                "fathom": py.get("fathom", {}),
            }
        )
    payload = {"dataset_id": DATASET_ID, "case_count": len(cases), "cases": cases}
    (root / "parity_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_review_queue(repo: Path) -> list[dict[str, Any]]:
    root = campaign_root(repo)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    queue_path = root / "review_queue.csv"
    previous: dict[str, dict[str, str]] = {}
    if queue_path.exists():
        with queue_path.open(newline="", encoding="utf-8") as handle:
            previous = {row["case_id"]: row for row in csv.DictReader(handle)}
    grid_counts: dict[str, int] = {}
    grid_path = root / "manual_grid.json"
    if grid_path.exists():
        stored_grids = json.loads(grid_path.read_text(encoding="utf-8"))
        grid_counts = {
            case_id: sum(cell.get("status") == "MEASURED" for cell in review.get("cells", []))
            for case_id, review in stored_grids.items()
        }
    rows: list[dict[str, Any]] = []
    for item in manifest["cases"]:
        py = _read_json(root / "runs/python-latest" / item["case_id"] / "python_summary.json") or {}
        matlab = (
            _read_json(
                repo
                / ".validation/matlab-oracle/runs/r2026a-latest"
                / item["case_id"]
                / "controlled"
                / "summary.json"
            )
            or {}
        )
        py_controlled = py.get("controlled_input", {})
        fathom = py.get("fathom", {})
        matlab_center = matlab.get("gauss_b1")
        python_center = py_controlled.get("gauss_b1")
        fathom_center = fathom.get("section_median_um")
        mp_diff = (
            100 * abs(python_center - matlab_center) / abs(matlab_center)
            if matlab_center not in {None, 0} and python_center is not None
            else None
        )
        method_diff = (
            100 * abs(fathom_center - matlab_center) / abs(matlab_center)
            if matlab_center not in {None, 0} and fathom_center is not None
            else None
        )
        flags = sorted(set(py_controlled.get("flags", [])) | set(fathom.get("flags", [])))
        reasons: list[str] = []
        if mp_diff is None:
            reasons.append("MATLAB_PYTHON_RESULT_MISSING")
        elif mp_diff > 5:
            reasons.append(f"MATLAB_PYTHON_DIVERGENCE_{mp_diff:.1f}%")
        if method_diff is None:
            reasons.append("FATHOM_RESULT_MISSING")
        elif method_diff > 30:
            reasons.append(f"FATHOM_SIMPOLY_METHOD_DIFFERENCE_{method_diff:.1f}%")
        if fathom.get("resolution_status") != "RESOLUTION_OK":
            reasons.append(str(fathom.get("resolution_status")))
        if flags:
            reasons.append("AUTOMATIC_QUALITY_FLAGS_PRESENT")
        high = (
            mp_diff is None
            or mp_diff > 5
            or method_diff is None
            or method_diff > 30
            or fathom.get("resolution_status") != "RESOLUTION_OK"
            or bool(flags)
        )
        medium = not high and (mp_diff > 1 or method_diff > 15)
        manual = previous.get(item["case_id"], {})
        rows.append(
            {
                "case_id": item["case_id"],
                "filename": item["filename"],
                "pixel_size": item["conversion_um_per_px"],
                "resolution_class": item["resolution_class"],
                "MATLAB_SIMPoly_center": matlab_center,
                "Python_SIMPoly_center": python_center,
                "Fathom_section_median": fathom_center,
                "MATLAB_Python_difference_percent": mp_diff,
                "Fathom_SIMPoly_method_difference_percent": method_diff,
                "algorithm_agreement": "UNKNOWN"
                if mp_diff is None
                else "CLOSE"
                if mp_diff <= 5
                else "DIVERGENT",
                "quality_flags": ";".join(flags),
                "manual_priority": "HIGH" if high else "MEDIUM" if medium else "LOW",
                "manual_status": manual.get("manual_status", "NOT_MEASURED"),
                "manual_measurement_count": grid_counts.get(
                    item["case_id"], int(manual.get("manual_measurement_count", 0))
                ),
                "manual_target_count": 25,
                "review_notes": manual.get("review_notes")
                or ("AUTO: " + "; ".join(reasons) if reasons else ""),
            }
        )
    with queue_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate_selected_diagnostics(repo: Path, case_id: str = "ZEISS_001") -> Path | None:
    """Render full diagnostics for one selected divergent case, not every TIFF."""
    matlab_path = repo / ".validation/matlab-oracle/controlled-first/intermediates.mat"
    if case_id != "ZEISS_001" or not matlab_path.exists():
        return None
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from scipy.io import loadmat

    from ..oracles.simpoly_source import (
        PROFILE_CONTROLLED_INPUT_V1,
        SIMPolySourceConfig,
        run_simpoly_source_pipeline,
    )

    manifest = json.loads(
        (campaign_root(repo) / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    case = next(item for item in manifest["cases"] if item["case_id"] == case_id)
    image = FathomEngine().open_image(case["absolute_path"])
    body = np.asarray(image.pixels)[: image.valid_body.shape[0], :, 0]
    result, inter = run_simpoly_source_pipeline(
        body,
        SIMPolySourceConfig(
            profile=PROFILE_CONTROLLED_INPUT_V1,
            conversion_um_per_px=case["conversion_um_per_px"],
        ),
    )
    fathom = FathomEngine().run_fathom(image)
    matlab = loadmat(matlab_path)
    output = campaign_root(repo) / "latest" / case_id
    output.mkdir(parents=True, exist_ok=True)
    arrays = (
        ("Original controlled body", body),
        ("MATLAB BW", matlab["BW_thickened"]),
        ("Python BW", inter.thickened_mask),
        ("Mask difference", matlab["BW_thickened"].astype(bool) ^ inter.thickened_mask),
        ("MATLAB skeleton", matlab["SK_valid"]),
        ("Python skeleton", inter.valid_skeleton),
        ("Skeleton difference", matlab["SK_valid"].astype(bool) ^ inter.valid_skeleton),
        ("MATLAB diameter map", matlab["Dist"]),
        ("Python diameter map", inter.distance_map),
    )
    figure, axes = plt.subplots(3, 3, figsize=(15, 11))
    for axis, (title, array) in zip(axes.ravel(), arrays, strict=True):
        axis.imshow(array, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "stage-diagnostics.png", dpi=130)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    matlab_edges = matlab["hist_edges"].ravel()
    matlab_counts = matlab["hist_values"].ravel()
    axis.step(matlab_edges[:-1], matlab_counts, where="post", label="MATLAB")
    axis.step(result.histogram_edges[:-1], result.histogram_counts, where="post", label="Python")
    fit_x = np.linspace(
        min(matlab_edges[0], result.histogram_edges[0]) - 2,
        max(matlab_edges[-1], result.histogram_edges[-1]),
        500,
    )
    matlab_fit = matlab["gauss_a1"].item() * np.exp(
        -(((fit_x - matlab["gauss_b1"].item()) / matlab["gauss_c1"].item()) ** 2)
    )
    axis.plot(fit_x, matlab_fit, label="MATLAB gauss1")
    if result.gaussian_amplitude is not None and result.reported_center is not None:
        python_c1 = result.gaussian_c1_px * case["conversion_um_per_px"]
        python_fit = result.gaussian_amplitude * np.exp(
            -(((fit_x - result.reported_center) / python_c1) ** 2)
        )
        axis.plot(fit_x, python_fit, label="Python gauss1")
    axis.set(xlabel="Diameter (µm)", ylabel="Count", title=f"{case_id} histogram")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "histogram-fits.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 8))
    axis.imshow(image.valid_body, cmap="gray")
    for candidate in fathom.candidates:
        for proposal in candidate.proposed_measurements:
            axis.plot(
                (proposal.p1[0], proposal.p2[0]),
                (proposal.p1[1], proposal.p2[1]),
                color="#ffd166",
                linewidth=0.4,
            )
    axis.set_title(f"{case_id} Fathom proposals (review required)")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output / "fathom-overlay.png", dpi=130)
    plt.close(figure)
    return output


def generate_report(repo: Path) -> Path:
    root = campaign_root(repo)
    first_intermediates = (
        repo / ".validation/matlab-oracle/runs/r2026a-latest/ZEISS_001/controlled/intermediates.mat"
    )
    if first_intermediates.exists():
        run_detailed_parity_campaign(repo)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    environment = _read_json(repo / ".validation/matlab-oracle/environment.json") or {}
    toolbox_items = "".join(
        f"<li>{html.escape(str(toolbox.get('Name')))} {html.escape(str(toolbox.get('Version')))} {html.escape(str(toolbox.get('Release')))}</li>"
        for toolbox in environment.get("toolboxes", [])
    )
    rows = build_review_queue(repo)
    parity = compile_parity(repo)
    parity_by_case = {case["case_id"]: case for case in parity["cases"]}
    table_rows = []
    complete = 0
    for row in rows:
        py = _read_json(root / "runs/python-latest" / row["case_id"] / "python_summary.json") or {}
        matlab = (
            _read_json(
                repo
                / ".validation/matlab-oracle/runs/r2026a-latest"
                / row["case_id"]
                / "controlled"
                / "summary.json"
            )
            or {}
        )
        divergence = parity_by_case[row["case_id"]]["controlled_input"]["first_divergence"]
        if py.get("status") == "COMPLETE" and matlab.get("status") == "COMPLETE":
            complete += 1
        values = (
            row["case_id"],
            row["filename"],
            f"{float(row['pixel_size']):.6g} µm/px",
            row["resolution_class"],
            row["MATLAB_SIMPoly_center"],
            row["Python_SIMPoly_center"],
            row["MATLAB_Python_difference_percent"],
            row["Fathom_section_median"],
            row["manual_status"],
            row["quality_flags"],
            row["manual_priority"],
            divergence,
        )
        table_rows.append(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>"
        )
    source_match = manifest.get("source_matlab_sha256") == CANONICAL_SOURCE_SHA256
    parity_pass = all(
        row["MATLAB_Python_difference_percent"] is not None
        and row["MATLAB_Python_difference_percent"] <= 5
        for row in rows
    )
    reviewed_count = sum(row["manual_status"] in {"REVIEWED", "SKIPPED"} for row in rows)
    manual_measurements = sum(int(row["manual_measurement_count"]) for row in rows)
    worst = sorted(
        rows,
        key=lambda row: (
            row["MATLAB_Python_difference_percent"]
            if row["MATLAB_Python_difference_percent"] is not None
            else float("inf")
        ),
        reverse=True,
    )[:5]
    worst_items = "".join(
        f"<li>{html.escape(row['case_id'])}: Δ MATLAB/Python = {row['MATLAB_Python_difference_percent']!s}%</li>"
        for row in worst
    )
    failures = [
        case["case_id"]
        for case in parity["cases"]
        if case["controlled_input"]["status"] != "COMPARED"
    ]
    resolution_items = (
        "".join(
            f"<li>{html.escape(row['case_id'])}: {html.escape(row['resolution_class'])}</li>"
            for row in rows
            if row["resolution_class"] == "LOW_MAG_NETWORK"
        )
        or "<li>No LOW_MAG_NETWORK preset cases.</li>"
    )
    headers = (
        "case",
        "filename",
        "calibration",
        "resolution",
        "MATLAB SIMPoly",
        "Python SIMPoly",
        "Δ MATLAB/Python %",
        "Fathom",
        "manual",
        "flags",
        "priority",
        "first divergence",
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>{DATASET_ID}</title>
<style>body{{font:14px system-ui;margin:2rem;color:#20242a}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccd2d8;padding:.35rem;text-align:left}}th{{background:#eef1f4}}code,pre{{background:#f4f5f6;padding:.3rem}}</style></head><body>
<h1>Fathom Fibers — MATLAB oracle campaign</h1><pre>DATASET: {DATASET_ID}
EXPECTED CASES: 16
FOUND CASES: {manifest["case_count"]}

MATLAB_ORACLE_AVAILABLE
MATLAB_SOURCE_HASH_{"MATCH" if source_match else "MISMATCH"}
MATLAB_PYTHON_PARITY_{"PASS" if parity_pass else "FAIL"}
REAL_TIFF_CAMPAIGN_{"COMPLETE" if complete == 16 else "PARTIAL"}
AUTOMATIC_CAMPAIGN: {complete} / 16 COMPLETE
MANUAL_REVIEW: {reviewed_count} / 16
MANUAL_REVIEW_{"COMPLETE" if reviewed_count == 16 else "PENDING"}</pre>
<h2>Master inventory and outcomes</h2><table><thead><tr>{"".join(f"<th>{h}</th>" for h in headers)}</tr></thead><tbody>{"".join(table_rows)}</tbody></table>
<p><img src='method-comparison.png' alt='Method comparison across all 16 cases' style='max-width:100%'></p>
<h2>Environment</h2><p>MATLAB {html.escape(str(environment.get('matlab_version', 'unavailable')))} on {html.escape(str(environment.get('os', 'unknown OS')))}.</p>
<h3>MATLAB toolbox versions</h3><ul>{toolbox_items}</ul><p>Full local report: <code>.validation/matlab-oracle/environment.json</code>.</p>
<h2>Source provenance</h2><p>Canonical SHA-256: <code>{CANONICAL_SOURCE_SHA256}</code>.</p>
<h2>Morphology probes</h2><p>Deterministic probe arrays: <code>.validation/matlab-oracle/morphology_probes.mat</code>.</p>
<h2>Stage parity and first divergences</h2><p>Exact hashes are used for integer/mask stages. Float parity requires numerical metrics; a hash alone is not treated as evidence.</p>
<p>Machine-readable per-profile evidence: <code>../parity_summary.json</code>.</p>
<h2>Fathom method comparison</h2><p>Fathom section medians and SIMPoly Gaussian centers are different estimands; differences are not accuracy errors.</p>
<h2>Worst disagreements</h2><ol>{worst_items}</ol>
<h2>Resolution warnings</h2><ul>{resolution_items}</ul>
<h2>Failure cases</h2><p>{html.escape(", ".join(failures)) if failures else "None; all 16 attempts are represented."}</p>
<h2>Per-image diagnostics</h2><p>Selected divergent case: <a href='ZEISS_001/stage-diagnostics.png'>stage arrays</a>, <a href='ZEISS_001/histogram-fits.png'>histograms/fits</a>, <a href='ZEISS_001/fathom-overlay.png'>Fathom overlay</a>.</p>
<h2>Manual review progress</h2><p>{reviewed_count}/16 images reviewed; {manual_measurements}/400 numeric measurements. Every image has a 25-position MANUAL_5X5_REFERENCE target, but non-measurable cells are valid outcomes.</p>
<h2>Limitations</h2><p>MATLAB is an external validation adapter and is not a core dependency. SIMPoly is not ground truth. Projected 2D geometry only.</p>
<h2>Reproduction</h2><pre>fathom-fibers campaign run --methods matlab-simpoly,python-simpoly,fathom --resume
fathom-fibers campaign report</pre></body></html>"""
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    report = latest / "index.html"
    report.write_text(document, encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        x = np.arange(1, 17)
        matlab_values = [row["MATLAB_SIMPoly_center"] for row in rows]
        python_values = [row["Python_SIMPoly_center"] for row in rows]
        fathom_values = [row["Fathom_section_median"] for row in rows]
        figure, axis = plt.subplots(figsize=(11, 5))
        axis.plot(x, matlab_values, "o-", label="MATLAB SIMPoly controlled")
        axis.plot(x, python_values, "s-", label="Python SIMPoly controlled")
        axis.plot(x, fathom_values, "^-", label="Fathom section median")
        axis.set(xlabel="ZEISS case", ylabel="Reported value (µm)", xticks=x)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(latest / "method-comparison.png", dpi=150)
        plt.close(figure)
        generate_selected_diagnostics(repo)
    except (ImportError, ValueError):
        pass
    return report
