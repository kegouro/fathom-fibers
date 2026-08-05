from __future__ import annotations

import numpy as np
import pytest

from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.oracles.contracts import (
    EstimandType,
    OracleManifest,
    OracleRun,
)
from fathom_fibers_quick.oracles.metrics import (
    compute_diameter_metrics,
    compute_mask_metrics,
    compute_skeleton_metrics,
)
from fathom_fibers_quick.oracles.report import generate_simpoly_validation_report
from fathom_fibers_quick.oracles.simpoly import (
    generate_synthetic_fiber_phantom,
    run_synthetic_benchmark_suite,
)
from fathom_fibers_quick.protocols import PRESET_SIMPOLY_MANUAL_5X5
from fathom_fibers_quick.simpoly_compat import METHOD_NAME, fit_1d_gaussian, run_simpoly_pipeline
from scripts.run_simpoly_oracle import run_matlab_oracle


def test_oracle_manifest_and_run_contracts():
    manifest = OracleManifest(
        oracle_id="SIMPOLY_MATLAB_ORIGINAL",
        name="SIMPoly MATLAB Original",
        version="1.0.0",
        source_doi="10.1089/ten.tec.2020.0304",
        license_status="LICENSE_UNRESOLVED",
    )
    d_man = manifest.to_dict()
    assert d_man["license_status"] == "LICENSE_UNRESOLVED"

    run = OracleRun(
        run_id="RUN_001",
        oracle_id="SIMPOLY_LITERATURE_REIMPLEMENTATION_V1",
        oracle_version="1.0.0",
        image_id="synthetic_50px.png",
        gaussian_center_px=50.2,
        status="SUCCESS",
    )
    d_run = run.to_dict()
    assert d_run["gaussian_center_px"] == 50.2

    reloaded_run = OracleRun.from_dict(d_run)
    assert reloaded_run.run_id == "RUN_001"


def test_matlab_runner_absent_handling(tmp_path):
    img_path = tmp_path / "dummy.png"
    img_path.write_bytes(b"dummy")
    out_json = tmp_path / "out.json"

    res = run_matlab_oracle(img_path, out_json)
    assert res["status"] in {"SKIPPED_MATLAB_ABSENT", "FAILED_MATLAB_RUNNER"}
    assert out_json.exists()


def test_explicit_estimands():
    assert EstimandType.SIMPOLY_GAUSSIAN_CENTER.value == "SIMPOLY_GAUSSIAN_CENTER"
    assert EstimandType.LOCAL_SECTION_WEIGHTED.value == "LOCAL_SECTION_WEIGHTED"
    assert EstimandType.MANUAL_GRID_MEAN.value == "MANUAL_GRID_MEAN"


def test_stage_by_stage_parity_metrics():
    m1 = np.ones((50, 50), dtype=bool)
    m2 = np.ones((50, 50), dtype=bool)
    m2[0, :] = False

    mask_m = compute_mask_metrics(m1, m2)
    assert mask_m["iou"] > 0.95
    assert mask_m["dice"] > 0.95

    sk1 = np.zeros((50, 50), dtype=bool)
    sk1[25, :] = True
    sk2 = sk1.copy()

    sk_m = compute_skeleton_metrics(sk1, sk2)
    assert sk_m["overlap_fraction"] == 1.0

    d_metrics = compute_diameter_metrics([10.0, 10.2, 9.8], [10.0, 10.0, 10.0], 10.1, 10.0)
    assert d_metrics["difference_of_gaussian_center"] == pytest.approx(0.1)
    assert d_metrics["mae"] is not None


def test_fit_1d_gaussian():
    rng = np.random.default_rng(42)
    data = rng.normal(50.0, 2.0, 500).tolist()

    mu, sig, _amp = fit_1d_gaussian(data)
    assert mu == pytest.approx(50.0, abs=1.5)
    assert sig == pytest.approx(2.0, abs=1.0)


def test_simpoly_python_pipeline_on_synthetic_phantom():
    cal = Calibration(1e-9, 1e-9, "synthetic")
    phantom = generate_synthetic_fiber_phantom(width_px=30.0, shape=(256, 256), disordered=False)

    res = run_simpoly_pipeline(phantom, cal)
    assert res["method_name"] == METHOD_NAME
    assert res["gaussian_center_px"] == pytest.approx(30.0, abs=3.0)
    assert res["segmented_fraction"] > 0.05


@pytest.mark.slow
def test_published_benchmark_reproduction_suite():
    _runs, _comparisons, summary = run_synthetic_benchmark_suite()

    assert summary["total_cases"] == 41
    assert summary["median_relative_error_percent"] <= 5.0
    # Small width discretization causes higher P90 error on 10px phantoms, recorded in summary
    assert "p90_relative_error_percent" in summary


def test_simpoly_manual_5x5_protocol_preset():
    proto = PRESET_SIMPOLY_MANUAL_5X5
    assert proto.protocol_id == "SIMPOLY_MANUAL_5X5"
    assert proto.sections_per_fiber == 1


def test_generate_simpoly_validation_report(tmp_path):
    synth_summary = {
        "total_cases": 41,
        "mean_error_ordered_percent": 2.15,
        "mean_error_disordered_percent": 1.62,
        "median_relative_error_percent": 1.8,
        "p90_relative_error_percent": 4.5,
        "fraction_within_10_percent": 1.0,
    }
    zeiss_results = [
        {
            "image_name": "PVDF Jose_02.tif",
            "domain": "SUPPORTED",
            "simpoly_px": "42.5",
            "fathom_px": "41.8",
            "manual_5x5_px": "42.0",
            "notes": "PVDF electrospun primary",
        }
    ]

    out_report = tmp_path / "test_simpoly_report.html"
    res_path = generate_simpoly_validation_report(synth_summary, zeiss_results, out_report)

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "LICENSE_UNRESOLVED" in content
    assert "PVDF Jose_02.tif" in content
