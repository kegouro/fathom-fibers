from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.core.distributions import summarize_distribution, weighted_quantile
from fathom_fibers_quick.core.methods import MethodId
from fathom_fibers_quick.reports import (
    _long_tail_ratio,
    _primary_x_max,
    build_final_dataset_report,
    build_image_report,
    series_distributions,
)
from fathom_fibers_quick.workspace import (
    WorkspaceCache,
    WorkspaceDataset,
    WorkspaceImage,
    load_workspace_dataset,
)

pytestmark = pytest.mark.qt


def _dataset_root() -> Path:
    root = os.environ.get("FATHOM_ZEISS_DATASET")
    if not root:
        pytest.skip("private Zeiss dataset not present; set FATHOM_ZEISS_DATASET")
    return Path(root)


def real_comparison() -> object:
    cache = WorkspaceCache()
    comparison = cache.load_comparison("PVDF Jose_01")
    if comparison is None:
        pytest.skip("cached comparison not present; run the private campaign first")
    return comparison


def test_primary_range_rule_is_data_driven():
    comparison = real_comparison()
    series = series_distributions(comparison)
    x_max = _primary_x_max(series)
    assert x_max is not None
    for name, distribution in series:
        if name == "Fathom Local":
            continue
        # the primary range covers at least the P99 of primary estimators
        p99 = weighted_quantile(distribution.diameter, distribution.weight, np.array([0.99]))[0]
        assert x_max >= float(p99)


def test_long_tail_criterion():
    comparison = real_comparison()
    series = series_distributions(comparison)
    local_ratio = _long_tail_ratio(series, "Fathom Local")
    assert local_ratio is not None
    assert local_ratio > 2.0  # PVDF Jose_01 is a long-tailed image


def test_image_report_scientific_summary():
    comparison = real_comparison()
    from fathom_fibers_quick.api import FathomEngine

    image = FathomEngine().open_image(str(_dataset_root() / "PVDF Jose_01.tif"))
    out = build_image_report(comparison, image, output_dir=Path("/tmp/opencode/report-v2/test-image"))
    text = out.read_text()
    assert "Scientific Summary" in text
    assert "Centerline refinement effect" in text
    for estimator in ("Ribbon EDT", "Ribbon Edge", "Ribbon Profile"):
        assert estimator in text
    assert "figure-primary-histogram.png" in text
    assert "figure-full-histogram.png" in text
    assert "Fathom Local — distribution tail" in text
    assert "Common sample distribution unavailable from current cache" in text
    assert "INCOMPLETE REFERENCE" in text
    assert "Full flag breakdown" in text
    assert "Provenance" in text
    assert "Limitations" in text


def test_image_report_numerical_invariance():
    comparison = real_comparison()
    from fathom_fibers_quick.api import FathomEngine

    image = FathomEngine().open_image(str(_dataset_root() / "PVDF Jose_01.tif"))
    out = build_image_report(comparison, image, output_dir=Path("/tmp/opencode/report-v2/test-invariance"))
    text = out.read_text()
    # every median that appears in the refinement table must equal the
    # weighted summary computed directly from the cached MethodResult
    field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
    expected = {
        "FATHOM_FIELD_PAIRED_EDGE_DIAMETER": field.secondary_distributions["FATHOM_FIELD_PAIRED_EDGE_DIAMETER"],
        "FATHOM_FIELD_PROFILE_DIAMETER": field.secondary_distributions["FATHOM_FIELD_PROFILE_DIAMETER"],
        "FATHOM_FIELD_REFINED_EDT_DIAMETER": field.secondary_distributions["FATHOM_FIELD_REFINED_EDT_DIAMETER"],
        "FATHOM_FIELD_REFINED_EDGE_DIAMETER": field.secondary_distributions["FATHOM_FIELD_REFINED_EDGE_DIAMETER"],
        "FATHOM_FIELD_REFINED_PROFILE_DIAMETER": field.secondary_distributions["FATHOM_FIELD_REFINED_PROFILE_DIAMETER"],
    }
    for key, distribution in expected.items():
        summary = summarize_distribution(distribution)
        median_text = f"{summary.weighted_median * 1000.0:.5g}"
        # median must appear in the report (either in the summary cards or tables)
        assert median_text in text, (key, median_text)
        p95_text = f"{summary.p95 * 1000.0:.5g}"
        assert p95_text in text, (key, p95_text)


def test_dataset_report_sections():
    repo = Path("/tmp/fathom-worktrees/unified-methods")
    dataset = load_workspace_dataset(_dataset_root(), repo=repo)
    out = build_final_dataset_report(
        repo, dataset=dataset, manual_store=None, output_dir=repo / ".validation/final-report-v2-test"
    )
    text = out.read_text()
    for needle in (
        "Dataset overview",
        "Dataset method summary",
        "Oriented Ribbon dataset behavior",
        "Per-image navigation",
        "Quality overview",
        "Manual 5×5",
        "Provenance",
        "Limitations",
    ):
        assert needle in text, needle
    # 16 image sections with anchors
    assert text.count("id='image-") == 16
    # diameters are reported in nanometres with an explicit mean ± 1 SD summary
    assert "Mean of image medians ± 1 SD (nm)" in text
    assert "Mean ± 1 SD (nm)" in text
    assert "Median of image medians (nm)" in text
    assert "1 µm = 1000 nm" in text
    assert "values are physical nm" in text
    assert "How to read this report" in text
    # per-image figures generated
    assert (out.parent / "images/PVDF Jose_01/figure-A-primary-histogram.png").exists()
    assert (out.parent / "dataset-figure-F.png").exists()
    # image-level aggregation wording (not pooled)
    assert "Summary across images" in text
    assert "not pooled" in text or "aggregate image-level" in text


def test_dataset_ribbon_section_computed_not_hardcoded():
    repo = Path("/tmp/fathom-worktrees/unified-methods")
    cache = WorkspaceCache(repo)
    comparison = cache.load_comparison("PVDF Jose_01")
    if comparison is None:
        pytest.skip("cached comparison not present; run the private campaign first")
    from fathom_fibers_quick.reports import _dataset_ribbon_metrics

    metrics = _dataset_ribbon_metrics([comparison])
    assert metrics["improved_count"] >= 0
    assert metrics["total"] == 1
    assert len(metrics["w1_edt_edge_raw"]) == 1
    assert metrics["w1_edt_edge_refined"][0] < metrics["w1_edt_edge_raw"][0]


def test_flag_definitions_cover_known_flags():
    from fathom_fibers_quick.report_style import FLAG_DEFINITIONS

    comparison = real_comparison()
    field = next(r for r in comparison.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
    for flag in field.quality_flags:
        assert flag in FLAG_DEFINITIONS or flag in {"EXPERIMENTAL_FIELD_MEASURING", "FIELD_STAGE_IMPLEMENTED", "GRAPH_STAGE_NOT_IMPLEMENTED"}


def test_report_css_includes_print_and_responsive():
    from fathom_fibers_quick.report_style import CSS

    assert "@media print" in CSS
    assert "@media (max-width: 900px)" in CSS
    assert "max-width: 1320px" in CSS


def test_export_bundle_uses_new_report():

    repo = Path("/tmp/fathom-worktrees/unified-methods")
    cache = WorkspaceCache(repo)
    comparison = cache.load_comparison("PVDF Jose_01")
    if comparison is None:
        pytest.skip("cached comparison not present; run the private campaign first")
    dataset = WorkspaceDataset(
        "ZEISS_TEST_DATASET",
        (
            WorkspaceImage(
                "ZEISS_001",
                "PVDF Jose_01.tif",
                Path("/tmp/zeiss-dataset/PVDF Jose_01.tif"),
            ),
        ),
    )
    from fathom_fibers_quick.reports import build_final_dataset_report

    out = build_final_dataset_report(
        repo, dataset=dataset, manual_store=None, output_dir=Path("/tmp/opencode/report-v2/bundle-report"),
        comparisons=[comparison],
    )
    text = out.read_text()
    assert "Scientific Morphological Fiber Analysis" in text
    assert "Oriented Ribbon dataset behavior" in text
