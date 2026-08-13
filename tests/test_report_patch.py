from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from fathom_fibers_quick.core.distributions import weighted_quantile
from fathom_fibers_quick.core.methods import MethodId
from fathom_fibers_quick.reports import (
    DATASET_FIGURE_A_TITLE,
    _calibration_audit,
    _dataset_mixture,
    build_final_dataset_report,
)
from fathom_fibers_quick.workspace import (
    WorkspaceCache,
    load_workspace_dataset,
)

pytestmark = pytest.mark.qt

REPO = Path("/tmp/fathom-worktrees/unified-methods")
DATASET_DIR = "/home/kegouro/HIBRIS/Workshop ⁄ Proyectos/fathom-fibers/local_data/zeiss/30-07-26"


def _dataset():
    return load_workspace_dataset(DATASET_DIR, repo=REPO)


def _comparisons():
    cache = WorkspaceCache(REPO)
    dataset = _dataset()
    return dataset, [c for c in (cache.load_comparison(i.stem) for i in dataset.images) if c]


def test_calibration_audit_covers_all_images():
    dataset, comparisons = _comparisons()
    audit = _calibration_audit(dataset, comparisons)
    assert len(audit) == 16
    calibrated = [entry for entry in audit if entry["calibrated"]]
    assert len(calibrated) == 16
    assert all(entry["isotropic"] for entry in calibrated)
    # calibration is mixed across the dataset
    sizes = [entry["pixel_size_x_nm"] for entry in calibrated]
    assert min(sizes) < 10.0 and max(sizes) > 500.0
    # every image uses the full usable field excluding the footer
    for entry in audit:
        assert entry["roi"] == (0, 0, 3072, 2071), entry["image"]
        assert entry["footer"] == (2071, 2240)
        assert entry["shape"] == (2304, 3072)


def test_calibration_summary_not_dash():
    dataset, comparisons = _comparisons()
    audit = _calibration_audit(dataset, comparisons)
    from fathom_fibers_quick.reports import _calibration_audit_summary, _calibration_short

    short = _calibration_short(audit)
    assert "—" not in short
    assert "mixed" in short
    assert "16 / 16" in short
    html = _calibration_audit_summary(audit)
    assert "Calibration" in html
    assert "Per-image calibration and ROI audit" in html


def test_toc_anchors_all_resolve():
    dataset, comparisons = _comparisons()
    out = build_final_dataset_report(
        REPO, dataset=dataset, manual_store=None, output_dir=REPO / ".validation/toc-test",
        comparisons=comparisons,
    )
    text = out.read_text()
    hrefs = re.findall(r"href='#([^']+)'", text)
    ids = set(re.findall(r"id='([^']+)'", text))
    assert hrefs, "no TOC links found"
    missing = [href for href in hrefs if href not in ids]
    assert missing == [], missing
    assert "id='dataset-overview'" in text
    assert "id='dataset-distribution'" in text
    assert "id='ribbon-dataset-behavior'" in text


def test_manual_warning_uses_design_system_class():
    from fathom_fibers_quick.workspace import Manual5x5Store

    dataset, comparisons = _comparisons()
    out = build_final_dataset_report(
        REPO, dataset=dataset,
        manual_store=Manual5x5Store(REPO / ".validation/unified-method-comparison", dataset.dataset_id),
        output_dir=REPO / ".validation/toc-test2",
        comparisons=comparisons,
    )
    text = out.read_text()
    assert "notebox" not in text
    assert "INCOMPLETE REFERENCE" in text
    assert "0 / 400" in text
    assert "sparse human reference" in text.lower() or "not ground truth" in text.lower()


def test_figure_a_terminology():
    import inspect

    from fathom_fibers_quick import reports

    source = inspect.getsource(reports)
    assert DATASET_FIGURE_A_TITLE in source
    assert "SIMPoly Python" not in source.replace(DATASET_FIGURE_A_TITLE, "")
    # the figure legend uses full method names, not short ones
    assert "label = \"Python SIMPoly\" if method_id == MethodId.PYTHON_SIMPOLY else \"Fathom Local\"" in source


def test_equal_image_weight_does_not_dominate():
    """Image A (100 samples @ 1 um) must weight equally to image B (10000 @ 3 um)."""
    from fathom_fibers_quick.core.methods import (
        DiameterDistribution,
        Estimand,
        MethodStatus,
    )

    class FakeResult:
        def __init__(self, method_id, distribution):
            self.method_id = method_id
            self.common_distribution = distribution
            self.secondary_distributions = {}
            self.native_distribution = None
            self.valid_roi = (0, 0, 64, 48)
            self.status = MethodStatus.COMPLETE
            self.quality_flags = ()
            self.native_statistics = {}
            self.method_version = "test"
            self.image_id = "synthetic"
            self.unit = "um"
            self.native_estimand = None
            self.native_result = None
            self.runtime_seconds = None
            self.provenance = {}

    a = FakeResult(MethodId.PYTHON_SIMPOLY, DiameterDistribution(
        np.full(100, 1.0), np.ones(100), "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.PYTHON_SIMPOLY))
    b = FakeResult(MethodId.PYTHON_SIMPOLY, DiameterDistribution(
        np.full(10000, 3.0), np.ones(10000), "um", Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER, MethodId.PYTHON_SIMPOLY))
    from fathom_fibers_quick.unified_comparison import compare_method_results

    comparison_a = compare_method_results((a,))
    comparison_b = compare_method_results((b,))

    from fathom_fibers_quick.reports import _dataset_mixture

    diameter, weight = _dataset_mixture([comparison_a, comparison_b], "PYTHON_SIMPOLY")
    # equal image weight: each image contributes unit total mass
    image_a_mass = weight[diameter == 1.0].sum()
    image_b_mass = weight[diameter == 3.0].sum()
    assert image_a_mass == pytest.approx(1.0)
    assert image_b_mass == pytest.approx(1.0)
    # mixture median sits between the two modes, close to the equal-weight center
    median = float(weighted_quantile(diameter, weight, np.array([0.5]))[0])
    # with equal image weight the mixture median sits at the 1.0-mode boundary;
    # a sample-count-dominated mixture would give a median at 3.0
    assert 1.0 <= median < 2.0
    # image A influence is ~50%, not ~1/100
    assert image_a_mass / (image_a_mass + image_b_mass) == pytest.approx(0.5)


def test_mixture_histogram_mass_normalized():
    _dataset, comparisons = _comparisons()
    diameter, weight = _dataset_mixture(comparisons, "PYTHON_SIMPOLY")
    assert diameter.size > 0
    assert np.sum(weight) == pytest.approx(len(comparisons), rel=1e-9)


def test_no_methodresult_modified():
    cache = WorkspaceCache(REPO)
    before = cache.load_comparison("PVDF Jose_01")
    field_before = next(r for r in before.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
    median_before = float(np.median(field_before.common_distribution.diameter))
    _dataset, comparisons = _comparisons()
    build_final_dataset_report(
        REPO, dataset=_dataset, manual_store=None, output_dir=REPO / ".validation/toc-test3",
        comparisons=comparisons,
    )
    after = cache.load_comparison("PVDF Jose_01")
    field_after = next(r for r in after.results if r.method_id == MethodId.FATHOM_FIELD_GRAPH_V1)
    assert float(np.median(field_after.common_distribution.diameter)) == median_before


def test_dataset_report_keeps_ribbon_and_caveats():
    from fathom_fibers_quick.workspace import Manual5x5Store

    dataset, comparisons = _comparisons()
    out = build_final_dataset_report(
        REPO, dataset=dataset, manual_store=Manual5x5Store(REPO / ".validation/unified-method-comparison", dataset.dataset_id),
        output_dir=REPO / ".validation/toc-test4",
        comparisons=comparisons,
    )
    text = out.read_text()
    for needle in (
        "Oriented Ribbon dataset behavior",
        "EXPERIMENTAL",
        "not known absolute accuracy",
        "consensus pseudo-reference",
        "sparse human reference",
        "INCOMPLETE REFERENCE",
        "Provenance",
        "Limitations",
    ):
        assert needle in text, needle
