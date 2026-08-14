from __future__ import annotations

import numpy as np

from fathom_fibers_quick.api import FathomEngine
from fathom_fibers_quick.core.distributions import (
    common_histogram_edges,
    compare_distributions,
    consensus_pseudo_reference,
    summarize_distribution,
    weighted_quantile,
)
from fathom_fibers_quick.core.methods import (
    Capability,
    CapabilityState,
    DiameterDistribution,
    Estimand,
    MethodCapabilities,
    MethodId,
    MethodResult,
    MethodStatus,
    method_cache_key,
)
from fathom_fibers_quick.model import Calibration
from fathom_fibers_quick.unified_comparison import compare_method_results


def distribution(values, weights, method=MethodId.PYTHON_SIMPOLY):
    return DiameterDistribution(
        np.asarray(values, float),
        np.asarray(weights, float),
        "um",
        Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER,
        method,
    )


def test_capabilities_and_method_result_accept_missing_outputs():
    capabilities = MethodCapabilities({Capability.GRAPH: CapabilityState.UNAVAILABLE})
    result = MethodResult(
        MethodId.FATHOM_FIELD_GRAPH_V1,
        "V1",
        "image",
        {"pixel_size_x_m": 1e-9},
        None,
        "um",
        capabilities,
        MethodStatus.EXPERIMENTAL_NOT_YET_MEASURING,
    )
    assert not capabilities.supports(Capability.GRAPH)
    assert result.common_distribution is None


def test_weighted_quantiles_and_summary_use_weights():
    dist = distribution([1.0, 10.0], [9.0, 1.0])
    assert weighted_quantile(dist.diameter, dist.weight, np.array([0.5]))[0] == 1.0
    summary = summarize_distribution(dist)
    assert summary.weighted_mean == 1.9
    assert summary.weighted_median == 1.0


def test_common_bins_agreement_and_consensus_are_explicitly_not_truth():
    left = distribution([1, 2, 3], [1, 1, 1], MethodId.MATLAB_SIMPOLY)
    right = distribution([2, 3, 4], [1, 1, 1], MethodId.PYTHON_SIMPOLY)
    edges = common_histogram_edges([left, right], bins=4)
    assert np.array_equal(edges, common_histogram_edges([right, left], bins=4))
    agreement = compare_distributions(left, right)
    assert agreement.wasserstein_1 == 1.0
    consensus = consensus_pseudo_reference([left, right])
    assert consensus.participating_methods == (MethodId.MATLAB_SIMPOLY, MethodId.PYTHON_SIMPOLY)
    assert consensus.distribution is not None
    assert consensus.distribution.source_method == MethodId.CONSENSUS_PSEUDO_REFERENCE_V1


def test_cache_key_changes_with_scientific_input_but_not_path():
    first = method_cache_key(
        image_sha256="abc", valid_roi=(0, 0, 10, 10), calibration={"px": 1},
        method_id=MethodId.PYTHON_SIMPOLY, method_version="V1", parameters={"profile": "controlled"},
    )
    second = method_cache_key(
        image_sha256="abc", valid_roi=(0, 0, 10, 10), calibration={"px": 1},
        method_id=MethodId.PYTHON_SIMPOLY, method_version="V1", parameters={"profile": "controlled"},
    )
    changed = method_cache_key(
        image_sha256="abc", valid_roi=(0, 0, 11, 10), calibration={"px": 1},
        method_id=MethodId.PYTHON_SIMPOLY, method_version="V1", parameters={"profile": "controlled"},
    )
    assert first == second
    assert first != changed


def test_engine_unified_comparison_has_honest_missing_method_states():
    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    engine = FathomEngine()
    image = engine.from_array(pixels, calibration=Calibration(5e-9, 5e-9, "test"), image_id="synthetic")
    comparison = engine.compare_all_methods(image, roi_bbox=(0, 0, 128, 90))
    states = {result.method_id: result.status for result in comparison.results}
    assert states[MethodId.MATLAB_SIMPOLY] == MethodStatus.NOT_RUN
    assert states[MethodId.FATHOM_FIELD_GRAPH_V1] == MethodStatus.EXPERIMENTAL_FIELD_MEASURING
    assert MethodId.FATHOM_FIELD_GRAPH_V1 in comparison.summaries
    assert states[MethodId.MANUAL_5X5_REFERENCE] == MethodStatus.NOT_MEASURED
    assert MethodId.PYTHON_SIMPOLY in comparison.summaries


def test_comparison_excludes_incompatible_or_missing_common_distributions():
    left = MethodResult(MethodId.MATLAB_SIMPOLY, "V1", "image", {}, None, "um", MethodCapabilities(), MethodStatus.COMPLETE, common_distribution=distribution([1, 2], [1, 1], MethodId.MATLAB_SIMPOLY))
    missing = MethodResult(MethodId.FATHOM_FIELD_GRAPH_V1, "V1", "image", {}, None, "um", MethodCapabilities(), MethodStatus.EXPERIMENTAL_NOT_YET_MEASURING)
    comparison = compare_method_results((left, missing))
    assert comparison.consensus.participating_methods == (MethodId.MATLAB_SIMPOLY,)
    assert comparison.consensus.excluded_methods[MethodId.FATHOM_FIELD_GRAPH_V1.value] == "NO_COMMON_DISTRIBUTION"


def test_cli_lists_unified_methods(capsys, monkeypatch):
    from fathom_fibers_quick.cli import main

    monkeypatch.setattr("sys.argv", ["fathom-fibers", "methods", "list"])
    main()
    assert "FATHOM_FIELD_GRAPH_V1" in capsys.readouterr().out


def test_qt_comparison_table_shows_not_run_and_experimental_states(qtbot):
    from fathom_fibers_quick.ui.widgets.panels import ComparisonPanel

    pixels = np.zeros((96, 128), dtype=np.uint8)
    pixels[35:55, 16:112] = 220
    engine = FathomEngine()
    image = engine.from_array(pixels, calibration=Calibration(5e-9, 5e-9, "test"))
    panel = ComparisonPanel()
    qtbot.addWidget(panel)
    panel.set_unified_result(engine.compare_all_methods(image), image)
    statuses = [panel.table.item(row, 1).text() for row in range(panel.table.rowCount())]
    assert "NOT_RUN" in statuses
    assert "EXPERIMENTAL_FIELD_MEASURING" in statuses
