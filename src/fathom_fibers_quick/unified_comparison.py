"""Method-neutral comparison and report contracts; agreement is never truth."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from .core.distributions import (
    ConsensusPseudoReference,
    DistributionAgreement,
    DistributionSummary,
    common_histogram_edges,
    compare_distributions,
    consensus_pseudo_reference,
    summarize_distribution,
)
from .core.methods import Estimand, MethodId, MethodResult, MethodStatus


@dataclass(frozen=True, slots=True)
class UnifiedMethodComparison:
    image_id: str
    results: tuple[MethodResult, ...]
    summaries: dict[MethodId, DistributionSummary]
    agreements: tuple[DistributionAgreement, ...]
    consensus: ConsensusPseudoReference


@dataclass(frozen=True, slots=True)
class ImageMorphologyReport:
    image_id: str
    comparison: UnifiedMethodComparison
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetMorphologyReport:
    dataset_id: str
    images: tuple[ImageMorphologyReport, ...]
    failures: dict[str, str]
    limitations: tuple[str, ...]


def compare_method_results(results: Iterable[MethodResult]) -> UnifiedMethodComparison:
    all_results = tuple(results)
    comparable = [
        result.common_distribution
        for result in all_results
        if result.status == MethodStatus.COMPLETE and result.common_distribution is not None
        and result.common_distribution.estimand == Estimand.COMMON_LENGTH_WEIGHTED_DIAMETER
    ]
    bins = common_histogram_edges(comparable)
    summaries = {item.source_method: summarize_distribution(item, bins=bins) for item in comparable}
    agreements = tuple(compare_distributions(left, right) for left, right in combinations(comparable, 2))
    comparable_ids = {id(item) for item in comparable}
    excluded = {
        result.method_id.value: (
            "NO_COMMON_DISTRIBUTION" if result.common_distribution is None else f"STATUS_{result.status.value}"
        )
        for result in all_results
        if result.common_distribution is None or id(result.common_distribution) not in comparable_ids
    }
    raw_consensus = consensus_pseudo_reference(comparable)
    consensus = ConsensusPseudoReference(
        raw_consensus.distribution,
        raw_consensus.quantile_grid,
        raw_consensus.quantiles,
        raw_consensus.disagreement_mad,
        raw_consensus.participating_methods,
        {**raw_consensus.excluded_methods, **excluded},
    )
    image_id = all_results[0].image_id if all_results else "unknown-image"
    return UnifiedMethodComparison(image_id, all_results, summaries, agreements, consensus)


def build_image_report(comparison: UnifiedMethodComparison) -> ImageMorphologyReport:
    limitations = [
        "Measurements represent projected 2-D geometry.",
        "Cross-method agreement and consensus pseudo-reference are not ground truth.",
    ]
    if any(result.method_id == MethodId.PYTHON_SIMPOLY for result in comparison.results):
        limitations.append("Python SIMPoly retains KNOWN_LIBRARY_DIVERGENCE: bwskel.")
    return ImageMorphologyReport(comparison.image_id, comparison, tuple(limitations))
