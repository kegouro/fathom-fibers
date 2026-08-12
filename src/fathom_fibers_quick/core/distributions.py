"""Common distributions and honest cross-method agreement metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .methods import DiameterDistribution, Estimand, MethodId


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    n: int
    weight_sum: float
    weighted_mean: float | None
    weighted_median: float | None
    weighted_std: float | None
    p05: float | None
    p25: float | None
    p50: float | None
    p75: float | None
    p95: float | None
    histogram_edges: np.ndarray
    histogram_weights: np.ndarray
    ecdf_x: np.ndarray
    ecdf_y: np.ndarray


@dataclass(frozen=True, slots=True)
class DistributionAgreement:
    left_method: MethodId
    right_method: MethodId
    estimand: Estimand
    mean_difference: float | None
    median_difference: float | None
    relative_median_difference_percent: float | None
    wasserstein_1: float | None
    ks_statistic: float | None


@dataclass(frozen=True, slots=True)
class ConsensusPseudoReference:
    distribution: DiameterDistribution | None
    quantile_grid: np.ndarray
    quantiles: np.ndarray
    disagreement_mad: np.ndarray
    participating_methods: tuple[MethodId, ...]
    excluded_methods: dict[str, str]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float).ravel()
    weights = np.asarray(weights, float).ravel()
    quantiles = np.asarray(quantiles, float)
    if not values.size:
        return np.full(quantiles.shape, np.nan)
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    return np.interp(quantiles * cumulative[-1], cumulative, x)


def common_histogram_edges(distributions: Iterable[DiameterDistribution], bins: int = 32) -> np.ndarray:
    values = [item.diameter for item in distributions if item.diameter.size]
    if not values:
        return np.array([], dtype=float)
    joined = np.concatenate(values)
    if np.allclose(joined.min(), joined.max()):
        half = max(abs(float(joined[0])) * 0.05, 0.5)
        return np.array([joined[0] - half, joined[0] + half])
    return np.histogram_bin_edges(joined, bins=bins)


def summarize_distribution(distribution: DiameterDistribution, *, bins: np.ndarray | None = None) -> DistributionSummary:
    x, w = distribution.diameter, distribution.weight
    if not x.size:
        empty = np.array([], dtype=float)
        return DistributionSummary(0, 0.0, None, None, None, None, None, None, None, None, empty, empty, empty, empty)
    q = weighted_quantile(x, w, np.array([0.05, 0.25, 0.5, 0.75, 0.95]))
    mean = float(np.average(x, weights=w))
    variance = float(np.average((x - mean) ** 2, weights=w))
    edges = bins if bins is not None else common_histogram_edges([distribution])
    hist, edges = np.histogram(x, bins=edges, weights=w)
    order = np.argsort(x, kind="stable")
    ecdf_x = x[order]
    ecdf_y = np.cumsum(w[order]) / w.sum()
    return DistributionSummary(
        int(x.size), float(w.sum()), mean, float(q[2]), float(np.sqrt(variance)),
        *(float(value) for value in q), edges, hist.astype(float), ecdf_x, ecdf_y,
    )


def compare_distributions(left: DiameterDistribution, right: DiameterDistribution) -> DistributionAgreement:
    if left.unit != right.unit or left.estimand != right.estimand:
        raise ValueError("comparison requires matching unit and estimand")
    if not left.diameter.size or not right.diameter.size:
        return DistributionAgreement(left.source_method, right.source_method, left.estimand, None, None, None, None, None)
    left_summary = summarize_distribution(left)
    right_summary = summarize_distribution(right)
    median_difference = left_summary.weighted_median - right_summary.weighted_median
    denominator = abs(right_summary.weighted_median)
    return DistributionAgreement(
        left.source_method, right.source_method, left.estimand,
        left_summary.weighted_mean - right_summary.weighted_mean,
        median_difference,
        100.0 * median_difference / denominator if denominator else None,
        float(stats.wasserstein_distance(left.diameter, right.diameter, left.weight, right.weight)),
        float(stats.ks_2samp(left.diameter, right.diameter).statistic),
    )


def consensus_pseudo_reference(results: Iterable[DiameterDistribution], *, grid_size: int = 101) -> ConsensusPseudoReference:
    included = list(results)
    grid = np.linspace(0.0, 1.0, grid_size)
    if not included:
        return ConsensusPseudoReference(None, grid, np.array([]), np.array([]), (), {})
    unit = included[0].unit
    estimand = included[0].estimand
    compatible = [
        item
        for item in included
        if item.unit == unit and item.estimand == estimand and item.diameter.size
    ]
    compatible_ids = {id(item) for item in compatible}
    excluded = {
        item.source_method.value: "EMPTY_OR_INCOMPATIBLE_COMMON_DISTRIBUTION"
        for item in included
        if id(item) not in compatible_ids
    }
    if not compatible:
        return ConsensusPseudoReference(None, grid, np.array([]), np.array([]), (), excluded)
    curves = np.vstack([weighted_quantile(item.diameter, item.weight, grid) for item in compatible])
    quantiles = np.median(curves, axis=0)
    mad = np.median(np.abs(curves - quantiles), axis=0)
    # Quantile samples are equally weighted in V1; this is explicitly not truth.
    distribution = DiameterDistribution(quantiles, np.ones_like(quantiles), unit, estimand, MethodId.CONSENSUS_PSEUDO_REFERENCE_V1)
    return ConsensusPseudoReference(distribution, grid, quantiles, mad, tuple(item.source_method for item in compatible), excluded)
