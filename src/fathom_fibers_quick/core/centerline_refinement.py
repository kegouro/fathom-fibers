"""FATHOM_ORIENTED_RIBBON_V1 — stage 2: smooth refined centerline.

Converts the validated midpoint observations into ordered non-branching seed
runs and, on each continuous run of usable observations, fits a
confidence-weighted cubic smoothing spline parameterized by physical
arc length.

This stage deliberately does not: remeasure widths or boundaries, resolve
crossings or junctions, build a graph, or assign fiber identities.  The
refined curve exists only on the observed support of each run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.interpolate import make_smoothing_spline

from .oriented_ribbon import (
    ABSTAIN_EDGE_FLAGS,
    ABSTAIN_PROFILE_FLAGS,
    LOW_COHERENCE_FLAG,
    BoundaryMidpointObservation,
    CenterlineRefinementConfig,
    CenterlineRefinementResult,
)

STAGE = "SMOOTH_CENTERLINE_V1"
ORDERING = "SEED_RUN_ORDERING_V1"
SMOOTHING = "SCIPY_CUBIC_SMOOTHING_SPLINE"

FLAG_SEGMENT_TOO_SHORT = "REFINEMENT_SEGMENT_TOO_SHORT"
FLAG_NO_REFINEMENT = "NO_REFINABLE_SEGMENTS"

SEED_KEYS = ("seed_row", "seed_col")


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One non-branching 8-connected chain of the seed skeleton.

    Pixel coordinates follow the Field convention (rows are y, cols are x)
    and match the frame of the supplied seed mask.  ``xy_m`` and ``s_m`` are
    physical metres; ``s_m`` is cumulative arc length from the run start.
    """

    run_id: int
    rows: np.ndarray
    cols: np.ndarray
    xy_m: np.ndarray
    s_m: np.ndarray


@dataclass(frozen=True, slots=True)
class CenterlineSegment:
    """A confidence-weighted smooth curve over one refinable run fragment."""

    segment_id: int
    source_indices: np.ndarray
    s_m: np.ndarray
    seed_xy_m: np.ndarray
    observed_midpoint_xy_m: np.ndarray
    refined_xy_m: np.ndarray
    confidence: np.ndarray
    midpoint_source: np.ndarray
    length_m: float
    metadata: dict[str, Any]
    flags: tuple[str, ...]


def order_seed_runs(
    seed_mask: np.ndarray,
    *,
    pixel_size_xy_m: tuple[float, float],
) -> list[SeedRun]:
    """Partition a binary seed skeleton into ordered non-branching runs.

    Connectivity is 8-neighbour.  Endpoints, isolated pixels and true
    junctions (degree >= 4, or degree >= 3 with more than one continuation
    beyond the candidate pair) are cut points and are never traversed, so
    runs never bridge ambiguous topology.  Digital-line staircases -- where
    the candidate pair merges into a single continuation -- are treated as
    simple chain pixels, so a straight diagonal line stays one run.  This is
    purely an ordering device: no graph, no fiber identities, no crossing
    resolution.

    Complexity is O(N) via a pixel-coordinate dictionary.
    """
    binary = np.asarray(seed_mask, dtype=bool)
    pixel_coords = np.argwhere(binary)
    if pixel_coords.size == 0:
        return []
    px, py = (float(value) for value in pixel_size_xy_m)
    coord_set: set[tuple[int, int]] = {
        (int(row), int(col)) for row, col in pixel_coords
    }

    def neighbors(row: int, col: int) -> list[tuple[int, int]]:
        found: list[tuple[int, int]] = []
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                neighbour = (row + drow, col + dcol)
                if neighbour in coord_set:
                    found.append(neighbour)
        return found

    degree: dict[tuple[int, int], int] = {}
    for row, col in pixel_coords:
        degree[(int(row), int(col))] = len(neighbors(int(row), int(col)))

    visited: set[tuple[int, int]] = set()
    cut: set[tuple[int, int]] = set()
    runs: list[list[tuple[int, int]]] = []

    def candidates_of(
        point: tuple[int, int],
        previous: tuple[int, int] | None,
    ) -> list[tuple[int, int]]:
        # cut pixels remain visible as candidates (they shape the topology)
        # but are never entered by the walk
        return [
            neighbour
            for neighbour in neighbors(*point)
            if neighbour != previous
            and neighbour not in visited
            and degree.get(neighbour, 0) >= 2
        ]

    def is_junction(point: tuple[int, int], previous: tuple[int, int] | None) -> bool:
        """True when ``point`` has multiple non-converging continuations."""
        choices = candidates_of(point, previous)
        if len(choices) < 2:
            return False
        return not _candidates_merge(choices, point, previous, neighbors)

    def forward_walk(chain: list[tuple[int, int]]) -> None:
        previous: tuple[int, int] | None = None
        current = chain[-1]
        while True:
            choices = candidates_of(current, previous)
            if not choices:
                return
            if is_junction(current, previous):
                cut.add(current)
                return
            following = choices[0]
            if following in cut:
                cut.add(current)
                return
            if is_junction(following, current):
                cut.add(following)
                return
            visited.add(following)
            chain.append(following)
            previous, current = current, following

    for row, col in pixel_coords:
        start_pixel = (int(row), int(col))
        if start_pixel in visited or start_pixel in cut:
            continue
        visited.add(start_pixel)
        if degree[start_pixel] <= 1:
            continue
        if is_junction(start_pixel, None):
            cut.add(start_pixel)
            continue
        chain = [start_pixel]
        forward_walk(chain)
        chain.reverse()
        forward_walk(chain)
        runs.append(chain)

    ordered: list[SeedRun] = []
    for run_id, chain in enumerate(runs):
        rows = np.asarray([point[0] for point in chain], dtype=int)
        cols = np.asarray([point[1] for point in chain], dtype=int)
        xy = np.column_stack((cols * px, rows * py))
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        s = np.zeros(chain.__len__(), dtype=float)
        s[1:] = np.cumsum(steps)
        ordered.append(SeedRun(run_id, rows, cols, xy, s))
    return ordered


def _candidates_merge(
    candidates: list[tuple[int, int]],
    junction: tuple[int, int],
    previous: tuple[int, int] | None,
    neighbors: Any,
) -> bool:
    """True when the candidates are a digital-line staircase.

    A degree-3 pixel is a staircase when its candidate pair merges into at
    most one continuation beyond the pair (the classic diagonal bump).  A
    branch such as a T-junction has two distinct continuations.  The local
    3x3 neighborhood is identical in both cases; this one-pixel lookahead
    disambiguates them; the incoming neighbor is never counted as a
    continuation.
    """
    pair = {junction, *candidates}
    beyond: set[tuple[int, int]] = set()
    for candidate in candidates:
        for neighbour in neighbors(*candidate):
            if neighbour not in pair and neighbour != previous:
                beyond.add(neighbour)
    # count 8-connected components among the continuations: two adjacent
    # forward pixels belong to the same continuing chain
    remaining = set(beyond)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            point = stack.pop()
            for neighbour in neighbors(*point):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    stack.append(neighbour)
    return components <= 1


def _pixel_to_sample_index(
    samples: Mapping[str, np.ndarray],
) -> dict[tuple[int, int], int]:
    rows = np.asarray(samples["seed_row"], dtype=int)
    cols = np.asarray(samples["seed_col"], dtype=int)
    mapping: dict[tuple[int, int], int] = {}
    for index in range(rows.size):
        key = (int(rows[index]), int(cols[index]))
        if key in mapping:
            raise ValueError("duplicate seed pixel in samples; seed mapping is not one-to-one")
        mapping[key] = index
    return mapping


def _refinable(
    observation: BoundaryMidpointObservation,
    config: CenterlineRefinementConfig,
) -> bool:
    if not observation.accepted:
        return False
    if observation.preferred_midpoint_xy_m is None:
        return False
    if not all(math.isfinite(value) for value in observation.preferred_midpoint_xy_m):
        return False
    if not math.isfinite(observation.refinement_confidence):
        return False
    hard = (
        ABSTAIN_EDGE_FLAGS
        | ABSTAIN_PROFILE_FLAGS
        | {LOW_COHERENCE_FLAG}
    )
    return not any(flag in hard for flag in observation.flags)


def refine_centerline(
    refinement: CenterlineRefinementResult,
    samples: Mapping[str, np.ndarray],
    seed_mask: np.ndarray,
    *,
    pixel_size_xy_m: tuple[float, float],
    config: CenterlineRefinementConfig | None = None,
) -> CenterlineRefinementResult:
    """Build the smooth refined centerline from stage-1 observations.

    ``samples`` must contain the ``seed_row`` / ``seed_col`` pixel mapping
    (same frame as ``seed_mask``) in addition to the Field local samples.
    Only accepted observations on non-branching seed runs are fitted; gaps
    larger than ``max_gap_factor`` times the run's median seed step split
    segments, and fragments below ``min_segment_points`` are flagged instead
    of extrapolated.
    """
    if config is None:
        config = CenterlineRefinementConfig()
    missing = [key for key in SEED_KEYS if key not in samples]
    if missing:
        raise ValueError(f"refinement requires local_samples keys {missing}")
    n = int(refinement.original_xy_m.shape[0])
    runs = order_seed_runs(seed_mask, pixel_size_xy_m=pixel_size_xy_m)
    sample_index = _pixel_to_sample_index(samples)
    observations = {item.source_index: item for item in refinement.observations}
    normal = np.asarray(samples["normal_xy"], dtype=float)
    tangent = np.column_stack((normal[:, 1], -normal[:, 0]))

    refined_xy = np.full((n, 2), np.nan)
    refined_mask = np.zeros(n, dtype=bool)
    segment_ids = np.full(n, -1, dtype=int)
    smooth_shift = np.full(n, np.nan)
    smooth_normal = np.full(n, np.nan)
    smooth_tangent = np.full(n, np.nan)
    segments: list[CenterlineSegment] = []
    too_short = 0
    segment_counter = 0

    for run in runs:
        run_samples = _samples_in_run(run, sample_index)
        if not run_samples:
            continue
        median_step = _median_run_step(run)
        max_gap = config.max_gap_factor * median_step
        for fragment in _split_refinable_fragments(
            run, run_samples, observations, samples, config, max_gap=max_gap
        ):
            indices = fragment["indices"]
            if indices.size < config.min_segment_points:
                too_short += 1
                continue
            fitted = _fit_fragment(
                fragment, refinement, samples, normal, tangent, config
            )
            if fitted is None:
                too_short += 1
                continue
            (
                segment_refined,
                segment_smooth_shift,
                segment_smooth_normal,
                segment_smooth_tangent,
            ) = fitted
            segment = CenterlineSegment(
                segment_id=segment_counter,
                source_indices=indices,
                s_m=fragment["s_m"],
                seed_xy_m=fragment["seed_xy"],
                observed_midpoint_xy_m=fragment["midpoints"],
                refined_xy_m=segment_refined,
                confidence=fragment["confidence"],
                midpoint_source=fragment["source"],
                length_m=float(fragment["s_m"][-1] - fragment["s_m"][0]),
                metadata={
                    "run_id": int(run.run_id),
                    "run_length_m": float(run.s_m[-1] - run.s_m[0]),
                    "median_seed_step_m": float(median_step),
                    "smoothing": SMOOTHING,
                    "parameterization": "physical_arc_length_m",
                },
                flags=(),
            )
            segments.append(segment)
            refined_xy[indices] = segment_refined
            refined_mask[indices] = True
            segment_ids[indices] = segment_counter
            smooth_shift[indices] = segment_smooth_shift
            smooth_normal[indices] = segment_smooth_normal
            smooth_tangent[indices] = segment_smooth_tangent
            segment_counter += 1

    summary = _refined_summary(
        refinement,
        refined_mask,
        smooth_shift,
        segment_ids=segment_ids,
    )
    result_flags: list[str] = [STAGE]
    if too_short:
        result_flags.append(FLAG_SEGMENT_TOO_SHORT)
    if not segments:
        result_flags.append(FLAG_NO_REFINEMENT)
    metadata = {
        **dict(refinement.metadata),
        "stage": STAGE,
        "ordering": ORDERING,
        "smoothing": SMOOTHING,
        "config": asdict(config),
        "run_count": len(runs),
    }
    return CenterlineRefinementResult(
        observations=refinement.observations,
        original_xy_m=refinement.original_xy_m,
        accepted_mask=refinement.accepted_mask,
        mask_midpoint_xy_m=refinement.mask_midpoint_xy_m,
        profile_midpoint_xy_m=refinement.profile_midpoint_xy_m,
        preferred_midpoint_xy_m=refinement.preferred_midpoint_xy_m,
        midpoint_source=refinement.midpoint_source,
        confidence=refinement.confidence,
        shift_um=refinement.shift_um,
        signed_normal_shift_um=refinement.signed_normal_shift_um,
        tangential_shift_um=refinement.tangential_shift_um,
        width_um=refinement.width_um,
        coverage_fraction=refinement.coverage_fraction,
        summary=summary,
        flags=tuple(result_flags),
        metadata=metadata,
        segments=tuple(segments),
        refined_xy_m=refined_xy,
        refined_mask=refined_mask,
        segment_ids=segment_ids,
        smooth_shift_um=smooth_shift,
        smooth_normal_shift_um=smooth_normal,
        smooth_tangential_shift_um=smooth_tangent,
    )


def _samples_in_run(
    run: SeedRun,
    sample_index: dict[tuple[int, int], int],
) -> list[tuple[int, int, int]]:
    """Return (pixel_index, sample_index) pairs for this run in run order."""
    found: list[tuple[int, int, int]] = []
    for pixel_index in range(run.rows.size):
        key = (int(run.rows[pixel_index]), int(run.cols[pixel_index]))
        sample = sample_index.get(key)
        if sample is not None:
            found.append((pixel_index, sample))
    return found


def _median_run_step(run: SeedRun) -> float:
    if run.xy_m.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(run.xy_m, axis=0), axis=1)
    return float(np.median(steps)) if steps.size else 0.0


def _split_refinable_fragments(
    run: SeedRun,
    run_samples: list[tuple[int, int, int]],
    observations: dict[int, BoundaryMidpointObservation],
    samples: Mapping[str, np.ndarray],
    config: CenterlineRefinementConfig,
    *,
    max_gap: float,
) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    current: list[int] = []
    current_s: list[float] = []
    previous_s: float | None = None
    for pixel_index, sample_index in run_samples:
        observation = observations.get(sample_index)
        refinable = observation is not None and _refinable(observation, config)
        s_value = float(run.s_m[pixel_index])
        gap_ok = (
            previous_s is None
            or (max_gap > 0.0 and (s_value - previous_s) <= max_gap)
        )
        if (not refinable or not gap_ok) and current:
            fragments.append(
                _fragment_from_current(run, current, current_s, observations, samples)
            )
            current = []
            current_s = []
        if refinable:
            current.append(sample_index)
            current_s.append(s_value)
        previous_s = s_value
    if current:
        fragments.append(
            _fragment_from_current(run, current, current_s, observations, samples)
        )
    return fragments


def _fragment_from_current(
    run: SeedRun,
    indices: list[int],
    s_values: list[float],
    observations: dict[int, BoundaryMidpointObservation],
    samples: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    index_array = np.asarray(indices, dtype=int)
    s_array = np.asarray(s_values, dtype=float)
    midpoints = refinement_preferred_midpoints(index_array, observations)
    seed_xy = np.asarray(
        [np.asarray(observations[index].original_xy_m) for index in indices],
        dtype=float,
    )
    confidence = np.asarray(
        [observations[index].refinement_confidence for index in indices], dtype=float
    )
    source = np.asarray(
        [
            observations[index].preferred_midpoint_source or ""
            for index in indices
        ],
        dtype="<U8",
    )
    return {
        "indices": index_array,
        "s_m": s_array,
        "midpoints": midpoints,
        "seed_xy": seed_xy,
        "confidence": confidence,
        "source": source,
    }


def refinement_preferred_midpoints(
    indices: np.ndarray,
    observations: dict[int, BoundaryMidpointObservation],
) -> np.ndarray:
    midpoints = np.full((indices.size, 2), np.nan)
    for position, index in enumerate(indices):
        midpoint = observations[index].preferred_midpoint_xy_m
        if midpoint is not None:
            midpoints[position] = midpoint
    return midpoints


def _fit_fragment(
    fragment: dict[str, Any],
    refinement: CenterlineRefinementResult,
    samples: Mapping[str, np.ndarray],
    normal: np.ndarray,
    tangent: np.ndarray,
    config: CenterlineRefinementConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    indices = fragment["indices"]
    s_m = fragment["s_m"]
    midpoints = fragment["midpoints"]
    if indices.size < 2 or not np.isfinite(midpoints).all():
        return None
    start, end = float(s_m[0]), float(s_m[-1])
    length = end - start
    if length <= 0.0:
        return None
    width_um = refinement.width_um[indices]
    finite_width = width_um[np.isfinite(width_um)]
    char_width = float(np.median(finite_width)) * 1e-6 if finite_width.size else 1.0
    if not math.isfinite(char_width) or char_width <= 0.0:
        char_width = 1.0
    s_norm = (s_m - start) / length
    xy_norm = midpoints / char_width
    weights = np.clip(fragment["confidence"], 1e-6, 1.0)
    if np.sum(weights) <= 0.0:
        return None
    count = indices.size
    # smoothing_strength == 1.0 selects the data-driven GCV smoothing
    # (validated on synthetic truth); other values scale a documented
    # baseline lam = strength * n * 1e-4 in normalized units.
    lam = None if config.smoothing_strength == 1.0 else config.smoothing_strength * count * 1e-4
    try:
        spline_x = make_smoothing_spline(s_norm, xy_norm[:, 0], w=weights, lam=lam)
        spline_y = make_smoothing_spline(s_norm, xy_norm[:, 1], w=weights, lam=lam)
    except Exception:
        return None
    refined_norm = np.column_stack((spline_x(s_norm), spline_y(s_norm)))
    refined = refined_norm * char_width
    original = refinement.original_xy_m[indices]
    shift_vector = refined - original
    smooth_shift = np.linalg.norm(shift_vector, axis=1) * 1e6
    sample_normal = normal[indices]
    sample_tangent = tangent[indices]
    smooth_normal = (
        shift_vector[:, 0] * sample_normal[:, 0]
        + shift_vector[:, 1] * sample_normal[:, 1]
    ) * 1e6
    smooth_tangent = (
        shift_vector[:, 0] * sample_tangent[:, 0]
        + shift_vector[:, 1] * sample_tangent[:, 1]
    ) * 1e6
    return refined, smooth_shift, smooth_normal, smooth_tangent


def _refined_summary(
    refinement: CenterlineRefinementResult,
    refined_mask: np.ndarray,
    smooth_shift: np.ndarray,
    *,
    segment_ids: np.ndarray,
) -> dict[str, float | int | None]:
    n = refined_mask.size
    accepted_count = int(np.sum(refinement.accepted_mask))
    smoothed_count = int(np.sum(refined_mask))
    segment_count = _segment_count(segment_ids)
    accepted_shift = refinement.shift_um[refinement.accepted_mask]
    smooth_values = smooth_shift[refined_mask]
    widths = refinement.width_um[refined_mask]
    fractions = smooth_values / widths
    return {
        **refinement.summary,
        "observation_coverage": refinement.coverage_fraction,
        "smooth_coverage": smoothed_count / n if n else 0.0,
        "segment_count": segment_count,
        "accepted_observation_count": accepted_count,
        "smoothed_sample_count": smoothed_count,
        "median_observed_shift_um": float(np.median(accepted_shift)) if accepted_shift.size else None,
        "median_smooth_shift_um": float(np.median(smooth_values)) if smooth_values.size else None,
        "p90_smooth_shift_um": float(np.quantile(smooth_values, 0.9)) if smooth_values.size else None,
        "median_smooth_shift_fraction_of_width": float(np.median(fractions)) if fractions.size else None,
    }


def _segment_count(segment_ids: np.ndarray) -> int:
    """Number of distinct refined segment ids."""
    if segment_ids is None or not segment_ids.size:
        return 0
    return int(np.sum(np.unique(segment_ids) >= 0))


__all__ = [
    "FLAG_NO_REFINEMENT",
    "FLAG_SEGMENT_TOO_SHORT",
    "ORDERING",
    "SMOOTHING",
    "STAGE",
    "CenterlineSegment",
    "SeedRun",
    "order_seed_runs",
    "refine_centerline",
]
