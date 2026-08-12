"""FATHOM_ORIENTED_RIBBON_V1 — stage 1: boundary midpoint observations.

Given the paired mask boundaries and profile-refined boundaries that
``FATHOM_FIELD_GRAPH_V1`` already computes for each centerline sample, this
module builds explicit geometric midpoint observations:

    m      = (p_minus + p_plus) / 2
    width  = ||p_plus - p_minus||
    delta  = m - c0

Midpoints are observations only.  No refined centerline curve, spline or
remeasurement is produced in this stage; crossing ambiguities abstain and no
ground-truth claim is made.  All geometry is handled in physical (metre)
coordinates exactly as the Field backend already stores them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

ALGORITHM_ID = "FATHOM_ORIENTED_RIBBON_V1"
STAGE = "MIDPOINT_OBSERVATIONS_ONLY"

# Paired-edge flags that make the whole sample abstain: a wrong midpoint is
# worse than a missing midpoint.  Names reuse the Field backend verbatim.
ABSTAIN_EDGE_FLAGS = frozenset(
    {
        "POSSIBLE_CROSSING",
        "AMBIGUOUS_LOCAL_WIDTH",
        "MISSING_POSITIVE_EDGE",
        "MISSING_NEGATIVE_EDGE",
    }
)
# Profile-side abstention only invalidates the profile observation; the mask
# midpoint remains usable unless the paired edge itself abstains.
ABSTAIN_PROFILE_FLAGS = frozenset(
    {
        "PROFILE_AMBIGUOUS_EDGE",
        "PROFILE_EDGE_MINUS_NOT_FOUND",
        "PROFILE_EDGE_PLUS_NOT_FOUND",
        "PROFILE_NONPOSITIVE_WIDTH",
    }
)
LOW_COHERENCE_FLAG = "LOW_ORIENTATION_COHERENCE"

REQUIRED_KEYS = frozenset(
    {
        "x_m",
        "y_m",
        "qx",
        "qy",
        "coherence",
        "normal_xy",
        "minus_xy_m",
        "plus_xy_m",
        "radius_minus_um",
        "radius_plus_um",
        "edge_accepted",
        "edge_flags",
    }
)
PROFILE_KEYS = (
    "profile_minus_u_um",
    "profile_plus_u_um",
    "profile_accepted",
    "profile_flags",
    "profile_gradient_snr",
)


@dataclass(frozen=True, slots=True)
class CenterlineRefinementConfig:
    """Parameters used by the midpoint observation stage.

    ``min_coherence`` mirrors the Field backend's ``low_coherence`` threshold
    (0.15) so the ribbon stage never accepts what the orientation field
    itself already distrusts.  ``tangential_mismatch_fraction`` is a generic
    geometric sanity bound for the diagnostic flag only; it never rejects.
    """

    min_coherence: float = 0.15
    tangential_mismatch_fraction: float = 0.5


@dataclass(frozen=True, slots=True)
class BoundaryMidpointObservation:
    """One centerline sample's midpoint observation from its local boundaries.

    All positions are physical metres; widths and shifts are microns,
    following the Field ``local_samples`` convention.  ``preferred_midpoint``
    is PROFILE when profile geometry is valid, else MASK, else None.
    """

    source_index: int
    original_xy_m: tuple[float, float]
    tangent_xy: tuple[float, float]
    normal_xy: tuple[float, float]
    coherence: float

    mask_minus_xy_m: tuple[float, float] | None
    mask_plus_xy_m: tuple[float, float] | None
    mask_midpoint_xy_m: tuple[float, float] | None
    mask_width_um: float | None
    shift_mask_um: float | None
    signed_normal_shift_mask_um: float | None
    tangential_shift_mask_um: float | None

    profile_minus_xy_m: tuple[float, float] | None
    profile_plus_xy_m: tuple[float, float] | None
    profile_midpoint_xy_m: tuple[float, float] | None
    profile_width_um: float | None
    shift_profile_um: float | None
    signed_normal_shift_profile_um: float | None
    tangential_shift_profile_um: float | None
    profile_snr: float | None

    mask_profile_center_disagreement_m: float | None
    mask_profile_center_disagreement_fraction: float | None

    preferred_midpoint_source: str | None
    preferred_midpoint_xy_m: tuple[float, float] | None
    refinement_confidence: float
    accepted: bool
    flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CenterlineRefinementResult:
    """Stage-1 ribbon output: midpoint observations only, no refined curve."""

    observations: tuple[BoundaryMidpointObservation, ...]
    original_xy_m: np.ndarray
    accepted_mask: np.ndarray
    mask_midpoint_xy_m: np.ndarray
    profile_midpoint_xy_m: np.ndarray
    preferred_midpoint_xy_m: np.ndarray
    midpoint_source: np.ndarray
    confidence: np.ndarray
    shift_um: np.ndarray
    signed_normal_shift_um: np.ndarray
    tangential_shift_um: np.ndarray
    width_um: np.ndarray
    coverage_fraction: float
    summary: dict[str, float | int | None]
    flags: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def refined_xy_m(self) -> None:
        """Batch 2 contract placeholder; never computed in this stage."""
        return None


def _split_flags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    text = str(value)
    return tuple(flag for flag in text.split(";") if flag)


def compute_midpoint_observations(
    samples: Mapping[str, np.ndarray],
    *,
    config: CenterlineRefinementConfig | None = None,
    include_observations: bool = True,
) -> CenterlineRefinementResult:
    """Build midpoint observations from existing Field ``local_samples``.

    The mapping must contain the paired-edge fields the Field adapter already
    stores (``x_m``, ``y_m``, ``normal_xy``, ``minus_xy_m``, ``plus_xy_m``,
    radii, acceptance and flags); profile fields are optional and treated as
    absent when missing.  No pixel data, gradient search or resampling is
    performed: every midpoint is reconstructed from already-accepted
    boundaries.
    """
    if config is None:
        config = CenterlineRefinementConfig()
    missing = REQUIRED_KEYS - set(samples)
    if missing:
        raise ValueError(f"midpoint observations require local_samples keys {sorted(missing)}")

    def array(key: str) -> np.ndarray:
        return np.asarray(samples[key], dtype=float)

    n = int(array("x_m").size)
    for key in REQUIRED_KEYS:
        value = np.asarray(samples[key])
        if key in {"normal_xy", "minus_xy_m", "plus_xy_m"}:
            if value.shape != (n, 2):
                raise ValueError(f"local_samples[{key}] must have shape ({n}, 2)")
        elif value.size != n:
            raise ValueError(f"local_samples[{key}] must have size {n}")

    has_profile = all(key in samples for key in ("profile_minus_u_um", "profile_plus_u_um", "profile_accepted", "profile_flags"))
    x_m, y_m = array("x_m"), array("y_m")
    qx, qy = array("qx"), array("qy")
    coherence = array("coherence")
    normal = array("normal_xy")
    minus_xy = array("minus_xy_m")
    plus_xy = array("plus_xy_m")
    radius_minus = array("radius_minus_um") * 1e-6
    radius_plus = array("radius_plus_um") * 1e-6
    edge_accepted = np.asarray(samples["edge_accepted"], bool)
    edge_flags = np.asarray(samples["edge_flags"])
    if has_profile:
        u_minus = array("profile_minus_u_um") * 1e-6
        u_plus = array("profile_plus_u_um") * 1e-6
        profile_accepted = np.asarray(samples["profile_accepted"], bool)
        profile_flags = np.asarray(samples["profile_flags"])
        snr = (
            np.asarray(samples["profile_gradient_snr"], dtype=float)
            if "profile_gradient_snr" in samples
            else np.full(n, np.nan)
        )
    else:
        u_minus = np.full(n, np.nan)
        u_plus = np.full(n, np.nan)
        profile_accepted = np.zeros(n, bool)
        profile_flags = np.full(n, "", dtype="<U80")
        snr = np.full(n, np.nan)

    observations: list[BoundaryMidpointObservation] = []
    accepted = np.zeros(n, bool)
    mask_mid = np.full((n, 2), np.nan)
    profile_mid = np.full((n, 2), np.nan)
    preferred_xy = np.full((n, 2), np.nan)
    source = np.full(n, "", dtype="<U8")
    confidence = np.zeros(n)
    shift = np.full(n, np.nan)
    signed_shift = np.full(n, np.nan)
    tangential_shift = np.full(n, np.nan)
    width = np.full(n, np.nan)
    observed_flags: set[str] = set()

    for index in range(n):
        observation, sample_shift, sample_signed, sample_tangential, sample_width, is_accepted = _observe(
            index,
            x_m=x_m, y_m=y_m, qx=qx, qy=qy, coherence=coherence, normal=normal,
            minus_xy=minus_xy, plus_xy=plus_xy,
            radius_minus=radius_minus, radius_plus=radius_plus,
            edge_accepted=edge_accepted, edge_flags=edge_flags,
            u_minus=u_minus, u_plus=u_plus,
            profile_accepted=profile_accepted, profile_flags=profile_flags, snr=snr,
            has_profile=has_profile,
            config=config,
        )
        if include_observations:
            observations.append(observation)
        observed_flags.update(observation.flags)
        accepted[index] = is_accepted
        confidence[index] = observation.refinement_confidence
        if observation.mask_midpoint_xy_m is not None:
            mask_mid[index] = observation.mask_midpoint_xy_m
        if observation.profile_midpoint_xy_m is not None:
            profile_mid[index] = observation.profile_midpoint_xy_m
        midpoint = observation.preferred_midpoint_xy_m
        if midpoint is not None:
            preferred_xy[index] = midpoint
            source[index] = observation.preferred_midpoint_source
        shift[index] = sample_shift
        signed_shift[index] = sample_signed
        tangential_shift[index] = sample_tangential
        width[index] = sample_width

    accepted_count = int(np.sum(accepted))
    coverage = accepted_count / n if n else 0.0
    accepted_shift = shift[accepted]
    accepted_width = width[accepted]
    accepted_fraction = accepted_shift / accepted_width
    summary = {
        "accepted_count": accepted_count,
        "rejected_count": n - accepted_count,
        "median_shift_um": float(np.median(accepted_shift)) if accepted_shift.size else None,
        "p90_shift_um": float(np.quantile(accepted_shift, 0.9)) if accepted_shift.size else None,
        "median_shift_fraction": float(np.median(accepted_fraction)) if accepted_fraction.size else None,
    }
    metadata = {
        "algorithm": ALGORITHM_ID,
        "stage": STAGE,
        "config": asdict(config),
        "consumed_local_samples_keys": sorted(REQUIRED_KEYS | set(PROFILE_KEYS) & set(samples)),
        "coordinate_frame": "physical_meters",
    }
    result_flags = ("MIDPOINT_OBSERVATIONS_ONLY", *sorted(observed_flags - {"MIDPOINT_OBSERVATIONS_ONLY"}))
    return CenterlineRefinementResult(
        observations=tuple(observations),
        original_xy_m=np.column_stack((x_m, y_m)),
        accepted_mask=accepted,
        mask_midpoint_xy_m=mask_mid,
        profile_midpoint_xy_m=profile_mid,
        preferred_midpoint_xy_m=preferred_xy,
        midpoint_source=source,
        confidence=confidence,
        shift_um=shift,
        signed_normal_shift_um=signed_shift,
        tangential_shift_um=tangential_shift,
        width_um=width,
        coverage_fraction=coverage,
        summary=summary,
        flags=result_flags,
        metadata=metadata,
    )


def _observe(
    index: int,
    *,
    x_m: np.ndarray,
    y_m: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    coherence: np.ndarray,
    normal: np.ndarray,
    minus_xy: np.ndarray,
    plus_xy: np.ndarray,
    radius_minus: np.ndarray,
    radius_plus: np.ndarray,
    edge_accepted: np.ndarray,
    edge_flags: np.ndarray,
    u_minus: np.ndarray,
    u_plus: np.ndarray,
    profile_accepted: np.ndarray,
    profile_flags: np.ndarray,
    snr: np.ndarray,
    has_profile: bool,
    config: CenterlineRefinementConfig,
) -> tuple[BoundaryMidpointObservation, float, float, float, float, bool]:
    flags: list[str] = []
    c0 = (float(x_m[index]), float(y_m[index]))
    normal_vector = (float(normal[index, 0]), float(normal[index, 1]))
    tangent_vector = (normal_vector[1], -normal_vector[0])
    coherence_value = float(coherence[index])
    edge_flag_set = set(_split_flags(edge_flags[index]))
    profile_flag_set = set(_split_flags(profile_flags[index]))
    observed_snr = float(snr[index]) if np.isfinite(snr[index]) else None

    def normal_valid() -> bool:
        norm = np.hypot(*normal_vector)
        return bool(np.isfinite(norm) and norm > 0.0)

    if not normal_valid():
        flags.append("MIDPOINT_INVALID_NORMAL")
    else:
        normal_vector = (normal_vector[0] / np.hypot(*normal_vector), normal_vector[1] / np.hypot(*normal_vector))
        tangent_vector = (normal_vector[1], -normal_vector[0])

    low_coherence = bool(
        coherence_value < config.min_coherence
        or LOW_COHERENCE_FLAG in edge_flag_set
    )
    if low_coherence:
        flags.append(LOW_COHERENCE_FLAG)

    sample_abstain = bool(edge_flag_set & ABSTAIN_EDGE_FLAGS)

    # ---- mask geometry (paired mask boundaries) ---------------------------
    minus_m = (float(minus_xy[index, 0]), float(minus_xy[index, 1]))
    plus_m = (float(plus_xy[index, 0]), float(plus_xy[index, 1]))
    minus_finite = np.isfinite(minus_xy[index]).all()
    plus_finite = np.isfinite(plus_xy[index]).all()
    mask_valid = (
        bool(edge_accepted[index])
        and not sample_abstain
        and not low_coherence
        and normal_valid()
        and minus_finite
        and plus_finite
    )
    if not minus_finite:
        flags.append("MISSING_NEGATIVE_EDGE")
    if not plus_finite:
        flags.append("MISSING_POSITIVE_EDGE")
    mask_midpoint: tuple[float, float] | None = None
    mask_width_um: float | None = None
    shift_mask_um: float | None = None
    signed_mask_um: float | None = None
    tangent_mask_um: float | None = None
    if mask_valid:
        mask_midpoint = (
            0.5 * (minus_m[0] + plus_m[0]),
            0.5 * (minus_m[1] + plus_m[1]),
        )
        width_m = math.hypot(plus_m[0] - minus_m[0], plus_m[1] - minus_m[1])
        if width_m <= 0.0:
            mask_valid = False
            flags.append("MIDPOINT_ZERO_WIDTH")
        else:
            mask_width_um = width_m * 1e6
            shift_mask_um, signed_mask_um, tangent_mask_um = _shift_components(
                mask_midpoint, c0, normal_vector, tangent_vector
            )

    # ---- profile geometry (reconstructed from existing u positions) -------
    profile_minus_m: tuple[float, float] | None = None
    profile_plus_m: tuple[float, float] | None = None
    profile_midpoint: tuple[float, float] | None = None
    profile_width_um: float | None = None
    shift_profile_um: float | None = None
    signed_profile_um: float | None = None
    tangent_profile_um: float | None = None
    profile_valid = False
    if has_profile and normal_valid() and not low_coherence and not sample_abstain:
        u_minus_value, u_plus_value = float(u_minus[index]), float(u_plus[index])
        profile_valid = (
            bool(profile_accepted[index])
            and not (profile_flag_set & ABSTAIN_PROFILE_FLAGS)
            and np.isfinite(u_minus_value)
            and np.isfinite(u_plus_value)
        )
        if not profile_valid:
            flags.extend(sorted(profile_flag_set))
        if profile_valid:
            profile_minus_m = (
                c0[0] + u_minus_value * normal_vector[0],
                c0[1] + u_minus_value * normal_vector[1],
            )
            profile_plus_m = (
                c0[0] + u_plus_value * normal_vector[0],
                c0[1] + u_plus_value * normal_vector[1],
            )
            profile_midpoint = (
                0.5 * (profile_minus_m[0] + profile_plus_m[0]),
                0.5 * (profile_minus_m[1] + profile_plus_m[1]),
            )
            width_m = math.hypot(
                profile_plus_m[0] - profile_minus_m[0],
                profile_plus_m[1] - profile_minus_m[1],
            )
            if width_m <= 0.0:
                profile_valid = False
                flags.append("MIDPOINT_ZERO_WIDTH")
            else:
                profile_width_um = width_m * 1e6
                shift_profile_um, signed_profile_um, tangent_profile_um = _shift_components(
                    profile_midpoint, c0, normal_vector, tangent_vector
                )

    # ---- preferred midpoint (frozen rule: PROFILE > MASK > NONE) ----------
    if sample_abstain:
        flags.extend(sorted(edge_flag_set & ABSTAIN_EDGE_FLAGS))
    preferred_source: str | None
    if sample_abstain or low_coherence:
        preferred_source = None
    elif profile_valid:
        preferred_source = "PROFILE"
    elif mask_valid:
        preferred_source = "MASK"
    else:
        preferred_source = None
    accepted_flag = preferred_source is not None

    if preferred_source == "PROFILE":
        preferred_midpoint = profile_midpoint
        shift_value, signed_value, tangent_value, width_value = (
            shift_profile_um, signed_profile_um, tangent_profile_um, profile_width_um
        )
    elif preferred_source == "MASK":
        preferred_midpoint = mask_midpoint
        shift_value, signed_value, tangent_value, width_value = (
            shift_mask_um, signed_mask_um, tangent_mask_um, mask_width_um
        )
    else:
        preferred_midpoint = None
        shift_value = signed_value = tangent_value = width_value = None

    disagreement_m: float | None = None
    disagreement_fraction: float | None = None
    if profile_midpoint is not None and mask_midpoint is not None and mask_valid:
        disagreement_m = math.hypot(
            profile_midpoint[0] - mask_midpoint[0],
            profile_midpoint[1] - mask_midpoint[1],
        )
        if width_value and width_value > 0.0:
            disagreement_fraction = disagreement_m / (width_value * 1e-6)

    if (
        preferred_midpoint is not None
        and width_value
        and tangent_value is not None
        and abs(tangent_value) > config.tangential_mismatch_fraction * width_value
    ):
        flags.append("MIDPOINT_TANGENTIAL_MISMATCH")

    orientation_confidence = float(np.clip(coherence_value, 0.0, 1.0))
    pair_validity = 1.0 if (bool(edge_accepted[index]) and not sample_abstain and not low_coherence) else 0.0
    refinement_confidence = min(orientation_confidence, pair_validity)
    if profile_valid:
        refinement_confidence = min(refinement_confidence, 1.0)

    observation = BoundaryMidpointObservation(
        source_index=index,
        original_xy_m=c0,
        tangent_xy=tangent_vector,
        normal_xy=normal_vector,
        coherence=coherence_value,
        mask_minus_xy_m=minus_m if minus_finite else None,
        mask_plus_xy_m=plus_m if plus_finite else None,
        mask_midpoint_xy_m=mask_midpoint,
        mask_width_um=mask_width_um,
        shift_mask_um=shift_mask_um,
        signed_normal_shift_mask_um=signed_mask_um,
        tangential_shift_mask_um=tangent_mask_um,
        profile_minus_xy_m=profile_minus_m,
        profile_plus_xy_m=profile_plus_m,
        profile_midpoint_xy_m=profile_midpoint,
        profile_width_um=profile_width_um,
        shift_profile_um=shift_profile_um,
        signed_normal_shift_profile_um=signed_profile_um,
        tangential_shift_profile_um=tangent_profile_um,
        profile_snr=observed_snr,
        mask_profile_center_disagreement_m=disagreement_m,
        mask_profile_center_disagreement_fraction=disagreement_fraction,
        preferred_midpoint_source=preferred_source,
        preferred_midpoint_xy_m=preferred_midpoint,
        refinement_confidence=refinement_confidence,
        accepted=accepted_flag,
        flags=tuple(dict.fromkeys(flags)),
    )
    return (
        observation,
        shift_value if shift_value is not None else np.nan,
        signed_value if signed_value is not None else np.nan,
        tangent_value if tangent_value is not None else np.nan,
        width_value if width_value is not None else np.nan,
        accepted_flag,
    )


def _shift_components(
    midpoint: tuple[float, float],
    center: tuple[float, float],
    normal_vector: tuple[float, float],
    tangent_vector: tuple[float, float],
) -> tuple[float, float, float]:
    """Return (magnitude_um, signed_normal_um, tangential_um) of m - c0."""
    delta = (midpoint[0] - center[0], midpoint[1] - center[1])
    magnitude_m = math.hypot(*delta)
    signed_normal_m = delta[0] * normal_vector[0] + delta[1] * normal_vector[1]
    tangential_m = delta[0] * tangent_vector[0] + delta[1] * tangent_vector[1]
    return magnitude_m * 1e6, signed_normal_m * 1e6, tangential_m * 1e6


__all__ = [
    "ABSTAIN_EDGE_FLAGS",
    "ABSTAIN_PROFILE_FLAGS",
    "ALGORITHM_ID",
    "LOW_COHERENCE_FLAG",
    "STAGE",
    "BoundaryMidpointObservation",
    "CenterlineRefinementConfig",
    "CenterlineRefinementResult",
    "compute_midpoint_observations",
]
