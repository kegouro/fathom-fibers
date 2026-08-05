from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from .measurement_records import MeasurementRecord


def get_utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RepeatabilityItem:
    reference_group_id: str
    record_id: str
    original_value_m: float
    round_number: int
    measured_value_m: float | None = None
    timestamp: str = field(default_factory=get_utc_now_iso)


@dataclass
class RepeatabilitySession:
    study_id: str
    operator_id: str
    random_seed: int
    items: list[RepeatabilityItem] = field(default_factory=list)
    created_at: str = field(default_factory=get_utc_now_iso)
    is_completed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "operator_id": self.operator_id,
            "random_seed": self.random_seed,
            "items": [
                {
                    "reference_group_id": it.reference_group_id,
                    "record_id": it.record_id,
                    "original_value_m": it.original_value_m,
                    "round_number": it.round_number,
                    "measured_value_m": it.measured_value_m,
                    "timestamp": it.timestamp,
                }
                for it in self.items
            ],
            "created_at": self.created_at,
            "is_completed": self.is_completed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepeatabilitySession:
        items = [
            RepeatabilityItem(
                reference_group_id=d["reference_group_id"],
                record_id=d["record_id"],
                original_value_m=d["original_value_m"],
                round_number=d.get("round_number", 1),
                measured_value_m=d.get("measured_value_m"),
                timestamp=d.get("timestamp", get_utc_now_iso()),
            )
            for d in data.get("items", [])
        ]
        return cls(
            study_id=data.get("study_id", "STUDY_001"),
            operator_id=data.get("operator_id", "OPERATOR_1"),
            random_seed=int(data.get("random_seed", 42)),
            items=items,
            created_at=data.get("created_at", get_utc_now_iso()),
            is_completed=bool(data.get("is_completed", False)),
        )


def create_blind_session(
    records: Sequence[MeasurementRecord],
    operator_id: str = "OPERATOR_1",
    seed: int | None = None,
) -> RepeatabilitySession:
    """Creates a randomized blind repeatability session hiding original values."""
    if seed is None:
        seed = random.randint(1000, 999999)

    study_id = f"STUDY_{seed:06d}"
    items: list[RepeatabilityItem] = []

    # Filter records with primary values
    valid_records = [r for r in records if r.primary_value is not None]
    rng = random.Random(seed)
    shuffled = list(valid_records)
    rng.shuffle(shuffled)

    for idx, r in enumerate(shuffled):
        grp_id = f"GRP_{idx + 1:03d}"
        items.append(
            RepeatabilityItem(
                reference_group_id=grp_id,
                record_id=r.measurement_id,
                original_value_m=float(r.primary_value),
                round_number=1,
            )
        )

    return RepeatabilitySession(study_id=study_id, operator_id=operator_id, random_seed=seed, items=items)


def analyze_repeatability(session: RepeatabilitySession) -> dict[str, Any]:
    """Calculates intra-operator SD, CV, bias, and Bland-Altman limits when N is sufficient."""
    completed = [it for it in session.items if it.measured_value_m is not None]
    n_pairs = len(completed)

    if n_pairs < 2:
        return {
            "n_pairs": n_pairs,
            "status": "N_INSUFFICIENT",
            "message": "N insuficiente para métricas de repetibilidad (se requieren al menos 2 pares).",
        }

    origs = np.array([it.original_value_m for it in completed], dtype=np.float64)
    reps = np.array([it.measured_value_m for it in completed], dtype=np.float64)

    abs_diffs = np.abs(origs - reps)
    mean_val = float(np.mean((origs + reps) / 2.0))

    # Intra-operator SD: S_intra = sqrt(sum((x1 - x2)^2) / (2N))
    s_intra = float(np.sqrt(np.sum((origs - reps) ** 2) / (2 * n_pairs)))
    cv = (s_intra / mean_val) if mean_val > 0 else 0.0

    # Inter-round bias d = mean(x2 - x1)
    diffs = reps - origs
    bias = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if n_pairs > 1 else 0.0

    # Bland-Altman limits if N >= 5
    ba_lower, ba_upper = None, None
    if n_pairs >= 5:
        ba_lower = bias - 1.96 * sd_diff
        ba_upper = bias + 1.96 * sd_diff

    return {
        "n_pairs": n_pairs,
        "status": "SUFFICIENT",
        "mean_value_m": mean_val,
        "mean_abs_diff_m": float(np.mean(abs_diffs)),
        "mean_rel_diff": float(np.mean(abs_diffs / np.maximum(origs, 1e-15))),
        "s_intra_m": s_intra,
        "cv": cv,
        "bias_m": bias,
        "sd_diff_m": sd_diff,
        "bland_altman_lower_m": ba_lower,
        "bland_altman_upper_m": ba_upper,
    }


def compare_automatic_and_manual(records: Sequence[MeasurementRecord]) -> list[dict[str, Any]]:
    """Compares automatic candidates with reviewed manual references."""

    def get_source_str(r: MeasurementRecord) -> str:
        return r.source.value if hasattr(r.source, "value") else str(r.source)

    def get_status_str(r: MeasurementRecord) -> str:
        return r.status.value if hasattr(r.status, "value") else str(r.status)

    auto_recs = [r for r in records if "AUTO" in get_source_str(r).upper()]
    manual_recs = [r for r in records if get_source_str(r) == "MANUAL" and get_status_str(r) in {"ACCEPTED", "MANUALLY_EDITED"}]

    results: list[dict[str, Any]] = []
    if not auto_recs or not manual_recs:
        return results

    for a_rec in auto_recs:
        a_val = a_rec.primary_value
        if a_val is None:
            continue
        a_center = a_rec.center

        # Find closest manual reference
        best_m: tuple[float, MeasurementRecord] | None = None
        for m_rec in manual_recs:
            m_val = m_rec.primary_value
            if m_val is None:
                continue
            m_center = m_rec.center
            d_px = math.hypot(a_center[0] - m_center[0], a_center[1] - m_center[1])
            if best_m is None or d_px < best_m[0]:
                best_m = (d_px, m_rec)

        if best_m and best_m[0] <= 150.0:  # Proximity threshold
            m_rec = best_m[1]
            m_val = float(m_rec.primary_value)
            abs_diff = abs(a_val - m_val)
            rel_diff = abs_diff / max(m_val, 1e-15)

            results.append({
                "auto_id": a_rec.measurement_id,
                "manual_id": m_rec.measurement_id,
                "auto_value_m": a_val,
                "manual_value_m": m_val,
                "reference_label": "Referencia manual revisada",
                "absolute_difference_m": abs_diff,
                "relative_difference": rel_diff,
                "center_distance_px": best_m[0],
                "automatic_method": get_source_str(a_rec),
                "automatic_flags": list(a_rec.quality_flags),
                "manual_status": get_status_str(m_rec),
            })

    return results
