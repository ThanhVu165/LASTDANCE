"""Two-stage score fusion required by the online baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


def minmax_scores(scores: Mapping[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    keys = list(scores)
    values = np.asarray([scores[key] for key in keys], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("score channel contains NaN or Inf")
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-12:
        return {key: 1.0 for key in keys}
    normalized = (values - low) / (high - low)
    return {key: float(value) for key, value in zip(keys, normalized)}


def _smooth_ranks(
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    *,
    beta: float,
    block_size: int = 512,
) -> np.ndarray:
    result = np.empty(len(candidate_scores), dtype=np.float64)
    references = np.asarray(reference_scores, dtype=np.float64)
    for start in range(0, len(candidate_scores), block_size):
        stop = min(len(candidate_scores), start + block_size)
        differences = references[None, :] - candidate_scores[start:stop, None]
        logits = np.clip(beta * differences, -60.0, 60.0)
        result[start:stop] = 0.5 + (1.0 / (1.0 + np.exp(-logits))).sum(axis=1)
    return result


@dataclass(frozen=True, slots=True)
class VisualFusionResult:
    scores: dict[int, float]
    consensus: dict[int, float]


def fuse_visual_channels(
    siglip_scores: Mapping[int, float],
    eva_scores: Mapping[int, float],
    *,
    siglip_reference_scores: Sequence[float],
    eva_reference_scores: Sequence[float],
    eta: float = 60.0,
    beta: float = 40.0,
) -> VisualFusionResult:
    """Fuse SigLIP and EVA only; CLIP must never enter this function."""

    union = sorted(set(siglip_scores) | set(eva_scores))
    if not union:
        return VisualFusionResult(scores={}, consensus={})
    if not siglip_scores or not eva_scores:
        raise RuntimeError("SRRF requires both SigLIP and EVA score channels")
    sig_values = np.asarray([siglip_scores[uid] for uid in union], dtype=np.float64)
    eva_values = np.asarray([eva_scores[uid] for uid in union], dtype=np.float64)
    sig_ranks = _smooth_ranks(sig_values, np.asarray(siglip_reference_scores), beta=beta)
    eva_ranks = _smooth_ranks(eva_values, np.asarray(eva_reference_scores), beta=beta)
    fused = 1.0 / (eta + sig_ranks) + 1.0 / (eta + eva_ranks)
    normalized = minmax_scores({uid: float(score) for uid, score in zip(union, fused)})
    direct_sig = set(siglip_scores)
    direct_eva = set(eva_scores)
    consensus = {
        uid: 1.0 if uid in direct_sig and uid in direct_eva else 0.5
        for uid in union
    }
    return VisualFusionResult(scores=normalized, consensus=consensus)


def combine_query_scores(
    per_query: Sequence[Mapping[int, float]],
    *,
    consensus_bonus: float = 0.1,
) -> dict[int, float]:
    if not per_query:
        return {}
    union = set().union(*(scores.keys() for scores in per_query))
    combined: dict[int, float] = {}
    for uid in union:
        values = sorted((float(scores.get(uid, 0.0)) for scores in per_query), reverse=True)
        best = values[0]
        support = values[1] if len(values) > 1 else 0.0
        combined[int(uid)] = best + consensus_bonus * support
    return minmax_scores(combined)


def fuse_modalities(
    visual_scores: Mapping[int, float],
    *,
    ocr_scores: Mapping[int, float] | None = None,
    asr_scores: Mapping[int, float] | None = None,
    modality_weights: Mapping[str, float] | None = None,
) -> dict[int, float]:
    """Late-fuse the single visual score with optional text channels."""

    channels: dict[str, dict[int, float]] = {"visual": minmax_scores(visual_scores)}
    if ocr_scores:
        channels["ocr"] = minmax_scores(ocr_scores)
    if asr_scores:
        channels["asr"] = minmax_scores(asr_scores)
    requested = dict(modality_weights or {"visual": 1.0})
    weights = {name: max(0.0, float(requested.get(name, 0.0))) for name in channels}
    total = sum(weights.values())
    if total <= 0:
        weights = {name: (1.0 if name == "visual" else 0.0) for name in channels}
        total = 1.0
    weights = {name: weight / total for name, weight in weights.items()}
    union = set().union(*(scores.keys() for scores in channels.values()))
    fused = {
        int(uid): sum(weights[name] * scores.get(uid, 0.0) for name, scores in channels.items())
        for uid in union
    }
    return minmax_scores(fused)
