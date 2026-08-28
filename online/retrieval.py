"""Frame-level multimodal retrieval over stable keyframe_uid identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from shared.schemas.online import ArtifactAvailability, FrameEvidence, UnifiedQueryPlan

from .artifacts import ArtifactRegistry
from .config import OnlineConfig
from .encoders import TextEncoderRegistry
from .fts import FtsSearcher
from .fusion import combine_query_scores, fuse_modalities, fuse_visual_channels, minmax_scores


@dataclass(slots=True)
class RetrievalResult:
    evidence: list[FrameEvidence]
    per_query_scores: dict[str, dict[int, float]]
    candidate_uids: set[int]
    warnings: list[str] = field(default_factory=list)
    degraded_to_clip: bool = False


@dataclass(slots=True)
class _VisualContext:
    mode: str
    query_vectors: dict[str, np.ndarray]
    reference_scores: dict[str, list[np.ndarray]]
    direct_hits: dict[str, list[set[int]]]


def visual_query_texts(plan: UnifiedQueryPlan) -> list[str]:
    """Return independent global, scene and constraint queries in stable order."""

    result: list[str] = []
    for value in (
        plan.caption_en,
        *plan.retrieval_queries,
        *plan.scenes,
        *plan.must_have,
        *plan.should_have,
    ):
        text = value.strip()
        if text and text not in result:
            result.append(text)
    return result


class FrameRetriever:
    def __init__(
        self,
        registry: ArtifactRegistry,
        encoders: TextEncoderRegistry,
        config: OnlineConfig,
    ) -> None:
        self.registry = registry
        self.encoders = encoders
        self.config = config

    def search(self, plan: UnifiedQueryPlan) -> RetrievalResult:
        texts = visual_query_texts(plan)
        warnings: list[str] = []
        candidate_uids: set[int] = set()
        context = self._initial_visual_search(texts, candidate_uids, warnings)

        ocr_scores: dict[int, float] = {}
        asr_scores: dict[int, float] = {}
        if plan.visible_text and self.registry.statuses["ocr"].availability == ArtifactAvailability.READY:
            ocr_scores = FtsSearcher(self.registry.layout.ocr, "ocr").search(plan.visible_text, limit=self.config.visual_top_k)
            candidate_uids.update(uid for uid in ocr_scores if uid in self.registry.catalog.by_uid)
        elif plan.visible_text:
            warnings.append(f"OCR {self.registry.statuses['ocr'].availability.value}; visual weights were renormalized")
        if plan.spoken_text and self.registry.statuses["asr"].availability == ArtifactAvailability.READY:
            asr_scores = FtsSearcher(self.registry.layout.asr, "asr").search(plan.spoken_text, limit=self.config.visual_top_k)
            candidate_uids.update(uid for uid in asr_scores if uid in self.registry.catalog.by_uid)
        elif plan.spoken_text:
            warnings.append(f"ASR {self.registry.statuses['asr'].availability.value}; visual weights were renormalized")

        all_scored = set(candidate_uids)
        for uid in list(candidate_uids):
            all_scored.update(frame.keyframe_uid for frame in self.registry.catalog.neighbors(uid, self.config.neighbor_radius))
        per_query_visual, raw_channels, consensus = self._score_uids(context, texts, all_scored)
        combined_visual = combine_query_scores(per_query_visual, consensus_bonus=self.config.query_consensus_bonus)
        fused_all = fuse_modalities(
            combined_visual,
            ocr_scores=ocr_scores,
            asr_scores=asr_scores,
            modality_weights=plan.modality_weights,
        )

        reranked_raw: dict[int, float] = {}
        neighbor_support: dict[int, float] = {}
        for uid in candidate_uids:
            neighbors = self.registry.catalog.neighbors(uid, self.config.neighbor_radius)
            scores = sorted((fused_all.get(item.keyframe_uid, 0.0) for item in neighbors), reverse=True)[:2]
            support = float(np.mean(scores)) if scores else 0.0
            neighbor_support[uid] = support
            reranked_raw[uid] = fused_all.get(uid, 0.0) + self.config.neighbor_bonus * support
        final_scores = minmax_scores(reranked_raw)

        evidence: list[FrameEvidence] = []
        for uid in sorted(candidate_uids, key=lambda value: final_scores.get(value, 0.0), reverse=True):
            frame = self.registry.catalog.by_uid.get(uid)
            if frame is None:
                continue
            query_index = max(
                range(len(texts)),
                key=lambda index: per_query_visual[index].get(uid, 0.0),
            )
            evidence.append(
                FrameEvidence(
                    keyframe_uid=uid,
                    video_id=frame.video_id,
                    frame_id=frame.frame_id,
                    pts_time=frame.pts_time,
                    shot_id=frame.shot_id,
                    query_part=texts[query_index],
                    score_clip=raw_channels["clip"].get(uid),
                    score_siglip=raw_channels["siglip"].get(uid),
                    score_eva=raw_channels["eva_clip"].get(uid),
                    score_visual=combined_visual.get(uid, 0.0),
                    score_ocr=ocr_scores.get(uid),
                    score_asr=asr_scores.get(uid),
                    score_fused=fused_all.get(uid, 0.0),
                    neighbor_support=neighbor_support.get(uid, 0.0),
                    final_score=final_scores.get(uid, 0.0),
                    model_consensus=consensus.get(uid, 0.0),
                )
            )
        return RetrievalResult(
            evidence=evidence,
            per_query_scores={text: scores for text, scores in zip(texts, per_query_visual)},
            candidate_uids=candidate_uids,
            warnings=warnings,
            degraded_to_clip=context.mode == "clip",
        )

    def _initial_visual_search(
        self,
        texts: list[str],
        candidate_uids: set[int],
        warnings: list[str],
    ) -> _VisualContext:
        try:
            sig_vectors = self.encoders.encode("siglip", texts)
            eva_vectors = self.encoders.encode("eva_clip", texts)
            references = {"siglip": [], "eva_clip": []}
            direct = {"siglip": [], "eva_clip": []}
            for sig_vector, eva_vector in zip(sig_vectors, eva_vectors):
                sig_uids, sig_scores = self.registry.visual["siglip"].search(sig_vector, self.config.visual_top_k)
                eva_uids, eva_scores = self.registry.visual["eva_clip"].search(eva_vector, self.config.visual_top_k)
                candidate_uids.update(int(uid) for uid in sig_uids)
                candidate_uids.update(int(uid) for uid in eva_uids)
                references["siglip"].append(sig_scores)
                references["eva_clip"].append(eva_scores)
                direct["siglip"].append(set(int(uid) for uid in sig_uids))
                direct["eva_clip"].append(set(int(uid) for uid in eva_uids))
            query_vectors = {"siglip": sig_vectors, "eva_clip": eva_vectors}
            mode = "primary"
        except Exception as error:
            warnings.append(f"SigLIP/EVA primary failed; CLIP degraded rollback: {type(error).__name__}: {error}")
            clip_vectors = self.encoders.encode("clip", texts)
            references = {"clip": []}
            direct = {"clip": []}
            for vector in clip_vectors:
                uids, scores = self.registry.visual["clip"].search(vector, self.config.visual_top_k)
                candidate_uids.update(int(uid) for uid in uids)
                references["clip"].append(scores)
                direct["clip"].append(set(int(uid) for uid in uids))
            query_vectors = {"clip": clip_vectors}
            mode = "clip"

        if mode == "primary":
            try:
                clip_vectors = self.encoders.encode("clip", texts)
                query_vectors["clip"] = clip_vectors
                references["clip"] = []
                direct["clip"] = []
                for vector in clip_vectors:
                    uids, scores = self.registry.visual["clip"].search(vector, self.config.visual_top_k)
                    references["clip"].append(scores)
                    direct["clip"].append(set(int(uid) for uid in uids))
            except Exception as error:
                warnings.append(f"CLIP comparison unavailable: {type(error).__name__}: {error}")
        return _VisualContext(
            mode=mode,
            query_vectors=query_vectors,
            reference_scores=references,
            direct_hits=direct,
        )

    def _score_uids(
        self,
        context: _VisualContext,
        texts: list[str],
        uids: set[int],
    ) -> tuple[list[dict[int, float]], dict[str, dict[int, float]], dict[int, float]]:
        ordered_uids = sorted(uids)
        raw: dict[str, dict[int, float]] = {"clip": {}, "siglip": {}, "eva_clip": {}}
        per_query: list[dict[int, float]] = []
        consensus: dict[int, float] = {uid: 0.0 for uid in ordered_uids}
        if context.mode == "clip":
            for query_index, vector in enumerate(context.query_vectors["clip"]):
                scores = self.registry.visual["clip"].scores_for(vector, ordered_uids)
                normalized = minmax_scores(scores)
                per_query.append(normalized)
                for uid, value in scores.items():
                    raw["clip"][uid] = max(raw["clip"].get(uid, -float("inf")), value)
                for uid in context.direct_hits["clip"][query_index]:
                    consensus[uid] = max(consensus.get(uid, 0.0), 0.5)
            return per_query, raw, consensus

        for query_index in range(len(texts)):
            sig_scores = self.registry.visual["siglip"].scores_for(context.query_vectors["siglip"][query_index], ordered_uids)
            eva_scores = self.registry.visual["eva_clip"].scores_for(context.query_vectors["eva_clip"][query_index], ordered_uids)
            fused = fuse_visual_channels(
                sig_scores,
                eva_scores,
                siglip_reference_scores=context.reference_scores["siglip"][query_index],
                eva_reference_scores=context.reference_scores["eva_clip"][query_index],
                eta=self.config.srrf_eta,
                beta=self.config.srrf_beta,
            )
            per_query.append(fused.scores)
            for uid in ordered_uids:
                raw["siglip"][uid] = max(raw["siglip"].get(uid, -float("inf")), sig_scores[uid])
                raw["eva_clip"][uid] = max(raw["eva_clip"].get(uid, -float("inf")), eva_scores[uid])
                direct_both = uid in context.direct_hits["siglip"][query_index] and uid in context.direct_hits["eva_clip"][query_index]
                consensus[uid] = max(consensus[uid], 1.0 if direct_both else 0.5 if uid in context.direct_hits["siglip"][query_index] or uid in context.direct_hits["eva_clip"][query_index] else 0.0)
            if "clip" in context.query_vectors:
                clip_scores = self.registry.visual["clip"].scores_for(context.query_vectors["clip"][query_index], ordered_uids)
                for uid, value in clip_scores.items():
                    raw["clip"][uid] = max(raw["clip"].get(uid, -float("inf")), value)
        return per_query, raw, consensus
