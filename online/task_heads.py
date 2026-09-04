"""KIS, QA and TRAKE task-specific candidate generation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar

from shared.schemas.online import (
    FrameEvidence,
    AnswerResult,
    KISCandidate,
    QACandidate,
    TrakeCandidate,
    UnifiedQueryPlan,
    VideoHypothesis,
)

from .config import OnlineConfig
from .retrieval import RetrievalResult


class VideoAnswerer(Protocol):
    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> AnswerResult: ...


class UnavailableAnswerer:
    def answer(
        self,
        *,
        video_id: str,
        frames: Sequence[FrameEvidence],
        question: str,
    ) -> AnswerResult:
        return AnswerResult(warnings=["VQA unavailable; answer requires operator review"])


T = TypeVar("T")


def _adaptive_portfolio(
    grouped: list[tuple[str, float, list[T]]],
    *,
    max_results: int,
    max_per_video: int,
    primary_min: int,
) -> list[T]:
    """Confidence-adaptive weighted round-robin with the explicit Top-5 seed."""

    if not grouped:
        return []
    cursors = {video: 0 for video, _, _ in grouped}
    counts = {video: 0 for video, _, _ in grouped}
    selected: list[T] = []
    by_video = {video: values for video, _, values in grouped}
    scores = {video: max(float(score), 1e-6) for video, score, _ in grouped}

    def take(video: str) -> None:
        values = by_video.get(video, [])
        cursor = cursors.get(video, 0)
        if cursor < len(values) and counts[video] < max_per_video and len(selected) < max_results:
            selected.append(values[cursor])
            cursors[video] = cursor + 1
            counts[video] += 1

    videos = [video for video, _, _ in grouped]
    seed = [videos[0], videos[0]]
    if len(videos) > 1:
        seed.append(videos[1])
    seed.append(videos[0])
    if len(videos) > 2:
        seed.append(videos[2])
    for video in seed:
        take(video)

    available = {
        video: min(len(by_video[video]), max_per_video)
        for video in videos
    }
    total = min(max_results, sum(available.values()))
    targets = dict(counts)

    def reserve(video: str, minimum: int) -> None:
        room = total - sum(targets.values())
        if room <= 0:
            return
        desired = min(minimum, available[video])
        targets[video] += min(max(0, desired - targets[video]), room)

    for video in videos[:5]:
        reserve(video, 2)
    for video in videos[:12]:
        reserve(video, 1)
    reserve(videos[0], min(primary_min, max_per_video))

    rank = {video: index for index, video in enumerate(videos)}

    def next_candidate_strength(video: str) -> float:
        candidate = by_video[video][cursors[video]]
        evidence = getattr(candidate, "evidence", None)
        if isinstance(evidence, FrameEvidence):
            return float(evidence.final_score)
        if isinstance(evidence, list) and evidence:
            return sum(float(item.final_score) for item in evidence) / len(evidence)
        return float(getattr(candidate, "score", 0.0))

    while sum(targets.values()) < total:
        eligible = [video for video in videos if targets[video] < available[video]]
        if not eligible:
            break
        video = max(
            eligible,
            key=lambda item: (scores[item] / (targets[item] + 1.0), -rank[item]),
        )
        targets[video] += 1

    while len(selected) < total:
        eligible = [video for video in videos if counts[video] < targets[video]]
        if not eligible:
            break
        video = min(
            eligible,
            key=lambda item: (
                counts[item] / targets[item],
                -next_candidate_strength(item),
                rank[item],
            ),
        )
        take(video)
    return selected


def build_kis_candidates(
    hypotheses: list[VideoHypothesis],
    *,
    max_results: int,
    config: OnlineConfig,
) -> list[KISCandidate]:
    grouped: list[tuple[str, float, list[KISCandidate]]] = []
    for video_rank, video in enumerate(hypotheses):
        ranked = [
            KISCandidate(
                video_id=video.video_id,
                frame_id=frame.frame_id,
                score=0.55 * video.video_score + 0.45 * frame.final_score,
                evidence=frame,
            )
            for frame in video.best_frames
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        candidates = ranked
        if video_rank == 0:
            candidates = []
            used_shots: set[str] = set()
            for candidate in ranked:
                if candidate.evidence.shot_id in used_shots:
                    continue
                candidates.append(candidate)
                used_shots.add(candidate.evidence.shot_id)
                if len(candidates) == 3:
                    break
            selected_uids = {candidate.evidence.keyframe_uid for candidate in candidates}
            candidates.extend(
                candidate
                for candidate in ranked
                if candidate.evidence.keyframe_uid not in selected_uids
            )
        grouped.append((video.video_id, video.video_score, candidates))
    return _adaptive_portfolio(
        grouped,
        max_results=max_results,
        max_per_video=config.portfolio_max_per_video,
        primary_min=config.portfolio_primary_min,
    )


def build_qa_candidates(
    hypotheses: list[VideoHypothesis],
    plan: UnifiedQueryPlan,
    *,
    answerer: VideoAnswerer,
    max_results: int,
    config: OnlineConfig,
) -> tuple[list[QACandidate], list[str]]:
    grouped: list[tuple[str, float, list[QACandidate]]] = []
    warnings: list[str] = []
    question = plan.question or plan.raw_query
    for rank, video in enumerate(hypotheses):
        if not video.best_frames or rank >= config.qa_answer_video_top_k:
            continue
        # Providers select compact context; submission rows use only returned evidence.
        frames = video.best_frames[: config.portfolio_max_per_video]
        locator = min(1.0, max(0.0, 0.6 * video.video_score + 0.4 * frames[0].final_score))
        # Locator confidence controls auto-accept only. It must never prevent an
        # answer attempt on the configured Top-N candidate videos.
        result = answerer.answer(
            video_id=video.video_id,
            frames=frames,
            question=question,
        )
        result = AnswerResult.model_validate(result)
        warnings.extend(result.warnings)
        answer, answer_confidence = result.answer, result.confidence
        if not answer.strip() or answer.casefold() == "uncertain" or answer_confidence <= 0.0:
            warnings.append(
                f"QA answer unavailable for {video.video_id}; evidence remains available for manual review"
            )
            continue
        requires_review = (
            result.requires_review or locator <= config.qa_similarity_threshold
            or answer_confidence < config.qa_similarity_threshold
        )
        if any(f.video_id != video.video_id for f in result.evidence):
            raise ValueError("answerer returned cross-video evidence")
        candidates = [
            QACandidate(
                video_id=video.video_id,
                frame_id=frame.frame_id,
                answer=answer,
                score=0.55 * video.video_score + 0.45 * frame.final_score,
                confidence=min(locator, answer_confidence),
                requires_review=requires_review,
                evidence=frame,
            )
            for frame in result.evidence
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        grouped.append((video.video_id, video.video_score, candidates))
    portfolio = _adaptive_portfolio(
        grouped,
        max_results=max_results,
        max_per_video=config.portfolio_max_per_video,
        primary_min=config.portfolio_primary_min,
    )
    return portfolio, list(dict.fromkeys(warnings))


@dataclass(slots=True)
class _Beam:
    frames: list[FrameEvidence]
    score: float


def build_trake_candidates(
    hypotheses: list[VideoHypothesis],
    moment_results: list[RetrievalResult],
    *,
    max_results: int,
    config: OnlineConfig,
) -> list[TrakeCandidate]:
    moment_by_video: list[dict[str, list[FrameEvidence]]] = []
    for result in moment_results:
        grouped: dict[str, dict[int, FrameEvidence]] = defaultdict(dict)
        for frame in result.evidence:
            current = grouped[frame.video_id].get(frame.frame_id)
            if current is None or frame.final_score > current.final_score:
                grouped[frame.video_id][frame.frame_id] = frame
        moment_by_video.append(
            {
                video: sorted(frames.values(), key=lambda item: (-item.final_score, item.pts_time, item.frame_id))
                for video, frames in grouped.items()
            }
        )

    candidates_by_video: dict[str, list[TrakeCandidate]] = {}
    for hypothesis in hypotheses:
        video_id = hypothesis.video_id
        per_moment = [grouped.get(video_id, []) for grouped in moment_by_video]
        if any(not values for values in per_moment):
            continue
        # Prune unreachable suffixes before beam truncation, not valid same-shot frames.
        for index in range(len(per_moment) - 2, -1, -1):
            per_moment[index] = [frame for frame in per_moment[index]
                                 if any(nxt.pts_time > frame.pts_time and nxt.frame_id != frame.frame_id
                                        for nxt in per_moment[index + 1])]
        beams = [_Beam([frame], frame.final_score) for frame in per_moment[0]][: config.trake_beam_width]
        for values in per_moment[1:]:
            expanded: list[_Beam] = []
            for beam in beams:
                last = beam.frames[-1]
                used = {frame.frame_id for frame in beam.frames}
                for frame in values:
                    if frame.pts_time <= last.pts_time or frame.frame_id in used:
                        continue
                    delta = frame.pts_time - last.pts_time
                    score = beam.score + frame.final_score * math.exp(-config.trake_decay * delta)
                    expanded.append(_Beam(beam.frames + [frame], score))
            expanded.sort(key=lambda item: item.score, reverse=True)
            beams = expanded[: config.trake_beam_width]
            if not beams:
                break
        complete = [beam for beam in beams if len(beam.frames) == len(moment_results)]
        rows: list[TrakeCandidate] = []
        seen: set[tuple[int, ...]] = set()
        for beam in complete:
            frame_ids = tuple(frame.frame_id for frame in beam.frames)
            if frame_ids in seen:
                continue
            seen.add(frame_ids)
            rows.append(
                TrakeCandidate(
                    video_id=video_id,
                    frame_ids=list(frame_ids),
                    pts_times=[frame.pts_time for frame in beam.frames],
                    score=0.4 * hypothesis.video_score + 0.6 * (beam.score / len(beam.frames)),
                    evidence=beam.frames,
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        candidates_by_video[video_id] = rows

    grouped = [
        (hypothesis.video_id, hypothesis.video_score, candidates_by_video.get(hypothesis.video_id, []))
        for hypothesis in hypotheses
        if candidates_by_video.get(hypothesis.video_id)
    ]
    return _adaptive_portfolio(
        grouped,
        max_results=max_results,
        max_per_video=config.portfolio_max_per_video,
        primary_min=config.portfolio_primary_min,
    )
