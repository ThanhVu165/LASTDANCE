"""Bounded real-data smoke test for the dedicated multimodal reranker."""
from __future__ import annotations

import argparse
import json
import time

from app.rerank.model_reranker import (
    release_model_reranker,
    rerank_kis_with_generative_model,
    rerank_kis_with_model,
)
from app.services.clip_search import search_text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="a red car driving on a road")
    parser.add_argument("--candidates", type=int, default=12)
    args = parser.parse_args()

    import torch

    candidates = search_text(args.query, top_k=max(1, args.candidates))
    for row in candidates:
        row["query_coverage"] = 1.0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    reranked, report = rerank_kis_with_model(args.query, candidates)
    if not report.available:
        reranked, report = rerank_kis_with_generative_model(args.query, candidates)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_reserved() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    payload = {
        "report": report.__dict__,
        "elapsed_seconds": round(elapsed, 3),
        "peak_reserved_gib": round(peak, 3),
        "top_results": [
            {
                "video_id": row["video_id"],
                "local_idx": row["local_idx"],
                "score": round(float(row["score"]), 5),
                "model_score": row.get("model_relevance_score"),
            }
            for row in sorted(reranked, key=lambda item: float(item["score"]), reverse=True)[:5]
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    release_model_reranker()
    return 0 if report.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
