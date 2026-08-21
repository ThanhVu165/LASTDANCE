"""Bounded round-1 smoke checks using representative prior-year query shapes.

The reference workbook does not include ground-truth video/frame labels, so this
tool reports structural correctness, latency and the returned portfolio. It never
claims retrieval accuracy. Use a manually verified target separately when one is
available.
"""
from __future__ import annotations

import argparse
import json
import time

from app.pipelines.kis_pipeline import run_kis_query
from app.services.query_processing import parse_semantic_query
from app.services.query_planner import plan_visual_query


SMOKE_QUERIES = {
    "tkis-charity": (
        "Đoạn video đưa hình ảnh nhóm người làm thiện nguyện. Sản phẩm từ thiện "
        "là các hộp cơm. Nhóm người thiện nguyện mặc áo màu xanh, đeo bao tay "
        "nilon và cùng nhau làm hàng trăm suất cơm. Một trong các phân cảnh sau "
        "đó là nhóm thiện nguyện đang phát cơm chay 0 đồng trước cổng chùa "
        "Chánh Thiên, Bà Rịa."
    ),
    "tkis-festival": (
        "Đoạn video mô tả một lễ hội ở Nhật Bản, mọi người nhảy múa trên một "
        "chiếc thuyền và cầm quạt giấy in hình quốc kỳ Nhật Bản. Trên bờ có "
        "nhiều người đứng xem; hai người mặc áo đỏ đang khiêng một con cá được "
        "cột vào cây tre."
    ),
    "tkis-ocr": (
        "Tìm clip tại Úc có thùng thu gom pin để tái chế với nắp màu xanh dương, "
        "viền xanh lá cây, đặt sát tường màu gỗ. Cuối clip có cửa hàng với dòng "
        "chữ coles màu trắng trên nền đỏ."
    ),
    "vkis-short": "Múa lân",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", choices=tuple(SMOKE_QUERIES), default="tkis-charity")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--parser-only", action="store_true")
    parser.add_argument("--exact", action="store_true")
    args = parser.parse_args()

    text = SMOKE_QUERIES[args.query]
    semantic = parse_semantic_query(text)
    plan = plan_visual_query(text)
    report: dict = {
        "query": args.query,
        "characters": len(text),
        "scene_count": len(semantic.scenes),
        "temporal_edges": semantic.temporal_edges,
        "expansion_count": len(semantic.expansions),
        "planner_source": plan.source,
        "planned_scene_count": len(plan.scenes),
        "planned_prompt_count": len(plan.retrieval_prompts),
        "repair_query_count": len(plan.repair_queries),
    }
    if not args.parser_only:
        started = time.perf_counter()
        rows = run_kis_query(text, top_k=args.top_k, refine_exact=args.exact)
        report.update(
            {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "result_count": len(rows),
                "model_verified_count": sum(
                    bool(row.get("model_verified")) for row in rows
                ),
                "model_scored_count": sum(
                    row.get("model_relevance_score") is not None for row in rows
                ),
                "distinct_videos": len({row["video_id"] for row in rows}),
                "top_results": rows[:5],
            }
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
