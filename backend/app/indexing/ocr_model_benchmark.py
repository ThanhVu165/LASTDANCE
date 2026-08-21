"""Run the accent-sensitive OCR diagnostic set on real project keyframes.

The frozen PP-OCRv6 lines are the outputs observed before migration.  Keeping the
baseline here makes the comparison reproducible even after the live cache has
been rebuilt with EasyOCR.  This three-image set is a regression smoke test, not
a substitute for a larger labeled evaluation set.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from app.config import KEYFRAMES_DIR, OCR_DEVICE
from app.indexing.ocr_index import OcrItem, _create_pipeline, _predict_batch


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    cache_key: str
    image_path: Path
    expected: tuple[str, ...]
    ppocrv6_baseline: tuple[str, ...]


DEFAULT_CASES = (
    BenchmarkCase(
        "news-headline-subsidence",
        "L21_V001:6",
        KEYFRAMES_DIR / "L21_V001" / "006.jpg",
        ("tin chính", "tình trạng sụt lún ở đbscl đang diễn ra rất nhanh"),
        (
            "06:30:28",
            "tinchính",
            "giay",
            "tinh trang sut lún dbscl dang din ra rãt nhanh",
        ),
    ),
    BenchmarkCase(
        "news-headline-heart-transport",
        "L21_V001:7",
        KEYFRAMES_DIR / "L21_V001" / "007.jpg",
        (
            "tin chính",
            "trái tim được vận chuyển cấp tốc về huế",
            "ghép cho bệnh nhân",
        ),
        (
            "htv",
            "hd",
            "06:30:34",
            "tinchính",
            "giay",
            "trái tim duc vân chuyên cãp tõc vê huê",
            "ghép cho bênh nhân",
        ),
    ),
    BenchmarkCase(
        "warning-sign-and-headline",
        "L21_V001:17",
        KEYFRAMES_DIR / "L21_V001" / "017.jpg",
        (
            "cảnh báo",
            "sạt lở nguy hiểm",
            "tạm dừng lưu thông",
            "đối với xe 3 bánh trở lên",
            "người dân đi lại chú ý quan sát",
            "sụt lún ở đbscl đang diễn ra rất nhanh",
        ),
        (
            "htv",
            "hd",
            "06:31:28",
            "cánh báo",
            "sat l nguy hiêm",
            "tąm düng luu thông",
            "dói vöi xe 3 bánh trô lén",
            "nguèi dán di lai chú ý quan sát",
            "sut lún dbscl dang dižn ra rãt nhanh",
            "giây",
            "min bc mua lón dài ngày",
        ),
    ),
)


def _normalize_for_quality(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[^0-9a-zà-ỹđ]+", " ", text)
    return " ".join(text.split())


def _candidate_spans(lines: Sequence[str]) -> list[str]:
    normalized = [_normalize_for_quality(line) for line in lines]
    normalized = [line for line in normalized if line]
    spans = list(normalized)
    for width in (2, 3):
        spans.extend(
            " ".join(normalized[start : start + width])
            for start in range(0, len(normalized) - width + 1)
        )
    if normalized:
        spans.append(" ".join(normalized))
    return list(dict.fromkeys(spans))


def _phrase_similarity(expected: str, lines: Sequence[str]) -> float:
    target = _normalize_for_quality(expected)
    candidates = _candidate_spans(lines)
    if not target:
        return 1.0
    if not candidates:
        return 0.0
    return max(
        difflib.SequenceMatcher(None, target, candidate, autojunk=False).ratio()
        for candidate in candidates
    )


def _score(expected: Sequence[str], lines: Sequence[str]) -> dict[str, Any]:
    phrase_scores = [
        {"expected": phrase, "similarity": round(_phrase_similarity(phrase, lines), 4)}
        for phrase in expected
    ]
    mean = sum(row["similarity"] for row in phrase_scores) / max(len(phrase_scores), 1)
    return {"mean_similarity": round(mean, 4), "phrases": phrase_scores}


def _model_report(
    cases: Sequence[BenchmarkCase], outputs: Sequence[Sequence[str]]
) -> dict[str, Any]:
    report: dict[str, Any] = {
        case.name: {"lines": list(lines), "quality": _score(case.expected, lines)}
        for case, lines in zip(cases, outputs)
    }
    scores = [row["quality"]["mean_similarity"] for row in report.values()]
    report["_mean_similarity"] = round(sum(scores) / max(len(scores), 1), 4)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=OCR_DEVICE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    missing = [str(case.image_path) for case in DEFAULT_CASES if not case.image_path.exists()]
    if missing:
        raise FileNotFoundError("Missing benchmark images: " + ", ".join(missing))

    pipeline = _create_pipeline(args.device)
    outcomes = _predict_batch(
        pipeline,
        [OcrItem(case.cache_key, str(case.image_path)) for case in DEFAULT_CASES],
    )
    errors = [f"{outcome.key}: {outcome.error}" for outcome in outcomes if outcome.error]
    if errors:
        raise RuntimeError("OCR benchmark failed: " + "; ".join(errors))

    easyocr_outputs = [[line.text for line in outcome.lines] for outcome in outcomes]
    report = {
        "paddle_ppocrv6_medium_observed_baseline": _model_report(
            DEFAULT_CASES, [case.ppocrv6_baseline for case in DEFAULT_CASES]
        ),
        "easyocr_craft_latin_g2": _model_report(DEFAULT_CASES, easyocr_outputs),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
