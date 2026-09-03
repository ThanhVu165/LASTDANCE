"""Route CRAFT-positive shots under the committed Gemini frame/VND caps."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from offline.ocr_escalation import (
    CraftFrameFeatures,
    OcrEscalationPolicy,
    select_gemini_escalations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-jsonl", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/ocr_escalation_policy.json"),
    )
    parser.add_argument("--estimated-prompt-tokens-per-request", type=float, required=True)
    parser.add_argument("--estimated-output-tokens-per-request", type=float, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _load_features(path: Path) -> list[CraftFrameFeatures]:
    rows: list[CraftFrameFeatures] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(CraftFrameFeatures.model_validate_json(line))
            except Exception as error:
                raise RuntimeError(
                    f"invalid feature row at {path}:{line_number}"
                ) from error
    return rows


def main() -> int:
    args = build_parser().parse_args()
    policy = OcrEscalationPolicy.model_validate_json(
        args.policy.read_text(encoding="utf-8")
    )
    selection = select_gemini_escalations(
        _load_features(args.features_jsonl),
        policy=policy,
        estimated_prompt_tokens_per_request=args.estimated_prompt_tokens_per_request,
        estimated_output_tokens_per_request=args.estimated_output_tokens_per_request,
    )

    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(selection.model_dump(mode="json"), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        "PASS: "
        f"text_positive={selection.text_positive_records} "
        f"paid_frames={selection.selected_paid_frames} "
        f"paid_requests={selection.selected_paid_requests} "
        f"easyocr_overflow={selection.overflow_easyocr_frames} "
        f"estimated_vnd={selection.estimated_cost_vnd_with_reserve:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
