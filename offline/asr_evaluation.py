"""Small optional WER helpers for ASR Dev Gate comparisons."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_wer_text(value: str) -> list[str]:
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in value
    )
    return normalized.split()


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(
    references: Iterable[str], hypotheses: Iterable[str]
) -> dict[str, int | float]:
    reference_rows = list(references)
    hypothesis_rows = list(hypotheses)
    if not reference_rows or len(reference_rows) != len(hypothesis_rows):
        raise ValueError("WER requires equally sized, non-empty reference/hypothesis sets")
    errors = 0
    reference_words = 0
    for reference, hypothesis in zip(reference_rows, hypothesis_rows, strict=True):
        ref_tokens = normalize_wer_text(reference)
        hyp_tokens = normalize_wer_text(hypothesis)
        if not ref_tokens:
            raise ValueError("WER reference must contain at least one normalized word")
        errors += _edit_distance(ref_tokens, hyp_tokens)
        reference_words += len(ref_tokens)
    return {
        "sample_count": len(reference_rows),
        "word_errors": errors,
        "reference_words": reference_words,
        "wer": errors / reference_words,
    }
