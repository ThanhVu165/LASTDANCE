"""Controlled Vietnamese-to-English visual query expansion.

The organizer's image index was produced by the original English CLIP ViT-B/32.
The multilingual text tower remains the primary Vietnamese path; this module adds
one faithful English rendering so the original CLIP text tower can recover details
that its distilled multilingual student misses.  The source query is never removed.
"""
from __future__ import annotations

import re
from functools import lru_cache

from app.config import QUERY_TRANSLATION_ENABLED, QUERY_TRANSLATION_MAX_NEW_TOKENS
from app.services.query_processing import is_vietnamese_text
from app.services.visual_qa import generate_text


_SCENE_LINE = re.compile(r"^\s*S\s*([0-9]+)\s*[:=.-]\s*(.+?)\s*$", re.IGNORECASE)


def _clean_translation_line(line: str) -> str:
    translation = " ".join(line.strip().split())
    return re.sub(
        r"^(?:(?:caption|english|translation|english description)\s*)?"
        r"(?:\d+\s*[.):=-]|[-*•])\s*",
        "",
        translation,
        flags=re.IGNORECASE,
    ).strip(" \"'“”.")


@lru_cache(maxsize=512)
def translate_visual_queries(text: str) -> tuple[str, ...]:
    if not QUERY_TRANSLATION_ENABLED or not is_vietnamese_text(text):
        return ()
    prompt = (
        "Render the Vietnamese video-search description below as exactly three "
        "concise English visual captions for CLIP retrieval. Caption 1 is a faithful "
        "natural translation. Caption 2 must translate compound nouns to their "
        "conventional English category names instead of literal hypernyms (for example "
        "'xe mui trần' -> 'convertible', 'xe cứu hỏa' -> 'fire truck', and 'sân bóng "
        "rổ' -> 'basketball court'). It uses common concrete visual caption vocabulary. "
        "Caption 3 is an alternative phrasing. Preserve every "
        "visible entity, count, color, action, attribute, spatial relation, scene "
        "and proper name. Do not answer a question, infer missing facts, explain, or "
        "add details. Output only three captions, one per line, without numbering or "
        "bullets.\n"
        f"Vietnamese: {text}\nEnglish captions:"
    )
    try:
        decoded = generate_text(
            prompt,
            max_new_tokens=QUERY_TRANSLATION_MAX_NEW_TOKENS,
        )
    except (RuntimeError, OSError, ValueError):
        return ()
    translations: list[str] = []
    for line in decoded.splitlines():
        translation = _clean_translation_line(line)
        # English captions may legitimately retain Vietnamese proper names such
        # as "Chánh Thiên" or "Bà Rịa". Accent-based language detection would
        # discard exactly the named-place evidence needed by organizer queries.
        if translation and translation.casefold() != text.casefold():
            translations.append(translation)
    return tuple(dict.fromkeys(translations[:3]))


@lru_cache(maxsize=256)
def translate_visual_scenes(scenes: tuple[str, ...]) -> tuple[str, ...]:
    """Translate all open-vocabulary evidence units in one deterministic call.

    Each output remains tied to its source scene. This lets the English CLIP tower
    retrieve late details without treating translations as extra independent
    evidence. A malformed/failed generation returns empty strings and the caller
    keeps the multilingual source path unchanged.
    """
    if (
        not QUERY_TRANSLATION_ENABLED
        or not scenes
        or not any(is_vietnamese_text(scene) for scene in scenes)
    ):
        return tuple("" for _ in scenes)
    numbered = "\n".join(
        f"S{index}={scene}" for index, scene in enumerate(scenes, 1)
    )
    prompt = (
        f"Translate the {len(scenes)} Vietnamese visual evidence units below into "
        "concise English video-retrieval captions. Preserve every visible entity, "
        "count, colour, action, attribute, spatial relation, scene order and proper "
        "name. Use conventional visual category names. Do not merge units, infer "
        "facts, answer questions or explain. Output exactly one line per input in "
        "the same order and format S1=..., S2=..., with no other text.\n"
        f"{numbered}"
    )
    try:
        decoded = generate_text(
            prompt,
            max_new_tokens=max(
                QUERY_TRANSLATION_MAX_NEW_TOKENS,
                36 * len(scenes),
            ),
        )
    except (RuntimeError, OSError, ValueError):
        return tuple("" for _ in scenes)

    translated: dict[int, str] = {}
    for line in decoded.splitlines():
        match = _SCENE_LINE.match(line)
        if not match:
            continue
        index = int(match.group(1)) - 1
        if not 0 <= index < len(scenes):
            continue
        value = _clean_translation_line(match.group(2))
        if value and value.casefold() != scenes[index].casefold():
            translated[index] = value
    return tuple(translated.get(index, "") for index in range(len(scenes)))


def with_english_visual_expansion(text: str, expansions: list[str]) -> list[str]:
    translations = translate_visual_queries(text)
    if not translations:
        return expansions
    return list(dict.fromkeys([*expansions, *translations]))
