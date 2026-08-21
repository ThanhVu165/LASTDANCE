"""Shared natural-language understanding for KIS, Q&A and TRAKE.

Retrieval is semantic-first: the complete query is encoded by a multilingual CLIP
text tower aligned with the organizer's existing CLIP-ViT-B/32 image features.
Object detections and OCR are optional evidence, never substitutes for understanding
the sentence.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


VIETNAMESE_DIACRITICS = frozenset(
    "ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩ"
    "òóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ"
)
VIETNAMESE_FUNCTION_WORDS = frozenset(
    "mot chiec nguoi dang trong tren duoi voi va cua co la khi sau truoc "
    "tim canh video khoanh khac bao nhieu mau gi nao".split()
)

# Only concepts that can actually appear in the OpenImages object cache belong
# here. Adjectives, actions and arbitrary query tokens stay in semantic CLIP.
OBJECT_LABEL_HINTS = {
    "nguoi dan ong": "man",
    "nguoi phu nu": "woman",
    "dan ong": "man",
    "phu nu": "woman",
    "be trai": "boy",
    "be gai": "girl",
    "tre em": "child",
    "nguoi": "person",
    "person": "person",
    "man": "man",
    "woman": "woman",
    "boy": "boy",
    "girl": "girl",
    "child": "child",
    "xe o to": "car",
    "xe hoi": "car",
    "o to": "car",
    "car": "car",
    "xe buyt": "bus",
    "bus": "bus",
    "xe tai": "truck",
    "truck": "truck",
    "xe may": "motorcycle",
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "xe dap": "bicycle",
    "bicycle": "bicycle",
    "cho": "dog",
    "dog": "dog",
    "meo": "cat",
    "cat": "cat",
    "may bay": "airplane",
    "airplane": "airplane",
    "tau hoa": "train",
    "train": "train",
    "thuyen": "boat",
    "boat": "boat",
    "cay": "tree",
    "tree": "tree",
    "toa nha": "building",
    "building": "building",
    "micro": "microphone",
    "microphone": "microphone",
}

OCR_INTENT_MARKERS = (
    "dong chu",
    "chu tren",
    "bien hieu",
    "bien bao ghi",
    "man hinh ghi",
    "tieu de",
    "logo",
    "text on",
    "sign says",
    "written on",
    "caption says",
)

LONG_QUERY_THRESHOLD = 220
MAX_VISUAL_CLAUSES = 8


@dataclass(frozen=True)
class SemanticQuery:
    original_text: str
    language: str
    retrieval_text: str
    scenes: list[str] = field(default_factory=list)
    temporal_ordered: bool = False
    temporal_edges: list[tuple[int, int]] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    object_keywords: list[str] = field(default_factory=list)
    ocr_keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QaTask:
    original_text: str
    retrieval_text: str
    question: str
    answer_uppercase: bool = False
    answer_no_spaces: bool = False


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).strip().split())


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )
    return without_marks.replace("đ", "d").replace("Đ", "D")


def is_vietnamese_text(text: str) -> bool:
    lowered = unicodedata.normalize("NFC", text).casefold()
    if any(character in VIETNAMESE_DIACRITICS for character in lowered):
        return True
    ascii_tokens = set(re.findall(r"[a-z]+", strip_diacritics(lowered)))
    return len(ascii_tokens.intersection(VIETNAMESE_FUNCTION_WORDS)) >= 2


def _strip_search_scaffolding(text: str) -> str:
    normalized = _normalize(text)
    patterns = (
        r"^(?:hãy\s+)?tìm(?:\s+kiếm)?\s+(?:video|đoạn video|cảnh|hình ảnh)\s+(?:về|có|cho thấy)?\s*",
        r"^(?:please\s+)?find\s+(?:a\s+)?(?:video|video frame|scene|image|picture)\s+(?:of|with|showing)?\s*",
    )
    for pattern in patterns:
        cleaned = re.sub(pattern, "", normalized, flags=re.IGNORECASE)
        if cleaned != normalized and cleaned.strip():
            return cleaned.strip(" .")
    return normalized


def _visual_clauses(text: str) -> list[str]:
    """Split a long, multi-scene description into CLIP-sized visual evidence."""
    sentence_parts = re.split(r"(?<=[.!?])\s+|\s*;\s*", _normalize(text))
    clauses: list[str] = []
    for sentence in sentence_parts:
        sentence = sentence.strip(" ,;:.-")
        if not sentence:
            continue
        pieces = [sentence]
        if len(sentence) > 280:
            comma_parts = [part.strip(" ,;:.-") for part in sentence.split(",")]
            useful = [part for part in comma_parts if len(part.split()) >= 4]
            if len(useful) >= 2:
                pieces = useful
        for piece in pieces:
            cleaned = _strip_search_scaffolding(piece)
            if len(cleaned.split()) >= 3:
                clauses.append(cleaned)
    return list(dict.fromkeys(clauses))[:MAX_VISUAL_CLAUSES]


def _temporal_edges(scenes: list[str]) -> list[tuple[int, int]]:
    """Infer only explicit ordering edges; unrelated evidence stays unordered."""
    normalized_scenes = [
        " ".join(re.findall(r"[a-z0-9]+", strip_diacritics(scene.casefold())))
        for scene in scenes
    ]
    later_markers = (
        "sau do",
        "phan canh sau",
        "tiep theo",
        "ke tiep",
        "later",
        "after that",
        "following",
        "next",
    )
    start_markers = (
        "o dau",
        "dau doan",
        "dau clip",
        "bat dau",
        "at the beginning",
        "the clip begins",
    )
    end_markers = (
        "cuoi doan",
        "cuoi clip",
        "ket thuc",
        "at the end",
        "the clip ends",
    )
    edges: list[tuple[int, int]] = []
    for index, normalized in enumerate(normalized_scenes):
        if index > 0 and any(marker in normalized for marker in later_markers):
            edges.append((index - 1, index))
        if any(marker in normalized for marker in start_markers):
            edges.extend((index, later) for later in range(index + 1, len(scenes)))
        if any(marker in normalized for marker in end_markers):
            edges.extend((earlier, index) for earlier in range(index))
    return list(dict.fromkeys(edges))


def expand_query(text: str) -> list[str]:
    """Generate same-language visual prompts; never create mixed VI/EN strings.

    Short descriptions get prompt paraphrases. Long organizer queries are instead
    decomposed into scene-sized clauses so later scenes are not truncated by the
    text encoder and the retrieval pipeline can aggregate evidence at video level.
    """
    base = _strip_search_scaffolding(text)
    clauses = _visual_clauses(base)
    if len(base) >= LONG_QUERY_THRESHOLD or len(clauses) >= 3:
        # Single clauses fit CLIP's short text window but lose cross-scene
        # conjunctions (e.g. preparing boxed meals, then distributing them at a
        # temple). Adjacent pairs retain that identity signal without feeding the
        # full organizer paragraph into a truncated text encoder.
        clause_pairs = [
            f"{clauses[index]}. {clauses[index + 1]}"
            for index in range(len(clauses) - 1)
        ]
        variants = (base, *clauses, *clause_pairs)
    elif is_vietnamese_text(base):
        variants = (
            base,
            f"một khung hình video cho thấy {base}",
            f"cảnh có {base}",
        )
    else:
        variants = (
            base,
            f"a video frame showing {base}",
            f"a photo of {base}",
        )
    return list(dict.fromkeys(_normalize(variant) for variant in variants if variant.strip()))


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", normalized_text) is not None


def extract_object_keywords(text: str) -> list[str]:
    normalized = strip_diacritics(text.casefold())
    normalized = " ".join(re.findall(r"[a-z0-9]+", normalized))
    labels = [
        label
        for phrase, label in sorted(OBJECT_LABEL_HINTS.items(), key=lambda item: -len(item[0]))
        if _contains_phrase(normalized, phrase)
    ]
    return list(dict.fromkeys(labels))


def extract_ocr_keywords(text: str) -> list[str]:
    quoted = [
        _normalize(match)
        for match in re.findall(r"[\"“”']([^\"“”']{2,})[\"“”']", text)
    ]
    normalized = strip_diacritics(text.casefold())
    has_ocr_intent = any(marker in normalized for marker in OCR_INTENT_MARKERS)
    explicit_quotes = [
        phrase
        for phrase in quoted
        if has_ocr_intent or phrase.isupper() or len(phrase.split()) <= 3
    ]
    if explicit_quotes:
        return list(dict.fromkeys(explicit_quotes))
    if not has_ocr_intent:
        return []

    # If the query explicitly says that a sign/screen contains particular text,
    # retain only the content after the writing verb instead of every visual word.
    match = re.search(
        r"(?:ghi|viết|hiển thị|says|reads|written)\s+(?:là\s+)?(.+?)(?:[?.]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r"(?:dòng\s+chữ|cụm\s+chữ|text)\s+(.+?)"
            r"(?=\s+(?:màu|trên\s+nền|ở\s+)|[,.;?]|$)",
            text,
            flags=re.IGNORECASE,
        )
    if not match:
        return []
    phrase = _normalize(match.group(1).strip(" \"“”'"))
    return [phrase] if phrase else []


def parse_semantic_query(text: str) -> SemanticQuery:
    original = _normalize(text)
    if not original:
        raise ValueError("Query must not be empty.")
    retrieval_text = _strip_search_scaffolding(original)
    scenes = _visual_clauses(retrieval_text) or [retrieval_text]
    temporal_edges = _temporal_edges(scenes)
    return SemanticQuery(
        original_text=original,
        language="vi" if is_vietnamese_text(original) else "en",
        retrieval_text=retrieval_text,
        scenes=scenes,
        temporal_ordered=bool(temporal_edges),
        temporal_edges=temporal_edges,
        expansions=expand_query(retrieval_text),
        object_keywords=extract_object_keywords(original),
        ocr_keywords=extract_ocr_keywords(original),
    )


def parse_qa_query(text: str) -> QaTask:
    """Accept the organizer's complete Q&A query in one input field."""
    original = _normalize(text)
    if not original:
        raise ValueError("Q&A query must not be empty.")

    labeled = re.search(
        r"(?:câu hỏi|question)\s*:\s*(.+)$",
        original,
        flags=re.IGNORECASE,
    )
    directive_pattern = re.compile(
        r"(?P<question>(?:hãy\s+cho\s+biết|hỏi)\s+[^:]{3,240})\s*:\s*",
        flags=re.IGNORECASE,
    )
    directive = directive_pattern.search(original)
    question_sentence_pattern = re.compile(
        r"(?:^|(?<=[.!?]))\s*(?P<question>[^.!?]{3,300}\?)",
        flags=re.IGNORECASE,
    )
    question_sentences = list(question_sentence_pattern.finditer(original))
    question = (
        labeled.group(1).strip()
        if labeled
        else directive.group("question").strip() if directive else original
    )
    if not labeled and not directive and question_sentences:
        # Organizer prompts often repeat the same question after two different
        # visual descriptions.  The last explicit question is the clean answer
        # instruction; the surrounding prose remains retrieval evidence.
        question = question_sentences[-1].group("question").strip()
    event_labeled = re.search(
        r"(?:mô tả sự kiện|event description|mô tả|description)\s*:\s*(.+?)"
        r"(?=\s*(?:câu hỏi|question)\s*:)",
        original,
        flags=re.IGNORECASE,
    )
    # An explicit event description is a cleaner visual retrieval prompt than the
    # answer wording. For free-form organizer queries, preserve the complete text
    # because it contains all available event context.
    if event_labeled:
        retrieval_text = event_labeled.group(1).strip()
    elif labeled:
        # The question often reveals the requested answer type (colour, count,
        # name) but is not part of the event identity.  Keeping it in retrieval
        # lets a generic answer attribute dominate the distinctive scene, e.g. a
        # red sedan outranking the described red convertible with elderly riders.
        retrieval_text = original[: labeled.start()].strip(" ,;:.-")
    elif directive:
        retrieval_text = _normalize(directive_pattern.sub("", original))
    elif question_sentences:
        retrieval_text = _normalize(question_sentence_pattern.sub(" ", original))
        if not retrieval_text:
            retrieval_text = original
    else:
        retrieval_text = original
    normalized_ascii = strip_diacritics(original.casefold())
    return QaTask(
        original_text=original,
        retrieval_text=retrieval_text,
        question=question,
        answer_uppercase=(
            "viet hoa" in normalized_ascii
            or "in hoa" in normalized_ascii
            or "uppercase" in normalized_ascii
        ),
        answer_no_spaces=(
            "khong co khoang trang" in normalized_ascii
            or "khong khoang trang" in normalized_ascii
            or "no spaces" in normalized_ascii
        ),
    )


_NUMBERED_EVENT = re.compile(
    r"(?:^|(?<=[\s,;:]))(?:\(\s*\d+\s*\)|\d+[.)]|E\s*\d+\s*[:.)-]|"
    r"(?:event|moment|khoảnh\s*khắc|sự\s*kiện)\s*\d+\s*[:.)-])\s*",
    flags=re.IGNORECASE,
)


def _trake_context(prefix: str) -> str:
    context = prefix.strip(" \t\r\n,;:.-")
    context = re.sub(
        r"^(?:hãy\s+)?tìm\s+\d+\s+khoảnh\s+khắc(?:\s+chính)?\s*",
        "",
        context,
        flags=re.IGNORECASE,
    )
    context = re.sub(
        r"^(?:please\s+)?find\s+\d+\s+(?:key\s+)?(?:moments|events)\s*",
        "",
        context,
        flags=re.IGNORECASE,
    )
    context = re.sub(r"^(?:khi|trong|của|when|during|of)\s+", "", context, flags=re.IGNORECASE)
    context = re.sub(
        r"(?:tìm|xác định|find|identify)\s+(?:các\s+)?(?:khoảnh\s+khắc|moments|events)"
        r"(?:\s+(?:sau|following))?\s*$",
        "",
        context,
        flags=re.IGNORECASE,
    )
    return context.strip(" ,;:.-")


def split_trake_moments(text: str) -> list[str]:
    """Parse one complete organizer TRAKE query into its ordered moments."""
    original = _normalize(text.replace("\n", " "))
    if not original:
        return []

    markers = list(_NUMBERED_EVENT.finditer(original))
    if len(markers) >= 2:
        prefix = original[: markers[0].start()]
        context = _trake_context(prefix)
        moments: list[str] = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(original)
            moment = original[marker.end() : end].strip(" ,;:.-")
            if not moment:
                continue
            if context and len(context.split()) >= 3:
                moment = f"{context}, {moment}"
            moments.append(moment)
        return moments

    bullet_parts = [
        part.strip(" ,;:.-")
        for part in re.split(r"\s*[•▪◦]\s*|\s+[–-]\s+", original)
        if part.strip(" ,;:.-")
    ]
    if len(bullet_parts) >= 2:
        return bullet_parts

    connector = re.compile(
        r"\s*(?:;|\b(?:sau đó|tiếp theo|kế tiếp|cuối cùng|rồi|"
        r"after that|next|then|finally)\b)\s*",
        flags=re.IGNORECASE,
    )
    connected = [part.strip(" ,;:.-") for part in connector.split(original) if part.strip(" ,;:.-")]
    return connected if len(connected) >= 2 else [original]
