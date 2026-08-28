"""Load official AIC26 qualifier query packages without extracting untrusted files."""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable

from shared.schemas.online import QuerySpec, TaskType


_TASK_SUFFIX = re.compile(r"-(kis|qa|trake)$", re.IGNORECASE)
_TRAKE_EVENT = re.compile(r"(?im)^\s*E(\d+)\s*[:.\-]?\s*\S")
_NATURAL_NUMBER = re.compile(r"(\d+)")


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in _NATURAL_NUMBER.split(value))


def _task_type(query_name: str) -> TaskType:
    match = _TASK_SUFFIX.search(query_name)
    if match is None:
        raise ValueError(f"unsupported query filename: {query_name}.txt")
    return {
        "kis": TaskType.KIS,
        "qa": TaskType.QA,
        "trake": TaskType.TRAKE,
    }[match.group(1).lower()]


def _event_count(raw_query: str) -> int | None:
    event_numbers = [int(value) for value in _TRAKE_EVENT.findall(raw_query)]
    if not event_numbers:
        return None
    expected = list(range(1, max(event_numbers) + 1))
    if sorted(set(event_numbers)) != expected:
        raise ValueError(f"TRAKE event labels must be contiguous E1..E{max(event_numbers)}")
    return len(expected)


def query_spec_from_text(filename: str, raw_query: str) -> QuerySpec:
    source = PurePosixPath(filename)
    if source.name != filename or source.suffix.casefold() != ".txt":
        raise ValueError("query package entries must be root-level .txt files")
    query_name = source.stem
    task_type = _task_type(query_name)
    expected = _event_count(raw_query) if task_type == TaskType.TRAKE else None
    if task_type == TaskType.TRAKE and expected is None:
        raise ValueError(f"{filename} does not declare explicit E1..EN events")
    return QuerySpec(
        query_name=query_name,
        source_filename=filename,
        task_type=task_type,
        raw_query=raw_query,
        expected_event_count=expected,
    )


def load_query_specs_from_zip(payload: bytes) -> list[QuerySpec]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("query package is not a valid ZIP file") from error
    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("query package contains no files")
        specs: list[QuerySpec] = []
        for name in names:
            if PurePosixPath(name).name != name:
                raise ValueError("query package must contain .txt files at ZIP root")
            if not name.casefold().endswith(".txt"):
                raise ValueError(f"unexpected non-query file in package: {name}")
            raw = archive.read(name).decode("utf-8-sig")
            specs.append(query_spec_from_text(name, raw))
    return _validate_unique(specs)


def load_query_specs_from_directory(path: Path) -> list[QuerySpec]:
    specs = [
        query_spec_from_text(item.name, item.read_text(encoding="utf-8-sig"))
        for item in path.iterdir()
        if item.is_file()
    ]
    return _validate_unique(specs)


def _validate_unique(specs: Iterable[QuerySpec]) -> list[QuerySpec]:
    ordered = sorted(specs, key=lambda item: _natural_key(item.query_name))
    if not ordered:
        raise ValueError("query package contains no query .txt files")
    names = [item.query_name for item in ordered]
    if len(names) != len(set(names)):
        raise ValueError("query package contains duplicate query names")
    return ordered
