"""Official AIC26 per-query drafts and fail-closed submission bundle export."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from shared.schemas.online import (
    KISCandidate,
    QACandidate,
    QuerySpec,
    TaskType,
    TrakeCandidate,
)

from .config import OnlineLayout


Candidate = KISCandidate | QACandidate | TrakeCandidate
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_ZIP = re.compile(r"[^A-Za-z0-9]+")
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")
_PLACEHOLDER_ANSWERS = {"uncertain"}


@dataclass(frozen=True, slots=True)
class SubmissionExportReport:
    zip_path: Path
    csv_paths: dict[str, Path]
    row_counts: dict[str, int]
    csv_sha256: dict[str, str]
    zip_sha256: str


def sanitize_component(value: str, *, fallback: str) -> str:
    value = _SAFE_COMPONENT.sub("-", value.strip()).strip(".-")
    return (value or fallback)[:80]


def sanitize_zip_name(value: str) -> str:
    stem = _SAFE_ZIP.sub("", Path(value).stem)
    return f"{(stem or 'submission')[:80]}.zip"


def parse_candidate(value: Candidate | dict[str, Any]) -> Candidate:
    if isinstance(value, (KISCandidate, QACandidate, TrakeCandidate)):
        return value
    if "local_idx" in json.dumps(value, ensure_ascii=False):
        raise ValueError("submission payload must never contain local_idx")
    candidate_type = value.get("candidate_type")
    models = {"KIS": KISCandidate, "QA": QACandidate, "TRAKE": TrakeCandidate}
    if candidate_type not in models:
        raise ValueError("candidate_type must be KIS, QA or TRAKE")
    return models[candidate_type].model_validate(value)


def _candidate_key(item: Candidate) -> tuple[Any, ...]:
    if isinstance(item, TrakeCandidate):
        return (item.video_id, *item.frame_ids)
    return (item.video_id, item.frame_id)


class SubmissionWorkspace:
    """A bundle of one independent, rank-preserving draft per official query."""

    def __init__(
        self,
        *,
        folder_name: str,
        expected_queries: Iterable[QuerySpec | dict[str, Any]],
        layout: OnlineLayout | None = None,
        query_drafts: Mapping[str, Iterable[Candidate | dict[str, Any]]] | None = None,
        query_history: Iterable[dict[str, Any]] = (),
        provenance: dict[str, str] | None = None,
        catalog: Any = None,
    ) -> None:
        self.layout = layout or OnlineLayout.from_environment()
        self.folder_name = sanitize_component(folder_name, fallback="submission-review")
        specs = [
            item if isinstance(item, QuerySpec) else QuerySpec.model_validate(item)
            for item in expected_queries
        ]
        self.expected_queries = {item.query_name: item for item in specs}
        if not self.expected_queries:
            raise ValueError("submission bundle requires at least one query")
        if len(self.expected_queries) != len(specs):
            raise ValueError("duplicate query_name in submission bundle")
        raw_drafts = query_drafts or {}
        unknown = set(raw_drafts) - set(self.expected_queries)
        if unknown:
            raise ValueError(f"draft contains unknown queries: {sorted(unknown)}")
        self.query_drafts: dict[str, list[Candidate]] = {
            name: [parse_candidate(item) for item in raw_drafts.get(name, ())]
            for name in self.expected_queries
        }
        self.query_history = [dict(item) for item in query_history]
        self.provenance = dict(provenance or {})
        self._catalog = catalog
        self._catalog_cache: dict[tuple[str, int], float] | None = None
        for name, entries in self.query_drafts.items():
            if entries:
                self._validate_entries(self.expected_queries[name], entries)

    @property
    def directory(self) -> Path:
        target = (self.layout.submissions / self.folder_name).resolve()
        root = self.layout.submissions.resolve()
        if target != root and root not in target.parents:
            raise ValueError("submission directory escapes AIC_DATA/submissions")
        return target

    @property
    def state_path(self) -> Path:
        return self.directory / "workspace.json"

    @property
    def submission_directory(self) -> Path:
        return self.directory / "submission"

    def replace_query_draft(
        self,
        query_name: str,
        candidates: Iterable[Candidate | dict[str, Any]],
    ) -> None:
        spec = self._spec(query_name)
        replacement = [parse_candidate(item) for item in candidates]
        self._validate_entries(spec, replacement)
        self.query_drafts[query_name] = replacement

    def merge_ranked(
        self,
        query_name: str,
        candidates: Iterable[Candidate | dict[str, Any]],
        *,
        limit: int = 100,
    ) -> None:
        if not 1 <= limit <= 100:
            raise ValueError("merge limit must be between 1 and 100")
        spec = self._spec(query_name)
        current = list(self.query_drafts[query_name])
        incoming = [parse_candidate(item) for item in candidates]
        self._validate_entries(spec, incoming)
        seen = {_candidate_key(item) for item in current}
        merged = list(current)
        for item in incoming:
            key = _candidate_key(item)
            if key in seen:
                continue
            merged.append(item)
            seen.add(key)
            if len(merged) >= limit:
                break
        self._validate_entries(spec, merged)
        self.query_drafts[query_name] = merged

    def remove(self, query_name: str, indices: Iterable[int]) -> None:
        spec = self._spec(query_name)
        selected = set(indices)
        replacement = [
            item for index, item in enumerate(self.query_drafts[query_name]) if index not in selected
        ]
        if replacement:
            self._validate_entries(spec, replacement)
        self.query_drafts[query_name] = replacement

    def reorder(self, query_name: str, old_index: int, new_index: int) -> None:
        entries = list(self.query_drafts[self._spec(query_name).query_name])
        if not 0 <= old_index < len(entries) or not 0 <= new_index < len(entries):
            raise IndexError("draft reorder index is out of range")
        item = entries.pop(old_index)
        entries.insert(new_index, item)
        self.query_drafts[query_name] = entries

    def validate_complete(self) -> None:
        for name, spec in self.expected_queries.items():
            entries = self.query_drafts.get(name, [])
            if not entries:
                raise ValueError(f"missing submission rows for {name}")
            self._validate_entries(spec, entries)
            self.validate_csv_bytes(spec, self.csv_bytes(name))

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "profile": "AIC26_QUALIFIER_OFFICIAL",
            "folder_name": self.folder_name,
            "expected_queries": [
                item.model_dump(mode="json") for item in self.expected_queries.values()
            ],
            "query_drafts": {
                name: [item.model_dump(mode="json") for item in entries]
                for name, entries in self.query_drafts.items()
            },
            "query_history": self.query_history,
            "provenance": self.provenance,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="workspace-", suffix=".json", dir=self.directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, self.state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.state_path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        layout: OnlineLayout | None = None,
        catalog: Any = None,
    ) -> "SubmissionWorkspace":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError("unsupported workspace schema; expected schema_version=2")
        return cls(
            folder_name=payload["folder_name"],
            expected_queries=payload["expected_queries"],
            layout=layout,
            query_drafts=payload.get("query_drafts", {}),
            query_history=payload.get("query_history", []),
            provenance=payload.get("provenance", {}),
            catalog=catalog,
        )

    def csv_bytes(self, query_name: str) -> bytes:
        spec = self._spec(query_name)
        entries = self.query_drafts[query_name]
        if not entries:
            raise ValueError(f"submission draft is empty for {query_name}")
        self._validate_entries(spec, entries)
        text = io.StringIO(newline="")
        writer = csv.writer(text, delimiter=",", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        for item in entries:
            if isinstance(item, KISCandidate):
                writer.writerow([item.video_id, item.frame_id])
            elif isinstance(item, QACandidate):
                writer.writerow([item.video_id, item.frame_id, item.answer])
            else:
                writer.writerow([item.video_id, *item.frame_ids])
        payload = text.getvalue().encode("utf-8")
        self.validate_csv_bytes(spec, payload)
        return payload

    def validate_csv_bytes(self, spec: QuerySpec, payload: bytes) -> None:
        if payload.startswith(b"\xef\xbb\xbf"):
            raise ValueError(f"{spec.csv_filename} must be UTF-8 without BOM")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{spec.csv_filename} is not UTF-8") from error
        rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=","))
        if not 1 <= len(rows) <= 100:
            raise ValueError(f"{spec.csv_filename} must contain 1-100 rows")
        header_tokens = {"video_id", "frame_id", "answer", "query_id", "local_idx"}
        if any(cell.strip().casefold() in header_tokens for cell in rows[0]):
            raise ValueError(f"{spec.csv_filename} must not contain a header row")
        expected_width = (
            2
            if spec.task_type == TaskType.KIS
            else 3
            if spec.task_type == TaskType.QA
            else 1 + int(spec.expected_event_count or 0)
        )
        seen: set[tuple[Any, ...]] = set()
        catalog = self._catalog_index()
        for row_number, row in enumerate(rows, 1):
            if len(row) != expected_width:
                raise ValueError(
                    f"{spec.csv_filename} row {row_number} has {len(row)} columns; "
                    f"expected {expected_width}"
                )
            video_id = row[0]
            if not _VIDEO_ID.fullmatch(video_id) or video_id.casefold().endswith(".mp4"):
                raise ValueError(f"{spec.csv_filename} row {row_number} has invalid video_id")
            frame_values = row[1:2] if spec.task_type != TaskType.TRAKE else row[1:]
            if any(_INTEGER.fullmatch(value) is None for value in frame_values):
                raise ValueError(f"{spec.csv_filename} row {row_number} has a non-integer frame ID")
            frame_ids = [int(value) for value in frame_values]
            times: list[float] = []
            for frame_id in frame_ids:
                key = (video_id, frame_id)
                if key not in catalog:
                    raise ValueError(
                        f"{spec.csv_filename} row {row_number} references unknown frame {key}"
                    )
                times.append(catalog[key])
            if spec.task_type == TaskType.QA:
                answer = row[2]
                if not answer or not answer.strip() or len(answer) > 100:
                    raise ValueError(f"{spec.csv_filename} row {row_number} has an invalid QA answer")
                if answer.strip().casefold() in _PLACEHOLDER_ANSWERS:
                    raise ValueError(f"{spec.csv_filename} row {row_number} contains a placeholder answer")
                key = (video_id, frame_ids[0])
            elif spec.task_type == TaskType.TRAKE:
                if len(set(frame_ids)) != len(frame_ids):
                    raise ValueError(f"{spec.csv_filename} row {row_number} reuses a frame")
                if any(right <= left for left, right in zip(times, times[1:])):
                    raise ValueError(
                        f"{spec.csv_filename} row {row_number} is not strictly increasing in time"
                    )
                key = (video_id, *frame_ids)
            else:
                key = (video_id, frame_ids[0])
            if key in seen:
                raise ValueError(f"duplicate submission hypothesis in {spec.csv_filename}: {key}")
            seen.add(key)

    def export_csv(self, query_name: str) -> Path:
        spec = self._spec(query_name)
        payload = self.csv_bytes(query_name)
        self.submission_directory.mkdir(parents=True, exist_ok=True)
        destination = self.submission_directory / spec.csv_filename
        self._atomic_write(destination, payload)
        return destination

    def export_zip(self, *, zip_name: str = "submissionround1.zip") -> SubmissionExportReport:
        self.validate_complete()
        self.submission_directory.mkdir(parents=True, exist_ok=True)
        payloads = {name: self.csv_bytes(name) for name in self.expected_queries}
        csv_paths: dict[str, Path] = {}
        for name, payload in payloads.items():
            destination = self.submission_directory / self.expected_queries[name].csv_filename
            self._atomic_write(destination, payload)
            csv_paths[name] = destination
        safe_name = sanitize_zip_name(zip_name)
        destination = self.directory / safe_name
        temporary = destination.with_suffix(".zip.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, spec in self.expected_queries.items():
                    archive.writestr(f"submission/{spec.csv_filename}", payloads[name])
            self.validate_zip_path(temporary, payloads=payloads)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return SubmissionExportReport(
            zip_path=destination,
            csv_paths=csv_paths,
            row_counts={name: len(self.query_drafts[name]) for name in self.expected_queries},
            csv_sha256={
                name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
            },
            zip_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )

    def validate_zip_path(
        self,
        path: Path,
        *,
        payloads: Mapping[str, bytes] | None = None,
    ) -> None:
        expected = {
            f"submission/{spec.csv_filename}": name
            for name, spec in self.expected_queries.items()
        }
        try:
            archive = zipfile.ZipFile(path)
        except zipfile.BadZipFile as error:
            raise ValueError("submission is not a valid ZIP file") from error
        with archive:
            entries = [info.filename for info in archive.infolist() if not info.is_dir()]
            if len(entries) != len(expected) or set(entries) != set(expected):
                raise ValueError(
                    "submission ZIP entry list must contain every expected CSV under submission/"
                )
            for entry, query_name in expected.items():
                content = archive.read(entry)
                self.validate_csv_bytes(self.expected_queries[query_name], content)
                if payloads is not None and content != payloads[query_name]:
                    raise ValueError(f"submission ZIP content mismatch for {entry}")

    def _spec(self, query_name: str) -> QuerySpec:
        try:
            return self.expected_queries[query_name]
        except KeyError as error:
            raise ValueError(f"unknown query_name: {query_name}") from error

    def _validate_entries(self, spec: QuerySpec, entries: list[Candidate]) -> None:
        if not 1 <= len(entries) <= 100:
            raise ValueError(f"{spec.query_name} draft must contain 1-100 hypotheses")
        expected_type = {
            TaskType.KIS: KISCandidate,
            TaskType.QA: QACandidate,
            TaskType.TRAKE: TrakeCandidate,
        }[spec.task_type]
        seen: set[tuple[Any, ...]] = set()
        catalog = self._catalog_index()
        for item in entries:
            if not isinstance(item, expected_type):
                raise ValueError(f"{spec.query_name} contains the wrong task candidate shape")
            if not _VIDEO_ID.fullmatch(item.video_id) or item.video_id.casefold().endswith(".mp4"):
                raise ValueError("video_id must not contain .mp4 or unsafe characters")
            frame_ids = item.frame_ids if isinstance(item, TrakeCandidate) else [item.frame_id]
            times = []
            for frame_id in frame_ids:
                key = (item.video_id, frame_id)
                if key not in catalog:
                    raise ValueError(f"unknown submission frame: {key}")
                times.append(catalog[key])
            if isinstance(item, QACandidate):
                if not item.answer or not item.answer.strip() or len(item.answer) > 100:
                    raise ValueError("QA answer must contain 1-100 characters")
                if item.answer.strip().casefold() in _PLACEHOLDER_ANSWERS:
                    raise ValueError("QA answer must not be an Uncertain placeholder")
            if isinstance(item, TrakeCandidate):
                if len(item.frame_ids) != spec.expected_event_count:
                    raise ValueError(
                        f"TRAKE requires exactly {spec.expected_event_count} frame IDs"
                    )
                if any(frame.video_id != item.video_id for frame in item.evidence):
                    raise ValueError("TRAKE sequence must not mix videos")
                if any(right <= left for left, right in zip(times, times[1:])):
                    raise ValueError("TRAKE frames must be strictly increasing in catalog time")
            key = _candidate_key(item)
            if key in seen:
                raise ValueError(f"duplicate submission hypothesis: {key}")
            seen.add(key)

    def _catalog_index(self) -> dict[tuple[str, int], float]:
        if self._catalog_cache is not None:
            return self._catalog_cache
        if self._catalog is not None:
            self._catalog_cache = {
                (frame.video_id, int(frame.frame_id)): float(frame.pts_time)
                for frame in self._catalog.frames
            }
            return self._catalog_cache
        path = self.layout.catalog
        if not path.is_file():
            raise ValueError(f"frames.csv is required for submission validation: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle)
            required = {"video_id", "frame_id", "pts_time"}
            if rows.fieldnames is None or not required.issubset(rows.fieldnames):
                raise ValueError("frames.csv lacks video_id/frame_id/pts_time")
            self._catalog_cache = {
                (row["video_id"], int(row["frame_id"])): float(row["pts_time"])
                for row in rows
            }
        return self._catalog_cache

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f"{destination.stem}-",
            suffix=destination.suffix,
            dir=destination.parent,
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
