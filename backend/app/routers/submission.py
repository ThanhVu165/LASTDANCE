"""Submission management API: collect ranked answers per query_id, validate them
against the contest's CSV rules, export individual CSV files, and package everything
into a submission.zip ready to upload."""
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from app.config import MAX_SUBMISSION_ROWS
from app.models import SubmissionRow
from app.services.export_csv import build_submission_zip, rows_to_csv_text, validate_rows

router = APIRouter(prefix="/submission", tags=["submission"])

_store: Dict[str, List[SubmissionRow]] = {}


class AddRowsRequest(BaseModel):
    query_id: str
    rows: List[SubmissionRow]
    replace: bool = False


class ZipRequest(BaseModel):
    # query_id -> output csv filename, e.g. {"kis_q1": "query-1-kis.csv"}
    files: Dict[str, str]


@router.post("/add")
def add_rows(payload: AddRowsRequest) -> dict:
    if payload.replace:
        _store[payload.query_id] = []
    bucket = _store.setdefault(payload.query_id, [])
    remaining = MAX_SUBMISSION_ROWS - len(bucket)
    if remaining <= 0:
        raise HTTPException(status_code=400, detail=f"Submission đã có đủ {MAX_SUBMISSION_ROWS} dòng.")
    bucket.extend(payload.rows[:remaining])
    bucket.sort(key=lambda x: x.rank)
    return {"query_id": payload.query_id, "total_rows": len(bucket)}


@router.get("/{query_id}")
def get_rows(query_id: str) -> dict:
    return {"query_id": query_id, "rows": [row.model_dump() for row in _store.get(query_id, [])]}


@router.delete("/{query_id}")
def clear_rows(query_id: str) -> dict:
    _store.pop(query_id, None)
    return {"query_id": query_id, "cleared": True}


@router.get("/{query_id}/validate")
def validate(query_id: str, expected_trake_events: Optional[int] = None) -> dict:
    rows = _store.get(query_id, [])
    errors = validate_rows(rows, expected_trake_events=expected_trake_events)
    return {"query_id": query_id, "ok": len(errors) == 0, "errors": errors}


@router.get("/{query_id}/export")
def export_rows(query_id: str) -> PlainTextResponse:
    rows = _store.get(query_id, [])
    if not rows:
        raise HTTPException(status_code=404, detail="Không có dữ liệu cho query_id này.")
    errors = validate_rows(rows)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})
    return PlainTextResponse(content=rows_to_csv_text(rows), media_type="text/csv")


@router.post("/zip")
def export_zip(payload: ZipRequest) -> Response:
    csv_files: dict[str, str] = {}
    all_errors: dict[str, list[str]] = {}

    for query_id, filename in payload.files.items():
        rows = _store.get(query_id, [])
        if not rows:
            all_errors[query_id] = ["Không có dữ liệu."]
            continue
        errors = validate_rows(rows)
        if errors:
            all_errors[query_id] = errors
            continue
        csv_files[filename] = rows_to_csv_text(rows)

    if all_errors:
        raise HTTPException(status_code=400, detail={"errors": all_errors})

    zip_bytes = build_submission_zip(csv_files)
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=submission.zip"},
    )
