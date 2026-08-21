from fastapi import APIRouter, HTTPException

from app.models import TrakeQuery, TrakeResponse, TrakeResult
from app.pipelines.trake_pipeline import run_trake_query

router = APIRouter(prefix="/search/trake", tags=["trake"])


@router.post("", response_model=TrakeResponse)
def search_trake(payload: TrakeQuery) -> TrakeResponse:
    try:
        moments, rows = run_trake_query(payload.text, top_k=100)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrakeResponse(
        moments=moments,
        results=[TrakeResult(**r) for r in rows],
    )
