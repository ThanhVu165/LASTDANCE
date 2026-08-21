from fastapi import APIRouter, HTTPException

from app.models import QaQuery, QaResponse, QaResult
from app.pipelines.qa_pipeline import run_qa_query

router = APIRouter(prefix="/search/qa", tags=["qa"])


@router.post("", response_model=QaResponse)
def search_qa(payload: QaQuery) -> QaResponse:
    try:
        rows = run_qa_query(payload.text, top_k=100)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return QaResponse(results=[QaResult(**r) for r in rows])
