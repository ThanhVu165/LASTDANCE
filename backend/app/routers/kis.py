from fastapi import APIRouter

from app.models import KisQuery, KisResponse, KisResult
from app.pipelines.kis_pipeline import run_kis_query

router = APIRouter(prefix="/search/kis", tags=["kis"])


@router.post("", response_model=KisResponse)
def search_kis(payload: KisQuery) -> KisResponse:
    rows = run_kis_query(payload.text, top_k=100)
    return KisResponse(results=[KisResult(**r) for r in rows])
