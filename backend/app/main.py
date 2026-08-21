import re
import sys
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import (
    KEYFRAMES_DIR,
    KIS_EXACT_FRAME_ENABLED,
    KIS_LONG_QUERY_CANDIDATES,
    KIS_STORYBOARD_ENABLED,
    MODEL_QUERY_PLANNER_ENABLED,
    MODEL_REPAIR_ENABLED,
    MODEL_RERANK_ENABLED,
    MODEL_RERANK_NAME,
    MODEL_RERANK_TOP_VIDEOS,
    QA_TEMPORAL_SAMPLES,
    QA_VLM_TOP_VIDEOS,
    QUERIES_DIR,
    QUERY_TRANSLATION_ENABLED,
    RANKING_CUTOFFS,
    SIGLIP_INDEX_PATH,
    TRAKE_EXACT_FRAME_ENABLED,
    VIDEOS_DIR,
    VLM_RERANK_ENABLED,
    VLM_RERANK_FRAMES_PER_VIDEO,
    VLM_RERANK_GROUP_SIZE,
    VLM_RERANK_TOP_VIDEOS,
    VQA_DEVICE,
    VQA_MODEL_NAME,
    VIDEO_WINDOW_INDEX_PATH,
)
from app.rerank.model_reranker import model_reranker_status
from app.routers.kis import router as kis_router
from app.routers.qa import router as qa_router
from app.routers.submission import router as submission_router
from app.routers.trake import router as trake_router

app = FastAPI(title="AIC2026 Retrieval System", version="0.2.0")

app.include_router(kis_router)
app.include_router(qa_router)
app.include_router(trake_router)
app.include_router(submission_router)

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


@lru_cache(maxsize=256)
def _source_frame_jpeg(video_id: str, frame_id: int) -> bytes:
    import cv2

    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Cannot open source video {video_id}.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode frame {frame_id} from {video_id}.")
    encoded, buffer = cv2.imencode(".jpg", frame)
    if not encoded:
        raise ValueError(f"Cannot encode frame {frame_id} from {video_id}.")
    return buffer.tobytes()


@app.get("/health")
def health() -> dict:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
        torch_version = torch.__version__
        torch_cuda_build = torch.version.cuda
    except ImportError:
        cuda_available = False
        device_name = None
        torch_version = None
        torch_cuda_build = None
    vqa_requires_cuda = VQA_DEVICE.strip().lower().startswith(("cuda", "gpu"))
    return {
        "ok": True,
        "python_executable": sys.executable,
        "torch_version": torch_version,
        "torch_cuda_build": torch_cuda_build,
        "cuda_available": cuda_available,
        "cuda_device": device_name,
        "vqa_model": VQA_MODEL_NAME,
        "vqa_device": VQA_DEVICE,
        "vqa_ready": cuda_available or not vqa_requires_cuda,
        "query_translation_enabled": QUERY_TRANSLATION_ENABLED,
        "model_query_planner_enabled": MODEL_QUERY_PLANNER_ENABLED,
        "model_rerank_enabled": MODEL_RERANK_ENABLED,
        "model_rerank_model": MODEL_RERANK_NAME,
        "model_rerank_top_videos": MODEL_RERANK_TOP_VIDEOS,
        "model_repair_enabled": MODEL_REPAIR_ENABLED,
        "model_rerank_status": model_reranker_status(),
        "siglip_index_ready": SIGLIP_INDEX_PATH.exists(),
        "video_window_index_ready": VIDEO_WINDOW_INDEX_PATH.exists(),
        "vlm_rerank_enabled": VLM_RERANK_ENABLED,
        "vlm_rerank_top_videos": VLM_RERANK_TOP_VIDEOS,
        "vlm_rerank_frames_per_video": VLM_RERANK_FRAMES_PER_VIDEO,
        "vlm_rerank_group_size": VLM_RERANK_GROUP_SIZE,
        "qa_vlm_top_videos": QA_VLM_TOP_VIDEOS,
        "qa_temporal_samples": QA_TEMPORAL_SAMPLES,
        "kis_exact_frame_enabled": KIS_EXACT_FRAME_ENABLED,
        "kis_storyboard_enabled": KIS_STORYBOARD_ENABLED,
        "kis_long_query_candidates": KIS_LONG_QUERY_CANDIDATES,
        "trake_exact_frame_enabled": TRAKE_EXACT_FRAME_ENABLED,
        "ranking_cutoffs": RANKING_CUTOFFS,
    }


@app.get("/video/{video_id}/keyframe/{local_idx}")
def get_keyframe(video_id: str, local_idx: int):
    """Serve a keyframe JPEG by its LOCAL index (matches the on-disk NNN.jpg
    filename) — not the real video frame_id, which is only meaningful for
    submission export."""
    folder = KEYFRAMES_DIR / video_id
    for width in (3, 4):
        candidate = folder / f"{local_idx:0{width}d}.jpg"
        if candidate.exists():
            return FileResponse(path=str(candidate))
    raise HTTPException(status_code=404, detail="Keyframe not found.")


@app.get("/video/{video_id}/frame/{frame_id}")
def get_source_frame(video_id: str, frame_id: int):
    """Serve an exact source-video frame selected by coarse-to-fine refinement."""
    if not _VIDEO_ID_PATTERN.fullmatch(video_id) or frame_id < 0:
        raise HTTPException(status_code=422, detail="Invalid video/frame identifier.")
    try:
        content = _source_frame_jpeg(video_id, frame_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Source video not found.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(content=content, media_type="image/jpeg")


@app.get("/queries")
def list_query_files() -> dict:
    """List query .txt files provided by the organizers under queries/, following
    the naming convention query-N-<kis|qa|trake>.txt."""
    if not QUERIES_DIR.exists():
        return {"files": []}
    files = sorted(p.name for p in QUERIES_DIR.glob("*.txt"))
    return {"files": files}


@app.get("/queries/{filename}")
def get_query_file(filename: str) -> dict:
    path = QUERIES_DIR / filename
    if not path.exists() or path.suffix != ".txt":
        raise HTTPException(status_code=404, detail="Query file not found.")
    return {"filename": filename, "content": path.read_text(encoding="utf-8")}
