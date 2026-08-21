"""Central paths & settings for the AIC2026 retrieval system.

Actual data layout confirmed from the dataset downloaded by the team (differs from
the generic description in the contest PDF — this reflects what is really on disk):

  data/videos/<video_id>.mp4
  data/keyframes/<video_id>/<NNN>.jpg          # local index 1..K, NOT the real frame number
  data/objects/<video_id>/<NNN>.json           # Faster R-CNN / OpenImages detections, same NNN numbering
  data/features/<video_id>.npy                 # shape (K, 512) float16, row i == local index i+1
  data/map-keyframes/<video_id>.csv            # columns: n,pts_time,fps,frame_idx
                                                #   n        = local index (matches keyframe/object NNN)
                                                #   frame_idx = REAL frame number in the video —
                                                #               this is what must be submitted as frame_id!
  data/metadata/<video_id>.json                # YouTube metadata, optional (may be missing)
  data/index/                                  # built artifacts (faiss index, keyframe_index.json, caches)

Hardware note: dev machine has CPU i5-12450H + RTX 4050 (6GB VRAM). CLIP image
features are precomputed by the organizers. OCR indexing uses CUDA, while runtime
text-query encoding remains light enough to share the same PyTorch installation.
"""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"

VIDEOS_DIR = DATA_DIR / "videos"
KEYFRAMES_DIR = DATA_DIR / "keyframes"
OBJECTS_DIR = DATA_DIR / "objects"
FEATURES_DIR = DATA_DIR / "features"
MAP_KEYFRAMES_DIR = DATA_DIR / "map-keyframes"
INDEX_DIR = DATA_DIR / "index"

QUERIES_DIR = ROOT_DIR / "queries"

KEYFRAME_INDEX_PATH = INDEX_DIR / "keyframe_index.json"
FAISS_INDEX_PATH = INDEX_DIR / "clip.faiss"
OBJECTS_CACHE_PATH = INDEX_DIR / "objects_cache.json"
OCR_CACHE_PATH = INDEX_DIR / "ocr_cache.json"
OCR_STATE_PATH = INDEX_DIR / "ocr_state.json"
OCR_MODEL_DIR = DATA_DIR / "models" / "easyocr"

# CLIP model — must match the model used by the organizers to extract image features
# (clip-ViT-B-32) so text/image embeddings live in the same space. It is small enough
# (~150M params) to run comfortably on CPU for text-only encoding at query time.
CLIP_MODEL_NAME = "clip-ViT-B-32"
MULTILINGUAL_CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
CLIP_DEVICE = "cpu"

# General visual question answering. Qwen3-VL 2B uses about 4.2 GiB reserved VRAM
# in measured FP16 inference, so it fits the 6 GB RTX 4050 when the offline OCR
# worker is not occupying the GPU. Keep image tokens bounded.
VQA_MODEL_NAME = os.getenv("AIC_VQA_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
VQA_DEVICE = os.getenv("AIC_VQA_DEVICE", "cuda:0")
VQA_MIN_PIXELS = int(os.getenv("AIC_VQA_MIN_PIXELS", str(256 * 28 * 28)))
VQA_MAX_PIXELS = int(os.getenv("AIC_VQA_MAX_PIXELS", str(512 * 28 * 28)))
VQA_MAX_NEW_TOKENS = int(os.getenv("AIC_VQA_MAX_NEW_TOKENS", "32"))
QA_CONTEXT_RADIUS = int(os.getenv("AIC_QA_CONTEXT_RADIUS", "1"))
QA_MULTIFRAME_TOP_K = int(os.getenv("AIC_QA_MULTIFRAME_TOP_K", "20"))
QA_TEMPORAL_RADIUS_SECONDS = float(
    os.getenv("AIC_QA_TEMPORAL_RADIUS_SECONDS", "90")
)
QA_TEMPORAL_SAMPLES = int(os.getenv("AIC_QA_TEMPORAL_SAMPLES", "12"))
QA_VLM_TOP_VIDEOS = int(os.getenv("AIC_QA_VLM_TOP_VIDEOS", "12"))
QA_MATCH_WEIGHT = float(os.getenv("AIC_QA_MATCH_WEIGHT", "0.65"))
QA_JUDGMENT_MAX_NEW_TOKENS = int(
    os.getenv("AIC_QA_JUDGMENT_MAX_NEW_TOKENS", "64")
)

# The official preliminary-round metric averages R@{1,5,20,50,100}.  A small
# point-wise VLM pass therefore concentrates on the leading video hypotheses,
# while the final ranking deliberately diversifies at those exact cutoffs.
RANKING_CUTOFFS = (1, 5, 20, 50, 100)
VLM_RERANK_ENABLED = os.getenv("AIC_VLM_RERANK_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
VLM_RERANK_TOP_VIDEOS = int(os.getenv("AIC_VLM_RERANK_TOP_VIDEOS", "30"))
VLM_RERANK_GROUP_SIZE = int(os.getenv("AIC_VLM_RERANK_GROUP_SIZE", "3"))
VLM_RERANK_FRAMES_PER_VIDEO = int(
    os.getenv("AIC_VLM_RERANK_FRAMES_PER_VIDEO", "4")
)
VLM_RERANK_TOP_SEQUENCES = int(os.getenv("AIC_VLM_RERANK_TOP_SEQUENCES", "10"))
VLM_RERANK_VISUAL_WEIGHT = float(os.getenv("AIC_VLM_RERANK_VISUAL_WEIGHT", "0.65"))
VLM_RERANK_BEST_FRAME_WEIGHT = float(
    os.getenv("AIC_VLM_RERANK_BEST_FRAME_WEIGHT", "0.10")
)
VLM_RERANK_MAX_NEW_TOKENS = int(os.getenv("AIC_VLM_RERANK_MAX_NEW_TOKENS", "16"))
QUERY_TRANSLATION_ENABLED = os.getenv("AIC_QUERY_TRANSLATION_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
QUERY_TRANSLATION_MAX_NEW_TOKENS = int(
    os.getenv("AIC_QUERY_TRANSLATION_MAX_NEW_TOKENS", "128")
)

# Model-first query planning and multimodal verification.  The planner uses the
# existing instruction VLM text-only; the dedicated reranker is loaded only for
# the scoring phase because both 2B models cannot remain resident together on a
# 6 GiB GPU.
MODEL_QUERY_PLANNER_ENABLED = os.getenv(
    "AIC_MODEL_QUERY_PLANNER_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
MODEL_QUERY_PLANNER_MAX_NEW_TOKENS = int(
    os.getenv("AIC_MODEL_QUERY_PLANNER_MAX_NEW_TOKENS", "768")
)
MODEL_RERANK_ENABLED = os.getenv("AIC_MODEL_RERANK_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
MODEL_RERANK_NAME = os.getenv(
    "AIC_MODEL_RERANK_NAME", "Qwen/Qwen3-VL-Reranker-2B"
)
MODEL_RERANK_DEVICE = os.getenv("AIC_MODEL_RERANK_DEVICE", "cuda:0")
# Runtime search must never stall while trying to fetch a multi-gigabyte model.
# Download/build steps are explicit; an incomplete local cache falls back to the
# already-installed instruction VLM verifier.
MODEL_RERANK_LOCAL_FILES_ONLY = os.getenv(
    "AIC_MODEL_RERANK_LOCAL_FILES_ONLY", "1"
).strip().lower() not in {"0", "false", "no", "off"}
MODEL_RERANK_TOP_VIDEOS = int(os.getenv("AIC_MODEL_RERANK_TOP_VIDEOS", "40"))
MODEL_RERANK_FRAMES_PER_VIDEO = int(
    os.getenv("AIC_MODEL_RERANK_FRAMES_PER_VIDEO", "4")
)
MODEL_RERANK_BATCH_SIZE = int(os.getenv("AIC_MODEL_RERANK_BATCH_SIZE", "1"))
MODEL_RERANK_WEIGHT = float(os.getenv("AIC_MODEL_RERANK_WEIGHT", "0.75"))
MODEL_RERANK_CONFIDENCE_THRESHOLD = float(
    os.getenv("AIC_MODEL_RERANK_CONFIDENCE_THRESHOLD", "0.55")
)
MODEL_RERANK_MIN_VERIFIED_VIDEOS = int(
    os.getenv("AIC_MODEL_RERANK_MIN_VERIFIED_VIDEOS", "20")
)
MODEL_GENERATIVE_VERIFY_GROUP_SIZE = int(
    os.getenv("AIC_MODEL_GENERATIVE_VERIFY_GROUP_SIZE", "4")
)
MODEL_GENERATIVE_VERIFY_MAX_NEW_TOKENS = int(
    os.getenv("AIC_MODEL_GENERATIVE_VERIFY_MAX_NEW_TOKENS", "96")
)
MODEL_REPAIR_ENABLED = os.getenv("AIC_MODEL_REPAIR_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
MODEL_REPAIR_MAX_ROUNDS = int(os.getenv("AIC_MODEL_REPAIR_MAX_ROUNDS", "1"))

# Optional side indexes.  They never replace the organizer CLIP index until a
# complete checkpoint has been published.
SIGLIP_MODEL_NAME = os.getenv(
    "AIC_SIGLIP_MODEL", "google/siglip2-base-patch16-256"
)
SIGLIP_DEVICE = os.getenv("AIC_SIGLIP_DEVICE", "cuda:0")
SIGLIP_QUERY_DEVICE = os.getenv("AIC_SIGLIP_QUERY_DEVICE", "cpu")
SIDE_RETRIEVAL_TOP_K = int(os.getenv("AIC_SIDE_RETRIEVAL_TOP_K", "400"))
SIGLIP_INDEX_PATH = INDEX_DIR / "siglip2.faiss"
SIGLIP_FEATURES_PATH = INDEX_DIR / "siglip2_features.npy"
SIGLIP_STATE_PATH = INDEX_DIR / "siglip2_state.json"
VIDEO_WINDOW_EMBEDDING_MODEL_NAME = os.getenv(
    "AIC_VIDEO_WINDOW_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"
)
VIDEO_WINDOW_DEVICE = os.getenv("AIC_VIDEO_WINDOW_DEVICE", "cuda:0")
VIDEO_WINDOW_LOCAL_FILES_ONLY = os.getenv(
    "AIC_VIDEO_WINDOW_LOCAL_FILES_ONLY", "1"
).strip().lower() not in {"0", "false", "no", "off"}
VIDEO_WINDOW_INDEX_PATH = INDEX_DIR / "video_windows.faiss"
VIDEO_WINDOW_METADATA_PATH = INDEX_DIR / "video_windows.json"
VIDEO_WINDOW_FEATURES_PATH = INDEX_DIR / "video_window_features.npy"
VIDEO_WINDOW_STATE_PATH = INDEX_DIR / "video_window_state.json"
VIDEO_WINDOW_SIZE = int(os.getenv("AIC_VIDEO_WINDOW_SIZE", "6"))
VIDEO_WINDOW_STRIDE = int(os.getenv("AIC_VIDEO_WINDOW_STRIDE", "6"))
VIDEO_WINDOW_TOTAL_PIXELS = int(
    os.getenv("AIC_VIDEO_WINDOW_TOTAL_PIXELS", str(6 * 192 * 192))
)
VIDEO_WINDOW_EMBEDDING_DIM = int(
    os.getenv("AIC_VIDEO_WINDOW_EMBEDDING_DIM", "1024")
)
TRAKE_EXACT_FRAME_ENABLED = os.getenv("AIC_TRAKE_EXACT_FRAME_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
TRAKE_EXACT_FRAME_TOP_K = int(os.getenv("AIC_TRAKE_EXACT_FRAME_TOP_K", "1"))
TRAKE_EXACT_FRAME_COARSE_SAMPLES = int(
    os.getenv("AIC_TRAKE_EXACT_FRAME_COARSE_SAMPLES", "9")
)
TRAKE_EXACT_FRAME_FINE_SAMPLES = int(
    os.getenv("AIC_TRAKE_EXACT_FRAME_FINE_SAMPLES", "13")
)
TRAKE_EXACT_FRAME_MAX_RADIUS = int(
    os.getenv("AIC_TRAKE_EXACT_FRAME_MAX_RADIUS", "120")
)
TRAKE_EXACT_FRAME_MAX_NEW_TOKENS = int(
    os.getenv("AIC_TRAKE_EXACT_FRAME_MAX_NEW_TOKENS", "16")
)
KIS_EXACT_FRAME_ENABLED = os.getenv("AIC_KIS_EXACT_FRAME_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
KIS_EXACT_FRAME_TOP_K = int(os.getenv("AIC_KIS_EXACT_FRAME_TOP_K", "1"))
KIS_STORYBOARD_ENABLED = os.getenv("AIC_KIS_STORYBOARD_ENABLED", "1").strip().lower() \
    not in {"0", "false", "no", "off"}
KIS_STORYBOARD_HITS_PER_SCENE = int(
    os.getenv("AIC_KIS_STORYBOARD_HITS_PER_SCENE", "8")
)
KIS_STORYBOARD_BEAM_SIZE = int(os.getenv("AIC_KIS_STORYBOARD_BEAM_SIZE", "32"))
KIS_STORYBOARD_MIN_SCENE_SCORE = float(
    os.getenv("AIC_KIS_STORYBOARD_MIN_SCENE_SCORE", "0.25")
)
KIS_STORYBOARD_WEIGHT = float(os.getenv("AIC_KIS_STORYBOARD_WEIGHT", "0.55"))
KIS_STORYBOARD_ORDER_SLACK = int(
    os.getenv("AIC_KIS_STORYBOARD_ORDER_SLACK", "2")
)
KIS_LONG_QUERY_CANDIDATES = int(
    os.getenv("AIC_KIS_LONG_QUERY_CANDIDATES", "800")
)
TRAKE_MAX_GAP_SECONDS = float(os.getenv("AIC_TRAKE_MAX_GAP_SECONDS", "300"))
TRAKE_GAP_PENALTY_WEIGHT = float(
    os.getenv("AIC_TRAKE_GAP_PENALTY_WEIGHT", "0.15")
)

# OCR is built offline with EasyOCR's CRAFT detector and latin_g2 recognizer.
# Unlike PP-OCRv6's current recognition dictionary, latin_g2 contains the full
# precomposed Vietnamese alphabet. The builder validates this invariant before it
# accepts any cache entry.
OCR_DEVICE = os.getenv("AIC_OCR_DEVICE", "cuda:0")
OCR_DETECTION_MODEL_NAME = os.getenv("AIC_OCR_DET_MODEL", "craft")
OCR_RECOGNITION_MODEL_NAME = os.getenv("AIC_OCR_REC_MODEL", "latin_g2")
OCR_LANGUAGES = tuple(
    language.strip()
    for language in os.getenv("AIC_OCR_LANGUAGES", "vi,en").split(",")
    if language.strip()
)
OCR_RECOGNITION_BATCH_SIZE = int(os.getenv("AIC_OCR_REC_BATCH_SIZE", "16"))
OCR_INPUT_BATCH_SIZE = int(os.getenv("AIC_OCR_INPUT_BATCH_SIZE", "8"))
OCR_MAX_SIDE = int(os.getenv("AIC_OCR_MAX_SIDE", "2560"))
OCR_MAGNIFICATION = float(os.getenv("AIC_OCR_MAGNIFICATION", "1.0"))
OCR_MIN_TEXT_SIZE = int(os.getenv("AIC_OCR_MIN_TEXT_SIZE", "12"))
OCR_TEXT_THRESHOLD = float(os.getenv("AIC_OCR_TEXT_THRESHOLD", "0.65"))
OCR_LOW_TEXT = float(os.getenv("AIC_OCR_LOW_TEXT", "0.35"))
OCR_LINK_THRESHOLD = float(os.getenv("AIC_OCR_LINK_THRESHOLD", "0.4"))
OCR_MIN_CONFIDENCE = float(os.getenv("AIC_OCR_MIN_CONFIDENCE", "0.05"))
OCR_CHECKPOINT_EVERY = int(os.getenv("AIC_OCR_CHECKPOINT_EVERY", "500"))
OCR_MAX_RETRIES = int(os.getenv("AIC_OCR_MAX_RETRIES", "3"))

TOP_K_CANDIDATES = 400       # default candidates pulled per prompt from FAISS
MAX_SUBMISSION_ROWS = 100    # contest rule: max 100 answers per query file
MAX_ANSWER_LENGTH = 100      # contest rule: Q&A answer max length in characters
