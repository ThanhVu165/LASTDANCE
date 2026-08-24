# LASTDANCE — Baseline Hợp Nhất (AIC 2026)

> Tài liệu này là **nguồn chuẩn duy nhất** cho việc code. Mọi module code (offline/online)
> phải tuân theo đúng schema, interface và luồng xử lý mô tả ở đây. Nếu code và tài liệu
> lệch nhau, sửa code theo tài liệu, không sửa ngược lại — trừ khi có quyết định mới được
> ghi bổ sung vào mục "Changelog" ở cuối file.

**Cập nhật:** 23/08/2026
**Thời gian còn lại:** 6 ngày
**Máy tham chiếu:** Intel i5-12450H, RTX 4050 Laptop 6 GiB VRAM, Windows/Kaggle/Colab, Python 3.11

---

## 0. Nguyên tắc bắt buộc (không được vi phạm)

1. **Không average nhiều keyframe thành một vector đại diện cho shot/video.** Mỗi keyframe
   có vector riêng. Video được biểu diễn bằng tập nhiều evidence unit, không phải một vector
   duy nhất.
2. **Không dùng `local_idx` làm khóa nộp bài hoặc khóa dedup.** Khóa duy nhất luôn là
   `(video_id, frame_id)`.
3. **Không suy đoán mapping thời gian/frame.** Mọi tra cứu timestamp/frame phải đi qua
   `frames.csv`.
4. **Không hardcode đường dẫn tuyệt đối.** Toàn bộ path build từ biến môi trường `AIC_DATA`.
5. **Không để bất kỳ bước bắt buộc nào của pipeline phụ thuộc 100% vào internet.** Mọi lời
   gọi API cloud (Gemini, v.v.) phải có đường lùi (fallback) chạy local, trừ khi thể lệ xác
   nhận phòng thi có internet ổn định (xem mục 7 — Rủi ro treo).
6. **Không giả định GPU/VRAM vô hạn.** Mọi model local phải ghi rõ VRAM ước tính và cơ chế
   load/release theo pha nếu cần.
7. **Không tự ý đổi FPS, resolution, hoặc thời lượng video là hằng số** — các giá trị này
   phải đọc ra từ bước EDA/inventory, không giả định.
8. **Không cộng trực tiếp điểm CLIP/SigLIP/BEiT-3 vào chung công thức Min-Max liên-modal.**
   3 FAISS index là 3 nguồn điểm riêng, phải gộp về **1** điểm `score_visual` duy nhất trước
   (xem `§3.2` tầng 1) rồi mới đưa vào Late Fusion liên-modal (`§3.2` tầng 2) — nhầm bước này
   sẽ làm sai trọng số `modality_weights["visual"]`.

---

## 1. Data Contract — `frames.csv`

Catalog trung tâm duy nhất. Mọi kết quả từ FAISS, SQLite FTS5, ASR đều phải map ngược qua
file này để lấy timestamp/frame chuẩn.

| Trường | Kiểu | Vai trò |
|---|---|---|
| `video_id` | str | Định danh video |
| `local_idx` | int | Vị trí keyframe nội bộ — chỉ dùng để đọc file JPG/OCR cache, **không** dùng làm khóa nộp bài |
| `frame_id` | int | Chỉ số frame thật trong MP4 — dùng cho preview và submission |
| `pts_time` | float | Timestamp chuẩn (giây) — trục join giữa các modality |
| `shot_id` | str | Định danh shot (từ AutoShot) |
| `window_id` | str \| null | Định danh video-window (nếu dùng window-based retrieval) |
| `keyframe_uid` | int | Khóa deterministic (hash), dùng chung cho FAISS/OCR/ASR — xem `OFFLINE_INDEXING_SPEC.md` mục 3.2a |

```python
# shared/schemas/frame.py
from pydantic import BaseModel

class FrameRecord(BaseModel):
    video_id: str
    local_idx: int
    frame_id: int
    pts_time: float
    shot_id: str
    window_id: str | None = None
    keyframe_uid: int
```

Path convention: mọi file dữ liệu build từ `os.environ["AIC_DATA"]` (default: `data/`),
không hardcode path tuyệt đối trong bất kỳ file JSON/CSV nào.

---

## 2. Offline Pipeline (Nhánh 1)

```
video (.mp4)
  └─> [1] Inventory & EDA (đọc thật FPS/resolution/duration, không giả định)
  └─> [2] Shot Boundary Detection (CPU reference; Windows/Colab CUDA sau parity gate)
  └─> [3] Trích 3 keyframe/shot (Begin - Middle - End)
  └─> [4] Hậu lọc:
          - Laplacian Variance -> loại khung mờ
          - pHash / Cosine similarity > 0.9 -> loại khung gần trùng
  └─> [5] Dual-embedding cho MỖI keyframe (không mean-pool):
          - CLIP ViT-B/32 (baseline / rollback)
          - SigLIP (câu mô tả dài)
          - BEiT-3 (chi tiết vùng ảnh nhỏ)
          -> chuẩn hóa L2
  └─> [6] OCR mỗi keyframe:
          - Primary: Gemini 2.5 Flash-Lite API (JSON prompt)
          - Fallback offline: EasyOCR (CRAFT + latin_g2, đã hỗ trợ tiếng Việt)
  └─> [7] Ghi vào frames.csv + FAISS index + SQLite FTS5

[Nhánh 3 — song song, người phụ trách riêng, tài khoản Kaggle/Colab riêng]
audio (.wav tách từ video)
  └─> [8] ASR: Whisper Large-v3 / phoWhisper (tiếng Việt) — xem chi tiết ASR_SPEC.md
  └─> [9] Temporal Alignment: map segment → keyframe_uid gần nhất theo pts_time
  └─> [10] Ghi vào asr.sqlite (SQLite FTS5) — cùng cấu trúc với ocr.sqlite
```

### 2.1 Bước chi tiết

| # | Bước | Input | Output | Thư viện/model | Ghi chú |
|---|---|---|---|---|---|
| 1 | Inventory | video thô | bảng FPS/res/duration thật | `ffprobe` | Không giả định giá trị |
| 2 | Shot detection | video | manifest v2 gồm shot + transition range | AutoShot / TransNetV2 port tạm thời | CPU reference; Windows/Colab CUDA chỉ sau parity 100% trên dev-subset |
| 3 | Keyframe extraction | shot list | ảnh keyframe (jpg) | `ffmpeg` | 3/shot |
| 4 | Dedup/quality | keyframe | keyframe đã lọc | OpenCV (Laplacian), pHash | Threshold cần benchmark trên dev subset |
| 5 | Visual embedding | keyframe | vector CLIP + SigLIP + BEiT-3 | HuggingFace transformers | Batch processing, chạy trên Kaggle GPU |
| 6 | OCR | keyframe | text + bbox | Gemini API / EasyOCR | JSON prompt schema cố định — xem 2.2 |
| 7 | Indexing | vector + text | FAISS index, SQLite FTS5, frames.csv | `faiss-cpu`, `sqlite3` | IndexFlatIP, chuẩn hóa L2 trước khi add |
| 8 | ASR (nhánh 3, song song) | audio `.wav` | segment + timestamp | Whisper Large-v3 / phoWhisper | Xem `ASR_SPEC.md`, chạy trên Kaggle/Colab account riêng, không tranh quota với nhánh 1 |

### 2.2 JSON schema OCR prompt (Gemini) — schema chuẩn, dùng chung với `OFFLINE_INDEXING_SPEC.md`

```python
class OcrResult(BaseModel):
    frame_id: int
    detected_text: list[str]
    bbox: list[list[float]]  # bounding box tương ứng từng phần tử detected_text
    confidence: float
    language: str  # "vi" | "en" | "mixed"
```

### 2.3 Điều kiện publish index (bắt buộc — xem chi tiết đầy đủ ở `OFFLINE_INDEXING_SPEC.md` §5)

Một index chỉ được coi là production-ready khi:
- `complete = true` (tính theo từng `video_id`, không phải toàn catalog)
- tập `keyframe_uid` trong `frames.csv` khớp 100% với tập ID đã add vào **cả 3** file FAISS
  (diff bằng code, **không** đếm số dòng thô)
- không có NaN/Inf trong vector
- norm ≈ 1 (đã chuẩn hóa L2 đúng)
- mapping video/frame/timestamp khớp `frames.csv`
- checkpoint/resume hoạt động (để resume nếu Kaggle timeout giữa chừng)

File đang preallocate hoặc checkpoint dở dang **không** được set `complete=true`.

---

## 3. Online Pipeline (Nhánh 2)

```
[Query text] ──> QueryPlanner (xem interface 3.1)
                       │
        ┌──────────────┼───────────────────────────┐
        ▼ (Visual)      │                            │
  clip.faiss  siglip.faiss  beit3.faiss              │
        │            └──────┬──────┘                 │
        │ (rollback     SRRF (siglip+beit3)           │
        │  nếu thiếu)        │                        │
        └──────────┬─────────┘                        │
              score_visual (1 điểm/keyframe, §3.2 tầng 1)
                   │          ▼ (OCR)         ▼ (ASR — spoken_text)
                   │      SQLite FTS5    SQLite FTS5 (asr.sqlite)
                   │      (ocr.sqlite)   (BM25 trên transcribed_text)
                   └──────────────┼───────────────┘
                                  ▼
          Late Fusion tầng 2: Min-Max Normalization
                  + trọng số thích ứng (§3.2 tầng 2)
                         ▼
                   Top 100 candidates
                         ▼
          Neighbors-Based Reranking (CPU)
                         ▼
        ┌────────────────┴────────────────┐
        ▼ (TRAKE)                          ▼ (QA)
  Beam Search +                    Multiplicative Gating
  Exponential Decay                 (threshold 0.85)
  (λ=0.01, beam=8)                        │
        └────────────────┬────────────────┘
                         ▼
              frames.csv lookup (chuẩn hóa mốc thời gian)
                         ▼
              Dedup theo (video_id, frame_id)
                         ▼
                  Top 100 submission
```

### 3.1 Interface `QueryPlanner` — bắt buộc dùng chung cho mọi implementation

```python
# shared/interfaces/query_planner.py
from abc import ABC, abstractmethod
from pydantic import BaseModel

class UnifiedQueryPlan(BaseModel):
    raw_query: str
    caption_en: str
    scenes: list[str]
    must_have: list[str]
    should_have: list[str]
    negative_constraints: list[str]
    visible_text: list[str]      # clue cho OCR
    spoken_text: list[str]       # clue cho ASR
    modality_weights: dict[str, float]  # {"visual": 0.6, "ocr": 0.4, ...}
    question: str | None = None       # cho QA
    answer_format: str | None = None  # cho QA
    ordered_moments: list[str] | None = None  # cho TRAKE

class QueryPlanner(ABC):
    @abstractmethod
    def plan(self, text: str, task_type: str) -> UnifiedQueryPlan:
        ...
```

**3 lớp implementation, theo đúng thứ tự ưu tiên:**

| Lớp | Class | Điều kiện dùng | VRAM |
|---|---|---|---|
| Primary | `GeminiQueryPlanner` | Có internet | 0 MB (Cloud API) |
| Fallback 1 | `QwenLocalQueryPlanner` | Mất mạng, máy đủ VRAM | ~1.8 GB (Qwen3-VL-2B, 4-bit quantized) |
| Fallback 2 | `RuleBasedQueryPlanner` | Cả API và local model đều lỗi | 0 |

Fallback 2 = tạo 1 scene duy nhất chứa nguyên văn query, `modality_weights` mặc định
`{"visual": 1.0}`. Không được throw exception ra ngoài — pipeline luôn phải trả về một
`UnifiedQueryPlan` hợp lệ.

**Quy tắc code:** không viết `if internet_available: call_gemini() else: ...` rải rác trong
business logic. Chỉ một nơi duy nhất (factory function `get_query_planner()`) quyết định
implementation nào được dùng, có cơ chế retry + timeout + circuit breaker để tự động rơi
xuống lớp tiếp theo.

### 3.2 Retrieval & Fusion

Fusion chạy **2 tầng riêng biệt** — không được gộp chung thành 1 hàm/1 công thức. Lý do:
visual có **3 index độc lập** (`clip.faiss`, `siglip.faiss`, `beit3.faiss`), trong khi OCR và
ASR mỗi kênh chỉ có 1 nguồn điểm. Nếu code gộp cả 5 nguồn điểm (clip, siglip, beit3, ocr, asr)
vào chung 1 Min-Max, `modality_weights["visual"]` sẽ bị pha loãng sai vì visual chiếm 3/5 vote
thay vì đúng 1 vote như OCR/ASR.

**Tầng 1 — Intra-visual fusion (gộp 3 embedding thành 1 điểm `score_visual`):**

- Search song song cả 3 FAISS index cho mỗi query.
- Gộp điểm SigLIP + BEiT-3 bằng **Score-Reflected Reciprocal Rank Fusion (SRRF)** — giữ được
  phân phối điểm tương đồng thực tế, không chỉ dựa vào rank thuần như RRF gốc.
- **CLIP không tham gia SRRF.** CLIP giữ vai trò rollback: chỉ dùng làm `score_visual` chính
  cho keyframe nào mà `siglip.faiss` hoặc `beit3.faiss` **chưa Ready** (theo Publishing
  Criteria, `OFFLINE_INDEXING_SPEC.md` §5). Khi cả 3 index đã Ready cho video đó, `score_visual`
  = kết quả SRRF(SigLIP, BEiT-3); điểm CLIP chỉ log lại để so sánh/tie-break, không cộng thêm
  vào công thức.
- Output tầng 1: đúng **1** con số `score_visual`/keyframe — đây là input duy nhất của tầng 2.

**Tầng 2 — Inter-modal fusion (visual/ocr/asr):**

- Visual: `score_visual` từ tầng 1 (frame-level, không phải shot-level mean-pooled).
- OCR: SQLite FTS5 (`ocr.sqlite`), BM25, hỗ trợ exact phrase / full-term / partial / fuzzy match.
- ASR: SQLite FTS5 (`asr.sqlite`), BM25 trên `transcribed_text`, chỉ kích hoạt trọng số cao khi
  `UnifiedQueryPlan.spoken_text` không rỗng (câu hỏi có nhắc lời thoại/nhân vật nói).
- Fusion: Min-Max Normalization mỗi kênh về `[0,1]`, nhân với `modality_weights` từ
  `UnifiedQueryPlan` (3 khóa: `visual`, `ocr`, `asr`).

```
Score_normalized = (Score - Score_min) / (Score_max - Score_min)
Score_final = Σ (w_channel × Score_normalized_channel)
```

**Quy tắc code:** tầng 1 (`fuse_visual_channels()`) và tầng 2 (`fuse_modalities()`) là 2 hàm
riêng biệt trong `online/fusion.py`, gọi tuần tự — không viết gộp thành 1 hàm, để người đọc
code sau này không nhầm "visual" là 1 index khi thực ra là 3.

### 3.3 Reranking

- **Bắt buộc trong baseline:** Neighbors-Based Reranking (CPU, dựa trên stable local
  neighborhoods trong không gian đặc trưng).
- **Stretch goal, không bắt buộc:** BLIP-2 ITM qua Cloud GPU (RunPod). Chỉ triển khai nếu
  còn dư thời gian sau khi toàn bộ pipeline chính chạy ổn định. Không đưa vào critical path.

### 3.4 TRAKE

Beam Search + Exponential Decay gap penalty:

```
λ_i = e^(-decay × Δt_i),  decay = 0.01,  beam_width = 8
S_final_i = s_i × λ_i × b_i
```

### 3.5 QA — Multiplicative Gating

Gating có 2 lớp: ngưỡng similarity (**bắt buộc**, luôn chạy) + xác thực BLIP-2 (**tùy chọn**,
graceful degradation) — không chặn critical path nếu BLIP-2 chưa build kịp (§3.3 vẫn coi
BLIP-2 là stretch goal), nhưng khi model đã sẵn sàng thì bắt buộc dùng để giảm false positive
similarity (case kiểu BOGOTA/MEDELLIN: điểm similarity cao nhưng nội dung không thực sự khớp).

```python
SIMILARITY_THRESHOLD = 0.85

def blip2_verify(evidence, question) -> float:
    """Trả 1.0 nếu BLIP-2 ITM chưa build/chưa load (graceful degradation — KHÔNG throw
    exception). Trả điểm ITM thực (0-1) nếu model đã sẵn sàng. Giá trị 1.0 chỉ là default
    khi thiếu model, không phải kết quả xác thực thật."""
    ...

b_i = blip2_verify(best_evidence, query.question)
S_final = best_evidence.score_visual * b_i

if S_final > SIMILARITY_THRESHOLD:
    answer = qwen_vqa.generate(best_evidence)
else:
    answer = "Uncertain"  # chặn lỗi copy thực thể từ query (case BOGOTA/MEDELLIN)
```

**Quy tắc:** `blip2_verify()` không được để pipeline QA phụ thuộc cứng vào việc BLIP-2 tồn
tại — thiếu model thì log warning và trả 1.0, không crash. Interface đã có sẵn chỗ cắm `b_i`
ngay khi model sẵn sàng, không phải sửa lại signature giữa chừng thi.

### 3.6 Submission

- Dedup theo `(video_id, frame_id)` — không dùng `local_idx`.
- Validator bắt buộc kiểm tra đúng số dòng quy định (tối đa 100) trước khi xuất file nộp.
- Whitespace/empty query phải trả HTTP 422, không được để lọt thành 500.

---

## 4. Model & Hardware Budget

| Vai trò | Model | Môi trường | VRAM ước tính | Ghi chú |
|---|---|---|---|---|
| Query planning (primary) | Gemini 2.5 Flash-Lite | Cloud API | 0 MB | Cần internet |
| Query planning (fallback) | Qwen3-VL-2B-Instruct | Local GPU, 4-bit | ~1.8 GB | Load/release theo pha |
| Shot detection (offline) | AutoShot / TransNetV2 port tạm thời | Local CPU, Windows NVIDIA GPU hoặc Colab T4 | Ghi theo batch report | CUDA chọn tường minh, không fallback; phải qua parity gate CPU–CUDA |
| Frame recall baseline | CLIP ViT-B/32 | Local/Kaggle | ~300 MB | Rollback an toàn |
| Frame recall nâng cao | SigLIP + BEiT-3 | Kaggle GPU (offline) | N/A (chạy batch, không online) | Build index nền |
| OCR (primary) | Gemini 2.5 Flash-Lite | Cloud API | 0 MB | |
| OCR (fallback) | EasyOCR (CRAFT + latin_g2) | Local CPU | thấp | Đã xác nhận hỗ trợ tiếng Việt |
| Reranking | Neighbors-Based (thuật toán, không phải model) | Local CPU | 0 MB | |
| VQA / answer generation | Qwen3-VL-2B-Instruct | Local GPU, 4-bit | ~1.8 GB | Chỉ chạy khi qua Multiplicative Gating |
| ASR (nhánh 3, song song) | Whisper Large-v3 / phoWhisper | Kaggle/Colab GPU riêng (offline) | N/A (chạy batch, không online) | Xem `ASR_SPEC.md`; human-in-the-loop vẫn giữ làm backup nếu nhánh 3 không kịp |
| Reranking + QA verification (tùy chọn) | BLIP-2 ITM | Cloud GPU (RunPod) | N/A (cloud) | Stretch goal — dùng cho `b_i` trong QA gating (§3.5) và reranking (§3.3); graceful degradation nếu chưa sẵn sàng (`b_i = 1.0`), không chặn critical path |

**Lưu ý vận hành:** tắt các tiến trình ngầm chiếm VRAM (LM Studio, Epic Games Launcher...)
trước khi chạy inference local. Không chạy đồng thời Qwen planner + Qwen VQA nếu tổng VRAM
vượt 6 GB — load/release model theo pha.

---

## 5. Cấu trúc thư mục repo

```
app/        # FastAPI backend
offline/    # Pipeline tiền xử lý (shot detection, embedding, OCR, indexing)
online/     # Query planning, retrieval, fusion, reranking, submission
shared/     # Pydantic schemas, interfaces (QueryPlanner, FrameRecord), constants
scripts/    # Chạy batch Kaggle/Colab, tách nhỏ danh sách video song song
```

Quy tắc: Pydantic schema định nghĩa trong `shared/`, dùng chung giữa `offline/` và
`online/` — không định nghĩa lại schema trùng lặp ở hai nơi.

---

## 6. Rủi ro còn treo — PHẢI xác nhận trước khi code phần liên quan

| # | Câu hỏi | Ảnh hưởng nếu không xác nhận trước |
|---|---|---|
| 1 | Thể lệ AIC 2026 có cho phép internet trong phòng thi không? | Quyết định Gemini là primary hay optional; RunPod BLIP-2 có đáng làm không |
| 2 | Index frame-level (đã chốt) có đủ nhanh trên CPU khi dataset đầy đủ (kể cả batch 2) không? | Cần benchmark thật trên dev subset trước khi build full 177k+ keyframe |
| 3 | EasyOCR chạy full collection có kịp trong 6 ngày không (đã từng chưa hoàn thành ở bản cũ)? | Cần ưu tiên Gemini OCR trước, EasyOCR chạy nền song song làm backup |

Không viết code phần liên quan đến các mục trên cho tới khi có câu trả lời — tránh việc
phải viết lại toàn bộ khi câu trả lời ngược với giả định ban đầu.

---

## 7. Checklist triển khai theo thứ tự ưu tiên

1. Khóa schema `frames.csv` + `UnifiedQueryPlan` (mục 1, 3.1) — làm trước tiên, cả hai nhánh
   dùng chung.
2. Fix lỗi contract độc lập: whitespace → 422, dedup theo `(video_id, frame_id)`, validator
   đúng số dòng.
3. Build `QueryPlanner` interface + 3 lớp fallback (mục 3.1) trước khi code bất kỳ logic
   retrieval nào phụ thuộc vào nó.
4. Chạy AutoShot + dual-embedding trên dev subset nhỏ để benchmark tốc độ/VRAM trước khi
   chạy full 873 video.
5. Build FAISS + SQLite FTS5 index frame-level.
6. Nối luồng online: intra-visual fusion (SRRF, §3.2 tầng 1) → inter-modal fusion (Min-Max,
   §3.2 tầng 2) → reranking → TRAKE/QA logic.
7. A/B CLIP-only vs CLIP+SigLIP+BEiT-3 trên dev subset có ground truth, đo Recall@k thật —
   không kết luận "tốt hơn" nếu chưa đo.
8. Nhánh 3 (ASR): audio extraction → Whisper/phoWhisper → temporal alignment → `asr.sqlite`,
   chạy độc lập trên Kaggle/Colab account riêng — xem `ASR_SPEC.md`.
9. Nếu còn dư thời gian: build BLIP-2 Cloud (RunPod) như stretch goal — dùng chung cho cả
   reranking (§3.3) và QA verification (`b_i`, §3.5). Không chặn release chính; interface
   gating đã có graceful degradation sẵn nên có thể cắm vào bất kỳ lúc nào trong 6 ngày.

---

## Changelog

- **23/08/2026** — Chốt baseline hợp nhất: bỏ mean-pooling shot-level làm index chính, bỏ
  RunPod/BLIP-2 khỏi critical path, đổi OCR fallback từ PARSeq-Ti sang EasyOCR, thêm 3 lớp
  fallback bắt buộc cho `QueryPlanner`.
- **23/08/2026 (bản 2)** — Đồng bộ `frames.csv` sang khóa `keyframe_uid` (thay `faiss_row_id`
  positional, khớp `OFFLINE_INDEXING_SPEC.md`). Chuyển ASR từ "hoãn hoàn toàn" sang **Nhánh 3
  song song** (người phụ trách riêng, tài khoản Kaggle/Colab riêng, không tranh quota với
  Nhánh 1) — thêm kênh ASR vào fusion, `modality_weights` có 3 khóa. Tách chi tiết ASR ra
  `ASR_SPEC.md`. Human-in-the-loop giữ làm backup nếu Nhánh 3 không kịp tiến độ.
- **23/08/2026 (bản 3)** — Sửa xung đột phát hiện khi audit chéo 3 file: §2.3 đổi tiêu chí
  publish sang diff `keyframe_uid` (khớp `OFFLINE_INDEXING_SPEC.md`, bỏ kiểu đếm dòng thô cũ);
  đồng bộ schema `OcrResult` (thêm `bbox`, đổi `text`→`detected_text`, `conf`→`confidence`)
  khớp với `OFFLINE_INDEXING_SPEC.md`.
- **23/08/2026 (bản 4)** — Sau khi đối chiếu với 2 baseline tham khảo (FiftyOne BTC/
  `lducc-hcm-aic`, và 1 doc SOTA offline+online khác): (1) Thêm nguyên tắc bắt buộc §0.8 +
  tầng 1 "intra-visual fusion" vào §3.2 — gộp SigLIP+BEiT-3 bằng SRRF thành 1 `score_visual`
  duy nhất trước khi vào Min-Max liên-modal (CLIP giữ vai trò rollback khi thiếu index, không
  tham gia SRRF); trước bản này §3.2 coi "visual" là 1 điểm dù thực tế có 3 FAISS index riêng
  — rủi ro làm sai `modality_weights["visual"]` nếu nhiều người code song song không thống
  nhất cách hiểu. (2) Thêm hệ số `b_i` (BLIP-2 verification) vào công thức QA gating §3.5,
  có graceful degradation (`b_i = 1.0` khi BLIP-2 chưa sẵn sàng) — BLIP-2 vẫn là stretch goal
  không chặn critical path, nhưng interface đã sẵn chỗ cắm. (3) **Giữ nguyên** `keyframe_uid`
  (không đổi lại `faiss_row_id` positional dù baseline tham khảo thứ 2 dùng cách đó) — đã đối
  chiếu và xác nhận lý do bỏ `faiss_row_id` ở bản 2 vẫn đúng, không đảo ngược.
- **24/08/2026 (bản 5)** — Cho phép chia Shot Detection TransNetV2 sang Colab CUDA để rút
  ngắn batch 873 video. CPU vẫn là default; CUDA phải dùng cùng commit/config/weight, ghi
  device/runtime provenance, fail closed khi CUDA thiếu và chỉ được chạy production sau khi
  từng shot boundary + `excluded_transition_ranges` khớp 100% với reference CPU trên đủ 5
  video dev-subset. Không copy/reimplement hậu xử lý trong notebook.
- **24/08/2026 (bản 6)** — Chuyển worker Shot Detection của đồng đội từ CPU sang NVIDIA GPU
  trên Windows để chạy batch dài. Dùng environment CUDA tách biệt, CPU chỉ làm reference;
  vẫn bắt buộc parity 5/5, provenance và fail closed như Colab. Không overwrite environment
  CPU hoặc tự fallback khi driver/CUDA/VRAM không đáp ứng.
