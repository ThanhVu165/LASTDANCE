# LASTDANCE — Baseline Hợp Nhất (AIC 2026)

> Đây là **nguồn chuẩn kỹ thuật duy nhất** cho Offline, ASR và Online. Runbook/status chỉ mô
> tả cách vận hành hoặc tiến độ, không được định nghĩa schema/contract khác file này. Nếu code
> và tài liệu lệch nhau, sửa code theo tài liệu, trừ khi quyết định mới của người dùng được
> ghi vào Changelog. Hai spec tách nhánh cũ đã được hợp nhất và xóa để tránh lệch phiên bản.

**Cập nhật:** 26/08/2026
**Thời gian còn lại:** 6 ngày
**Máy tham chiếu local/Shot:** Intel i5-12450H, RTX 4050 Laptop 6 GiB VRAM,
Windows/Colab, Python 3.11. **Visual Embedding Kaggle GPU:** Python 3.12.x; giữ PyTorch CUDA
khớp image Kaggle thay vì cài đè wheel chung.

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
8. **Không cộng trực tiếp điểm CLIP/SigLIP/EVA-CLIP vào chung công thức Min-Max liên-modal.**
   3 FAISS index là 3 nguồn điểm riêng, phải gộp về **1** điểm `score_visual` duy nhất trước
   (xem `§3.2` tầng 1) rồi mới đưa vào Late Fusion liên-modal (`§3.2` tầng 2) — nhầm bước này
   sẽ làm sai trọng số `modality_weights["visual"]`.
9. **Inventory bằng `ffprobe` là bước bắt buộc trước Shot Detection full collection.**
   Chạy lại khi thêm/thay MP4 hoặc khi inventory thiếu/stale; không cần chạy lại nếu tập nguồn
   không đổi và `inventory.json` còn hợp lệ. Không publish inventory chạy với `--limit`.

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
| `shot_id` | str | Định danh shot (từ TransNetV2) |
| `window_id` | str \| null | Định danh video-window (nếu dùng window-based retrieval) |
| `keyframe_uid` | int | Khóa deterministic BLAKE2b, dùng chung cho FAISS/OCR/ASR — xem §2.1d |

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
  └─> [5] Ba visual embedding cho MỖI keyframe (không mean-pool):
          - CLIP ViT-B/32 (baseline / rollback)
          - SigLIP (câu mô tả dài)
          - EVA-CLIP (visual encoder bổ sung)
          -> chuẩn hóa L2
  └─> [6] OCR mỗi keyframe:
          - Primary: Gemini 2.5 Flash-Lite API (JSON prompt)
          - Fallback offline: EasyOCR (CRAFT + latin_g2, đã hỗ trợ tiếng Việt)
  └─> [7] Ghi vào frames.csv + FAISS index + SQLite FTS5

[Nhánh 3 — song song, người phụ trách riêng, tài khoản Kaggle/Colab riêng]
audio (.wav tách từ video)
  └─> [8] ASR: Whisper Large-v3 / phoWhisper (tiếng Việt) — xem §2A
  └─> [9] Temporal Alignment: map segment → keyframe_uid gần nhất theo pts_time
  └─> [10] Ghi vào asr.sqlite (SQLite FTS5) — cùng cấu trúc với ocr.sqlite
```

### 2.1 Bước chi tiết

| # | Bước | Input | Output | Thư viện/model | Ghi chú |
|---|---|---|---|---|---|
| 1 | Inventory | video thô | bảng FPS/res/duration thật | `ffprobe` | Không giả định giá trị |
| 2 | Shot detection | video | manifest v2 gồm shot + transition range | TransNetV2 (`transnetv2-pytorch==1.0.5`) | GPU production sau parity 100% trên dev-subset; CPU reference/fallback |
| 3 | Keyframe extraction | shot list | ảnh keyframe (jpg) | `ffmpeg` | 3/shot |
| 4 | Dedup/quality | keyframe | keyframe đã lọc | OpenCV (Laplacian), pHash | Threshold cần benchmark trên dev subset |
| 5 | Visual embedding | keyframe | vector CLIP + SigLIP + EVA-CLIP | Transformers + OpenCLIP | Batch processing, chạy trên Kaggle GPU |
| 6 | OCR | keyframe | text + bbox | Gemini API / EasyOCR | JSON prompt schema cố định — xem 2.2 |
| 7 | Indexing | vector + text | FAISS index, SQLite FTS5, frames.csv | `faiss-cpu`, `sqlite3` | IndexFlatIP, chuẩn hóa L2 trước khi add |
| 8 | ASR (nhánh 3, song song) | audio `.wav` | segment + timestamp | Whisper Large-v3 / phoWhisper | Xem §2A; chạy trên Kaggle/Colab account riêng, không tranh quota với nhánh 1 |
### 2.1a Inventory/EDA bằng `ffprobe` — bắt buộc

Inventory là bản kiểm kê metadata nguồn, không phải index và không sinh keyframe. Bước này
chạy **local CPU**, không dùng quota GPU.

- Input: video được tìm đệ quy dưới `AIC_DATA/videos/`.
- Output production: `AIC_DATA/index/inventory.json`, ghi atomic, `schema_version = 1`.
- Mỗi record: `video_id`, `relative_path`, `width`, `height`, `fps`, `duration`,
  `frame_count` (có thể `null` nếu container không khai báo) và `has_audio`.
- `fps`, resolution và duration phải đọc từ stream/format thật; thiếu video stream, FPS,
  duration hoặc resolution hợp lệ thì fail closed.
- `relative_path` phải nằm dưới `AIC_DATA`; `video_id` không được trùng.

Lệnh production, chạy **không có `--limit`**:

```powershell
$env:AIC_DATA = "D:\AIC2026"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_inventory
```

Chạy Inventory một lần trước Shot Detection full collection. Chạy lại nếu thêm, xóa, thay
nội dung/metadata hoặc đổi tên MP4; không cần chạy lại khi tập nguồn không đổi và
`inventory.json` còn đúng. `--limit` chỉ dùng smoke với `--output` riêng; nếu dùng
`--limit` cùng output mặc định, file tạo ra chỉ là inventory một phần và **không được
publish**.

Schema rút gọn:

```json
{
  "schema_version": 1,
  "videos": [
    {
      "video_id": "L21_V001",
      "relative_path": "videos/L21_V001.mp4",
      "width": 1280,
      "height": 720,
      "fps": 25.0,
      "duration": 1513.96,
      "frame_count": 37849,
      "has_audio": true
    }
  ]
}
```

### 2.1b Shot Detection — TransNetV2 production

- Input: `AIC_DATA/videos/<video_id>.mp4`; Inventory của collection phải có trước.
- Model: `transnetv2-pytorch==1.0.5`, threshold `0.5`, weight SHA-256
  `a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de`.
- Windows NVIDIA GPU là worker production sau parity 5/5; CPU là reference/fallback; Colab
  CUDA là worker phụ. CUDA phải chọn tường minh và không được fallback âm thầm.
- Output: `AIC_DATA/shots/<video_id>.json`, manifest schema v2, atomic publish.
- Provenance bắt buộc: implementation, package version, device, threshold, weight source/hash.
- Checkpoint riêng theo worker/device/output namespace: ghi `0/1` trước inference, chỉ ghi
  `1/1` sau atomic publish và validation. Ngắt giữa inference chạy lại video hiện tại;
  manifest hợp lệ đã publish được adopt; complete state thiếu/hỏng manifest phải fail closed.

Manifest phải chứa shot tăng dần, không overlap và mọi frame transition nằm ngoài shot:

```json
{
  "schema_version": 2,
  "video_id": "L21_V001",
  "relative_video_path": "videos/L21_V001.mp4",
  "detector": "transnetv2",
  "detector_signature": {
    "implementation": "transnetv2-pytorch",
    "package_version": "1.0.5",
    "device": "cuda",
    "threshold": 0.5,
    "weights_sha256": "a313d0..."
  },
  "shots": [
    {"shot_id": "s000000", "start_frame": 0, "end_frame": 48}
  ],
  "excluded_transition_ranges": [
    {
      "start_frame": 49,
      "end_frame": 49,
      "reason": "transition_score_above_threshold"
    }
  ],
  "transition_exclusion_validation": {
    "total_frame_count": 31720,
    "excluded_frame_count": 1,
    "excluded_frame_fraction": 0.0000315,
    "warning_threshold": 0.01,
    "exceeds_warning_threshold": false
  }
}
```

Các range transition phải khớp chính xác phần bù giữa các shot. Accounting/range sai phải
fail closed; tỷ lệ loại vượt 1% chỉ phát warning để kiểm tra threshold. Keyframe planner
không được chọn frame thuộc `excluded_transition_ranges`.

### 2.1c Keyframe extraction và quality/dedup

- Trích tối đa 3 keyframe/shot: Begin, Middle, End bằng FFmpeg và timestamp/frame thật.
- Tên file: `{video_id}/{shot_id}_{local_idx}.jpg`; `local_idx` chỉ định vị JPEG.
- Không dùng `frame_id` để đặt tên file và không dùng `local_idx` làm khóa submission.
- Lọc mờ bằng Laplacian variance; dedup pHash/cosine chỉ trong cùng shot.
- Threshold production phải được benchmark; luôn giữ ít nhất một keyframe/shot.
- Không xóa JPEG nguồn trong bước report/filter; artifact selection phải có signature.
- Checkpoint/resume phải giữ thứ tự canonical của plan, không sort lại theo `frame_id` hay
  `local_idx`.

### 2.1d `keyframe_uid`, embedding và index

`keyframe_uid` là khóa deterministic chung cho `frames.csv`, ba FAISS index, OCR và ASR:

```python
import hashlib

def make_keyframe_uid(video_id: str, shot_id: str, local_idx: int) -> int:
    raw = f"{video_id}:{shot_id}:{local_idx}"
    digest = hashlib.blake2b(raw.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False) >> 1
```

Không dùng row index/`faiss_row_id` làm khóa. CLIP, SigLIP và EVA-CLIP build độc lập bằng
`faiss.IndexIDMap(faiss.IndexFlatIP(dim))`; mỗi keyframe có vector riêng, L2-normalize và
ép `float16` trước khi lưu/push. Nhánh Offline chỉ bàn giao ba index độc lập, không build
index SRRF đã gộp.

OCR được ghi vào SQLite FTS5:

```sql
CREATE VIRTUAL TABLE ocr_fts USING fts5(
    video_id UNINDEXED,
    keyframe_uid UNINDEXED,
    detected_text,
    language UNINDEXED,
    confidence UNINDEXED
);
```

Artifact Nhánh 1:

1. `frames.csv`.
2. `clip.faiss`, `siglip.faiss`, `eva_clip.faiss`.
3. `ocr.sqlite`.
4. Shot/keyframe/quality manifests, checkpoint và publishing state cần cho audit/resume.
5. Mọi path lưu trong artifact là relative dưới `AIC_DATA`.



### 2.2 JSON schema OCR prompt (Gemini) — schema chuẩn

```python
class OcrResult(BaseModel):
    frame_id: int
    detected_text: list[str]
    bbox: list[list[float]]  # bounding box tương ứng từng phần tử detected_text
    confidence: float
    language: str  # "vi" | "en" | "mixed"
```

`OcrResult` giữ nguyên là schema nội dung chuẩn. Artifact trung gian bọc schema này bằng
**OCR record envelope schema v1** để checkpoint/resume và provenance không làm đổi schema
FTS cuối:

- Mỗi keyframe của batch có đúng một dòng JSONL với `video_id`, `keyframe_uid`, `frame_id`,
  relative `source_image`, `execution_mode`, `status`, `engine`, `fallback_used`, `attempts`
  và `result`. `execution_mode=gemini_primary` cho pipeline chính; một job EasyOCR chủ động,
  tách biệt phải ghi `easyocr_offline` và `fallback_used=false`, tránh đánh đồng với failover
  âm thầm do quota.
- `status` là `success | no_text | error`. `success` bắt buộc có `result: OcrResult` và ít
  nhất một text không rỗng; `no_text`/`error` dùng `result=null`, không tạo `language` hay
  confidence giả. Provenance/error chỉ nằm trong envelope/manifest, không thêm cột vào
  `ocr_fts`.
- `bbox` trong envelope chuẩn hóa thành quadrilateral 8 số
  `[x1,y1,x2,y2,x3,y3,x4,y4]`, theo chiều kim đồng hồ từ góc trên-trái trực quan, tọa độ
  `[0,1]` trên ảnh keyframe sau khi đã xoay đúng orientation. EasyOCR quy đổi 4 point sang
  dạng này; Gemini phải trả cùng convention.
- `confidence` Gemini giữ giá trị top-level đã validate từ JSON schema. Adapter EasyOCR
  tính trung bình confidence từng vùng, trọng số bằng số code point không-whitespace của
  text vùng đó; kết quả clamp vào `[0,1]`.
- Coverage terminal yêu cầu toàn bộ UID thuộc đúng một trong ba status. Completion gate chỉ
  PASS khi tập UID exact, không duplicate/foreign/missing và `error == 0`; `no_text` là kết
  quả hợp lệ, không bị diễn giải thành lỗi hoặc ép tạo row FTS.
- Chín batch ghi chín JSONL shard độc lập. Chỉ sau khi union 9 tập UID disjoint và exhaustive
  so với `frames.csv` mới build **một** `ocr.sqlite` cuối ở local; chỉ record `success` được
  nạp vào `ocr_fts`.

Fallback policy fail-closed: `429`/`5xx`/timeout retry exponential backoff + jitter;
`401`/`403` hoặc quota/global project failure dừng batch, không âm thầm đẩy toàn catalog sang
CPU. Chỉ response Gemini sai JSON/schema sau retry mới fallback EasyOCR theo từng keyframe.
Nếu EasyOCR cũng lỗi, envelope ghi `error` và batch không được complete. EasyOCR production
phải khởi tạo với `download_enabled=False` sau khi package/CRAFT/`latin_g2` đã kiểm tra
filename, size và SHA-256 theo registry pin.

### 2.3 Publishing Criteria — điều kiện `complete=true`

`complete` được tính theo từng `video_id`, không phải toàn catalog. Một video chỉ được đánh
dấu `complete=true` khi thỏa **tất cả**:

- [ ] Shot manifest schema v2 hợp lệ và checkpoint Shot Detection ở `1/1`.
- [ ] Tập `keyframe_uid` trong `frames.csv` khớp 100% với ID trong **cả ba** FAISS
      `clip.faiss`, `siglip.faiss`, `eva_clip.faiss` bằng set diff, không đếm dòng thô.
- [ ] Không có `NaN`/`Inf` trong bất kỳ vector nào.
- [ ] Vector đã L2-normalize, norm xấp xỉ 1 trên sample kiểm tra.
- [ ] Mapping `video_id`/`frame_id`/`pts_time` đã sanity-check với video gốc.
- [ ] Checkpoint/resume đã được thử bằng cách ngắt giữa batch rồi chạy lại, không duplicate
      và không mất dữ liệu.

File preallocate, artifact partial hoặc checkpoint dở dang không được set `complete=true`.
Không có cờ Ready chỉnh tay; readiness phải được validator suy ra từ artifact thật.

### 2.4 Đồng bộ artifact Kaggle ↔ local

Dùng HuggingFace Dataset (Git LFS):

```text
Kaggle build --push_to_hub()--> HF Dataset --snapshot_download()--> máy local
```

- Vector bắt buộc `float16` trước khi push.
- Gom 50–100 video hoặc cuối phiên mới push, không push từng video.
- Revision/commit đặt theo batch (`batch-01`, `batch-02`, ...).
- Chỉ tải full snapshot về local một lần trước khi thi; pull lại khi có patch khẩn.
- Không coi upload thành công là Ready nếu Publishing Criteria phía trên chưa PASS.

---

## 2A. ASR Pipeline (Nhánh 3)

Nhánh ASR chạy độc lập trên tài khoản Kaggle/Colab GPU riêng, không tranh quota với Visual
Embedding. Human-in-the-loop vẫn là fallback cho video chưa có coverage.

### 2A.1 Scope

| Việc | Quyết định |
|---|---|
| Tách audio 16 kHz mono bằng FFmpeg | Bắt buộc, local CPU |
| Whisper Large-v3 / phoWhisper + timestamp | Bắt buộc, GPU riêng |
| Temporal alignment về keyframe | Bắt buộc |
| Build `asr.sqlite` FTS5 + coverage report | Bắt buộc |
| BEATs/audio captioning phi ngôn ngữ | Ngoài scope |
| Speaker diarization | Ngoài scope |

### 2A.2 Luồng xử lý

```text
video (.mp4)
  -> FFmpeg: audio 16 kHz mono
  -> Whisper Large-v3 / phoWhisper
  -> segment(start_time, end_time, text, language)
  -> tìm keyframe_uid gần nhất từ frames.csv
  -> asr.sqlite + coverage report
```

Với mỗi segment, chọn keyframe có `pts_time` nằm trong `[start_time, end_time]` và gần
segment nhất; nếu không có keyframe trong khoảng, chọn keyframe gần `start_time` nhất trong
cùng video. Không tự tạo khóa ASR mới để thay `keyframe_uid`.

### 2A.3 Schema `AsrSegment`

```python
class AsrSegment(BaseModel):
    video_id: str
    segment_id: str
    start_time: float
    end_time: float
    transcribed_text: str
    language: Literal["vi", "en"]
    keyframe_uid_nearest: int
```

Invariant: identifier/text không rỗng, timestamp không âm, `end_time >= start_time`, và
`keyframe_uid_nearest` là signed-int64 dương tồn tại trong `frames.csv`.

```sql
CREATE VIRTUAL TABLE asr_fts USING fts5(
    video_id UNINDEXED,
    segment_id UNINDEXED,
    transcribed_text,
    language UNINDEXED,
    keyframe_uid_nearest UNINDEXED,
    start_time UNINDEXED,
    end_time UNINDEXED
);
```

Tên cột nội dung giữ là `transcribed_text`; OCR dùng `detected_text`. Hai bảng phải đủ
song song để cùng một `FtsSearcher` dùng được nhưng không đổi tên làm mất nguồn modality.

### 2A.4 Bàn giao và Publishing Criteria ASR

Bàn giao:

1. `asr.sqlite`.
2. Coverage report liệt kê trạng thái từng `video_id`.
3. Checkpoint/provenance cần để resume và audit.

Một video chỉ được đánh dấu ASR complete khi:

- [ ] Audio có thật và mọi segment hợp lệ; video không có segment phải được xác minh là
      không có thoại/âm thanh, không phải lỗi ASR.
- [ ] Mọi `keyframe_uid_nearest` tồn tại trong `frames.csv` của cùng video.
- [ ] `asr.sqlite` build FTS5 thành công và query mẫu trả đúng kết quả.
- [ ] Coverage report không đánh dấu hoàn tất cho checkpoint dở dang.

Online chỉ tăng trọng số ASR khi `UnifiedQueryPlan.spoken_text` không rỗng. Video thiếu
ASR coverage phải hiện rõ để thí sinh dùng human-in-the-loop, không im lặng diễn giải kết
quả rỗng thành “không có thoại”.

---

## 3. Online Pipeline (Nhánh 2)

```
[Query text] ──> QueryPlanner (xem interface 3.1)
                       │
        ┌──────────────┼───────────────────────────┐
        ▼ (Visual)      │                            │
  clip.faiss  siglip.faiss  eva_clip.faiss           │
        │            └──────┬──────┘                 │
        │ (rollback     SRRF (siglip+eva_clip)        │
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
visual có **3 index độc lập** (`clip.faiss`, `siglip.faiss`, `eva_clip.faiss`), trong khi OCR và
ASR mỗi kênh chỉ có 1 nguồn điểm. Nếu code gộp cả 5 nguồn điểm (clip, siglip, eva_clip, ocr, asr)
vào chung 1 Min-Max, `modality_weights["visual"]` sẽ bị pha loãng sai vì visual chiếm 3/5 vote
thay vì đúng 1 vote như OCR/ASR.

**Tầng 1 — Intra-visual fusion (gộp 3 embedding thành 1 điểm `score_visual`):**

- Search song song cả 3 FAISS index cho mỗi query.
- Gộp điểm SigLIP + EVA-CLIP bằng **Score-Reflected Reciprocal Rank Fusion (SRRF)** — giữ được
  phân phối điểm tương đồng thực tế, không chỉ dựa vào rank thuần như RRF gốc.
- **CLIP không tham gia SRRF.** CLIP giữ vai trò rollback: chỉ dùng làm `score_visual` chính
  cho keyframe nào mà `siglip.faiss` hoặc `eva_clip.faiss` **chưa Ready** (theo Publishing
  Criteria §2.3). Khi cả 3 index đã Ready cho video đó, `score_visual`
  = kết quả SRRF(SigLIP, EVA-CLIP); điểm CLIP chỉ log lại để so sánh/tie-break, không cộng thêm
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
| Shot detection (offline) | TransNetV2 (`transnetv2-pytorch==1.0.5`) | Windows NVIDIA GPU; CPU reference/fallback; Colab T4 phụ | Ghi theo batch report | CUDA chọn tường minh, không fallback; phải qua parity gate CPU–CUDA |
| Frame recall baseline | CLIP ViT-B/32 | Local/Kaggle | ~300 MB | Rollback an toàn |
| Frame recall nâng cao | SigLIP + EVA-CLIP | Kaggle GPU (offline) | N/A (chạy batch, không online) | Build index nền |
| OCR (primary) | Gemini 2.5 Flash-Lite | Cloud API | 0 MB | |
| OCR (fallback) | EasyOCR (CRAFT + latin_g2) | Local CPU | thấp | Đã xác nhận hỗ trợ tiếng Việt |
| Reranking | Neighbors-Based (thuật toán, không phải model) | Local CPU | 0 MB | |
| VQA / answer generation | Qwen3-VL-2B-Instruct | Local GPU, 4-bit | ~1.8 GB | Chỉ chạy khi qua Multiplicative Gating |
| ASR (nhánh 3, song song) | Whisper Large-v3 / phoWhisper | Kaggle/Colab GPU riêng (offline) | N/A (chạy batch, không online) | Xem §2A; human-in-the-loop giữ làm backup nếu coverage thiếu |
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

1. Khóa và test schema `FrameRecord`, `OcrResult`, `AsrSegment`,
   `UnifiedQueryPlan`.
2. Chạy Inventory full collection bằng `ffprobe`, **không `--limit`**, kiểm số video và
   metadata trước Shot Detection.
3. Dựng environment GPU, doctor PASS, parity TransNetV2 CPU–GPU 5/5 rồi chạy full shot batch
   với checkpoint/resume.
4. Trích keyframe Begin/Middle/End, benchmark blur/pHash trên dev subset rồi mới chốt threshold.
5. Chạy song song: CLIP/SigLIP/EVA-CLIP + OCR ở Nhánh 1, Whisper/phoWhisper ở Nhánh 3.
6. Build `frames.csv`, ba FAISS `IndexIDMap`, `ocr.sqlite`, `asr.sqlite`; chỉ publish
   video qua đủ Publishing Criteria.
7. Hoàn thiện Online: QueryPlanner fallback → SRRF visual → fusion visual/OCR/ASR →
   reranking → TRAKE/QA → dedup/submission.
8. A/B CLIP-only với SigLIP+EVA-CLIP và các threshold trên dev set có ground truth.
9. Chỉ khi critical path đã ổn định mới làm BLIP-2 Cloud/RunPod như stretch goal.

---

## Changelog

- **23/08/2026** — Chốt baseline hợp nhất: bỏ mean-pooling shot-level làm index chính, bỏ
  RunPod/BLIP-2 khỏi critical path, đổi OCR fallback từ PARSeq-Ti sang EasyOCR, thêm 3 lớp
  fallback bắt buộc cho `QueryPlanner`.
- **23/08/2026 (bản 2)** — Đồng bộ `frames.csv` sang khóa `keyframe_uid` thay
  `faiss_row_id` positional; chuyển ASR thành Nhánh 3 song song, thêm kênh ASR vào fusion và
  giữ human-in-the-loop làm backup.
- **23/08/2026 (bản 3)** — Publishing Criteria đổi sang set diff `keyframe_uid`; đồng bộ
  `OcrResult` với `bbox`, `detected_text` và `confidence`.
- **23/08/2026 (bản 4)** — Sau khi đối chiếu với 2 baseline tham khảo (FiftyOne BTC/
  `lducc-hcm-aic`, và 1 doc SOTA offline+online khác): (1) Thêm nguyên tắc bắt buộc §0.8 +
  tầng 1 "intra-visual fusion" vào §3.2 — ở kiến trúc bản 4 khi đó, gộp SigLIP+BEiT-3
  bằng SRRF thành 1 `score_visual`
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
- **24/08/2026 (bản 7)** — Chốt TransNetV2 làm shot detector production; dừng chờ/A-B
  AutoShot trên critical path. Windows NVIDIA GPU là worker production sau parity 5/5, CPU
  giữ làm reference/fallback. Batch runner bắt buộc checkpoint signature-aware theo từng
  video và chỉ nâng `1/1` sau khi manifest schema v2 đã atomic-publish rồi validate lại.
- **24/08/2026 (bản 8 — baseline duy nhất)** — Hợp nhất toàn bộ contract Offline Indexing
  và ASR vào file này; xóa hai spec tách nhánh để không còn nhiều nguồn chuẩn. Chốt Inventory
  `ffprobe` là bước bắt buộc trước Shot Detection full collection, chỉ chạy lại khi tập MP4
  thay đổi hoặc inventory stale; cấm publish kết quả `--limit` vào output production.
- **26/08/2026 (bản 9)** — Theo xác nhận rõ của người dùng, tách contract Python theo môi
  trường: local/Shot Detection giữ Python 3.11.x; Visual Embedding trên image Kaggle hiện tại
  dùng Python 3.12.x. Gate thật đã PASS 87 test (6 skip theo platform), Torch
  `2.10.0+cu128`, CUDA 12.8, Tesla T4 và immutable revision của CLIP/SigLIP. Manifest Visual
  phải ghi Python/system/machine cùng Torch/Transformers/CUDA/GPU; checkpoint dở dang chỉ
  được resume trong cùng runtime, không trộn shard giữa Python/Torch khác nhau.
- **26/08/2026 (bản 10)** — Theo quyết định vận hành cuối của người dùng, loại vĩnh viễn
  BEiT-3/Microsoft UniLM khỏi kiến trúc; không mở lại audit/checksum/conversion/adapter.
  Modality thứ ba chính thức đổi sang EVA-CLIP với khóa `eva_clip`, artifact
  `eva_clip.faiss`, và tầng intra-visual SRRF đổi thành SigLIP + EVA-CLIP; CLIP vẫn là
  rollback. Publishing Ready yêu cầu tập `keyframe_uid` khớp 100% trong cả
  `clip.faiss`, `siglip.faiss`, `eva_clip.faiss`. EVA-CLIP phải pin immutable HF revision,
  chỉ load checkpoint `.safetensors` đã xác thực, và qua dev-subset-5
  interrupt → process mới resume → validate trên Kaggle CUDA trước khi được viết/chạy
  production 9 batch.
- **26/08/2026 (bản 11)** — Chốt OCR artifact envelope schema v1 quanh `OcrResult` mà không
  đổi schema dùng chung/FTS: terminal status `success/no_text/error`, provenance attempt,
  bbox quadrilateral normalized, completion gate exact UID với `error=0`; siết fallback để
  auth/quota global dừng batch và chỉ response sai schema mới rơi EasyOCR từng frame. Chín
  batch hợp nhất qua JSONL shard trước khi build một `ocr.sqlite` cuối ở local; EasyOCR phải
  dùng weight pin checksum và `download_enabled=False`.
