# LASTDANCE — Baseline Hợp Nhất (AIC 2026)

> Đây là **nguồn chuẩn kỹ thuật duy nhất** cho Offline, ASR và Online. Runbook/status chỉ mô
> tả cách vận hành hoặc tiến độ, không được định nghĩa schema/contract khác file này. Nếu code
> và tài liệu lệch nhau, sửa code theo tài liệu, trừ khi quyết định mới của người dùng được
> ghi vào Changelog. Không duy trì spec tách nhánh hoặc tài liệu kiến trúc archived song
> song; chi tiết vận hành chỉ nằm trong các runbook được liên kết từ README.

**Cập nhật:** 29/08/2026
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
| `window_id` | str \| null | Trường tương thích catalog cũ; Online frame-level hiện hành không dùng để retrieval/rank |
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
          - Tầng 1: CRAFT detect trên toàn catalog
          - Tầng 2: EasyOCR latin_g2 đọc mọi region CRAFT-positive
          - Tầng 3: Vintern FP16 official xử lý region router v2 chọn
          - Tầng 4: Gemini chỉ nhận residual/arbiter sau khi báo exact count/cost
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
| 6 | OCR | keyframe | text + bbox | CRAFT + EasyOCR + Vintern + Gemini residual | Tầng 1–3 build trên Kaggle GPU; Gemini chỉ nhận residual sau budget gate — xem 2.2 |
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



### 2.2 OCR phân tầng CRAFT → EasyOCR → Vintern → Gemini residual — schema chuẩn

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

- CRAFT chạy trên mọi keyframe; EasyOCR chạy trên mọi region CRAFT-positive. Region rỗng,
  confidence `<0,40`, hoặc ở dải `0,40–0,60` có mixed/glyph/noise theo router v2 mới vào
  Vintern khi archive Vintern hoàn chỉnh đã sẵn sàng. Vintern không còn là barrier bắt buộc:
  nếu batch chưa có Vintern, toàn bộ candidate router v2 của batch đó đi thẳng vào residual
  Gemini với provenance `vintern_not_available`; EasyOCR text/bbox vẫn được giữ làm fallback.
  Output Vintern rỗng, prompt leak, giải thích hoặc phình bất thường cũng trở thành residual.
  Gemini không chạy tự động: trước Tầng 4 phải báo exact số residual region, unique
  `keyframe_uid`, unique shot/request, token và chi phí để người dùng quyết.
- Vintern **không** chạy song song trên mọi region để thi với EasyOCR. Trong Gate B, cùng
  inference Vintern đã chạy cho candidate router v2 phải ghi output length, khoảng cách tới
  length guard và mean token log-prob nếu runtime expose; thiếu log-prob ghi `null/
  unavailable`, không chạy model lần hai. Vì deadline và chỉ có một annotator, Gate B dùng
  evidence tier `emergency_single_annotator_100`: đúng 100 Vintern-region thuộc 100 frame
  khác nhau, cân bằng 20 frame/video trên full dev-subset-5 và chọn deterministic theo năm
  stratum router/confidence đã pre-register. Đây không được gọi là standard 300-frame PASS.
  Theo deadline override của người dùng, đúng hai crop thật sự unreadable được loại khỏi
  calibration; evidence usable là 98/100 region thuộc 98 frame. Report phải ghi tier
  `emergency_single_annotator_98_of_100`; không được hạ thêm dưới 98.
  Confidence Vintern là tỷ lệ exact-match thực đo trong bucket (Unicode NFC + casefold +
  collapse whitespace). Bucket fine/structural dưới 20 mẫu không được override; global chỉ
  dùng báo cáo, không được cấp quyền override. Không dùng số model tự khai.
- Sau guard, Vintern chỉ thay text của đúng region khi confidence calibrated **lớn hơn
  nghiêm ngặt** confidence EasyOCR gốc. Bằng nhau/thấp hơn, thiếu result hoặc guard fail thì
  giữ EasyOCR và/hoặc đưa residual sang Gemini. Mỗi candidate phải có audit JSONL gồm bucket,
  support/correct, text và confidence cũ/mới, guard, quyết định override và final engine.
  `ocr_fts` vẫn giữ năm cột: nguồn/calibration nằm trong envelope/audit sidecar, không thêm
  cột SQL. Snapshot sau Gate B ghi tier `easyocr_vintern_calibrated` (hiển thị
  “EasyOCR+Vintern calibrated”), không được ghi `easyocr_only` nếu đã materialize override.
- Do deadline và chỉ có một annotator, Gate A CRAFT dùng emergency contract schema v2 trên
  đúng 100 UID đã pin từ 100 dòng đầu của bundle bất biến: 60 `L21_V001` + 40 `L21_V002`.
  Ngưỡng chất lượng không hạ: region recall `>=0,98`, text-frame recall `>=0,99`. Evidence
  thực tế không có frame `no_text` và không cân bằng năm video, nên không được gọi là
  threshold PASS. Nếu không config nào đạt, explicit deadline override chỉ được giữ nguyên
  `recall_current` (`0,6/0,3/0,3`) để mở full dev-subset-5 Gate B; report phải mang decision
  `DEADLINE_OVERRIDE_KEEP_CURRENT`, exact UID-set/policy hash và ba limitation đã khóa.
  Không dùng 100 frame để ngoại suy ETA production; full dev5 Gate B vẫn là evidence kế tiếp.
- Mỗi terminal record ghi `video_id`, `keyframe_uid`, `frame_id`, relative `source_image`,
  `execution_mode=craft_easyocr_vintern_gemini`, `status`, `engine`, `fallback_used`,
  `attempts` và `result`. Attempt phải theo thứ tự CRAFT detection → EasyOCR recognition →
  Vintern recognition → Gemini recognition khi Vintern artifact có sẵn. Khi Vintern chưa có,
  được phép ghi attempt `vintern_not_available` rồi chuyển thẳng candidate router v2 sang
  Gemini; không được giả Vintern success/override. Được dừng ở EasyOCR/Vintern khi đã terminal.
  `fallback_used=true` khi record đã escalate vượt EasyOCR; tên field cũ được giữ để không
  đổi envelope schema. Mode cũ chỉ giữ để đọc evidence lịch sử, không dùng cho production.
- `status` là `success | no_text | error`. `success` bắt buộc có `result: OcrResult` và ít
  nhất một text không rỗng; `no_text`/`error` dùng `result=null`, không tạo `language` hay
  confidence giả. Provenance/error chỉ nằm trong envelope/manifest, không thêm cột vào
  `ocr_fts`.
- `bbox` trong envelope chuẩn hóa thành quadrilateral 8 số
  `[x1,y1,x2,y2,x3,y3,x4,y4]`, theo chiều kim đồng hồ từ góc trên-trái trực quan, tọa độ
  `[0,1]` trên ảnh keyframe sau khi đã xoay đúng orientation. EasyOCR quy đổi 4 point sang
  dạng này; Vintern/Gemini không được sinh hoặc sửa bbox.
- `confidence` Gemini giữ giá trị top-level đã validate từ JSON schema. Adapter EasyOCR
  tính trung bình confidence từng vùng, trọng số bằng số code point không-whitespace của
  text vùng đó; kết quả clamp vào `[0,1]`.
- Nếu CRAFT không tìm thấy region, record kết thúc `no_text`. Nếu có region, crop được phóng
  rõ, gắn `region_id` local duy nhất và giữ mapping `region_id -> keyframe_uid/frame_id/bbox`
  xuyên suốt ba tầng recognition. Contact-sheet Gemini chỉ chứa residual còn lại sau
  EasyOCR/Vintern, gom theo shot; Gemini chỉ trả `region_id`, `text`, `language`,
  `confidence`, không sinh bbox/UID.
- Request crop-sheet dùng global `media_resolution=MEDIA_RESOLUTION_MEDIUM` ở Gate 2. Không
  dùng LOW trên nguyên frame; MEDIUM cũng chưa được coi là production-safe cho chữ nhỏ nếu
  chưa qua A/B ảnh thật với HIGH/default. Strict `responseJsonSchema` phải exact-set mọi
  `region_id`, reject duplicate/missing/foreign row.
- CLIP/SigLIP/EVA-CLIP chỉ xếp lịch residual quan trọng trước. Chỉ được reuse kết quả OCR của
  một keyframe sang keyframe khác cùng shot khi **đồng thời** embedding cosine, CRAFT layout,
  crop SSIM và crop pHash đều qua threshold hash-bound. Thiếu một tín hiệu thì vẫn gửi crop
  của frame đó; embedding đơn độc không được kết luận `no_text` hoặc cho phép reuse.
- Gemini 2.5 Flash-Lite là lựa chọn người dùng dự kiến cho Tầng 4, nhưng chưa đổi pin runtime
  cho tới khi Paid key canary trả HTTP 200 + schema-valid. Không tự gọi Gemini hoặc phát
  sinh chi phí. Report pre-Gemini phải chốt exact residual region/frame/shot/request và
  token/cost dưới trần 400.000 VND để người dùng chọn phạm vi.
- Mốc bàn giao tạm 28/08/2026 đã có đủ 9/9 archive CRAFT+EasyOCR (293.336 keyframe), chưa có
  Vintern và chưa gọi Gemini. Exact residual khi bypass Vintern là 830.301 region trên
  253.177 frame, gom thành 92.768 shot/request và 106.183 contact-sheet. Estimate Standard
  có reserve 15% vượt trần (651.803 VND); estimate Batch trong trần (325.902 VND), nhưng
  không được dùng runner Standard hiện có để giả thực thi Batch. Nhánh 2 chỉ được coi artifact
  EasyOCR là development snapshot, không phải OCR final.
- Terminal coverage chỉ PASS khi toàn bộ UID exact,
  `success + no_text == expected`, `error == 0`, không duplicate/foreign/missing.
- Chín JSONL terminal shard phải disjoint và exhaustive so với `frames.csv`. Chỉ sau union
  mới build **một** `ocr.sqlite` cuối ở local; chỉ record `success` được nạp vào `ocr_fts`.

Để Nhánh 2 phát triển song song trước khi đủ chín batch, Nhánh 1 được bàn giao **OCR SQLite
snapshot** theo tiến độ. Snapshot không thay đổi hoặc rút gọn pipeline chính:

- Mỗi snapshot nằm trong thư mục bất biến
  `ocr/snapshots/ocr-snapshot-<UTC>-<source-hash>/`, gồm đúng `ocr.sqlite`,
  `coverage.json` và `SHA256SUMS`; không ghi đè snapshot cũ và không dùng path
  `ocr/archives/{batch_id}` của shard production.
- Bảng `ocr_fts` giữ nguyên năm cột chuẩn. Trạng thái tầng, coverage và provenance chỉ nằm
  trong sidecar, không thêm cột SQL khiến `FtsSearcher` lệch contract.
- `coverage.json` bắt buộc có `immutable=true`, `complete=false`,
  `production_ready=false`, `intended_use=online_development_only`, catalog/source/SQLite
  SHA-256, coverage toàn catalog và từng video, engine count, materialized text tier, cùng
  số Vintern required/completed/accepted/residual/pending region. Schema v2 thêm map
  `batches[batch_id]` với tier exact `craft_only|easyocr|vintern_calibrated|gemini_final`,
  `complete`, `frames`/expected/processed/status/missing/pending, `updated_utc`, danh sách
  video, assigned/observed UID-set SHA-256 và source artifact checksum. `craft_only` chỉ là
  coverage/checkpoint, không được materialize vào FTS.
- Snapshot chỉ nạp record `success`; `no_text`, `error` và UID còn thiếu vẫn được đếm trong
  coverage. Một video có đủ UID snapshot không đồng nghĩa Publishing Ready hoặc đã qua đủ
  Vintern/Gemini. Nhánh 2 phải đọc `snapshot_id`/coverage và không cache như artifact final.
- Snapshot mới có thể trỏ `parent_snapshot_id` để xác định thế hệ kế tiếp, nhưng luôn tạo
  thư mục mới. Chỉ `ocr.sqlite` cuối sau terminal union chín batch mới có thể đi qua
  Publishing Criteria; snapshot không bao giờ được đổi `complete=true`.
- Bốn worker không ghi chung SQLite. Mỗi worker chỉ xuất JSONL/manifest riêng; builder local
  kiểm tra partition video/UID của chín batch disjoint và exhaustive rồi atomic-build một
  SQLite snapshot mới. Dev-subset-5 không phải batch thứ mười và không được append cạnh
  batch-01: cùng UID xuất hiện hai nguồn phải fail closed.

Tầng 1–3 chỉ chạy Kaggle GPU, có thể chia chín batch UID-disjoint trên bốn tài khoản. Máy
Codex không GPU không chạy model; RTX 4050 máy thi chỉ pull artifact và dùng `ocr.sqlite`
đã build. Mỗi layer ghi JSONL/manifest, push theo batch vào cùng HF Dataset; local dùng
`snapshot_download()` theo §2.4 rồi mới union shard và build một SQLite cuối.

Failure policy fail-closed: CRAFT/`latin_g2` production phải dùng `download_enabled=False`
sau khi package/weights đã verify filename, size và SHA-256. CRAFT error ghi `error`, không
được coi là `no_text`; EasyOCR/Vintern error hoặc output bị guard chặn phải giữ provenance
và đi residual queue, không âm thầm biến thành success/no_text. Nếu người dùng mở Tầng 4,
Gemini `429`/`5xx`/timeout retry exponential backoff + jitter; `401`/`403`, quota/global
failure hoặc budget guard dừng cloud worker và xuất routing state, không vượt trần chi phí.

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

Invariant: identifier/text không rỗng, timestamp hữu hạn và không âm, `end_time >= start_time`, và
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

Envelope Kaggle v1 tiếp tục được đọc. `silent` không có `silence_verification` vẫn đọc được
để audit, nhưng `verified_complete=false`, `unverified_silent_videos` tăng và coverage video
bằng 0. Verification gồm audio SHA-256, người review và đường dẫn evidence tương đối; hash
phải khớp audio. Không đổi inference rỗng thành bằng chứng im lặng. Segment không vượt duration.
SQLite và `asr.coverage.json` phải khớp checksum/size, catalog SHA, số dòng và trạng thái từng
video, UID cùng video và nearest-PTS alignment. Online từ chối cặp file chưa đồng bộ; sidecar
không có hoặc sai thì ASR INVALID, visual vẫn hoạt động. `--allow-partial` chỉ phục vụ development;
không nâng `production_ready`. Snapshot ASR giữ bất biến và `complete=false`.

Online chỉ tăng trọng số ASR khi `UnifiedQueryPlan.spoken_text` không rỗng. Video thiếu
ASR coverage phải hiện rõ để thí sinh dùng human-in-the-loop, không im lặng diễn giải kết
quả rỗng thành “không có thoại”.

---

## 3. Online Pipeline Accuracy-Max (Nhánh 2)

Mục tiêu vòng sơ tuyển là tìm **đúng video trước**, sau đó chọn frame/answer/sequence đúng
cho KIS, QA hoặc TRAKE. Retrieval vẫn ở frame-level để giữ exact evidence; video ranking là
lớp tổng hợp evidence, không tạo video embedding và không mean-pool vector.

```text
ArtifactRegistry: frames.csv + 3 FAISS + keyframes/videos + OCR/ASR tùy chọn
        ↓
SearchRequest + QuerySpec
        ↓
Unified Query Planner (Gemini → Qwen local → rule)
        ↓
Operator review: VIDEO_LOCATOR | TARGET_MOMENT | ANSWER_EVIDENCE | ORDERED_EVENT
        ↓
Video retrieval theo toàn plan → target/event retrieval tách riêng
        ↓
Visual retrieval độc lập: SigLIP + EVA SRRF; CLIP comparison/rollback
        + OCR/ASR FTS theo intent
        ↓
Multimodal fusion → temporal-neighbor boost
        ↓
Frame evidence → video hypotheses → VLM verification Top 4
        ↓
KIS | QA | TRAKE task head
        ↓
Confidence-adaptive Top 100 → Streamlit review → official CSV/ZIP validator
```

### 3.1 Runtime boundary và code contract

Critical path nằm trong `online/`; Streamlit gọi trực tiếp `OnlineEngine`, **không có
FastAPI/backend trung gian**.

```python
OnlineEngine.plan(QuerySpec) -> UnifiedQueryPlan
OnlineEngine.search(SearchRequest, query_plan: UnifiedQueryPlan | None) -> SearchRun
```

`SearchRequest`:

```text
task_type: KIS | QA | TRAKE
raw_query: str
query_spec: QuerySpec | null
max_results: 1..100 (default 100)
mode: accurate
```

`plan()` tạo contract có thể review mà chưa chạy FAISS. CLI được phép gọi `search()` với
`query_plan=None` để planner tự chạy; Streamlit bắt buộc gọi `plan()`, cho operator sửa rồi
truyền plan đã duyệt vào `search()`. `QuerySpec` khóa `query_name`, `source_filename`,
`task_type`, nguyên văn query và `expected_event_count` bắt buộc cho TRAKE. `SearchRun` trả query plan, artifact status,
Top video hypotheses, Top task candidates, timing, provenance và warning. Mọi evidence dùng
`keyframe_uid`; candidate/submission chỉ dùng `video_id` + `frame_id`.

### 3.2 Startup preflight và ArtifactRegistry

Input bắt buộc:

1. `frames.csv` + state hợp lệ, đúng 293.336 keyframe/873 video cho catalog hiện hành.
2. `clip.faiss`, `siglip.faiss`, `eva_clip.faiss` + state; đúng
   `IndexIDMap(IndexFlatIP)`, dimension 512/768/768, revision model, catalog/UID-set và
   checkpoint-resume provenance.
3. Keyframe JPEG là bắt buộc để giữ đầy đủ thumbnail, contact sheet, VLM verification và
   manual review của Accuracy-Max. Source MP4 **không phải điều kiện startup của vector
   retrieval**, nhưng bắt buộc cho playback và FFmpeg exact-frame refinement; thiếu MP4
   phải hiện degraded mode, không được giả đã decode frame nguồn.

Visual lỗi làm Online `NOT_READY`. OCR/ASR được phân loại độc lập:

- `READY`: schema/integrity/UID join hợp lệ;
- `UNAVAILABLE`: artifact chưa có, pipeline tự renormalize kênh còn lại;
- `INVALID`: file tồn tại nhưng schema/hash/join sai, tắt modality và hiển thị lỗi.

Production OCR mặc định ở `$AIC_DATA/index/ocr.sqlite`. Snapshot development chỉ được chọn
bằng `AIC_OCR_SNAPSHOT_DIR=<snapshot directory>`; registry phải verify đúng ba file
`ocr.sqlite`, `coverage.json`, `SHA256SUMS`, checksum, catalog SHA/count/video/UID-set,
FTS5 count/integrity và join `(video_id,keyframe_uid)`. UI/provenance phải hiện snapshot ID,
tier, coverage, error/missing và `production_ready=false`; cấm copy/đổi tên snapshot thành
artifact final.

Data retention của máy thi được chia ba lớp:

- **Runtime core, cấm xóa:** `frames.csv` + state, ba FAISS + state, exact text-encoder
  snapshot khớp revision và OCR/ASR artifact đang bật.
- **Accuracy/review, nên giữ:** toàn bộ `$AIC_DATA/keyframes/`; giữ MP4 nếu cần playback
  hoặc exact-frame refinement. Xóa/move MP4 không làm vector search fail nhưng làm mất hai
  chức năng này.
- **Rebuild/resume only:** raw visual embedding shard/mirror, shot/quality/plan/batch state,
  OCR archive/source và detector/object intermediate. Online không đọc các path này sau khi
  FAISS/OCR snapshot đã được deep-validate; chỉ được dọn khi immutable HF revision còn truy
  cập được hoặc đã có backup ngoài máy.

`$AIC_DATA/tmp/online-refinement/` là cache tái tạo được. Draft/CSV/ZIP trong
`$AIC_DATA/submissions/` là dữ liệu người dùng, không tự động xóa.

### 3.3 Unified Query Planner

Thứ tự provider duy nhất:

```text
Gemini (default hiện hành gemini-3.5-flash-lite)
  → Qwen3-VL-2B-Instruct local
  → deterministic rule fallback
```

Planner output chuẩn là role-aware; một unit được phép mang nhiều role:

```text
VIDEO_LOCATOR   = clue dùng tìm đúng video
TARGET_MOMENT   = cảnh/frame KIS hoặc QA cần định vị
ANSWER_EVIDENCE = nơi chứa dữ kiện để trả lời QA
ORDERED_EVENT   = đúng một event phải xuất hiện trong sequence TRAKE
```

```text
UnifiedQueryPlan:
  raw_query, global_context_en, retrieval_queries[≤2], query_units[],
  answer_target, submission_target_ids[], ordered_event_ids[], planner_warnings[]

QueryUnit:
  unit_id, description_original, retrieval_query_en, roles[], requiredness,
  modalities[], temporal_group, temporal_order, known_text_literals[],
  visual_text_attributes[], confidence
```

Quy tắc:

- giữ nguyên query gốc để audit nhưng mọi visual query từ model phải là tiếng Anh;
- `global_context_en` là faithful global query bắt buộc; thêm tối đa một discriminative query;
- `description_original` phải là span có thật trong query; thuộc tính đồng thời ở cùng
  `temporal_group`, chỉ tách unit mới khi thực sự đổi cảnh/thời gian;
- visual query từ model phải bằng tiếng Anh, chạy độc lập và không mean-pool text vector;
- không thêm người/vật/hành động/tên riêng; provider không được tạo negative constraint;
- `known_text_literals` chỉ chứa text mà query đã nêu chính xác, ví dụ “giá dầu mazut”. Mô tả
  “biển đỏ có 6 ký tự chữ Hán” thuộc `visual_text_attributes`; số chưa biết đang được hỏi là
  `AnswerTarget`, tuyệt đối không đưa hai loại này vào FTS exact/prefix;
- QA bắt buộc có `AnswerTarget(value_is_unknown=true)` và evidence unit; TRAKE chỉ dùng ID
  mang role `ORDERED_EVENT`, context locator không được tính thành event;
- fallback được ghi warning; rule fallback không được giả là đã dịch tiếng Anh.

Schema cũ `scenes/anchor_moment_index/ordered_moments` chỉ được adapter đọc để migration;
Online Core không dùng các trường này để quyết định frame, answer evidence hoặc event.

### 3.4 Visual retrieval và SRRF

Mỗi visual query được encode độc lập bằng đúng text encoder/revision của index, `float32` và
L2-normalized. Cache key gồm model revision + query text. Trên Windows, Torch text encoder và
Qwen mặc định chạy trong worker tách process để tránh xung đột OpenMP với FAISS.

Với mỗi query:

1. search Top 1.000 SigLIP và Top 1.000 EVA;
2. tạo union `keyframe_uid`;
3. reconstruct vector qua internal position chỉ bên trong adapter để tính missing channel;
4. fuse SigLIP/EVA bằng SRRF:

```text
smooth_rank_m(u) = 0.5 + Σ sigmoid(beta × (score_m(j) - score_m(u)))
SRRF(u) = 1/(eta + smooth_rank_siglip(u)) + 1/(eta + smooth_rank_eva(u))
eta=60, beta=40
```

SRRF được min-max normalize. Các query độc lập được gộp boost-only:

```text
score_visual(u) = normalize(best_query_score + 0.1 × second_best_query_score)
```

CLIP được search song song để ghi model agreement và tie-break video có score cách nhau tối
đa `0,02`; CLIP không cộng vào `score_visual`. Nếu SigLIP hoặc EVA lỗi query-time, toàn
retrieval rơi tường minh về CLIP và sinh warning; không trộn rollback âm thầm.

### 3.5 OCR/ASR, fusion và temporal neighbors

OCR FTS dùng `detected_text`; ASR FTS dùng `transcribed_text`. Search cascade:

1. exact phrase;
2. đủ token bằng AND;
3. prefix candidate pool 5.000 row rồi rerank theo token coverage;
4. fuzzy chỉ trên candidate hẹp.

Các MATCH stage không so trực tiếp trị tuyệt đối BM25 giữa query khác nhau; thứ tự cascade
luôn thắng trước, BM25 chỉ quyết định rank trong cùng stage.

Cascade chỉ chạy với `known_text_literals`. Khi QA hỏi một giá trị chưa biết, answer head
đọc OCR/ASR trực tiếp theo UID của evidence frame và tối đa hai temporal neighbor trước/sau
trong candidate video; không biến câu hỏi hoặc mô tả kiểu chữ thành MATCH expression.

Trọng số intent mặc định:

```text
visual-only:                visual=1.00
visible text:               visual=0.55, OCR=0.45
spoken text:                visual=0.55, ASR=0.45
visible + spoken:           visual=0.50, OCR=0.25, ASR=0.25
```

Mỗi channel min-max riêng rồi mới late-fuse. Modality thiếu không nhận score 0; trọng số các
channel hiện có được renormalize về tổng 1.

Với mỗi candidate, lấy tối đa hai neighbor trước/sau, ưu tiên cùng shot rồi theo `pts_time`:

```text
neighbor_support = mean(top-2 valid neighbor scores)
score_reranked = normalize(frame_score + 0.15 × neighbor_support)
```

Đây là boost-only; không mean-pool neighbor và không loại sự kiện ngắn vì neighbor yếu.

### 3.6 Frame evidence thành video hypothesis

Khi tính score video, chỉ giữ evidence tốt nhất mỗi shot. Với mỗi query unit `j`:

```text
evidence_j(v) = 0.7 × max(shot_scores) + 0.3 × mean(top-3 shot_scores)
```

```text
base_video(v) =
    0.35 × locator_evidence
  + 0.45 × target_or_event_evidence
  + 0.10 × global_query
  + 0.10 × model_consensus
```

Mỗi role evidence gộp `0,50 × coverage + 0,30 × mean + 0,20 × weakest`; coverage tính video
xuất hiện trong Top 50 của từng unit. Sau khi giữ Top 12 video KIS/QA, pipeline chạy target
retrieval riêng: locator frame không được đi thẳng vào submission rank. Target pool không
hard-dedup theo shot vì keyframe lân cận có thể là exact KIS frame hoặc OCR rõ hơn. TRAKE giữ
Top 20. Không hard-filter chỉ còn một video trước task head.

### 3.7 VLM verification/reranking

Verifier chạy tối đa Top 4 video, tối đa 36 frame/video, 12 frame/contact sheet. Provider:
Gemini trước, Qwen local CUDA sau; thiếu cả hai giữ nguyên retrieval score và sinh warning.
Quota mỗi search: tối đa 8 Gemini call và 300 giây; safe defaults 14 RPM, 225k TPM, 450 RPD.

VLM trả scene match, must/should match và danh sách frame rank có cấu trúc:

```text
verified_video = 0.70 × base_video + 0.20 × must_have + 0.10 × should_have
final_frame = 0.70 × retrieval_score + 0.30 × VLM_frame_score
```

Chỉ frame ID thật sự có trong output VLM mới được fuse. Frame bị VLM bỏ qua giữ nguyên
retrieval score; partial output không phải negative evidence.

### 3.8 Task heads

#### KIS

`VIDEO_LOCATOR` và `TARGET_MOMENT` cùng tham gia rank video. Sau đó chỉ các unit trong
`submission_target_ids` được retrieval lại để rank frame nộp; locator-only frame không được
chiếm Top 100. Query chỉ có một khoảnh khắc gắn đồng thời hai role. Target candidate pool
không hard-dedup theo shot để exact frame không bị thay bởi shot leader.

Top 100 có seed `video1, video1, video2, video1, video3`, sau đó weighted round-robin theo
confidence: tối đa 40 row/video, video đầu mục tiêu 30–40 nếu đủ, Top 5 tối thiểu hai row và
Top 12 tối thiểu một row. Ba frame đầu video #1 ưu tiên ba shot khác nhau; không padding hoặc
duplicate.

#### QA

Dùng `VIDEO_LOCATOR` để rank video và `ANSWER_EVIDENCE` để retrieval riêng frame/shot chứa
đáp án; không đưa đáp án phỏng đoán vào retrieval. `AnswerTarget` mô tả value type/source
nhưng luôn là unknown. OCR/ASR đọc frame-local + neighbor; confidence thấp tiếp tục qua
Gemini → Qwen CUDA nếu có, nếu vẫn chưa chắc giữ candidate cho operator review. Locator:

```text
locator = 0.60 × video_score + 0.40 × best_frame_score
```

Answerer luôn được gọi cho ít nhất Top 3 video có evidence, kể cả `locator <= 0,85`. Ngưỡng
`0,85` chỉ kiểm soát auto-accept; answer thấp hơn ngưỡng mang `requires_review=true`. Không
tạo portfolio gồm các dòng `Uncertain`: video/evidence vẫn hiện để operator xem nhưng chỉ
answer thật mới thành `QACandidate`. `Uncertain`, answer rỗng, quá 100 ký tự hoặc
`requires_review=true` bị bulk-add/workspace export chặn; operator chọn và xác minh thủ công
sẽ hạ cờ review.

`VideoAnswerer.answer()` trả `AnswerResult` gồm answer, value_type, unit, evidence frame,
confidence, requires_review, provider và warnings. Mỗi row QA chỉ dùng frame do answerer
chỉ ra (panel A–F hoặc source frame ID có trong context), không broadcast một đáp án sang
mọi frame của video. Hai lần hỏi chỉ auto-confirm khi answer/type/unit khớp chính xác sau
NFC/case/whitespace normalization và có chung evidence; free text vẫn cần review. Similarity
ký tự không phải semantic agreement. OCR/ASR numeric extraction phải ràng buộc theo câu hỏi,
trường hợp nhiều giá trị hoặc không rõ thì abstain. Contact sheet letterbox toàn ảnh.
UI vẫn có form QA thủ công cho video chưa có answer; thay frame/answer phải xác nhận lại.

#### TRAKE

Chỉ unit trong `ordered_event_ids` chạy event retrieval độc lập; locator context chạy một
lượt riêng để rank video và không tham gia Beam Search. Video score:

```text
0.15 × locator_evidence + 0.45 × event_coverage
+ 0.20 × weakest_event + 0.10 × mean_event + 0.10 × model_consensus
```

Mỗi moment/video dedup theo `(video_id, frame_id)`, giữ các frame khác nhau trong cùng shot.
Không cắt Top 32 trước khi kiểm tra tính khả thi của chuỗi; loại frame không có suffix hợp lệ
trước khi cắt beam. `trake_frame_top_k` được đọc để tương thích config cũ, không còn cắt pool.
Beam width 8, chỉ mở rộng frame cùng video,
`pts_time` tăng nghiêm ngặt và không lặp frame. `trake_decay=0.0` mặc định để không phạt
khoảng cách thời gian chưa được ground-truth calibrate. Sequence phải đủ đúng
`expected_event_count`; UI cho thay neighbor nhưng vẫn validate thứ tự.

### 3.9 Streamlit review và Top 100

Trước hai tab kết quả, UI có bước `Phân tích truy vấn`: gọi `OnlineEngine.plan()`, hiển thị
global English context và từng QueryUnit, cho sửa retrieval query, nhiều role, modality,
known literal/visual attribute, target/evidence ID và thứ tự TRAKE. KIS thiếu target, QA
thiếu AnswerTarget/evidence hoặc TRAKE sai `expected_event_count` bị chặn trước FAISS. Plan
đã duyệt/chỉnh sửa và SHA-256 được ghi vào provenance.

UI kết quả có hai tab:

- `Top 100`: đúng thứ tự sẽ serialize vào CSV; KIS hiển thị thumbnail phân trang, QA là
  hypothesis frame+answer, TRAKE là sequence hoàn chỉnh;
- `Theo video`: nhóm evidence, score breakdown, scene matched/missing, temporal neighbor,
  source video và exact-frame decode bằng FFmpeg `select=eq(n,frame_id)`.

Search không tự thêm vào draft. Bulk-add là atomic và task-aware; nếu draft đã có dữ liệu,
operator chọn replace hoặc merge-fill/dedup. Workspace tách theo `query_name`, giới hạn 100
row riêng từng query và lưu atomic history/provenance ngoài submission ZIP. Merge vào draft
đủ 100 dòng giữ nguyên thứ tự hiện có. Cả ba task có điều khiển source frame ±1/5/10 và dải
21 frame liên tiếp (cắt ở biên video), một lượt FFmpeg decode. Source verification chạy CPU;
cache gắn SHA-256 video và được kiểm tra lại trước ZIP, không tiêu quota GPU.

### 3.10 Submission profile duy nhất

Profile duy nhất là `AIC26_QUALIFIER_OFFICIAL`:

- một query = một CSV; tên `.txt` đổi đúng thành `.csv`;
- 1–100 dòng, UTF-8 không BOM, comma, LF, không header, không `query_id`;
- KIS: `video_id,frame_id`;
- QA: `video_id,frame_id,answer`, answer tối đa 100 ký tự, CSV quoting chuẩn và giữ Unicode;
- TRAKE: `video_id,frame_id_1,...,frame_id_N`, đúng N event, cùng video, timestamp tăng;
- cấm `.mp4`, `local_idx`, path, score và duplicate;
- frame thuộc `frames.csv` dùng mapping catalog đã validate; frame ngoài catalog bắt buộc
  có `VerifiedFrameRef(video_id, frame_id, pts_time, source_sha256)` trong workspace.
  `frame_id` là số thứ tự decoded frame, `pts_time` lấy từ ffprobe từng frame, không tính
  bằng FPS trung bình. Verifier kiểm tra giới hạn frame và fingerprint MP4; source thiếu,
  đổi nội dung hoặc timestamp không khớp thì chặn export. Metadata xác thực nằm ngoài CSV/ZIP.
  Không tạo `keyframe_uid` mới cho source frame; retrieval evidence vẫn trỏ UID gốc.

ZIP phải chứa đủ và chỉ đủ `submission/<query_name>.csv`. Exporter bắt buộc serialize → đọc
lại/validate CSV bytes → tạo ZIP → mở lại/validate entry và byte content → hiển thị row count
và SHA-256. CSV riêng chỉ để inspection; ZIP PASS mới là file nộp chính thức.

---

### 3.11 Qualifier evaluation và acceptance

`shared/evaluation.py` chấm theo thông tin vòng sơ tuyển đã cung cấp: với từng dòng, KIS
được 1 khi đúng video và frame nằm trong interval; QA thêm điều kiện answer thuộc tập alias
ngữ nghĩa đã được người gán nhãn duyệt; TRAKE được tỷ lệ event nằm trong interval tương ứng
khi đúng video. `R@k = max(row_score[:k])`, `k ∈ {1,5,20,50,100}`; Final Score là trung bình
năm R@k. Tối đa 100 dòng; bộ chấm giữ nguyên thứ tự nộp.

Acceptance cần 60 câu có người xác minh: 20 KIS, 20 QA, 20 TRAKE; development 30 và held-out
30, mỗi loại 10/split, không trùng video giữa hai split. Query cũ chỉ dùng regression. Chọn
config/ablation trên development, khóa hash config và bộ nhãn rồi mới chạy held-out. Báo cáo
phải phân biệt Final Score, video recall, catalog event support, khoảng lệch frame, latency
và tỷ lệ query cần operator review. Phiếu gán nhãn trống và unit test không phải ground truth.
Lệnh và tình trạng thực nghiệm: `docs/QUALIFIER_ACCEPTANCE_RUNBOOK.md`.

---

## 4. Model & Hardware Budget

| Vai trò | Model | Môi trường | VRAM ước tính | Ghi chú |
|---|---|---|---|---|
| Query planning (primary) | Gemini 3.5 Flash-Lite (operator có thể override) | Cloud API | 0 MB | Free-tier/quota được rate-limit; key chỉ ở environment |
| Query planning (fallback) | Qwen3-VL-2B-Instruct | Local CUDA qua Torch worker | Peak đo thật ~4,13 GiB | Dùng chung runtime với VQA, không nạp hai bản |
| Shot detection (offline) | TransNetV2 (`transnetv2-pytorch==1.0.5`) | Windows NVIDIA GPU; CPU reference/fallback; Colab T4 phụ | Ghi theo batch report | CUDA chọn tường minh, không fallback; phải qua parity gate CPU–CUDA |
| Frame recall baseline | CLIP ViT-B/32 | Local/Kaggle | ~300 MB | Rollback an toàn |
| Frame recall nâng cao | SigLIP + EVA-CLIP | Kaggle GPU (offline) | N/A (chạy batch, không online) | Build index nền |
| OCR Tầng 1 | CRAFT (EasyOCR package) | Kaggle GPU theo shard | 9/9 archive hiện có | Phủ 293.336 UID; snapshot Online vẫn development-only |
| OCR Tầng 2 | EasyOCR latin_g2 | Kaggle GPU theo shard | 278.091 FTS row hiện có | 15.188 no-text, 57 error; không gọi là final |
| OCR Tầng 3 | Vintern-1B-v3.5 FP16 official | Kaggle T4 theo shard | Runtime PASS, router v2 dev-only | Revision/checksum pin; INT4 bị loại |
| OCR Tầng 4 | Gemini 2.5 Flash-Lite dự kiến | Cloud API | 0 MB | Chưa pin/chưa gọi; chỉ residual sau exact count/token/cost và người dùng duyệt |
| Reranking | Temporal neighbors + video evidence + VLM verification | Local CPU + Gemini/Qwen tùy chọn | Qwen peak đo thật ~4,19 GiB | VLM lỗi giữ retrieval score |
| VQA / answer generation | Gemini → Qwen3-VL-2B-Instruct → unavailable | Cloud/local CUDA | Qwen peak đo thật ~4,19 GiB | Chỉ auto-answer khi locator >0,85; operator review là gate cuối |
| ASR (nhánh 3, song song) | Whisper Large-v3 / phoWhisper | Kaggle/Colab GPU riêng (offline) | N/A (chạy batch, không online) | Xem §2A; human-in-the-loop giữ làm backup nếu coverage thiếu |

**Lưu ý vận hành:** tắt các tiến trình ngầm chiếm VRAM (LM Studio, Epic Games Launcher...)
trước khi chạy inference local. Không chạy đồng thời Qwen planner + Qwen VQA nếu tổng VRAM
vượt 6 GB — load/release model theo pha.

---

## 5. Cấu trúc thư mục repo

```
offline/    # Pipeline tiền xử lý (shot detection, embedding, OCR, indexing)
online/     # Query planning, retrieval, fusion, reranking, submission
shared/     # Pydantic schemas, interfaces (QueryPlanner, FrameRecord), constants
scripts/    # Chạy batch Kaggle/Colab, tách nhỏ danh sách video song song
```

Quy tắc: Pydantic schema định nghĩa trong `shared/`, dùng chung giữa `offline/` và
`online/` — không định nghĩa lại schema trùng lặp ở hai nơi.

---

## 6. Rủi ro/độ chưa hoàn thiện hiện hành

| # | Câu hỏi | Ảnh hưởng nếu không xác nhận trước |
|---|---|---|
| 1 | Accuracy regression trên bộ kiểm tra thủ công chưa đạt acceptance target đã đặt | Không tuyên bố baseline accuracy-complete chỉ từ preflight/test contract; phải chạy lại ground-truth Recall@12/100 |
| 2 | OCR mới ở snapshot EasyOCR development, còn 57 error và chưa có Vintern/Gemini final | Dùng được để tăng recall visible-text nhưng phải hiện provenance và không gọi production-ready |
| 3 | `asr.sqlite` chưa có | Spoken-text query chạy visual-only có warning; đây là lỗ hổng recall còn thật |
| 4 | Gemini cloud model/quota có thể đổi | Pin bằng config lúc thi, rate-limit, giữ Qwen/rule fallback và prefetch local snapshot |

Không dùng HTTP 200, đủ 100 dòng hoặc artifact `READY` để suy ra accuracy PASS. Mọi tuyên bố
accuracy phải gắn bộ query/ground truth, revision/config và metric tái lập.

---

## 7. Checklist closure hiện hành

1. **CLOSED:** Inventory/shot/keyframe catalog 873 video, 293.336 keyframe.
2. **CLOSED:** CLIP/SigLIP/EVA `IndexIDMap`, UID diff/norm/resume validation.
3. **IMPLEMENTED:** Online Accuracy-Max, Top 100 UI, official CSV/ZIP validator.
4. **INTEGRATED-DEVELOPMENT:** OCR EasyOCR snapshot 100% UID coverage; chưa final do 57 error
   và chưa có tầng Vintern/Gemini terminal.
5. **OPEN:** ASR artifact và spoken-text regression.
6. **OPEN:** chạy lại bộ kiểm tra thủ công/ground-truth, chốt Video Recall@12,
   Frame/shot Recall@100, QA evidence và TRAKE sequence accuracy.
7. **OPEN:** freeze model/config/revision và deep preflight ngay trước vòng thi.

---

## Changelog

- **04/09/2026 — qualifier audit theo plan đã được người dùng duyệt:** cho phép source frame
  đã xác thực ngoài catalog; giữ retrieval UID; QA structured evidence và review; TRAKE giữ
  frame cùng shot và kiểm tra suffix trước beam. Thêm evaluator R-Score/Final Score, khóa
  development/held-out, finite timestamps và guard inventory smoke. Publishing cần manifest
  Shot/checkpoint/log/mapping bound artifact, không chấp nhận chỉ cờ true. Consumer ASR đọc
  legacy v1 nhưng không coi silent chưa xác minh là complete. Theo chỉ đạo bổ sung cùng phiên,
  job Kaggle ASR đang chạy được giữ nguyên, chờ hoàn thành mới tải HF; OCR tạm bỏ qua vì sắp
  thay artifact. Các sửa runtime/partition ASR và kiểm chứng GPU chưa được thực hiện trong đợt này.

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
- **27/08/2026 (bản 12)** — Theo quyết định rõ của người dùng với budget cứng 400.000 VND,
  chốt OCR production là CRAFT detector gate toàn catalog rồi Gemini recognition chỉ trên
  crop có chữ; `latin_g2` nhận overflow/cloud fallback. Ba Begin/Middle/End cùng shot được
  đóng gói thành một contact-sheet request; Gemini chỉ trả `region_id/text/language/
  confidence`, bbox và UID do adapter local giữ. Paid bị chặn bởi tối đa 20.000 frame,
  token/cost ledger và reserve 15%. Embedding chỉ ưu tiên; reuse một frame chỉ khi embedding
  + CRAFT layout + crop SSIM + crop pHash cùng pass. Terminal JSONL vẫn exact một record/UID
  trước khi build SQLite cuối.
- **28/08/2026 (bản 13)** — Thay vai trò OCR theo quyết định mới: CRAFT detect toàn catalog,
  EasyOCR trở thành recognizer Tầng 2 cho mọi region, Vintern FP16 official revision
  `b98f263eab246eb5269ade64edbdca8a887dc44d` là Tầng 3 theo router v2, Gemini chỉ còn
  residual/arbiter Tầng 4. Tầng 1–3 chỉ chạy Kaggle GPU, chia chín batch UID-disjoint trên
  tối đa bốn tài khoản; máy Codex không GPU không chạy model và RTX 4050 máy thi chỉ đọc
  `ocr.sqlite` đã build sẵn. Mỗi layer phải bàn giao JSONL/manifest qua đúng HF Dataset
  chung rồi local `snapshot_download()` và hợp nhất; artifact chỉ nằm trên Kaggle chưa phải
  pipeline hoàn tất. Model Gemini 2.5 chưa pin/call cho tới Paid canary và quyết định chi phí
  sau khi báo exact residual region/frame/shot/request.
- **28/08/2026 (bản 14)** — Cho phép bàn giao OCR SQLite snapshot bất biến để Nhánh 2 code
  và test FTS/fusion song song khi production OCR vẫn chạy. Snapshot giữ nguyên schema
  `ocr_fts`, version bằng UTC + semantic source hash, đi kèm coverage/checksum và luôn
  `complete=false`, `production_ready=false`, development-only. Coverage tách rõ text đang
  materialize từ EasyOCR hay Vintern/Gemini và trạng thái Vintern theo video; snapshot không
  thay thế terminal union hoặc bất kỳ Publishing Criteria nào.
- **28/08/2026 (bản 15)** — Gate B dùng ground-truth để calibrate confidence Vintern theo
  bucket tín hiệu nội tại từ chính inference candidate đã chạy, không chạy Vintern lần hai
  và không dùng self-confidence. Vintern chỉ override đúng region router v2 khi guard PASS
  và empirical bucket accuracy lớn hơn confidence EasyOCR gốc; toàn bộ quyết định có audit
  trail ngoài SQLite. Snapshot materialize kết quả này mang tier
  `easyocr_vintern_calibrated`; snapshot EasyOCR-only cũ vẫn bất biến.
- **28/08/2026 (bản 16)** — Pre-register Gate A CRAFT threshold pilot trên 300 frame cân
  bằng 60 unique shot cho mỗi dev video. Bắt buộc human-ground-truth region recall `>=0,98`
  và text-frame recall `>=0,99`; chỉ chọn cấu hình nhẹ nhất trong tập đạt recall, nếu không
  đạt thì giữ threshold recall hiện tại và cấm mở Gate B. Full dev5 rerun sau Gate A mới là
  evidence để tính ETA Tầng 1–3.
- **28/08/2026 (bản 17)** — Theo quyết định deadline của người dùng, thay Gate A 300 nhãn
  bằng emergency contract 100 nhãn đã hoàn thành (60 V001 + 40 V002), pin exact UID-set.
  Cả ba config đều fail region recall 98%; giữ `recall_current` bằng decision riêng
  `DEADLINE_OVERRIDE_KEEP_CURRENT` để chạy full dev5 Gate B. Không gọi đây là balanced PASS,
  không hạ recall gate và ghi rõ không đo được no-text false-positive trên mẫu này.
- **28/08/2026 (bản 18)** — Theo quyết định deadline, Gate B vẫn chạy model trên đủ 4.164
  frame dev5 nhưng giảm human calibration Vintern xuống đúng 100 candidate/100 distinct
  frame, cân bằng 20/video và stratify deterministic. Evidence mang tier
  `emergency_single_annotator_100`, không được gọi là standard 300-frame PASS. Bucket
  fine/structural cần support `>=20`; cấm dùng global bucket để override. Candidate thiếu
  support, guard fail, thiếu result hoặc calibrated confidence không hơn EasyOCR đều giữ
  EasyOCR và được đếm vào Gemini residual để báo exact cost trước Tầng 4.
- **28/08/2026 (bản 19)** — Người dùng quyết định loại hai crop Gate B thật sự unreadable
  thay vì chạy replacement review. Pool vẫn pin đúng 100 row và mọi ngưỡng an toàn giữ
  nguyên, nhưng calibration dùng 98 labeled region/98 distinct frame. Policy schema v3 cho
  phép đúng hai `exclude_unreadable`, cấm exclusion thứ ba và vẫn cấm global-bucket override.
- **28/08/2026 (bản 20)** — Chốt incremental OCR snapshot schema v2 để Nhánh 2 nhận dữ liệu
  khi bốn worker hoàn thành lệch tầng. Coverage ghi tier/count/checksum/UID hash riêng từng
  batch; `craft_only` không vào FTS. Worker không ghi chung SQLite: JSONL shard được union
  fail-closed theo partition catalog, duplicate/foreign UID bị từ chối, rồi local atomic-build
  một snapshot mới gồm đúng SQLite/coverage/checksum. Dev5 không được append trùng batch-01.
- **28/08/2026 (bản 21)** — Theo quyết định trực tiếp của người dùng, Vintern không còn là
  barrier bắt buộc trước pre-Gemini. Preflight yêu cầu đủ chín archive EasyOCR; batch có
  Vintern hoàn chỉnh vẫn dùng calibrated override để giảm chi phí, batch chưa có Vintern
  chuyển toàn bộ router-v2 candidate thẳng sang Gemini với provenance
  `vintern_not_available`. Không được giả kết quả Vintern, không bỏ EasyOCR fallback và
  Gemini vẫn khóa sau exact count/cost + paid canary + duyệt riêng của người dùng.
- **28/08/2026 (bản 22)** — Tạm chốt nhánh OCR ở tầng CRAFT+EasyOCR production 9/9, đúng
  293.336 keyframe đã verify trên private HF Dataset. Vintern chưa chạy, Gemini chưa gọi.
  Pin evidence exact pre-Gemini (830.301 region, 253.177 frame, 92.768 request) và cấm mở
  full production bằng runner Standard khi estimate 651.803 VND vượt budget; estimate Batch
  325.902 VND chỉ là cơ sở cho quyết định/implementation Batch riêng. Cho phép Nhánh 2 dùng
  snapshot tier `easyocr` với `complete=false`, `production_ready=false`.
- **28/08/2026 (bản 23)** — Sửa regression Online KIS sau phục hồi worktree: anchor
  retrieval chuyển sang merge/boost-only, VLM không còn phạt frame bị bỏ qua trong partial
  structured output, và Top 100 dùng weighted round-robin ngay sau seed Top 5 thay vì chèn
  liền 30 frame của video đầu. Diagnostic Q8 xác nhận raw FAISS đã tìm đúng
  `L23_V021/frame 6471`; bỏ hard-dedup một frame/shot khỏi candidate KIS vì frame `6471`
  từng bị frame `6480` cùng shot thay thế. Bổ sung `caption_en` làm faithful query bắt buộc vì code cũ chỉ
  search query expansion/scene/constraint. Vì vậy không rebuild embedding cho lỗi thuộc
  Online Core này.
- **28/08/2026 (bản 24)** — Tích hợp handoff OCR EasyOCR vào Online bằng snapshot path tường
  minh `AIC_OCR_SNAPSHOT_DIR`, không copy/alias thành `$AIC_DATA/index/ocr.sqlite`. Registry
  bắt buộc verify ba file snapshot, SHA-256, catalog/count/UID-set, FTS5/integrity và join
  `(video_id, keyframe_uid)`; UI/provenance phải hiện snapshot ID, tier, coverage, error và
  `production_ready=false`. Snapshot local từ HF revision
  `a5dcff74326f43421553481793d4a1e51eb59ce5` phủ 293.336/293.336 UID, 873 video và có
  278.091 FTS row; 57 frame CRAFT-detected nhưng EasyOCR text rỗng vẫn giữ `error`, không hạ
  gate hoặc gọi artifact này là final. Planner đồng thời nhận diện cụm visible-text dạng
  “giá dầu mazut được hiển thị” và tìm OCR bằng literal discriminative thay vì toàn câu.
- **28/08/2026 (bản 25)** — Hợp nhất Online Accuracy-Max vào nguồn chuẩn duy nhất: bỏ
  contract Min-Max/window-first/BLIP-2/FastAPI cũ khỏi §3, khóa `OnlineEngine` + Streamlit
  trực tiếp, video-first evidence ranking, VLM verification, ba task head, Top 100 portfolio
  và submission profile `AIC26_QUALIFIER_OFFICIAL`. FTS khóa thứ tự exact → AND → prefix →
  fuzzy; prefix lấy pool 5.000 rồi rerank token coverage để từ phổ biến không lấn clue hiếm.
  `CURRENT_STATUS.md` đổi từ append-only log sang snapshot hiện hành; README không còn xem
  spec tách nhánh/tài liệu archived là instruction.
- **29/08/2026 (bản 26)** — Theo xác nhận của người dùng, xóa implementation legacy
  `backend/`, `frontend/`, 12 tài liệu archived và ZIP recovery; runtime duy nhất còn lại là
  `online/` + Streamlit trực tiếp `OnlineEngine`. Khóa data-retention contract theo ba lớp:
  runtime core, accuracy/review media và rebuild/resume-only; MP4 không chặn vector retrieval
  nhưng thiếu MP4 làm mất playback/exact-frame refinement.
- **29/08/2026 (bản 27)** — Chuyển Unified Query Planner sang contract role-aware đa vai trò
  `VIDEO_LOCATOR/TARGET_MOMENT/ANSWER_EVIDENCE/ORDERED_EVENT` và API hai pha
  `plan() → operator review → search(plan)`. Video score tách locator/target; locator-only
  frame không đi vào KIS/QA submission rank và TRAKE context không thể trở thành event. QA
  luôn thử Top 3, ngưỡng 0,85 chỉ auto-accept; unknown OCR/ASR được đọc theo evidence UID +
  neighbors thay vì FTS cả câu hỏi, candidate `requires_review` bị chặn export. Schema cũ
  chỉ còn adapter migration; Online config nâng schema v3.
