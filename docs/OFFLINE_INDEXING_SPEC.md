# Nhánh 1 — Offline Indexing Pipeline (Bản chốt)

> File này là nguồn chuẩn riêng cho **Offline Indexing**, trích và sửa từ `BASELINE_SPEC.md`.
> Đây là artifact mà Nhánh 1 phải bàn giao cho Nhánh 2 (Online Retrieval) — Nhánh 2 chỉ được
> code dựa trên đúng schema và điều kiện nghiệm thu mô tả ở đây, không tự suy đoán thêm.

**Cập nhật:** 23/08/2026
**Mục tiêu:** chuyển 500GB video thô → ~80GB keyframe + <5GB vector/text index, nạp vừa RAM máy local.

---

## 0. Nguyên tắc bắt buộc

1. Mỗi keyframe có vector riêng — **không** mean-pooling nhiều keyframe/shot thành 1 vector
   (làm mất khả năng trả đúng `frame_id` cụ thể mà KIS yêu cầu).
2. Ba FAISS index (CLIP/SigLIP/EVA-CLIP) dùng khóa **`keyframe_uid`** (deterministic hash, xem
   mục 3.2a) qua `IndexIDMap` — **không** dùng vị trí insert (`row index`) làm khóa. Nhờ vậy
   3 worker/máy khác nhau có thể build 3 modality độc lập, không đồng bộ thời gian/thứ tự.
3. Không hardcode path tuyệt đối — mọi đường dẫn build từ biến môi trường `AIC_DATA`.
4. Không giả định FPS/resolution/duration — lấy từ bước Inventory/EDA thật.
5. Một index chỉ được đánh dấu "Ready" khi thỏa đủ **Publishing Criteria** ở mục 5 — không
   được set `complete=true` khi còn checkpoint dở dang.

---

## 1. Tầng Phân đoạn & Trích chọn (Preprocessing Stage)

Chạy **Local CPU** — tiết kiệm GPU quota Kaggle cho tầng embedding.

| Bước | Model/Thư viện | Hành động | Output |
|---|---|---|---|
| 1.1 Shot Detection | **AutoShot** | Phân tích thay đổi ngữ nghĩa theo thời gian để chia video thành shot đồng nhất nội dung | `.json`: `[shot_id, start_frame, end_frame]` |
| 1.2 Keyframe Extraction | **FFmpeg** | Trích 3 keyframe đại diện/shot (Begin - Middle - End), **không** cắt window cố định | Ảnh `.jpg` (quality 80–90) |
| 1.3 Lọc nhiễu | **OpenCV** (Laplacian Variance) | Loại khung hình bị mờ (blur) | Tập keyframe đã lọc |
| 1.4 Lọc trùng | **pHash** hoặc **Cosine > 0.9** | Xóa khung gần như giống hệt trong cùng shot | Tập keyframe tối giản (mục tiêu: giảm ~80% dung lượng thô) |

---

## 2. Tầng Trích xuất Đặc trưng Đa phương thức (Extraction Stage)

Chạy **Kaggle GPU** theo batch.

| Bước | Model | Hành động | Output |
|---|---|---|---|
| 2.1 Visual Embedding | **CLIP ViT-B/32** (baseline/rollback) + **SigLIP** + **EVA-CLIP** | Chạy song song 3 model, lấy vector cho **từng keyframe riêng lẻ**. **Bắt buộc ép về `float16`** trước khi lưu file (giảm 50% dung lượng, giảm băng thông push/pull HF Dataset — xem mục 6) | `.npy`/`.bin`, dtype `float16` |
| 2.2 OCR Extraction | **Gemini 2.5 Flash-Lite API** (primary) / **EasyOCR** (fallback offline) | Đọc chữ tĩnh trên keyframe (biển hiệu, logo, chữ chạy) theo JSON prompt schema cố định — dùng chung `OcrResult` với `BASELINE_SPEC.md` §2.2 | `.json`/keyframe: `{"frame_id": ..., "detected_text": [...], "bbox": [...], "confidence": 0.95, "language": "vi"}` |

---

## 3. Tầng Lập chỉ mục & Công bố (Indexing Stage)

Chạy **Local**.

| Bước | Công cụ | Hành động | Output |
|---|---|---|---|
| 3.1 Unified Catalog | Python (Pandas) | Ánh xạ toàn bộ metadata frame vào catalog trung tâm duy nhất — xem schema đầy đủ ở mục 3.1a | **`frames.csv`** |
| 3.2 Vector DB Build | FAISS `IndexFlatIP` + `IndexIDMap` | Chuẩn hóa L2, build 3 index **độc lập** — khóa bằng `keyframe_uid`, không cần cùng thứ tự (xem 3.2a) | `clip.faiss`, `eva_clip.faiss`, `siglip.faiss` |
| 3.3 Text DB Build | SQLite FTS5 | Nạp dữ liệu OCR vào bảng ảo hỗ trợ Full-Text Search (BM25) — schema ở mục 3.3a | `ocr.sqlite` |

### 3.3a Schema SQL `ocr.sqlite` — cấu trúc chuẩn, đối chiếu với `asr.sqlite` (`ASR_SPEC.md` §3.2)

```sql
CREATE VIRTUAL TABLE ocr_fts USING fts5(
    video_id UNINDEXED,
    keyframe_uid UNINDEXED,
    detected_text,
    language UNINDEXED,
    confidence UNINDEXED
);
```

### 3.1a Schema `frames.csv` — đầy đủ, dùng `keyframe_uid` thay cho row-index

```python
# shared/schemas/frame.py
from pydantic import BaseModel

class FrameRecord(BaseModel):
    video_id: str
    local_idx: int       # vị trí keyframe nội bộ — dùng để trỏ đúng file ảnh .jpg trên đĩa
    frame_id: int         # số frame thật trong video — dùng cho preview & submission
    pts_time: float        # timestamp chuẩn (giây) — trục join giữa các modality
    shot_id: str           # định danh shot (từ AutoShot)
    window_id: str | None = None   # chỉ điền nếu còn giữ window-based retrieval
    keyframe_uid: int      # khóa dùng chung cho cả 3 FAISS index + OCR + ASR (xem 3.2a)
```

**Quy tắc đặt tên file ảnh:** `{video_id}/{shot_id}_{local_idx}.jpg` — `local_idx` bắt buộc
phải tồn tại vì `frame_id` (số frame thật trong mp4) **không** dùng để đặt tên file ảnh trên
đĩa. Thiếu `local_idx`, module đọc ảnh ở Nhánh 2 sẽ phải đoán ngược tên file → dễ lỗi.

### 3.2a `keyframe_uid` — khóa nội dung (content-addressed), không phụ thuộc thứ tự build

Thay vì dùng vị trí insert (`faiss_row_id`) làm khóa — vốn buộc 3 FAISS index phải build
đồng thời, cùng thứ tự, dễ vỡ khi phân tán trên nhiều Kaggle worker — dùng **ID xác định
(deterministic)** tính từ nội dung, để bất kỳ worker nào tính cũng ra cùng một số:

```python
import faiss
import hashlib

def make_keyframe_uid(video_id: str, shot_id: str, local_idx: int) -> int:
    raw = f"{video_id}:{shot_id}:{local_idx}"
    h = hashlib.blake2b(raw.encode(), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False) >> 1  # ép dương, vừa int64

index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
index.add_with_ids(vectors, keyframe_uids)  # không quan tâm thứ tự add
```

**Lợi ích so với ràng buộc thứ tự cũ:**
- 3 index (`clip.faiss`, `eva_clip.faiss`, `siglip.faiss`) build độc lập, không đồng bộ thời
  gian/thứ tự — mỗi worker Kaggle chạy xong modality nào add luôn modality đó.
- Thiếu 1 modality cho 1 video: phát hiện bằng cách diff tập `keyframe_uid` trong file
  `.faiss` với tập trong `frames.csv` thuộc đúng `video_id` — không lan ra toàn catalog.
- Sửa lỗi thiếu sót: chỉ cần add bổ sung đúng `keyframe_uid` còn thiếu, **không phải rebuild
  toàn bộ** index.
- Nhánh 2 có thể query được ngay cả khi 1 modality chưa build xong (file `.faiss` đó đơn
  giản là chưa có ID đó, không gây lỗi ngầm kiểu `null`).

---

## 4. Data Contract — bàn giao cho Nhánh 2

1. **`frames.csv`** — trục join bắt buộc, schema đầy đủ ở mục 3.1a.
2. **`clip.faiss` / `eva_clip.faiss` / `siglip.faiss`** — index frame-level, khóa bằng
   `keyframe_uid` qua `IndexIDMap` (mục 3.2a), độc lập thời gian build.

   > **Lưu ý ranh giới trách nhiệm:** Nhánh 1 chỉ bàn giao **3 file `.faiss` độc lập**, không
   > build sẵn 1 index visual đã gộp. Việc gộp 3 điểm số (SRRF cho SigLIP+EVA-CLIP, CLIP làm
   > rollback khi thiếu index) là việc của Nhánh 2 tại thời điểm query — xem
   > `BASELINE_SPEC.md` §3.2 tầng 1. Không cần thêm bước xử lý nào ở Nhánh 1 cho việc này.
3. **`ocr.sqlite`** — toàn bộ text OCR, tìm kiếm BM25.
4. **Relative Path Convention** — mọi path build từ `AIC_DATA`, không hardcode tuyệt đối.

---

## 5. Publishing Criteria — điều kiện để index được coi là "Ready"

Một bộ index chỉ được Nhánh 2 sử dụng khi thỏa **toàn bộ** điều kiện sau (không được bỏ bớt
để tiết kiệm thời gian):

- [ ] `complete = true` (điền theo từng `video_id`, không phải toàn catalog — một video có
      thể publish xong CLIP trước, SigLIP/EVA-CLIP sau)
- [ ] Với mỗi `video_id` đã đánh dấu complete: tập `keyframe_uid` trong `frames.csv` khớp
      100% với tập ID đã add vào **cả 3** file FAISS (diff bằng code, không đếm số dòng thô)
- [ ] Không có `NaN`/`Inf` trong bất kỳ vector nào
- [ ] Norm vector ≈ 1 sau khi chuẩn hóa L2 (kiểm tra sample ngẫu nhiên, không cần kiểm tra 100%)
- [ ] Mapping `video_id`/`frame_id`/`pts_time` đã xác thực qua Sanity Check (đối chiếu vài
      chục dòng ngẫu nhiên với video gốc bằng mắt)
- [ ] Cơ chế checkpoint/resume hoạt động đúng (test bằng cách ngắt giữa chừng 1 batch rồi
      chạy lại, không bị duplicate hoặc mất dữ liệu)

File đang preallocate hoặc checkpoint dở dang **không** được set `complete=true`.

---

## 6. Đồng bộ artifact qua HuggingFace Dataset

Kaggle build index (GPU) → cần chuyển về máy local (RTX 4050) để chạy Online Retrieval lúc
thi. Dùng **HuggingFace Dataset** (Git LFS) làm kho trung gian.

```
Kaggle Notebook (build) --push_to_hub()--> HF Dataset (Git LFS) --snapshot_download()--> Local máy
```

**Bắt buộc để tránh bị throttle băng thông (tài khoản free có soft limit):**

- [ ] Toàn bộ vector `.npy`/`.bin` đã ép `float16` trước khi push (mục 2.1) — giảm 50%
      dung lượng mỗi lần push/pull.
- [ ] **Không push từng video một** — gom theo batch (ví dụ mỗi 50–100 video hoặc cuối mỗi
      phiên Kaggle) rồi push một lần, giảm số lượt gọi API HF.
- [ ] Chỉ pull về máy local những gì cần dùng ngay (tránh `snapshot_download()` full repo
      nhiều lần trong ngày thi — pull 1 lần trước khi thi, không pull lại giữa chừng trừ khi
      có patch khẩn).
- [ ] Đặt tên revision/commit rõ ràng theo batch (ví dụ `batch-01`, `batch-02`) để dễ rollback
      nếu 1 lần push bị lỗi giữa chừng, không phải re-push toàn bộ dataset.

---

## Changelog

- **23/08/2026** — Tách riêng thành file offline indexing độc lập từ `BASELINE_SPEC.md`.
  Bổ sung lại `local_idx`/`window_id` vào schema `frames.csv`. Thay `faiss_row_id`
  (positional, dễ vỡ khi build phân tán) bằng `keyframe_uid` (deterministic hash qua
  `IndexIDMap`) — cho phép 3 FAISS index build độc lập, không đồng bộ thứ tự/thời gian, sửa
  lỗi thiếu sót bằng cách add bổ sung thay vì rebuild toàn bộ. Thêm bước ép `float16` bắt
  buộc ở Visual Embedding. Khôi phục đầy đủ 6 điều kiện Publishing Criteria. Thêm mục 6 quy
  trình đồng bộ qua HuggingFace Dataset (batch push, revision naming, tránh throttle).
- **23/08/2026 (bản 2)** — Sửa xung đột phát hiện khi audit chéo 3 file: xóa câu ràng buộc
  "build cùng thứ tự" còn sót lại trong bảng §3 (mâu thuẫn với §0/§3.2a); đồng bộ tên field
  JSON OCR (`detected_text`, `confidence`) khớp `OcrResult` trong `BASELINE_SPEC.md`; thêm
  schema SQL `ocr_fts` (mục 3.3a) để đối chiếu trực tiếp với `asr_fts` trong `ASR_SPEC.md`.
- **23/08/2026 (bản 3)** — `BASELINE_SPEC.md` bổ sung tầng "intra-visual fusion" (SRRF) ở
  Nhánh 2 (không phải Nhánh 1). Thêm ghi chú ở mục 4 làm rõ Nhánh 1 **không** cần build thêm
  index gộp — chỉ bàn giao đúng 3 file `.faiss` độc lập như cũ, tránh người phụ trách Nhánh 1
  làm trùng việc với Nhánh 2.
- **26/08/2026 (bản 4)** — Loại vĩnh viễn BEiT-3 và thay modality thứ ba bằng EVA-CLIP.
  Contract bàn giao đổi trực tiếp từ `beit3.faiss` sang `eva_clip.faiss`; công thức
  `keyframe_uid`, `IndexIDMap`, khả năng build độc lập và toàn bộ Publishing Criteria giữ
  nguyên.
