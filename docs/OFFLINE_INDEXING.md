# Thiết kế offline indexing

> **ARCHIVED 24/08/2026:** Thiết kế Qwen video-window cũ đã bị thay thế. Xem
> `BASELINE_SPEC.md` §2 cho pipeline frame-level hiện hành.

Offline indexing phải biến video thô thành một kho evidence có timestamp và
provenance. Mục tiêu không phải tạo một “vector toàn video”, mà tạo nhiều biểu
diễn bổ sung cho nhau để query-time có thể tìm cảnh, hành động, chữ, tên riêng,
object, quan hệ và chuỗi sự kiện.

Quyết định model và các challenger được khóa trong
[`MODEL_SELECTION.md`](MODEL_SELECTION.md); không đổi model trực tiếp trong builder
chỉ vì benchmark nhà phát hành cao hơn.

## 1. Evidence cần trích xuất

| Evidence | Dùng để tìm | Cách biểu diễn | Trạng thái |
|---|---|---|---|
| Source manifest | tính toàn vẹn video, duration, FPS | JSON/SQLite metadata | Cần chuẩn hóa |
| Keyframe map | frame quan sát và frame thật | `local_idx`, `frame_id`, `pts_time` | Đã có |
| Shot/scene boundary | tránh ghép frame khác cảnh, tạo window hợp lý | interval timestamp | Cần triển khai |
| Frame embedding | object/cảnh tĩnh/thuộc tính nổi bật | CLIP 512, optional SigLIP2 | CLIP có; SigLIP2 chưa build |
| Video-window embedding | hành động, quan hệ, nhiều scene gần nhau | Qwen3-VL Embedding 1024 | Builder có, index partial |
| Structured window caption | lexical retrieval, debug, planner grounding | JSON + text index | Cần triển khai |
| OCR | biển hiệu, phụ đề, tên riêng, số | line text + confidence + box | Builder có, chưa full |
| Object evidence | object, count thô | labels/counts/scores | Đã có cache |
| ASR | lời nói, tên riêng, nội dung không nhìn thấy | segment text + timestamp | Nhánh 3 chạy song song; xem `BASELINE_SPEC.md` §2A |
| Exact-frame access | refine đáp án nộp | decode từ MP4 theo `frame_id` | Đã có |

Không nguồn nào tự nó là ground truth. Query-time planner quyết định evidence nào
cần, retrieval hợp nhất kết quả đã hiệu chuẩn, và verifier nhìn lại frame/window.

## 2. Canonical data contract

Mọi artifact phát sinh phải truy được về source:

```json
{
  "video_id": "L21_V001",
  "window_id": "L21_V001:w000042",
  "start_time": 81.2,
  "end_time": 96.8,
  "local_idxs": [41, 42, 43, 44, 45, 46],
  "frame_ids": [2030, 2215, 2400, 2584, 2768, 2952],
  "keyframe_paths": ["..."],
  "shot_ids": [12],
  "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
  "embedding_dim": 1024,
  "index_signature": "..."
}
```

Quy tắc bắt buộc:

- `local_idx` chỉ dùng để đọc artifact keyframe/object/OCR;
- `frame_id` là frame thật để preview/nộp bài;
- `pts_time` là chuẩn cho alignment giữa modality;
- mọi record có model/version/config signature;
- artifact partial không được publish như production.

## 3. Các stage offline

### Stage 0 — Inventory và validation

1. Quét video, keyframe, map CSV, feature và object JSON.
2. Kiểm tra video mở được, FPS/duration hợp lệ và path không trùng.
3. Kiểm tra mỗi keyframe có đúng row map; `frame_id` tăng hợp lý.
4. Sinh manifest chứa checksum hoặc size/mtime để phát hiện dataset thay đổi.

Nếu Stage 0 lỗi, dừng build downstream. Không tạo index trên dataset lệch.

### Stage 1 — Base frame index

`app.indexing.build_index` hợp nhất organizer feature, keyframe metadata và object
cache, sau đó tạo `clip.faiss`. Đây là baseline/fallback luôn phải giữ được.

### Stage 2 — Shot và scene segmentation

Tách shot bằng thay đổi histogram/content trên video hoặc keyframe sequence. Sau
đó gộp shot gần nhau thành event candidate theo thời gian. Boundary được dùng để:

- không tạo một window chứa hai nội dung không liên quan;
- lấy temporal context đúng scene cho QA;
- tăng chất lượng ordered alignment cho TRAKE;
- tạo window ngắn/trung bình theo nội dung thay vì chỉ theo số frame cố định.

Phiên bản đầu có thể dùng OpenCV content-difference có threshold được benchmark;
sau đó mới cân nhắc model shot-boundary chuyên dụng. Boundary là cấu trúc, không
phải quy tắc hiểu query.

### Stage 3 — Video-window embedding

Profile đầu tiên dùng 6 keyframe/window, stride 6. Khi có shot boundary, window
không vượt boundary trừ event window được đánh dấu rõ. Qwen embedding nhận chuỗi
ảnh + relative timestamp, ghi feature float16 và publish FAISS khi hoàn tất.

Sau baseline, thử multi-resolution bằng các artifact riêng:

- `short`: hành động cục bộ;
- `scene`: toàn một shot/scene;
- `event`: chuỗi nhiều bước trong khoảng dài hơn.

Không average các vector scale thành một vector video.

### Stage 4 — Structured window caption

Dùng `Qwen/Qwen3-VL-2B-Instruct` offline trên window đã chọn để sinh JSON giới
hạn schema:

```json
{
  "entities": [],
  "attributes": [],
  "actions": [],
  "relations": [],
  "setting": "",
  "visible_text_hint": [],
  "event_summary_vi": "",
  "event_summary_en": ""
}
```

Caption là evidence có thể tìm bằng lexical/BM25 và dense text embedding, đồng
thời giúp debug vì con người đọc được. Caption không thay thế visual embedding;
mọi field không chắc chắn cần confidence hoặc để trống. OCR text chính thức vẫn
lấy từ OCR pipeline, không dùng caption để bịa chữ.

Để kiểm soát chi phí, ưu tiên caption theo shot/window đại diện hoặc window được
retrieval log truy cập nhiều; không nhất thiết caption toàn bộ collection ngay.

### Stage 5 — OCR

EasyOCR CRAFT + `latin_g2` ghi từng line gồm text, confidence và polygon. Search
dùng fuzzy phrase matching tổng quát. OCR chạy offline, checkpoint được và không
được dùng như answer generator độc lập.

EasyOCR là baseline ổn định hiện tại. PP-OCRv5 Latin là challenger ưu tiên;
PP-OCRv6 chỉ benchmark sau khi kiểm tra dictionary dấu tiếng Việt và Paddle CUDA
trong environment riêng. Qwen/PaddleOCR-VL chỉ làm second-pass trên frame khó,
không thay OCR chuyên dụng cho toàn collection.

### Stage 6 — Object evidence

Chuẩn hóa organizer detection thành label, count và score theo keyframe/window.
Object chỉ là tín hiệu phụ vì detector có thể bỏ sót thuộc tính, hành động hoặc
nhận nhầm object.

### Stage 7 — ASR song song (Nhánh 3)

ASR sinh segment có `start_time`, `end_time`, `language`, `transcribed_text` và
`keyframe_uid_nearest`. Pipeline này chạy song song bằng tài khoản Kaggle/Colab riêng,
align về `frames.csv` theo timestamp rồi build `asr.sqlite` FTS5. Contract hiện hành và
Publishing Criteria nằm tại `BASELINE_SPEC.md` §2A; không dùng thiết kế window-first cũ
làm runtime instruction.

## 4. Artifact layout mục tiêu

```text
data/index/
  manifest.json
  keyframe_index.json
  clip.faiss
  objects_cache.json
  shots.json
  video_windows.json
  video_window_features.npy
  video_windows.faiss
  video_window_state.json
  window_captions.jsonl
  window_captions_lexical/
  ocr_cache.json
  ocr_state.json
  asr_segments.jsonl          # checkpoint/provenance trung gian Nhánh 3
  asr.sqlite                  # SQLite FTS5 bàn giao cho Online
  asr_coverage.csv            # trạng thái từng video
```

Mỗi builder cần `--limit`, `--checkpoint-every`, resume theo signature và atomic
rename khi publish. Nếu cấu hình/model/dataset signature thay đổi, build sang
artifact mới; không ghi tiếp lên checkpoint cũ.

## 5. Build order

```text
validate dataset
  → base keyframe/CLIP/object index
  → shot boundaries
  → video-window metadata
  → Qwen video-window embedding
  → optional structured captions
  → OCR
  → optional SigLIP2
  → ASR chạy song song trên GPU account riêng; alignment/FTS hoàn tất khi frames.csv sẵn sàng
```

Các job GPU chạy tuần tự. Có thể sửa code trong lúc builder chạy nếu không restart
process hoặc thay file mà builder đang import, nhưng không chạy backend/model GPU
khác cùng lúc trên máy 6 GiB.

## 6. Validation trước khi publish

Với mỗi index:

1. state `complete=true`, số vector bằng số metadata;
2. vector không NaN/Inf và norm nằm gần 1;
3. random sample mở đúng video/frame/window;
4. timestamp và frame tăng đúng thứ tự;
5. query smoke trả đúng schema và provenance;
6. A/B Recall@k trên dev subset;
7. đo dung lượng, throughput, peak VRAM và ETA full build;
8. thử cố ý dừng ngang và resume.

Chỉ sau các bước này mới bật index trong runtime hoặc ghi nó là production trong
`CURRENT_STATUS.md`.
