# Bối cảnh dự án LASTDANCE

> **ARCHIVED 24/08/2026:** Tài liệu này mô tả kiến trúc Qwen video-window cũ. Nguồn chuẩn
> hiện tại là `BASELINE_SPEC.md`, `OFFLINE_INDEXING_SPEC.md` và `ASR_SPEC.md`.

LASTDANCE hỗ trợ người tham gia AIC2026 tìm video bằng câu truy vấn tự nhiên. Đây
là tài liệu bối cảnh sản phẩm; kiến trúc chi tiết nằm trong
[`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md).

## 1. Ba bài toán

- **KIS**: tìm và xếp hạng `(video_id, frame_id)` phù hợp với mô tả.
- **QA**: tìm đúng video/time window rồi trả lời câu hỏi dựa trên evidence đó.
- **TRAKE**: tìm đúng video và một frame cho mỗi moment theo đúng thứ tự.

Người dùng chỉ nhập nguyên câu query. Backend chịu trách nhiệm planning, recall,
aggregation, verification, ranking và output Top 100.

## 2. Mục tiêu đánh giá

Thể lệ quan tâm các cutoff 1, 5, 20, 50 và 100. Vì vậy hệ thống phải đồng thời:

- đưa hypothesis mạnh nhất lên Top 1;
- giữ các video/window hợp lý trong Top 5/20;
- bảo toàn recall và đa dạng evidence ở Top 50/100;
- không lãng phí slot cho frame gần trùng nếu chúng không thêm evidence;
- map đúng `frame_id` thật khi nộp.

`backend/app/evaluation/official_metric.py` là implementation metric offline.
Chỉ báo Recall@k khi dev set có ground truth tương ứng.

## 3. Bản chất query

Query thực tế thường mô tả một scene hoặc diễn biến video, gồm nhiều thành phần:

- nhân vật, object, số lượng và thuộc tính;
- màu sắc, quan hệ không gian và bối cảnh;
- hành động và thay đổi trạng thái;
- nhiều scene theo thứ tự;
- chữ trên màn hình, logo hoặc tên riêng;
- câu hỏi chỉ trả lời được tại một thời điểm.

Một frame có thể khớp object nhưng thiếu hành động; một window có thể khớp hành
động nhưng thiếu scene khác. Vì thế hệ thống phải tìm và tổng hợp **một tập
evidence theo thời gian**, không dùng một vector đại diện toàn video.

## 4. Quyết định hướng phát triển

Hướng chuẩn là **offline video evidence indexing + window-first model retrieval**:

1. Offline tạo manifest, keyframe map, shot boundary, frame embedding,
   video-window embedding, structured caption, OCR và object evidence.
2. Qwen planner biến query thành scene/evidence plan có schema.
3. Frame retriever và video-window retriever tạo candidate bổ sung cho nhau.
4. Candidate được gom thành video hypothesis theo scene coverage và temporal order.
5. Multimodal reranker/verifier chấm toàn bộ query với evidence window.
6. Repair retrieval chỉ chạy khi model chỉ ra evidence còn thiếu.
7. QA và TRAKE tái sử dụng verified retrieval thay vì xây recall riêng yếu hơn.

KIS, QA và TRAKE sẽ dùng cùng `UnifiedQueryPlan`; khác biệt chỉ nằm ở task type và
output. Chi tiết schema/migration nằm trong [`QUERY_PROCESSING.md`](QUERY_PROCESSING.md).

OCR không phải trọng tâm duy nhất. Nó giải quyết visible text; video embedding giải
quyết hành động/diễn biến; caption hỗ trợ lexical semantics; object hỗ trợ entity;
shot/timestamp hỗ trợ temporal reasoning; ASR sau này giải quyết lời nói.

## 5. Dữ liệu và định danh

Dataset tham chiếu có 873 video và 177.321 keyframe:

```text
data/
  videos/
  keyframes/
  objects/
  features/
  map-keyframes/
  metadata/
  index/
```

| Trường | Ý nghĩa |
|---|---|
| `video_id` | định danh video |
| `local_idx` | thứ tự keyframe, dùng để đọc JPG/object/OCR |
| `frame_id` | frame thật trong MP4, dùng cho API chính xác/submission |
| `pts_time` | timestamp để join evidence và align moment |
| `window_id` | định danh một cửa sổ thời gian trong video |

`keyframe_index.json` có đường dẫn tuyệt đối nên máy mới phải rebuild base index.

## 6. Vai trò model

| Model | Vai trò |
|---|---|
| organizer CLIP ViT-B/32 | baseline frame recall nhanh |
| multilingual CLIP text tower | query tiếng Việt trong cùng image space |
| Qwen3-VL-2B-Instruct | planner, verifier, VQA, offline structured caption |
| Qwen3-VL-Embedding-2B | text/video-window embedding chung |
| Qwen3-VL-Reranker-2B | query–video/window cross-modal rerank |
| SigLIP2 base | optional complementary frame recall |
| EasyOCR CRAFT + latin_g2 | visible-text extraction |

Model được chọn theo vai trò, phần cứng, latency và metric; không theo độ nổi
tiếng. Một model score là evidence, không phải xác suất đúng đã hiệu chuẩn. Xem
quyết định và challenger trong [`MODEL_SELECTION.md`](MODEL_SELECTION.md).

## 7. Pipeline đích

### KIS

```text
query plan
  → multi-channel frame/window recall
  → calibrated fusion
  → video/scene/temporal aggregation
  → model rerank + verification
  → one bounded repair round
  → cutoff-aware Top 100
  → exact-frame refinement
```

### QA

```text
query plan
  → shared verified video-window retrieval
  → temporal context
  → grounded VQA
  → answer verification
  → Top 100
```

### TRAKE

```text
ordered moment plan
  → shared window recall per moment
  → video coverage ranking
  → monotonic alignment
  → sequence verification
  → exact-frame refinement
  → Top 100 hypotheses
```

## 8. Giới hạn phần cứng

Máy tham chiếu: i5-12450H, RTX 4050 Laptop 6 GiB, Windows, Python 3.11. Các
model 2B không cùng resident; builder và backend GPU chạy tuần tự. KIS có ngân
sách 2–3 phút, QA/TRAKE 3–5 phút, nhưng mọi milestone vẫn phải đo P50/P95.

## 9. Trạng thái và giới hạn hiện tại

- CLIP, KIS planner, generative verifier, repair và KIS/QA basic đang hoạt động.
- Unified planner cho cả QA/TRAKE là target; hai pipeline này còn parser riêng.
- Video-window builder có nhưng index mới 300/29.938 và bị vô hiệu vì partial.
- Dedicated reranker chưa active; runtime dùng generative verifier.
- SigLIP2 chưa có full index.
- OCR collection chưa xác nhận complete; ASR hoãn.
- TRAKE chưa dùng shared window/verifier.
- Chưa có dev set đủ lớn để chứng minh accuracy toàn collection.

Xem số liệu mới nhất trong [`CURRENT_STATUS.md`](CURRENT_STATUS.md) và thứ tự
phát triển trong [`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md).
