# Bối cảnh dự án LASTDANCE

Đây là tài liệu bối cảnh chuẩn để thành viên mới hoặc AI hiểu dự án trước khi sửa
code. Trạng thái vận hành đo gần nhất nằm trong
`docs/model_first_runtime_report_2026-08-21.md`.

## 1. Bài toán

LASTDANCE hỗ trợ ba dạng truy vấn video của vòng sơ tuyển AIC2026:

- **KIS**: người dùng nhập một mô tả tự nhiên và hệ thống xếp hạng các
  `(video_id, frame_id)` phù hợp.
- **QA**: truy vấn chứa cả mô tả sự kiện và câu hỏi. Hệ thống phải tìm đúng video,
  đúng thời điểm rồi mới sinh câu trả lời.
- **TRAKE**: truy vấn mô tả nhiều khoảnh khắc có thứ tự. Hệ thống phải tìm đúng
  video và một frame cho từng khoảnh khắc.

Trong giao diện thi, người dùng chỉ nhập nguyên văn câu truy vấn. Backend chịu
trách nhiệm hiểu query, tìm kiếm, kiểm chứng, xếp hạng và trả tối đa 100 kết quả.

## 2. Mục tiêu đánh giá

Theo tài liệu vòng sơ tuyển trong `docs/Thong tin vong So tuyen AIC2026.pdf`, hệ
thống được quan tâm tại các mốc `1, 5, 20, 50, 100`. Vì vậy:

- Top 1 phải ưu tiên hypothesis có bằng chứng đầy đủ nhất.
- Top 5/20 phải phủ những video/segment hợp lý khác, không lặp vô ích.
- Top 50/100 giữ recall và đa dạng nhưng vẫn phải xếp evidence mạnh lên trước.
- QA chỉ đúng khi video/frame và answer cùng đúng.
- TRAKE sai video nhận điểm rất thấp hoặc bằng không; đúng video nhưng lệch một số
  moment có thể nhận điểm từng phần.

`backend/app/evaluation/official_metric.py` là implementation chuẩn của công thức
đánh giá dùng trong kiểm thử offline.

## 3. Vì sao một vector hoặc một frame là chưa đủ

Truy vấn thực tế thường mô tả một scene video, không phải một object đơn lẻ. Một
query có thể đồng thời chứa:

- nhiều nhân vật và vật thể;
- màu sắc, số lượng và quan hệ không gian;
- hành động hoặc thay đổi trạng thái;
- nhiều cảnh theo thứ tự;
- chữ xuất hiện trên màn hình;
- câu hỏi chỉ trả lời được ở một thời điểm cụ thể.

Một keyframe có thể khớp “xe hơi” nhưng bỏ sót “màu đỏ”, hành động hoặc cảnh sau.
Do đó kiến trúc gom nhiều evidence frame/window theo video trước khi rerank toàn
bộ query.

## 4. Dữ liệu và định danh

Dataset không nằm trong Git. Cấu trúc chuẩn:

```text
data/
  videos/<video_id>.mp4
  keyframes/<video_id>/<NNN>.jpg
  objects/<video_id>/<NNN>.json
  features/<video_id>.npy
  map-keyframes/<video_id>.csv
  metadata/<video_id>.json
  index/
```

Dataset tham chiếu hiện có 873 video và 177.321 keyframe.

Hai chỉ số không được nhầm:

| Trường | Ý nghĩa | Nơi sử dụng |
|---|---|---|
| `local_idx` | Số thứ tự keyframe trong một video | Đọc JPG/object/OCR nội bộ |
| `frame_id` | Frame thật trong MP4, từ `frame_idx` của map CSV | API kết quả và file nộp |

`keyframe_index.json` chứa đường dẫn tuyệt đối. Khi chuyển project sang thư mục
khác trên máy khác, phải chạy lại `app.indexing.build_index`.

## 5. Nguyên tắc kiến trúc model-first

1. **Model hiểu query**: Qwen sinh structured plan gồm scene, retrieval caption,
   `must_have`, visible text, temporal edge và repair query.
2. **Retriever mở rộng recall**: CLIP là đường production; OCR/object và side
   indexes chỉ bổ sung evidence.
3. **Video/window hypothesis**: nhiều frame cùng video được gom, làm mượt theo
   thời gian và kiểm tra storyboard.
4. **Model kiểm chứng**: multimodal model đánh giá toàn bộ điều kiện, không chỉ
   object nổi bật nhất.
5. **Repair có giới hạn**: khi pool đã kiểm chứng còn thiếu, planner sinh caption
   nhắm vào evidence bị bỏ sót rồi truy hồi thêm một vòng.
6. **Fallback rõ ràng**: lỗi model/index không được làm mất kết quả; hệ thống quay
   về đường retrieval/tournament đã kiểm thử.

Heuristic vẫn cần cho validation schema, mapping frame, giới hạn thời gian và
fallback. Không dùng heuristic để vá từng query mẫu.

## 6. Pipeline hiện tại

### KIS

```text
query
  → Qwen structured planner
  → CLIP/multilingual CLIP recall cho từng scene caption
  → optional SigLIP2 và Qwen-window recall nếu index complete
  → rank calibration + candidate union
  → object/OCR fusion khi planner yêu cầu visible evidence
  → temporal smoothing + ordered storyboard
  → dedicated Qwen reranker nếu local checkpoint đầy đủ
     hoặc Qwen instruct generative verifier theo nhóm
  → một repair retrieval round nếu evidence chưa đủ
  → chỉ ưu tiên verified pool khi có đủ 100 row
  → cutoff-aware ranking + exact-frame Top 1
```

API KIS trả thêm `model_relevance_score` và `model_verified` để phân biệt kết quả
đã được model nhìn thấy với retrieval fallback.

### QA

```text
query hoàn chỉnh
  → tách event description và question
  → KIS verified retrieval, không refine frame hai lần
  → temporal contact sheet cho các video dẫn đầu
  → Qwen kiểm tra event, chọn panel và trả answer
  → chuẩn hóa ngôn ngữ/format
  → cutoff-aware Top 100
```

QA không được trả lời từ tên object hoặc OCR cache mà không nhìn evidence frame.

### TRAKE

```text
query
  → tách ordered moments
  → recall riêng cho mỗi moment
  → rank video theo coverage
  → K-best monotonic alignment
  → visual sequence rerank
  → coarse-to-fine exact-frame refinement
  → Top 100 sequence hypotheses
```

TRAKE chưa dùng đầy đủ structured planner/shared verifier mới. Đây là backlog sau
KIS và QA.

## 7. Model và trạng thái

| Vai trò | Model | Trạng thái |
|---|---|---|
| CLIP image space | `clip-ViT-B-32` | Production, feature organizer có sẵn |
| Vietnamese query tower | `clip-ViT-B-32-multilingual-v1` | Production, CPU |
| Planner/VQA/verifier | `Qwen/Qwen3-VL-2B-Instruct` | Production trên GPU 6 GB |
| Dedicated reranker | `Qwen/Qwen3-VL-Reranker-2B` | Mục tiêu; checkpoint local chưa đầy đủ |
| Frame side recall | `google/siglip2-base-patch16-256` | Code có, full index chưa build |
| Video-window embedding | `Qwen/Qwen3-VL-Embedding-2B` | Model đã tải; partial index bị vô hiệu hóa |
| OCR | EasyOCR CRAFT + `latin_g2` | Offline, có checkpoint; không chạy lúc thi |
| ASR | Chưa chọn runtime production | Tạm hoãn |

Không coi tên model trong cấu hình là bằng chứng model đang được dùng. Kiểm tra
`/health`, state `complete` và runtime report.

## 8. Giới hạn phần cứng và chi phí

Máy tham chiếu: i5-12450H, RTX 4050 Laptop 6 GiB, Windows, Python 3.11,
PyTorch CUDA. Qwen instruct dùng khoảng 4,1–4,4 GiB VRAM nên:

- không chạy OCR/index builder cùng backend;
- dedicated embedding/reranker phải load tuần tự;
- model 8B không phù hợp local FP16;
- full video-window embedding có thể mất nhiều giờ dù online FAISS search nhanh.

Runtime đã đo, không phải accuracy:

- KIS HTTP Top 100: khoảng 94,9 giây, 100/100 row có model score;
- QA HTTP Top 100: khoảng 82,6 giây trên query smoke;
- 56/56 unit test pass tại checkpoint runtime.

## 9. Những gì chưa được chứng minh

- Chưa có dev set đủ lớn có ground truth trên dataset hiện tại.
- Smoke test chứng minh contract, latency và khả năng chạy; không chứng minh mọi
  Top 1 đều đúng.
- Side index chưa được A/B bằng Recall@1/5/20/50/100.
- Generative relevance score không hiệu chuẩn như xác suất; dedicated reranker vẫn
  là hướng tốt hơn khi có checkpoint và benchmark.
- OCR chưa hoàn tất toàn collection; ASR chưa có.

## 10. Thứ tự phát triển

1. Giữ KIS production ổn định và tạo dev set ground truth.
2. Tải/benchmark dedicated reranker, chỉ thay verifier khi cải thiện rõ ràng.
3. A/B SigLIP2 trên subset; không full build nếu recall không tăng.
4. Chỉ resume Qwen-window index khi subset chứng minh lợi ích so với chi phí.
5. Thêm answer verification cho QA.
6. Đưa shared planner/verifier vào TRAKE.
7. Bổ sung ASR timestamped sau cuộc thi.

