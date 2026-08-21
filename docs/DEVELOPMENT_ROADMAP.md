# Lộ trình phát triển LASTDANCE

Lộ trình này thay thế toàn bộ plan/checkpoint cũ. Mỗi milestone phải có baseline,
ground truth và điều kiện rollback; không đổi production chỉ vì một model mới tải
được hoặc một smoke query nhìn có vẻ tốt.

## M0 — Giữ baseline có thể vận hành

Trạng thái: hoàn thành.

- CLIP frame index và multilingual query recall hoạt động.
- Qwen structured planner, generative verifier và repair đã nối vào KIS.
- KIS/QA trả Top 100 theo schema; TRAKE có monotonic alignment.
- Optional index chưa hoàn chỉnh tự fail closed.
- Compile và unit test là gate bắt buộc.

Điều kiện giữ: mọi milestone sau phải chạy được A/B với baseline này.

## M1 — Thống nhất model-based query processing

1. Định nghĩa `UnifiedQueryPlan` và JSON schema dùng chung.
2. Tạo `plan_query(text, task_type)` bằng `Qwen3-VL-2B-Instruct`.
3. Đưa source-language và English CLIP caption vào cùng plan.
4. Model gắn scene, must-have, modality, temporal edge, QA answer request và
   TRAKE moments.
5. JSON validator chỉ giữ contract; lỗi model fallback nguyên query một scene.
6. Chuyển KIS, sau đó QA và TRAKE sang cùng plan/evidence contract.
7. Xóa parser/translator semantic riêng sau regression.

Gate chấp nhận: schema-valid ≥99%, scene/question/moment coverage không giảm,
không có query-specific rule và cả ba task dùng cùng entry point.

## M2 — Chuẩn hóa offline evidence và hoàn thành video-window embedding

Ưu tiên cao nhất.

1. Tạo manifest và validator cho video/keyframe/map/feature/object artifact.
2. Tạo shot/scene boundary để window không ghép nội dung rời rạc.
3. Tạo dev subset đại diện query ngắn, nhiều scene, hành động, chữ, màu/thuộc tính
   và tiếng Việt.
4. Gán ground truth video + khoảng frame/window, không chỉ một keyframe.
5. Chạy builder profile hiện tại trên subset: size 6, stride 6, dim 1024.
6. Kiểm tra checkpoint/resume, metadata, vector norm và mapping timestamp.
7. Publish index chỉ khi `next_index == total` và `complete=true`.
8. A/B CLIP-only với CLIP + Qwen-window tại các cutoff chính thức.
9. Nếu subset tăng recall và nằm trong ngân sách, resume full 29.938 window.
10. Sinh structured caption cho subset/window đại diện và đo phần recall bổ sung.

Gate chấp nhận:

- không giảm Recall@1/5 quá sai số thống kê;
- tăng recall video/window cho query nhiều ngữ nghĩa;
- 100% metadata map đúng video/local/frame/timestamp;
- index dở không bao giờ xuất hiện trong runtime;
- KIS P95 không vượt ngân sách 3 phút trên máy tham chiếu.

Nếu không đạt, giữ artifact để phân tích nhưng tắt window retrieval; thử profile
khác trên subset, không full-build mù.

## M3 — Hoàn thiện dedicated reranker

1. Tải đủ `Qwen/Qwen3-VL-Reranker-2B` ngoài request path.
2. Smoke riêng trên cặp positive, hard negative và partial match.
3. Đo peak VRAM, latency, score distribution và lỗi parse.
4. A/B với generative verifier hiện tại trên cùng candidate pool.
5. Hiệu chuẩn threshold/weight trên dev set, không trên query cuộc thi đang thi.

Gate chấp nhận: tăng Top 1/5, mọi candidate mục tiêu có relevance score, latency
KIS vẫn dưới 3 phút. Nếu không đạt, generative verifier tiếp tục là backend chính.

## M4 — Chuyển KIS sang window-first có kiểm chứng

1. Planner sinh whole-query caption, scene caption và must-have evidence.
2. CLIP và Qwen-window chạy song song về mặt logic, tuần tự về GPU.
3. Hợp nhất bằng calibrated rank fusion.
4. Gom window thành video hypothesis theo coverage và temporal edges.
5. Rerank top video/window, repair tối đa một vòng có mục tiêu.
6. Mở rộng verification cho tới Top 100 hoặc hết ngân sách; luôn ghi trạng thái.
7. Refine source frame chỉ sau khi chọn đúng video/window.

Gate chấp nhận: cải thiện mean official score và tỷ lệ đúng video; không phá
contract Top 100 hoặc mapping `frame_id`.

## M5 — QA dùng shared verified retrieval

1. Tách event description khỏi answer request bằng planner có schema.
2. Nhận verified window từ KIS thay vì tìm lại từ đầu.
3. Tạo temporal contact sheet quanh window.
4. VQA sinh answer cùng ngôn ngữ và đúng format.
5. Thêm answer verifier độc lập: answer phải được hỗ trợ bởi panel đã chọn.
6. Với chữ/tên riêng, đưa OCR evidence vào prompt nhưng không copy mù.

Gate chấp nhận: đúng video/window trước, sau đó mới tính answer accuracy; P95 dưới
5 phút và không còn lỗi CUDA fallback âm thầm.

## M6 — TRAKE dùng shared window retrieval

1. Planner sinh danh sách ordered moments và temporal constraints.
2. Retrieve window cho từng moment trong cùng video collection.
3. Xếp video theo coverage toàn bộ moment trước alignment.
4. Dùng K-best monotonic alignment theo `pts_time`.
5. Reranker nhìn toàn chuỗi, loại hypothesis chỉ khớp một phần.
6. Refine frame thật cho từng moment sau khi sequence đã được chọn.

Gate chấp nhận: tăng tỷ lệ đúng video và sequence score; mọi hypothesis có đúng
số moment và frame tăng theo thời gian khi query yêu cầu.

## M7 — Multi-resolution và modality bổ sung

Chỉ bắt đầu sau M1–M6:

- thử window ngắn/trung bình/dài bằng index signature riêng;
- SigLIP2 chỉ giữ nếu tăng frame recall bổ sung cho Qwen-window;
- ASR timestamped cho lời thoại/tên riêng;
- OCR full collection khi có thời gian offline;
- học fusion/calibration từ dev set thay vì đặt weight thủ công.

## Quy trình cho mỗi thay đổi model

1. Ghi rõ vai trò model và license/artifact cần thiết.
2. Tạo baseline và test set trước khi build full index.
3. Smoke subset, đo VRAM/latency và failure behavior.
4. A/B đúng metric cuộc thi.
5. Chỉ bật khi index/model hoàn chỉnh và có rollback.
6. Cập nhật `CURRENT_STATUS.md`, `SYSTEM_ARCHITECTURE.md` và lệnh setup.

## Thứ tự ưu tiên

```text
unified model query plan
  → video-window/offline evidence recall
  → KIS verified ranking
  → QA retrieval + answer verification
  → TRAKE shared retrieval + sequence verification
  → ASR/multi-resolution/fusion learning
```
