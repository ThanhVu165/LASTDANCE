# Trạng thái hiện tại của LASTDANCE

Cập nhật: 21/08/2026. Đây là nguồn trạng thái runtime chuẩn; không suy trạng thái
từ tên model trong config hoặc từ artifact đang tải dở.

## Thành phần đang hoạt động

| Thành phần | Trạng thái |
|---|---|
| Organizer CLIP FAISS + multilingual text tower | Production |
| Qwen structured planner cho KIS | Production |
| Qwen generative video verifier | Production fallback chính |
| Repair retrieval một vòng | Production |
| KIS Top 100 + exact-frame refinement | Production |
| QA dùng KIS retrieval + temporal VQA | Production, còn cần answer verification |
| TRAKE monotonic alignment + visual rerank | Đang hoạt động; chưa dùng shared window layer |
| EasyOCR CRAFT + `latin_g2` | Builder hoạt động; collection chưa xác nhận complete |

## Thành phần đang triển khai

### Unified query processing

- Model đã chọn: `Qwen/Qwen3-VL-2B-Instruct`.
- Schema đích và migration đã được định nghĩa trong `QUERY_PROCESSING.md`.
- KIS đã có model planner; QA và TRAKE còn parser/task flow riêng.
- Chưa được ghi là hoàn thành cho tới khi ba pipeline gọi cùng `plan_query` và
  cùng `VerifiedEvidenceBundle`.

### Qwen video-window embedding

- Model: `Qwen/Qwen3-VL-Embedding-2B`.
- Builder và runtime loader đã có.
- Cấu hình: 6 keyframe/window, stride 6, 1024 chiều, 221.184 pixel/window.
- Collection: 177.321 keyframe → 29.938 window.
- Checkpoint hiện tại: 300/29.938, `complete=false`.
- Runtime bắt buộc bỏ qua partial index; `/health.video_window_index_ready=false`.

Đây là hướng recall chính cho query nhiều ngữ nghĩa, nhưng chưa phải production
cho tới khi build đủ và A/B trên ground truth.

### Dedicated reranker

- Model đích: `Qwen/Qwen3-VL-Reranker-2B`.
- Adapter đã có, runtime local-only.
- Checkpoint local chưa được xác nhận đầy đủ; health hiện báo backend
  `generative`.
- Không được coi model là active chỉ vì thư mục cache tồn tại.

### SigLIP2

- Model: `google/siglip2-base-patch16-256`.
- Builder/loader đã có nhưng chưa có state/index complete.
- Chỉ là side recall; không thay thế hướng video-window.

## Kiểm thử gần nhất

- `compileall`: pass.
- Unit tests sau cleanup: 55/55 pass.
- KIS HTTP Top 100 đã đo khoảng 94,9 giây, 100 row có model score trong smoke.
- QA HTTP Top 100 đã đo khoảng 82,6 giây trên query smoke.
- `/health`: CUDA và VQA ready trên RTX 4050 Laptop GPU 6 GiB.

Các số trên chứng minh runtime/contract, không chứng minh accuracy toàn dataset.
Báo cáo lỗi cụ thể nằm trong [`e2e_test_report_2026-08-21.md`](e2e_test_report_2026-08-21.md).

## Việc cần làm tiếp theo

1. Triển khai `UnifiedQueryPlan` dùng chung KIS/QA/TRAKE.
2. Chuẩn hóa manifest/validation và shot boundary cho offline indexing.
3. Tạo dev set có ground truth video + window.
4. Benchmark Qwen-window trên subset rồi mới resume full index.
5. Thử structured window caption như một kênh lexical/dense có provenance.
6. Benchmark OCR challenger trong environment riêng; không thay EasyOCR trước A/B.
7. Hoàn tất/benchmark dedicated reranker.
8. Chuyển KIS, QA rồi TRAKE sang shared verified retrieval.

Chi tiết và acceptance gate nằm trong
[`DEVELOPMENT_ROADMAP.md`](DEVELOPMENT_ROADMAP.md).
