# Báo cáo runtime model-first — 21/08/2026

## Đã bật và có thể dùng

- Qwen3-VL-2B-Instruct structured planner thay parser regex trên đường KIS chính.
- Model verification theo nhóm 4, tối đa 40 video; Top 100 ưu tiên tuyệt đối các
  row đã chấm và trả `model_relevance_score`, `model_verified` qua API.
- Một vòng repair retrieval từ các caption distinctive do planner sinh.
- QA gọi trực tiếp KIS model-first nên dùng cùng verified video retrieval trước
  khi temporal VQA trả lời.
- Optional SigLIP2 và video-window retrieval đã nối vào KIS; chỉ tự kích hoạt khi
  state/index được publish hoàn chỉnh.
- Backend chạy bằng `.venv`, PID tại thời điểm báo cáo: `4520`, cổng `8000`.

## Kết quả kiểm thử thật

| Kiểm thử | Kết quả |
|---|---:|
| Unit tests | 56/56 pass |
| Generative verifier 4 video | 4/4 parsed, 24,656 giây gồm load model |
| Peak VRAM verifier | 4,082 GiB reserved |
| KIS module Top 20 | 20 row, 20 được chấm, KIS phase 29,586 giây |
| KIS HTTP Top 100 | 100 row, 100 verified, 100 model-scored, 94,9 giây |
| QA HTTP Top 100 | 100 row, 82,6 giây; câu test màu trả `đỏ` ở Top 1 |

Các smoke trên chứng minh khả năng vận hành và contract, không chứng minh accuracy
toàn dataset do chưa có ground truth tương ứng.

## Chưa hoàn tất

1. `Qwen/Qwen3-VL-Reranker-2B`: trọng số 3,96 GiB chưa tải xong; hiện dùng Qwen
   instruct generative verifier. Tournament chỉ chạy nếu cả hai verifier lỗi.
2. SigLIP2: builder checkpointable và fusion đã có, nhưng full index 177.321
   keyframe chưa build/publish nên `/health.siglip_index_ready=false`.
3. Qwen video-window: `Qwen3-VL-Embedding-2B` đã tải đủ (~3,96 GiB). Builder dùng
   6 frame/window, stride 6, timestamp thật, 1024 chiều và ngân sách tổng 221.184
   pixel/window. Benchmark batch 8 đã ghi checkpoint 300/29.938; full build được
   chủ động dừng vì cần vài giờ và chưa có ground truth chứng minh lợi ích. State
   chưa complete nên `/health.video_window_index_ready=false` và KIS tự bỏ qua.
4. TRAKE chưa chuyển sang cùng tầng verified retrieval.

## Vận hành ngay

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 127.0.0.1 --port 8000
```

Kiểm tra `http://127.0.0.1:8000/health`: planner/rerank/repair phải là `true`, CUDA
và VQA phải ready. Không chạy OCR, SigLIP builder hay video-window builder đồng
thời với backend vì Qwen đang giữ khoảng 4,3 GiB VRAM.

## Bước tiếp theo

1. Khi mạng ổn định, tải đủ `Qwen/Qwen3-VL-Reranker-2B`, chạy
   `app.evaluation.model_rerank_smoke` rồi xác nhận `backend=dedicated`.
2. Dừng backend, build SigLIP2 và A/B Recall@1/5/20/50/100 trước khi coi nó là
   production.
3. Chỉ resume Qwen video-window khi A/B subset chứng minh cải thiện recall; model
   đã có local và builder sẽ tiếp tục từ checkpoint 300 nếu giữ nguyên cấu hình.
4. Sau khi KIS có ground-truth regression, truyền verified window/hypothesis sang
   TRAKE và làm sequence verification.
