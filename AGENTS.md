# AGENTS.md — LASTDANCE repository contract

File này áp dụng cho toàn repository và là contract mặc định cho người và coding
agent.

## Đọc trước khi sửa

1. `docs/PROJECT_CONTEXT.md`
2. `docs/SYSTEM_ARCHITECTURE.md`
3. `docs/QUERY_PROCESSING.md` nếu đụng query/KIS/QA/TRAKE
4. `docs/OFFLINE_INDEXING.md` nếu đụng dữ liệu/index/model offline
5. `docs/MODEL_SELECTION.md` nếu đề xuất/thay model
6. `docs/CURRENT_STATUS.md`
7. `docs/DEVELOPMENT_ROADMAP.md`
8. `docs/TEAM_SETUP.md` nếu thay dependency/lệnh/model artifact

Báo cáo E2E có ngày là evidence snapshot. PDF là nguồn thể lệ. Không dùng lịch sử
chat hoặc file đã bị loại bỏ làm runtime instruction.

## Mục tiêu sản phẩm

LASTDANCE là hệ thống video retrieval cho KIS, QA và TRAKE. Tối ưu các cutoff
`R@1, R@5, R@20, R@50, R@100` theo thứ tự:

1. tìm đúng video và time window;
2. phủ đủ scene/điều kiện của query;
3. rerank bằng evidence nhìn thấy được;
4. chỉ sau đó mới refine frame, trả lời QA hoặc align TRAKE.

## Kiến trúc bắt buộc

- Không biểu diễn toàn video bằng một vector hoặc một keyframe.
- Video là tập window/evidence có timestamp và provenance.
- `Qwen3-VL-Embedding-2B` là hướng video-window recall chính; organizer CLIP là
  baseline/fallback cho tới khi window index complete và thắng A/B.
- Một Qwen `UnifiedQueryPlan` là đường hiểu semantic chung cho KIS/QA/TRAKE.
- Dedicated Qwen reranker là đích query–video/window scoring; generative verifier
  là fallback model hiện tại.
- OCR, object, caption, shot và ASR là evidence bổ sung, không phải query parser
  hoặc answer generator độc lập.
- Regex/heuristic chỉ dùng cho schema, ID mapping và resource bound. Fallback
  semantic là nguyên query, không tự suy scene/question/moment bằng rule.
- Không thêm patch riêng cho query mẫu, ngôn ngữ, màu hoặc object.

## Data và index invariants

- `local_idx` là keyframe nội bộ; `frame_id` là frame thật dùng cho submission.
- `pts_time` là trục join giữa window, OCR, caption, object và ASR.
- Mọi artifact có model/config/dataset signature.
- State `complete=false` phải fail closed; file feature/index tồn tại chưa đủ.
- Không cộng raw cosine từ các embedding space khác nhau; dùng rank calibration.
- Không xóa `data/`, OCR state, production index hoặc model cache như cleanup.
- Không commit dataset, cache, `.venv`, query, submission, log hoặc credential.

## Model và GPU

Máy tham chiếu là RTX 4050 Laptop 6 GiB. Không chạy đồng thời backend Qwen, OCR,
SigLIP builder hoặc Qwen embedding/reranker build. Các model 2B phải load/release
theo phase. Runtime không tự tải model nhiều GB và không âm thầm chuyển CUDA sang
CPU.

## Workflow thay đổi

1. Đọc pipeline, config, tests và status liên quan.
2. Nếu thay retrieval/index, tạo baseline/dev subset trước.
3. Giữ behavior mới sau feature/index completeness gate.
4. Builder phải checkpoint/resume và publish atomic.
5. Chạy từ `backend/`:

   ```powershell
   .\.venv\Scripts\python.exe -m compileall -q app
   .\.venv\Scripts\python.exe -m unittest discover -s tests -q
   ```

6. Với ranking/model/index, báo Recall@k khi có ground truth, result/verified
   count, distinct video, latency P50/P95, peak VRAM và rollback.
7. Cập nhật `CURRENT_STATUS.md`, kiến trúc, indexing, roadmap và setup nếu contract,
   model, artifact hoặc command thay đổi.

## Không được làm

- Không publish partial FAISS index hoặc sửa state complete thủ công.
- Không mix score không hiệu chuẩn.
- Không cho answer model chạy trước khi retrieval chọn được evidence credible.
- Không tuyên bố Top 100 đều đúng hoặc model tốt hơn chỉ từ smoke test.
- Không bỏ CLIP/tournament fallback trước khi replacement qua regression test.
- Không dùng `git reset --hard` hoặc ghi đè dirty worktree.

## Definition of done

Một thay đổi retrieval hoàn tất khi code compile, test pass, API giữ schema/count,
index/failure fallback được kiểm tra, mapping frame đúng, đo latency/VRAM, có A/B
trên ground truth nếu thay ranking, và tài liệu phản ánh đúng active/partial/planned.
