# LASTDANCE — AIC2026 video retrieval

LASTDANCE nhận một câu truy vấn tự nhiên và trả Top 100 kết quả cho KIS, QA hoặc
TRAKE. Hướng phát triển hiện tại là **video-window retrieval + model-based
planning/reranking/verification**. OCR, object, caption và sau này ASR là các
kênh evidence bổ sung; không kênh nào thay thế việc hiểu toàn bộ video scene.

## Tài liệu chuẩn

Đọc theo thứ tự:

1. [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) — bài toán và quyết định sản phẩm.
2. [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) — pipeline và vai trò model.
3. [`docs/QUERY_PROCESSING.md`](docs/QUERY_PROCESSING.md) — một model-based query plan cho KIS/QA/TRAKE.
4. [`docs/OFFLINE_INDEXING.md`](docs/OFFLINE_INDEXING.md) — cách biến video thành evidence/index.
5. [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) — quyết định và challenger cho từng model.
6. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — phần đang hoạt động và đang dở.
7. [`docs/DEVELOPMENT_ROADMAP.md`](docs/DEVELOPMENT_ROADMAP.md) — thứ tự phát triển và acceptance gate.
8. [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md) — dựng trên máy khác.
9. [`docs/AI_COLLABORATION_GUIDE.md`](docs/AI_COLLABORATION_GUIDE.md) — quy tắc làm việc với AI.

[`docs/Thong tin vong So tuyen AIC2026.pdf`](docs/Thong%20tin%20vong%20So%20tuyen%20AIC2026.pdf)
là nguồn thể lệ. Báo cáo E2E có ngày là bằng chứng kiểm thử tại một checkpoint,
không phải hướng dẫn runtime.

## Kiến trúc ngắn gọn

```text
query
  → unified Qwen structured planner cho KIS/QA/TRAKE
  → frame recall + video-window recall + indexed evidence
  → video/scene coverage và temporal aggregation
  → Qwen multimodal reranker/verifier
  → bounded repair retrieval
  → Top 100 + exact source-frame refinement
```

Không dùng một vector để đại diện toàn video. Offline pipeline giữ một tập vector
window cùng timestamp, frame mapping, shot, OCR, object và structured caption.
Chi tiết: [`docs/OFFLINE_INDEXING.md`](docs/OFFLINE_INDEXING.md).

## Model theo vai trò

| Vai trò | Model | Trạng thái |
|---|---|---|
| Baseline frame recall | organizer `clip-ViT-B-32` | Production |
| Vietnamese CLIP query | `clip-ViT-B-32-multilingual-v1` | Production |
| Unified planner, verifier, VQA, offline caption | `Qwen/Qwen3-VL-2B-Instruct` | KIS planner/verifier/VQA active; unified QA/TRAKE plan và caption là roadmap |
| Video-window embedding | `Qwen/Qwen3-VL-Embedding-2B` | Builder có, index partial |
| Dedicated query–video rerank | `Qwen/Qwen3-VL-Reranker-2B` | Adapter có, checkpoint chưa active |
| Optional frame recall | `google/siglip2-base-patch16-256` | Chưa build full index |
| OCR | EasyOCR CRAFT + `latin_g2` | Builder hoạt động, chưa xác nhận full collection |
| OCR challenger | PP-OCRv5 Latin; PP-OCRv6 sau dictionary/runtime validation | Benchmark riêng, chưa thay baseline |
| ASR | Chưa khóa model | Sau KIS/QA |

Tên model trong config không có nghĩa model đang active. Kiểm tra `/health`, file
state và [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md). Lý do chọn model nằm
trong [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

## Dữ liệu

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

`local_idx` là số keyframe nội bộ. `frame_id` là frame thật trong MP4 và là giá
trị dùng cho preview chính xác/nộp bài. Không được hoán đổi hai trường này.

## Cài đặt nhanh trên Windows

```powershell
cd C:\LASTDANCE\backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install `
  torch==2.12.1 torchvision==0.27.1 `
  --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Không cài VietOCR vào environment production vì pin Pillow xung đột. Hướng dẫn
đầy đủ và cách xử lý Git ownership nằm trong [`docs/TEAM_SETUP.md`](docs/TEAM_SETUP.md).

## Build index

Base index bắt buộc:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m app.indexing.build_index
```

Video-window index — hiện là builder đang phát triển, chạy subset trước:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.video_window_index `
  --limit 100 --batch-size 1 --checkpoint-every 25
```

Chỉ bỏ `--limit` sau khi subset có ground truth chứng minh lợi ích. Index chỉ được
dùng khi `video_window_state.json` có `complete=true`. Không chạy builder GPU cùng
backend/OCR/side builder trên GPU 6 GiB.

OCR offline:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.ocr_index `
  --limit 20 --checkpoint-every 5
```

OCR có checkpoint/resume; `Ctrl+C` không làm mất checkpoint đã publish. Không xóa
cache/state để chạy lại trừ khi chủ động đổi signature và đã sao lưu artifact.

## Kiểm thử

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Smoke GPU chỉ chạy khi backend và các builder khác đã dừng.

## Khởi động

Backend:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd C:\LASTDANCE\frontend
C:\LASTDANCE\backend\.venv\Scripts\python.exe -m streamlit run `
  streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Kiểm tra `http://127.0.0.1:8000/health`. Với index partial, trường
`video_window_index_ready=false` là đúng và hệ thống giữ CLIP fallback.

## API

| Endpoint | Chức năng |
|---|---|
| `GET /health` | Model, CUDA, index và fallback status |
| `POST /kis/search` | KIS Top 100 |
| `POST /qa/search` | QA Top 100 |
| `POST /trake/search` | TRAKE Top 100 sequence hypotheses |
| `GET /video/{video_id}/keyframe/{local_idx}` | Keyframe nội bộ |
| `GET /video/{video_id}/frame/{frame_id}` | Frame thật từ MP4 |
| `/submission/*` | Validate/export CSV/ZIP |

## Nguyên tắc phát triển

- Retrieval đúng video/window trước, rerank và answer sau.
- Model hiểu semantic; rule chỉ giữ contract và fallback.
- Không cộng cosine thô từ các embedding space khác nhau.
- Không publish index dở.
- Mọi thay đổi model/index phải có dev set, Recall@k, latency, VRAM và rollback.
- Không commit `data/`, model cache, `.venv`, query, submission hoặc credential.
