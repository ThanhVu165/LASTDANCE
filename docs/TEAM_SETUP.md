# Dựng LASTDANCE trên máy khác

> **ARCHIVED 24/08/2026:** Lệnh setup/build dưới đây dành cho backend window-first cũ.
> Không dùng để build artifact frame-level mới nếu chưa đối chiếu `BASELINE_SPEC.md`.

Tài liệu này dành cho thành viên clone repository, nhận dataset/model artifact,
build index và chạy hệ thống trên Windows PowerShell.

## 1. Git chứa gì

Git chứa source, test, frontend và tài liệu. Git không chứa:

- `data/` và generated indexes;
- `.venv`;
- Hugging Face/EasyOCR/model cache;
- query thật, submission, log và credential.

Clone code chưa đủ để search. Dataset và model artifact phải được chia sẻ qua kênh
nội bộ đúng quyền sử dụng của cuộc thi.

## 2. Yêu cầu máy

- Windows 10/11 64-bit, Python 3.11;
- tối thiểu 16 GiB RAM, khuyến nghị 32 GiB;
- NVIDIA GPU 6 GiB VRAM trở lên cho các model Qwen 2B;
- driver tương thích PyTorch CUDA;
- đủ đĩa cho dataset, model cache và feature/index trung gian.

Máy 6 GiB chỉ chạy một workload GPU tại một thời điểm.

## 3. Clone và Git ownership

```powershell
cd C:\
git clone https://github.com/ThanhVu165/LASTDANCE.git
cd C:\LASTDANCE
git remote -v
git status
```

Nếu Git báo `detected dubious ownership`, sau khi xác minh đúng repo:

```powershell
git config --global --add safe.directory C:/LASTDANCE
git status
```

Không dùng `safe.directory '*'`.

## 4. Backend environment

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

Không cài VietOCR vào environment này vì phiên bản hiện dùng pin Pillow 10.2.0,
trong khi project dùng Pillow 11.0.0. Nếu cần benchmark, tạo environment riêng.

Kiểm tra CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Frontend

Có thể dùng chung backend venv hoặc tạo venv riêng:

```powershell
cd C:\LASTDANCE\frontend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 6. Nhận và kiểm tra dataset

Cấu trúc tối thiểu:

```text
data/features/<video_id>.npy
data/keyframes/<video_id>/<NNN>.jpg
data/map-keyframes/<video_id>.csv
```

Để có đầy đủ evidence cần thêm:

```text
data/videos/<video_id>.mp4
data/objects/<video_id>/<NNN>.json
data/metadata/<video_id>.json
```

Dataset tham chiếu hiện có 873 video và 177.321 keyframe. So sánh inventory trước
khi build. Thiết kế manifest/shot validator là milestone tiếp theo; hiện thành viên
phải kiểm tra count và sample thủ công.

## 7. Tải model

Model runtime bắt buộc:

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-2B-Instruct
.\.venv\Scripts\hf.exe download sentence-transformers/clip-ViT-B-32-multilingual-v1
```

Model cho hướng video-window/rerank:

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Embedding-2B
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Reranker-2B
```

Optional frame recall:

```powershell
.\.venv\Scripts\hf.exe download google/siglip2-base-patch16-256
```

Model lớn phải tải trước, không để API request tự tải. Có thể đặt `HF_TOKEN` qua
environment; không ghi token vào Git hoặc remote URL.

Không cài Paddle OCR challenger vào production venv. Tạo environment riêng để
tránh lặp lỗi CUDA/Pillow; chỉ chuyển baseline sau benchmark theo
[`MODEL_SELECTION.md`](MODEL_SELECTION.md).

## 8. Build base index

Mỗi máy phải rebuild vì `keyframe_index.json` lưu path tuyệt đối:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m app.indexing.build_index
```

Artifact bắt buộc:

- `keyframe_index.json`;
- `clip.faiss`;
- `objects_cache.json`.

## 9. Build video-window index

Builder hiện có và checkpoint được. Chạy subset đầu tiên:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.video_window_index `
  --limit 100 --batch-size 1 --checkpoint-every 25 `
  --window-size 6 --stride 6
```

Kiểm tra `data/index/video_window_state.json`. Chỉ resume full sau khi subset đã
được đánh giá:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.video_window_index `
  --batch-size 1 --checkpoint-every 50 `
  --window-size 6 --stride 6
```

Builder resume từ `next_index`. Nếu đổi model/dim/window/stride/pixel/dataset,
signature sẽ khác; di chuyển artifact cũ sang nơi lưu trữ trước khi build mới.
Không sửa `complete=true` thủ công.

Shot boundary, manifest validator và structured caption index được mô tả trong
[`OFFLINE_INDEXING.md`](OFFLINE_INDEXING.md) nhưng **chưa có command production**.

## 10. OCR và optional SigLIP2

OCR smoke:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.ocr_index `
  --limit 20 --checkpoint-every 5
```

Resume OCR bằng cùng lệnh không `--limit`. Có thể dừng bằng `Ctrl+C`; checkpoint
đã atomic-write vẫn còn. Không xóa cache/state chỉ vì muốn chạy lại.

SigLIP2 subset:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.siglip_index `
  --limit 100 --batch-size 8 --checkpoint-every 50
```

SigLIP2 là optional side index. Chỉ build full nếu A/B cho thấy recall bổ sung so
với CLIP + Qwen-window.

## 11. Cấu hình runtime

```powershell
$env:AIC_VQA_DEVICE = "cuda:0"
$env:AIC_MODEL_QUERY_PLANNER_ENABLED = "1"
$env:AIC_MODEL_RERANK_ENABLED = "1"
$env:AIC_MODEL_RERANK_TOP_VIDEOS = "40"
$env:AIC_MODEL_REPAIR_ENABLED = "1"
$env:AIC_MODEL_RERANK_LOCAL_FILES_ONLY = "1"
```

Không cần flag để “ép bật” side index. Loader chỉ dùng nó khi state complete và
artifact tồn tại. Khi model 2B chạy query window, runtime giải phóng instruct VLM
trước để tránh OOM.

## 12. Test

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Smoke GPU chỉ chạy sau khi dừng backend và các builder khác.

## 13. Khởi động

Backend:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 127.0.0.1 --port 8000
```

Frontend dùng backend venv:

```powershell
cd C:\LASTDANCE\frontend
C:\LASTDANCE\backend\.venv\Scripts\python.exe -m streamlit run `
  streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

`/health` phải báo CUDA/VQA ready. `video_window_index_ready=false` là đúng nếu
state partial. Sau khi publish full index, restart backend để loader nhìn artifact.

## 14. Vận hành GPU

Trước khi chạy offline builder:

1. dừng backend;
2. kiểm tra `nvidia-smi`;
3. chạy đúng một builder;
4. ghi ETA/progress/state;
5. dừng builder trước khi mở API.

Không chạy OCR, SigLIP, Qwen embedding/reranker và backend đồng thời trên GPU 6 GiB.

## 15. Commit và push

```powershell
cd C:\LASTDANCE
git status
git diff --check
git add README.md AGENTS.md docs backend frontend
git status
git commit -m "Adopt video-window model-first architecture"
git push origin main
```

Kiểm tra staged files để chắc chắn `data/`, cache, query và secret không xuất hiện.

## 16. Troubleshooting

| Triệu chứng | Xử lý |
|---|---|
| Git dubious ownership | thêm đúng `C:/LASTDANCE` vào safe.directory |
| VQA requested CUDA | kiểm tra đúng `.venv`, wheel CUDA và driver |
| CUDA OOM | dừng workload GPU khác; không âm thầm chuyển CPU |
| API đứng tải model | tải model trước; giữ local-only cho model lớn |
| Sai `frame_id` | rebuild base index và kiểm tra map-keyframes |
| Window index không active | kiểm tra metadata/FAISS/state `complete=true` |
| Signature mismatch | lưu artifact cũ và build mới, không resume chéo config |
| Path keyframe máy cũ | rebuild `app.indexing.build_index` |
