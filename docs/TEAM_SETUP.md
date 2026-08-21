# Hướng dẫn dựng LASTDANCE trên máy khác

Tài liệu này dành cho thành viên nhóm cần clone code, nhận dataset/model artifacts
và chạy hệ thống độc lập. Các lệnh ưu tiên Windows PowerShell.

## 1. Những gì Git có và không có

Git chứa source code, test, frontend và tài liệu. Git không chứa:

- `data/`: video, keyframe, feature, object, metadata và generated indexes;
- `.venv`;
- Hugging Face/EasyOCR/Paddle model cache;
- query thật, submission, log và secret.

Vì vậy clone repository chưa đủ để search. Thành viên cần nhận dataset qua ổ cứng,
NAS hoặc cloud nội bộ đúng quyền sử dụng của cuộc thi.

## 2. Yêu cầu máy

Khuyến nghị tối thiểu:

- Windows 10/11 64-bit;
- Python 3.11;
- 16 GiB RAM, khuyến nghị 32 GiB;
- khoảng 50 GiB trống ngoài dung lượng dataset;
- NVIDIA GPU 6 GiB VRAM trở lên cho Qwen 2B;
- driver NVIDIA tương thích wheel PyTorch CUDA đang dùng.

CPU-only có thể build CLIP/FAISS và chạy API cơ bản nhưng Qwen planner/QA sẽ rất
chậm. Nếu dùng CPU, phải đặt device rõ ràng; không để hệ thống âm thầm fallback.

## 3. Clone và xử lý Git ownership

```powershell
cd C:\
git clone https://github.com/ThanhVu165/LASTDANCE.git
cd C:\LASTDANCE
git remote -v
git status
```

Nếu repository được tạo/chỉnh bởi tài khoản sandbox hoặc administrator, Git có
thể báo `detected dubious ownership`. Với repository đã xác minh đúng đường dẫn:

```powershell
git config --global --add safe.directory C:/LASTDANCE
git status
```

Không dùng `safe.directory '*'`. Không đổi owner toàn ổ đĩa chỉ để sửa một repo.
Có thể kiểm tra các exception bằng:

```powershell
git config --global --get-all safe.directory
```

## 4. Tạo backend environment

```powershell
cd C:\LASTDANCE\backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Máy NVIDIA tham chiếu
.\.venv\Scripts\python.exe -m pip install `
  torch==2.12.1 torchvision==0.27.1 `
  --index-url https://download.pytorch.org/whl/cu130

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Không cài VietOCR vào environment production hiện tại: phiên bản đó pin Pillow
khác với frontend/backend. Nếu cần benchmark VietOCR, tạo environment riêng.

Kiểm tra CUDA:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. Frontend dependencies

Streamlit hiện có thể dùng chung backend `.venv` nếu đã cài đủ package. Cách tách
environment nhẹ hơn:

```powershell
cd C:\LASTDANCE\frontend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu dùng frontend venv riêng, thay đường dẫn Python trong lệnh khởi động ở mục 10.

## 6. Chép dataset

Sau khi chép, cấu trúc tối thiểu phải là:

```text
C:\LASTDANCE\data\features\<video_id>.npy
C:\LASTDANCE\data\keyframes\<video_id>\001.jpg
C:\LASTDANCE\data\map-keyframes\<video_id>.csv
```

Để dùng object/OCR/exact-frame còn cần:

```text
data\objects\<video_id>\001.json
data\videos\<video_id>.mp4
data\metadata\<video_id>.json
```

Máy tham chiếu hiện có 873 video và 177.321 keyframe. So sánh số lượng trước khi
build để tránh index lệch dataset.

## 7. Build production indexes

Chạy trên từng máy sau khi data nằm đúng vị trí, vì `keyframe_index.json` lưu
đường dẫn tuyệt đối:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m app.indexing.build_index
```

Kết quả bắt buộc:

- `data/index/keyframe_index.json`;
- `data/index/clip.faiss`;
- `data/index/objects_cache.json`.

OCR cache/state dùng key `video_id:local_idx` và có thể chuyển giữa các máy nếu
dataset và chữ ký model/config giống hệt. Tuy vậy phải kiểm tra state trước, không
copy một cache dở rồi đánh dấu complete.

## 8. Model artifacts

Các model public có thể tải bằng Hugging Face CLI:

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-2B-Instruct
.\.venv\Scripts\hf.exe download sentence-transformers/clip-ViT-B-32-multilingual-v1
```

Optional, không chặn production:

```powershell
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Reranker-2B
.\.venv\Scripts\hf.exe download google/siglip2-base-patch16-256
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Embedding-2B
```

Đặt `HF_TOKEN` nếu tài khoản nhóm có token để tránh rate limit. Cảnh báo symlink
trên Windows không làm model sai nhưng có thể tốn thêm dung lượng; bật Developer
Mode nếu muốn Hugging Face dùng symlink hiệu quả.

Runtime model-first giữ `local_files_only` cho model lớn. Mục tiêu là request
không bao giờ đứng vì tự tải checkpoint giữa cuộc thi.

## 9. Cấu hình qua environment variables

Ứng dụng đọc trực tiếp biến môi trường; hiện không tự load `.env`. Đặt biến trong
terminal trước khi khởi động. Profile RTX 4050 6 GiB mặc định:

```powershell
$env:AIC_VQA_DEVICE = "cuda:0"
$env:AIC_MODEL_QUERY_PLANNER_ENABLED = "1"
$env:AIC_MODEL_RERANK_ENABLED = "1"
$env:AIC_MODEL_RERANK_TOP_VIDEOS = "40"
$env:AIC_MODEL_REPAIR_ENABLED = "1"
$env:AIC_MODEL_RERANK_LOCAL_FILES_ONLY = "1"
```

Chế độ khẩn cấp nếu KIS vượt ngân sách:

```powershell
$env:AIC_MODEL_RERANK_TOP_VIDEOS = "24"
```

Không tắt VQA cho QA. Không bật SigLIP/video-window bằng biến môi trường; các kênh
này chỉ hoạt động khi index và state hoàn chỉnh được publish.

## 10. Kiểm thử trước khi chạy

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Smoke KIS không exact-frame:

```powershell
.\.venv\Scripts\python.exe -m app.evaluation.round1_smoke `
  --query tkis-charity --top-k 100
```

Không chạy smoke GPU khi backend hoặc OCR đang giữ VRAM.

## 11. Khởi động

Backend:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app `
  --host 127.0.0.1 --port 8000
```

Frontend dùng chung backend venv:

```powershell
cd C:\LASTDANCE\frontend
C:\LASTDANCE\backend\.venv\Scripts\python.exe -m streamlit run `
  streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Kiểm tra:

- `http://127.0.0.1:8000/health`;
- `http://127.0.0.1:8501`.

`/health` phải có `cuda_available=true`, `vqa_ready=true`, planner/rerank/repair
đúng profile. `siglip_index_ready=false` hoặc `video_window_index_ready=false` là
bình thường nếu chưa build side index.

## 12. Chạy trên hai máy hoặc LAN

Frontend hiện gọi backend tại `127.0.0.1:8000`, nên cấu hình chuẩn là frontend và
backend cùng một máy. Muốn tách máy cần đưa API base URL thành environment setting
trước; không mở `0.0.0.0`/firewall trong lúc thi nếu chưa kiểm tra bảo mật mạng.

Dataset/model license và thể lệ cuộc thi phải được tuân thủ khi chia sẻ artifact.

## 13. Dừng dịch vụ và chuyển sang indexing

Đóng terminal dịch vụ hoặc dừng đúng PID trước khi chạy OCR/builder. Xác nhận VRAM:

```powershell
nvidia-smi
```

Không chạy đồng thời backend Qwen với OCR, SigLIP hoặc Qwen embedding builder trên
GPU 6 GiB.

## 14. Commit và push

Từ repository root:

```powershell
cd C:\LASTDANCE
git status
git diff --check
git add README.md AGENTS.md docs backend frontend
git status
git commit -m "Document model-first architecture and team setup"
git push origin main
```

Xem lại danh sách staged trước commit để chắc chắn `data/`, model hoặc secret
không xuất hiện. Nếu push yêu cầu đăng nhập, dùng Git Credential Manager hoặc SSH
key của tài khoản có quyền với `ThanhVu165/LASTDANCE`; không ghi token vào remote
URL hoặc file trong repository.

## 15. Troubleshooting nhanh

| Triệu chứng | Cách xử lý |
|---|---|
| `dubious ownership` | Thêm đúng repo vào `safe.directory`, không dùng wildcard |
| `VQA requested cuda` | Kiểm tra đúng `.venv`, wheel CUDA và driver |
| CUDA OOM | Dừng OCR/builder/backend khác; giảm top video nếu khẩn cấp |
| KIS treo tải model | Giữ local-only; tải model bằng CLI trước khi chạy API |
| Kết quả trả `local_idx` sai | Rebuild index và kiểm tra map-keyframes |
| Side index không hoạt động | Kiểm tra state `complete=true` và file FAISS tồn tại |
| Máy mới không mở được keyframe | `keyframe_index.json` có path máy cũ; rebuild |

