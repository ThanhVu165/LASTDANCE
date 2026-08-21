# Runbook đợt 1 — cấu hình đã khóa

## Không thay đổi trước giờ thi

- Không cài thêm package, không tải Qwen3-VL-Embedding/SigLIP2 và không rebuild
  `clip.faiss`.
- Không chạy OCR khi backend đang giữ Qwen3-VL trên GPU 6 GB.
- Không chạy Uvicorn với `--reload` trong lúc thi.
- Không xóa cache Hugging Face, `data/index` hoặc thư mục keyframe.

## Khởi động

Terminal backend:

```powershell
cd C:\LASTDANCE\backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal frontend:

```powershell
cd C:\LASTDANCE\frontend
streamlit run streamlit_app.py
```

Kiểm tra `http://127.0.0.1:8000/health`:

- `cuda_available=true`, `vqa_ready=true`.
- `vlm_rerank_enabled=true`, `vlm_rerank_group_size=3`.
- `kis_storyboard_enabled=true`, `kis_long_query_candidates=800`.
- `kis_exact_frame_enabled=true`.

## Smoke trước khi thi

Không cần chạy OCR. Sau khi backend chưa khởi động hoặc đã dừng, có thể chạy:

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m app.evaluation.round1_smoke `
  --query tkis-charity --top-k 100 --exact
```

Kết quả tham chiếu ngày 21/08/2026: 100 dòng, 32 video khác nhau, Top 1
`L22_V004`; exact-frame chọn `frame_id=810` từ video nguồn. Tổng thời gian 72,056
giây, nằm trong ngân sách KIS 2–3 phút.

Ngân sách mục tiêu khi thi: KIS không quá 180 giây; Q&A và TRAKE không quá 300
giây. Đây là giới hạn vận hành, không phải timeout cứng của pipeline.

## Cấu hình mặc định đã đo

| Biến | Mặc định | Ý nghĩa |
|---|---:|---|
| `AIC_KIS_LONG_QUERY_CANDIDATES` | 800 | Recall mỗi evidence unit dài |
| `AIC_VLM_RERANK_TOP_VIDEOS` | 30 | Video vào bracket KIS |
| `AIC_VLM_RERANK_GROUP_SIZE` | 3 | Số video/contact sheet comparison |
| `AIC_VLM_RERANK_FRAMES_PER_VIDEO` | 4 | Evidence frame/video |
| `AIC_KIS_EXACT_FRAME_ENABLED` | 1 | Refine Top 1 trên video gốc |

## Chế độ khẩn cấp

Nếu KIS vượt quá giới hạn thời gian nhưng CUDA vẫn ổn, khởi động lại backend với:

```powershell
$env:AIC_VLM_RERANK_TOP_VIDEOS = "18"
```

Nếu Qwen gặp lỗi trong rerank, code tự giữ ranking retrieval. Chỉ khi cần bảo đảm
KIS còn trả kết quả mới tắt VLM rồi khởi động lại backend:

```powershell
$env:AIC_VLM_RERANK_ENABLED = "0"
```

Không dùng cấu hình tắt VLM cho Q&A vì Q&A cần model để trả lời. Khắc phục Q&A bằng
cách kiểm tra đúng `.venv`, `cuda_available` và dừng mọi tiến trình OCR trước.

## Sau đợt 1

Mở `docs/retrieval_upgrade_plan.md`, mục P2 VCMR. Tạo environment và side index
riêng cho `Qwen3-VL-Embedding-2B`; không sửa trực tiếp môi trường thi cho đến khi
A/B hoàn tất.
