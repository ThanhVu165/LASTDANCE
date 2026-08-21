# AIC2026 Retrieval System — Vòng sơ tuyển

Hệ thống truy xuất video hỗ trợ cả 3 dạng truy vấn của vòng sơ tuyển AIC2026:
**Textual KIS**, **Q&A**, **TRAKE**. Backend FastAPI + module re-rank riêng, frontend
Streamlit để tra cứu và xuất bài nộp đúng chuẩn hệ thống thi
(https://sotuyenaic.oj.io.vn/).

## 1) Cấu trúc dữ liệu thực tế (đã xác nhận từ dataset đã tải)

```
data/
  videos/<video_id>.mp4
  keyframes/<video_id>/<NNN>.jpg        # đánh số cục bộ 001, 002, ... (local_idx)
  objects/<video_id>/<NNN>.json         # Faster R-CNN / OpenImages, cùng đánh số local_idx
  features/<video_id>.npy               # shape (K, 512) float16 — mỗi video 1 file riêng
  map-keyframes/<video_id>.csv          # cột: n, pts_time, fps, frame_idx
                                         #   n = local_idx; frame_idx = SỐ FRAME THẬT
                                         #   → đây là giá trị BẮT BUỘC dùng khi nộp bài
  metadata/<video_id>.json               # metadata YouTube (có thể thiếu ở 1 số video)
  index/                                  # sinh ra bởi bước build index bên dưới
```

⚠️ **Lưu ý quan trọng**: mỗi keyframe có 2 chỉ số khác nhau:
- `local_idx`: số thứ tự file jpg/json (dùng nội bộ để lấy ảnh/object/OCR).
- `frame_id`: số frame thật trong video, lấy từ cột `frame_idx` — **đây là giá trị
  phải xuất ra file CSV nộp bài**, không phải `local_idx`.

## 2) Cài đặt

### Backend
```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# PyTorch CUDA 13.0 cho RTX 4050/driver hiện tại
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 `
  --index-url https://download.pytorch.org/whl/cu130

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

OCR dùng **EasyOCR CRAFT + `latin_g2`** trên `cuda:0`. `latin_g2` có đầy đủ ký tự
tiếng Việt dựng sẵn; script kiểm tra toàn bộ bảng chữ cái khi khởi tạo và từ chối
ghi cache nếu model không đáp ứng. Trọng số CRAFT/`latin_g2` được tải vào
`data/models/easyocr/` ở lần chạy đầu. Pipeline chỉ dùng PyTorch nên tránh xung
đột DLL CUDA/cuDNN giữa Paddle và PyTorch trên Windows.

Hai model runtime được tự tải từ Hugging Face ở truy vấn đầu tiên:

- `sentence-transformers/clip-ViT-B-32-multilingual-v1` (~539 MB) cho truy vấn
  tiếng Việt. Text embedding 512 chiều của model này được distill vào đúng không
  gian ảnh của CLIP ViT-B/32, nên dùng trực tiếp index ảnh sẵn có, không phải
  trích xuất lại 177.321 feature.
- `Qwen/Qwen3-VL-2B-Instruct` (~4–5 GB cache) cho Q&A thị giác tổng quát.

Không chạy OCR và Q&A cùng lúc trên GPU 6 GB. Hãy hoàn tất/dừng tiến trình
`ocr_index` trước khi dùng trang Q&A. Backend giữ Qwen3-VL trong VRAM sau lần hỏi
đầu để các câu tiếp theo nhanh hơn; muốn chạy OCR trở lại thì dừng backend trước.

### Vì sao chọn các model này

- PP-OCRv6 medium đã chạy được GPU nhưng model recognition phát hành hiện tại
  không thể phát ra đầy đủ 88 ký tự tiếng Việt dựng sẵn (xem
  [PaddleOCR issue #18254](https://github.com/PaddlePaddle/PaddleOCR/issues/18254)).
  Đây là lỗi vocabulary/model, không thể giải quyết triệt để bằng thay ký tự sau OCR.
- `latin_g2` khai báo đầy đủ nhóm ký tự tiếng Việt trong
  [cấu hình chính thức của EasyOCR](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py).
  Trên bộ chẩn đoán 3 keyframe thật, điểm giữ nguyên dấu đạt **0,9441**, so với
  **0,8724** của PP-OCRv6 medium. CRAFT + `latin_g2` cũng chạy ổn định trên RTX
  4050 6 GB với batch 8.
- HunyuanOCR 1.5 (1B) là ứng viên fallback cho một tập nhỏ ảnh khó; trọng số BF16
  khoảng 2,24 GB có thể nạp trên GPU này, nhưng
  [benchmark chính thức](https://github.com/Tencent-Hunyuan/HunyuanOCR/blob/main/docs/benchmark.md)
  công bố độ trễ khoảng 3,03 giây/ảnh ở chế độ cơ bản ngay trên H20. Vì vậy không
  dùng model VLM này để index toàn bộ 177.321 keyframe.
- Truy vấn tiếng Việt không còn được dịch bằng từ điển nhỏ rồi ghép thành câu lai
  Việt/Anh. Model
  [`clip-ViT-B-32-multilingual-v1`](https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1)
  hỗ trợ hơn 50 ngôn ngữ, có tiếng Việt, và tương thích trực tiếp với feature ảnh
  CLIP ViT-B/32 hiện có. Trên index thật, cả `a red car` và `một chiếc ô tô màu đỏ`
  đều trả đủ 100 kết quả trong khoảng 4 giây sau khi model đã có trong cache; hai
  hạng đầu của truy vấn tiếng Việt là khung hình xe đỏ thật.
- [`Qwen3-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
  là VLM 2B thế hệ mới có nâng cấp nhận dạng ảnh, hiểu văn bản và liên kết
  ảnh–ngôn ngữ. Đo trực tiếp FP16 trên RTX 4050 6 GB: ba câu hỏi màu trên keyframe
  thật đều đúng bằng tiếng Việt (`đỏ`, `xám`, `đỏ`), VRAM reserved cực đại khoảng
  4,14 GiB; sau khi model đã nạp, mỗi ảnh thử mất khoảng 0,36–0,49 giây. Đây là
  model cho bước đọc ảnh/trả lời của Q&A, không thay thế retrieval CLIP.

### Frontend
```powershell
cd frontend
python -m pip install -r requirements.txt
```

## 3) Build index (chạy 1 lần sau khi có dữ liệu trong `data/`)

```powershell
cd backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m app.indexing.build_index

# Bắt buộc smoke-test trước; kiểm tra GPU, model và output trên 20 ảnh
.\.venv\Scripts\python.exe -m app.indexing.ocr_index --limit 20 --checkpoint-every 5

# Smoke-test phân bố rộng (100 ảnh rải đều trong toàn bộ collection)
.\.venv\Scripts\python.exe -m app.indexing.ocr_index `
  --sample-stride 1700 --limit 100 --checkpoint-every 20

# Chạy lại bộ benchmark giữ nguyên dấu trên các keyframe đã gán nhãn
.\.venv\Scripts\python.exe -m app.indexing.ocr_model_benchmark

# Sau khi kết quả smoke-test hợp lý, chạy toàn bộ phần còn thiếu
.\.venv\Scripts\python.exe -m app.indexing.ocr_index
```

`build_index.py` sinh:
- `data/index/keyframe_index.json` — ánh xạ global_idx ↔ (video_id, local_idx, frame_id, path)
- `data/index/clip.faiss` — FAISS index (cosine similarity) trên toàn bộ CLIP feature
- `data/index/objects_cache.json` — nhãn object theo từng keyframe

`ocr_index.py` sinh:
- `data/index/ocr_cache.json` — schema v2: text theo từng dòng, confidence và tọa
  độ polygon; search vẫn đọc được cache chuỗi cũ trong giai đoạn chuyển đổi.
- `data/index/ocr_state.json` — phân biệt `success`, `no_text`, `error`, lưu số lần
  thử và chữ ký model. Cache rỗng cũ không có state sẽ tự được OCR lại.

Mặc định script đưa 8 ảnh/lần vào detector và recognition batch size là 16, phù
hợp RTX 4050 6 GB. Nếu gặp OOM, batch lỗi tự được chia đôi đến khi chạy được; có thể chủ
động đặt `$env:AIC_OCR_INPUT_BATCH_SIZE = "4"`. Khi đổi model/ngưỡng, chữ ký cache
thay đổi và script tự rebuild các entry liên quan. Lỗi inference được retry tối đa
3 lần; dùng `--retry-failed` để thử lại các entry đã vượt giới hạn.

Đo thực tế trên máy RTX 4050 6 GB: mẫu tuần tự nhiều chữ đạt 2,13 ảnh/giây, mẫu
100 ảnh rải đều collection đạt 3,67 ảnh/giây, không có lỗi. Thời gian toàn bộ ước
tính khoảng 13–24 giờ tùy mật độ chữ; tiến trình có thể dừng và chạy tiếp từ state.

## 4) Chạy hệ thống

```powershell
# Terminal 1 — backend
cd backend
$env:PYTHONPATH = "C:\LASTDANCE\backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend
streamlit run streamlit_app.py
```

Mở http://localhost:8501, chọn trang **Textual KIS / Q&A / TRAKE / Export**.

## 5) Kiến trúc & phần công nghệ chính

- **Semantic retrieval**: giữ nguyên feature ảnh `clip-ViT-B-32`; truy vấn tiếng
  Anh dùng text tower gốc, tiếng Việt dùng
  `sentence-transformers/clip-ViT-B-32-multilingual-v1`. Cả hai chạy trên CPU và
  tìm trong cùng FAISS index 512 chiều.
- **Object filter**: đọc `detection_class_entities` + `detection_scores` từ Objects
  JSON (Faster R-CNN/OpenImages), lọc ngưỡng confidence ≥ 0.3.
- **OCR**: EasyOCR 1.7.2, CRAFT + `latin_g2`, chạy offline trên GPU; giữ ảnh tối đa
  2560 px để nhận chữ nhỏ, lưu các dòng confidence ≥ 0,05 cùng polygon, có state,
  retry và atomic
  checkpoint. So khớp OCR dùng ranh giới từ/cụm từ và edit-similarity theo độ dài,
  không dùng bảng sửa ký tự thủ công.
- **Query understanding dùng chung** (`app/services/query_processing.py`): cả KIS,
  Q&A và TRAKE đều nhận nguyên văn một truy vấn hoàn chỉnh; tự nhận diện VI/EN,
  tạo các visual prompt cùng ngôn ngữ, tách phần mô tả/câu hỏi của Q&A và tách
  chuỗi khoảnh khắc có thứ tự của TRAKE. Không còn từ điển dịch chắp vá. Object và
  OCR chỉ là bằng chứng re-rank phụ khi truy vấn thực sự chứa object hoặc yêu cầu
  đọc chữ; semantic CLIP vẫn là tín hiệu chính.
- **KIS nhiều cảnh trong đợt 1**: mỗi evidence unit tiếng Việt được Qwen3-VL dịch
  sang một caption tiếng Anh tương ứng trong một lần gọi; hai encoder được max-pool
  trên cùng unit nên bản dịch không được tính thành một phiếu ngữ nghĩa mới. Truy
  vấn nhiều cảnh lấy 800 candidate/prompt, truy vấn ngắn giữ 400.
- **Module Re-rank** (`app/rerank/`) — trọng tâm của hệ thống:
  - `fusion_scoring.py`: kết hợp điểm CLIP + object-match + OCR-match.
  - `temporal_smoothing.py`: khuếch đại điểm nếu các keyframe liền kề trong cùng
    video cũng có điểm cao.
  - `storyboard_alignment.py`: gom nhiều keyframe/evidence unit trên cùng video và
    kiểm tra các cạnh thời gian được nêu rõ trong truy vấn.
  - `visual_reranker.py`: Qwen3-VL so sánh tương đối các contact sheet theo bracket
    nhóm 3. Không dùng điểm tự chấm tuyệt đối của VLM vì model 2B không được hiệu
    chuẩn làm relevance scorer. Query ngắn được bổ sung keyframe trước/giữa/sau.
  - `contest_ranking.py`: portfolio tại đúng mốc 1/5/20/50/100, luôn điền đủ 100
    dòng nếu candidate pool đủ lớn.
- **Pipelines riêng** cho từng loại truy vấn (`app/pipelines/`):
  - `kis_pipeline.py`: bilingual scene recall → fusion → temporal/storyboard
    alignment → Qwen tournament → cutoff portfolio → exact-frame Top 1.
  - `qa_pipeline.py`: dùng toàn bộ mô tả để retrieval 100 khung hình, sau đó
    Qwen3-VL nhìn từng khung hình và trả lời toàn bộ truy vấn; không suy đoán answer từ số
    lượng object/OCR.
  - `trake_pipeline.py`: semantic retrieval cho từng khoảnh khắc đã tách, xếp hạng
    nhiều giả thuyết video theo độ phủ toàn chuỗi, rồi K-best beam alignment đảm
    bảo frame tăng theo thời gian và sinh top 100 tổ hợp.

## 6) Export & nộp bài (khớp đúng hướng dẫn chính thức)

Module `app/services/export_csv.py` sinh CSV theo đúng chuẩn:
- Không header, UTF-8, dấu phẩy, tên video **không có đuôi `.mp4`**.
- Q&A: tự động bọc `"..."` khi answer có dấu phẩy/ngoặc kép/xuống dòng.
- Validator kiểm tra: tối đa 100 dòng, đúng số cột theo loại truy vấn, TRAKE đúng số
  frame theo số event, answer ≤ 100 ký tự — giúp tránh mất lượt nộp (chỉ được 3
  lần/gói truy vấn).
- Trang **Export** trên UI có thể xuất từng CSV hoặc đóng gói nhiều CSV vào
  `submission.zip` (đúng cấu trúc thư mục `submission/` bắt buộc).

## 7) API chính

| Method | Path | Mô tả |
|---|---|---|
| POST | `/search/kis` | Một trường `text`; luôn yêu cầu top 100 Textual KIS |
| POST | `/search/qa` | Một trường `text` chứa cả mô tả và câu hỏi; top 100 |
| POST | `/search/trake` | Một trường `text` chứa toàn bộ chuỗi; tự tách moments, top 100 |
| GET | `/video/{video_id}/keyframe/{local_idx}` | Trả ảnh keyframe (dùng local_idx) |
| POST | `/submission/add` | Thêm danh sách kết quả đã xếp hạng vào submission |
| GET | `/submission/{query_id}/validate` | Kiểm tra định dạng trước khi xuất |
| GET | `/submission/{query_id}/export` | Xuất 1 file CSV |
| POST | `/submission/zip` | Đóng gói nhiều CSV thành `submission.zip` |

Ví dụ payload cho cả ba endpoint đều có cùng hình dạng:

```json
{
  "text": "Nguyên văn truy vấn do ban tổ chức cung cấp"
}
```

UI không còn thanh chọn `top_k` hay các ô nhập thủ công từng moment. Mỗi trang chỉ
có một ô truy vấn; backend khóa giới hạn ở 100 đúng theo thể lệ vòng sơ tuyển.

## 8) Hướng phát triển tiếp theo (backlog, không chặn MVP)

- Sau đợt 1: side index `Qwen3-VL-Embedding-2B` trên các cửa sổ video chồng lấn
  6–10 giây và 20–40 giây; `Qwen3-VL-Reranker-2B` làm cross-encoder. Không dùng một
  vector cho toàn video và không thay production trước khi A/B có ground truth.
- Learned fusion weights (khi có tập truy vấn/đáp án mẫu).
- ASR timestamped và OCR đầy đủ được hợp nhất như kênh evidence độc lập.
- Relevance feedback ngay trong phiên làm việc.
- Temporal alignment nâng cao cho TRAKE (HMM/CRF thay vì DP đơn giản).

Chi tiết quyết định đợt 1/đợt 2 nằm trong `docs/retrieval_upgrade_plan.md` và
runbook thi ở `docs/round1_contest_runbook.md`.
