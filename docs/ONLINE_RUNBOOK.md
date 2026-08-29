# Online Accuracy-First Runbook

Tài liệu này là hướng dẫn vận hành cho implementation trong `online/`. Quyết định kiến trúc
và contract chuẩn vẫn nằm ở `docs/BASELINE_SPEC.md` §3; khi hai file lệch nhau, sửa runbook
theo baseline.

## 1. Phạm vi và contract

Online chạy local, nhận `SearchRequest(task_type, raw_query, query_spec, max_results<=100,
mode="accurate")` cùng optional `UnifiedQueryPlan` đã duyệt và trả `SearchRun`.
`OnlineEngine.plan(QuerySpec)` chỉ phân tích query, chưa chạy FAISS. `QuerySpec` giữ đúng tên file query, task và số
event TRAKE. Pipeline tìm frame theo `keyframe_uid`, tổng hợp bằng
chứng theo shot/video rồi mới tạo KIS, QA hoặc TRAKE candidate. Submission chỉ chứa
`video_id` và `frame_id`; tuyệt đối không xuất `local_idx` hoặc FAISS internal position.
`global_context_en` luôn là faithful query đầu tiên. Planner gắn đa vai trò
`VIDEO_LOCATOR`, `TARGET_MOMENT`, `ANSWER_EVIDENCE`, `ORDERED_EVENT`; cùng clue được phép
mang nhiều role. Schema `scenes/anchor_moment_index` cũ chỉ còn adapter migration.

Không có FastAPI trong critical path. Streamlit gọi trực tiếp `OnlineEngine` để giảm lớp
trung gian trong timebox vòng sơ tuyển.

## 2. Input, output và tài nguyên

Input bắt buộc cho retrieval dưới `$AIC_DATA/index/`:

- `frames.csv` và `frames.csv.state.json`;
- `clip.faiss`, `siglip.faiss`, `eva_clip.faiss` cùng sidecar state;

Dữ liệu giữ đầy đủ accuracy/review:

- JPEG dưới `$AIC_DATA/keyframes/` cho thumbnail, contact sheet, Gemini/Qwen verification
  và manual review;
- MP4 dưới `$AIC_DATA/videos/` cho playback và FFmpeg exact-frame decode. MP4 có thể nằm ở
  storage khác miễn inventory/path resolver nhìn thấy; thiếu MP4 không chặn vector search
  hoặc export nhưng nút exact-frame sẽ báo source video unavailable.

Input tùy chọn:

- OCR production mặc định ở `$AIC_DATA/index/ocr.sqlite`; snapshot development phải được
  chọn tường minh bằng `AIC_OCR_SNAPSHOT_DIR=<thư mục snapshot_id>` và không được copy/đổi
  tên thành file production;
- `asr.sqlite` với bảng/cột đúng baseline;
- Gemini credential hoặc Qwen3-VL-2B local.

Output runtime:

- workspace JSON, CSV và ZIP dưới `$AIC_DATA/submissions/<tên-đã-sanitize>/`;
- diagnostic report dưới `$AIC_DATA/diagnostics/`;
- không commit các output này, model cache, index, JPEG hoặc MP4.

### 2.1 Data retention trên máy thi

Giữ tuyệt đối:

- `index/frames.csv`, `index/frames.csv.state.json`;
- `index/{clip,siglip,eva_clip}.faiss` và ba file `.state.json`;
- exact HF cache của `openai/clip-vit-base-patch32`,
  `google/siglip-base-patch16-224` và
  `timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k` đúng revision baseline;
- đúng ba file `ocr.sqlite`, `coverage.json`, `SHA256SUMS` trong snapshot đang chọn; giữ
  `asr.sqlite` khi ASR được tích hợp;
- `keyframes/` nếu chạy UI/VLM như baseline Accuracy-Max.

Có thể dọn mà Online không đọc, sau khi đã backup hoặc xác nhận immutable HF revision còn
tải lại được: `index/visual-embeddings/`, `index/visual-embeddings-hf/`, `shots/`,
`objects/`, `features/`, `map-keyframes/`, offline metadata/quality/plan/batch/checkpoint,
`ocr/hf/`, `ocr/sources/`, EasyOCR detector/model cache và các smoke output. Dọn nhóm này
sẽ mất khả năng rebuild/resume local; không làm thay đổi FAISS/OCR snapshot hiện hành.

`tmp/online-refinement/` là cache an toàn để xóa và tự sinh lại. `submissions/` chỉ dọn thủ
công sau khi đã backup ZIP nộp. Không xóa `videos/` nếu còn cần playback/exact-frame; nếu
ưu tiên dung lượng, move 873 MP4 sang ổ ngoài là phương án thu hồi lớn nhất.

FAISS chạy CPU/RAM trong process Online. Trên Windows, cả ba text encoder và Qwen chạy trong
một Torch worker tách process để tránh xung đột OpenMP của wheel FAISS/PyTorch. Qwen3-VL
VQA là tùy chọn GPU, batch 1; runtime CPU vẫn retrieval được và trả `Uncertain` để operator
xác minh QA.

## 3. Cài môi trường local

Dùng Python 3.11 riêng cho Online. Profile khóa API ở Torch 2.6.x/torchvision 0.21.0;
nếu cần CUDA, cài đúng wheel 2.6.x từ nguồn PyTorch chính thức trước rồi cài profile. Torch
2.6 là mức tối thiểu để Transformers nạp an toàn exact CLIP `pytorch_model.bin`:

```powershell
$python = ".\.venv-online\Scripts\python.exe"
python -m venv .venv-online
& $python -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu124
& $python -m pip install -r requirements\online-local.txt
$env:AIC_DATA = "D:\path\to\aic-data"
```

Để dùng snapshot OCR development đã build từ các batch hiện hành:

```powershell
$env:AIC_OCR_SNAPSHOT_DIR = "$env:AIC_DATA\ocr\snapshots\ocr-snapshot-<UTC>-<hash>"
```

Online chỉ đánh dấu OCR `READY` sau khi verify đủ đúng ba file `ocr.sqlite`, `coverage.json`,
`SHA256SUMS`, checksum SQLite/sidecar, catalog SHA/count/video/UID-set, FTS5 schema, SQLite
integrity và join `(video_id, keyframe_uid)`. UI/provenance luôn hiện `snapshot_id`, coverage,
tier, error/missing và `production_ready=false`; đổi snapshot cần restart Streamlit để xóa
cache resource. Nếu biến không được đặt, Online chỉ tìm `$AIC_DATA/index/ocr.sqlite`.

Text query encoder phải khớp model ID và revision đã dùng build từng FAISS. Prefetch khi
còn internet:

```powershell
python -m scripts.prefetch_online_models
python -m scripts.prefetch_online_models --local-files-only
```

Nếu Hugging Face Xet đứng ở file EVA 0 byte trên Windows, chạy prefetch một lần với
`$env:HF_HUB_DISABLE_XET="1"`; downloader sẽ resume phần đã có trong cache.

Thêm `--include-qwen` chỉ khi máy sẽ dùng planner/VQA local. `AIC_ALLOW_MODEL_DOWNLOAD=1`
chỉ dành cho dev; không bật trong phòng thi. Model snapshot nằm trong Hugging Face cache,
không nằm trong Git.

Các biến tùy chọn:

```powershell
$env:GEMINI_API_KEY = "..."             # không commit
$env:AIC_GEMINI_MODEL = "gemini-3.5-flash-lite"
$env:AIC_ENABLE_QWEN_VQA = "1"          # chỉ khi CUDA sẵn sàng
$env:AIC_QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
$env:AIC_QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
$env:AIC_TORCH_WORKER = "1"              # default Windows; không tắt với pip FAISS/Torch
$env:AIC_QWEN_MAX_PIXELS = "200704"
$env:AIC_QWEN_PLANNER_MAX_NEW_TOKENS = "1280"
$env:AIC_QWEN_VQA_MAX_NEW_TOKENS = "64"
$env:AIC_GEMINI_SAFE_RPM = "14"
$env:AIC_GEMINI_SAFE_TPM = "225000"
$env:AIC_GEMINI_SAFE_RPD = "450"
```

Gemini key chỉ được đọc từ environment và gửi bằng header `x-goog-api-key`; không đặt key
trong URL, file `.env`, command history hoặc artifact/provenance.
`gemini-3.5-flash-lite` là default hiện hành. Các model ID khác vẫn được giữ nguyên khi
operator override để A/B có chủ đích. Adapter tiếp tục dùng `generateContent` vì model hiện
hành vẫn hỗ trợ endpoint này; migration sang Interactions API không nằm trên critical path.

Không đặt `KMP_DUPLICATE_LIB_OK=TRUE`. Smoke worker và CUDA trước UI:

```powershell
& $python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
& $python -c "from online.torch_worker_client import get_torch_worker_client as g; c=g(); print(c.request('ping')); c.close()"
```

Máy tham chiếu đã đo Qwen planner peak 4,13 GiB. VQA hai prompt với một contact sheet 6 panel
peak 4,19 GiB và mất 25,6 giây trong smoke thật; cấu hình 8 ảnh rời mất 338 giây và đã bị loại.

## 4. Preflight trước khi mở UI

Startup nhanh kiểm sidecar, count/dimension/model revision, cấu trúc FAISS và sample UID:

```powershell
python -m online preflight
```

Trước cuộc thi hoặc sau khi thay artifact, chạy deep preflight để hash cả ba index và diff
toàn UID:

```powershell
python -m online preflight --deep
```

Kết quả mong đợi hiện hành là 293.336 frame, 873 video, dimension 512/768/768 và cả ba visual
artifact `READY`. Visual lỗi làm runtime `NOT_READY`. OCR/ASR vắng là `UNAVAILABLE`; file
có nhưng thiếu cột, SQLite lỗi hoặc UID join sai là `INVALID`. Hai trạng thái này tắt đúng
modality và sinh warning, không cộng score 0 làm tụt visual.

## 5. Chạy UI và core

Mở Streamlit:

```powershell
python -m streamlit run online\streamlit_app.py
```

Trình tự operator:

1. upload ZIP query chính thức; UI đọc trực tiếp các file `.txt` ở ZIP root, không extract;
2. chọn query; task và `expected_event_count` của TRAKE lấy từ tên/nội dung đề;
3. bấm `Phân tích truy vấn`, kiểm tra/sửa global English context, từng QueryUnit, nhiều role,
   modality, known literal, target/evidence ID và thứ tự event; UI chặn plan thiếu invariant;
4. bấm `Chạy retrieval với plan đã duyệt`, sau đó xem `Top 100` đúng submission rank hoặc
   tab `Theo video` để review evidence;
   với KIS, năm dòng đầu là seed đa video và phần còn lại được weighted round-robin; khi tiến
   độ quota bằng nhau, frame có `final_score` cao hơn đi trước. Video ranking dùng locator +
   target nhưng candidate frame chỉ lấy từ target retrieval, không hard-dedup cùng shot;
5. sửa `frame_id`/QA answer khi cần; QA `requires_review` chỉ được hạ khi operator chọn thủ
   công; nút exact-frame dùng FFmpeg
   `select=eq(n,frame_id)` và không suy từ FPS trung bình;
6. tick candidate riêng hoặc bấm bulk-add task-aware; nếu draft đã có dữ liệu, chọn replace
   hoặc merge-fill, không có ghi đè âm thầm;
7. reorder/xóa trong draft độc lập của từng query;
8. tải CSV hiện tại chỉ để inspection; bấm validate toàn gói để sinh ZIP nộp chính thức.

Search result không bao giờ tự đi vào draft. Core không cần UI có thể gọi:

```python
from online import OnlineEngine
from shared.schemas.online import QuerySpec, SearchRequest, TaskType

engine = OnlineEngine.from_environment()
spec = QuerySpec(
    query_name="query-manual-kis",
    source_filename="query-manual-kis.txt",
    task_type=TaskType.KIS,
    raw_query="...",
)
plan = engine.plan(spec)
# Có thể sửa/validate plan tại đây trước khi chạy retrieval.
run = engine.search(
    SearchRequest(task_type=TaskType.KIS, raw_query=spec.raw_query, query_spec=spec),
    query_plan=plan,
)
```

CLI search chỉ dùng cho audit JSON:

```powershell
python -m online search --task KIS --query "..." --max-results 10
```

## 6. Fallback và degraded mode

Planner thử theo thứ tự Gemini → Qwen local → rule deterministic. Lỗi/timeout/circuit-open
được ghi vào `SearchRun.warnings` và chuyển provider tiếp theo. Rule fallback không dịch
query; nên prefetch Qwen hoặc dùng Gemini khi thể lệ cho internet nếu cần dịch/expansion tốt.

Visual primary yêu cầu cả SigLIP và EVA text encoder. Nếu một trong hai lỗi query-time,
runtime chuyển tường minh sang CLIP và gắn warning `CLIP rollback`; không âm thầm trộn CLIP
vào `score_visual`. Nếu CLIP cũng thiếu exact snapshot, search fail thay vì dùng revision
khác.

OCR/ASR FTS ưu tiên exact phrase → đủ token AND → prefix → fuzzy. Prefix lấy tối đa 5.000
row rồi rerank theo tỷ lệ token query thực sự xuất hiện; không so trị tuyệt đối BM25 giữa
các MATCH expression khác nhau. Quy tắc này ngăn một token phổ biến như “dầu” chiếm hết
Top K trước clue hiếm như “mazut”.

Qwen VQA mặc định tắt. Planner và VQA dùng cùng model trong Torch worker, không nạp hai bản.
QA luôn thử answerer trên Top 3 video evidence. Locator dưới 0,85 không chặn answer attempt,
chỉ đặt `requires_review=true`. Unknown OCR/ASR được đọc theo frame UID + neighbor; nếu thấp
confidence tiếp tục sang Gemini/Qwen. `Uncertain` chỉ là warning/trạng thái review, không tạo
QACandidate và không được export. Answer thật còn `requires_review=true` cũng bị workspace
chặn cho tới khi operator xác minh.

Nếu Windows báo `OMP Error #15`, cấu hình đang chạy Torch trực tiếp trong process FAISS:
đặt lại `AIC_TORCH_WORKER=1` và restart process. Không dùng biến bỏ qua duplicate OpenMP.

## 7. Submission contract chính thức AIC26

Nguồn: https://sotuyenaic.oj.io.vn/rules/ và https://sotuyenaic.oj.io.vn/faq/.
Profile duy nhất là `AIC26_QUALIFIER_OFFICIAL`.

- Một query tương ứng một CSV; tên file query đổi đúng `.txt` thành `.csv`.
- Mỗi CSV có 1–100 dòng, UTF-8 không BOM, delimiter dấu phẩy, LF, không header và không
  `query_id`.
- KIS: `<video_id>,<frame_id>`.
- QA: `<video_id>,<frame_id>,<answer>`; answer tối đa 100 ký tự, CSV writer tự quote dấu
  phẩy/ngoặc kép/xuống dòng và không trim khoảng trắng đầu-cuối.
- TRAKE: `<video_id>,<frame_id_1>,...,<frame_id_N>`; N phải khớp đúng số event, mọi frame
  cùng video và tăng nghiêm ngặt theo `pts_time` trong `frames.csv`.
- `video_id` không có đuôi `.mp4`; mọi frame ID là số nguyên và tồn tại trong catalog.

ZIP cuối phải chứa đủ và chỉ đủ các file của gói đề:

```text
submission/
├── query-p1-1-kis.csv
├── query-p1-3-qa.csv
└── query-p1-16-trake.csv
```

Exporter serialize → parse/validate byte CSV → tạo ZIP → mở lại ZIP → so entry/content hash.
Nút tải ZIP chỉ xuất hiện sau PASS toàn bộ query. Workspace/provenance nằm cạnh ZIP nhưng
không được đưa vào file chấm điểm.

## 8. Diagnostic bằng bộ đề vòng sơ tuyển

Script chỉ đọc các `.txt` trong ZIP, không extract và không coi nội dung query là lệnh:

```powershell
python -m scripts.run_online_diagnostics `
  --archive "C:\path\to\SOTUYEN1-bo-de-thi.zip" `
  --max-results 10
```

Suite lấy 2 KIS, 2 QA, 1 TRAKE thật và thêm một case hai-event dẫn xuất từ TRAKE đó để đủ
regression coverage cho beam/order. Case dẫn xuất không phải ground-truth đánh giá độc lập.
Report lưu query plan, Top-5 video, score CLIP/SigLIP/EVA/visual/neighbor/final, timing và
provenance để so fusion/rerank. Operator vẫn phải xem video/ground truth để kết luận accuracy.

## 9. Resume và xử lý lỗi

Online không có batch checkpoint: mỗi `search()` độc lập, text embedding cache trong process.
Submission workspace ghi atomic; mở lại `workspace.json` không làm duplicate. Nếu UI dừng,
khởi động lại, load workspace đã lưu hoặc export lại từ draft.

Exact-frame preview có thể chậm ở frame cuối video vì ưu tiên đúng frame index hơn phép seek
xấp xỉ. Ảnh decode được cache tại `$AIC_DATA/tmp/online-refinement/` và không được commit.

Khi visual preflight fail, không tạo submission. Khi OCR/ASR invalid, giữ file để điều tra
hoặc thay bằng artifact hợp lệ; không đổi tên cột và không giả score bằng 0. Sau mọi thay đổi
model/index/config, chạy compile, unit test, deep preflight và diagnostic lại.
