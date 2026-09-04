# Nhánh 1 - Offline Indexing

Thư mục này triển khai Nhánh 1 theo nguồn chuẩn duy nhất `docs/BASELINE_SPEC.md` §2.
Nhánh Online hiện hành nằm độc lập trong `online/` và chỉ nhận artifact qua contract đã khóa.

## Lát cắt hiện có

- schema dùng chung trong `shared/schemas/`;
- `keyframe_uid` deterministic bằng BLAKE2b;
- inventory đọc FPS, resolution và duration thật qua `ffprobe`;
- interface `ShotDetector` và adapter TransNetV2 lazy-load;
- keyframe plan Begin/Middle/End dùng timestamp thật theo từng decoded frame;
- exact-frame extraction bằng một lượt FFmpeg decode/video, stage đủ batch rồi atomic-publish
  từng JPEG với checkpoint/resume;
- Laplacian variance + pHash report/filter theo threshold CLI, không xóa JPEG nguồn;
- `frames.csv` builder yêu cầu plan/quality UID + SHA khớp và publish state hash fail-closed;
- checkpoint theo signature, atomic write và không có cờ ready chỉnh tay;
- evaluator fail-closed cho Publishing Criteria của ba visual index.
- visual embedding shard builder chạy một modality/lệnh, float16 + L2, signature/checkpoint
  riêng và có intentional-stop gate để xác minh resume thật trên Kaggle.
- FAISS builder chạy CPU local, một modality/lệnh, add batch rời nhau bằng `keyframe_uid`
  và diff ID thật với `frames.csv`; không chờ hoặc rebuild modality khác.

TransNetV2 đã được chốt làm shot detector production. Môi trường Python 3.11 dùng package
PyTorch đã pin; Windows NVIDIA GPU là worker production sau parity 5/5, còn CPU giữ làm
reference/fallback. Module không tải weight trong lúc xử lý video: bundled weight của package
được kiểm tra SHA-256 cố định; external override phải cung cấp checksum. Shot manifest ghi
package, device, threshold, weight source và weight hash.

## Chạy pipeline một video

Chạy từ root repository sau khi bootstrap theo `docs/ENVIRONMENT_SETUP.md`:

```powershell
$env:AIC_DATA = "data"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "offline-local")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_inventory `
  -PythonArguments @("--limit", "1")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.detect_shots `
  -PythonArguments @("F:\LASTDANCE-DATA\videos\L01_V001.mp4")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_keyframe_plan `
  -PythonArguments @(
    "F:\LASTDANCE-DATA\videos\L01_V001.mp4",
    "$env:AIC_DATA\shots\L01_V001.json"
  )

.\scripts\run_offline_windows.ps1 `
  -Module scripts.extract_keyframes `
  -PythonArguments @(
    "$env:AIC_DATA\index\keyframe-plans\L01_V001.json",
    "--checkpoint-every", "25"
  )

.\scripts\run_offline_windows.ps1 `
  -Module scripts.filter_keyframes `
  -PythonArguments @("$env:AIC_DATA\index\keyframe-plans\L01_V001.json")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_frames_catalog `
  -PythonArguments @(
    "--plan", "$env:AIC_DATA\index\keyframe-plans\L01_V001.json",
    "--quality", "$env:AIC_DATA\index\keyframe-quality\L01_V001.json"
  )
```

## Chạy keyframe toàn collection

Chỉ chạy sau khi parity Shot Detection CPU–GPU đủ 5/5:

```powershell
python -m scripts.run_keyframe_batch `
  --collection `
  --data-root "$env:AIC_DATA" `
  --fail-fast
```

Hoặc chạy hai terminal bằng hai shard deterministic, không chạy đồng thời với lệnh
collection tuần tự ở trên:

```powershell
# Terminal 1
python -m scripts.run_keyframe_batch --collection --data-root "$env:AIC_DATA" `
  --shard-count 2 --shard-index 0 --fail-fast

# Terminal 2
python -m scripts.run_keyframe_batch --collection --data-root "$env:AIC_DATA" `
  --shard-count 2 --shard-index 1 --fail-fast
```

Hai shard lấy vị trí chẵn/lẻ trong inventory đã sort, nên tập ID disjoint và hợp lại đúng
toàn collection. Mỗi shard tự dùng checkpoint, report và extraction state riêng dưới
`AIC_DATA/index/keyframe-batches/`.

Chạy thêm finalizer với đúng `--shard-count` đã dùng. Nó chờ mọi shard, kiểm tra coverage
exact/disjoint, rồi tự build và validate `frames.csv`:

```powershell
python -m scripts.finalize_keyframe_collection `
  --data-root "$env:AIC_DATA" `
  --shard-count 2 `
  --poll-seconds 30
```

Batch hiện tại trên máy tham chiếu dùng 4 shard, nên finalizer tương ứng dùng
`--shard-count 4`.

Runner kiểm tra tập shot manifest khớp chính xác inventory trước khi ghi, rồi chạy tuần tự
plan → exact extraction → quality cho từng video. Không truyền threshold nghĩa là quality
report-only, giữ toàn bộ keyframe để tránh filter mù. Artifact vận hành mặc định:

```text
AIC_DATA/index/keyframe-batches/collection.checkpoint.json
AIC_DATA/index/keyframe-batches/collection.json
AIC_DATA/index/keyframe_extraction_state.json
```

Ngắt bằng `Ctrl+C` rồi chạy lại đúng lệnh: runner xác minh signature của MP4, shot
manifest, JPEG config và quality config; stage hợp lệ được skip, extraction dở tiếp tục từ
checkpoint. Signature lệch hoặc file đã đánh dấu xong nhưng mất/rỗng sẽ fail closed.

## Build `frames.csv` toàn collection

Sau khi mọi video đã có đủ plan và quality manifest, không cần liệt kê 873 cặp file:

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_frames_catalog `
  -PythonArguments @("--collection")
```

Collection mode lấy tập `video_id` canonical từ
`AIC_DATA/index/inventory.json`, rồi yêu cầu khớp chính xác:

```text
AIC_DATA/index/keyframe-plans/<video_id>.json
AIC_DATA/index/keyframe-quality/<video_id>.json
```

Thiếu hoặc thừa dù chỉ một manifest thì lệnh dừng, không publish catalog partial. Output là
`AIC_DATA/index/frames.csv` và `frames.csv.state.json`; sidecar khóa SHA-256, số record,
số video và provenance của từng cặp plan/quality. Cách truyền lặp `--plan/--quality` vẫn
được giữ cho smoke/dev subset, không dùng nó để publish production toàn collection.

Các CLI còn lại dùng runner theo cùng mẫu `-Module ... -PythonArguments @(…)`. Khi
environment đã activate đúng, có thể gọi `python -m ...` trực tiếp.

Lần chạy `filter_keyframes` không truyền threshold là report-only để benchmark dev subset.
Chỉ truyền `--blur-threshold`/`--phash-max-distance` sau khi chốt ngưỡng; selection manifest
luôn giữ ít nhất một keyframe/shot. `build_frames_catalog` chỉ nhận plan + quality manifest
khớp SHA và không được dùng catalog smoke/partial làm production index.

Inventory/keyframe/dedup là local-CPU. Shot detection dùng CPU làm reference và có worker
Windows/Colab CUDA fail-closed sau parity gate 5 video. Visual embedding chạy trên Kaggle:
CLIP/SigLIP/EVA-CLIP đều đã qua dev gate T4 và production 9/9 batch; ba modality
publish/resume độc lập theo `keyframe_uid`.

## Handoff shot detection giữa nhiều máy

Runbook vận hành chuẩn nằm tại `docs/SHOT_DETECTION_RUNBOOK.md`. Đồng đội phải đọc file này
trước khi chạy batch. Tóm tắt contract: checkout cùng commit, khóa
package/device/threshold/weight checksum, chia `video_id` không giao nhau và dùng checkpoint
riêng cho mỗi worker/namespace. Mỗi video chỉ lên `1/1` sau khi manifest schema v2 đã
atomic-publish và validate lại; worker CUDA phải đối chiếu đủ 5 video dev-subset trước
production batch.

## Handoff visual embedding Kaggle

Đọc `docs/VISUAL_EMBEDDING_RUNBOOK.md` trước khi upload input/chạy GPU. CLIP và SigLIP đã
pin immutable revision và qua dev gate Kaggle T4; mỗi modality publish/resume độc lập.
EVA-CLIP đã PASS đúng quy trình exit 75 → process mới resume → validator trên Kaggle T4;
bằng chứng SHA-256 và notebook production 9 batch nằm trong Visual runbook. Production EVA
đã hoàn tất bằng `notebooks/kaggle_eva_clip_production.ipynb`, batch size 32 và namespace HF
riêng; snapshot handoff cuối là commit `938aefd437ab8db61fc6599d613aedcf4921d71e`.
BEiT-3 đã bị loại vĩnh viễn, không mở lại audit/checkpoint/adapter.

## OCR v2 — recognition 9/9 và snapshot development local đã validate

Nhận kết quả và tích hợp Nhánh 2: [OCR_V2_ONLINE_HANDOFF.md](../docs/OCR_V2_ONLINE_HANDOFF.md).
Ground truth được hoãn theo quyết định người dùng để bàn giao development; không phải accuracy PASS.

Nguồn chuẩn duy nhất: [BASELINE_SPEC.md](../docs/BASELINE_SPEC.md) §2.2, cập nhật 04/09/2026.
Pipeline: CRAFT bbox cache từ 9 archive HF → VietOCR mọi crop gốc → Paddle có điều kiện
→ residual tùy chọn Gemini → JSONL → union/SQLite local. Không chạy lại EasyOCR/Vintern;
không bật làm nét. Bốn T4 có log 30 giây, checkpoint local từng minibatch + HF verified
theo spec; không dùng máy Codex chạy model. Theo dõi checklist trong
[OCR_V2_PRODUCTION_PLAN.md](../docs/OCR_V2_PRODUCTION_PLAN.md).

Planner/canary/recognition: `notebooks/kaggle_ocr_v2_production.ipynb`, hướng dẫn tại
[OCR_V2_PRODUCTION_RUNBOOK.md](../docs/OCR_V2_PRODUCTION_RUNBOOK.md). Bốn worker đã chạy
xong chín batch T4/HF thật; output recognition vẫn không phải terminal envelope/SQLite final.
Planner và worker đọc `frames.csv`/state từ dataset Kaggle qua `CATALOG_PATH` (hoặc tự tìm
dưới `INPUT_ROOT`); HF chỉ cung cấp archive OCR và lưu kết quả, không cần upload catalog.
Cell cuối có log từng bước, heartbeat 20 giây và stream stdout/stderr; có thể copy từ
`scripts/kaggle_ocr_v2_production_launch.py` để cập nhật notebook đang dùng theo runbook.

Migration/union dùng `offline/ocr_v2_snapshot.py`, pin source bằng
`scripts/sync_ocr_v2_results.py`, build/validate bằng `scripts/build_ocr_v2_snapshot.py` và
`scripts/validate_ocr_v2_snapshot.py`; xem
[OCR_V2_SNAPSHOT_RUNBOOK.md](../docs/OCR_V2_SNAPSHOT_RUNBOOK.md). Schema coverage v3 giữ
engine VietOCR/Paddle thật và tương thích song song với reader snapshot legacy. Artifact
thật `ocr-snapshot-20260904T081629Z-66ecea73cce1` đã qua validator độc lập với
293.336/293.336 UID và 269.259 FTS row; chưa sửa Online hoặc đổi snapshot đang phục vụ.

### Notebook thử nghiệm và evidence (không phải runner production)

Khi cần đánh giá lỗi EasyOCR Batch 01 trước khi dùng thêm GPU, đọc
`docs/OCR_V2_EXPERIMENT_RUNBOOK.md`. Quy trình tạo bundle cân bằng 100 frame/120 crop từ JPEG
và archive có checksum bằng `notebooks/kaggle_ocr_v2_review.ipynb`, chấm Gate A bằng nhãn
tay, rồi dùng
`notebooks/kaggle_ocr_v2_gate_b.ipynb` để so cached EasyOCR với PaddleOCR/VietOCR trên một T4.
Notebook không gọi Vintern/Gemini, không build SQLite và không được đánh dấu artifact
production ready. Người dùng đã chọn triển khai v2 theo deadline và evidence visual/runtime;
không được diễn giải quyết định này thành Gate A/B PASS định lượng.

Thử nghiệm làm nét bổ sung: đọc `docs/OCR_V2_SHARPEN_TRIAL_RUNBOOK.md`, dùng
`notebooks/kaggle_ocr_v2_sharpen.ipynb` trên session T4 đã có VietOCR. So 30 crop × 3
phương án, có log/checkpoint và sheet full text. Trial đã xong 90/90; review không xác nhận
đủ ba crop cải thiện rõ nên giữ gốc, không cần chạy lại trước production. Các notebook
Gate A/B/sharpen chỉ tái lập evidence, không thực hiện chạy đủ chín batch.

## Build FAISS local

Sau khi một embedding modality đã hoàn tất và được tải về, đọc
`docs/FAISS_INDEX_RUNBOOK.md`. Builder yêu cầu checkpoint/resume verified, dùng đúng
`IndexIDMap(IndexFlatIP)`, hỗ trợ add batch video rời nhau và publish sidecar SHA-bound sau
cùng. `complete=true` của một sidecar chỉ thuộc modality/video đã khai báo, chưa phải trạng
thái Ready của toàn pipeline.

Closure Visual ngày 27/08/2026: local đã build và validate PASS `clip.faiss` (dim 512),
`siglip.faiss` (dim 768) và `eva_clip.faiss` (dim 768), mỗi index có đúng 293.336 UID/873
video, 9 source batch, finite/L2 norm và checkpoint/resume verified. Các artifact này nằm
dưới `data/index` và không được commit vào Git.

## Nhánh 3 — Offline ASR (Audio Stream Recognition)

**Tóm tắt:** ASR chạy Kaggle GPU độc lập, không tranh quota với visual embedding. Engine là
**faster-whisper large-v3** (CTranslate2, đa ngôn ngữ, tự phát hiện vi/en). Luồng:

1. **Local CPU:** Extract 16kHz mono FLAC từ video bằng FFmpeg
   - Input: `F:\LASTDANCE-DATA\videos\*.mp4` (external path)
   - Output: `data/audio/{video_id}.flac` (project folder)
   - Command: `python -m scripts.extract_asr_audio --collection --videos-root "F:\LASTDANCE-DATA\videos" --data-root "data"`
   - Checkpoint/resume qua `CheckpointStore`

2. **Kaggle GPU (2–4 worker song song):** Transcribe FLAC → segment list (per-video JSONL
   envelope)
   - faster-whisper large-v3 + Silero VAD filter (mặc định)
   - Checkpoint mỗi 5 video, push batch archival lên HF Dataset

3. **Local handoff + snapshot builder:** pin HF revision, validate manifest/checkpoint,
   dedupe overlap tương đương, quarantine conflict và chuẩn hóa timestamp có audit; sau đó
   build union JSONL → `asr.sqlite` (FTS5, 7 cột khớp `online/fts.py`) + `coverage.json`
   - Source: `data/asr/hf-staging/asr/archives/` và `asr/checkpoints/`
   - Handoff: `python -m scripts.materialize_asr_handoff` → union JSONL + audit JSON
   - Output: `data/asr/snapshots/asr-snapshot-...`

4. **Publish:** Atomic copy snapshot vào `data/index/asr.sqlite` (Online đã mount sẵn)

**Paths cheatsheet:**
- Videos (external): `F:\LASTDANCE-DATA\videos` — KHÔNG sửa, đường dẫn cố định
- Project data: `data/` hoặc set `$env:AIC_DATA = "data"`
- Audio output: `data/audio/`
- HF cache: `data/hf-cache/asr/`
- Snapshots: `data/asr/snapshots/`
- Final index: `data/index/asr.sqlite` ← Online đọc từ đây

**Đặc điểm:**
- Không cascade/tầng như OCR — chỉ một engine, tránh rủi ro deadline
- Batch = tái dùng 9 ranh giới video OCR/visual (`batch-01`..`batch-09`), cùng
  `video_set_sha256`
- HF namespace: `asr/audio/batch-XX/`, `asr/archives/batch-XX/`, `asr/snapshots/...`
- Granularity = per-video (1 JSONL = 1 video + list segment), không như OCR per-keyframe

**Publishing Criteria (§2A.4):** Video `complete=true` khi thỏa audio thật + segment hợp lệ
+ keyframe_uid_nearest match + no error status + FTS5 query text đúng.

**Runbook đầy đủ:** `docs/ASR_RUNBOOK.md` — chứa lệnh extract local, push HF, chạy Kaggle,
pull snapshot, build/publish index, checkpoint drill, troubleshoot degraded mode.

Xem `docs/ENVIRONMENT_SETUP.md` để dựng environment giống nhau trên Windows/Kaggle.
