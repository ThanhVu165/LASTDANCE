# Nhánh 1 - Offline Indexing

Thư mục này triển khai Nhánh 1 theo nguồn chuẩn duy nhất `docs/BASELINE_SPEC.md` §2.
`backend/app` là implementation window-first cũ và không được import vào nhánh này.

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
$env:AIC_DATA = "D:\path\to\aic-data"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "offline-local")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_inventory `
  -PythonArguments @("--limit", "1")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.detect_shots `
  -PythonArguments @("$env:AIC_DATA\videos\L01_V001.mp4")

.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_keyframe_plan `
  -PythonArguments @(
    "$env:AIC_DATA\videos\L01_V001.mp4",
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
Windows/Colab CUDA fail-closed sau parity gate 5 video. Các bước CLIP/SigLIP/BEiT-3 chưa
được chạy hoặc tải model trong lát cắt này.

## Handoff shot detection giữa nhiều máy

Runbook vận hành chuẩn nằm tại `docs/SHOT_DETECTION_RUNBOOK.md`. Đồng đội phải đọc file này
trước khi chạy batch. Tóm tắt contract: checkout cùng commit, khóa
package/device/threshold/weight checksum, chia `video_id` không giao nhau và dùng checkpoint
riêng cho mỗi worker/namespace. Mỗi video chỉ lên `1/1` sau khi manifest schema v2 đã
atomic-publish và validate lại; worker CUDA phải đối chiếu đủ 5 video dev-subset trước
production batch.

## Handoff visual embedding Kaggle

Đọc `docs/VISUAL_EMBEDDING_RUNBOOK.md` trước khi upload input/chạy GPU. CLIP và SigLIP có
candidate dev đã pin immutable revision; mỗi modality publish/resume độc lập. BEiT-3 đang
fail-closed chờ chốt official Microsoft UniLM retrieval checkpoint, không được thay bằng
BEiT thường. Chưa modality nào được coi là checkpoint/resume verified cho tới khi chạy quy
trình exit 75 → process mới resume → validator PASS trên Kaggle thật.

## Build FAISS local

Sau khi một embedding modality đã hoàn tất và được tải về, đọc
`docs/FAISS_INDEX_RUNBOOK.md`. Builder yêu cầu checkpoint/resume verified, dùng đúng
`IndexIDMap(IndexFlatIP)`, hỗ trợ add batch video rời nhau và publish sidecar SHA-bound sau
cùng. `complete=true` của một sidecar chỉ thuộc modality/video đã khai báo, chưa phải trạng
thái Ready của toàn pipeline.

Xem `docs/ENVIRONMENT_SETUP.md` để dựng environment giống nhau trên Windows/Kaggle.
