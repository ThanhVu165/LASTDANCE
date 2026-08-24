# Nhánh 1 - Offline Indexing

Thư mục này triển khai pipeline frame-level theo `docs/OFFLINE_INDEXING_SPEC.md`.
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

TransNetV2 là default tạm thời trong khi AutoShot weight chưa được xác nhận. Môi trường
Python 3.11 dùng package PyTorch đã pin; adapter vẫn tương thích API
`predict_video()`/`predictions_to_scenes()`. Module không tải weight trong lúc xử lý video:
bundled weight của package được kiểm tra SHA-256 cố định; external override phải cung cấp
checksum. Shot manifest ghi package, device, threshold, weight source và weight hash.

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
trước khi chạy batch. Tóm tắt contract: checkout cùng commit, import nguyên adapter của repo,
khóa package/threshold/weight checksum, chia `video_id` không giao nhau và chỉ bàn giao
manifest schema v2; worker CUDA phải đối chiếu đủ 5 video dev-subset trước production batch.

Xem `docs/ENVIRONMENT_SETUP.md` để dựng environment giống nhau trên Windows/Kaggle.
