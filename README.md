# LASTDANCE - AIC 2026 frame-level video retrieval

LASTDANCE nhận query tự nhiên và trả tối đa 100 kết quả cho KIS, QA hoặc TRAKE.
Từ ngày 24/08/2026, kiến trúc chính đã pivot từ Qwen video-window sang retrieval
frame-level: mỗi keyframe có vector riêng và được join bằng `keyframe_uid`.

## Nguồn chuẩn

Đọc theo thứ tự:

1. [`AGENTS.md`](AGENTS.md) - phạm vi và các invariant bắt buộc.
2. [`docs/BASELINE_SPEC.md`](docs/BASELINE_SPEC.md) - contract tổng thể offline/online.
3. [`docs/OFFLINE_INDEXING_SPEC.md`](docs/OFFLINE_INDEXING_SPEC.md) - Nhánh 1.
4. [`docs/ASR_SPEC.md`](docs/ASR_SPEC.md) - data contract Nhánh 3.
5. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) - trạng thái triển khai thực tế.
6. [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) - môi trường Windows/Kaggle.
7. [`docs/SHOT_DETECTION_RUNBOOK.md`](docs/SHOT_DETECTION_RUNBOOK.md) - chạy và bàn giao
   TransNetV2 giữa nhiều máy.

Các tài liệu window-first cũ được giữ lại làm lịch sử, không còn là runtime instruction.

## Kiến trúc mới

```text
video
  -> inventory bằng ffprobe
  -> shot detection (TransNetV2 tạm thời; AutoShot khi có weight)
  -> 3 keyframe/shot + blur/dedup filtering
  -> CLIP + SigLIP + BEiT-3 vector riêng cho từng keyframe
  -> frames.csv + 3 FAISS IndexIDMap + ocr.sqlite

audio
  -> Whisper/phoWhisper
  -> temporal alignment bằng pts_time
  -> asr.sqlite
```

Nhánh online sẽ gộp SigLIP + BEiT-3 bằng SRRF thành đúng một `score_visual`; CLIP là
rollback. Sau đó mới fusion visual/OCR/ASR. Nhánh 1 không build index visual đã gộp.

## Cấu trúc repo trong giai đoạn migration

```text
offline/          # Nhánh 1 mới
shared/schemas/   # FrameRecord, OcrResult, AsrSegment
scripts/          # CLI local/Kaggle
tests/            # contract test cho kiến trúc mới
backend/app/      # implementation window-first cũ, chưa wire vào pipeline mới
```

## Chạy lát cắt offline hiện tại

Bootstrap Windows dựng Python 3.11 + FFmpeg đã pin, kiểm tra checksum, doctor, compile và
test. Toàn bộ dữ liệu được định vị qua `AIC_DATA`; không ghi path tuyệt đối vào artifact.

```powershell
.\scripts\bootstrap_miniforge_windows.ps1
$env:AIC_DATA = "D:\path\to\aic-data"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_inventory -PythonArguments @("--limit", "10")
```

Pipeline hiện có inventory, TransNetV2 shot detection, Begin/Middle/End extraction,
Laplacian/pHash quality manifest và `frames.csv` catalog builder fail-closed. Weight
TransNetV2 bundle trong package pin và được kiểm tra SHA-256 trước load; không tải weight
trong lúc xử lý video. Embedding/OCR GPU chưa chạy trong lát cắt này.

## Invariant quan trọng

- Submission và dedup dùng `(video_id, frame_id)`, không dùng `local_idx`.
- FAISS/OCR/ASR join bằng `keyframe_uid` deterministic.
- Không mean-pool keyframe thành một vector shot/video.
- Ba FAISS index build độc lập và vector lưu `float16`.
- Không set `complete=true` nếu thiếu bất kỳ Publishing Criteria nào.
- Không hardcode FPS, resolution, duration hoặc đường dẫn dữ liệu.
- Không tự động phụ thuộc cloud khi internet phòng thi chưa được xác nhận.
