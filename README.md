# LASTDANCE - AIC 2026 frame-level video retrieval

LASTDANCE nhận query tự nhiên và trả tối đa 100 kết quả cho KIS, QA hoặc TRAKE.
Từ ngày 24/08/2026, kiến trúc chính đã pivot từ Qwen video-window sang retrieval
frame-level: mỗi keyframe có vector riêng và được join bằng `keyframe_uid`.

## Nguồn chuẩn

Đọc theo thứ tự:

1. [`AGENTS.md`](AGENTS.md) - phạm vi và các invariant bắt buộc.
2. [`docs/BASELINE_SPEC.md`](docs/BASELINE_SPEC.md) - baseline hợp nhất, thắng khi tài liệu
   chi tiết còn lệch.
3. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) - snapshot trạng thái triển khai thực tế.
4. [`docs/ONLINE_RUNBOOK.md`](docs/ONLINE_RUNBOOK.md) - Online, OCR snapshot, UI và submission.
5. [`docs/ENVIRONMENT_SETUP.md`](docs/ENVIRONMENT_SETUP.md) - môi trường Windows/Kaggle.
6. [`offline/README.md`](offline/README.md) - entry point Nhánh 1.
7. [`docs/SHOT_DETECTION_RUNBOOK.md`](docs/SHOT_DETECTION_RUNBOOK.md) - chạy và bàn giao
   TransNetV2 giữa nhiều máy.
8. [`docs/VISUAL_EMBEDDING_RUNBOOK.md`](docs/VISUAL_EMBEDDING_RUNBOOK.md) - chuẩn bị input,
   chạy CLIP/SigLIP độc lập trên Kaggle và test checkpoint/resume thật.
9. [`docs/FAISS_INDEX_RUNBOOK.md`](docs/FAISS_INDEX_RUNBOOK.md) - build/add/validate từng
   `IndexIDMap` độc lập trên CPU local sau khi embedding batch hoàn tất.
10. [`docs/OCR_V2_PRODUCTION_PLAN.md`](docs/OCR_V2_PRODUCTION_PLAN.md) - checklist OCR v2 đã
    chốt: CRAFT cache → VietOCR → Paddle có điều kiện; bốn T4, log và checkpoint HF.
11. [`docs/OCR_RUNBOOK.md`](docs/OCR_RUNBOOK.md) - đầu mối vận hành OCR, phân biệt pipeline
    v2 hiện hành với lệnh EasyOCR/Vintern lịch sử. Contract chỉ nằm trong baseline §2.2.
12. [`docs/OCR_V2_SNAPSHOT_RUNBOOK.md`](docs/OCR_V2_SNAPSHOT_RUNBOOK.md) - tải đúng chín
    result production, union/validate và atomic-build SQLite development trên CPU local.
13. [`docs/OCR_V2_ONLINE_HANDOFF.md`](docs/OCR_V2_ONLINE_HANDOFF.md) - **nhóm Online đọc trước**:
    nhận code/dữ liệu, revision/checksum, setup CPU, adapter schema v3 và rollback.
14. [`docs/ASR_RUNBOOK.md`](docs/ASR_RUNBOOK.md) - ASR faster-whisper large-v3 chạy Kaggle
    GPU độc lập (Nhánh 3), audio extraction local, batch checkpoint, snapshot + publish index.
15. [`docs/QUALIFIER_ACCEPTANCE_RUNBOOK.md`](docs/QUALIFIER_ACCEPTANCE_RUNBOOK.md) - source-frame review,
    bộ chấm chính thức, gán nhãn 60 câu và điều kiện nghiệm thu.

ASR contract nằm trực tiếp trong `BASELINE_SPEC.md` §2A. Không duy trì spec/kiến trúc
archived song song; lịch sử quyết định nằm trong Git và Changelog của baseline.

## Kiến trúc mới

```text
video
  -> inventory bằng ffprobe
  -> shot detection (TransNetV2; production chạy GPU sau parity, CPU làm reference)
  -> 3 keyframe/shot + blur/dedup filtering
  -> CLIP + SigLIP + EVA-CLIP vector riêng cho từng keyframe
  -> frames.csv + 3 FAISS IndexIDMap + ocr.sqlite

audio
  -> Whisper/phoWhisper
  -> temporal alignment bằng pts_time
  -> asr.sqlite
```

Nhánh online sẽ gộp SigLIP + EVA-CLIP bằng SRRF thành đúng một `score_visual`; CLIP là
rollback. Sau đó mới fusion visual/OCR/ASR. Nhánh 1 không build index visual đã gộp.

## Cấu trúc repo hiện hành

```text
offline/          # Nhánh 1 Offline Indexing
online/           # Nhánh 2 Accuracy-Max + Streamlit trực tiếp
shared/           # Schema/interface dùng chung
scripts/          # CLI local/Kaggle
tests/            # contract/regression test
```

## Chạy lát cắt offline hiện tại

Bootstrap Windows dựng Python 3.11 + FFmpeg đã pin, kiểm tra checksum, doctor, compile và
test. Toàn bộ dữ liệu (ngoài videos) được định vị trong project folder `data/` (hoặc set env
`AIC_DATA`, default `data/`). Videos nằm ngoài project tại `F:\LASTDANCE-DATA\videos`.

```powershell
.\scripts\bootstrap_miniforge_windows.ps1
$env:AIC_DATA = "data"  # project folder (hoặc mount khác nếu cần)
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_inventory -PythonArguments @("--limit", "10")
```

Pipeline hiện có inventory, TransNetV2 shot detection, Begin/Middle/End extraction,
Laplacian/pHash quality manifest, `frames.csv` catalog builder và visual embedding shard
builder cùng FAISS `IndexIDMap` builder/validator fail-closed. Weight
TransNetV2 bundle trong package pin và được kiểm tra SHA-256 trước load; không tải weight
trong lúc xử lý video. Batch runner có checkpoint/resume riêng theo từng video và chỉ nâng
trạng thái sau khi manifest đã atomic-publish rồi validate lại. CLIP/SigLIP/EVA-CLIP đã
hoàn tất 9/9 batch; local FAISS đã validate đủ 293.336 UID/873 video.

OCR v2 đã chốt 04/09/2026 theo [`BASELINE_SPEC.md §2.2`](docs/BASELINE_SPEC.md): tái sử dụng
CRAFT bbox từ chín archive HF → VietOCR mọi crop gốc → Paddle có điều kiện → residual
Gemini tùy chọn, cần duyệt riêng. Không chạy lại EasyOCR/Vintern, không bật làm nét sau
trial 30 crop. Bốn worker đã hoàn tất chín batch trên T4 và upload result/report
content-addressed lên HF. Migration/validator/build SQLite v2 đã materialize và validate
snapshot development thật `ocr-snapshot-20260904T081629Z-66ecea73cce1`, phủ đủ
293.336 UID với 269.259 FTS row; vẫn giữ `complete=false`, `production_ready=false`. Xem
[hướng dẫn recognition](docs/OCR_V2_PRODUCTION_RUNBOOK.md) và
[hướng dẫn snapshot](docs/OCR_V2_SNAPSHOT_RUNBOOK.md).
Gate B có kết quả thử nghiệm, không phải PASS định lượng hay production-ready.
Online vẫn dùng snapshot EasyOCR development qua `AIC_OCR_SNAPSHOT_DIR`; chưa đổi artifact
đang phục vụ. Snapshot bất biến, có coverage/checksum; không commit snapshot/JSONL vào Git.
Xem [plan triển khai](docs/OCR_V2_PRODUCTION_PLAN.md) và [runbook](docs/OCR_RUNBOOK.md).
ASR hiện còn thiếu.

## Invariant quan trọng

- Submission và dedup dùng `(video_id, frame_id)`, không dùng `local_idx`.
- FAISS/OCR/ASR join bằng `keyframe_uid` deterministic.
- Không mean-pool keyframe thành một vector shot/video.
- Ba FAISS index build độc lập và vector lưu `float16`.
- Không set `complete=true` nếu thiếu bất kỳ Publishing Criteria nào.
- Không hardcode FPS, resolution, duration hoặc đường dẫn dữ liệu.
- Không tự động phụ thuộc cloud khi internet phòng thi chưa được xác nhận.
