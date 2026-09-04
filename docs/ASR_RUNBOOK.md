# ASR Runbook — faster-whisper Large-v3 on Kaggle

Nhánh 3 — Audio Stream Recognition (ASR) dùng **faster-whisper Large-v3** (CTranslate2,
đa ngôn ngữ, tự phát hiện vi/en per-video). Chạy trên Kaggle GPU độc lập, không tranh quota
với Nhánh 1 (keyframe/embedding/OCR). Kết quả sync về local qua HuggingFace Dataset, build
immutable SQLite snapshot, publish vào Online index.

> **Nguồn chuẩn:** `docs/BASELINE_SPEC.md` §2A.

---

## 1. Scope & Design

### 1.1 Pipeline Flow

```text
AIC_DATA/videos/*.mp4 (873 video)
  ↓ [local CPU, FFmpeg]
AIC_DATA/audio/{video_id}.flac (16kHz mono FLAC, per-video)
  ↓ push audio/batch-XX/ theo 9 batch (batch-01..batch-09) lên HF Dataset
  ↓ [Kaggle GPU, 2–4 worker account song song]
  faster-whisper large-v3 + VAD filter, per-video checkpoint/resume
  → AsrRecordEnvelope JSONL (1 dòng = 1 video chứa list segment)
  + AsrShardManifest (batch completion gate)
  → push asr/archives/batch-XX/ lên HF Dataset
  ↓ [local, snapshot_download toàn bộ asr/]
  ↓ [local, build_asr_snapshot]
  asr.sqlite (FTS5, 7 cột khớp online/fts.py contract)
  + coverage.json (per-video status, Publishing Criteria audit)
  + SHA256SUMS (atomic integrity)
  → asr/snapshots/asr-snapshot-<UTC>-<hash>/ (immutable development snapshot)
  ↓ [local, publish_asr_index]
  Publishing Criteria §2A.4 validation + atomic copy
  → AIC_DATA/index/asr.sqlite (Online đã mount sẵn)
```

### 1.2 Key Differences from OCR

- **Granularity:** ASR công việc = 1 video (1 JSONL dòng chứa list segment), không phải 1 keyframe/region.
- **Engine:** single model (faster-whisper large-v3), không cascade/tầng (tránh rủi ro
  OCR gặp lúc deadline).
- **Batch:** tái dùng đúng 9 batch video đã pin OCR/visual
  (`batch-01`..`batch-09`, cùng `video_set_sha256`), không phát minh ranh giới mới.
- **Index publish:** `publish_asr_index.py` (mới, không có tương đương OCR) vì Online đọc
  thẳng `AIC_DATA/index/asr.sqlite` (không có OCR snapshot-dir override), nên không sửa
  `online/config.py`.

---

## 2. Local Setup: Audio Extraction

### 2.1 Extract 16kHz Mono FLAC (per-video)

Dùng `scripts/extract_asr_audio.py` (CLI wrapping `offline/asr_audio.py`). Yêu cầu:
- `ffmpeg` hệ thống (đã dùng cho Shot Detection/keyframe)
- `AIC_DATA` env var (default `data/`)

#### Single Video

```powershell
$env:AIC_DATA = ""
python -m scripts.extract_asr_audio `
  --video "$env:AIC_DATA\videos\L21_V001.mp4" `
  --output-dir "$env:AIC_DATA\audio"
```

Output: `AIC_DATA/audio/L21_V001.flac` (16kHz mono, ~40% size of original WAV).

#### Batch (All 873 Videos)

```powershell
python -m scripts.extract_asr_audio `
  --collection `
  --data-root "$env:AIC_DATA"
```

Hoặc 2 shard song song (tương tự keyframe extraction):

```powershell
# Terminal 1
python -m scripts.extract_asr_audio --collection --data-root "$env:AIC_DATA" `
  --shard-count 2 --shard-index 0

# Terminal 2
python -m scripts.extract_asr_audio --collection --data-root "$env:AIC_DATA" `
  --shard-count 2 --shard-index 1
```

**Checkpoint:** `AIC_DATA/audio/collection.checkpoint.json` (signature-aware resume).
Ngắt + chạy lại = skip đã xong, tiếp tục từ checkpoint.

**Kết quả:** ~873 FLAC file, tổng ~25GB (phút cắm tùy clip length).

---

## 3. Push Audio to HuggingFace

Audio batch được tổ chức theo 9 batch đã pin:

```text
asr/audio/batch-01/  (100 video, e.g. L21_V001..L24_V016)
asr/audio/batch-02/  (100 video, e.g. L24_V017..L25_V072)
...
asr/audio/batch-09/  (73 video, e.g. L30_V024..L30_V096)
```

**Tay push (recommended)**

```powershell
git clone https://huggingface.co/datasets/MinhThuw0103/lastdance-visual-embeddings --local-dir=hf_repo
cd hf_repo
# Copy audio batch (e.g., batch-01)
Copy-Item "$env:AIC_DATA\audio\L21_V001.flac" asr/audio/batch-01/
Copy-Item "$env:AIC_DATA\audio\L21_V002.flac" asr/audio/batch-01/
# ... copy all 100 video cho batch-01
git add asr/audio/batch-01/
git commit -m "Add audio batch-01 (100 video, $timestamp)"
git push
```

Mỗi batch được push riêng (không push 873 file cùng lúc) để tránh timeout. Revision tên
batch (`audio-batch-01`, `audio-batch-02`) để roll-back + audit dễ.

**Verify:** `huggingface-hub` CLI:

```powershell
huggingface-cli repo ls-files MinhThuw0103/lastdance-visual-embeddings --repo-type=dataset | grep asr/audio/batch-01
```

---

## 4. Kaggle Notebook Production

### 4.1 Build Notebook

```powershell
python -m scripts.build_kaggle_asr_production_notebook `
  --worker-count 4 `
  --output-notebook "notebooks\kaggle_asr_production.ipynb"
```

Output: `.ipynb` chứa embedded runtime (`scripts/kaggle_asr_production_runtime.py` written to
`kaggle_asr_runtime.py` on disk via `%%writefile`), WORKER_SLOT configuration, dual-GPU
subprocess launcher, push-to-HF script.

### 4.2 Run on Kaggle

Upload `kaggle_asr_production.ipynb` vào tài khoản Kaggle (worker 1, worker 2, v.v.).
**Không cần "Add Input" dataset** — Cell 1 tự tải audio + `frames.csv` từ HF Dataset
`Vu165/lastdance-asr` qua `snapshot_download()`. Yêu cầu notebook account có **Internet
enabled** (Kaggle Phone-verified hoặc Pro; tài khoản chưa verify sẽ báo lỗi
`kernelSessions.enableInternet`).

Notebook config (mỗi account):

1. **Accelerator**: chọn **GPU T4 x2** (khuyên dùng — chạy song song 2 batch/2 GPU); nếu
   chỉ có GPU T4 x1, sửa `GPU_COUNT = 1` ở Cell 2.
2. **Internet**: bật ON (bắt buộc để `snapshot_download` từ HF).
3. **Secrets**: thêm `HF_TOKEN` tại https://www.kaggle.com/settings/secrets (dùng để push
   archive JSONL/manifest lên HF sau khi xong).
4. Sửa Cell 2 (Parameters), chỉ đổi `WORKER_SLOT`:
   ```python
   WORKER_SLOT = 1  # hoặc 2, 3, 4 tuỳ account
   WORKER_BATCHES = {
       1: ("batch-01", "batch-09"),
       2: ("batch-02", "batch-03", "batch-04"),
       3: ("batch-05", "batch-08"),
       4: ("batch-06", "batch-07"),
   }
   GPU_COUNT = 2  # số GPU vật lý khả dụng (T4 x2 = 2, T4 x1 = 1)
   ```
5. Run **Cell 1** (setup: pip install + `snapshot_download` audio/catalog từ HF) → chờ
   `Audio ready at /kaggle/working/hf-asr`. Snapshot và HF cache phải nằm trong
   `/kaggle/working` vì `/kaggle/input` là read-only.
6. Run **Cell 2** (parameters) → **Cell 3** (`%%writefile kaggle_asr_runtime.py`, ghi
   runtime ra file, không exec inline) → **Cell 4** (invocation).
7. **Cell 4** chia đều batch của worker theo `GPU_COUNT` (round-robin) và launch một
   subprocess `python kaggle_asr_runtime.py` riêng cho từng GPU (`CUDA_VISIBLE_DEVICES`
   pin theo subprocess), chạy song song. Mỗi subprocess tự transcribe, segment align,
   ghi AsrRecordEnvelope JSONL, checkpoint, rồi push `asr/archives/batch-XX/` lên HF khi
   batch hoàn tất (`processed_videos == expected_videos` và không có `error` row).

> Lý do dùng `subprocess.Popen` thay vì `multiprocessing`: các hàm được định nghĩa trực
> tiếp trong notebook cell không pickling/import lại được bởi child process khi dùng
> `multiprocessing` start method `spawn` trong Jupyter kernel. Ghi runtime ra file riêng rồi
> `subprocess.Popen` mỗi GPU một tiến trình Python độc lập tránh hoàn toàn vấn đề này.

**Runtime:**
- faster-whisper large-v3 (~2–3GB VRAM/process, fp16, batched inference)
- ~60–80 video/hr/GPU trên Kaggle T4 (tùy clip length)
- Với GPU_COUNT=2: 2 batch chạy song song → tổng thời gian ~ giảm gần một nửa so với
  chạy tuần tự trên 1 GPU (ví dụ worker 2 có 3 batch: 2 batch chạy song song GPU 0/1,
  batch thứ 3 chạy tiếp trên GPU vừa xong trước — vẫn theo thứ tự round-robin cố định
  ở đầu Cell 4, không dynamic load balancing).

**Checkpoint & resume qua Kaggle timeout / tắt máy giữa chừng:**

- Cứ sau mỗi `CHECKPOINT_EVERY` video (mặc định 10), mỗi subprocess **push JSONL +
  `batch-checkpoint.json` dở dang** lên HF tại `asr/checkpoints/batch-XX/` (path riêng,
  khác với `asr/archives/batch-XX/` — path archive chỉ được ghi khi cả batch hoàn tất qua
  completion gate).
- Nếu Kaggle timeout (giới hạn phiên GPU) hoặc máy tắt **giữa batch**: `/kaggle/working` bị
  xoá, nhưng tiến trình đã có trên HF `asr/checkpoints/`. Mở lại notebook, chạy lại Cell 1
  → 4 y hệt — mỗi subprocess tự động **tải checkpoint từ HF về trước khi transcribe**, skip
  các video đã xong, chỉ tiếp tục phần còn thiếu.
- Khi batch hoàn tất, checkpoint tạm trên `asr/checkpoints/` được **tự xoá** sau khi
  `asr/archives/batch-XX/` publish thành công (dọn dẹp best-effort, không chặn run nếu lỗi).
- Push checkpoint/restore là best-effort (bọc try/except) — lỗi mạng khi push/tải checkpoint
  không làm crash tiến trình transcribe, chỉ log cảnh báo rồi tiếp tục.

---

## 5. Download & Build Snapshot (Local)

### 5.1 Pull All ASR Artifacts from HF

```powershell
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='MinhThuw0103/lastdance-visual-embeddings',
    repo_type='dataset',
    allow_patterns='asr/*',
    local_dir='$env:AIC_DATA/hf-cache',
    token=<HF_TOKEN>,
    force_download=False
)
"
```

Output: `AIC_DATA/hf-cache/asr/audio/`, `asr/archives/batch-XX/`, `asr/snapshots/`
(nếu có snapshot đã tồn tại từ run trước).

### 5.2 Build Immutable Snapshot

```powershell
python -m scripts.build_asr_snapshot `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --source-jsonl "$env:AIC_DATA\hf-cache\asr\archives\batch-01\asr-envelope.jsonl" `
  --source-jsonl "$env:AIC_DATA\hf-cache\asr\archives\batch-02\asr-envelope.jsonl" `
  --source-jsonl "$env:AIC_DATA\hf-cache\asr\archives\batch-03\asr-envelope.jsonl" `
  ... `
  --source-format "asr_envelope_v1" `
  --output-root "$env:AIC_DATA\asr\snapshots"
```

Hoặc script automation qua glob:

```powershell
python -c "
import json
from pathlib import Path
jsonl_files = sorted(Path('$env:AIC_DATA/hf-cache/asr/archives').glob('batch-*/asr-envelope.jsonl'))
args = ' '.join([f'--source-jsonl {f}' for f in jsonl_files])
# ... build_asr_snapshot với args đó
"
```

**Output:**
```text
AIC_DATA/asr/snapshots/asr-snapshot-20260904T001000Z-a1b2c3d4e5f6/
  ├── asr.sqlite (FTS5, 7 cột: video_id, segment_id, transcribed_text, language,
  │                       keyframe_uid_nearest, start_time, end_time)
  ├── coverage.json (per-video status, Publishing Criteria audit)
  └── SHA256SUMS (integrity bind)
```

**Validation fail-closed:**
- Mỗi video trong `frames.csv` phải có đúng một row
- Mỗi segment `keyframe_uid_nearest` phải tồn tại trong `frames.csv`
- No NaN/Inf, language in [vi, en, mixed]
- FTS5 build thành công
- SHA256 verify atomic

---

## 6. Publishing Criteria (§2A.4)

ASR video `complete=true` khi **thỏa đủ**:

1. ✅ Audio thật (`audio_duration_seconds > 0`)
2. ✅ ≥1 segment hợp lệ (`list[AsrSegment]` không rỗng hoặc status `silent` được log)
3. ✅ Mỗi segment `keyframe_uid_nearest` tồn tại trong `frames.csv`
4. ✅ `keyframe_uid_nearest` valid (signed int64 > 0)
5. ✅ `language` in [vi, en, mixed]
6. ✅ FTS5 query thử `transcribed_text` token ≥2 chars được điểm (bm25 không null)
7. ✅ No error status (`error_status == null` hoặc status code không `RETRYABLE_ERROR`)

**Degraded mode (ALLOWED if documented):**
- Snapshot `complete=false` nhưng `production_ready=false` có thể publish nếu coverage
  > 90% + error < 5%, với explicit `--allow-partial` flag + warning trong coverage.json.
- Xem mục mở #3 của `docs/CURRENT_STATUS.md`.

---

## 7. Publish to Online Index

```powershell
python -m scripts.publish_asr_index `
  --snapshot-dir "$env:AIC_DATA\asr\snapshots\asr-snapshot-20260904T001000Z-a1b2c3d4e5f6" `
  --output-report "$env:AIC_DATA\asr\publish-report.json"
```

**Hành động:**
1. Validate snapshot local (SHA256SUMS, coverage.json, asr.sqlite columns)
2. Kiểm Publishing Criteria per-video (nếu strict mode) hoặc skip (degraded)
3. Atomic copy `asr.sqlite` + `coverage.json` vào `AIC_DATA/index/`
4. Online đã mount `AIC_DATA/index/` → sử dụng ngay lập tức

**Output report:** `publish-report.json` ghi action (publish|already_present|blocked),
timestamp, schema version, coverage summary.

---

## 8. Checkpoint & Resume

### 8.1 Local Audio Extraction

Checkpoint lưu tại: `AIC_DATA/audio/collection.checkpoint.json`

```json
{
  "schema_version": 1,
  "videos": {
    "L21_V001": {
      "stages": {
        "extract_audio": {
          "signature": "sha256_of_ffmpeg_config",
          "next_index": 42,  // video 42 đã xong trong list inventory
          "total": 873
        }
      }
    }
  }
}
```

Chạy lại lệnh extraction = auto-skip đã xong, tiếp tục từ index 42.

### 8.2 Kaggle Batch Checkpoint

Notebook save sau mỗi 5 video (configurable `CHECKPOINT_EVERY`):
`asr/archives/batch-XX/batch-checkpoint.json` + push lên HF.

Restart notebook (nếu timeout) = resume từ video kế tiếp trong checkpoint.

---

## 9. Troubleshooting

### 9.1 Audio Extraction Fails

- **ffprobe không tìm:** cài `ffmpeg` hệ thống (Windows: `choco install ffmpeg`)
- **AIC_DATA not found:** export `$env:AIC_DATA = "D:\path"`
- **Staging write error:** kiểm `AIC_DATA/audio/` có write permission

### 9.2 Kaggle Notebook Timeout

- Notebook không save checkpoint → retry mất tiến độ. Để `CHECKPOINT_EVERY` nhỏ (5).
- Interrupt mid-batch → checkpoint.json lưu state video hiện tại, restart = resume tự động.
- Quota GPU hết → chạy lại sau 24h hoặc dùng account khác (worker 2, 3, 4).

### 9.3 ASR Snapshot Build Fails

- **Missing JSONL từ batch:** kiểm `asr/archives/batch-XX/asr-envelope.jsonl` trong HF.
- **UID mismatch:** snapshot audit so sánh `frames.csv` UID set → nếu không khớp fail
  closed.
- **FTS5 error:** recheck SQL syntax `offline/asr_snapshot.py` line có tạo bảng.

### 9.4 Degraded Mode (Coverage < 100%)

- Snapshot `complete=false` nhưng coverage tốt → chạy:
  ```powershell
  python -m scripts.publish_asr_index `
    --snapshot-dir ... `
    --allow-partial
  ```
- Online sẽ cảnh báo "ASR incomplete" khi query, tuy nhiên vẫn search được text available.

---

## 10. Checklist Before Production Run

- [ ] `offline/asr_audio.py`, `scripts/extract_asr_audio.py` test pass (`pytest tests/test_asr_audio.py -q`)
- [ ] Audio batch extracted + pushed đầy đủ (9 batch × batch-size video, ~873 total)
- [ ] `configs/asr_whisper_model.json` pinned with real model hash
- [ ] Notebook `kaggle_asr_production.ipynb` built + tested on Kaggle with sample batch
- [ ] Worker assignment plan disjoint + exhaustive (lệnh `validate_asr_worker_plan.py`)
- [ ] Notebook run 1 batch thành công → checkpoint save → pull snapshot từ HF
- [ ] `build_asr_snapshot.py` build snapshot local + coverage.json valid
- [ ] Publishing Criteria validation PASS (hoặc `--allow-partial` + warning nếu cover <100%)
- [ ] `publish_asr_index.py` publish thành công → `AIC_DATA/index/asr.sqlite` tồn tại
- [ ] Online restart + `/health` endpoint trả 200 (mục `asr` status not null)
- [ ] Full pytest -q pass (242 test + 9 test_asr_* không phá gì)

---

## 11. References

- `docs/BASELINE_SPEC.md` §2A — ASR contract, UID alignment, Publishing Criteria
- `offline/README.md` — Nhánh 1 (keyframe/embedding/OCR) + Nhánh 3 ASR summary
- `offline/asr_*.py` — Modular schema, alignment, worker plan, snapshot
- `scripts/extract_asr_audio.py` — CLI extraction (local CPU)
- `scripts/kaggle_asr_production_runtime.py` — Kaggle runtime (GPU)
- `online/fts.py` — FTS5 contract (asr_fts table schema, nếu Online cần modify)
- `online/config.py` — `OnlineLayout.asr` path (đã mount `AIC_DATA/index/asr.sqlite`)
