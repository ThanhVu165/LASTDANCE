# Trạng thái hiện tại của LASTDANCE

Cập nhật: 25/08/2026.

## Quyết định kiến trúc

Repo đã pivot hoàn toàn từ Qwen video-window sang baseline frame-level. Nguồn chuẩn kỹ
thuật duy nhất là `BASELINE_SPEC.md`; contract Offline và ASR đã được hợp nhất vào đây.

`AGENTS.md` đã được thay bằng context mới. Các tài liệu window-first cũ chỉ còn là lịch sử.
`backend/app` chưa bị sửa hoặc chuyển thư mục trong Nhánh 1; implementation này không được
wire vào dispatch logic mới.

## Đã triển khai trong Nhánh 1

- Tạo package `offline/`, `shared/schemas/`, `scripts/` và test độc lập.
- Khóa `FrameRecord`, `OcrResult`, `AsrSegment` theo baseline mới.
- `make_keyframe_uid()` hash đúng chuỗi canonical theo công thức BLAKE2b trong spec, ép về
  signed-int64 dương và từ chối ID chứa whitespace thay vì âm thầm chuẩn hóa.
- Inventory đọc FPS, resolution, duration, frame count và audio stream thật bằng `ffprobe`.
- Artifact inventory chỉ lưu path tương đối dưới `AIC_DATA` và publish bằng atomic replace.
- Có interface `ShotDetector`; TransNetV2 đã được chốt làm detector production, lazy-load
  và không tự tải model/weight. Windows NVIDIA GPU là worker production sau parity 5/5; CPU
  giữ làm reference/fallback.
- Shot manifest schema v2 validate boundary tăng dần, không overlap, ghi rõ mọi
  `excluded_transition_ranges` và cảnh báo (không fail) nếu tỷ lệ frame transition bị loại
  vượt 1%; lỗi coverage accounting vẫn fail closed.
- Shot adapter hỗ trợ device `cpu`/`cuda` tường minh, CPU là default và CUDA fail closed nếu
  không khả dụng. Batch runner production chọn `--device cuda`, dùng chung một model, ghi
  runtime/provenance/peak CUDA và atomic-publish từng manifest; parity checker so exact
  boundary/range CPU–CUDA.
- Batch runner có checkpoint riêng theo worker/device/output namespace: ghi `0/1` trước
  inference và chỉ ghi `1/1` sau atomic publish + validation. Resume xử lý được cả ngắt
  giữa inference lẫn crash-window sau publish; complete state thiếu manifest sẽ fail closed.
- Visual embedding builder chạy độc lập từng modality, khóa model revision + catalog/UID +
  batch size trong signature, ghi shard atomic với UID `int64`, vector L2-normalized
  `float16` và chỉ publish final manifest sau khi scan/hash/health check đủ shard.
- CLIP/SigLIP có adapter Kaggle CUDA và revision Hugging Face bất biến đã verify; BEiT-3
  fail-closed chờ official Microsoft UniLM retrieval checkpoint, không thay bằng BEiT
  thường. Intentional stop/resume phải được chứng minh bằng hai process thật.
- FAISS builder chạy CPU local bằng `IndexIDMap(IndexFlatIP)`, add batch video rời nhau theo
  `keyframe_uid`, không chờ modality khác. Sidecar SHA-bound và validator diff ID/vector thật
  với `frames.csv`; source overlap hoặc khác model/catalog/dimension bị từ chối.
- Keyframe plan chọn Begin/Middle/End theo timestamp frame thật từ `ffprobe`, dedup shot
  ngắn và sinh `FrameRecord` với path tương đối dưới `AIC_DATA`. Builder load manifest v2
  qua validator, truyền `excluded_transition_ranges` tường minh cho planner và planner
  fail-closed nếu một representative frame rơi vào range transition.
- Keyframe plan loader yêu cầu `frame_id` unique và tăng nghiêm ngặt. Extractor giữ nguyên
  thứ tự canonical này, không tự sort theo `frame_id` hoặc `local_idx`, vì checkpoint
  `next_index` tham chiếu trực tiếp vào thứ tự item trong plan.
- Exact-frame extraction dùng một lượt FFmpeg decode/video với `select` theo decoded frame
  index, stage/kiểm đủ batch rồi atomic-publish từng JPEG; checkpoint vẫn cập nhật theo nhóm
  và `--limit` không đổi full-plan signature.
- Quality stage đo Laplacian variance + 64-bit pHash, chỉ dedup trong cùng shot, nhận ngưỡng
  qua CLI (không hardcode production threshold), luôn giữ ít nhất một frame/shot và không
  xóa JPEG nguồn. Manifest selection có plan/config signature và ghi atomic.
- Catalog builder yêu cầu quality manifest khớp 100% UID/metadata/SHA của plan, cấm trùng
  `keyframe_uid`, `(video_id, local_idx)` và `(video_id, frame_id)`, rồi ghi `frames.csv`
  atomic với sidecar state khóa bằng SHA-256 và validator fail-closed.
- Collection mode của catalog lấy tập ID canonical từ `inventory.json`, tự ghép đúng
  `keyframe-plans/<video_id>.json` với `keyframe-quality/<video_id>.json` cho toàn bộ
  collection và từ chối publish nếu thiếu/thừa dù chỉ một video. Manual
  `--plan/--quality` vẫn được giữ cho smoke/dev subset.
- Checkpoint theo `video_id` + stage + signature, không cho resume chéo signature hoặc lùi
  progress.
- Publishing readiness được suy ra fail-closed từ tập `keyframe_uid` của cả CLIP/SigLIP/
  BEiT-3, vector health, mapping sanity và checkpoint/resume verification. Không có API set
  `complete=true` thủ công.

## Kiểm thử

- Compile `offline shared scripts tests`: pass.
- Test Nhánh 1: 87/87 pass bằng Python 3.11.9 trên nền hợp nhất
  `origin/codex/offline-shot-detection@02d6d3e` ngày 25/08/2026. Đây gồm test batch
  Shot/Keyframe/Catalog mới và 15 test Visual Embedding/FAISS.
- Compile legacy `backend/app`: pass.
- Không chạy được toàn bộ backend test trong môi trường hiện tại vì thiếu `fastapi` và
  `opencv-python`; đây là dependency-environment blocker, không phải regression đã quan sát.

## Gate production Shot Detection Windows GPU

**ACCEPTED — parity CPU–GPU exact 5/5 PASS ngày 25/08/2026.** Tập 873 manifest CUDA trong
`AIC_DATA/shots` là nguồn shot production được phép dùng cho keyframe downstream.

| Điều kiện | Trạng thái đã xác minh |
|---|---|
| Driver/GPU và `check_nvidia_windows.ps1` PASS | PASS — RTX 4050 6141 MiB, driver 581.15 |
| `.venv-shot-gpu`, CUDA và bundled weight hợp lệ | PASS — batch CUDA hoàn tất |
| `L21_V001` exact manifest compare CPU–GPU | PASS |
| `L21_V002` exact manifest compare CPU–GPU | PASS |
| `L21_V003` exact manifest compare CPU–GPU | PASS |
| `L21_V005` exact manifest compare CPU–GPU | PASS |
| `L21_V006` exact manifest compare CPU–GPU | PASS |

Exact compare bao gồm từng shot boundary và mọi `excluded_transition_ranges`; không được
chỉ so shot count. Lệch một transition frame có thể làm keyframe plan/mapping
`keyframe_uid → frame_id` khác nhau và phá join `frames.csv` về sau.

Artifact CUDA hiện có trong `AIC_DATA/shots`: 873/873 manifest hợp lệ, batch report không
có failure, checkpoint đủ 873/873; tổng 97.810 shot và 3.658 excluded transition ranges.
CPU reference được giữ riêng tại `AIC_DATA/index/shot-parity-cpu` để audit.

## Điều phối worker Shot Detection

| Worker | Người phụ trách | File phân công | Phạm vi video ID | Trạng thái |
|---|---|---|---|---|
| Windows NVIDIA GPU local | Người dùng | `worker-01.txt` | 873/873 | ACCEPTED — parity 5/5 PASS |
| Colab CUDA | CHƯA PHÂN CÔNG | `worker-colab.txt` | CHƯA CHỐT | DISABLED |

Không cần chạy lại batch shot 873. Colab vẫn disabled; nếu bật về sau, hai batch report có
cùng `video_id` phải bị xem là xung đột và không merge tự động.

### Handoff triển khai 24/08/2026

- Working tree đang có thay đổi chưa commit cho pivot frame-level; không có thay đổi nào
  trong `backend/app`.
- Bốn source-of-truth (`AGENTS.md` và ba spec trong `docs/`) đã được kiểm tra SHA-256 khớp
  chính xác với file nguồn do người dùng cung cấp.
- File/package mới: `offline/`, `shared/schemas/`, `scripts/`, `tests/` và
  `offline/README.md`.
- CLI đã smoke: `build_inventory`, `detect_shots`, `build_keyframe_plan`,
  `extract_keyframes`, `filter_keyframes` và `build_frames_catalog`.
- Lệnh kiểm tra cuối đã chạy:

  ```powershell
  python -m compileall -q offline shared scripts tests
  python -m unittest discover -s tests -q
  git diff --check
  ```

- Kết quả gần nhất: compile pass, 64/64 test Nhánh 1 pass và `git diff --check` sạch.
- Chưa có artifact production mới: chưa sinh `inventory.json`, shot manifest,
  `frames.csv`, FAISS hoặc SQLite từ dữ liệu thật; do đó mọi video vẫn phải được xem là
  `complete=false`.
- Chưa đo throughput, peak VRAM, disk size hoặc ETA vì chưa chạy model/dataset thật.
- Checkout hiện tại không có `.venv-shot-gpu` hoặc `.venv-offline`. System Python 3.11.9
  đủ để compile và chạy 64/64 unit test, nhưng không phải environment production.
- Doctor `shot-windows-gpu` trên system Python FAIL đúng vì Torch 2.5.1 CPU, thiếu
  `transnetv2-pytorch`, `ffmpeg-python`, `ImageHash` và CUDA không khả dụng trong Torch.
- Lock/bootstrap Windows GPU đã có trong repo; cần tạo `.venv-shot-gpu` trên máy đích rồi
  chạy lại doctor trước parity.
- Weight contract vẫn khóa SHA-256
  `a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de` và sẽ được doctor
  kiểm tra sau khi cài package.
- Smoke thật trên video tổng hợp 100 frame đỏ→xanh tách đúng 2 shot `[0..49]` và `[50..99]`;
  manifest có package version, CPU, threshold, weight source và hash.
- Smoke preprocessing E2E trên video đó: plan/extract 6/6 Begin/Middle/End, report quality
  giữ 6/6, pHash distance 0 giữ đúng 2/6 (một/shot), catalog có đúng 2 record ở frame 0 và
  50; CSV/state hash validator pass. Tất cả chỉ nằm trong `tmp/`, không phải production.

## Dev-subset thật 5 video — 24/08/2026

- `AIC_DATA=D:\LT\AIC2026`; doctor offline-local pass, không dùng GPU/Kaggle.
- Inventory: 5 video, 158.873 frame, tổng khoảng 91,6 phút, 1280×720 và đều có audio.
- TransNetV2 CPU: 1.388 shot. Manifest schema v2 được regenerate từ raw prediction mới;
  chỉ 4 frame transition bị loại (`L21_V002:49`, `L21_V005:57`,
  `L21_V006:0,1063`), tương đương khoảng 0,0025%; không video nào vượt warning 1%.
- Keyframe plan/extraction: 4.164/4.164 Begin/Middle/End JPEG, không zero-byte, checkpoint
  đủ 5/5 video, tổng dung lượng khoảng 397 MB; staging cleanup sạch. Batch decode mới hoàn
  tất toàn subset trong khoảng 2,5 phút quan sát, thay vì mở FFmpeg 4.164 lần.
- Quality report-only: Laplacian min/p05/median lần lượt khoảng 75,56/180,16/439,83. Visual
  audit cho thấy các frame dưới 100 vẫn chứa evidence hữu ích, nên chưa dùng blur threshold.
  pHash distance 0 sẽ loại 240 frame nhưng có thể làm mất ticker/OCR thay đổi; chưa filter
  khi chưa có ground-truth A/B, giữ 4.164/4.164.
- Dev catalog riêng tại `index/dev-subset-5/frames.csv`: 4.164 record/5 video, SHA sidecar
  validator pass; UID, `(video_id, local_idx)` và `(video_id, frame_id)` đều unique 100%.
- Đây chỉ là dev catalog complete trong namespace riêng, không phải production index: chưa
  có CLIP/SigLIP/BEiT-3 FAISS, OCR hoặc ASR nên collection production vẫn fail closed.

## Chưa triển khai hoặc chưa chạy thật

- Shot Detection CUDA 873 video đã được accept sau parity 5/5. Keyframe
  plan/extraction/quality report-only đang chạy thành hai shard disjoint 437 + 436 video,
  mỗi shard có checkpoint/report/extraction state riêng.
- Ngưỡng blur/pHash production chưa chốt vì visual audit chưa thay thế ground-truth A/B.
  Cosine dedup chờ embedding và không nằm trên critical path hiện tại.
- Chưa sinh CLIP/SigLIP/BEiT-3 embedding hoặc FAISS `IndexIDMap` bằng model/dữ liệu thật.
  Code builder/validator/runbook đã có và synthetic regression PASS; BEiT-3 còn blocked.
- Chưa có `ocr.sqlite` hoặc `asr.sqlite`.
- Chưa dùng Kaggle/Colab hoặc push Hugging Face Dataset. Windows CUDA local đã dùng cho shot
  batch; `.venv-shot-gpu` chỉ là environment shot detection, không thay thế environment
  local-CPU cho inventory/keyframe/quality.
- Internet trong phòng thi chưa xác nhận; Gemini không được coi là dependency bắt buộc.

## Bước tiếp theo

1. Chờ hai shard keyframe hoàn tất và xác minh đủ 873 plan/quality cùng JPEG không rỗng.
2. Chạy `scripts.build_frames_catalog --collection`; lệnh chỉ publish khi inventory,
   plan và quality khớp chính xác toàn collection.
3. Khi catalog production PASS, chạy CLIP/SigLIP Kaggle theo gate exit 75 → process mới
   resume → validator; BEiT-3 chỉ chạy sau khi chốt checkpoint chính thức.
4. Tải modality hoàn tất về local và build FAISS modality đó ngay, không đợi hai modality
   còn lại; validator diff `keyframe_uid` theo video.
5. Chốt ngưỡng blur/pHash bằng ground-truth A/B; không filter mù artifact report-only.

## Log phiên

> Cập nhật CUỐI mỗi phiên Codex, trước khi đóng session — dù account nào cũng đọc file này
> đầu tiên sau `AGENTS.md`. Không xóa lịch sử cũ, chỉ thêm mục mới lên đầu.

---

### [26/08/2026] Sửa pin dependency Kaggle Visual

- Kaggle resolver phát hiện `requirements/kaggle-gpu.txt` khóa
  `huggingface-hub==1.3.5` và `safetensors==0.6.2`, mâu thuẫn trực tiếp với metadata của
  `transformers==5.15.1` (`huggingface-hub>=1.5.0`, `safetensors>=0.8.0`).
- Nâng pin tương ứng lên `huggingface-hub==1.28.0` và `safetensors==0.8.0`, đồng bộ
  environment doctor và Visual runbook. Pip resolver dry-run trên Python 3.11 PASS; Kaggle
  clean-install/doctor/inference thật vẫn cần chạy lại trước khi coi gate môi trường PASS.
- Thay đổi hiện chỉ ở working tree; chưa commit, chưa push và không được coi là provenance
  Kaggle chính thức cho tới khi chủ repo yêu cầu rõ hai hành động đó.

---

### Đang làm dở (task hiện tại, nếu có)

- Task hiện tại: hoàn tất keyframe plan → exact extraction → quality → `frames.csv` cho
  toàn bộ 873 video.
- Parity CPU–GPU 5/5 exact PASS; 873 shot manifest đã accepted. Batch runner mới khóa
  inventory/shot exact membership, input/config/implementation signature và resume
  fail-closed; 72/72 test PASS.
- Hai shard keyframe disjoint đang chạy. Sau khi cả hai report `complete=true`, build và
  validate `scripts.build_frames_catalog --collection`.
- Branch Visual đã được dựng lại từ nền Shot/Keyframe mới `02d6d3e`; 87/87 test PASS. Code
  có thể đưa lên Kaggle sau khi push lại branch và catalog/input dev hoặc production sẵn sàng.
- **Lệnh để tự kiểm tra trạng thái thật** (không tin lời mô tả, chạy lại):
  ```powershell
  git status --short
  git diff --stat
  .\scripts\run_offline_windows.ps1 `
    -Module scripts.environment_doctor `
    -PythonArguments @("--profile", "offline-local", "--skip-data")
  .\scripts\run_offline_windows.ps1 `
    -Module unittest -PythonArguments @("discover", "-s", "tests", "-q")
  ```

---

### Quyết định mới phát sinh

- Môi trường production Nhánh 1 dùng CPython 3.11.9; dependency Windows local và Kaggle
  tách profile để Kaggle không bị thay wheel PyTorch/CUDA có sẵn.
- TransNetV2 dùng adapter tổng quát, không tự tải weight trong request path. Package
  `transnetv2-pytorch==1.0.5` đã được chốt làm runtime shot detection production; GPU chạy
  batch sau parity 5/5, CPU làm reference/fallback.
- Package trên bundle weight; default kiểm tra SHA-256
  `a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de`. External
  weight chỉ là override và bắt buộc checksum. Shot manifest lưu đầy đủ provenance.
- `--limit` của keyframe extraction là điểm dừng có thể resume, không được đưa vào artifact
  signature hoặc làm giảm `total` của full plan.

### [24/08/2026] Chốt TransNetV2 GPU + checkpoint shot batch

- Người dùng chốt TransNetV2 làm detector production; AutoShot được bỏ khỏi critical path.
- `run_shot_batch` ghi checkpoint `0/1` trước inference và `1/1` sau atomic publish +
  validation; signature khóa detector/device/weight, stat MP4 và output namespace.
- Test đã cover ngắt trong inference, crash sau publish trước checkpoint, adopt manifest và
  fail closed khi complete state thiếu manifest. Compile + 64/64 test PASS, diff check sạch.
- NVIDIA preflight đã PASS; chưa chạy CUDA thật vì `.venv-shot-gpu` chưa tồn tại. Production
  vẫn BLOCKED bởi GPU doctor, parity 5/5 và registry phân công worker.


---

### Việc đã xong, đã review (Claude hoặc người đã check qua)

- [24/08/2026] — `offline/`, `shared/schemas/`, `scripts/`, `tests/` — scaffold Nhánh 1,
  inventory bằng ffprobe, shot adapter, keyframe Begin/Middle/End theo timestamp thật,
  exact-frame extraction, quality/catalog, checkpoint/publishing fail-closed — 41/41 unit
  test PASS.
- [24/08/2026] — `requirements/`, `environment.yml`, `.python-version`, `.env.example`,
  `scripts/environment_doctor.py` — khóa profile và kiểm tra version/path/binary/checksum —
  clean-install và idempotent update PASS trên Python 3.11.9 + FFmpeg 7.1.1; lock Windows
  Conda-native và pip-transitive đã tạo.
- [24/08/2026] — TransNetV2 smoke thật — video tổng hợp 100 frame đỏ→xanh được tách đúng
  `[0..49]`, `[50..99]`; manifest ghi package/device/threshold/source/SHA — PASS.
- [24/08/2026] — Bộ ba spec ban đầu đã được kiểm tra chéo; về sau contract Offline/ASR
  được hợp nhất trở lại `docs/BASELINE_SPEC.md` để chỉ còn một nguồn chuẩn.

---

### Việc CHƯA làm, ưu tiên tiếp theo

1. Hoàn tất và validate hai shard keyframe 437 + 436 video.
2. Build `frames.csv --collection` và xác minh inventory/plan/quality membership 873/873.
3. Chạy CLIP/SigLIP trên Kaggle bằng artifact float16 + real checkpoint/resume gate.
4. Chốt official BEiT-3 retrieval checkpoint + checksum rồi mới bật modality.
5. Build FAISS độc lập cho modality đã xong và diff UID thật với catalog.
6. Chạy ground-truth A/B trước khi bật blur/pHash filter hoặc đổi detector.

### [25/08/2026] Phiên #11
- Theo yêu cầu người dùng, xóa remote branch Visual cũ sau khi lưu backup local
  `codex/backup-visual-pre-refresh-20260825@f8d421f`; không mất đường phục hồi.
- Fetch/pull nền mới `origin/codex/offline-shot-detection@02d6d3e`, gồm Shot Detection đã
  accepted 873/873 và code batch Keyframe/Catalog mới. Đọc lại toàn bộ baseline/offline/ASR
  spec cùng Shot runbook trước khi tích hợp.
- Tạo lại `codex/offline-visual-embeddings` từ `02d6d3e`, áp code Visual/FAISS thành commit
  `21bdbbe`. README là conflict duy nhất; đã giữ cả nguồn chuẩn mới, checkpoint Shot và hai
  runbook Visual/FAISS, không ghi đè nguyên một phía.
- Doctor `offline-local`, compile, `git diff --check` và toàn bộ 87/87 test PASS trên nền
  hợp nhất. Chưa chạy Kaggle/model thật, chưa sinh vector/index và BEiT-3 vẫn fail-closed.
- Trạng thái này được commit riêng trước khi push lại branch; remote SHA phải được đối chiếu
  với local sau push.

### [24/08/2026] Phiên #10
- Ghi production gate Windows GPU thành BLOCKED rõ ràng: 60/60 unit test không phải parity
  CUDA thật; cả 5 video hiện đều `CHƯA CHẠY` exact compare.
- Bổ sung yêu cầu reboot bắt buộc sau update driver rồi mới rerun preflight/doctor; thêm lại
  điều kiện này vào checklist treo máy.
- Thêm registry điều phối Windows/Colab. Vì chưa có tên người và phạm vi ID từ người dùng,
  Windows vẫn blocked và Colab disabled; cấm khởi chạy cho tới khi hai tập ID được ghi rõ,
  xác nhận không giao nhau.
- Nhấn mạnh exact compare phải gồm `excluded_transition_ranges`; lệch transition frame có
  thể làm mapping keyframe/UID khác và phá join `frames.csv`.

### [24/08/2026] Phiên #9
- Theo quyết định mới, chuyển worker Shot Detection của đồng đội từ CPU sang Windows NVIDIA
  GPU; Colab giữ làm lựa chọn phụ. CPU vẫn là reference và không có fallback âm thầm.
- Thêm `.venv-shot-gpu` tách biệt, profile PyTorch 2.12.1 CUDA 12.6 chính thức, preflight
  driver >=528.33 và doctor `shot-windows-gpu`. Không cài CUDA wheel vào `.venv-offline`.
- Batch runner có `--shots-dir` nằm trong `AIC_DATA` để parity GPU không overwrite reference
  CPU. Runbook bổ sung setup, parity 5/5, resume và checklist chống sleep khi treo máy.
- Verification local: 60/60 test, compile và PowerShell syntax PASS; preflight trên máy
  Codex fail đúng thiết kế vì driver 516.40. Chưa chạy CUDA thật trên máy đồng đội.

### [24/08/2026] Phiên #8
- Cho phép Shot Detection dùng Colab CUDA theo quyết định mới, vẫn giữ CPU làm default và
  fail closed nếu CUDA không khả dụng; cập nhật Changelog của cả hai spec liên quan.
- Thêm profile `shot-colab-gpu`, batch runner dùng một model, runtime/VRAM report, exact
  CPU–CUDA manifest comparator và danh sách parity 5 video cố định trong repo.
- Viết lại runbook với cell Colab và danh sách artifact cần đưa lên Drive. Code CUDA mới chỉ
  được test bằng mock/local; parity thật trên T4 chưa chạy nên production gate vẫn đóng.
- Verification local: doctor `offline-local` PASS, targeted 19/19, toàn suite 58/58,
  compile PASS và `git diff --check` sạch.

### [24/08/2026] Phiên #7
- Thêm `docs/SHOT_DETECTION_RUNBOOK.md` làm hướng dẫn vận hành chuẩn cho đồng đội: checkout,
  bootstrap, data layout, doctor, chạy một video/danh sách, reference validation, batch
  report và quy tắc bàn giao artifact.
- Liên kết runbook từ root/offline README và environment setup. Bổ sung contract trong
  `AGENTS.md`: mọi CLI/workflow/artifact mới phải cập nhật hướng dẫn sử dụng trong cùng
  thay đổi, ghi rõ phần đã/chưa được xác minh trước khi push.

### [24/08/2026] Phiên #6
- Chuẩn bị handoff trên branch `codex/offline-shot-detection`: bổ sung chuỗi lệnh E2E một
  video và contract chia shot detection nhiều máy vào `offline/README.md`.
- Ghi rõ worker phải dùng cùng commit/postprocessing, package/threshold/weight checksum;
  adapter hiện chạy CPU và chỉ được trộn manifest CUDA sau benchmark boundary/range khớp
  100% trên `L21_V001` + `L21_V006`.
- Audit trước commit không thấy dataset/JPEG/vector/checkpoint/log/credential trong danh sách
  source. Doctor offline-local, compile, 50/50 unit test và `git diff --check` đều PASS.

### [24/08/2026] Phiên #5
- Đóng contract chéo manifest v2 → keyframe planner: manifest được load/validate bằng code
  production, `excluded_transition_ranges` được truyền tường minh và planner có assertion
  fail-closed nếu representative frame chạm transition.
- Xác nhận `scripts/extract_keyframes.py` không sort item. Plan loader nay bắt buộc
  `frame_id` unique/tăng nghiêm ngặt trước extraction; không sort âm thầm để giữ đúng ý
  nghĩa checkpoint `next_index` theo canonical plan order.
- Thêm integration test manifest v2 theo hai range thật của `L21_V006` và test plan đảo thứ
  tự. Audit read-only cả 5 artifact hiện có: 5/5 plan tăng nghiêm ngặt, 0 keyframe chạm
  transition; không cần regenerate plan/JPEG. Compile và 50/50 unit test PASS.

### [24/08/2026] Phiên #4
- Chốt transition nằm ngoài shot; manifest schema v2 lưu range/reason/coverage stats và
  warning >1%. Regenerate 5 manifest từ raw prediction mới, không vá JSON từ inference cũ.
- Dev-subset thật hoàn tất tới quality + catalog: 5 video, 1.388 shot, 4.164 keyframe và
  `frames.csv` dev validator pass; giữ toàn bộ frame sau visual audit, chưa filter mù.
- Thay extractor N lần decode bằng một batch decode/video; exact-frame, atomic JPEG và
  checkpoint giữ nguyên contract. Compile và 48/48 unit test PASS.

### [24/08/2026] Phiên #3
- `make_keyframe_uid()` không còn chuẩn hóa ID bằng `.strip()`; ID rỗng hoặc chứa bất kỳ
  whitespace nào đều raise `ValueError`, còn input canonical được hash nguyên văn theo spec.
- Thêm golden-vector `8984422734592370359` và đối chiếu implementation với công thức
  BLAKE2b viết độc lập; targeted 5/5 và toàn suite 43/43 test PASS, compile PASS.
- Ghi rõ trong `AGENTS.md` rằng `transnetv2-pytorch==1.0.5` là port PyTorch bên thứ ba;
  A/B với AutoShot không được diễn giải thành so sánh hai implementation gốc trong paper.
- Dọn `CURRENT_STATUS.md` về một mục `Log phiên`; file `docs/STATUS.md` đứng riêng đã xóa.

### [24/08/2026] Phiên #2
- Pivot Nhánh 1 đã được hiện thực hóa tới exact-frame extraction; không sửa `backend/app`.
- Đã thêm requirements theo profile, Conda manifest, doctor và hướng dẫn portability.
- Verification mới nhất: Python 3.11.9 toolchain doctor PASS, compile PASS, 41/41 unit test
  PASS, `git diff --check` sạch.
- Lần bootstrap đầu đã tải đúng installer Miniforge và checksum pass nhưng silent installer
  trả code 2 vì repo path có khoảng trắng; không có environment/toolchain dở được tạo.
  Bootstrap đã đổi sang base toolchain ở `%LOCALAPPDATA%` theo khuyến nghị upstream, còn
  `.venv-offline` vẫn nằm trong repo. Clean-install và lần update idempotent sau đó đều PASS.
- Đã thêm `run_offline_windows.ps1`, profile dev/full, lock Windows và smoke model thật.
- Đã smoke E2E tới `frames.csv` trên video tổng hợp: 2 shot, 6 keyframe, pHash giữ 2,
  catalog frame 0/50 và state hash pass; artifact chỉ ở `tmp/`, không publish production.

### [23/08/2026] Phiên #1
- Đã tạo: `AGENTS.md`, khóa schema `shared/schemas/frame.py` + `ocr.py` + `asr.py`
- Quyết định: archive Qwen video-window sang `legacy/qwen_window/`, dùng TransNetV2 làm
  default shot detector (AutoShot Baidu chưa xác nhận)
- Next: scaffold `offline/shot_detection.py` với interface `ShotDetector` tổng quát
