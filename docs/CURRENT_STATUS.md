# Trạng thái hiện tại của LASTDANCE

Cập nhật: 28/08/2026.

## Visual Embedding — CLOSED/PASS ngày 27/08/2026

- CLIP, SigLIP và EVA-CLIP production đều PASS đủ 9/9 batch, 873 video và 293.336
  `keyframe_uid` tại snapshot HF cuối
  `938aefd437ab8db61fc6599d613aedcf4921d71e`.
- Local đã khôi phục đúng 293.336 JPEG từ catalog production, không có file rỗng; ba
  `IndexIDMap(IndexFlatIP)` đã build và validator độc lập PASS: CLIP dim 512, SigLIP dim
  768, EVA-CLIP dim 768. Cả ba dùng cùng UID-set SHA-256
  `5bada00bd4a93928e48af3a6cbe7189a3b465eafb00cc8f829941edee536e660`, khớp catalog
  SHA-256 `ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37`.
- Finite/L2 norm, source `float16`, FAISS `float32`, 9 source batch và
  `checkpoint_resume_verified=true` đều PASS. Sanity mapping đầu/giữa/cuối catalog đã đối
  chiếu ảnh, FPS, `frame_id` và `pts_time`. Các mô tả Visual “chưa chạy/chưa build” ở log
  lịch sử phía dưới đã được mục closure này thay thế.

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
- CLIP/SigLIP có adapter Kaggle CUDA, immutable Hugging Face revision và dev-subset-5 đã
  PASS intentional stop → process mới resume → validate trên Tesla T4. EVA-CLIP là modality
  thứ ba mới: official HF revision + safetensors đã pin, adapter fail-closed không load
  pickle; chưa được production cho tới khi qua cùng dev gate. BEiT-3 đã bị loại vĩnh viễn.
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
  EVA-CLIP, vector health, mapping sanity và checkpoint/resume verification. Không có API set
  `complete=true` thủ công.

## Kiểm thử

- Compile `offline shared scripts tests`: pass.
- Test Nhánh 1 local: 87/87 pass bằng Python 3.11.9 trên nền hợp nhất
  `origin/codex/offline-shot-detection@02d6d3e` ngày 25/08/2026. Đây gồm test batch
  Shot/Keyframe/Catalog mới và 15 test Visual Embedding/FAISS.
- Kaggle Visual: 87 test PASS (`skipped=6` theo platform) bằng Python 3.12.13 ngày
  26/08/2026; doctor chỉ còn fail contract Python cũ trước khi được người dùng duyệt cập
  nhật. Torch `2.10.0+cu128`, CUDA 12.8, Tesla T4 và revision CLIP/SigLIP đều PASS.
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
  có đủ CLIP/SigLIP/EVA-CLIP FAISS, OCR hoặc ASR nên collection production vẫn fail closed.

## Chưa triển khai hoặc chưa chạy thật

- Shot Detection CUDA 873 video đã được accept sau parity 5/5. Keyframe
  plan/extraction/quality report-only đang chạy thành hai shard disjoint 437 + 436 video,
  mỗi shard có checkpoint/report/extraction state riêng.
- Ngưỡng blur/pHash production chưa chốt vì visual audit chưa thay thế ground-truth A/B.
  Cosine dedup chờ embedding và không nằm trên critical path hiện tại.
- Dev-subset CLIP/SigLIP embedding thật đã PASS và archive handoff đã verify. Production
  CLIP đang chạy theo 9 batch; production SigLIP chuẩn bị chạy độc lập. EVA-CLIP chưa qua
  dev-subset CUDA gate; chưa có đủ ba FAISS `IndexIDMap` production ở máy local.
- Chưa có `ocr.sqlite` hoặc `asr.sqlite`.
- Chưa dùng Kaggle/Colab hoặc push Hugging Face Dataset. Windows CUDA local đã dùng cho shot
  batch; `.venv-shot-gpu` chỉ là environment shot detection, không thay thế environment
  local-CPU cho inventory/keyframe/quality.
- Internet trong phòng thi chưa xác nhận; Gemini không được coi là dependency bắt buộc.

## Bước tiếp theo

1. Chờ hai shard keyframe hoàn tất và xác minh đủ 873 plan/quality cùng JPEG không rỗng.
2. Chạy `scripts.build_frames_catalog --collection`; lệnh chỉ publish khi inventory,
   plan và quality khớp chính xác toàn collection.
3. Tiếp tục production CLIP/SigLIP độc lập; chạy EVA-CLIP dev-subset theo gate exit 75 →
   process mới resume → validator. Chỉ sau EVA dev PASS mới tạo production 9-batch của nó.
4. Tải modality hoàn tất về local và build FAISS modality đó ngay, không đợi hai modality
   còn lại; validator diff `keyframe_uid` theo video.
5. Chốt ngưỡng blur/pHash bằng ground-truth A/B; không filter mù artifact report-only.

## Log phiên

> Cập nhật CUỐI mỗi phiên Codex, trước khi đóng session — dù account nào cũng đọc file này
> đầu tiên sau `AGENTS.md`. Không xóa lịch sử cũ, chỉ thêm mục mới lên đầu.

---

### [28/08/2026] OCR Gate A — 300-frame CRAFT threshold pilot sẵn sàng chạy

- Thêm policy pre-register `configs/ocr_craft_gate_a_policy.json`: đúng 300 frame/
  5 video/60 unique shot mỗi video, ba config `0,6/0,3/0,3`, `0,7/0,4/0,4`,
  `0,8/0,5/0,5`; bắt buộc region recall `>=0,98` và text-frame recall `>=0,99`.
- Thêm `scripts/build_ocr_gate_a_mini_dataset.py` và chuyển notebook Kaggle
  `scripts/kaggle_ocr_craft_gate_a_threshold_pilot.ipynb` sang Dataset mini 300 ảnh đã pin
  ZIP/manifest SHA-256. Notebook verify UID + checksum từng ảnh, CRAFT checksum, chạy 900
  detector pass có resume và xuất review bundle 300 ảnh 2×2; không còn quét Dataset AIC đầy
  đủ. Không chạy recognition, Vintern, Gemini hay tự PASS khi CSV nhãn còn trống.
- Thêm evaluator CPU `scripts.evaluate_ocr_craft_gate_a`: validate exact sample/config/
  threshold/human labels, đo recall/false-positive/regions-per-frame và chỉ mở Gate B nếu
  recall gate đạt. Bundle cũ 250 crop không thay thế Gate A vì thiếu no-text control và nhãn
  human hoàn chỉnh. Notebook mới compile tĩnh PASS; chưa chạy Kaggle nên Gate A vẫn pending.
- Không commit/push Git; chưa mở production 9 batch và chưa dùng Gemini.

### [28/08/2026] OCR Gate A — emergency contract 100 nhãn đã đánh giá

- Người dùng quyết định giảm human review còn 100 do chỉ có một annotator và deadline.
  Exact UID-set là 100 dòng đầu bundle: 60 V001 + 40 V002; tất cả đều được gán có chữ,
  vì vậy no-text false-positive rate không đo được và evidence không cân bằng năm video.
- Kết quả: `recall_current=92,56%`, `balanced=89,73%`, `strict=85,57%` region recall;
  text-frame recall cả ba là 100%. Không config nào đạt region recall 98%.
- Policy schema v2 không hạ quality gate và không gọi đây là PASS. Decision là
  `DEADLINE_OVERRIDE_KEEP_CURRENT`, giữ CRAFT `0,6/0,3/0,3` và cho phép chạy full dev5 Gate B
  để lấy evidence rộng hơn. Report/UID-set/policy hash và limitation đều fail-closed.

- Gate B tại `scripts/kaggle_ocr_gate_b_dev5_calibrated.ipynb` đọc
  report/hash/selected threshold thay vì hardcode và vẫn rerun full dev5. Theo deadline,
  review Vintern giảm xuống đúng 100 candidate/100 distinct frame, 20/video, stratify
  deterministic; đây là `emergency_single_annotator_100`, không phải standard 300-frame
  PASS. Checkpoint refresh mỗi 250 item, tự restore qua session mới và dựng lại crop từ bbox;
  nếu Vintern đã hoàn tất thì bỏ qua tải model. Runtime Kaggle đã PASS: đủ 4.164 frame,
  41.212 region, 14.883/14.883 Vintern result, không runtime error; EasyOCR 24,0 phút và
  Vintern 73,9 phút trên một T4.

---

### [28/08/2026] OCR Gate B — chốt Vintern calibrated override; HF snapshot đã verify

- Theo quyết định người dùng, Vintern vẫn chỉ chạy sau EasyOCR trên candidate router v2,
  không chạy song song mọi region. Thêm calibration CPU dùng đúng result từ inference Gate B:
  exact-match ground-truth theo bucket output length/guard margin/log-prob (nếu có), support
  tối thiểu và backoff deterministic; không dùng self-confidence, không chạy model lần hai.
- Rule materialize: guard PASS và empirical Vintern confidence phải lớn hơn nghiêm ngặt
  EasyOCR confidence của cùng region. JSONL audit giữ bucket/support/correct, text/confidence
  cũ-mới, quyết định và final engine; SQLite vẫn đúng năm cột. Snapshot kế tiếp sẽ mang tier
  `easyocr_vintern_calibrated`/“EasyOCR+Vintern calibrated”. Ground-truth deadline tier đã
  materialize từ 98/100 frame; hai crop sample 15/24 bị loại là unreadable. Đây vẫn là
  calibration-set evidence, chưa phải holdout accuracy PASS.
- Thêm `scripts.calibrate_ocr_vintern_gate_b`, policy
  `configs/ocr_vintern_calibration_policy.json`, adapter snapshot calibrated và contract test.
- Policy calibration đã nâng schema v2 cho emergency 100: bucket fine/structural thiếu
  support 20 không được lùi về global để override. Vùng không đủ bằng chứng giữ EasyOCR và
  được ghi rõ là Gemini residual; chưa gọi Gemini.
- Excel đã làm tròn 272 ô immutable trong CSV người dùng; script
  `scripts.repair_ocr_gate_b_ground_truth` phục hồi toàn bộ từ review ZIP, chỉ giữ năm cột
  human-label. Policy deadline schema v3 cho phép đúng 2 exclusion và dùng 98 labeled frame.
  Calibration CPU PASS, không gọi model: EasyOCR exact-match `10/98=10,2%`, Vintern
  `34/98=34,7%`; Vintern-only đúng 31, EasyOCR-only đúng 7, cả hai sai 57. Materialization
  tạm thời override 8.997/14.883 candidate và còn 5.886 residual trên 2.886 frame/1.271
  shot dev5. Chưa mở production vì residual ngoại suy còn khoảng 414.644 region và evidence
  98 mẫu không phải holdout.
- Snapshot EasyOCR-only hiện tại đã upload thành công lên private Dataset
  `MinhThuw0103/lastdance-visual-embeddings` dưới
  `ocr/snapshots/ocr-snapshot-20260827T195734Z-85dd095d6ba9/` bằng đúng một commit
  `15f2f3bed29a9f89683b01ba24b30578849b20bd`. Uploader đã
  `snapshot_download(revision=<commit>)` lại đúng ba file và verify SHA hai chiều:
  `ocr.sqlite=b442d2d0...03f0f7`, `coverage.json=54327f74...0453fb`,
  `SHA256SUMS=8cd49270...a27ed`. Handoff phải pin commit này, không dùng `main/latest`.
  Publish report nằm ngoài repo cạnh snapshot với hậu tố `.hf-publish-report.json`.

### [28/08/2026] OCR pre-Gemini — audit logo/overlay tĩnh trên residual dev5

- Audit CPU-only đọc artifact calibrated Gate B, không gọi model/Gemini và không sửa router
  production. Policy `configs/ocr_low_information_overlay_audit_policy.json` chỉ xét overlay
  ngắn lặp ổn định ở góc trên-phải; ticker/subtitle đáy màn hình được bảo vệ. Text/bbox gốc
  luôn giữ nguyên, row chỉ được gắn cờ suppression candidate.
- Trên dev5, residual giảm mô phỏng từ 5.886 region / 2.886 frame / 1.271 shot-request xuống
  4.079 region / 1.957 frame / 1.023 shot-request; 1.807 region (30,70%) thuộc 31 nhóm
  overlay góc trên-phải. Trong 98 nhãn calibration có 56 row thuộc residual, 26 row bị bộ
  lọc gắn cờ và cả 26 đều là logo/đồng hồ kênh; không thấy semantic collision trong mẫu này.
- Ngoại suy tuyến tính cho 293.336 keyframe là khoảng 287.348 residual region / 137.862
  frame / 72.066 shot-request sau lọc. Đây **không phải exact production count** và không đủ
  để tự mở Gemini: dev5 đều là nội dung HTV, nhãn calibration không phải holdout overlay đa
  kênh. Report/artifact sinh tự động nằm ngoài repo; production vẫn phải chạy chín batch rồi
  mới báo exact count/cost và xin người dùng duyệt Tầng 4.
  Không commit/push Git.

---

### [28/08/2026] OCR snapshot tăng dần — batch/tier contract sẵn sàng

- Thêm builder local CPU `scripts.build_ocr_incremental_snapshot`: nhận plan partition toàn
  catalog, đọc JSONL độc lập của từng worker, reject overlap/foreign/duplicate UID rồi
  atomic-build một SQLite. Bốn worker không bao giờ ghi chung SQLite.
- `coverage.json` schema v2 ghi từng batch với tier
  `craft_only|easyocr|vintern_calibrated|gemini_final`, complete/count/status/missing/pending,
  timestamp, video list, assigned/observed UID SHA-256 và source checksum. `craft_only` chỉ
  làm coverage, không vào FTS. Dev5 không thể trở thành batch thứ mười vì video partition
  bắt buộc disjoint/exhaustive.
- Snapshot vẫn luôn immutable, `complete=false`, `production_ready=false`; final SQLite chỉ
  được build sau terminal union đủ chín batch. Không commit/push và chưa upload snapshot mới.

### [28/08/2026] OCR production phase 1 — notebook CRAFT+EasyOCR sẵn sàng

- Thêm notebook self-contained `scripts/kaggle_ocr_production_easyocr.ipynb`; bốn tài khoản
  chỉ đổi `WORKER_SLOT`. Assignment cân theo frame: slot 1 = 01+09, slot 2 = 02+03+04,
  slot 3 = 05+08, slot 4 = 06+07.
- Notebook pin catalog/mapping và UID-set SHA từng batch, CRAFT/latin_g2 checksum, resume
  JSONL theo UID, retry error ở lần chạy sau, tạo router-v2 Vintern candidate không kèm crop,
  đóng archive không media và chỉ publish batch completion gate PASS. HF Dataset phải private;
  archive được tải lại theo revision và verify SHA-256.
- Phase này không gọi Vintern/Gemini, không ghi SQLite chung. Snapshot Online được build local
  từ những batch EasyOCR đã pull bằng incremental snapshot builder. Notebook mới chỉ qua
  static compile/contract test local; Kaggle production runtime chưa chạy. Không commit/push.

### [28/08/2026] OCR production phase 2 — notebook Vintern FP16 sẵn sàng

- Thêm notebook self-contained `scripts/kaggle_ocr_production_vintern.ipynb`, dùng lại đúng
  bốn slot phase 1. Batch nào có archive EasyOCR hoàn chỉnh trên private HF Dataset thì có
  thể chạy Vintern ngay; archive chưa có dừng rõ `WAIT_EASYOCR_ARCHIVE`.
- Notebook pin official `5CD-AI/Vintern-1B-v3_5` FP16 revision
  `b98f263eab246eb5269ade64edbdca8a887dc44d`, weight SHA-256 và Git blob OID của runtime;
  chỉ xử lý router-v2 candidate, dựng crop tạm từ bbox + keyframe, resume theo
  `candidate_id`, checkpoint 10.000 item và in progress 100 item.
- Output raw per batch được exact-set validate, không chứa media, chỉ upload vào
  `ocr/archives/{batch_id}/vintern/` khi `error=0`, rồi tải lại verify SHA-256. Manifest ghi
  `calibrated=false`, `searchable=false`; không gọi Gemini, không build SQLite và chưa thay
  EasyOCR text. Materialization calibrated vẫn là bước local sau khi pull cả hai layer.
- Sửa lookup mount Kaggle của notebook phase 1 để nhận thêm depth
  `*/*/*/keyframes-batch-*`, đúng cấu trúc Dataset thực người dùng vừa xác minh. Code mới chỉ
  qua compile/contract tests local; Vintern production Kaggle chưa chạy. Không commit/push.

### [28/08/2026] OCR pre-Gemini production gate + locked execution notebook

- Thêm `scripts.prepare_ocr_gemini_production`: local CPU tự kéo đúng 9 cặp archive
  EasyOCR/Vintern từ một private HF revision (hoặc nhận artifact root local), verify
  source-chain/checksum/completion, áp calibration table emergency 98 đã pin rồi báo **exact**
  residual region/frame/shot/request/contact-sheet. Không gọi Gemini.
- Preflight xuất bundle không media gồm report, residual mapping và one-request-per-shot
  manifest. Shot dày được chia nhiều contact-sheet image part trong cùng request. Cost chỉ là
  planning estimate với giá official snapshot 28/08/2026; usage/token thật chờ paid canary.
- Thêm notebook CPU-only `scripts/kaggle_ocr_production_gemini.ipynb`. Default
  `EXECUTION_MODE=preflight`, hai cờ paid đều false và code không đọc API key. Canary tối đa
  100 request cần exact report SHA + duyệt riêng; production cần thêm approved
  `model_version`, request cap và VND cap `<=400.000`.
- Runtime candidate là `gemini-2.5-flash-lite`, global MEDIUM, strict structured output exact
  string `region_id`; Gemini không sở hữu bbox/UID. Có retry/backoff, actual usage ledger,
  checkpoint/resume và per-batch HF round-trip verify. Model chỉ được coi pin runtime sau
  canary HTTP 200/schema-valid một version. Chưa gọi API, chưa phát sinh chi phí, chưa có
  exact production count vì 18 upstream archive chưa hoàn tất. Không commit/push.

### [28/08/2026] OCR — bỏ Vintern khỏi critical-path barrier

- Người dùng quyết định pre-Gemini không phải chờ Vintern. Nguồn bắt buộc còn đúng chín
  archive EasyOCR; với từng batch, preflight tự dùng Vintern calibrated nếu archive hoàn
  chỉnh có mặt, nếu thiếu thì đưa toàn bộ candidate router v2 thẳng sang Gemini và ghi
  `vintern_status=not_available_bypassed_to_gemini`/`vintern_bypassed=true`.
- EasyOCR text/confidence/bbox luôn được giữ làm fallback; không giả Vintern result hay
  override. Report tách `completed_batches_used` và `bypassed_batches`, nên chi phí tăng do
  bypass được nhìn thấy chính xác trước khi mở paid canary.
- Gemini vẫn khóa như cũ: exact preflight report SHA → paid canary HTTP 200/schema-valid →
  pin model version → người dùng duyệt request/VND cap. Thay đổi này chỉ bỏ barrier Vintern,
  không tự gọi API và không tự phát sinh phí. Không commit/push.

### [28/08/2026] OCR tạm bàn giao — EasyOCR production 9/9, chưa Vintern/Gemini

- CRAFT+EasyOCR production đã hoàn thành và round-trip verify trên private HF Dataset cho đủ
  chín batch, phủ đúng **293.336 keyframe**. Đây là tầng text hiện có để Nhánh 2 build
  development snapshot và test `FtsSearcher`; `materialized_text_tier=easyocr`.
- Pre-Gemini audit local đã verify đủ 9 archive EasyOCR, không có archive Vintern và không gọi
  API. Exact residual khi bypass Vintern: **830.301 region / 253.177 frame / 92.768 shot và
  request / 106.183 contact-sheet**. Report SHA-256:
  `f48b490d74bc043ebf1e7c14c1ba51fcf02d773b0779ea7c269bf17efef8cb55`.
- Estimate có reserve retry 15%: Standard khoảng **651.803 VND** (vượt trần 400.000 VND),
  Batch khoảng **325.902 VND** (trong trần). Đây chỉ là planning estimate; **Gemini API chưa
  được gọi, chưa phát sinh phí** và runner hiện tại là Standard nên không được mở full
  production theo estimate Batch.
- Vintern production chưa chạy. OCR tạm chốt ở EasyOCR để bàn giao song song; mọi SQLite
  snapshot tạo từ nguồn này phải ghi `complete=false`, `production_ready=false`, tier
  `easyocr`. OCR final còn thiếu Vintern/Gemini decision, terminal union và validation
  `ocr.sqlite` cuối.

---

### [28/08/2026] OCR — thêm SQLite snapshot development, không đổi production pipeline

- Theo quyết định người dùng, Nhánh 1 được bàn giao snapshot OCR bất biến để Nhánh 2 code/
  test FTS/fusion song song. Gate A threshold → Gate B full dev5 → chín batch production
  vẫn giữ nguyên; snapshot không phải đường tắt chất lượng.
- Thêm builder CPU `scripts.build_ocr_snapshot`: validate catalog state + UID, atomic-build
  đúng bảng `ocr_fts`, integrity/count/FTS probe, xuất thư mục version UTC + source hash với
  `coverage.json` và `SHA256SUMS`. Snapshot luôn `complete=false`,
  `production_ready=false`, `online_development_only` và không ghi đè thế hệ cũ.
- Coverage theo từng video ghi expected/observed/success/no_text/error/missing, engine count,
  text tier và Vintern required/completed/accepted/residual/pending. Adapter Gate 2 chỉ nạp
  text/confidence EasyOCR; kết quả Vintern hiện chỉ làm coverage evidence vì chưa có
  confidence calibrate để materialize đúng contract.
- Snapshot dev thật đã build ngoài repo:
  `ocr-snapshot-20260827T195734Z-85dd095d6ba9`, 4.164 FTS row/5 video,
  1,4195% catalog, SQLite SHA-256
  `b442d2d0a75bf3d004caa84e4d607d0a06fa70f7936b895d3b946ea15d03f0f7`. V001/V002/
  V003/V005 có Vintern v2 coverage xong nhưng còn residual; V006 partial, còn 3.007 region.
  Text trong SQLite vẫn được đánh dấu rõ `easyocr_only`; snapshot là child của lần build
  thử đầu và là bản handoff local mới nhất.
- Namespace HF được khóa riêng `ocr/snapshots/{snapshot_id}/...` trong cùng Dataset. Snapshot
  dev này đã upload và round-trip verify tại commit `15f2f3bed29a9f89683b01ba24b30578849b20bd`.
  Đây vẫn là snapshot partial/easyocr-only, không làm production pipeline READY. Không
  commit/push Git.

---

### [28/08/2026] OCR — chốt môi trường Tầng 1–4 và luồng HF Dataset

- Kiến trúc được người dùng chốt lại: CRAFT toàn catalog → EasyOCR mọi region → Vintern
  FP16 official/router v2 → Gemini residual/arbiter. Gemini 2.5 chỉ được canary/call sau khi
  Tầng 1–3 báo exact residual region, unique `keyframe_uid`, unique shot/request, token và
  chi phí; hiện chưa đổi runtime pin và chưa phát sinh API cost.
- Máy Codex hiện tại không GPU chỉ orchestration/code/validate artifact. Tầng 1–3 chỉ chạy
  Kaggle GPU, chia chín batch UID-disjoint trên tối đa bốn tài khoản OCR. RTX 4050 máy thi
  chỉ chạy Online, đọc `ocr.sqlite` build sẵn và không chạy model OCR.
- Timing thật từ dev-subset-5: CRAFT+EasyOCR xử lý 4.164 frame/41.212 region trong
  1.351,592 giây, 3,081 frame/s trên một T4. Ngoại suy thô 293.336 frame là 26,45 giờ/T4;
  bốn worker lý tưởng 6,61 giờ, cộng buffer 25% là 8,27 giờ. Cả 4.164 frame đều bị CRAFT
  đánh dấu `text_detected`, nên đây là ETA bảo thủ và là dấu hiệu cần audit false-positive,
  không phải tỷ lệ text đại diện toàn catalog.
- Artifact Tầng 1–3 bắt buộc đi qua đúng HF Dataset chung: Kaggle push archive/checksum/
  manifest dưới `ocr/archives/{batch_id}/`, verify remote, local `snapshot_download()` rồi
  mới union và build một `ocr.sqlite`. Worker production, HF upload/verify, local pull/merge
  và SQLite builder hiện chưa hoàn tất; pipeline vẫn **NOT READY**, không `complete=true`.
- Envelope đã có mode mới `craft_easyocr_vintern_gemini` và engine Vintern, giữ mode cũ chỉ
  để đọc evidence. Không đổi `shared/schemas/ocr.py::OcrResult`, `ocr_fts`, `online/` hoặc `app/`.
  Không commit/push.

---

### [28/08/2026] OCR Gate 2 — router v2 dev-only PASS, production vẫn đóng

- Checkpoint recovery được xác minh lại đúng SHA-256: EasyOCR 4.164 frame, 41.212 region;
  Vintern 27.927 result, không duplicate/foreign ID. Review bundle có 250 crop, cân bằng
  50 mẫu/video và 50 mẫu/category; đánh giá trực quan có hỗ trợ AI, không được ghi là human
  ground truth.
- Router seed cũ FAIL vì `mixed_frame` đẩy 34.335/41.212 region (83,31%) sang Vintern,
  ngoại suy khoảng 2,42 triệu crop / 210,6 giờ single T4. Không cần chạy nốt 6.408 candidate
  của checkpoint cũ.
- Thêm router v2 dev-only theo từng region: hard threshold `<0,40`; dải `0,40–0,60` chỉ
  xét mixed/glyph bất thường của chính region, không propagation toàn frame. Kết quả thật:
  14.807/41.212 candidate (35,93%), giảm 19.528 candidate / 56,87%; cả năm video đều dưới
  trần audit 40%.
- Trong 11.800 candidate v2 đã có kết quả Vintern, output guard giữ 11.659 success và chặn
  141 output qua tổng 153 reason rỗng, giải thích/prompt leak hoặc phình độ dài. Ngoại suy
  còn khoảng 1,043 triệu crop / 90,8 giờ single T4, chưa gồm
  EasyOCR/I/O/retry/Gemini; production chưa được phép mở chỉ từ số này.
- `PASS_ROUTER_V2_DEV_ONLY` chỉ cho phép bước kế tiếp là pilot end-to-end nhỏ có guard và
  đo temporal reuse/static-overlay. Gate 2 accuracy vẫn chưa PASS do thiếu human ground
  truth; không sửa production contract, `shared/schemas`, `online/` hoặc `app/`.
- Artifact audit/candidate JSONL được sinh ngoài repo. Chưa commit/push.

---

### [27/08/2026] OCR Gate 2 — Vintern runtime PASS, dev-subset-5 harness chờ chạy

- Official `5CD-AI/Vintern-1B-v3_5` revision
  `b98f263eab246eb5269ade64edbdca8a887dc44d` đã verify weight SHA-256
  `296a16a6bf28e6d3f0fb9298deba70b3cfa1d7519f4aa326e2f862bf2e63be05` và sáu remote-code
  Git OID. FP16 trên một T4 chạy 300/300 crop không lỗi/OOM; `max_num=1` P50 0,371 s,
  P95 0,662 s, peak reserved 2.508 MiB. Artifact benchmark local SHA-256
  `68edc4a6bee12f0413e44f934c807a69b4665a84937666af12992e51ea589118`; manifest 100 crop
  SHA-256 `6203c79d11e5a05c0b9d9deb89dd18f4b45c8e6f664835e40ad7a66a5038ff67`.
- Benchmark trên chỉ lấy 35 frame của `L21_V001`, nên PASS provenance/runtime nhưng chưa PASS
  accuracy. Người dùng chưa cho phép tích hợp Vintern vào production contract/code.
- Đã tạo dev-only `scripts/kaggle_ocr_gate2_dev5.ipynb`: chạy đúng 4.164 keyframe của năm
  video `L21_V001/V002/V003/V005/V006`, đối chiếu count + SHA-256 tập UID từ catalog thật,
  tách CRAFT detect/EasyOCR recognize, reload + resume JSONL, chạy Vintern cho toàn bộ seed
  escalation, xuất threshold sweep/ETA và 250 review row gồm CRAFT `no_text` control. Notebook
  không gọi Gemini; kết quả không lỗi vẫn là `PENDING_MANUAL_ACCURACY` cho tới khi review có
  ground truth. Notebook đã JSON-parse và compile tĩnh 8/8 code cell; chưa chạy dev5 thật.
- Không commit/push, không sửa `online/`, `backend/app`, `shared/schemas` hoặc production OCR
  envelope trong vòng này.

---

### [27/08/2026] OCR Gate 1 — chốt CRAFT-gated Gemini, final canary còn pending

- Người dùng đã làm rõ `BASELINE_SPEC.md` bản 12: CRAFT detect vùng chữ trên toàn bộ 293.336
  keyframe; frame không region terminal `no_text`, frame có region được crop/phóng và gom
  Begin/Middle/End thành một contact-sheet request/shot cho Gemini recognition. `latin_g2`
  chỉ nhận overflow/cloud stop. Paid cap vẫn là min(20.000 frame, 400.000 VND theo token
  ledger có reserve 15%).
- AI Studio của đúng project Free `gen-lang-client-0009440353` hiển thị
  `gemini-3.1-flash-lite` 15 RPM/250K input TPM/500 RPD (usage snapshot 3/15,
  6,96K/250K, 8/500); model 3.5 cùng trần và đã hết ngày ở 505/500 RPD. Model 2.5 vẫn 404
  với user mới; candidate escalation pin 3.1, không alias `latest`.
- Canary 3.1 crop-sheet cũ giữ bbox từ detector local: 10/10 synthetic shot request PASS,
  strict schema 100%, synthetic line recall 90/90, trung bình 1.250 input + 257,4 output
  token/shot, latency min/max 1.984/6.114 ms. Summary local SHA-256
  `97944101f2eaec1533b6cfc8c59d4cae1934f17224821d6af2955f6e745d7fab`; bị supersede vì
  chưa trả exact `region_id` và chưa dùng `MEDIA_RESOLUTION_MEDIUM`. Final canary chưa chạy
  lại vì `GEMINI_API_KEY` hiện không có trong process/User scope của shell Codex.
- Envelope production đổi sang `craft_gated_gemini`, attempt tách detection/recognition.
  Router schema v2 gom shot, chỉ reuse một frame khi embedding + CRAFT layout + crop SSIM +
  pHash cùng pass; embedding đơn độc không gate `no_text`. Gemini response không sinh bbox/
  UID, adapter map `region_id` về ba `keyframe_uid` và bbox CRAFT.
- Gate 2 vẫn đóng vì catalog local recover không có JPEG. Cần đúng dev-subset-5 để benchmark
  CRAFT/`latin_g2` throughput/recall, routing rate, MEDIUM-vs-HIGH, interrupt→resume và token
  cost ảnh thật trước production. Verification local: 126/126 unit test, compileall và
  `git diff --check` PASS. Không có commit/push mới trong vòng thay đổi này.

---

### [27/08/2026] OCR Gate 1 — catalog/model/quota/strict-schema canary

- Catalog private HF đã audit tại immutable revision
  `72848939bdc5ebd57b5cd45370e685aee036cafa`: `frames.csv` SHA-256
  `ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37`, đúng 293.336
  keyframe/873 video; UID formula mismatch 0, duplicate 0, chín batch disjoint/exhaustive.
- `gemini-2.5-flash-lite` trả HTTP 404 và API nói không còn cấp cho user mới. Model pin đã
  verify cho audit là `gemini-3.5-flash-lite`, catalog version `3.5-flash-lite-07-2026`;
  không dùng alias `latest`. Canary loose-schema 390 request trước đó thực tế cũng dùng
  model 3.5; model 2.5 chỉ có đúng preflight 404, không được trộn vào baseline.
- Google AI Studio Rate Limit của đúng project `gen-lang-client-0009440353`, Free tier,
  hiển thị trần thật: **15 RPM, 250.000 TPM, 500 RPD**. Snapshot sau audit là
  `17/15 RPM`, `25,28K/250K TPM`, `502/500 RPD`; console báo đã chạm rate limit.
- Đã đổi canary sang Gemini `responseJsonSchema`: mỗi bbox bắt buộc `minItems=maxItems=8`,
  từng tọa độ và confidence trong `[0,1]`, `additionalProperties=false`. Strict preflight
  HTTP 200/schema-valid. Baseline ghép hai JSONL shard đủ 100 unique request: 97 terminal
  success, 3 terminal timeout/quota, 117 HTTP attempt; **97/97 HTTP 200 schema-valid, 0
  invalid bbox**, normalized text và language đúng 97/97. Latency success P50 2.190 ms,
  P95 2.647 ms; có 13 timeout-attempt và 7 attempt HTTP 429 được giữ provenance.
- Rate harness cũ có lỗi catch-up burst sau timeout; đã sửa để mọi HTTP attempt, kể cả
  retry, được pace từ thời điểm attempt thật và không đuổi lịch. Peak 17 vượt RPM=15 đã trả
  `429 RESOURCE_EXHAUSTED`; ramp tiếp theo dừng vì RPD đã 502/500, không đốt thêm quota.
- Capacity lower bound cho 293.336 keyframe: chỉ xét 15 RPM là khoảng 13,58 ngày; áp trần
  500 RPD là khoảng 586,67 ngày. Trong 6 ngày free tier tối đa 3.000 request (~1,023%
  catalog), nên Gate 1 **chưa PASS** và không sang Gate 2 với project Free tier hiện tại.
- Evidence local sanitize nằm dưới `tmp/ocr-gate1/` và bị `.gitignore`; chưa tạo
  `ocr.sqlite` production. OCR targeted unit test 9/9 PASS bằng Python 3.11 local.

---

### [26/08/2026] Thay BEiT-3 bằng EVA-CLIP

- Quyết định cuối: BEiT-3/Microsoft UniLM bị loại vĩnh viễn; không mở lại audit, checksum,
  sandbox conversion hoặc adapter. Registry giữ row `blocked_model_selection` chỉ để ghi
  nhận trạng thái cuối.
- Modality thứ ba chính thức là `eva_clip`; index bàn giao là `eva_clip.faiss`; Publishing
  Ready yêu cầu CLIP + SigLIP + EVA-CLIP khớp 100% `keyframe_uid`.
- Pin `timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k` tại immutable revision
  `bf4190eb65dd5204ffb03e980108beb1200e0873`; chỉ cho phép
  `open_clip_model.safetensors` với SHA-256 đã pin. Code/dev runner được chuẩn bị nhưng gate
  Kaggle T4 dev-subset-5 vẫn phải chạy thật trước production.
- `scripts.run_eva_clip_dev_gate` tự chạy verifier → intentional exit 75 → process mới
  resume → validator và assert dimension/dtype/CUDA; cố ý không có archive/upload/9-batch.

---

### [26/08/2026] Chốt Python 3.12 cho Kaggle Visual

- Người dùng xác nhận rõ cho phép đổi contract: chỉ profile `kaggle-gpu` dùng Python 3.12.x;
  local CPU, Shot Windows GPU và Shot Colab vẫn giữ Python 3.11.x.
- Gate Kaggle thật: Python 3.12.13, Torch `2.10.0+cu128`, CUDA 12.8, Tesla T4; 87 test PASS
  với 6 skip theo platform; CLIP và SigLIP resolve đúng immutable revision.
- Visual manifest ghi thêm Python/system/machine bên cạnh Torch/Transformers/CUDA/GPU.
  Artifact `.npy` vẫn `int64` UID + `float16` vector, `allow_pickle=False`, nên handoff về
  local Python 3.11 để build FAISS không đổi contract.
- Cấm resume một checkpoint dở dang qua runtime Python/Torch/Transformers khác; session đổi
  runtime phải dùng batch ID/output mới để không trộn shard khó audit.

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

- Môi trường local/Shot Nhánh 1 dùng CPython 3.11.9; Kaggle Visual dùng Python 3.12.x.
  Dependency tách profile để Kaggle không bị thay wheel PyTorch/CUDA có sẵn.
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
3. Tiếp tục CLIP/SigLIP production trên Kaggle bằng artifact float16 và checkpoint/resume.
4. Chạy EVA-CLIP dev-subset CUDA gate; không tạo production runner trước khi PASS.
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
