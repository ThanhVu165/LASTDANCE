# OCR runbook — CRAFT → EasyOCR → Vintern → Gemini residual

Nguồn chuẩn kỹ thuật là `docs/BASELINE_SPEC.md` §2.1d/§2.2. Runbook này chỉ mô tả cách
vận hành Gate 1–production; không thay schema `OcrResult` hoặc `ocr_fts`.

## Môi trường và luồng artifact đã chốt

- Máy Codex hiện tại không GPU: chỉ code, orchestration và validate JSONL/manifest; cấm chạy
  CRAFT/EasyOCR/Vintern.
- Tầng 1–3 chạy Kaggle GPU theo chín batch UID-disjoint, được phép phân công tối đa bốn tài
  khoản OCR. Không dùng tài khoản/quota của Visual hoặc ASR.
- RTX 4050 máy thi chỉ chạy Online và đọc `ocr.sqlite` đã build sẵn; không chạy model OCR.
- Sau mỗi batch hoàn chỉnh, archive/checksum được push dưới `ocr/archives/{batch_id}/` trong
  cùng HF Dataset đã dùng cho Visual. Máy local dùng `snapshot_download()` theo
  `BASELINE_SPEC.md` §2.4; không có luồng đồng bộ OCR riêng. Artifact chỉ còn trên Kaggle
  chưa được coi là bàn giao.

## Gate 1 — preflight bắt buộc

API audit/canary local dùng profile nhỏ:

```powershell
python -m pip install -r requirements/ocr-api.txt
```

Không dùng urllib mặc định của một environment Windows chưa có CA bundle làm bằng chứng
API: lỗi TLS transport phải được tách khỏi HTTP status/model/quota. Harness chính thức dùng
`httpx` đã pin và không ghi API key/header vào artifact.

### 1. Catalog production

Input là đúng `AIC_DATA/index/frames.csv` cùng `frames.csv.state.json` đã publish. Không dùng
con số ghi trong notebook/status làm bằng chứng thay cho file thật.

```powershell
$env:AIC_DATA = "D:\path\to\aic-data"
python -m scripts.audit_ocr_catalog `
  --expected-records 293336 `
  --expected-videos 873
```

Lệnh fail closed nếu state/hash/schema sai, có UID duplicate hoặc UID không khớp công thức
BLAKE2b trong baseline. Hai expected count chỉ dùng sau khi đã lấy được catalog thật; nếu
catalog trả count khác thì dừng để điều tra, không sửa expected cho khớp tùy tiện.

### 2. Gemini project/tier/quota (chỉ làm sau Tầng 1–3)

Tier là thuộc tính project/billing account, không suy ra từ API key. Bằng chứng Gate 1 phải
ghi cùng timestamp:

1. Google AI Studio Projects/Billing: project ID, Plan/Tier và trạng thái billing/credit.
2. Dashboard > Usage/Rate limits: RPM, input TPM, RPD đang active cho đúng model ID dự kiến
   production và đúng project. Candidate người dùng chọn là `gemini-2.5-flash-lite`, nhưng
   runtime pin vẫn giữ nguyên cho tới khi Paid key canary HTTP 200 + schema-valid; không dùng
   alias `latest` và không lấy canary model khác thay bằng chứng.
3. Canary bằng chính key của project đó trên một tập ảnh nhỏ: số request, concurrency,
   latency p50/p95, input/output token và toàn bộ 429/5xx/timeout. Không log API key.

Runtime lower-bound chỉ được tính sau ba bằng chứng trên:

```text
max(N / RPM, input_tokens / TPM, ceil(N / RPD) * 1440 phút)
```

Sau đó so với throughput đo ở canary; lấy thời gian lớn hơn và cộng margin retry. `401/403`
hoặc quota/global project failure dừng cloud worker và xuất routing state; phần crop chưa
xử lý chuyển sang job `latin_g2` tường minh, không fallback âm thầm trong exception handler.

### 3. EasyOCR offline weight pin

Registry committed: `configs/ocr_easyocr_models.json`. Tải package/ZIP trước phiên chạy,
extract đúng hai `.pth` vào một model storage directory, rồi chạy:

```bash
python -m pip install -r requirements/ocr-kaggle-gpu.txt
```

Trên Kaggle phải ghi `python`, `torch`, `torchvision`, CUDA và GPU trước/sau install; dừng nếu
pip thay Torch/Torchvision của image. Profile này mới pin dependency, **chưa** là bằng chứng
tương thích Python 3.12/Torch 2.10 cho tới khi preflight Gate 2 chạy thật.

```powershell
python -m scripts.verify_ocr_model_weights `
  --model-storage-dir "D:\models\easyocr" `
  --archive-dir "D:\downloads\easyocr" `
  --wheel "D:\downloads\easyocr\easyocr-1.7.2-py3-none-any.whl"
```

Production preflight không được dùng `--skip-package-check`. Reader phải nhận
`download_enabled=False`, `lang_list=["vi", "en"]`, CRAFT detector và auto-selected
`latin_g2`; thiếu/sai một byte thì dừng trước inference. Weight/ZIP không commit vào Git và
không để lẫn vào archive OCR.

### 4. Artifact envelope/checkpoint

- Canonical terminal model: `offline.ocr_artifacts.OcrRecordEnvelope`; pipeline mới dùng
  `execution_mode=craft_easyocr_vintern_gemini`. Attempt theo thứ tự CRAFT detection →
  EasyOCR mọi region → Vintern router v2 → Gemini residual. Có thể terminal ở CRAFT
  (`no_text`), EasyOCR hoặc Vintern. Theo baseline bản 21, Vintern được bypass khi archive
  batch chưa sẵn sàng, nhưng phải ghi `vintern_not_available` và route toàn bộ candidate v2
  sang Gemini; không được giả override.
- CRAFT `no_text` kết thúc record mà không gọi Gemini. CRAFT phát hiện region nhưng không có
  recognizer attempt là artifact dở dang, không được terminal-publish.
- Shard manifest: `offline.ocr_artifacts.OcrShardManifest`.
- Mỗi batch có một append-safe JSONL shard. Khi resume, JSONL hiện có là authority; đọc lại,
  validate từng line và reject duplicate UID trước khi tiếp tục. State/manifest được ghi
  atomic sau shard, không dùng `next_index` đơn lẻ làm authority.
- `success` có `OcrResult`; `no_text`/`error` có `result=null`. Error provenance phải sanitize,
  tuyệt đối không chứa API key/header/request body nhạy cảm.
- Batch gate PASS khi `success + no_text == expected`, `error == 0`, không duplicate,
  missing hoặc foreign UID. Đây chỉ là OCR-stage completion; không tự nâng publishing Ready
  toàn pipeline khi EVA-CLIP/FAISS còn thiếu.

### 5. Shot grouping, reuse gate và budget cap Gemini

Chỉ tạo queue Gemini sau khi Tầng 1–3 kết thúc. Trước mọi API call, report phải ghi exact
residual region, unique frame (`keyframe_uid`), unique shot/request, token và chi phí; dừng
để người dùng chọn số lượng. Contact-sheet chỉ chứa residual crop còn lại.

Feature JSONL/CLI dưới đây là evidence Gate 1 cũ của kiến trúc CRAFT→Gemini. Giữ để đọc
artifact lịch sử, không dùng làm production selection mới:

```powershell
python -m scripts.rank_ocr_escalation `
  --features-jsonl "D:\path\batch-01.craft-features.jsonl" `
  --estimated-prompt-tokens-per-request 1250 `
  --estimated-output-tokens-per-request 260 `
  --output-json "D:\path\batch-01.gemini-selection.json"
```

Hai token estimate tính theo **một request/shot**, bắt buộc lấy từ canary ảnh thật cùng
contact-sheet/model/media resolution. Queue mới gom residual trong cùng shot thành một
request và chặn theo exact token ledger + **400.000 VND**; số paid frame không tự chốt trước
khi người dùng xem report Tầng 1–3.

`visual_priority` chỉ xếp lịch shot chạy Free/paid trước. Một target chỉ reuse text từ frame
khác khi `embedding_cosine_to_source`, `craft_layout_similarity_to_source`,
`crop_ssim_to_source` và `crop_phash_distance_to_source` cùng qua policy. Report ghi SHA-256
semantic của policy/features; không sửa output JSON để vượt cap.

Contact sheet chỉ chứa residual crop sau EasyOCR/Vintern đã phóng rõ. Mỗi crop có `region_id` duy nhất;
prompt gửi mapping local nhưng Gemini response chỉ được trả `region_id`, `text`, `language`,
`confidence`. `responseJsonSchema` ép exact số region; bbox/keyframe UID không do model sinh.
Canary hiện đặt global `MEDIA_RESOLUTION_MEDIUM`; phải A/B với HIGH/default trên chữ nhỏ thật,
không dùng LOW trên full frame. Tham chiếu vận hành: Google
[Media resolution](https://ai.google.dev/gemini-api/docs/generate-content/media-resolution).

## Gate 2 và production

Dev-subset-5 đã tạo evidence CRAFT+EasyOCR và Vintern partial; router v2 chỉ PASS ở mức
operational/fan-out, chưa có human ground truth để gọi accuracy PASS. Production vẫn đóng.

### Gate A — CRAFT emergency review 100 frame

Policy schema v2 tại `configs/ocr_craft_gate_a_policy.json` pin đúng 100 UID đã được một
annotator hoàn thành: 60 V001 + 40 V002. Đây là emergency deadline override, không phải mẫu
cân bằng năm video. Ba cấu hình vẫn là `recall_current`, `balanced`, `strict`; yêu cầu region
recall `>=0,98` và text-bearing frame recall `>=0,99` không bị hạ. Nếu không config nào đạt,
evaluator giữ `recall_current` và phát decision `DEADLINE_OVERRIDE_KEEP_CURRENT` để Gate B
full dev5 đo lại; limitation bắt buộc nằm trong report.

Chạy một lần `scripts/build_ocr_gate_a_mini_dataset.py` với Dataset gốc để tạo
`ocr-gate-a-mini.zip`, rồi upload ZIP thành Kaggle Dataset private. Notebook lịch sử đã chạy
300 frame; contract v2 dùng đúng 100 dòng đầu đã pin, không cần chạy GPU lại. Nếu cần tái lập,
upload và chạy
`scripts/kaggle_ocr_craft_gate_a_threshold_pilot.ipynb` trên một Kaggle T4, chỉ gắn Dataset
mini này; không gắn lại `thvu165/aic-2026-keyframes`. Notebook:

- pin SHA-256 của ZIP/manifest, verify đủ 300 ảnh nguồn rồi chọn exact 100 UID contract v2;
- verify CRAFT weight checksum, chỉ chạy detector 300 lượt (100 × 3 config), không chạy
  recognizer/Vintern/Gemini;
- checkpoint/resume theo `(keyframe_uid, config_id)`;
- xuất `/kaggle/working/ocr_gate_a_review_bundle.zip` gồm 100 ảnh 2×2, result JSONL, sample
  manifest, CSV nhãn trống, instructions và report `PENDING_HUMAN_GROUND_TRUTH`.

Người thật xem ảnh gốc cùng ba overlay rồi điền mọi dòng CSV: `gt_has_text=yes|no`, số vùng
chữ thật, số vùng chữ thật mỗi config bỏ sót, `annotator` và ghi chú nếu cần. Không dùng
EasyOCR/Vintern agreement làm ground truth. Tải bundle về local, giải nén và chạy:

```powershell
python -m scripts.evaluate_ocr_craft_gate_a `
  --results-jsonl "D:\gate-a\craft-threshold-results.jsonl" `
  --review-csv "D:\gate-a\craft-gate-a-human-review.csv" `
  --output-json "D:\gate-a\craft-gate-a-evaluation.json"
```

Gate B chấp nhận `PASS_THRESHOLD_SELECTED` hoặc exact
`DEADLINE_OVERRIDE_KEEP_CURRENT`, đều phải có `gate_b_allowed=true` và policy hash đúng.
ETA Tầng 1–3 chỉ được tính lại từ full dev5 Gate B, không ngoại suy từ 100 frame.

### Gate B — full dev5 và ground-truth Vintern

Code Gate B có thể chuẩn bị trước nhưng execution phụ thuộc report Gate A. Notebook
`scripts/kaggle_ocr_gate_b_dev5_calibrated.ipynb` yêu cầu attach keyframe Dataset và đúng
một `craft-gate-a-evaluation.json`; nó dừng trước model load nếu report thiếu/không được phép,
policy SHA lệch, selected threshold bị sửa hoặc deadline-override limitation bị thay đổi.

Khi Gate A được authorize bằng PASS thường hoặc deadline override, notebook dùng threshold
từ report để chạy lại đủ 4.164 frame, không
hardcode lựa chọn: CRAFT → EasyOCR mọi region → router v2 → Vintern FP16 official. JSONL
resume bị khóa bằng run signature gồm Gate A report SHA, policy, threshold, router và Vintern
revision/checksum. Vintern result cùng inference ghi output length, guard margin và
`mean_token_logprob=null` khi `model.chat` không expose; không chạy model lần hai.

Output cần tải:

- `/kaggle/working/ocr_gate_b_checkpoint.zip`: EasyOCR frame JSONL, candidate JSONL,
  Vintern result JSONL, signature/report; refresh mỗi 250 item. Khi attach ZIP này vào một
  session mới, notebook tự verify member/signature, phục hồi JSONL và dựng lại crop bị thiếu
  từ bbox + ảnh nguồn; nếu Vintern đã đủ result thì bỏ qua tải model.
- `/kaggle/working/ocr_gate_b_ground_truth_review.zip`: đúng 100 candidate từ 100 frame khác
  nhau, 20/video, chọn deterministic theo năm stratum router/confidence; gồm context/crop,
  CSV và selection manifest có SHA-256.

Trong CSV Vintern, người thật đặt `label_status=labeled`; nếu detector bắt nhiễu không có
chữ thật thì `ground_truth_is_empty=yes` và để `human_text` rỗng. Nếu hai người vẫn không
đọc được, đặt `label_status=exclude_unreadable`; không bịa text. Deadline override hiện tại
cho phép đúng hai row unreadable ở sample 15/24; calibration dùng 98 row `labeled` thuộc 98
distinct `keyframe_uid` với policy
`configs/ocr_vintern_calibration_policy_emergency_98.json`. Exclusion thứ ba phải fail
closed. Đây là evidence tier `emergency_single_annotator_98_of_100`, không phải standard
300-frame PASS. CSV mở bằng Excel có thể làm tròn `keyframe_uid` 64-bit và confidence; trước
calibration phải phục hồi mọi cột immutable từ CSV gốc trong review ZIP theo `sample_index`,
chỉ lấy năm cột human-label từ file đã chỉnh.

Trước khi mở bốn Kaggle worker, tạo một plan JSON hash-bound với đúng chín batch và tối đa
bốn assignment `enabled=true`, sau đó chạy:

```powershell
python -m scripts.validate_ocr_worker_plan --plan "D:\path\ocr-worker-plan.json"
```

`offline.ocr_production.OcrWorkerPlan` fail closed khi overlap, thiếu hoặc thừa batch.
`OcrLayerShardManifest` khóa mỗi output Tầng 1–3 bằng catalog/config/input/output SHA-256,
tập UID được giao và count item/keyframe. Account/token không ghi trong plan Git; người vận
hành giữ mapping `worker_id → Kaggle account` ở ghi chú riêng. Contract namespace HF chỉ cho
phép `ocr/archives/{batch_id}/...`, không thể ghi sang `clip/`, `siglip/` hay `eva_clip/`.

## Snapshot bàn giao sớm cho Nhánh 2

Snapshot cho phép code/test `FtsSearcher` và fusion bằng dữ liệu thật trong khi Gate A,
Gate B và production chín batch vẫn chạy bình thường. Builder chạy CPU, không chạy model và
không dùng quota Kaggle:

```powershell
$env:AIC_DATA = "D:\path\to\aic-data"
python -m scripts.build_ocr_snapshot `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --source-format ocr_envelope_v1 `
  --source-jsonl "$env:AIC_DATA\ocr\terminal\batch-01.jsonl"
```

Để tạo snapshot ngay từ checkpoint Gate 2 dev5 đã cứu, chỉ materialize text/confidence
EasyOCR; Vintern được dùng làm **coverage evidence**, chưa thay text vì Vintern không có
confidence đã calibrate:

```powershell
python -m scripts.build_ocr_snapshot `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --source-format gate2_easyocr_dev_v1 `
  --source-jsonl "D:\evidence\easyocr-frames.jsonl" `
  --vintern-results-jsonl "D:\evidence\vintern-results.jsonl"
```

Output mặc định:

```text
$AIC_DATA/ocr/snapshots/ocr-snapshot-<UTC>-<source-hash>/
  ocr.sqlite
  coverage.json
  SHA256SUMS
```

Khi nhiều worker hoàn thành lệch tầng, tạo một plan local (không chứa token/account):

```json
{
  "schema_version": 1,
  "catalog_sha256": "<sha256 frames.csv>",
  "expected_batch_ids": ["batch-01", "batch-02"],
  "batches": [
    {
      "batch_id": "batch-01",
      "tier": "easyocr",
      "video_ids": ["L21_V001"],
      "source_format": "ocr_envelope_v1",
      "source_jsonl": "artifacts/batch-01.easyocr.jsonl",
      "updated_utc": "2026-08-28T10:00:00+00:00"
    },
    {
      "batch_id": "batch-02",
      "tier": "craft_only",
      "video_ids": ["L21_V002"],
      "source_format": "craft_jsonl_v1",
      "source_jsonl": "artifacts/batch-02.craft.jsonl",
      "updated_utc": "2026-08-28T10:05:00+00:00"
    }
  ]
}
```

Plan production thực phải liệt kê đủ chín batch và toàn bộ 873 video đúng một lần. Build:

```powershell
python -m scripts.build_ocr_incremental_snapshot `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --plan "$env:AIC_DATA\ocr\incremental-snapshot-plan.json" `
  --parent-snapshot-id "ocr-snapshot-..."
```

Script pin `catalog_sha256`, suy tập UID từ video partition, reject video/UID overlap,
foreign/duplicate và source thiếu. Mỗi worker chỉ giao JSONL; tuyệt đối không để nhiều máy
ghi chung SQLite. `craft_only` xuất coverage nhưng không vào FTS. Dev5 thuộc batch-01 nên
plan chỉ chọn một nguồn cho batch-01, không append snapshot dev5 như batch riêng.

Quy tắc bắt buộc:

- Builder validate `frames.csv.state.json`, reject UID duplicate/foreign/video mismatch,
  build vào staging rồi atomic-rename; cùng timestamp/source không được ghi đè.
- `ocr.sqlite` chỉ có đúng bảng `ocr_fts` năm cột từ baseline và chỉ nạp record `success`.
  Builder chạy `PRAGMA integrity_check`, đếm row và một FTS5 probe thật trước khi publish.
- `coverage.json` luôn ghi `complete=false`, `production_ready=false`, development-only;
  coverage từng video tách `materialized_text_tier` khỏi trạng thái Vintern
  `not_run|partial|complete|complete_with_residual`. Schema v2 còn có `batches` với tier
  `craft_only|easyocr|vintern_calibrated|gemini_final`, complete/count/status/pending,
  `updated_utc`, video list, UID-set hash và source checksum để Online biết chất lượng thật.
- Nhánh 2 phải trỏ rõ thư mục `snapshot_id`; không copy/đổi tên snapshot thành artifact
  production và phải invalidate cache khi đổi snapshot ID.
- Upload snapshot bằng `scripts.publish_ocr_snapshot_hf`: script assert repo private, local
  checksum, namespace chưa có/đã đủ đúng ba file, tạo một commit rồi
  `snapshot_download()` lại theo commit SHA và so hash hai chiều. Repo public/gated, remote
  partial hoặc hash lệch đều dừng trước khi ghi đè:

```powershell
python -m scripts.publish_ocr_snapshot_hf `
  --snapshot-dir "D:\path\ocr-snapshot-<UTC>-<hash>" `
  --repo-id "MinhThuw0103/lastdance-visual-embeddings" `
  --output-report "D:\path\ocr-snapshot.hf-publish-report.json"
```

- Snapshot dev `ocr-snapshot-20260827T195734Z-85dd095d6ba9` đã upload và round-trip verify
  thành công tại private Dataset commit
  `15f2f3bed29a9f89683b01ba24b30578849b20bd`. Nhánh 2 phải pull pin revision này:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="MinhThuw0103/lastdance-visual-embeddings",
    repo_type="dataset",
    revision="15f2f3bed29a9f89683b01ba24b30578849b20bd",
    allow_patterns=[
        "ocr/snapshots/ocr-snapshot-20260827T195734Z-85dd095d6ba9/*"
    ],
)
```

  Sau download phải verify `SHA256SUMS`; không dùng `main/latest`. Đây là snapshot
  EasyOCR-only, partial development, không phải production final.
- Không commit `ocr.sqlite`, JSONL, `coverage.json`, `SHA256SUMS`, crop hoặc archive sinh ra.

### Vintern pilot và production phase 2

Người dùng đã chốt kiến trúc `CRAFT -> EasyOCR -> Vintern -> Gemini residual`, nhưng chưa
cho phép gọi Gemini hay đánh dấu production complete. Notebook
`scripts/kaggle_ocr_gate2_dev5.ipynb` chạy trên Kaggle T4 x2 nhưng mask GPU 1, không gọi
Gemini và không tạo `OcrRecordEnvelope` production. Gắn Dataset
`thvu165/aic-2026-keyframes`, bật Internet rồi Run All.

Notebook fail closed theo count + SHA-256 tập UID của đủ 4.164 keyframe dev-subset-5, tách
CRAFT detection khỏi EasyOCR recognition, checkpoint JSONL theo UID, reload EasyOCR giữa
seed 50 frame và phần resume, rồi chạy Vintern FP16 trên toàn bộ seed escalation region.
Ngưỡng confidence `0.60`, heuristic mixed-language và `max_num=1/2` chỉ là seed để đo sweep;
không được copy sang production trước manual accuracy review.

Artifact cần tải về:

- `/kaggle/working/ocr_gate2_dev5_report.json`;
- `/kaggle/working/ocr_gate2_manual_review.csv`;
- `/kaggle/working/ocr_gate2_manual_review_sheets.zip`;
- `/kaggle/working/ocr_gate2_gemini_residual_canary.jsonl`.

`decision=PENDING_MANUAL_ACCURACY` là kết quả mong đợi của một run không lỗi; không được đổi
thành PASS chỉ vì EasyOCR và Vintern đồng ý nhau. Manual review phải phủ cả năm video, gồm
CRAFT `no_text` control để đo false-negative detector, EasyOCR pass, Vintern escalation và
Gemini residual. Chỉ sau khi người dùng duyệt accuracy mới được đề xuất sửa production
contract/code; Gemini residual chưa được gọi API trong notebook này.

Nếu phiên Kaggle toàn phần bị mất nhưng đã tải được
`ocr_gate2_resume_checkpoint.zip`, dùng notebook recovery
`scripts/kaggle_ocr_gate2_partial_review.ipynb`. Notebook này chạy CPU, không cài hoặc chạy
CRAFT/EasyOCR/Vintern/Gemini. Input bắt buộc là Dataset keyframe
`thvu165/aic-2026-keyframes` và ZIP chứa đúng ba file `easyocr-frames.jsonl`,
`vintern-candidates.jsonl`, `vintern-results.jsonl`. Notebook fail closed theo record count,
SHA-256, duplicate ID, foreign candidate ID và tham chiếu `keyframe_uid`; checksum hiện pin
đúng checkpoint đã cứu `4.164 / 34.335 / 27.927` record.

Recovery chỉ tạo evidence `PARTIAL_VINTERN_CANARY_PENDING_MANUAL_ACCURACY`: loại region
Vintern chưa hoàn tất, lấy deterministic 50 mẫu/video, dựng lại crop từ `bbox_px` + JPEG gốc
và xuất:

- `/kaggle/working/ocr_gate2_partial_review_artifacts.zip`;
- bên trong có summary JSON, CSV 250 dòng, ZIP contact sheet và checkpoint manifest;
- sau khi điền nhãn `yes/no` vào CSV, hai cell cuối tạo
  `/kaggle/working/ocr_gate2_partial_accuracy.json`.

Không commit ZIP/JSONL/JPEG/CSV/report sinh ra. Partial report không thay thế full Gate 2,
không tự đổi production contract và không được diễn giải model agreement là accuracy.

Nếu chỉ cần audit checkpoint đã tải trên máy, chạy
`scripts/local_ocr_gate2_partial_audit.ipynb` bằng Python 3.11. Notebook tự tìm ba JSONL ở
working directory hoặc thư mục cha; có thể set `OCR_GATE2_CHECKPOINT_DIR` nếu đặt ở nơi
khác. Output `ocr_gate2_partial_auto_audit.json` được ghi cạnh checkpoint và không commit.
Audit local chốt checksum, duplicate, coverage, runtime và router fan-out; khi `AIC_DATA`
không chứa JPEG/crop thật, trường manual accuracy phải giữ `BLOCKED_MISSING_LOCAL_KEYFRAME_IMAGES`.

Sau khi có ba JSONL recovery, tính lại router Vintern Gate 2 v2 trên local CPU, không chạy
lại model và không dùng quota Kaggle/API:

```powershell
python -m scripts.audit_ocr_gate2_router_v2 `
  --easyocr-jsonl "D:\path\easyocr-frames.jsonl" `
  --vintern-results-jsonl "D:\path\vintern-results.jsonl" `
  --output-json "D:\path\ocr_gate2_router_v2_audit.json" `
  --output-candidates-jsonl "D:\path\ocr_gate2_router_v2_candidates.jsonl"
```

Policy dev-only nằm ở `configs/ocr_vintern_gate2_policy.json`. Router xét độc lập từng
region: text rỗng hoặc confidence `<0.40` vào Vintern; dải `0.40–0.60` chỉ vào khi chính
region đó mixed hoặc có glyph bất thường. Tuyệt đối không truyền cờ mixed của một region
sang toàn frame. `offline.ocr_vintern_gate2.vintern_output_rejection_reasons` chặn output
dạng giải thích/prompt leak, rỗng hoặc phình độ dài bất thường; output bị chặn phải giữ
EasyOCR hoặc chuyển Gemini residual, không được publish như text thật.

Script fail closed khi JSON sai, trùng/foreign ID hoặc confidence ngoài `[0,1]`; chạy lại
toàn bộ từ đầu là cơ chế resume vì chỉ đọc JSONL và hoàn tất trong vài giây. Hai output là
artifact audit sinh tự động, để ngoài repo và không commit. `PASS_ROUTER_V2_DEV_ONLY` chỉ
cho phép pilot end-to-end nhỏ; không đồng nghĩa Gate 2 accuracy hay production PASS.

1. Tạo worker plan bốn tài khoản với chín batch UID-disjoint; ghi account owner trước khi
   chạy, không tự cân lại batch giữa phiên.
   Với một người vận hành bốn tài khoản, assignment production đã cân theo 293.336 frame:
   slot 1 = batch 01+09 (69.784), slot 2 = 02+03+04 (83.798), slot 3 = 05+08 (76.705),
   slot 4 = 06+07 (63.049). Dùng cùng notebook
   `scripts/kaggle_ocr_production_easyocr.ipynb`, mỗi tài khoản chỉ đổi `WORKER_SLOT=1..4`.
2. Chạy CRAFT+EasyOCR, ngắt giữa batch, khởi động process mới và chứng minh resume JSONL
   không duplicate/mất UID. Timing dev5 hiện có: 4.164 frame/41.212 region trong 1.351,6 s,
   khoảng 3,08 frame/s trên một T4; ETA full catalog chỉ là ngoại suy cho tới pilot batch.
   Notebook pin checksum catalog/mapping/từng batch và EasyOCR weights, refresh checkpoint
   ZIP mỗi 250 frame, tạo candidate Vintern nhưng không archive crop/media. Mỗi batch chỉ
   upload HF sau completion gate exact UID + `error=0`, rồi tải lại theo revision vừa tạo và
   so SHA-256. Để demo resume một lần: đặt `INTERRUPT_AFTER_NEW_FRAMES=50`, tải checkpoint,
   attach vào session mới, trả về 0 và Run All. Bình thường luôn để 0.
3. Chạy `scripts/kaggle_ocr_production_vintern.ipynb` trên đúng bốn tài khoản/slot phase 1.
   Mỗi batch chỉ được bắt đầu sau khi EasyOCR của batch đó đã in `BATCH_COMPLETE` và archive
   đã round-trip verify trên HF. Notebook tự tải archive tại
   `ocr/archives/{batch_id}/easyocr/`, pin revision tải xuống, verify ZIP/checksum/manifest,
   dựng crop tạm từ JPEG nguồn + `bbox_px`, rồi chạy official Vintern FP16 revision
   `b98f263eab246eb5269ade64edbdca8a887dc44d` **chỉ** trên candidate router v2.
   Result ghi `output_length`, `guard_margin_ratio`, `mean_token_logprob=null` khi
   `model.chat` không expose log-prob và không sửa bbox. Progress in mỗi 100 candidate;
   checkpoint ZIP mỗi 10.000 candidate. Để chứng minh resume một lần, đặt
   `INTERRUPT_AFTER_NEW_CANDIDATES=50`, tải checkpoint, attach session mới, trả về 0 rồi
   Run All. Archive raw được upload riêng tại `ocr/archives/{batch_id}/vintern/` chỉ sau
   exact candidate-set + `error=0`, rồi tải lại verify SHA-256. Nó luôn ghi
   `calibrated=false`, `searchable=false`: chưa phải SQLite/snapshot cho Online.
4. Calibrate và materialize trên CPU từ chính result Gate B cùng ground-truth:

```powershell
python -m scripts.calibrate_ocr_vintern_gate_b `
  --easyocr-jsonl "D:\gate-b\easyocr-frames.jsonl" `
  --vintern-results-jsonl "D:\gate-b\vintern-results.jsonl" `
  --ground-truth "D:\gate-a\ground-truth.csv" `
  --calibration-policy "configs\ocr_vintern_calibration_policy_emergency_98.json" `
  --output-calibration-json "D:\gate-b\vintern-calibration.json" `
  --output-materialized-jsonl "D:\gate-b\ocr-calibrated-frames.jsonl" `
  --output-audit-jsonl "D:\gate-b\vintern-overrides-audit.jsonl" `
  --output-summary-json "D:\gate-b\vintern-calibration-summary.json"
```

   Ground-truth phải có `keyframe_uid`, `candidate_id`/`region_id`, `label_status`,
   `ground_truth_is_empty`, `human_text`; pool giữ 100 row cân bằng 20/video nhưng deadline
   tier dùng 98 labeled frame/region và đúng hai `exclude_unreadable`. Policy schema v3 tại
   `configs/ocr_vintern_calibration_policy_emergency_98.json`: support tối thiểu 20 cho fine/structural;
   global bucket chỉ báo cáo và không bao giờ cho phép override. Audit ghi cả candidate không
   override và cờ/reason Gemini residual. Rule duy nhất:
   guard PASS và `vintern_confidence_calibrated > easyocr_confidence_old`; field nguồn nằm
   trong JSONL/envelope, không thêm cột vào `ocr_fts`.

   Snapshot calibrated được build bằng:

```powershell
python -m scripts.build_ocr_snapshot `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --source-format gate2_calibrated_dev_v1 `
  --source-jsonl "D:\gate-b\ocr-calibrated-frames.jsonl" `
  --calibration-json "D:\gate-b\vintern-calibration.json" `
  --override-audit-jsonl "D:\gate-b\vintern-overrides-audit.jsonl" `
  --parent-snapshot-id "ocr-snapshot-..."
```

5. Trước Gemini, có thể chạy audit-only cho logo/đồng hồ kênh lặp ở góc trên-phải:

```powershell
python -m scripts.audit_ocr_overlay_residuals `
  --calibrated-frames-jsonl "D:\gate-b\ocr-calibrated-frames.jsonl" `
  --ground-truth-jsonl "D:\gate-b\ground-truth-repaired.jsonl" `
  --policy "configs\ocr_low_information_overlay_audit_policy.json" `
  --output-report-json "D:\gate-b\overlay-audit.json" `
  --output-residual-jsonl "D:\gate-b\gemini-residual-overlay-audit.jsonl"
```

   Đây chỉ là simulation: không xóa text/bbox, không sửa router production và luôn trả
   decision `AUDIT_ONLY_DO_NOT_APPLY_TO_PRODUCTION`. Subtitle/ticker đáy màn hình được bảo
   vệ. Chỉ sau khi evidence đa video được duyệt mới được nâng policy thành execution rule.

6. EasyOCR/Vintern notebook tự push archive/checksum/manifest đã PASS vào namespace layer
   riêng dưới `ocr/archives/{batch_id}/` và verify checksum remote. Sau đủ chín batch, local
   `snapshot_download()` một lần, join EasyOCR candidate với Vintern result theo
   `candidate_id`, materialize calibration rồi validate union UID. Bốn Kaggle worker không
   ghi chung SQLite.
7. Sau khi đủ chín archive EasyOCR đã verify trên HF, máy local CPU tạo báo cáo exact và
   bundle Gemini mà không gọi API. Archive Vintern là tùy chọn theo từng batch: có thì dùng
   calibrated override, thiếu thì candidate router v2 đi thẳng Gemini:

```powershell
python -m scripts.prepare_ocr_gemini_production `
  --output-dir "$env:AIC_DATA\ocr\gemini-preflight-20260828"
```

   Nếu không truyền `--artifact-root`, script dùng `HF_TOKEN`, resolve một HF revision rồi
   tải đủ EasyOCR và mọi Vintern archive hiện có. Nó verify ZIP/SHA/manifest/source-chain,
   áp table calibration 98-label đã pin cho batch có Vintern tại
   `configs/ocr_vintern_calibration_table_emergency_98.json`, materialize override và xuất:

   - `ocr-gemini-preflight-report.json`: exact residual region/frame/shot/request/contact
     sheet theo batch, reason và planning cost;
   - `gemini-residual-regions.jsonl`: mapping region → UID/source/bbox;
   - `gemini-request-manifest.jsonl`: một request/shot; shot dày dùng nhiều contact-sheet
     image part nhưng vẫn một request;
   - `ocr-gemini-preflight.zip`: đúng ba file trên + `SHA256SUMS`, không chứa media.

   Token/cost trong report là planning estimate, không được gọi là billing fact. Công thức
   pin Gemini 2.5 MEDIUM ≈256 token/contact sheet và snapshot giá official ngày 28/08/2026:
   Standard `$0.10/M input + $0.40/M output`, Batch `$0.05/M + $0.20/M`, reserve 15%.
   Exact token/model version chỉ có sau paid canary trả `usageMetadata`.

   Evidence production đã chốt tạm ngày 28/08/2026: 9/9 archive EasyOCR, 293.336 frame;
   không có Vintern và không gọi Gemini. Preflight SHA
   `f48b490d74bc043ebf1e7c14c1ba51fcf02d773b0779ea7c269bf17efef8cb55` đếm exact
   830.301 residual region, 253.177 frame, 92.768 request và 106.183 contact-sheet. Estimate
   Standard +15% là 651.803 VND; Batch +15% là 325.902 VND. Runner notebook hiện tại gọi
   Standard đồng bộ, vì vậy không được chạy full catalog dựa trên estimate Batch.

8. Upload `ocr-gemini-preflight.zip` làm Kaggle input cùng Dataset keyframe và mở
   `scripts/kaggle_ocr_production_gemini.ipynb` trên **CPU-only**. Lần đầu giữ
   `EXECUTION_MODE='preflight'`: notebook không đọc `GEMINI_API_KEY`, chỉ verify và in report
   SHA/count/cost. Sau khi người dùng duyệt canary, điền exact report SHA, budget canary,
   đặt `EXECUTION_MODE='canary'`, `APPROVE_PAID_CANARY=True`, tối đa 100 request. Canary
   phải HTTP 200, schema-valid và chỉ có một `model_version`; tải
   `ocr-gemini-paid-canary-report.json` về để duyệt.

9. Chỉ sau lần duyệt thứ hai mới đặt `EXECUTION_MODE='production'`, pin
   `APPROVED_MODEL_VERSION`, exact `APPROVED_MAX_REQUESTS`, `APPROVED_MAX_VND<=400000` và
   `APPROVE_GEMINI_PRODUCTION=True`. `REQUESTS_PER_MINUTE` lấy từ console paid project thật,
   không đoán. Runtime dùng strict `responseJsonSchema` exact-set string `region_id`,
   `MEDIA_RESOLUTION_MEDIUM`, retry `429/5xx` với backoff, fail ngay `401/403`, ledger token/
   VND và resume `request_id`. Gemini không sinh/sửa bbox. Chỉ batch exact request-set,
   `error=0`, đúng canary model version mới upload
   `ocr/archives/{batch_id}/gemini/` và verify round-trip SHA. Sau terminal union mới
   build/query SQLite FTS5.

Không commit weight, ảnh, secret, JSONL, archive hoặc canary output. Không đánh dấu production
complete khi còn artifact chưa push/verify HF, residual chưa có quyết định hoặc SQLite chưa
build/validate.

### Handoff tạm cho Nhánh 2

Trong lúc Vintern/Gemini chưa chạy, Nhánh 2 được dùng đủ chín archive EasyOCR trên private HF
Dataset để build snapshot bằng `scripts.build_ocr_incremental_snapshot`. Snapshot phải bất
biến và kèm `coverage.json` + `SHA256SUMS`; mọi batch ghi tier `easyocr`, toàn snapshot ghi
`complete=false` và `production_ready=false`. Không copy JSONL/ZIP vào Git và không đặt tên
snapshot này là `final`. Khi có tầng mới, tạo snapshot version mới thay vì ghi đè bản cũ.

Sau khi build/validate, Online phải trỏ đúng thư mục bất biến, không trỏ thẳng một file lẻ:

```powershell
$env:AIC_OCR_SNAPSHOT_DIR = "$env:AIC_DATA\ocr\snapshots\ocr-snapshot-<UTC>-<hash>"
python -m streamlit run online\streamlit_app.py
```

Mỗi lần đổi `snapshot_id` phải restart Streamlit. Registry fail closed nếu checksum,
`catalog_sha256`, UID-set, FTS schema/count hoặc join video/UID sai; snapshot EasyOCR dù phủ
đủ UID vẫn phải hiện `online_development_only`, tier/error count và
`production_ready=false` cho đến terminal union cuối.
