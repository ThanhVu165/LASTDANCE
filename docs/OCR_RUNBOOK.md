# OCR runbook — Gemini primary, EasyOCR offline fallback

Nguồn chuẩn kỹ thuật là `docs/BASELINE_SPEC.md` §2.1d/§2.2. Runbook này chỉ mô tả cách
vận hành Gate 1–production; không thay schema `OcrResult` hoặc `ocr_fts`.

## Gate 1 — preflight bắt buộc

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

### 2. Gemini project/tier/quota

Tier là thuộc tính project/billing account, không suy ra từ API key. Bằng chứng Gate 1 phải
ghi cùng timestamp:

1. Google AI Studio Projects/Billing: project ID, Plan/Tier và trạng thái billing/credit.
2. Dashboard > Usage/Rate limits: RPM, input TPM, RPD đang active cho đúng model ID dự kiến
   production và đúng project. Baseline hiện ghi `gemini-2.5-flash-lite`, nhưng nếu project
   không còn quyền truy cập thì phải dừng và xin duyệt đổi model; không lấy canary của model
   khác để thay bằng chứng.
3. Canary bằng chính key của project đó trên một tập ảnh nhỏ: số request, concurrency,
   latency p50/p95, input/output token và toàn bộ 429/5xx/timeout. Không log API key.

Runtime lower-bound chỉ được tính sau ba bằng chứng trên:

```text
max(N / RPM, input_tokens / TPM, ceil(N / RPD) * 1440 phút)
```

Sau đó so với throughput đo ở canary; lấy thời gian lớn hơn và cộng margin retry. `401/403`
hoặc quota/global project failure dừng batch, không kích hoạt EasyOCR hàng loạt.

### 3. EasyOCR offline weight pin

Registry committed: `configs/ocr_easyocr_models.json`. Tải package/ZIP trước phiên chạy,
extract đúng hai `.pth` vào một model storage directory, rồi chạy:

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

- Canonical line model: `offline.ocr_artifacts.OcrRecordEnvelope`; `execution_mode` phân biệt
  pipeline `gemini_primary` với job backup chủ động `easyocr_offline`.
- Shard manifest: `offline.ocr_artifacts.OcrShardManifest`.
- Mỗi batch có một append-safe JSONL shard. Khi resume, JSONL hiện có là authority; đọc lại,
  validate từng line và reject duplicate UID trước khi tiếp tục. State/manifest được ghi
  atomic sau shard, không dùng `next_index` đơn lẻ làm authority.
- `success` có `OcrResult`; `no_text`/`error` có `result=null`. Error provenance phải sanitize,
  tuyệt đối không chứa API key/header/request body nhạy cảm.
- Batch gate PASS khi `success + no_text == expected`, `error == 0`, không duplicate,
  missing hoặc foreign UID. Đây chỉ là OCR-stage completion; không tự nâng publishing Ready
  toàn pipeline khi EVA-CLIP/FAISS còn thiếu.

## Gate 2 và production

Catalog, quota và strict-schema canary đã có bằng chứng thật, nhưng Gate 1 vẫn chưa PASS vì
Free tier hiện tại không đủ năng lực production. Chưa chạy Gate 2 cho tới khi phương án
model/throughput/cost được duyệt và canary lại đúng cấu hình đó. Gate 2 phải dùng đúng 5 video
`L21_V001`, `L21_V002`, `L21_V003`, `L21_V005`, `L21_V006`, demo interrupt → process mới
resume → validate, và test fallback bằng response schema-invalid có kiểm soát. Fallback test
không được giả global 401/403 thành per-frame fallback.
