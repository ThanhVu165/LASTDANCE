# OCR v2 — đồng bộ, union và build SQLite development ở local

Nguồn contract duy nhất là [`BASELINE_SPEC.md`](BASELINE_SPEC.md) §2.2c. Bước này chạy
**CPU local**, không dùng Kaggle GPU và không gọi Gemini/API. Input là đúng chín ZIP
production của recognition worker, catalog `frames.csv` + state và worker plan bất biến;
output là snapshot development versioned, không thay `AIC_DATA/index/ocr.sqlite` hoặc
snapshot EasyOCR Online đang dùng.

**Nhóm nhận bàn giao đọc [OCR_V2_ONLINE_HANDOFF.md](OCR_V2_ONLINE_HANDOFF.md) trước** để có
link tải từng input, revision/hash đã pin, setup CPU tối thiểu và checklist adapter Online.

## 1. Input và output

Input bắt buộc:

- `frames.csv` và `frames.csv.state.json` đã validate;
- `ocr-v2-worker-plan.json` đã tạo ở Kaggle planner (dataset hiện hành:
  `thvu165/ocr-v2-worker-plan`);
- bốn run ID production đã pin trong
  [`configs/ocr_v2_production_run_ids.json`](../configs/ocr_v2_production_run_ids.json);
- quyền **read** private HF Dataset `MinhThuw0103/lastdance-visual-embeddings`.

Output đồng bộ nằm mặc định ở `$AIC_DATA/ocr/v2-production/`: chín ZIP, chín summary và
`ocr-v2-production-sources-<hash>.json` pin đúng HF revision + content hash. Output build:

```text
$AIC_DATA/ocr/snapshots/ocr-snapshot-<UTC>-<source-hash>/
  ocr.sqlite
  coverage.json
  SHA256SUMS
```

`coverage.json` dùng schema 3 riêng cho OCR v2, giữ engine region thật
`vietocr|paddle|unresolved`. Nó luôn ghi `immutable=true`, `complete=false`,
`production_ready=false`, `intended_use=online_development_only`; không ép output mới vào
engine/tier EasyOCR/Vintern cũ. `shared/schemas/ocr.py` và năm cột `ocr_fts` không đổi.

## 2. Setup và đồng bộ HF

Dùng local Python 3.11 (`requirements/ocr-v2-artifacts.txt`, hoặc môi trường offline đã có). Không dán token vào
notebook, source, command history hoặc log. Trên máy này đăng nhập lại nếu `hf auth whoami`
báo token cũ không hợp lệ:

```powershell
hf auth login --force
hf auth whoami
```

Đặt root dữ liệu và đường dẫn thật của hai input local:

```powershell
$env:AIC_DATA = "D:\path\to\aic-data"
$catalog = "$env:AIC_DATA\catalog\frames.csv"
$plan = "$env:AIC_DATA\ocr\ocr-v2-worker-plan.json"
```

Đồng bộ theo revision HF bất biến và tạo source manifest:

```powershell
python -m scripts.sync_ocr_v2_results `
  --worker-plan $plan `
  --run-ids configs/ocr_v2_production_run_ids.json `
  --revision 8ca4271dd0218d3f3f3967a4d8a5c6aeebeaddc5
```

Script yêu cầu ít nhất một cặp `results-*.zip` + `reports/summary-*.json` trong namespace
production của mỗi batch/run ID. Với nhiều export do resume, script hash toàn bộ member;
chỉ khi `run-signature`, prediction, frame selection và residual giống hệt nhau mới chọn
bản mới nhất theo commit HF. Source manifest ghi cả commit được chọn và hash các cặp tương
đương bị loại. Candidate khác nội dung, summary mồ côi hoặc thiếu cặp đều làm sync dừng.
Script còn kiểm tra path content-addressed, hash tải xuống, summary khớp report trong ZIP,
worker/batch/run/signature và các cờ readiness. Ghi lại đường dẫn `source_manifest` được in
ở cuối lệnh.

## 3. Union và atomic-build SQLite

```powershell
$sources = "$env:AIC_DATA\ocr\v2-production\ocr-v2-production-sources-<hash>.json"

python -m scripts.build_ocr_v2_snapshot `
  --catalog $catalog `
  --worker-plan $plan `
  --source-manifest $sources
```

Builder fail closed trước khi publish thư mục snapshot:

- kiểm checksum mọi member trong từng ZIP và summary ngoài ZIP;
- kiểm run ID từ resources, signature từ run/batch/tasks, worker plan và catalog hash;
- kiểm raw prediction `(model, region_id, task_sha256, signature)` không thiếu/trùng/foreign;
- kiểm selection/result/residual, engine thật, status và report count;
- kiểm chín tập UID disjoint/exhaustive đúng 293.336 UID của `frames.csv`;
- chỉ insert frame `success`, giữ `no_text/error/residual` trong coverage;
- chạy SQLite `integrity_check`, exact row count và FTS probe rồi mới atomic-rename.

Nếu sync/build bị dừng, chạy lại cùng lệnh. File HF đã tải được cache/reuse; snapshot tạm
không được publish. Mỗi lần build thành công tạo thư mục immutable mới; không xóa hoặc ghi
đè snapshot trước để “resume”.

## 4. Validate độc lập và bàn giao

```powershell
$snapshot = "$env:AIC_DATA\ocr\snapshots\ocr-snapshot-<UTC>-<source-hash>"

python -m scripts.validate_ocr_v2_snapshot `
  --snapshot $snapshot `
  --catalog $catalog
```

Validator đọc-only kiểm `SHA256SUMS`, manifest schema 3, SQLite integrity/schema năm cột,
UID/video join với catalog, confidence/text/language, FTS probe và tổng coverage
batch/video. `recognition_coverage_complete=true` chỉ có nghĩa chín shard phủ đủ catalog;
không đồng nghĩa accuracy gate hoặc Publishing Ready. Error/residual được giữ nguyên để
quyết định bước Gemini/review sau, không đổi thành `no_text`.

Chưa đổi `AIC_OCR_SNAPSHOT_DIR` trong bước này. Việc kiểm consumer hoặc chuyển Online sang
snapshot v2 cần yêu cầu phạm vi Nhánh 2 riêng. Không commit ZIP, JSONL, source manifest,
SQLite, token, cache HF hoặc thư mục `$AIC_DATA`; commit/push source code cũng cần người dùng
yêu cầu riêng.

## 5. Test không dùng GPU/network

```powershell
python -m unittest tests.test_ocr_v2_snapshot -v
python -m unittest discover -s tests -q
```

Fixture tạo đủ chín batch tổng hợp, gồm VietOCR, Paddle, `no_text`, `error` và residual;
không tải model, không truy cập HF và không thay artifact local đang dùng.

## 6. Blind audit chất lượng output production — tùy chọn, đã hoãn

`recognition_complete=true` chỉ chứng minh mọi task model đã chạy; không phải accuracy.
Ngày 04/09/2026 người dùng yêu cầu bỏ qua chấm ground truth để bàn giao code/development
snapshot cho nhóm Online kịp thi. Không chặn commit/push vì thiếu nhãn; không gọi đây là
quality PASS hoặc đổi cờ readiness. Protocol bên dưới giữ để tái lập khi có thời gian.
Nếu thực hiện audit, tạo từ chính chín ZIP production đã pin. Lệnh
này chạy CPU local, tải duy nhất các JPEG được lấy mẫu từ Kaggle public Dataset và kiểm
SHA-256 ảnh theo task production; không chạy lại EasyOCR/VietOCR/Paddle và không dùng GPU:

```powershell
$review = "$env:AIC_DATA\ocr\quality-review\production-v2-20260904"

python -m scripts.ocr_v2_production_quality build `
  --source-manifest $sources `
  --output-dir $review `
  --sample-size 180
```

Mẫu chia đều 20 region cho mỗi batch và stratify theo output cuối:
`unresolved`, `paddle`, `vietocr_changed`, `vietocr_agree`. Mở
`blind-crop-sheets.zip`; sheet không hiển thị text từ model để tránh bias. Chỉ sửa bốn cột
cuối trong `ground-truth.csv`:

- `label_status=labeled`: điền phiên âm nhìn thấy chính xác vào `human_text`, cùng
  `text_type=ordinary|ticker|numeric_or_name|other`;
- `label_status=exclude_unreadable`: con người thực sự không đọc được, để text/type trống;
- `label_status=false_positive`: bbox không chứa chữ thật, để text/type trống;
- `notes` tùy chọn. Không sửa ID, sheet mapping hoặc `sample_row_sha256`.

Cần đủ 180 quyết định, ít nhất 150 crop readable và ít nhất năm crop số/tên. Sau đó chấm
EasyOCR cache, raw VietOCR, Paddle ở tập conditional đã gọi và quan trọng nhất là
`production_selected`:

```powershell
python -m scripts.ocr_v2_production_quality score `
  --review-dir $review `
  --output "$review\quality-score.json"
```

`PASS_BLIND_SAMPLE` cần output chọn cuối cải thiện ít nhất 5 điểm phần trăm exact-token
recall **hoặc** giảm CER tương đối ít nhất 10% so với EasyOCR cache, đồng thời độ chính xác
số/tên không giảm quá 2 điểm phần trăm. Đây là difficult-strata blind audit, không được gọi
là ước lượng accuracy ngẫu nhiên toàn catalog. `machine-results.jsonl` phải giữ kín cho tới
khi ground truth hoàn tất. Dù sample PASS, các frame `error`, region `unresolved`, consumer
schema và Publishing Criteria vẫn là gate độc lập trước khi `production_ready=true`.
