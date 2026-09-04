# Bàn giao OCR v2 cho nhóm Online — đọc file này trước

Cập nhật 04/09/2026. Contract duy nhất: [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2c.
Nhánh code: `codex/ocr-v2-experiment`. Recognition **9/9 batch đã xong**, snapshot local
đã validate. Không chạy lại notebook OCR, EasyOCR, VietOCR hoặc Paddle để nhận kết quả.

Người dùng quyết định **bỏ qua việc chấm ground truth trong đợt bàn giao này** để kịp thi.
Đó không phải accuracy PASS: chưa chứng minh OCR v2 tốt hơn bản cũ. Snapshot giữ
`complete=false`, `production_ready=false`, `intended_use=online_development_only`.

## 1. Nhận code và chọn cách nhận dữ liệu

Trong clone của mình, chạy `git fetch origin`. Nếu tạo checkout mới và working tree sạch:

```powershell
git switch --track origin/codex/ocr-v2-experiment
```

Nếu đang làm trên nhánh Online riêng, giữ nguyên thay đổi và tích hợp nhánh này theo quy
trình merge của nhóm; không reset/ghi đè working tree. Commit này không sửa `online/`,
`app/` hay `shared/schemas/`; việc tích hợp consumer bên dưới thuộc nhóm Online.

**GitHub chỉ có code/config/docs/tests/notebook.** ZIP, SQLite, catalog, plan và token
không nằm trong Git. Có hai cách nhận dữ liệu:

- **Nhanh nhất:** nhận từ người bàn giao cả thư mục
  `ocr-snapshot-20260904T081629Z-66ecea73cce1` gồm đúng `ocr.sqlite`, `coverage.json`,
  `SHA256SUMS`, cùng catalog/state. Copy nguyên thư mục vào `$AIC_DATA/ocr/snapshots/`,
  giữ tên, không sửa file. Làm setup ở §2 rồi validate ở §4; bỏ qua sync/build §3.
- **Tự tái lập từ HF:** lấy ba input nhỏ ở §2, rồi chạy sync/build §3 trên CPU.
  Chỉ cần quyền **read** HF, không cần GPU hoặc quyền write.

**Snapshot SQLite này hiện chỉ có ở máy người bàn giao, chưa upload lên HF.** Vì vậy
clone/pull Git không tự tải được SQLite. Cách tự phục vụ hoàn chỉnh là tải raw results từ
HF rồi build. Không dùng script publish snapshot legacy cho schema v3.

## 2. Setup local CPU và lấy catalog/plan

Dùng Python **3.11** và environment riêng, từ thư mục gốc repo. Ví dụ tạo venv trong
`tmp/` (đã gitignore); nếu đã có môi trường offline phù hợp thì dùng môi trường đó:

```powershell
py -3.11 -m venv tmp/ocr-handoff-venv
& ./tmp/ocr-handoff-venv/Scripts/Activate.ps1
python -m pip install -r requirements/ocr-v2-artifacts.txt
$env:AIC_DATA = Join-Path (Get-Location) "data"
$catalog = Join-Path $env:AIC_DATA "catalog/frames.csv"
$plan = Join-Path $env:AIC_DATA "ocr/ocr-v2-worker-plan.json"
```

Không cần cài profile OCR Kaggle hoặc model weights. Nếu Windows chưa có Python 3.11,
xem [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md); không đổi runtime Online đang chạy.

Tải **từng file**, không cần tải toàn bộ JPEG dataset:

| Nguồn Kaggle | File cần lấy | Nơi đặt dưới `$AIC_DATA` |
|---|---|---|
| [aic-2026-keyframes](https://www.kaggle.com/datasets/thvu165/aic-2026-keyframes) | `frames.csv` | `catalog/frames.csv` |
| Cùng dataset trên | `frames.csv.state.json` | `catalog/frames.csv.state.json` |
| [ocr-v2-worker-plan](https://www.kaggle.com/datasets/thvu165/ocr-v2-worker-plan) | `ocr-v2-worker-plan.json` | `ocr/ocr-v2-worker-plan.json` |

Có thể nhận trực tiếp ba file này từ người bàn giao. Nếu Kaggle tải dạng ZIP thì giải nén
để lấy đúng file trước khi kiểm hash; không đổi tên ZIP thành CSV/JSON. Đối chiếu SHA-256:

```powershell
Get-FileHash $catalog, "$catalog.state.json", $plan -Algorithm SHA256
```

```text
frames.csv
ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37
frames.csv.state.json
6be1f3cf25629eed76264311af20ee2c0bb1b4d912580f1ee0ce3161aca5bf31
ocr-v2-worker-plan.json
4cd9ead3e34d8749accf78403555c30b804ebe1758bc7eb6b5d663c0e960f72f
```

Khác hash thì dừng và xin đúng bản, không sửa state/hash để vượt kiểm tra. Hash file plan
ở đây khác semantic `plan_sha256` trong JSON; không nhầm hai loại.

## 3. Tải đúng kết quả trên HF rồi build SQLite

Repo dữ liệu private:
[MinhThuw0103/lastdance-visual-embeddings — bản bàn giao](https://huggingface.co/datasets/MinhThuw0103/lastdance-visual-embeddings/tree/8ca4271dd0218d3f3f3967a4d8a5c6aeebeaddc5/ocr/archives).
Chủ repo phải cấp quyền đọc cho người nhận; không gửi token chung hoặc commit token.
Đăng nhập tương tác bằng token của mình, không dán token vào lệnh/source:

```powershell
hf auth login
hf auth whoami
python -m scripts.sync_ocr_v2_results `
  --worker-plan $plan `
  --run-ids configs/ocr_v2_production_run_ids.json `
  --revision 8ca4271dd0218d3f3f3967a4d8a5c6aeebeaddc5
```

Script chỉ tải result/summary đúng namespace production; không tải full HF repo,
checkpoint dở dang hay archive EasyOCR cũ. Chín batch được chia như sau:

| Worker | Batch |
|---|---|
| 1 | 01 |
| 2 | 05, 08 |
| 3 | 02, 04, 06 |
| 4 | 03, 07, 09 |

Có nhiều ZIP do resume không có nghĩa là chạy trùng: script so member hash và chỉ chọn
export tương đương mới nhất. Nếu khác nội dung/checksum/identity thì dừng, không tự bỏ lỗi.
401/403/404 ở repo private: kiểm tra quyền đọc và tài khoản `hf auth whoami` trước.

Cuối sync có JSON `source_manifest`. Gán `$sources` bằng **đường dẫn được in ra**, rồi build:

```powershell
$sources = Join-Path $env:AIC_DATA "ocr/v2-production/ocr-v2-production-sources-6807fbb667a5.json"
python -m scripts.build_ocr_v2_snapshot `
  --catalog $catalog `
  --worker-plan $plan `
  --source-manifest $sources
```

Tên source manifest trên là bản tham chiếu của revision đã pin; nếu output thực tế khác,
kiểm revision/input và dùng đúng đường dẫn script trả về, không đổi tên JSON bằng tay.
Build stream chín ZIP và validate prediction/selection/residual/UID; không gọi model.
Cuối lệnh in đường dẫn `snapshot` mới. Khi bị ngắt, chạy lại cùng lệnh sync/build: download
có cache, build tạo snapshot immutable mới, không ghi đè snapshot đang dùng.

## 4. Validate bản nhận được

Đặt `$snapshot` bằng đường dẫn build in ra, hoặc bản copy nguyên thư mục:

```powershell
$snapshot = Join-Path $env:AIC_DATA "ocr/snapshots/ocr-snapshot-20260904T081629Z-66ecea73cce1"
python -m scripts.validate_ocr_v2_snapshot --snapshot $snapshot --catalog $catalog
if ($LASTEXITCODE -ne 0) { throw "OCR v2 snapshot validation failed" }
```

Bản tham chiếu có `ocr.sqlite` 45.228.032 byte, SHA-256:
`9b80eed3ef376655b6a4ad6c9496072f2cf215dec38f5af9e5095ebb491ed78e`.
Bản tự build có ID/thời gian/checksum riêng; dùng validator và `SHA256SUMS` của chính nó,
không sửa manifest để giả danh snapshot tham chiếu.

Kết quả coverage mong đợi từ các source này:

| Chỉ số | Giá trị |
|---|---:|
| UID đã đối chiếu / catalog | 293.336 / 293.336 (873 video) |
| Success / FTS row | 269.259 |
| No text | 15.188 |
| Error | 8.889 |
| Residual frame / region | 240.976 / 763.395 |
| Region được chọn VietOCR / Paddle | 1.131.448 / 70.470 |
| Region unresolved (không đưa text vào FTS) | 501.523 |

100% UID coverage **không phải** 100% frame có text đúng. Một frame success có thể còn
region residual; residual và unresolved không phải cùng một tập. Không đổi error thành
no_text; không gán engine mới thành EasyOCR/Vintern để qua schema cũ.

## 5. Phần nhóm Online cần làm trước khi bật snapshot v2

Hiện `online/artifacts.py::_inspect_ocr` gọi validator legacy rồi parse
`OcrSnapshotManifest` schema 1/2; nó sẽ từ chối schema 3. FTS reader truy vấn trực tiếp DB
được, nhưng **chỉ đổi biến môi trường là chưa đủ**. Nhóm Online cần:

1. Dispatch theo `schema_version`/`source_format` cho **cả validator và parser**. Schema 3
   dùng `offline.ocr_v2_snapshot.validate_ocr_v2_snapshot` với catalog/state đúng, và
   `OcrV2SnapshotManifest`; giữ nguyên đường legacy v1/v2. Unknown schema phải INVALID.
2. Đối chiếu hash catalog, số frame/video, UID-set, checksum, SQLite integrity và FTS count;
   schema v3 có `totals.observed_uid_sha256`; tính coverage bằng
   `totals.processed_keyframes / totals.expected_keyframes` (guard mẫu số 0), và đọc
   `totals.error_keyframes`, `totals.missing_keyframes`. Không đọc các field đó như field
   top-level legacy, không giả `batch.tier` cho v3. Hiển thị snapshot ID, source format,
   engine provenance, coverage/errors/residual và `production_ready=false` trên status/UI.
3. Giữ đúng năm cột `ocr_fts`:
   `video_id, keyframe_uid, detected_text, language, confidence`. Join với catalog bằng
   UID/video để lấy `frame_id`/`pts_time`; khóa nộp bài là `(video_id, frame_id)`.
   Không có OCR model chạy trong Online; không dùng `local_idx` để nộp.
4. Thêm regression test: v1/v2 vẫn đọc được, v3 hợp lệ dùng được ở chế độ development;
   file bị sửa/checksum sai/catalog lệch/schema lạ phải INVALID, không fallback âm thầm.
   Smoke các query `giá dầu mazut`, `Việt Nam`, `Hà Nội`, `2026` rồi kiểm frame mapping.
   Smoke không phải accuracy benchmark.
5. Sau khi adapter + test PASS mới cấu hình và restart process Online:

   ```powershell
   $env:AIC_OCR_SNAPSHOT_DIR = $snapshot
   ```

Giữ snapshot cũ để rollback:
`ocr-snapshot-20260828T153736Z-65e6f8bf8850`. Muốn rollback thì trỏ biến môi trường về
thư mục cũ, restart và validate lại; không overwrite `ocr.sqlite` final hoặc sửa checksum.

## 6. Kiểm thử và phạm vi đã xác minh

```powershell
python -m unittest tests.test_ocr_v2_snapshot -v
```

Profile artifact tối thiểu chỉ dành cho sync/build/validate và test snapshot. Chạy toàn bộ
repo trong môi trường dev/offline đã có dependency theo [requirements](../requirements/README.md):

```powershell
python -m unittest discover -s tests -q
```

Trước commit bàn giao: **305/305 test PASS trên Windows Python 3.11.9**; fixture Gate A
CSV đã sửa đúng cột `video_id`, review notebook đã regenerate từ source hiện hành.
Snapshot thật đã qua validator độc lập. Model T4/HF evidence đến từ production đã chạy,
không chạy lại GPU trong đợt commit này. Adapter/activation Online **chưa triển khai**.
Profile artifact tối thiểu đã clean-install trong venv Python 3.11.9, chạy 3/3 test
snapshot và validate SQLite thật thành công; không cần dependency model từ offline profile.
Ground-truth accuracy **chưa kiểm định**, được người dùng hoãn, không phải gate PASS.

Không commit/push `data/`, ZIP/JSONL/SQLite, ảnh audit, HF cache, token hoặc môi trường Python.
Nếu nhóm muốn URL tải thẳng bộ SQLite, cần người bàn giao publish riêng cả ba file qua
kênh dữ liệu được duyệt; commit Git này không thực hiện việc đó.
