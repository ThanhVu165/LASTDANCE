# OCR v2 — chạy recognition worker trên Kaggle T4

Contract chuẩn: [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2. Đây là lát cắt triển khai
**planner + recognition/selection + checkpoint**, không phải toàn bộ handoff SQLite.
Notebook: `notebooks/kaggle_ocr_v2_production.ipynb`.

Đã có test CPU bằng predictor/HF giả và **recognition thật 9/9 batch trên bốn Kaggle T4,
result/report đã HF verified**. Nhận kết quả theo [bàn giao Online](OCR_V2_ONLINE_HANDOFF.md),
không chạy lại worker chỉ để tải dữ liệu. Không gọi kết quả này là production-ready. Không sửa hay cần chạy lại Gate A,
Gate B hoặc sharpening notebook. Không gọi API trả phí, EasyOCR hoặc Vintern.

## 1. Tạo plan một lần, không dùng GPU

Import notebook mới vào một notebook Kaggle **CPU**, bật Internet; thêm Secret `HF_TOKEN`
có quyền đọc/ghi dataset private. Giữ `ACTION='plan'`, repo mặc định theo baseline.
Gắn dataset Kaggle có catalog **đầy đủ** `frames.csv` và `frames.csv.state.json` bên cạnh.
Catalog theo `FrameRecord` trong `shared/schemas/frame.py`; không dùng CSV subset từng batch.
Keyframes/catalog có sẵn trên Kaggle; HF lưu artifact OCR/embedding, không phải nguồn catalog.

- `CATALOG_PATH=''` tự tìm `frames.csv` dưới `INPUT_ROOT` (mặc định `/kaggle/input`).
  Nhiều bản chỉ được tự chọn khi **CSV và state giống nhau từng byte**; khác nhau thì dừng,
  in candidate để điền `CATALOG_PATH` bằng exact path Kaggle. Thiếu state thì gắn bản gốc;
  không tạo state giả hoặc bỏ kiểm tra `complete`/checksum.
- Planner resolve một HF commit cho chín archive rồi pin. Plan ghi
  `catalog_source=kaggle_input`, tên file catalog và SHA-256 của CSV/state, không khóa
  đường dẫn mount của một tài khoản. Worker đọc catalog local rồi kiểm tra cả hai hash.
- Planner tải và validate **chín archive một lần trên máy tạo plan**, kiểm tra checksum,
  số region thật, UID/video disjoint và union bằng catalog. Không có JPEG/model inference.
  Các worker sau đó chỉ tải archive được giao, không tải lại đủ chín archive trên mỗi tài khoản.
- Log `PLAN_COMPLETE` cho biết batch/region của worker 1–4. Tải
  `/kaggle/working/ocr-v2-production/ocr-v2-worker-plan.json` (default, hoặc dưới `AIC_DATA`).
  Gắn **cùng file này** vào bốn worker bằng Kaggle Input. Không tạo bốn plan trôi revision.
- Planner không tự upload/commit plan lên HF; người dùng bàn giao file plan cho các worker.

Đây là bước cần chạy đầu tiên để biết chính xác mỗi tài khoản phải gắn batch JPEG nào.
Không yêu cầu upload lại các ZIP `ocr-production-batch-0X-easyocr.zip` đã có trên HF.

Nếu cần xem đường dẫn catalog trong notebook (chỉ đọc, không GPU/HF):

```python
from pathlib import Path
for path in sorted(Path(INPUT_ROOT).rglob('frames.csv')):
    print(str(path), '| state:', path.with_name(path.name + '.state.json').is_file())
```

Nếu notebook cũ báo `Set CATALOG_HF_PATH`, import bản mới
`notebooks/kaggle_ocr_v2_production.ipynb` và chạy lại các cell từ đầu với `ACTION='plan'`.
Chỉ sửa parameters trong notebook cũ không đủ vì cell source nhúng cả runtime.
Plan kiểu cũ đọc catalog HF phải tạo lại bằng bản mới; không sửa JSON/hash bằng tay.

## 2. Canary + ngắt/resume trên mỗi worker

Mở notebook mới trên từng tài khoản, chọn **GPU T4**, Internet ON, Secret `HF_TOKEN`.
Gắn plan, cùng catalog/state và JPEG datasets theo phân công. Canary chỉ dùng batch đầu của worker đó;
production cần đủ JPEG của tất cả batch được giao.

```python
ACTION = 'run'
WORKER_SLOT = 1  # các tài khoản khác dùng 2, 3, 4; không chạy trùng
RUN_MODE = 'canary'
INTERRUPT_AFTER_MINIBATCHES = 1
APPROVED_CANARY_SHA256 = ''
RUN_SETUP = True
```

`WORKER_PLAN=''` tự tìm file `ocr-v2-worker-plan.json` duy nhất dưới `/kaggle/input`;
nếu cần thì điền exact path trong Kaggle. `KEYFRAMES_ROOT` mặc định `/kaggle/input`.
`CATALOG_PATH` có thể khác giữa tài khoản/session; nội dung CSV/state phải khớp hash plan.
Không đổi `VIETOCR/PADDLE` batch 64/128 bằng sửa code giữa một run; runner tự giảm khi OOM.

Setup kế thừa pin Gate B (VietOCR 0.3.13, Paddle GPU 3.2.2, PaddleOCR 3.7.0), kiểm tra
checksum wheel VietOCR, cài Python dependency sau dry-run với constraint giữ Torch,
Torchvision, Torchaudio, NVIDIA, NumPy và Pillow. Probe Torch/VietOCR và Paddle trong
process riêng. Lỗi resolver/probe phải sửa trước inference; không cài đè bộ CUDA.
Nếu đã có đúng environment, có thể đặt `RUN_SETUP=False`; runtime vẫn kiểm tra version.

Canary chọn deterministic **256 region từ batch đầu của worker**, không benchmark 5.000.
VietOCR đọc cả 256; Paddle chỉ đọc candidate, nên tổng call không cố định 512.
Đây là kiểm tra kỹ thuật, không thay bộ nhãn/chất lượng Gate B và không đại diện accuracy.

Lần đầu sẽ thấy `MINIBATCH_SAVED` → `HF_VERIFIED` → `INTENTIONAL_STOP` sau một minibatch.
Đây là dừng có chủ đích, không phải treo/lỗi CUDA. Sau đó:

```python
INTERRUPT_AFTER_MINIBATCHES = 0
```

Chạy lại cell launch: nó tạo process mới và restore checkpoint HF, skip phần đã lưu,
tiếp tục VietOCR rồi Paddle ở process riêng. Muốn chứng minh mất VM thì mở session mới,
gắn lại catalog/state/JPEG/plan và dùng cùng notebook/plan/worker/config; không cần file checkpoint local.
Không đổi source notebook trong lúc resume: code và package version nằm trong signature.

Kết quả in `report_sha256`, ZIP path và các count. Canary chỉ đủ điều kiện mở recognition
full khi `recognition_complete=true` và `resume_with_new_work=true`; không có nghĩa quality
PASS, đạt deadline hoặc SQLite-ready. Nếu chạy hết canary không ngắt thì không có chứng cứ
resume/new-work; không sửa report/flag để vượt gate.

## 3. Mở recognition đủ batch sau khi review canary

Copy **hash report của đúng worker** đã chạy canary:

```python
RUN_MODE = 'production'
INTERRUPT_AFTER_MINIBATCHES = 0
APPROVED_CANARY_SHA256 = '<report_sha256 được in từ canary của worker này>'
```

Runner lấy report đã verify từ HF, kiểm tra cùng code/config/environment/plan/worker,
đúng sample image hash và có resume với phần việc mới. Sai hash/missing report sẽ dừng.
Không nhập hash worker khác. Không chạy hai session cùng slot/run đồng thời.

Không cần đợi worker khác kết thúc recognition. Mỗi worker xử lý nguyên batch được giao;
canary và production có namespace/checkpoint khác nhau, không append mẫu thử vào full run.
Phải kiểm tra ETA thực tế từ log; elapsed của lần resume chỉ bao gồm phần chạy lại hiện tại,
không lấy con số đó chia cho tổng 256 prediction để suy throughput toàn run.

## 4. Log, checkpoint và sự cố

- `ReadTimeoutError` khi tải `paddlepaddle-gpu==3.2.2` từ CDN: timeout chờ mạng của
  bước này là 120 giây, retries 3; giới hạn cả lệnh vẫn 1.200 giây. Đây không phải
  resume tải wheel hoặc bảo đảm tốc độ CDN. Giữ `RUN_SETUP=True` tới khi có `ENV_READY`.
  Notebook cũ có thể vá sau cell SOURCES bằng:
  `SOURCES['ocr_v2_environment.py'] = SOURCES['ocr_v2_environment.py'].replace('"--timeout", "30", "--retries", "2"', '"--timeout", "120", "--retries", "3"')`.
  Chạy lại cell cuối trong cùng session; không cần đổi model, Torch/CUDA hay tạo lại plan.
- `ENV_READY` rồi `FileNotFoundError` tại `/kaggle/working/.../ocr-v2-worker-plan.json`:
  setup đã qua, nhưng cấu hình còn trỏ file plan của session tạo plan. Gắn cùng plan vào
  Kaggle Input, đặt `WORKER_PLAN=''` hoặc exact path của file Input rồi chạy lại. Trong
  **cùng session đã có `ENV_READY`**, đặt `RUN_SETUP=False`; session mới giữ `True`.
  Launcher mới kiểm tra file plan trước setup. Không đổi về `ACTION='plan'` để che lỗi
  ở worker và không bỏ qua kiểm tra signature của plan.
- Cell launch in ngay `[START]`/`[DONE]` cho 5 bước: ghi runtime, tìm plan, đọc Secret,
  setup môi trường, chạy planner/worker. Trong lúc chờ có `[WAIT]` mỗi 20 giây và elapsed;
  đây là thời gian chờ, không phải phần trăm hoàn thành hoặc ETA. stdout/stderr subprocess
  được chuyển trực tiếp vào output notebook, Python chạy unbuffered. Lỗi in `[STOP]` và
  giữ exit code; Interrupt dừng nhóm tiến trình con trên Kaggle Linux.
- Notebook cũ đang im lặng: dừng cell đang chạy trước, thay **cell cuối** bằng nội dung
  `scripts/kaggle_ocr_v2_production_launch.py`, giữ cell cấu hình và `SOURCES`, rồi chạy lại.
  File này là mã cell dùng biến notebook, không phải CLI standalone. Chỉ sửa launcher
  không đổi runtime/model signature; dùng đúng `RUN_SETUP` theo trạng thái setup thực tế.
  Bản notebook regenerate đã nhúng cell mới. Log không in token hoặc toàn bộ environment.
- `HEARTBEAT` tối đa 30 giây khi download/validate/setup/init/inference/sync; heartbeat chỉ
  báo phase còn hoạt động. `MINIBATCH_SAVED` có worker, batch, video, done/total, tốc độ,
  ETA phần recognition đang chạy và hash HF cuối.
- SQLite **checkpoint nội bộ từng batch** dùng transaction + synchronous FULL sau mỗi
  minibatch; đây không phải `ocr.sqlite` FTS dùng chung. Không ghi chung giữa worker.
- HF chỉ upload **delta prediction mới** trong các ZIP có sequence, previous hash,
  signature và checksum, thay vì upload lại toàn bộ lịch sử mỗi lần. Sync tối đa 5 phút
  giữa mốc kiểm tra minibatch, cuối pha và trước intentional stop. Model hang có thể trì
  hoãn sync; `CHECKPOINT_DUE` báo khi đã đến/vượt mốc ở lần kiểm tra kế tiếp.
- Chỉ checkpoint round-trip SHA hợp lệ là durable. Lỗi HF retry hữu hạn rồi dừng inference;
  giữ local DB, không tự tiếp tục hàng giờ mà không backup. Mất VM có thể phải chạy lại
  minibatch hoặc phần sau checkpoint HF cuối, không hứa zero rework.
- Resume kiểm tra chuỗi delta không gap/conflict, reject duplicate/foreign/signature drift,
  so local/remote overlap. Không chữa mismatch bằng xóa checkpoint hoặc bỏ validator.
- OOM giảm batch một nửa; size 1 vẫn OOM thì dừng. Không fallback CPU.

HF namespace: `ocr/archives/{batch_id}/ocr-v2/{run_id}/{canary|production}/`.
Không ghi vào archive EasyOCR cũ, trial sharpen hoặc snapshot Online đang dùng.

## 5. File worker và bước bàn giao local

Mỗi batch tạo `ocr-v2-{batch_id}-{mode}-results.zip`, có:

- `report.json`: count/runtime/residual, phase report (GPU/peak VRAM khi model thật expose),
  recognition completion, resume evidence; giá trị không đo được là null, không phải zero;
- `predictions.jsonl`: raw text/confidence VietOCR/Paddle, đúng model/region/signature;
- `frame-selections.jsonl`: UID/frame mapping, bbox, kết quả chọn và provenance từng region;
- `residual.jsonl`: region còn bất đồng/guard fail; chưa gọi Gemini;
- `run-signature.json`, `SHA256SUMS` để kiểm tra nguồn/cấu hình/toàn vẹn.

Gửi **plan JSON, ZIP canary và log từ ENV đến WORKER_BATCH_COMPLETE** để kiểm tra trước
khi treo full. ZIP được upload HF theo content hash; vẫn có file local để tải từ Kaggle.

`frame-selections.jsonl` là `ocr_v2_frame_selection_v1`, **không phải legacy
OcrRecordEnvelope**. `result` giữ đúng nội dung `OcrResult`, language tạm ghi `mixed`
(undetermined), không giả ASCII là tiếng Anh. Còn residual được ghi riêng dù frame có text
được chọn; không coi đủ prediction là tất cả chữ đã đúng. Snapshot/terminal đều chưa ready.

Bước tiếp theo sau worker đã được triển khai riêng bằng schema coverage v3, source pin,
union/validator và SQLite development versioned; làm theo
[OCR_V2_SNAPSHOT_RUNBOOK.md](OCR_V2_SNAPSHOT_RUNBOOK.md). **Không đưa JSONL mới vào builder
legacy hoặc đổi `AIC_OCR_SNAPSHOT_DIR`**.
Nếu cần sửa consumer Online phải xin phạm vi riêng. Không bỏ publishing gate vì deadline.

## Kiểm tra CPU cho người phát triển

```powershell
python scripts/build_kaggle_ocr_v2_production_notebook.py
python -m unittest discover -s tests -p test_ocr_v2_production.py
```

Test dùng archive/catalog/JPEG tổng hợp, predictor giả và HF giả; không download model,
không gọi API/GPU hoặc ghi HF thật. Test gồm partition chín batch, mapping frame_id,
crop/source drift, guard/override, ngắt rồi restore DB mới, duplicate/signature/chain,
sync lỗi, OOM và notebook compile. Regression catalog: HF chỉ có archive, catalog local
auto/explicit, bản trùng/khác nhau, thiếu state, hash lệch archive/plan, worker đổi mount path.
Không commit ZIP, ảnh, JSONL, checkpoint, model, token,
dependency report hoặc runtime output; notebook phải sạch outputs. Commit/push cần yêu cầu riêng.

Kiểm tra code: 22 test riêng worker/launcher PASS, gồm heartbeat khi chờ và subprocess
CPU thật để kiểm tra stdout/stderr, che token, exit code lỗi. Bốn worker sau đó đã chạy
thật trên Kaggle T4, kết thúc `[DONE]` và upload result/report `HF_VERIFIED`. Trước commit
bàn giao, fixture Gate A đã bổ sung đúng cột `video_id`, review notebook regenerate từ
runtime hiện tại; toàn repo **305/305 test PASS trên Windows Python 3.11.9**. Test không
thay thế accuracy ground truth; không chạy lại Kaggle/model trong đợt bàn giao code.
