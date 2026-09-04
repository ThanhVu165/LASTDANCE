# OCR v2 — thử làm nét 30 crop

Thử nghiệm bổ sung đã được duyệt trong `BASELINE_SPEC.md` §2.2a. Không thay Gate A/B,
không bật preprocessing production, không build SQLite, không chạy EasyOCR/Paddle/Vintern
hay API. Không đổi `OcrResult`, bbox hoặc khóa nội dung. Bốn worker production và input
chín archive trên HF `ocr/archives` giữ nguyên phạm vi; notebook này **không** triển khai
worker production. Contract chuẩn duy nhất ở `BASELINE_SPEC.md` §2.2; checklist triển khai
và các lời nhắc vận hành nằm ở `OCR_V2_PRODUCTION_PLAN.md`.

## Kết quả đã chốt 04/09/2026 — không bật làm nét

Trial thực tế hoàn tất 90/90 (30 crop × ba phương án), original khớp Gate B 30/30,
10/10 đối chứng không đổi. Bicubic đổi text 8/30; thêm UnsharpMask đổi 10/30. Review
ảnh gốc chưa xác nhận crop cải thiện rõ, không đạt ngưỡng ít nhất ba crop tốt hơn và
không có crop readable tệ hơn. **Giữ crop gốc Gate B; không bật upscale/sharpen production.**
Mẫu chọn 20 low/10 control không đại diện accuracy toàn catalog; confidence tăng không là gain.

Evidence: `ocr-v2-sharpen-results.zip`, SHA-256
`5ea998ca9a2718f362de322bca1330b5a11b9c7340d8e013f4c9ed4ea062edf5`.
Signature `ebeb37e7b967786691387dbb29cfcef7d2a26510711c22baad1eabe070ce0d97`;
31,79 giây recognition + sync trong report, không phải ETA production. ZIP giữ nguyên
`PENDING_VISUAL_REVIEW`; kết luận review được ghi trong baseline §2.2a và tài liệu này,
không sửa artifact gốc hoặc giả có CSV ground truth đã được người dùng điền.

Hướng dẫn dưới đây chỉ để tái lập/khôi phục trial khi cần; **không cần chạy lại trial**
trước khi triển khai production. Resume của trial không xác minh resume bốn worker v2.

## Chạy nhanh trên Kaggle

Dùng notebook riêng `notebooks/kaggle_ocr_v2_sharpen.ipynb` trên **Kaggle T4**, bật Internet.
Có thể dùng môi trường Gate B đã chạy; không cần chạy lại Gate B. Chạy ba cell code theo
thứ tự. Bước `[0/5] ENVIRONMENT_CHECK_START` tự tải wheel VietOCR 0.3.13 khi thiếu, verify
SHA giống Gate B rồi cài `--no-deps`; bổ sung einops 0.8.1/gdown 5.2.0/PyYAML 6.0.2 chỉ
khi gói tương ứng chưa có. Không cài Paddle hoặc resolve CUDA dependencies; kiểm tra
Torch/torchvision/NVIDIA không đổi và probe import Predictor trước khi tạo crop.
GPU không phải T4 hoặc VietOCR đã có khác pin sẽ báo lỗi. Setup/download nằm ngoài ngân
sách 600 giây recognition. Máy Codex chỉ build/kiểm tra code, không chạy model.

Input cần có:

- `ocr-v2-review-bundle.zip` hoặc thư mục tự giải nén từ Gate A.
- `ocr-v2-gate-b-results.zip` hoặc thư mục tự giải nén từ Gate B. Nếu dùng session cũ,
  notebook tự tìm ZIP trong working directory.
- Dataset JPEG Batch 01, gồm đủ `L21_V001`, `L21_V002`, `L21_V003`, `L21_V005`, `L21_V006`.

Không cần gắn/tải cả chín EasyOCR archive cho phép thử này: metadata/bbox đã được khóa
qua sample hash từ Gate B. Không dùng ảnh contact sheet làm ảnh đầu vào. Nếu tự tìm path
bị trùng, điền `REVIEW_BUNDLE`, `GATE_B_RESULTS`, `KEYFRAMES_ROOT` trong cell tham số.
Các path là cấu hình, không hardcode path máy cá nhân. Output mặc định là
`$AIC_DATA/ocr-v2-sharpen`; trên Kaggle `AIC_DATA` mặc định `/kaggle/working`.

Notebook chọn bốn confidence VietOCR thấp nhất và hai cao nhất còn lại mỗi video; null/
NaN/Inf được xếp thấp, tie-break bằng `region_id`. Không bắt mẫu Gate B cũ có đúng 24 crop
mỗi video. Cả 120 region phải có đủ ba model, cùng sample hash, không trùng/foreign/error.

Ba phiên bản: crop Gate B gốc; bicubic 2×; bicubic 2× +
`UnsharpMask(radius=1, percent=100, threshold=3)`. Cùng model/config/resize nội bộ VietOCR,
không sửa chính tả, không dùng AI sinh ảnh. Model/config/weights kế thừa checksum Gate B.
Model cache mặc định dùng lại thư mục `ocr-v2-gate-b`; thiếu mới tải weights/config đã pin.

`crop_width/crop_height` trong sample là metadata của cách cắt EasyOCR cũ; không dùng để
assert kích thước crop QUAD + padding của Gate B. Log `CROP_SIZE_METADATA_DIFFERENCE` chỉ
ghi khác biệt này để audit. JSONL giữ `archive_crop_size` và `gate_b_crop_size`; pixel vẫn
theo đúng helper Gate B, source/crop hash vẫn khóa checkpoint.

## Tiến độ, ngắt và resume

- Log `[1/5]` đến `[5/5]`, `VARIANT_START`, `MINIBATCH_SAVED`; heartbeat 30 giây trong các
  bước lâu. Heartbeat chỉ cho biết pha còn hoạt động, không giả đã nhận dạng thêm.
- Minibatch mặc định sáu crop. Sau mỗi minibatch ghi `checkpoint.json` bằng flush/fsync và
  atomic replace, rồi cập nhật `ocr-v2-sharpen-checkpoint.zip` trong thư mục output.
- Mặc định `DURABLE_CHECKPOINT_TO_HF=True`: cần Kaggle Secret `HF_TOKEN` có quyền ghi repo
  private `HF_REPO_ID` (mặc định repo OCR đã cấu hình `MinhThuw0103/lastdance-visual-embeddings`).
  Checkpoint được upload và verify round-trip SHA sau **mỗi minibatch**, dưới prefix mới
  `ocr/archives/batch-01/ocr-v2-sharpen/<signature>/`. Không chạm archive EasyOCR hoặc model khác.
  Mỗi ZIP là immutable/content-addressed; không có file latest chung cho nhiều worker.
  Mất quyền upload thì dừng trước inference (preflight) hoặc sau minibatch vừa lưu local;
  lỗi mạng retry ba lần có backoff rồi dừng, không chạy tiếp âm thầm mà không có backup.
  Log `MINIBATCH_SAVED` sau HF verify; `HF_CHECKPOINT_VERIFIED` cho biết dữ liệu đã lưu remote.
- Signature khóa nguồn Gate A/B, crop/source-image hash, code/helper hash, model/config,
  phiên bản package, batch size và các phương án. Không resume khi khác signature.
- Rerun cùng cell bỏ qua đúng `(region_id, variant)` đã hoàn tất. Không commit minibatch
  thiếu output. Confidence không hữu hạn ghi null, không ghi NaN vào JSON.
- Để demo: đặt `INTERRUPT_AFTER_NEW=6`, chạy đến intentional interrupt; đặt lại `0` và
  chạy cell runtime lần nữa. Sẽ báo `RESUME completed=6 remaining=84`.
- Sau khi mất VM, mặc định notebook tự tìm và tải checkpoint HF mới nhất cùng signature;
  chỉ các minibatch đã verify trên HF được bảo đảm phục hồi. Minibatch đang chạy/chưa upload
  có thể phải chạy lại. Nếu chủ động đặt `DURABLE_CHECKPOINT_TO_HF=False`, cần **download ZIP
  trước đó**, gắn vào Input session mới và điền `RESTORE_CHECKPOINT`. Results ZIP cũng có
  checkpoint bên trong. Môi trường mới phải khớp signature.
- Nếu cắt điện máy cá nhân nhưng session remote còn chạy, không cần chạy lại; nếu session
  Kaggle bị hủy thì phải khởi chạy lại notebook; không tự khởi động lại VM. Không chạy hai
  bản trial cùng signature đồng thời: checkpoint xung đột sẽ bị từ chối.

Nhận dạng có ngân sách 600 giây **sau khi model sẵn sàng**; không có warmup riêng hoặc
benchmark 5.000 crop. Kaggle/Linux có SIGALRM và kiểm tra giữa minibatch. Native CUDA bị
kẹt có thể trì hoãn Python xử lý tín hiệu; không hứa hard timeout khi driver treo.
Setup/download, tạo crop và export sheet nằm ngoài timer. Khi timeout/lỗi/ngắt cell có
thể bắt được, xuất ZIP partial; mất process đột ngột vẫn còn minibatch đã checkpoint.
HF checkpoint sync nằm trong ngân sách nhận dạng; final ZIP upload nằm ở pha export.

## Xem kết quả và quyết định

Tải `/kaggle/working/ocr-v2-sharpen-results.zip` (hoặc thư mục cha output tùy `AIC_DATA`).
Trong đó có:

- `sheets/compare-01.png` … `compare-30.png`: ba cột ảnh và **toàn bộ text**, không cắt chữ.
- `crops/`: 90 PNG nguyên bản của ba biến thể; sheet chỉ là bản hiển thị.
- `selected-sample.jsonl`, `recognizer-results.jsonl`, `runtime-report.json`,
  `run-signature.json`, `checkpoint.json`, `visual-review.csv`, `SHA256SUMS`.

Khi bật HF, results ZIP cũng được lưu theo checksum dưới cùng prefix trial; bản partial
và complete không ghi đè nhau. Token không được đưa vào signature, report hay log.

### Khôi phục bản cũ lỗi font sau khi đã chạy đủ 90/90

Giữ nguyên các cell/runtime/signature đang chạy, thêm cell sau trong cùng session:

```python
import os
from pathlib import Path
import matplotlib
from PIL import ImageFont

font_path = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
ImageFont.truetype(str(font_path), 20)
os.environ["OCR_REVIEW_FONT"] = str(font_path)
main()
```

`main()` sẽ thấy checkpoint 90/90 và xuất sheet/ZIP, không load model hoặc nhận dạng lại.
Không thay runtime bằng bản mới khi khôi phục run cũ: code hash nằm trong signature.
Bản mới tự tìm font Matplotlib và kiểm tra font ngay đầu trước inference.

Điền review: `original_readable=yes/no`; mỗi cột phương án điền
`better/same/worse/unreadable`, so với text VietOCR gốc và chữ nhìn thấy trên **ảnh gốc**.
Nếu ảnh gốc không đọc được thì không đoán đúng/sai từ ảnh làm nét. Không coi confidence
tăng là thắng, không lấy đồng thuận các phương án làm ground truth.

Chỉ **cân nhắc** một phương án nếu có ít nhất ba crop readable tốt hơn và không có crop
readable nào tệ hơn, đặc biệt số, dấu và tên. Đánh giá riêng từng phương án, không chọn
variant tốt nhất cho mỗi crop rồi cộng thành score giả. Nếu không rõ, giữ Gate B. Nếu tốt,
chỉ đề xuất preprocessing cho nhóm khó, chưa tự bật production. Report luôn
`production_ready=false` và `PENDING_VISUAL_REVIEW` (hoặc `INCOMPLETE`). File review đã
điền không bị ghi đè khi rerun.

## Build và kiểm tra CPU cho người phát triển

```powershell
python scripts/build_kaggle_ocr_v2_sharpen_notebook.py
python -m unittest discover -s tests -p test_ocr_v2_sharpen.py
```

Test chỉ dùng Pillow/NumPy và predictor giả để kiểm tra chọn mẫu, ba ảnh biến thể, checkpoint,
ngắt/resume, dữ liệu lỗi, text dài và notebook compile. Không tải weights/chạy model.
Kaggle inference thật và resume qua session mới cần người vận hành xác minh; không gán
CPU mock test thành chứng nhận GPU/Publishing Ready. Không commit PNG, CSV review,
checkpoint, ZIP, JSONL kết quả hoặc weights; chỉ commit code/notebook sạch và tài liệu khi
được yêu cầu riêng.
