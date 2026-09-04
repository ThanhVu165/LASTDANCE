# OCR v2 — plan triển khai đã chốt

Cập nhật 04/09/2026 theo cuộc trao đổi với người dùng.
**Nguồn chuẩn kỹ thuật duy nhất: [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2.**
File này là checklist triển khai để đối chiếu, không phải một baseline khác. Nếu cần đổi
model/router/schema, sửa spec và Changelog theo quyết định người dùng trước, rồi đồng bộ plan.

## Những gì đã chốt — không mở lại khi viết code

- CRAFT bbox cache từ chín archive EasyOCR → VietOCR `vgg_seq2seq` mọi crop →
  Paddle `latin_PP-OCRv5_mobile_rec` có điều kiện → guard/selection + residual →
  JSONL shard → union/SQLite local. Gemini tùy chọn, cần duyệt riêng.
- Giữ crop gốc Gate B; **không bật bicubic 2×/UnsharpMask** sau trial 30 crop.
  Không chạy lại EasyOCR/Vintern, không tự thêm deblur, xoay/tách bbox hay lấy frame lân cận.
- Bốn tài khoản Kaggle, một T4/worker; VietOCR batch khởi đầu 64, Paddle 128,
  pha/process riêng, giảm minibatch khi OOM. Máy Codex không chạy model.
- Input OCR đã ở HF `ocr/archives`, tên `ocr-production-batch-0X-easyocr.zip`.
  Không yêu cầu upload lại chín ZIP; JPEG keyframe gốc và `frames.csv`/state đọc từ
  dataset Kaggle đã gắn. Mỗi worker gắn catalog đầy đủ cùng hash và JPEG batch được giao;
  không yêu cầu HF có catalog.
- Giữ `OcrResult`, năm cột `ocr_fts`, UID và `frame_id` mapping. Không sửa
  `online/`/`app/`, không đổi snapshot Online đang dùng nếu chưa kiểm tra tương thích.

## Ba lời nhắc bắt buộc khi triển khai

1. **Báo tiến độ:** log/heartbeat tối đa 30 giây, có phase/worker/batch/video,
   done/total, tốc độ/ETA và checkpoint HF cuối đã verify. Chưa đo thì ghi unavailable,
   không ước lượng từ Batch 01 nhân chín.
2. **Lưu giữa chừng:** persist từng minibatch, sync HF theo §2.2b tối đa 5 phút giữa các
   mốc kiểm tra và cuối pha/batch. Resume đúng signature ở session mới; dừng nếu sync
   thất bại sau retry hữu hạn. Không hứa tự sống lại khi VM chết hoặc không mất minibatch
   chưa lưu. Không chỉ lưu `/kaggle/working`.
3. **Chín input HF bất biến:** pin cùng revision, validate manifest/checksum/catalog,
   chia nguyên batch theo số region thật, bốn worker disjoint/exhaustive, namespace
   output riêng `ocr/archives/{batch_id}/ocr-v2/{run_id}/`. Secret không nằm trong code.

## Checklist triển khai tiếp theo

| Bước | Việc phải hoàn tất | Đối chiếu spec | Trạng thái |
|---|---|---|---|
| 1 | Đồng bộ quyết định và evidence trong tài liệu local | §2.2, Changelog bản 28 | Đã cập nhật tài liệu |
| 2 | Pin input revision, lấy count thật đủ 9 batch, xuất plan 4 worker và kiểm tra partition | §2.2b | Đã xong; cùng plan được cả bốn worker dùng |
| 3 | Dựng environment và runner stream crop, VietOCR/Paddle + guard/selection | §2.2 | Đã chạy thật trên bốn Kaggle T4 |
| 4 | Log, local/HF checkpoint, stop/resume không thiếu/trùng/conflict | §2.2b | HF checkpoint/restore đúng signature đã thể hiện trong log production |
| 5 | Bổ sung engine/mode/tier provenance v2 và validator tương thích ngược | §2.2c | Đã có coverage schema v3 riêng; legacy v1/v2 giữ nguyên |
| 6 | Canary end-to-end trên T4, đo I/O + inference + HF; ngắt rồi resume process mới | §2.2b–c | Đã qua canary bắt buộc để mở production; report production giữ resume evidence |
| 7 | Chạy bốn worker sau preflight, union/validate và build snapshot local | §2.2c, §2.3 | Recognition 9/9 xong; snapshot thật đã validate 293.336 UID, 269.259 FTS row; adapter Online còn thiếu |
| 8 | Đếm residual, đề xuất API canary/cost nếu cần | §2.2d | Tùy chọn; chưa được duyệt gọi API |

Không giả kết quả VietOCR/Paddle thành EasyOCR/Vintern để qua schema cũ. Consumer snapshot
hiện dùng validator trong `offline/`; phải kiểm tra tương thích khi migration. Nếu thực
sự cần thay `online/`/`app/`, báo người dùng và dừng phần tích hợp đó, không tự sửa chéo.

## Evidence đã có và giới hạn

- Gate B có kết quả thực tế; là evidence runtime/visual, **chưa PASS recall/CER định lượng**.
  Người dùng chọn triển khai v2 theo deadline; giữ ngưỡng evaluation cũ để tái lập.
- Trial làm nét đủ 90/90; original khớp Gate B 30/30, đối chứng giữ nguyên 10/10,
  không xác nhận đủ ba crop cải thiện rõ. Quyết định **giữ gốc**, không chạy lại trial.
  SHA/signature và giới hạn mẫu ở baseline §2.2a; ZIP evidence không bị sửa.
- Bốn worker production đã kết thúc `[DONE]`; từng batch có result/report `HF_VERIFIED` và
  `recognition_complete=true`. Log chạy lại cho thấy prediction đúng signature được resume,
  không gọi model lại cho phần đã có. Xem [hướng dẫn recognition](OCR_V2_PRODUCTION_RUNBOOK.md)
  và [hướng dẫn snapshot](OCR_V2_SNAPSHOT_RUNBOOK.md).
- Online vẫn dùng snapshot EasyOCR development, không phải artifact v2/final.

## Deadline và bàn giao

Người dùng đã đặt cửa sổ khoảng 10 tiếng **gồm cả chuẩn bị**. Mốc phân bổ ban đầu
(0–1,5h chuẩn bị; 1,5–7,5h chạy; 7,5–9h chốt phần chính; 9–10h union/validate) chỉ là
ngân sách, không tự reset và không phải ETA đã đo. Cập nhật khả năng kịp sau canary
end-to-end và workload thật; không kéo API/thử nghiệm mới vào critical path.

Bàn giao `ocr.sqlite` + `coverage.json` + `SHA256SUMS` versioned; prediction/residual
riêng. Còn lỗi/residual chưa chốt hoặc publishing gate chưa đạt thì giữ
`complete=false`, `production_ready=false`. Gemini không chặn bản development, nhưng
không vì bỏ API mà tự gọi bản có residual là final. Không tự commit/push/upload kết quả
từ máy Codex. Union/SQLite v2 dùng builder schema v3 riêng; không dùng builder legacy cho
JSONL mới và chưa đổi consumer Online.

## Đọc tài liệu nào

- Nhóm Online tải kết quả và tích hợp: [OCR_V2_ONLINE_HANDOFF.md](OCR_V2_ONLINE_HANDOFF.md).
- Kiến trúc, router, checkpoint, schema và acceptance: [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2.
- Thứ tự triển khai và trạng thái: file plan này.
- Sync/union/validate SQLite development: [OCR_V2_SNAPSHOT_RUNBOOK.md](OCR_V2_SNAPSHOT_RUNBOOK.md).
- Đầu mối vận hành và lệnh lịch sử: [OCR_RUNBOOK.md](OCR_RUNBOOK.md).
- Chạy planner/canary/recognition worker: [OCR_V2_PRODUCTION_RUNBOOK.md](OCR_V2_PRODUCTION_RUNBOOK.md).
- Tái lập Gate A/B: [OCR_V2_EXPERIMENT_RUNBOOK.md](OCR_V2_EXPERIMENT_RUNBOOK.md).
- Tái lập trial làm nét: [OCR_V2_SHARPEN_TRIAL_RUNBOOK.md](OCR_V2_SHARPEN_TRIAL_RUNBOOK.md).
