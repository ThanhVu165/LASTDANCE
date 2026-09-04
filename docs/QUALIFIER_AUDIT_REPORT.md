# Qualifier audit — kết quả và các điều kiện còn mở

Đọc tài liệu vòng sơ tuyển, baseline, runbook và code offline/online/shared/scripts/tests.
Backup trước sửa: `632d0d5a154f5e0b4060c9eef25420f14ae95c50`, đã push. Phạm vi triển khai
được người dùng mở sang Online/ASR theo plan; chỉ đạo sau cùng: ASR đang chạy Kaggle,
chờ hoàn tất mới tải HF; OCR tạm bỏ qua vì chuẩn bị thay index.

## Các vấn đề đã tái hiện và xử lý

| Vấn đề | Ảnh hưởng | Xử lý hiện tại |
|---|---|---|
| Export chỉ nhận frame catalog thưa | Không thể nộp khoảnh khắc ở giữa hai keyframe | VerifiedFrameRef gắn SHA-256 nguồn, ffprobe PTS thật, dùng chung ba task; không tạo UID |
| TRAKE dedup theo shot/cắt Top 32 trước beam | Mất chuỗi đúng dù có frame phù hợp cùng shot | Dedup frame, prune suffix không khả thi trước beam, giữ frame cùng shot |
| QA fuzzy agreement, chọn số dài, broadcast answer | Hai đáp án khác ngữ nghĩa có thể được auto-accept; row không có bằng chứng | AnswerResult có frame/panel/type/unit, agreement chính xác + cùng evidence; numeric extraction theo câu hỏi; còn mơ hồ thì review |
| Contact sheet crop giữa ảnh | Mất chữ/đối tượng ở mép | Letterbox toàn ảnh |
| Đổi frame/answer nhưng giữ checkbox QA | Xác nhận cũ có thể áp sang nội dung khác | Confirmation key gắn video/frame/answer; form QA thủ công vẫn có khi VQA abstain |
| Merge khi draft đủ 100 | Có thể thêm row thứ 101 và báo lỗi | Dừng trước khi append, giữ thứ tự/dedup |
| `--limit` ghi inventory mặc định | Smoke có thể thay inventory full | Bắt buộc output riêng |
| NaN/Inf timestamp và frame bị ép kiểu | Validation vẫn có thể nhận dữ liệu không hợp lệ | Finite timestamp và frame ID integer strict ở các ranh giới liên quan |
| ASR empty output = silent = complete | Coverage có thể khai khống video không được nhận dạng | Consumer đọc legacy v1, nhưng silent chưa có proof vẫn incomplete; chưa can thiệp producer đang chạy |
| ASR publish không đối chiếu catalog/coverage actual | SQLite stale, UID/mapping hoặc count sai vẫn được dùng | Validate catalog SHA, FTS5, checksum/size, UID cùng video, nearest PTS, per-video counts/coverage; torn pair bị INVALID |
| Publishing tin hai cờ thủ công | Artifact/checkpoint dở dang có thể được coi complete | Cần manifest Shot và file bằng chứng mapping/interrupt/resume có checksum; cờ true đơn lẻ không đủ |
| Không có evaluator/split khóa | Dễ tối ưu riêng query đã biết và báo accuracy không kiểm chứng | R-Score/Final Score, diagnostics, 60 assignment, runner resume và freeze development trước held-out |

## Bằng chứng dữ liệu và giới hạn

Catalog hiện có 293.336 keyframe, 873 video, 97.810 shot. Khoảng cách keyframe trong cùng
video: median 22 frame, P90 74, P99 265; 187.635/292.463 khoảng cách lớn hơn 10 frame.
Kho dữ liệu nguồn có 12.258.989 frame. Vì thế retrieval đúng keyframe/video chưa đảm bảo
đúng interval hẹp của TRAKE; review raw frame là bắt buộc để giảm rủi ro này.

Ba FAISS thật đều validate UID set, checksum và vector finite/norm: CLIP 512, SigLIP 768,
EVA-CLIP 768; mỗi index 293.336 UID thuộc 873 video. Report ở
`data/evaluation/qualifier-20260904/artifact-validation.json`. Không rebuild index.

Kiểm chứng tích hợp FFmpeg trên fixture VFR: 40 decoded frames, PTS quanh frame 20 là
0.95/1.00/1.10 giây, giải mã strip 21 frame và xuất ZIP ba task với frame ngoài catalog.
Bằng chứng `data/tmp/qualifier-vfr-smoke/result.json`; đây là kiểm chứng cơ chế, không phải
accuracy trên dữ liệu thi. Test có đổi thứ tự ứng viên, frame cùng shot, UID/frame offset,
video-disjoint split, query có số gây nhiễu, unit/negation mismatch và fingerprint đổi.

Kiểm tra cuối: 288/288 unit/regression test qua; AST 188 file Python; evaluator CLI và
bốn CLI help chạy thành công; git diff --check qua. Chưa commit/push các thay đổi triển khai.

## Chưa thể đóng acceptance

- 60 phiếu `data/evaluation/qualifier-20260904/labels.pending.json` chưa có query/interval/
  QA alias/tên người duyệt. Cần gán nhãn thật trước dev ablation và held-out. Không dùng
  fixture/test/historical query như ground truth.
- Chưa chạy phép đo latency/VRAM 6 GiB hoặc đánh giá operator nhiều vòng trên máy thi.
- ASR còn chạy trên Kaggle: chưa tải HF, chưa có artifact thực để kiểm chứng. Rủi ro
  producer cũ còn cần xử lý sau job: model revision/checksum chưa pin, resume bỏ qua error,
  JSONL tail bị cắt, state chưa bound model/audio/catalog/config, partition phụ thuộc script
  và notebook. Không gọi các rủi ro này là đã sửa chỉ vì consumer kiểm tra chặt hơn.
- OCR được bỏ qua theo yêu cầu; kiểm tra UID/checksum/coverage lại sau khi nạp bản mới.
- Chưa có bộ mapping người và bằng chứng ngắt/chạy lại Kaggle đầy đủ theo publishing proof
  mới; không đánh dấu complete trên cơ sở unit test hoặc cờ trạng thái.

Hướng dẫn chạy và danh sách artifact bàn giao: [QUALIFIER_ACCEPTANCE_RUNBOOK.md](QUALIFIER_ACCEPTANCE_RUNBOOK.md).
