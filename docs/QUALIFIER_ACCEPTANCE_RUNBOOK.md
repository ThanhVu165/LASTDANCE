# Qualifier acceptance và bàn giao ngày 04/09/2026

Báo cáo vấn đề và giới hạn: [QUALIFIER_AUDIT_REPORT.md](QUALIFIER_AUDIT_REPORT.md).

Nguồn contract: [BASELINE_SPEC.md](BASELINE_SPEC.md) §2, §2A và §3.11.
Backup trước triển khai: `632d0d5a154f5e0b4060c9eef25420f14ae95c50`, đã push lên
`codex/online-accuracy-max-v4`. Các thay đổi sau backup chưa commit/push.

## Phạm vi và môi trường

Input: catalog `AIC_DATA/index/frames.csv` + state, ba FAISS độc lập, source video theo
inventory, query và nhãn người. Output: workspace có VerifiedFrameRef, CSV/ZIP chuẩn,
report điểm/diagnostic, lock cấu hình và bằng chứng publishing. Schema dùng chung:
`FrameRecord`, `VerifiedFrameRef`, `AnswerResult`, `AsrSegment`, `EvaluationCase`.
Mọi timestamp phải hữu hạn. Không sinh UID cho raw frame, không mean-pool/rebuild index.

Chạy validation, ffprobe/FFmpeg và evaluator trên Windows Python 3.11 CPU. Không dùng
quota GPU cho các bước này. Runner benchmark dùng provider Online đang cấu hình, có thể
chạy Qwen trên GPU local hoặc gọi cloud; chỉ gọi khi có `--execute`. Máy thi giới hạn
6 GiB VRAM: giữ cơ chế model theo pha hiện có; chưa đo lại peak sau đợt sửa này.
Không chạy ASR/OCR inference trên máy local.

Theo chỉ đạo mới: **ASR đang chạy Kaggle, chờ hoàn thành mới tải Hugging Face**; không đổi
notebook/runtime của job, không restore/download checkpoint đang dở. **OCR tạm bỏ qua** vì
sắp nạp artifact thay thế; giữ snapshot/index hiện tại, không retry hoặc publish OCR.

## Setup và kiểm tra CPU

Chạy từ repo root, dùng môi trường đã thiết lập theo `ENVIRONMENT_SETUP.md`:

```powershell
$env:AIC_DATA = Join-Path $PWD 'data'
& .venv-online/Scripts/python.exe -m unittest discover -s tests -q
& .venv-online/Scripts/python.exe -m scripts.evaluate_qualifier --help
& .venv-online/Scripts/python.exe -m scripts.run_qualifier_benchmark --help
& .venv-online/Scripts/python.exe -m scripts.freeze_qualifier_config --help
```

Giữ FFmpeg/ffprobe trên PATH hoặc cấu hình `AIC_FFMPEG`, `AIC_FFPROBE`. Inventory smoke
bắt buộc output riêng; lệnh dưới chỉ là ví dụ, chưa chạy lại inventory:

```powershell
python -m scripts.build_inventory --limit 1 --output "$env:AIC_DATA/tmp/inventory-smoke.json"
```

`--limit` với output mặc định hoặc trỏ inventory production bị chặn trước khi đọc video.

## Review frame và QA

Ở tab Theo video, dùng số frame nguồn hoặc ±1/5/10, bấm xem dải 21 frame. FFmpeg giải mã
một lượt, ffprobe cấp PTS thực từng decoded frame. Khi chọn raw frame ngoài catalog,
workspace lưu `video_id/frame_id/pts_time/source_sha256`. Export kiểm tra lại nguồn; khi
video không có, ngoài giới hạn hoặc fingerprint đổi, export bị chặn. CSV không thêm cột.

QA chỉ đề xuất row tại frame evidence đã chỉ định. Nếu model không trả lời, form answer
rỗng vẫn có ở video evidence để operator nhập. Xác nhận “Tôi đã xác minh đáp án tại frame
này” sau khi xem ảnh/video; đổi frame hoặc answer làm mất xác nhận trước. Không bulk-add
answer chưa duyệt hoặc placeholder. Không chuyển answer của một panel sang toàn video.

TRAKE giữ frame cùng shot và kiểm tra suffix khả thi trước beam; chỉ cùng video, đúng N
và tăng theo PTS thực. `trake_frame_top_k` và ngưỡng similarity ký tự QA trong config cũ
được giữ để đọc cấu hình cũ; chúng không cắt pool/không xác nhận semantic agreement nữa.

## 60 phiếu gán nhãn và chia tập

Đã tạo `AIC_DATA/evaluation/qualifier-20260904/labels.pending.json`: 60 video riêng biệt,
10 câu mỗi loại mỗi split. Các file `.media.json` chỉ gợi ý keyframe để mở video; không
phải ground truth. `verified_by`, query và interval để trống có chủ đích, evaluator từ chối.
Tạo lại ở một directory mới nếu cần; script không ghi đè công việc gán nhãn:

```powershell
python -m scripts.prepare_qualifier_labels --output "$env:AIC_DATA/evaluation/new-label-assignment"
```

Người gán nhãn phải xem video nguồn, viết câu truy vấn mô tả đúng nội dung, xác định
interval frame gốc cho từng event, khai báo `expected_event_count` TRAKE, và điền tên
`verified_by`. QA điền các đáp án tương đương đã được người duyệt vào `accepted_answers`.
Lưu thành `labels.reviewed.json`. Giữ 30 development và 30 held-out video-disjoint;
không dùng query cũ làm acceptance. Không xem kết quả held-out để chọn cấu hình.

## Benchmark, ablation và khóa cấu hình

Các lệnh dưới là workflow **chưa chạy thực nghiệm**, vì chưa có 60 nhãn người. Chạy mỗi
cấu hình vào output riêng. Baseline và từng ablation phải dùng cùng catalog, nhãn và
provider/model revision. Không gọi cloud trong đợt audit này.

```powershell
python -m scripts.run_qualifier_benchmark --labels "$env:AIC_DATA/evaluation/labels.reviewed.json" --config configs/online_baseline.json --split development --output "$env:AIC_DATA/evaluation/dev-baseline" --execute
python -m scripts.evaluate_qualifier --labels "$env:AIC_DATA/evaluation/labels.reviewed.json" --predictions "$env:AIC_DATA/evaluation/dev-baseline/predictions.json" --runs "$env:AIC_DATA/evaluation/dev-baseline/runs.json" --catalog "$env:AIC_DATA/index/frames.csv" --config configs/online_baseline.json --split development --acceptance --output "$env:AIC_DATA/evaluation/dev-baseline/score.json"
```

Ablation dùng một bản config riêng cho mỗi thay đổi, chạy lại hai lệnh trên vào directory
khác. Thêm `--baseline .../dev-baseline/score.json` vào evaluator để so Final Score từng
loại; non-regression thất bại trả exit code 1. Không điều chỉnh theo vài query cụ thể.

```powershell
python -m scripts.freeze_qualifier_config --reports "$env:AIC_DATA/evaluation/dev-baseline/score.json" "$env:AIC_DATA/evaluation/dev-candidate/score.json" --output "$env:AIC_DATA/evaluation/selected-config.lock.json"
```

Lock ghi hash config, nhãn và report development, không ghi đè file đã có. Với held-out,
truyền `--split held_out --freeze .../selected-config.lock.json` cho cả runner và evaluator,
dùng đúng config đã khóa, output mới. Không tự đánh dấu acceptance vì test xanh.

Resume benchmark: chạy cùng lệnh, cùng output; signature config/nhãn/catalog, code, provider settings và artifact phải khớp.
Mỗi query hoàn chỉnh có checkpoint JSON atomic; query dở chạy lại. Giữ model/provider
cùng revision giữa lần chạy, không trộn trace từ điều kiện khác nhau. Output chính:
`predictions.json`, `runs.json`, `queries/*.json`, `run-signature.json`, `score.json`.

Report có R@1/5/20/50/100, Final Score từng query/từng task, video recall, catalog support
cho từng event, lệch frame tới interval, latency P50/P95 và tỷ lệ query có cờ cần review.
Không có timing thì giá trị null, không giả là 0. Điểm tự động chưa phản ánh hiệu quả
operator sửa nhiều vòng; cần lưu và chấm thêm draft đã review để so assisted accuracy.

## Bằng chứng publishing

`assess_publishing_readiness` là bộ tổng hợp điều kiện; `mapping_verified=true` và
`checkpoint_resume_verified=true` đơn lẻ không còn đủ để `complete=true`.
`PublishingProofs` phải trỏ tới data root, catalog, manifest Shot schema v2, checkpoint
Shot khớp signature/source/output namespace, mapping report và ba resume report độc lập.

Mapping report JSON: `catalog_sha256`, `video_id`, `source_sha256`, `reviewed_by`, và
`samples` gồm `frame_id`, `source_frame_id`, `source_pts_time`. Sample phải trỏ frame
thật trong catalog và khớp manifest shot. Không tạo report từ cờ true.

Mỗi resume report JSON: `catalog_sha256`, `video_id`, `modality`, `embedding_artifact_dir`,
`interruption_exit_code` khác 0, `resume_exit_code=0`; `artifacts` gồm
`interrupted_checkpoint`, `resumed_checkpoint`, `run_log`, mỗi item có `path` tương đối
nằm dưới folder report và `sha256`. Checkpoint theo format visual hiện tại; checkpoint
trước phải nằm giữa batch, sau phải hoàn tất, cùng signature và có lịch sử interrupt/resume.
Validator đọc lại raw embedding shards qua `validate_completed_visual_embedding`, kiểm
tra checksum các bằng chứng, UID/shot và không chấp nhận report thiếu/stale. Ba FAISS
thật vẫn phải được validate độc lập trước khi dùng kết quả tổng hợp. Không công bố đầy
đủ Publishing Criteria khi chưa có đủ mapping người và demo Kaggle ngắt/chạy lại thật.

## Tiếp nhận ASR sau khi Kaggle hoàn tất

Hiện chưa tải Hugging Face và chưa chạy bước tích hợp thật. Sau khi người vận hành xác nhận
batch hoàn tất: pin revision archive, kiểm tra manifest/checksum và partition actual video
set trước khi build snapshot theo `ASR_RUNBOOK.md`. Model revision và signature của runtime
cũ phải được audit, không suy ra đã pin chỉ từ tên `large-v3`.

Envelope v1 `silent` thiếu proof vẫn đọc được nhưng coverage=0 và tăng
`unverified_silent_videos`; không diễn giải thành không có thoại. `SilenceVerification`
gồm audio SHA-256, tên người duyệt và đường dẫn evidence. Lỗi được giữ để xử lý sau khi
job kết thúc. Snapshot development không bao giờ tự thành production-ready.

`publish_asr_index` đòi catalog/state tại data root đích; kiểm tra SQLite/coverage và
nearest-PTS UID từng segment. Incomplete/unverified silence bị chặn, trừ `--allow-partial`
cho development. Hai file được thay atomic từng file; nếu bị ngắt giữa hai lần thay,
reader phát hiện checksum không khớp và báo INVALID, không coi pair là ready. Chạy lại
publish snapshot bất biến để phục hồi; đây không phải transaction atomic hai file.

## File bàn giao và phần còn chờ

Code/schema/test/runbook được review trong Git diff. `data/`, `tmp/`, audio/video,
model weights, notebook outputs, SQLite/FAISS, prediction/label/evidence artifacts không
commit. Bàn giao chúng qua kênh dữ liệu riêng sau khi xác minh, không tự publish từ script.

Đã kiểm tra CPU: toàn suite 288 test; AST 188 file Python; FFmpeg VFR smoke 40 frame,
strip 21 frame và ZIP cả ba task dùng raw frame ngoài catalog. Bốn CLI --help đã chạy;
evaluator CLI fixture cho Final Score 0.6 đúng kết quả tính tay. Bằng chứng nằm tại
`AIC_DATA/tmp/qualifier-vfr-smoke/result.json`; không phải đo retrieval accuracy.
Catalog/FAISS report: `AIC_DATA/evaluation/qualifier-20260904/artifact-validation.json`.

Còn chờ: 60 nhãn người, dev ablation và held-out, đo latency/VRAM trên máy thi, demo
Kaggle interrupt/resume nếu bằng chứng hiện có không đủ, ASR hoàn tất và tiếp nhận thật.
Các sửa ASR producer/partition/notebook trong plan chưa áp dụng lên job đang chạy.
OCR đã được người dùng yêu cầu bỏ qua đợt này. Không có kết luận accuracy-complete.
