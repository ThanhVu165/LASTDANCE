# AGENTS.md — LASTDANCE (AIC 2026) — Context cho Codex

> File này được Codex tự động đọc ở đầu mỗi phiên làm việc trong repo. Không xóa, chỉ cập
> nhật khi có thay đổi thật trong spec. Nếu Codex và file này mâu thuẫn nhau — Codex sai,
> sửa theo file này.

---

## 1. Bối cảnh cuộc thi (bắt buộc hiểu trước khi code bất kỳ dòng nào)

- **Cuộc thi:** AI Challenge (AIC) 2026 — nhiệm vụ Video Corpus Moment Retrieval.
- **3 dạng câu hỏi:**
  - **KIS** (Known-Item Search) — tìm đúng khoảnh khắc từ mô tả text.
  - **Q&A** — trả lời câu hỏi dựa trên nội dung video (có gating BLIP-2 + Qwen VQA).
  - **TRAKE** — tìm chuỗi khoảnh khắc theo trình tự thời gian (Beam Search + decay).
- **Chấm điểm:** R-Score và Final Score dựa trên **tối đa 100 kết quả** dạng
  `(video_id, frame_id)`. **Khóa nộp bài luôn là `(video_id, frame_id)`, KHÔNG BAO GIỜ dùng
  `local_idx`.**
- **Dữ liệu:** 500GB video thô, 873 video, mục tiêu nén còn ~80GB keyframe + <5GB
  vector/text index để nạp vừa RAM máy local.
- **Máy tham chiếu local/Shot:** Intel i5-12450H, RTX 4050 Laptop **6 GiB VRAM**,
  Python 3.11 trên Windows/Colab. **Visual Embedding Kaggle GPU dùng Python 3.12.x** theo
  image Kaggle đã validate; không áp Python 3.12 ngược sang profile local/Shot.
- **Mốc lịch sử:** từ 23/08/2026 từng đặt hạn 6 ngày cho toàn pipeline; không xem đây là
  đồng hồ còn lại. Ngân sách OCR người dùng đặt khi chốt v2 là khoảng 10 tiếng gồm chuẩn bị,
  không tự reset sau mỗi lần sửa; xem `docs/BASELINE_SPEC.md` §2.2b.

---

## 2. Nguồn chuẩn duy nhất (Source of Truth)

Chỉ có **một** nguồn chuẩn kỹ thuật: `docs/BASELINE_SPEC.md`. File này chứa toàn bộ
contract Offline, ASR và Online. Runbook/status/roadmap chỉ hướng dẫn vận hành hoặc ghi tiến
độ, không được tạo schema hay quyết định kiến trúc khác baseline.

**Nếu code hiện tại lệch với spec → sửa code theo spec, không sửa spec theo code**, trừ khi
người dùng xác nhận rõ ràng đây là thay đổi có chủ đích (và phải ghi vào Changelog của spec).

Nếu vận hành hoặc sửa Shot Detection, đọc thêm `docs/SHOT_DETECTION_RUNBOOK.md` trước khi
chạy batch.

---

## 3. Phạm vi phiên làm việc này (Nhánh 1 — Offline Indexing)

Codex đang làm việc trong scope **Nhánh 1** (video preprocessing + offline indexing).
Nhánh 2 (`online/`) do người khác phụ trách — **không tự ý sửa code trong `online/` hoặc
`app/`** trừ khi được yêu cầu rõ; nếu phát hiện online/ cần đổi để khớp thay đổi ở offline/,
**dừng lại và báo cáo**, không tự sửa chéo nhánh.

### Việc thuộc phạm vi:
- `offline/` — TransNetV2 shot detection, keyframe extraction, dedup/lọc nhiễu, visual
  embedding (CLIP/SigLIP/EVA-CLIP), OCR v2 (CRAFT bbox cache → VietOCR → Paddle có điều kiện;
  Gemini residual tùy chọn sau duyệt riêng), build FAISS + SQLite FTS5.
- `shared/schemas/` — Pydantic schema dùng chung (`FrameRecord`, `OcrResult`, `AsrSegment`)
  — sửa ở đây ảnh hưởng cả 2 nhánh, cần cẩn trọng và thông báo khi đổi.
- `scripts/` — script chạy batch trên Kaggle, tách nhỏ danh sách video song song.

---

## 4. Nguyên tắc bất di bất dịch (không được vi phạm dù tối ưu thế nào)

1. **Không mean-pool nhiều keyframe/shot thành 1 vector.** Mỗi keyframe có vector riêng.
2. **`keyframe_uid`** là khóa nội dung (deterministic hash `blake2b` từ
   `video_id:shot_id:local_idx`) dùng cho FAISS/OCR/ASR — **không dùng vị trí insert
   (`faiss_row_id`/row index)** làm khóa. Xem công thức chính xác ở
   `docs/BASELINE_SPEC.md` §2.1d.
3. **`local_idx`** chỉ dùng nội bộ để trỏ file ảnh (`{video_id}/{shot_id}_{local_idx}.jpg`)
   — không dùng làm khóa nộp bài hay dedup.
4. **Không hardcode path tuyệt đối** — mọi path build từ biến môi trường `AIC_DATA`
   (default `data/`).
5. **Không giả định FPS/resolution/duration** — phải chạy Inventory full bằng `ffprobe`
   trước Shot Detection; `--limit` chỉ dùng smoke với output riêng, không publish.
6. **Không giả định VRAM vô hạn** — 6GB VRAM local, phải ghi rõ ước tính và cơ chế
   load/release model theo pha nếu chạy local.
7. **Vector bắt buộc ép `float16`** trước khi lưu/push (giảm dung lượng + băng thông HF).
8. **3 FAISS index (CLIP/SigLIP/EVA-CLIP) build độc lập, không cần đồng bộ thời gian/thứ tự**
   — vì dùng `keyframe_uid` làm khóa qua `IndexIDMap`.
9. **Một `video_id` chỉ `complete = true` khi thỏa đủ Publishing Criteria** (mục 6 dưới) —
   không set `complete=true` khi còn checkpoint dở dang, không được bỏ bớt điều kiện để
   tiết kiệm thời gian.

---

## 5. Môi trường vận hành thực tế — Kaggle + HuggingFace Dataset

- **Keyframe extraction + dedup** → chạy **local CPU**. Shot Detection giữ CPU làm reference
  nhưng production batch được phép chạy **Windows NVIDIA GPU hoặc Colab CUDA** khi dùng đúng
  cùng commit/config và toàn bộ 5 video dev-subset đã qua parity 100% từng shot/range với
  manifest CPU. CUDA phải được chọn tường minh, ghi provenance và fail closed; tuyệt đối
  không fallback âm thầm về CPU.
- **Embedding (CLIP/SigLIP/EVA-CLIP)** → chạy Kaggle GPU theo batch. **OCR v2
  (VietOCR + Paddle có điều kiện, tái sử dụng CRAFT bbox cache)** chỉ chạy Kaggle GPU,
  chia chín batch UID-disjoint trên bốn tài khoản OCR, một T4/worker. Máy Codex chỉ viết code,
  orchestration và validate artifact; RTX 4050 máy thi chỉ chạy Online và đọc
  `ocr.sqlite` đã build sẵn, không chạy model OCR lúc thi. Nhánh OCR và Nhánh 3 (ASR) không
  dùng chung tài khoản/quota.
- **Đồng bộ artifact Kaggle ↔ máy local** qua **HuggingFace Dataset (Git LFS)**:
  ```
  Kaggle Notebook (build) --push_to_hub()--> HF Dataset (Git LFS) --snapshot_download()--> Local
  ```
  Bắt buộc:
  - Vector đã ép `float16` trước khi push.
  - **Không push từng video một** — gom batch (50–100 video/lần hoặc cuối mỗi phiên Kaggle).
    Ngoại lệ checkpoint OCR v2: sync HF giữa batch theo §2.2b, tối đa 5 phút giữa mốc kiểm
    tra minibatch; đây không phải publish final từng video. Log/heartbeat tối đa 30 giây.
  - Chỉ `snapshot_download()` full repo về local 1 lần trước khi thi, không pull lại giữa
    chừng trừ khi có patch khẩn.
  - Đặt tên revision/commit theo batch (`batch-01`, `batch-02`...) để rollback không phải
    re-push toàn bộ.
  - OCR được phép bàn giao SQLite snapshot development versioned để Nhánh 2 làm song song,
    nhưng mỗi snapshot phải bất biến, có coverage/checksum và luôn `complete=false`,
    `production_ready=false`; không được thay thế `ocr.sqlite` final.

---

## 6. Publishing Criteria — điều kiện "Ready" để Nhánh 2 dùng được

- [ ] `complete = true` tính theo từng `video_id` (không phải toàn catalog).
- [ ] Tập `keyframe_uid` trong `frames.csv` khớp 100% với tập ID đã add vào **cả 3** file
      FAISS (diff bằng code, không đếm dòng thô).
- [ ] Không có `NaN`/`Inf` trong bất kỳ vector nào.
- [ ] Norm vector ≈ 1 sau L2 normalize (kiểm tra sample ngẫu nhiên).
- [ ] Mapping `video_id`/`frame_id`/`pts_time` đã xác thực qua Sanity Check thủ công.
- [ ] Checkpoint/resume hoạt động đúng (test bằng cách ngắt giữa batch rồi chạy lại — không
      duplicate, không mất dữ liệu).

---

## 7. Rủi ro đã biết — KHÔNG cần re-investigate, đã có quyết định

- **Shot detector production đã chốt là TransNetV2.** Không tiếp tục chờ weight AutoShot,
  không làm A/B AutoShot trên critical path và không đổi detector nếu chưa có quyết định mới
  từ người dùng. Runtime dùng `transnetv2-pytorch==1.0.5`, là port PyTorch bên thứ ba chứ
  không phải TensorFlow SavedModel gốc của tác giả. CPU giữ vai trò reference/fallback;
  production ưu tiên Windows NVIDIA GPU sau parity 5/5. Mỗi worker phải dùng checkpoint
  signature-aware và chỉ đánh dấu xong video sau khi manifest schema v2 đã atomic-publish và
  validate lại thành công.
- **OCR v2 chốt 04/09/2026:** CRAFT bbox cache từ chín archive EasyOCR trên HF `ocr/archives`
  → VietOCR mọi crop → Paddle có điều kiện → Gemini residual tùy chọn sau duyệt riêng.
  Đọc `docs/BASELINE_SPEC.md` §2.2 và checklist `docs/OCR_V2_PRODUCTION_PLAN.md` trước khi
  triển khai. Không chạy lại EasyOCR/Vintern, không bật làm nét sau trial 30 crop, không
  so confidence chéo model. Gate B chỉ có evidence runtime/visual, không giả PASS định lượng.
  Bốn worker chia nguyên batch bằng số region thật, cùng input revision, checkpoint local
  từng minibatch + HF verified, namespace riêng, fail closed khi sync lỗi. Recognition worker
  đã hoàn tất 9/9 batch T4/HF thật; snapshot schema v3 local đã validate, adapter Online
  còn thiếu. Người dùng hoãn ground truth để bàn giao development, không giả quality PASS;
  xem `docs/OCR_V2_ONLINE_HANDOFF.md`. Không gán
  output mới thành engine/tier cũ. Giữ snapshot EasyOCR đang phục vụ Online cho tới khi
  handoff tương thích đã validate. Same-shot reuse vẫn cần đồng thời pass embedding cosine,
  CRAFT layout, crop SSIM và crop pHash; embedding không gate `no_text`.
- **BEiT-3 đã bị loại vĩnh viễn khỏi kiến trúc.** Không mở lại audit/checksum/conversion
  Microsoft UniLM; modality thứ ba chính thức là EVA-CLIP với checkpoint safetensors đã pin.
- **Thể lệ AIC 2026 có internet trong phòng thi hay không — CHƯA XÁC NHẬN.** Ảnh hưởng trực
  tiếp đến Query Planning online; OCR offline không phụ thuộc internet lúc thi vì
  `ocr.sqlite` phải build sẵn. Không code phần Online phụ thuộc giả định "chắc chắn có
  internet" cho đến khi có xác nhận.

---

## 8. Việc KHÔNG được tự ý làm

- Không tự chạy `git commit`, `git push`, tạo Pull Request hoặc publish thay đổi lên remote.
  Chỉ được thực hiện từng hành động này khi người dùng yêu cầu rõ trong phiên hiện tại. Trước
  khi commit phải báo `git status`, danh sách file dự kiến add và loại artifact sinh tự động;
  yêu cầu làm code/"làm tiếp" không được hiểu ngầm là cho phép commit hoặc push.
- Không đổi `keyframe_uid` trở lại kiểu positional/row-index.
- Không tự thêm audio captioning phi ngôn ngữ (BEATs) hay speaker diarization — đã bị cắt
  khỏi scope Nhánh 3 để kịp deadline.
- Không tự ý build thêm 1 visual index đã gộp sẵn (SRRF) ở Nhánh 1 — việc gộp 3 điểm
  CLIP/SigLIP/EVA-CLIP thành `score_visual` là việc của Nhánh 2 lúc query, không phải lúc index.
- Không đổi tên cột SQL giữa `ocr.sqlite` và `asr.sqlite` trái
  `docs/BASELINE_SPEC.md` §2.1d/§2A.3; giữ `detected_text` và `transcribed_text` để Nhánh 2
  dùng chung một module `FtsSearcher`.

---

## 9. Khi bắt đầu 1 task cụ thể trong phiên này

Trước khi code, Codex nên tự trả lời (và nói rõ trong phản hồi):
1. Task này thuộc bước nào trong pipeline (`docs/BASELINE_SPEC.md` §2–§3)?
2. Input/output chính xác là gì, khớp schema nào trong `shared/schemas/`?
3. Chạy ở đâu — local CPU hay Kaggle GPU? Có đụng tới quota GPU không?
4. Có điều kiện nào trong Publishing Criteria (§6 ở trên) liên quan đến task này không?

---

## 10. Handoff và tài liệu vận hành

- Mọi CLI, dependency, artifact contract hoặc workflow mới phải cập nhật hướng dẫn sử dụng
  cho đồng đội trong **cùng thay đổi/commit**; không chỉ giao code.
- Hướng dẫn phải ghi rõ input/output, môi trường CPU/GPU, lệnh setup/run/validate, cách
  resume, file được bàn giao và file không được commit.
- Workflow chuyên biệt phải có runbook trong `docs/` và được liên kết từ README liên quan.
- Trước khi push, chạy lại các lệnh trong hướng dẫn trên môi trường mục tiêu hoặc ghi rõ phần
  nào mới chỉ là planned/chưa được xác minh.
- Commit và push là hai quyền riêng biệt: được yêu cầu commit không đồng nghĩa được phép push.
  Không dùng auto-commit/auto-push trong script hoặc workflow của repo.
- Trước khi chạy Shot Detection phân tán, bảng **Điều phối worker Shot Detection** trong
  `docs/CURRENT_STATUS.md` phải ghi người phụ trách + phạm vi ID cho từng worker và các tập ID
  phải không giao nhau. Worker còn `CHƯA PHÂN CÔNG`/`DISABLED` tuyệt đối không được khởi chạy.
- Visual embedding/FAISS phải có checkpoint riêng cho từng modality và được phép
  publish/resume độc lập theo `keyframe_uid`; không tạo barrier chờ cả 3 modality. Trước khi
  đánh dấu checkpoint/resume verified, bắt buộc demo ngắt giữa batch rồi chạy lại thật.
- Logic CLIP rollback, SRRF và `score_visual` thuộc Nhánh 2 online; Nhánh 1 chỉ build artifact
  độc lập, không tự thêm fusion/rollback vào embedding hoặc FAISS builder.
