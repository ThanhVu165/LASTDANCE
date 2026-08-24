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
- **Máy tham chiếu:** Intel i5-12450H, RTX 4050 Laptop **6 GiB VRAM**, Python 3.11,
  Windows/Kaggle/Colab.
- **Thời gian còn lại:** tính từ 23/08/2026, hạn 6 ngày cho toàn bộ pipeline offline + online.

---

## 2. Nguồn chuẩn duy nhất (Source of Truth)

Ba file sau nằm ở `docs/` trong repo — **PHẢI đọc cả 3 trước khi sửa code liên quan đến
offline pipeline**, không suy đoán, không tự sáng tác schema mới:

1. `docs/BASELINE_SPEC.md` — tổng thể toàn hệ thống (offline + online + fusion).
2. `docs/OFFLINE_INDEXING_SPEC.md` — chi tiết Nhánh 1 (phạm vi làm việc chính của Codex
   trong repo này).
3. `docs/ASR_SPEC.md` — chi tiết Nhánh 3, chạy độc lập, chỉ cần biết **data contract**
   (`asr.sqlite`) để không phá vỡ format khi Nhánh 2 tích hợp.

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
- `offline/` — AutoShot shot detection, keyframe extraction, dedup/lọc nhiễu, visual
  embedding (CLIP/SigLIP/BEiT-3), OCR (Gemini + EasyOCR fallback), build FAISS + SQLite FTS5.
- `shared/schemas/` — Pydantic schema dùng chung (`FrameRecord`, `OcrResult`, `AsrSegment`)
  — sửa ở đây ảnh hưởng cả 2 nhánh, cần cẩn trọng và thông báo khi đổi.
- `scripts/` — script chạy batch trên Kaggle, tách nhỏ danh sách video song song.

---

## 4. Nguyên tắc bất di bất dịch (không được vi phạm dù tối ưu thế nào)

1. **Không mean-pool nhiều keyframe/shot thành 1 vector.** Mỗi keyframe có vector riêng.
2. **`keyframe_uid`** là khóa nội dung (deterministic hash `blake2b` từ
   `video_id:shot_id:local_idx`) dùng cho FAISS/OCR/ASR — **không dùng vị trí insert
   (`faiss_row_id`/row index)** làm khóa. Xem công thức chính xác ở
   `OFFLINE_INDEXING_SPEC.md` §3.2a.
3. **`local_idx`** chỉ dùng nội bộ để trỏ file ảnh (`{video_id}/{shot_id}_{local_idx}.jpg`)
   — không dùng làm khóa nộp bài hay dedup.
4. **Không hardcode path tuyệt đối** — mọi path build từ biến môi trường `AIC_DATA`
   (default `data/`).
5. **Không giả định FPS/resolution/duration** — luôn đọc thật từ `ffprobe` (bước Inventory).
6. **Không giả định VRAM vô hạn** — 6GB VRAM local, phải ghi rõ ước tính và cơ chế
   load/release model theo pha nếu chạy local.
7. **Vector bắt buộc ép `float16`** trước khi lưu/push (giảm dung lượng + băng thông HF).
8. **3 FAISS index (CLIP/SigLIP/BEiT-3) build độc lập, không cần đồng bộ thời gian/thứ tự**
   — vì dùng `keyframe_uid` làm khóa qua `IndexIDMap`.
9. **Một `video_id` chỉ `complete = true` khi thỏa đủ Publishing Criteria** (mục 6 dưới) —
   không set `complete=true` khi còn checkpoint dở dang, không được bỏ bớt điều kiện để
   tiết kiệm thời gian.

---

## 5. Môi trường vận hành thực tế — Kaggle + HuggingFace Dataset

- **Keyframe extraction + dedup** → chạy **local CPU**. Shot Detection mặc định vẫn chạy
  CPU local, nhưng được phép chia sang **Colab CUDA** khi dùng đúng cùng commit/config và
  toàn bộ 5 video dev-subset đã qua parity 100% từng shot/range với manifest CPU. CUDA phải
  được chọn tường minh, ghi provenance và fail closed; tuyệt đối không fallback âm thầm về
  CPU.
- **Embedding (CLIP/SigLIP/BEiT-3) + OCR fallback (EasyOCR)** → chạy **Kaggle GPU theo
  batch**. Quota Kaggle free: **30h/tuần** — Nhánh 1 và Nhánh 3 (ASR) dùng **2 tài khoản
  Kaggle/Colab riêng biệt**, không tranh chấp quota với nhau.
- **Đồng bộ artifact Kaggle ↔ máy local** qua **HuggingFace Dataset (Git LFS)**:
  ```
  Kaggle Notebook (build) --push_to_hub()--> HF Dataset (Git LFS) --snapshot_download()--> Local
  ```
  Bắt buộc:
  - Vector đã ép `float16` trước khi push.
  - **Không push từng video một** — gom batch (50–100 video/lần hoặc cuối mỗi phiên Kaggle).
  - Chỉ `snapshot_download()` full repo về local 1 lần trước khi thi, không pull lại giữa
    chừng trừ khi có patch khẩn.
  - Đặt tên revision/commit theo batch (`batch-01`, `batch-02`...) để rollback không phải
    re-push toàn bộ.

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

- **AutoShot model weight chỉ có trên Baidu Netdisk** (không có trên GitHub Release/Google
  Drive, cần passcode). Repo gốc `wentaozhu/AutoShot` không có `pip install` sẵn, không có
  script inference đơn giản cho 1 video — chỉ có script eval trên benchmark SHOT dataset,
  cần tự viết wrapper inference riêng cho 873 video của tụi mình.
  **Trạng thái hiện tại: đang thử tải Baidu trước.** Nếu không tải được trong thời gian hợp
  lý, phương án dự phòng đã bàn: **TransNetV2** (có pip package + weight GitHub, dễ triển
  khai hơn, AutoShot chỉ hơn ~4.2% F1 trên benchmark riêng của họ — không phải chênh lệch
  bắt buộc phải theo bằng mọi giá). Runtime dev hiện dùng `transnetv2-pytorch==1.0.5`, là
  port PyTorch bên thứ ba chứ không phải TensorFlow SavedModel gốc của tác giả. Khi A/B với
  AutoShot phải ghi rõ đây là AutoShot gốc so với TransNetV2 port; không dùng kết quả này để
  suy ngược hoặc xác nhận chênh lệch F1 trong paper giữa hai implementation gốc.
- **EasyOCR full 873 video có thể không kịp 6 ngày** — ưu tiên Gemini API OCR trước,
  EasyOCR chạy nền song song làm fallback, không chặn critical path.
- **Thể lệ AIC 2026 có internet trong phòng thi hay không — CHƯA XÁC NHẬN.** Ảnh hưởng trực
  tiếp đến việc Gemini API (OCR + Query Planning) là primary hay phải coi Qwen3-VL local là
  primary thật sự. Không code phần phụ thuộc giả định "chắc chắn có internet" cho đến khi
  có xác nhận.

---

## 8. Việc KHÔNG được tự ý làm

- Không đổi `keyframe_uid` trở lại kiểu positional/row-index.
- Không tự thêm audio captioning phi ngôn ngữ (BEATs) hay speaker diarization — đã bị cắt
  khỏi scope Nhánh 3 để kịp deadline.
- Không tự ý build thêm 1 visual index đã gộp sẵn (SRRF) ở Nhánh 1 — việc gộp 3 điểm
  CLIP/SigLIP/BEiT-3 thành `score_visual` là việc của Nhánh 2 lúc query, không phải lúc index.
- Không đổi tên cột SQL giữa `ocr.sqlite` và `asr.sqlite` mà không đối chiếu cả 2 spec —
  hai bảng phải giữ cấu trúc song song (`detected_text` vs `transcribed_text`) để Nhánh 2
  dùng chung 1 module `FtsSearcher`.

---

## 9. Khi bắt đầu 1 task cụ thể trong phiên này

Trước khi code, Codex nên tự trả lời (và nói rõ trong phản hồi):
1. Task này thuộc bước nào trong pipeline (`OFFLINE_INDEXING_SPEC.md` §1–3)?
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
