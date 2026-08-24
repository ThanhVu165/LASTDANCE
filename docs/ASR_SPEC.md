# Nhánh 3 — ASR Pipeline (Bản chốt)

> File này là nguồn chuẩn riêng cho **kênh Audio/ASR**, tách từ `BASELINE_SPEC.md`. Trước
> đây ASR bị hoãn hoàn toàn để dồn quota GPU cho Nhánh 1 (Offline Indexing). Nay tách thành
> **Nhánh 3 độc lập**, chạy song song, dùng tài khoản Kaggle/Colab riêng — không tranh chấp
> tài nguyên với Nhánh 1.

**Cập nhật:** 23/08/2026
**Lý do tồn tại:** không có ASR, hệ thống bất lực với các câu hỏi KIS/QA có mốc chứng cứ
duy nhất nằm ở lời hội thoại nhân vật — đây là lỗ hổng recall tự động, không thể bù hoàn
toàn bằng human-in-the-loop khi thời gian thi có hạn.

---

## 0. Nguyên tắc bắt buộc

1. **Chạy trên tài khoản Kaggle/Colab riêng, độc lập với Nhánh 1.** Không dùng chung quota
   GPU 30h/tuần với nhánh Visual Embedding — đây là lý do duy nhất khiến việc tách nhánh này
   khả thi.
2. **Không tạo định dạng lưu trữ riêng.** Output phải là SQLite FTS5 (`asr.sqlite`), cùng
   cấu trúc truy vấn với `ocr.sqlite` đã có — để Nhánh 2 chỉ cần thêm 1 kênh search vào fusion
   logic sẵn có, không phải viết module đọc dữ liệu mới.
3. **Join ngược về `frames.csv` qua `keyframe_uid`**, không tự chế khóa riêng.
4. **Scope rút gọn — chỉ làm phần lõi**, bỏ phần nâng cao (audio captioning phi ngôn ngữ) để
   đảm bảo kịp tiến độ trong 6 ngày.
5. **Human-in-the-loop vẫn giữ làm backup** — nếu Nhánh 3 không kịp xử lý hết 873 video, phần
   còn thiếu vẫn có thể bù bằng việc thí sinh tự nghe lại video lúc thi.

---

## 1. Scope — làm gì, bỏ gì

| Việc | Quyết định | Lý do |
|---|---|---|
| Audio extraction (FFmpeg, tách `.wav`) | ✅ Làm | Nhanh, chạy CPU, không tốn GPU quota |
| Whisper Large-v3 transcription + timestamp | ✅ Làm | Lõi của nhánh — lấp đúng lỗ hổng KIS/QA hội thoại |
| phoWhisper (tiếng Việt) | ✅ Ưu tiên nếu video chủ yếu tiếng Việt | Độ chính xác hội thoại tiếng Việt cao hơn Whisper gốc |
| Temporal Alignment (map segment → keyframe gần nhất) | ✅ Bắt buộc | Không làm thì dữ liệu ASR không join được với `frames.csv`, vô dụng |
| BEATs (audio captioning phi ngôn ngữ: mưa, còi xe, tiếng khóc...) | ❌ Bỏ | Nâng cao, không phải blocker của KIS/QA hội thoại, tốn thời gian không cần thiết trong 6 ngày |
| Speaker diarization (phân biệt ai nói) | ❌ Bỏ | Không có trong scope thi, không ưu tiên |

---

## 2. Pipeline chi tiết

```
video (.mp4)
  └─> [1] Audio Extraction: FFmpeg tách audio -> .wav/.mp3 (local CPU)
  └─> [2] ASR: Whisper Large-v3 hoặc phoWhisper (Kaggle/Colab GPU riêng)
          -> segment text + start_time + end_time
  └─> [3] Temporal Alignment: map mỗi segment -> keyframe_uid gần nhất
          theo pts_time (tra cứu frames.csv)
  └─> [4] Ghi vào asr.sqlite (SQLite FTS5, cùng cấu trúc ocr.sqlite)
```

| Bước | Model/Thư viện | Hành động | Output |
|---|---|---|---|
| 1 | FFmpeg | Tách luồng audio gốc ra khỏi video | `.wav` (16kHz mono, chuẩn input Whisper) |
| 2 | Whisper Large-v3 / phoWhisper | Speech-to-text, sinh segment kèm timestamp | Danh sách `(start_time, end_time, text)` |
| 3 | Python (Pandas) | Với mỗi segment, tìm `keyframe_uid` có `pts_time` gần nhất trong khoảng `[start_time, end_time]` (hoặc gần `start_time` nhất nếu không có keyframe nằm trong khoảng) | Segment đã gắn `keyframe_uid` |
| 4 | SQLite FTS5 | Insert vào bảng ảo hỗ trợ BM25 full-text search | `asr.sqlite` |

---

## 3. Data Contract — bàn giao cho Nhánh 2

### 3.1 Schema `AsrSegment`

```python
# shared/schemas/asr.py
from pydantic import BaseModel

class AsrSegment(BaseModel):
    video_id: str
    segment_id: str
    start_time: float
    end_time: float
    transcribed_text: str
    language: str              # "vi" | "en"
    keyframe_uid_nearest: int  # join ngược về frames.csv
```

### 3.2 `asr.sqlite` — cấu trúc bảng

```sql
CREATE VIRTUAL TABLE asr_fts USING fts5(
    video_id UNINDEXED,
    segment_id UNINDEXED,
    transcribed_text,
    language UNINDEXED,
    keyframe_uid_nearest UNINDEXED,
    start_time UNINDEXED,
    end_time UNINDEXED
);
```

Giữ đúng tên cột kiểu với `ocr.sqlite` (nếu `ocr.sqlite` dùng `detected_text`, cân nhắc đặt
tên cột nội dung ASR là `transcribed_text` để phân biệt rõ nguồn — tránh nhầm 2 bảng khi
Nhánh 2 code chung 1 module `FtsSearcher` cho cả OCR và ASR).

### 3.3 Bàn giao

1. **`asr.sqlite`** — sẵn sàng để `FtsSearcher` (module dùng chung với OCR) query trực tiếp.
2. **Coverage report** — file `.csv` liệt kê `video_id` nào đã xử lý xong ASR, `video_id` nào
   chưa (do hết thời gian) — để Nhánh 2 biết video nào cần fallback human-in-the-loop khi thi.

---

## 4. Publishing Criteria

- [ ] Mỗi `video_id` đã xử lý: số segment > 0 (trừ video thật sự không có thoại/âm thanh —
      cần đối chiếu bằng tai nghe thử vài video để phân biệt "không có thoại" với "lỗi ASR")
- [ ] `keyframe_uid_nearest` của mọi segment tồn tại thật trong `frames.csv` (không bị lệch
      do sai công thức tìm gần nhất)
- [ ] `asr.sqlite` build FTS5 thành công, test thử 1 câu truy vấn mẫu trả về đúng kết quả
- [ ] Coverage report cập nhật đúng — không đánh dấu "đã xử lý" cho video còn dở dang

---

## 5. Tích hợp vào Online Pipeline (tham chiếu, chi tiết ở `BASELINE_SPEC.md` mục 3.2)

- Kênh ASR chỉ được **tăng trọng số** trong `modality_weights` khi `UnifiedQueryPlan` (do
  `QueryPlanner` sinh ra) phát hiện query có `spoken_text` không rỗng — tức câu hỏi có nhắc
  đến lời thoại/nhân vật nói (ví dụ: "nhân vật nói...", "lời thoại...", "câu nói...").
- Nếu 1 video chưa có `asr.sqlite` coverage (theo coverage report mục 3.3), hệ thống Online
  cần hiển thị rõ trên UI để thí sinh biết cần tự nghe lại video (human-in-the-loop), không
  im lặng trả về rỗng khiến thí sinh tưởng video không có thoại.

---

## Changelog

- **23/08/2026** — Tạo mới. Tách kênh ASR thành Nhánh 3 độc lập (trước đó bị hoãn hoàn toàn
  trong `BASELINE_SPEC.md`). Scope rút gọn: chỉ Whisper/phoWhisper + temporal alignment, bỏ
  BEATs và speaker diarization. Human-in-the-loop giữ làm backup cho phần video chưa kịp xử
  lý.
- **23/08/2026 (bản 2)** — Sửa xung đột phát hiện khi audit chéo 3 file: thêm cột `language`
  còn thiếu vào bảng `asr_fts` (mục 3.2) — trước đó có trong `AsrSegment` Pydantic nhưng
  chưa có trong schema SQL, gây mất dữ liệu khi insert thật.
- **23/08/2026 (bản 3)** — `BASELINE_SPEC.md` bổ sung tầng "intra-visual fusion" (SRRF) và
  hệ số BLIP-2 verification trong QA gating — cả hai chỉ nằm ở nhánh visual/QA, **không**
  đổi gì ở kênh ASR. Đã audit chéo và xác nhận: kênh ASR vẫn tham gia Late Fusion tầng 2
  (`BASELINE_SPEC.md` §3.2) đúng như cũ, không có việc phát sinh cho Nhánh 3.
