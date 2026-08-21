# KẾ HOẠCH XÂY DỰNG HỆ THỐNG DỰ THI — AIC2026 (VÒNG SƠ TUYỂN)

> Nguồn tham khảo:
> 1. `docs/Thong tin vong So tuyen AIC2026.pdf` — mô tả 3 dạng truy vấn & cách chấm điểm.
> 2. https://sotuyenaic.oj.io.vn/ — hướng dẫn nộp bài chính thức (định dạng file, cách đóng gói, luật nộp).
>
> **Trạng thái: bắt đầu lại từ đầu.** Kế hoạch này thay thế toàn bộ bản trước, tập trung
> tối ưu phần **re-rank** để đưa kết quả truy vấn tốt hơn, đồng thời khớp chính xác với
> quy cách nộp bài thực tế của BTC.

---

## 1. TÓM TẮT ĐỀ BÀI

### 1.1. Ba dạng truy vấn

| Dạng truy vấn | Đầu vào | Format 1 dòng kết quả | Ghi chú |
|---|---|---|---|
| **Textual KIS** | 1 câu mô tả sự kiện | `<video_name>, <frame_id>` | Không có phần đuôi `.mp4` trong tên video |
| **Q&A** | mô tả sự kiện + câu hỏi | `<video_name>, <frame_id>, <answer>` | Answer ≤ 100 ký tự, so khớp **ngữ nghĩa**, có thể VI/EN |
| **TRAKE** | chuỗi N sự kiện tuần tự | `<video_name>, <frame_id_1>, ..., <frame_id_N>` | Số lượng frame phải khớp đúng số event yêu cầu, đúng thứ tự thời gian |

### 1.2. Cách chấm điểm

- **R-Score** mỗi dòng (0–1):
  - KIS: 1 nếu đúng video và `frame_id ∈ [s, e]`, ngược lại 0.
  - Q&A: 1 nếu đúng video + đúng khoảng frame + đúng answer về ngữ nghĩa, ngược lại 0.
  - TRAKE: sai video → 0 điểm toàn bộ dòng; đúng video → điểm = tỉ lệ số khoảnh khắc
    khớp đúng khoảng `[sⱼ, eⱼ]` trong tổng N khoảnh khắc (mỗi đoạn thường rất ngắn, <10 frame).
- **Final Score** = trung bình `R@k` với `k ∈ {1, 5, 20, 50, 100}`, `R@k = max` R-Score
  trong k dòng đầu tiên của danh sách đã nộp.
  → **Ý nghĩa cho thiết kế**: đáp án đúng phải được xếp **càng gần đầu danh sách càng
  tốt**, R@1 ảnh hưởng nặng nhất đến Final Score. Đây là **lý do chính khiến module
  re-rank là phần quan trọng nhất của toàn hệ thống**, không chỉ retrieval thô.

### 1.3. Quy cách nộp bài (theo hướng dẫn chính thức)

- Mỗi gói truy vấn: BTC cấp các file `.txt` chứa câu truy vấn, đặt tên có hậu tố
  loại truy vấn: `query-1-kis.txt`, `query-2-kis.txt`, `query-3-qa.txt`, `query-4-trake.txt`...
- Đội thi nộp lại **1 file `.csv` tương ứng cho mỗi file truy vấn**, cùng tên gốc nhưng
  đuôi `.csv` (ví dụ `query-1-kis.csv`), tối đa **100 dòng**.
- Quy tắc CSV: encoding UTF-8, delimiter `,`, **không có header**, tên video **không**
  có đuôi `.mp4`, frame_id so sánh dạng số nguyên, answer so sánh dạng chuỗi ngữ nghĩa
  (≤100 ký tự). Answer cần bọc `"..."` khi chứa dấu phẩy/ngoặc kép/xuống dòng.
- Đóng gói: tất cả CSV cho vào thư mục `submission/`, rồi nén `submission/` thành
  1 file `.zip` (không nén trực tiếp các file CSV).
- Nộp qua tài khoản BTC cấp trên hệ thống thi (https://sotuyenaic.oj.io.vn/), **tối đa
  3 lần nộp/gói truy vấn**, tính điểm theo **lần nộp cuối cùng**. Nộp sai định dạng
  vẫn tính là 1 lần nộp → hệ thống nội bộ cần validate định dạng trước khi xuất/nộp
  để tránh mất lượt.
- Public Leaderboard: 50% đáp án BTC. Private Leaderboard (xếp hạng chính thức vòng
  sơ tuyển): 100% đáp án.

### 1.4. Dữ liệu được cung cấp

```
Videos/       *.mp4 — dữ liệu thi chính thức
Keyframes/    <video_id>/<NNNN>.jpg — thứ tự tăng dần, vị trí frame ghi trong metadata
Objects/      <video_id>/<NNNN>.json — object detection (Faster R-CNN, OpenImages V4)
CLIP features/  1 file .npy duy nhất, thứ tự vector khớp thứ tự keyframe toàn cục
Metadata/     <video_id>.json — lấy từ YouTube, có thể thiếu ở một số video
```
Model CLIP dùng để trích feature: **clip-ViT-B-32**. Dữ liệu mẫu là batch 1 AIC2025;
batch 2 sẽ bổ sung sau → pipeline cần hỗ trợ cập nhật/tái sử dụng index tăng dần.

---

## 2. PHẠM VI MVP

| Thành phần | Lựa chọn |
|---|---|
| Backend | Python + **FastAPI** |
| Frontend | **Streamlit** (đơn giản, đủ dùng để tra cứu, chọn kết quả & xuất bài nộp) |
| Kênh tìm kiếm cốt lõi | (1) CLIP text→image search; (2) lọc/boost theo Object detection; (3) OCR khung hình |
| Yêu cầu bắt buộc | Hoàn thiện **cả 3 dạng truy vấn** (KIS, QA, TRAKE), không dở dang |
| Đầu ra | File `.csv` đúng chuẩn BTC (mục 1.3), đóng gói đúng cấu trúc `submission/` + `.zip` |
| **Trọng tâm ưu tiên** | **Tối ưu module re-rank** để tối đa hoá Final Score (R@1 → R@100) |

---

## 3. KIẾN TRÚC HỆ THỐNG

```
                     ┌─────────────────────────────────────────┐
                     │        DỮ LIỆU GỐC (data/)               │
                     │ videos/ keyframes/ objects/ features/ metadata/ │
                     └───────────────────┬───────────────────────┘
                                         ▼
                     ┌─────────────────────────────────────────┐
                     │   TIỀN XỬ LÝ OFFLINE (indexing/)          │
                     │  build_index.py : keyframe index + FAISS  │
                     │  ocr_index.py   : cache OCR text/keyframe │
                     └───────────────────┬───────────────────────┘
                                         ▼
        ┌───────────────────────────────────────────────────────────┐
        │        CORE RETRIEVAL SERVICES (dùng chung 3 pipeline)      │
        │  clip_search.py | object_filter.py | ocr_search.py           │
        │  query_processing.py (xử lý ngữ nghĩa truy vấn — mục 4)     │
        └───────────────────┬───────────────┬───────────────┬────────┘
                             ▼               ▼               ▼
                  ┌─────────────────┐ ┌─────────────┐ ┌──────────────────┐
                  │ kis_pipeline.py │ │qa_pipeline.py│ │trake_pipeline.py │
                  └────────┬────────┘ └──────┬──────┘ └────────┬─────────┘
                           └─────────────────┬┴──────────────────┘
                                              ▼
                          ┌─────────────────────────────────┐
                          │   RERANK ENGINE (mục 5 — trọng tâm) │
                          │  fusion scoring, dedup, diversity,  │
                          │  temporal smoothing, DP alignment   │
                          └───────────────────┬─────────────────┘
                                              ▼
              ┌───────────────────────────────────────────────────────┐
              │  FASTAPI ROUTERS: /search/kis  /search/qa  /search/trake │
              │  + /video/.../frame/...  + /submission (CRUD + export)   │
              │  export module: sinh đúng file .csv / .zip theo mục 1.3  │
              └───────────────────────────┬───────────────────────────┘
                                          ▼
              ┌───────────────────────────────────────────────────────┐
              │  STREAMLIT UI — 4 trang: KIS | Q&A | TRAKE | Export      │
              └───────────────────────────────────────────────────────┘
```

---

## 4. XỬ LÝ NGỮ NGHĨA TRUY VẤN (Query Understanding)

Dùng chung cho cả 3 pipeline, thực hiện **trước** retrieval — chất lượng bước này
quyết định trần điểm số mà rerank có thể đạt tới:

1. **Đọc file truy vấn `.txt`** theo đúng quy ước tên (`query-N-kis.txt`,
   `query-N-qa.txt`, `query-N-trake.txt`) để tự động nhận diện loại truy vấn và số
   thứ tự gói, tránh nhập tay sai định dạng.
2. **Chuẩn hoá & dịch ngôn ngữ**: câu mô tả tiếng Việt → dịch sang tiếng Anh để
   encode CLIP (model hiểu tiếng Anh tốt hơn); vẫn giữ câu gốc tiếng Việt để so khớp
   OCR/sinh answer tiếng Việt.
3. **Tách thành phần ngữ nghĩa**: đối tượng chính, thuộc tính (màu sắc/trang phục/
   số lượng), hành động, bối cảnh (trong nhà/ngoài trời...).
   - Q&A: tách riêng "mô tả sự kiện" và "câu hỏi" để định tuyến đúng kiểu
     answer-extraction.
   - TRAKE: tách chuỗi mô tả thành N mô tả khoảnh khắc con theo thứ tự liệt kê.
4. **Query expansion**: sinh biến thể diễn đạt đồng nghĩa (VD "áo đỏ" → "red shirt",
   "man in red"), encode nhiều biến thể, lấy điểm tốt nhất (max-pooling) → tăng độ
   ổn định đầu vào cho rerank.
5. **Trích từ khoá cho object/OCR filter**: map từ khoá sang nhãn gần nhất trong tập
   nhãn OpenImages, và từ khoá tra cứu OCR.

---

## 5. MODULE RE-RANK — TRỌNG TÂM TỐI ƯU (chi tiết theo yêu cầu)

Vì Final Score ưu tiên nặng vào R@1/R@5, rerank không thể chỉ là "cộng điểm tuyến
tính" đơn giản. Thiết kế theo nhiều tầng, có thể bật/tắt độc lập để dễ thử nghiệm:

### 5.1. Tầng 1 — Candidate generation (đầu vào cho rerank)
- Lấy top-N lớn từ CLIP (N ≈ 300–500) thay vì chỉ top-100, để rerank có đủ không
  gian "vớt" đáp án đúng bị retrieval thô xếp thấp.
- Chạy retrieval trên **nhiều biến thể query expansion** (mục 4.4), gộp candidate
  theo `(video_id, frame_id)`, giữ điểm cao nhất mỗi ứng viên.

### 5.2. Tầng 2 — Fusion scoring (điểm tổng hợp đa tín hiệu)
Kết hợp nhiều tín hiệu độc lập thay vì chỉ dựa CLIP:
- **CLIP similarity** (tín hiệu chính, trọng số cao nhất).
- **Object-match score**: đối chiếu nhãn vật thể được nhắc trong mô tả với nhãn
  Object detection tại keyframe (boost nếu khớp, phạt nhẹ nếu không có object nào
  liên quan dù ảnh có object detection).
- **OCR-match score**: đối chiếu từ khoá/chữ trong mô tả với text OCR trích từ
  keyframe (quan trọng với truy vấn nhắc biển hiệu, chữ trên màn hình...).
- **Temporal smoothing theo video**: nếu các keyframe liền kề trong cùng 1 video
  đều có điểm cao → đây là tín hiệu mạnh rằng cả đoạn video đúng ngữ cảnh, dùng để
  nâng điểm đồng loạt cho các frame trong đoạn đó (tăng khả năng chọn đúng frame dù
  frame "đỉnh" ban đầu không phải frame tối ưu nhất).
- Trọng số fusion ban đầu đặt cố định (heuristic), nhưng thiết kế dạng **có thể học
  lại (learned fusion weights)** khi có tập truy vấn/đáp án mẫu để hiệu chỉnh.

### 5.3. Tầng 3 — Diversity & Deduplication (tối ưu cho R@k ở ngưỡng thấp)
- Giới hạn số kết quả tối đa mỗi video trong top-N kết quả cuối (tránh 1 video
  chiếm hết vị trí đầu bảng khi chỉ có 100 dòng để nộp).
- Áp dụng **Maximal Marginal Relevance (MMR)**: cân bằng giữa độ liên quan (điểm
  fusion) và độ đa dạng (tránh các keyframe gần trùng nhau về nội dung/thời gian)
  để tận dụng tối đa 100 vị trí nộp cho các "giả thuyết" khác nhau.

### 5.4. Tầng 4 — Answer-aware rerank (riêng cho Q&A)
- Với câu hỏi đếm số lượng: dùng số lượng object đếm được làm tín hiệu phụ để
  rerank lại thứ tự các khoảnh khắc ứng viên (ưu tiên khoảnh khắc có số đếm hợp lý
  hơn theo ngữ cảnh câu hỏi).
- Sinh nhiều biến thể answer (số/chữ, VI/EN) cho cùng 1 khoảnh khắc để tăng khả
  năng khớp ngữ nghĩa khi chấm, nhưng **không nhân dòng answer khác nhau cho cùng
  1 frame** — chỉ chọn 1 answer tốt nhất/dòng vì mỗi dòng chỉ có 1 answer.

### 5.5. Tầng 5 — TRAKE-specific rerank (căn chỉnh chuỗi)
- **Chọn video**: tổng hợp điểm fusion của toàn bộ N khoảnh khắc theo từng video
  ứng viên (tổng hoặc trung bình có trọng số theo độ tin cậy từng khoảnh khắc), chọn
  video có tổng điểm cao nhất để giảm rủi ro "mất trắng" (TRAKE chấm 0 điểm toàn bộ
  nếu sai video).
- **Căn chỉnh khung hình**: dùng quy hoạch động (DP, kiểu Dynamic Time Warping) để
  chọn tổ hợp N khung hình có tổng điểm fusion cao nhất **và đảm bảo đúng thứ tự thời
  gian** (ràng buộc monotonic: frame khoảnh khắc sau > khoảnh khắc trước).
- Sinh thêm các biến thể tổ hợp gần-tối-ưu (đổi 1–2 khung hình so với phương án tốt
  nhất) để lấp đầy 100 dòng nộp, tăng cơ hội trúng nhiều khoảnh khắc hơn ở các
  ngưỡng R@5/R@20 trở lên.

### 5.6. Hướng phát triển nâng cao (sau khi rerank cơ bản chạy ổn)
- **Cross-encoder re-ranker**: mô hình chấm điểm ảnh+văn bản đồng thời (thay vì
  2 tháp độc lập như CLIP) để rerank lại top-50/100 ứng viên, chính xác hơn nhưng
  tốn chi phí — chỉ áp dụng cho tập nhỏ đã lọc qua tầng 1–3.
- **Learned fusion weights**: nếu có tập truy vấn/đáp án mẫu (từ đề luyện tập hoặc
  kết quả các đợt nộp trước), huấn luyện trọng số fusion bằng logistic regression/
  gradient boosting thay vì trọng số cố định.
- **Ensemble nhiều embedding model**: kết hợp thêm CLIP ViT-L/14, SigLIP, hoặc mô
  hình đa ngôn ngữ hỗ trợ tiếng Việt trực tiếp, lấy trung bình/bỏ phiếu điểm số.
- **Relevance feedback trong phiên làm việc**: cho phép người dùng đánh dấu kết quả
  đúng/gần đúng trên UI, dùng feedback để re-query (Rocchio-like) ngay trong phiên.
- **Temporal alignment nâng cao cho TRAKE**: thay DP đơn giản bằng mô hình chuỗi
  (HMM/CRF) học khoảng cách thời gian hợp lý giữa các khoảnh khắc liên tiếp dựa trên
  thống kê dữ liệu mẫu.

---

## 6. CHI TIẾT TỪNG PIPELINE (tổng hợp candidate + rerank ở mục 5)

### 6.1. Textual KIS
Candidate generation (5.1) → Fusion scoring (5.2) → Diversity/dedup (5.3) →
xuất tối đa 100 dòng `(video_name, frame_id)` xếp hạng theo điểm cuối.

### 6.2. Q&A
Candidate + fusion + diversity như KIS → Answer extraction (đếm object cho câu hỏi
số lượng, OCR/nhận diện màu sắc cho câu hỏi nội dung) → Answer-aware rerank (5.4)
→ xuất tối đa 100 dòng `(video_name, frame_id, answer)`.

### 6.3. TRAKE
Candidate + fusion cho từng khoảnh khắc con (trong toàn bộ dataset) → chọn video
(5.5) → DP alignment trong video đã chọn → sinh biến thể gần-tối-ưu → xuất tối đa
100 dòng `(video_name, frame_id_1, ..., frame_id_N)`.

---

## 7. MODULE EXPORT / NỘP BÀI (khớp chính xác yêu cầu BTC)

- Đọc danh sách kết quả đã chọn/xếp hạng cho từng `query_id`, sinh file `.csv`:
  - Không header, encoding UTF-8, delimiter `,`.
  - Tên video **bỏ đuôi `.mp4`**.
  - Q&A: tự động bọc `"..."` cho answer nếu chứa dấu phẩy/ngoặc kép/xuống dòng;
    escape `"` thành `""` bên trong.
  - Tên file output = tên file query gốc nhưng đổi đuôi `.txt` → `.csv`
    (ví dụ `query-3-qa.txt` → `query-3-qa.csv`).
- **Validator trước khi xuất**: kiểm tra tối đa 100 dòng, đúng số cột theo loại
  truy vấn, TRAKE đúng số Frame ID theo số event yêu cầu, answer ≤ 100 ký tự — để
  tránh mất lượt nộp (tối đa 3 lần/gói) do sai định dạng.
- **Đóng gói nộp bài**: gom toàn bộ file `.csv` của gói vào thư mục `submission/`,
  nén thành `.zip` (không nén trực tiếp các file csv) — cung cấp 1 nút "Export &
  Zip" trên UI để tự động hoá bước này, giảm rủi ro thao tác tay sai.

---

## 8. CẤU TRÚC THƯ MỤC DỰ KIẾN

```
LASTDANCE/
  docs/                                 (tài liệu đề thi)
  data/                                 (videos/keyframes/objects/features/metadata/index)
  queries/                              (các file .txt truy vấn BTC cấp theo từng đợt)
  submissions/<round>/submission/       (các .csv sinh ra, sẵn sàng để nén .zip)
  backend/
    requirements.txt
    app/
      config.py
      models.py
      main.py
      indexing/
        build_index.py
        ocr_index.py
      services/
        clip_search.py
        object_filter.py
        ocr_search.py
        query_processing.py
      rerank/
        fusion_scoring.py
        contest_ranking.py
        temporal_smoothing.py
      pipelines/
        kis_pipeline.py
        qa_pipeline.py
        trake_pipeline.py
      routers/
        kis.py | qa.py | trake.py | submission.py
  frontend/
    requirements.txt
    streamlit_app.py
    pages/
      1_Textual_KIS.py
      2_QA.py
      3_TRAKE.py
      4_Export.py
  README.md
```

---

## 9. DANH SÁCH CÔNG VIỆC (todos, xem SQL để theo dõi tiến độ)

Tất cả công việc dưới đây **bắt đầu lại từ `pending`**, chưa có việc nào tính là đã
hoàn thành trong kế hoạch này:

1. Khảo sát cấu trúc dữ liệu mẫu thực tế + tải file truy vấn mẫu từ hệ thống thi.
2. Xây script tiền xử lý: build keyframe index + FAISS từ CLIP `.npy`.
3. Parse Objects JSON thành bảng tra cứu nhãn vật thể theo keyframe (kèm đếm số lượng).
4. Xây OCR index (cache text theo keyframe).
5. Xây core retrieval service dùng chung (clip_search, object_filter, ocr_search).
6. Xây module xử lý ngữ nghĩa truy vấn (query_processing): dịch, tách thành phần,
   query expansion, đọc file `.txt` theo quy ước tên.
7. Xây **module rerank** (fusion scoring, temporal smoothing, diversity/MMR) — đây
   là công việc trọng tâm, cần thử nghiệm & hiệu chỉnh kỹ.
8. Xây pipeline Textual KIS (dùng core retrieval + rerank).
9. Xây pipeline Q&A (retrieval + rerank + answer extraction + answer-aware rerank).
10. Xây pipeline TRAKE (chọn video theo tổng điểm chuỗi + DP alignment + rerank biến thể).
11. Xây router FastAPI riêng cho từng loại truy vấn + API ảnh keyframe/video.
12. Xây module export/validator đúng chuẩn CSV + đóng gói `submission/`.zip theo
    hướng dẫn BTC.
13. Xây UI Streamlit riêng: trang KIS, trang Q&A, trang TRAKE.
14. Xây UI Streamlit trang Export (kèm nút Export & Zip tự động).
15. Viết README hướng dẫn cài đặt, tải dữ liệu, chạy tiền xử lý, chạy app.
16. Kiểm thử end-to-end: chạy thử với dữ liệu mẫu + ví dụ đề bài, đối chiếu định
    dạng CSV với hướng dẫn nộp bài, đo thử R-Score/Final Score nếu có đáp án mẫu.
17. (Backlog, không chặn MVP) Nâng cấp rerank nâng cao: cross-encoder, learned
    fusion weights, ensemble embedding, relevance feedback, temporal HMM/CRF cho TRAKE.

---

## 10. GHI CHÚ / RỦI RO

- File CLIP `.npy` chung cho toàn bộ keyframe → phải đảm bảo đúng thứ tự khi build
  index (ghép theo thứ tự tăng dần từng video, đúng thứ tự global index).
- Một số video không có metadata → xử lý optional, không được làm lỗi pipeline.
- OCR chạy offline bằng EasyOCR CRAFT + `latin_g2` trên GPU; model phải vượt kiểm
  tra đầy đủ bảng chữ cái tiếng Việt trước khi ghi cache. Cần smoke-test và chạy bộ
  benchmark giữ nguyên dấu trước khi chạy toàn bộ; cache schema v2 lưu text theo
  dòng, confidence, polygon, còn state hỗ trợ checkpoint/retry/tiếp tục an toàn.
- Dữ liệu sẽ có thêm batch 2 → thiết kế build_index dạng có thể chạy lại/bổ sung
  incremental, không bắt buộc build lại toàn bộ từ đầu.
- **Tên video trong export phải bỏ đuôi `.mp4`** — dễ sót nếu không kiểm tra kỹ, gây
  sai toàn bộ điểm dù retrieval đúng.
- Chỉ có **tối đa 3 lần nộp/gói truy vấn**, tính theo lần cuối → cần validate kỹ
  định dạng CSV trong hệ thống nội bộ trước khi xuất, tránh nộp sai định dạng làm
  mất lượt oan.
- Rerank là phần ảnh hưởng lớn nhất đến Final Score (do cách tính R@k) → nên dành
  thời gian thử nghiệm/đánh giá nhiều nhất cho mục 5, có thể tách thành nhiều
  phương án bật/tắt để so sánh hiệu quả trước khi chốt cấu hình nộp bài chính thức.
