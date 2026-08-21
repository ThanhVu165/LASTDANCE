# Báo cáo kiểm thử end-to-end KIS và Q&A — 21/08/2026

> Đây là evidence snapshot tại thời điểm kiểm thử, không phải roadmap hoặc mô tả
> runtime hiện tại. Kiến trúc chuẩn nằm trong `SYSTEM_ARCHITECTURE.md`, trạng thái
> mới nhất nằm trong `CURRENT_STATUS.md`.

## Kết luận

Hệ thống **hoạt động đúng ở mức vận hành**: frontend gọi được backend, CUDA/Qwen
sẵn sàng, API trả và render Top 100, ảnh tải được, submission xuất đúng 100 dòng.
Tuy nhiên hệ thống **chưa đạt ở mức chất lượng ngữ nghĩa**, đặc biệt là Q&A và KIS
nhiều scene. Không nên coi `HTTP 200` hoặc đủ 100 dòng là truy vấn đúng.

## Môi trường

- Backend: `C:\LASTDANCE\backend\.venv\Scripts\python.exe`.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU, CUDA 13.0.
- VQA: `Qwen/Qwen3-VL-2B-Instruct`, `cuda:0`.
- Sau chuỗi test: 4.457/6.141 MiB VRAM, `vqa_ready=true`.
- Nguồn hình dạng query: `DanhSachTruyVanAIC_Chungket.xlsx`, `Sheet1!A1:C30`.

## KIS

| Case | Thời gian | Kết quả cấu trúc | Đánh giá ngữ nghĩa |
|---|---:|---|---|
| Thiện nguyện, tiếng Việt | 58,509 s | 100 dòng, 32 video, ảnh hợp lệ | PASS: `L22_V004` Top 1, exact `frame_id=810` |
| Thiện nguyện, tiếng Anh | 27,783 s | 100 dòng, 27 video, ảnh hợp lệ | PASS: `L22_V004` Top 1 |
| `Múa lân` | 31,050 s; cache nóng 4,48 s | 100 dòng, UI render đủ #1–#100 | PASS trực quan: `L24_V023`, local 14 là múa lân |
| Lễ hội Nhật trên thuyền, quạt quốc kỳ, khiêng cá | 57,816 s | 100 dòng, 43 video | FAIL Top 1: `L22_V013` là đoàn rước/lễ hội Việt Nam, thiếu thuyền Nhật và cá |
| Thùng tái chế pin + chữ `COLES` khi chưa có OCR index | 28,070 s | 100 dòng | FAIL Top 1 trực quan; có một cặp `(video_id, frame_id)` bị lặp |

KIS ngắn và canary có mục tiêu biết trước hoạt động tốt. KIS dài vẫn có thể chọn
video chỉ đúng chủ đề chung "lễ hội" nhưng bỏ các thuộc tính phân biệt.

## Q&A

| Case | Thời gian | Retrieval/frame | Answer |
|---|---:|---|---|
| Lâu đài Bavaria, hỏi thương hiệu | 51,765 s | FAIL: Top 1 là video bài giảng/cảnh núi, không phải lâu đài cần tìm | FAIL: `THANHNIEN`, kỳ vọng `DISNEY` |
| Lễ hội silleta, hỏi thành phố | 96,282 s cold; 25,800 s warm | Tìm đúng video `L22_V012`, nhưng Top 1 là frame mở đầu bản tin; frame lễ hội ở Top 2 | FAIL: sao chép `BOGOTA` từ đề thay vì suy ra `MEDELLIN` |
| Sandwich trái cây cuộn chiên, hỏi số phần | 45,936 s | FAIL trực quan: Top 1 là bánh mì kẹp thông thường, không có chuỗi dưa hấu–thơm–lê–cuộn–chiên | `2` không có bằng chứng đáng tin |

Tỉ lệ `UNKNOWN` trong hai case đã đo lần lượt là 81/100, 56/100 và 52/100.
Q&A hiện có hai kiểu lỗi độc lập:

1. Retrieval chọn video chỉ khớp chủ đề chung.
2. Khi video đúng, model vẫn có thể chọn sai frame và sao chép một thực thể trong
   đề thay vì trả lời câu hỏi suy luận.

## Protocol, dữ liệu và submission

- `text=""` và thiếu trường `text`: trả 422 đúng.
- `text` chỉ chứa whitespace: KIS và Q&A trả 500; traceback đi từ
  `parse_semantic_query` / `parse_qa_query` vì router không chuyển `ValueError`
  thành 422.
- Index có 177.321 dòng nhưng có 614 nhóm trùng `(video_id, frame_id)`, ảnh hưởng
  192 video. Mỗi nhóm có hai local keyframe cùng trỏ một frame nguồn. Ranking hiện
  deduplicate theo `local_idx`, vì vậy có thể lãng phí dòng Top 100.
- Validator submission chưa phát hiện cặp video/frame trùng và không bắt buộc đúng
  100 dòng.
- Vòng KIS thực tế `search -> add -> validate -> export -> clear` thành công:
  100 kết quả, 100 dòng được thêm, validator OK, CSV đúng 100 dòng UTF-8.

## Độ ổn định

- Một backend đã chạy lâu trả HTTP 500 cho QA lễ hội sau 40,123 giây và lặp lại
  ngay sau 0,344 giây.
- Sau khi khởi động sạch bằng đúng `.venv`, cùng truy vấn chạy thành công ba lần
  liên tiếp qua API/UI. CUDA vẫn sẵn sàng sau test.
- Không thu được traceback của lỗi trạng thái ban đầu vì tiến trình cũ thuộc terminal
  khác. Cần logging theo request và cơ chế giải phóng/khôi phục CUDA để xác định dứt
  điểm nếu lỗi tái diễn.

## Frontend và regression

- Streamlit KIS: nhập query, hiển thị đủ #1–#100, ảnh và source-frame tải được.
- Streamlit Q&A: nhập query, hiển thị đủ #1–#100 và answer; kết quả sai `BOGOTA`
  giống API, nên frontend không làm biến đổi payload.
- Toàn bộ trang frontend qua `py_compile`.
- Backend tại checkpoint này: 56/56 test qua trong 11,24 giây. Sau cleanup dead
  code, suite chuẩn hiện là 55 test; xem `CURRENT_STATUS.md`.

## Ánh xạ lỗi vào kiến trúc hiện tại

1. KIS multi-scene sai vì frame recall/video hypothesis chưa phủ đủ scene: ưu tiên
   shot-aware video-window embedding và query–window reranker.
2. QA phải nhận verified window từ shared retrieval rồi mới answer; thêm answer
   verification để ngăn copy thực thể không được evidence hỗ trợ.
3. Query chứa chữ cần OCR/structured caption lexical evidence, nhưng OCR chỉ là
   một modality trong offline evidence store.
4. Deduplicate `(video_id, frame_id)`, whitespace 422, submission validation và
   request logging là các lỗi contract độc lập, cần regression test riêng.
5. Acceptance gate và thứ tự triển khai nằm trong `DEVELOPMENT_ROADMAP.md`.
