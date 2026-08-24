# Quyết định lựa chọn model

> **ARCHIVED 24/08/2026:** Bảng model này thuộc kiến trúc window-first cũ. Model budget
> hiện hành nằm trong `BASELINE_SPEC.md`.

Cập nhật: 21/08/2026. Các quyết định dưới đây tối ưu cho RTX 4050 Laptop 6 GiB,
query KIS 2–3 phút, QA/TRAKE 3–5 phút và khả năng build offline không mất phí API.
Model chỉ được thay khi thắng A/B trên dữ liệu video của dự án.

## Kết luận

| Vị trí | Lựa chọn chuẩn | Quyết định |
|---|---|---|
| Video-window embedding | `Qwen/Qwen3-VL-Embedding-2B` | Giữ |
| Structured video/window caption | `Qwen/Qwen3-VL-2B-Instruct` | Giữ |
| Full-collection OCR | EasyOCR CRAFT + `latin_g2` | Giữ làm baseline production |
| OCR challenger | `latin_PP-OCRv5_mobile_rec`; PP-OCRv6 sau khi kiểm tra dictionary/runtime | Benchmark trong environment riêng |
| Unified query planner | `Qwen/Qwen3-VL-2B-Instruct` | Dùng chung KIS/QA/TRAKE |
| Query–window reranker | `Qwen/Qwen3-VL-Reranker-2B` | Model đích |

## 1. Video-window embedding

Giữ `Qwen/Qwen3-VL-Embedding-2B`. Model card chính thức ghi model có 2B tham số,
nhận text, image, video và mixed-modal input, hỗ trợ hơn 30 ngôn ngữ, context 32K
và dimension tùy chỉnh 64–2048. Repository chính thức liệt kê tiếng Việt trong
33 ngôn ngữ. Đây đúng vai trò dual-tower recall và ghép trực tiếp với
`Qwen3-VL-Reranker-2B` ở tầng precision.

Profile dự án dùng 1024 chiều để giảm một nửa dung lượng so với 2048 nhưng vẫn giữ
khả năng biểu diễn đủ lớn. Dimension 512/1024/2048 cần được A/B; không chọn chỉ
dựa trên kích thước.

Không chọn bản 8B trên máy hiện tại: riêng weight FP16 đã vượt 6 GiB, chưa tính
activation và video tokens. SigLIP2 nhẹ hơn nhưng là image encoder, không thay thế
joint temporal video embedding; nó chỉ là kênh frame recall bổ sung.

Nguồn chính thức:

- https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- https://github.com/QwenLM/Qwen3-VL-Embedding

## 2. Structured window caption

Giữ `Qwen/Qwen3-VL-2B-Instruct` vì model đã có trong runtime, nhận multi-image/
video, hiểu timestamp và có khả năng text/OCR đa ngôn ngữ. Dùng cùng model cho
planner, caption, verifier và QA giảm số checkpoint, dependency và rủi ro GPU.

Caption phải là JSON schema giới hạn, không phải mô tả văn xuôi tự do. Chạy offline
trên window/shot đại diện và lưu provenance; không caption toàn bộ collection trước
khi subset chứng minh lexical recall tăng.

Các lựa chọn không thay thế chính:

- `SmolVLM2-2.2B-Instruct` hỗ trợ video và model card ghi khoảng 5,2 GB GPU RAM,
  nhưng ngôn ngữ công bố là English; biên VRAM quá sát và không phù hợp query/
  caption tiếng Việt bằng Qwen.
- `Florence-2-large` mạnh ở caption, dense region caption, detection và OCR trên
  ảnh, nhưng không phải video temporal model. Có thể benchmark như frame/region
  enrichment, không dùng cho window caption chính.
- Qwen 4B/8B có thể tăng chất lượng nhưng không phù hợp FP16 6 GiB; quantization
  chỉ xem xét khi có benchmark ổn định trên Windows và không làm latency vượt gate.

Nguồn chính thức:

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- https://huggingface.co/microsoft/Florence-2-large

## 3. OCR

EasyOCR CRAFT + `latin_g2` vẫn là baseline vì:

- đã chạy ổn định trong PyTorch environment hiện tại;
- dictionary `latin_g2` chứa các ký tự dấu tiếng Việt dựng sẵn;
- benchmark nội bộ trên keyframe dự án tốt hơn PP-OCRv6 ở lần thử trước;
- không gặp lỗi `cublasLt64_13.dll` của Paddle runtime trên máy tham chiếu.

Điều này không có nghĩa EasyOCR là model tốt nhất tuyệt đối. Challenger ưu tiên:

1. `latin_PP-OCRv5_mobile_rec`: tài liệu Paddle ghi hỗ trợ tiếng Việt và 84,7%
   trên Latin multilingual dataset; chạy cùng detector phù hợp trong venv riêng.
2. PP-OCRv6 small/medium: tài liệu công bố 1,5M–34,5M tham số, hỗ trợ 50 ngôn ngữ
   gồm tiếng Việt và cải thiện so với v5. Tuy nhiên repository từng có issue về
   dictionary thiếu các ký tự dấu tiếng Việt dựng sẵn; phải kiểm tra dictionary
   model thực tế trước benchmark.
3. PaddleOCR-VL 0.9B: phù hợp document parsing phức tạp hơn scene-text extraction,
   nên không dùng full collection mặc định. Có thể dùng second-pass cho bảng hoặc
   layout đặc biệt nếu query/dev set cần.

Không dùng Qwen VLM thay OCR chuyên dụng trên toàn collection: chậm hơn, tốn VRAM
và có nguy cơ “sửa” chữ theo language prior. Qwen chỉ làm second-pass verification
trên Top window hoặc frame OCR khó.

Nguồn chính thức:

- https://github.com/JaidedAI/EasyOCR
- https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md
- https://github.com/PaddlePaddle/PaddleOCR/issues/18254
- https://huggingface.co/PaddlePaddle/PaddleOCR-VL

## 4. Unified query planner

Dùng `Qwen/Qwen3-VL-2B-Instruct` cho mọi KIS/QA/TRAKE thay vì ba parser semantic
khác nhau. Query planning là text-only nên không tăng image token; sau planning,
cùng checkpoint được tái sử dụng cho verifier/VQA.

Không thêm text LLM riêng lúc này vì model switching và cache tăng nhưng chưa có
evidence planner tốt hơn. Nếu planner JSON/semantic không đạt, benchmark model
text-only nhỏ bằng exact schema-validity, scene coverage và latency trước khi đổi.

## 5. Điều kiện thay model

Một challenger chỉ được thay baseline khi:

1. chạy được trên đúng Windows/GPU/venv hoặc environment offline tách biệt;
2. không thiếu ký tự/modality cần thiết;
3. thắng trên dev set dự án, không chỉ benchmark nhà phát hành;
4. báo throughput, peak VRAM, disk và full-build ETA;
5. artifact có checkpoint/resume và fallback;
6. tăng Recall@k/QA answer accuracy đủ bù latency và độ phức tạp vận hành.
