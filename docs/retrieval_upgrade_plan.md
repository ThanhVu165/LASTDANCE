# Kế hoạch nâng cấp retrieval từ bộ truy vấn chung kết tham khảo

## 1. Những gì bộ truy vấn cho thấy

Workbook `DanhSachTruyVanAIC_Chungket.xlsx` có 29 truy vấn tham khảo: 14 TKIS,
6 Q&A, 4 TRAKE và 5 VKIS.

- TKIS không phải các caption ngắn. Phần lớn là mô tả nhiều cảnh theo trình tự,
  thường trộn thuộc tính nhìn thấy, chữ trên màn hình, địa danh và manh mối kiến
  thức. Một frame không thể chứa toàn bộ bằng chứng.
- Q&A gồm hai nhóm rõ rệt: quan sát theo thời gian/đếm tại thời điểm chính xác và
  suy luận thực thể (thành phố, danh họa, thương hiệu, logo). Nhiều truy vấn còn
  yêu cầu định dạng answer như viết hoa và bỏ khoảng trắng.
- TRAKE dùng trực tiếp `E1: ... E4:`, có khái niệm lần đầu/lần thứ hai và yêu cầu
  chọn đúng frame chạm/vượt/vào lưới, không chỉ đúng video.
- VKIS là truy vấn chủ đề rất ngắn. Đây là bài toán khác với TKIS dài và nên giữ
  đường retrieval semantic trực tiếp, không ép qua planner phức tạp.

Bộ truy vấn này dùng để kiểm tra độ phủ năng lực và parser. Vì video hiện tại không
phải bộ chung kết năm trước, không được dùng thứ hạng retrieval trên dataset hiện
tại như một phép đo accuracy có ground truth.

## 2. P0 — đã triển khai

1. Runtime preflight cho Q&A: `/health` công bố Python executable, Torch/CUDA và
   `vqa_ready`; UI chặn sớm khi backend chạy nhầm Python global.
2. Long-query decomposition: TKIS dài được tách thành tối đa tám mô tả cảnh vừa
   với text encoder, nhưng vẫn giữ truy vấn gốc.
3. Video-level consensus: bằng chứng của nhiều clause được tổng hợp trên toàn
   video trước khi chọn frame, thay vì để một frame khớp một chi tiết chung thắng.
4. OCR intent tốt hơn: nhận cả chữ không đặt trong ngoặc kép như `dòng chữ coles`,
   đồng thời tránh coi mọi câu trích dẫn ẩn dụ là chữ xuất hiện trên màn hình.
5. QA parsing/formatting: loại chỉ dẫn câu hỏi lặp khỏi retrieval; giữ nguyên toàn
   bộ query cho VLM; thực thi `UPPERCASE` và `NO SPACES` ở bước chuẩn hóa answer.
6. QA multi-frame: top 20 dùng cửa sổ ba keyframe theo thời gian; khung giữa là
   ứng viên chính. 80 kết quả sau dùng một frame để cân bằng latency.
7. TRAKE parser nhận cả `(1)`, `1.`, `Moment 1` và `E1:`; K-best alignment vẫn
   cưỡng chế frame tăng nghiêm ngặt theo thời gian.
8. Query tiếng Việt được Qwen3-VL chuyển thành caption tiếng Anh có kiểm soát theo
   từng evidence unit, chạy song song với multilingual CLIP. Hai bản của cùng unit
   được max-pool trước khi tính coverage; điểm được hiệu chuẩn bằng min-max + RRF.
9. Q&A tách hẳn câu hỏi khỏi event retrieval; toàn query chỉ dùng ở bước sinh answer.
10. Video-first Qwen rerank tối đa 30 video, 4 frame/video, bằng tournament tương
    đối nhóm 3. Query ngắn dùng cửa sổ keyframe trước/giữa/sau; TRAKE rerank 10
    sequence.
11. Xếp hạng theo portfolio tại đúng mốc R@1/5/20/50/100 thay cho một quota MMR
    cố định trên toàn danh sách.
12. TRAKE exact-frame coarse-to-fine trên video gốc cho moment có điều kiện biên;
    contact sheet giảm image token và lỗi/OOM.

## 3. Hàm mục tiêu chính thức

Theo tài liệu vòng sơ tuyển, `R@k` là R-Score lớn nhất trong k kết quả đầu với
`k = 1, 5, 20, 50, 100`; Final Score là trung bình năm mốc. KIS/Q&A cần đúng đồng
thời video và khoảng frame (Q&A thêm answer đúng ngữ nghĩa). TRAKE sai video là 0,
còn đúng video nhận điểm từng phần theo tỉ lệ moment khớp; khoảng moment thường
dưới 10 frame.

Vì vậy tầng rank hiện dùng hai mức:

- Qwen rerank tăng xác suất đúng hoàn toàn ở hạng 1.
- Cutoff portfolio giữ tối đa 2 kết quả/video trong Top 5, rồi tăng quota ở Top
  20/50/100 để phủ video/segment khác. TRAKE giữ nhiều alignment trong video mạnh
  hơn KIS vì một alignment khác có thể tăng R-Score từng phần.

`backend/app/evaluation/official_metric.py` là implementation duy nhất của công
thức chấm offline; test tái hiện đúng ví dụ Final Score 0,74 và TRAKE 0,75 trong PDF.

## 4. P1 — còn lại

### 4.1. ASR index có timestamp — tạm hoãn

Người dùng đã chủ động tạm bỏ qua ASR. Nhiều query TKIS/Q&A chứa thông tin chỉ có trong lời dẫn
bản tin: tên địa danh, sự kiện, con số, nguyên liệu hoặc lời giải thích khoa học.
CLIP, object detection và OCR không thể truy xuất phần này.

Đề xuất:

- Chạy offline `faster-whisper` với `large-v3-turbo`, `int8_float16`, VAD bật.
- Lưu từng segment `{video_id, start_sec, end_sec, text, language}`.
- Lập hai index: lexical/BM25 cho tên riêng-con số và multilingual dense embedding
  cho diễn đạt tương đương.
- Khi search, hợp nhất ASR với CLIP/OCR bằng reciprocal-rank fusion; ánh xạ timestamp
  ASR về keyframe gần nhất.
- Chạy ASR như một job riêng, không đồng thời với OCR hoặc Qwen3-VL.

`faster-whisper` công bố bản cài đặt nhanh hơn OpenAI Whisper và hỗ trợ INT8 để
giảm VRAM. Tài liệu chính thức:
https://github.com/SYSTRAN/faster-whisper

### 4.2. Query planner có kiểm soát — đã có bản visual translation

Dùng Qwen3-VL-2B ở chế độ text-only cho các query dài/gián tiếp để sinh JSON:

```json
{
  "visual_clauses": [],
  "named_entities": [],
  "ocr_phrases": [],
  "asr_terms": [],
  "temporal_order": [],
  "answer_format": {}
}
```

Hiện tại planner text-only đã sinh các caption visual tiếng Anh canonical và cache
theo query. Bản JSON đầy đủ ở trên vẫn là backlog khi ASR quay lại. Planner chỉ bổ sung tín hiệu; luôn giữ query gốc và không được dùng suy luận của
model như ground truth. Cache kế hoạch theo hash query để không trả thêm latency
khi người dùng xem lại kết quả.

### 4.3. Video-first rerank bằng multi-frame VLM — đã triển khai

- Retrieval rẻ chọn 5 video tốt nhất theo cấu hình mặc định.
- Lấy tối đa 9 frame đại diện cho từng video, ghép contact sheet có số để Qwen vừa
  chấm video vừa chọn đúng frame trong segment.
- Qwen3-VL đánh giá mức bao phủ toàn query và rerank video/segment.
- Chỉ áp dụng cho top video, không chạy VLM trên toàn bộ 177 nghìn keyframe.

Cách này phù hợp với TKIS nhiều cảnh và tận dụng Qwen3-VL đã có mà không làm latency
tăng theo toàn collection.

## 5. P2 — thử nghiệm trước khi đổi index production

### 5.1. Side index SigLIP2

Thử A/B `google/siglip2-so400m-patch14-384` trên một tập keyframe nhỏ đã gán nhãn.
SigLIP2 là encoder ảnh-văn bản đa ngôn ngữ và có API retrieval chính thức, nhưng
không tương thích với feature CLIP 512 chiều hiện tại; muốn dùng phải encode lại
toàn bộ ảnh và dựng FAISS side index.

Tài liệu chính thức:
https://huggingface.co/docs/transformers/model_doc/siglip2

Không thay index production cho tới khi Recall@1/5/20/50/100 trên tập dev tốt hơn
ensemble hiện tại đủ rõ để bù chi phí build và runtime.

`jinaai/jina-clip-v2` cũng hỗ trợ long-context và nhiều ngôn ngữ, nhưng checkpoint
công khai dùng giấy phép CC-BY-NC-4.0; cần kiểm tra điều kiện cuộc thi trước khi
đưa vào hệ thống chính.

### 5.2. TRAKE exact-frame refinement — đã triển khai cho Top 1

Sau alignment, hệ thống decode cửa sổ giới hạn bởi hai keyframe lân cận, lấy mẫu
coarse rồi fine, ghép contact sheet có số và để Qwen chọn frame. Chỉ moment có từ
chỉ biên/trạng thái chính xác mới refine; lỗi decode/model giữ nguyên keyframe.

## 6. Đánh giá bắt buộc trước mỗi thay đổi model

Tạo tập dev có ground truth trên chính video hiện tại và báo cáo riêng:

- Recall@1, @5, @20, @50, @100 và MRR cho KIS/Q&A retrieval.
- Exact/normalized answer accuracy cho Q&A.
- Video accuracy, moment recall và sai số frame theo giây cho TRAKE.
- P50/P95 latency, peak RAM/VRAM, kích thước index và thời gian build.
- Ablation: CLIP; CLIP+OCR; +long-query consensus; +ASR; +VLM rerank; +side index.

Không dùng cảm giác nhìn vài top result để quyết định thay model production.

## 7. Quyết định khóa cho đợt 1

Không đổi model/index lõi trong 10 giờ trước đợt thi. Lý do: re-index khoảng 90.000
cửa sổ video bằng model 2B không còn đủ thời gian để vừa chạy vừa đo Recall@100,
Top-1, latency và regression QA/TRAKE.

Thay đổi cầu nối đã kiểm tra trên index hiện tại:

- Truy vấn nhiều cảnh lấy 800 candidate cho mỗi evidence unit; truy vấn ngắn 400.
- Sinh bản tiếng Anh cho từng scene trong một lần gọi Qwen; tên riêng có dấu như
  `Chánh Thiên`, `Bà Rịa` không còn bị bộ dò ngôn ngữ loại bỏ.
- Storyboard giữ nhiều vector/keyframe trên mỗi video, không coi một frame là toàn
  bộ video.
- Qwen3-VL-2B không dùng pointwise `SCORE`: đo thực tế model trả 100 cho cả video
  đúng và sai. Production chỉ dùng so sánh tương đối theo bracket nhóm 3, output
  tối giản `BESTVIDEO/BESTPANEL`, lỗi format tự rơi về retrieval.
- Smoke TKIS thiện nguyện trả đủ 100 kết quả; video kiểm chứng thủ công `L22_V004`
  tăng từ khoảng hạng 35 ở baseline cũ lên hạng 3 sau scene recall và Top 1 sau
  comparative rerank. Không coi đây là accuracy toàn tập vì workbook không có
  ground truth.

## 8. P2 — kiến trúc VCMR ngay sau đợt 1

Ứng viên chính thay cho side index SigLIP2 là cặp chính thức:

- `Qwen/Qwen3-VL-Embedding-2B`: text/image/video, 33 ngôn ngữ có tiếng Việt,
  embedding 64–2048 chiều.
- `Qwen/Qwen3-VL-Reranker-2B`: cross-encoder cho cặp query–video.

Thiết kế không tạo một vector cho cả video:

1. Cắt video thành cửa sổ chồng lấn 6–10 giây và event window 20–40 giây.
2. Mỗi cửa sổ có vector, `video_id`, `start_sec`, `end_sec`; FAISS truy hồi đoạn.
3. Gom nhiều đoạn theo video, giữ thứ tự, rồi rerank nguyên query với ordered
   evidence pack.
4. OCR/ASR là index evidence riêng hợp nhất sau retrieval, không nhét vào ontology.
5. A/B trên một tập gán nhãn nhỏ trước; chỉ full re-index khi Recall@100 và Top-1
   vượt cấu hình đợt 1 đủ rõ.

Môi trường hiện tại đã xác minh RTX 4050 6 GB, PyTorch 2.12.1+cu130 và còn khoảng
52,9 GB đĩa. Bản 2B phải được thử trong environment/side index riêng; embedding và
reranker không giữ đồng thời trên GPU 6 GB.

Nguồn chính thức:

- https://github.com/QwenLM/Qwen3-VL-Embedding
- https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B
- https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B
