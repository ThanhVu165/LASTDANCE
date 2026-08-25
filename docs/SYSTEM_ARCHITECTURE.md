# Kiến trúc hệ thống LASTDANCE

> **ARCHIVED 24/08/2026:** Kiến trúc window-first trong tài liệu này đã bị thay thế bởi
> baseline frame-level tại `BASELINE_SPEC.md`.

Tài liệu này là mô tả kỹ thuật chuẩn của kiến trúc mới. Mục tiêu chính là tìm
đúng **video window** chứa đầy đủ ngữ nghĩa của truy vấn, sau đó mới chọn frame,
rerank, trả lời QA hoặc căn chỉnh TRAKE. Trạng thái triển khai thực tế nằm trong
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## 1. Quyết định kiến trúc

Không dùng một vector keyframe hoặc một vector gộp cho toàn video để biểu diễn
mọi nội dung. Mỗi video được biểu diễn bằng một tập cửa sổ thời gian có metadata:

```text
video
  → keyframe + frame_id + pts_time
  → các window theo thời gian
  → một embedding cho từng window
  → tập vector + metadata, không average thành một vector video duy nhất
```

Một video hypothesis gồm nhiều evidence unit: window phù hợp, frame đại diện,
OCR/object evidence, mức phủ từng scene và thứ tự thời gian. Đây là đơn vị được
rerank và kiểm chứng.

## 2. Các tầng hệ thống

| Tầng | Trách nhiệm | Model/kỹ thuật chính | Module |
|---|---|---|---|
| Query planning | Một schema chung cho KIS/QA/TRAKE, tách scene/moment/answer/evidence | `Qwen/Qwen3-VL-2B-Instruct` | `services/query_planner.py` hiện tại; unified planner là target |
| Frame recall | Recall nhanh cho cảnh đơn hoặc chi tiết tĩnh | organizer CLIP ViT-B/32, multilingual text tower | `services/clip_search.py` |
| Video-window recall | Tìm chuỗi frame cùng biểu diễn hành động và nhiều ngữ nghĩa | `Qwen/Qwen3-VL-Embedding-2B`, FAISS IP | `indexing/video_window_index.py`, `services/side_search.py` |
| Optional frame side recall | Bổ sung open-vocabulary frame recall | `google/siglip2-base-patch16-256` | `indexing/siglip_index.py` |
| Structured evidence | Chữ và object quan sát được | EasyOCR CRAFT + `latin_g2`, object cache | `services/ocr_search.py`, `services/object_filter.py` |
| Candidate aggregation | Hợp nhất nhiều retriever và gom theo video/scene | rank calibration, coverage, storyboard | `pipelines/kis_pipeline.py`, `rerank/storyboard_alignment.py` |
| Cross-modal rerank | Chấm query với evidence window/video | `Qwen/Qwen3-VL-Reranker-2B` | `rerank/model_reranker.py` |
| Verification/VQA | Kiểm tra điều kiện, chọn panel, sinh answer | `Qwen/Qwen3-VL-2B-Instruct` | `rerank/model_reranker.py`, `pipelines/qa_pipeline.py` |
| Temporal alignment | Giữ đúng thứ tự nhiều moment | K-best monotonic alignment | `pipelines/trake_pipeline.py` |
| Final ranking | Tối ưu các cutoff chính thức | cutoff-aware ranking | `rerank/contest_ranking.py` |

Regex và heuristic chỉ được dùng để kiểm tra schema, mapping ID, giới hạn tài
nguyên và fallback khi model không dùng được. Không thêm quy tắc riêng cho màu,
object, tiếng Việt hoặc query mẫu.

Thiết kế toàn bộ quá trình trích xuất evidence offline nằm trong
[`OFFLINE_INDEXING.md`](OFFLINE_INDEXING.md). Video embedding là tầng chính nhưng
được bổ sung bởi shot boundary, structured caption, OCR, object và ASR timestamped.

## 3. Unified model-based query processing

Mọi endpoint gọi một entry point `plan_query(text, task_type)`. Model luôn nhìn
nguyên query và sinh `UnifiedQueryPlan` gồm whole-query caption, scenes, must-have,
evidence modality, temporal edges, QA answer request hoặc TRAKE moments.

Model đồng thời sinh caption tiếng Việt cho multilingual/video embedding và caption
tiếng Anh cho original CLIP. Vì vậy translation không còn là semantic pipeline
độc lập. JSON validator không được tự hiểu query bằng regex; nếu model lỗi sau một
bounded repair, fallback giữ toàn bộ query như một scene.

KIS/QA/TRAKE dùng cùng `EvidenceHit` và `VerifiedEvidenceBundle`. QA chỉ thêm answer
stage, TRAKE chỉ thêm moment alignment. Schema và migration cụ thể nằm trong
[`QUERY_PROCESSING.md`](QUERY_PROCESSING.md).

## 4. Video-window embedding

### 4.1. Dữ liệu đầu vào

Builder đọc `keyframe_index.json`, nhóm theo `video_id`, sắp xếp theo `local_idx`
và giữ đồng thời:

- danh sách `local_idxs` để đọc keyframe nội bộ;
- danh sách `frame_ids` để xuất kết quả;
- `pts_times` để mô hình biết khoảng cách thời gian;
- đường dẫn keyframe và frame trung tâm.

Profile hiện tại dùng 6 keyframe/window, stride 6 và tổng ngân sách 221.184 pixel.
Đây là profile khởi đầu phù hợp RTX 4050 6 GiB, không phải tham số đã tối ưu bằng
ground truth. Mỗi cấu hình có signature riêng; không được resume checkpoint từ
một signature khác.

### 4.2. Encoding và index

`Qwen/Qwen3-VL-Embedding-2B` nhận chuỗi ảnh cùng relative timestamp, sinh vector
chuẩn hóa 1024 chiều. Builder ghi tuần tự vào `video_window_features.npy`, cập
nhật `video_window_state.json`, và chỉ khi hoàn tất mới publish:

- `video_windows.json` — metadata của từng window;
- `video_windows.faiss` — `IndexFlatIP` trên vector đã chuẩn hóa;
- state có `complete=true`.

Runtime phải bỏ qua toàn bộ index nếu thiếu file hoặc `complete=false`. Một file
feature đã preallocate không phải index production.

### 4.3. Query-time retrieval

Planner sinh ba loại text:

1. nguyên query hoặc caption giữ nhận dạng toàn scene;
2. caption cho từng scene/evidence unit;
3. repair query nhắm vào điều kiện còn thiếu.

Các caption được encode trong cùng embedding space với video window. Mỗi caption
truy hồi Top N window; kết quả được gom theo video và tính:

- mức phủ scene;
- điểm tốt nhất của từng scene;
- tính nhất quán giữa các window;
- thứ tự thời gian nếu query yêu cầu;
- evidence OCR/object nếu query yêu cầu chữ hoặc object cụ thể.

Không cộng trực tiếp cosine của CLIP, SigLIP2 và Qwen. Mỗi retriever được chuyển
thành rank/min-max score trong chính không gian của nó, sau đó hợp nhất bằng
calibrated rank fusion. Trọng số chỉ được đổi sau ablation trên dev set.

### 4.4. Hướng multi-resolution

Sau khi profile 6/6 có baseline, có thể tạo index signature riêng cho:

- window ngắn: hành động cục bộ và quan hệ tức thời;
- window trung bình: scene có nhiều bước;
- event window dài hơn: diễn biến video.

Không gộp mọi scale vào một vector. Mỗi scale trả evidence độc lập và hợp nhất ở
tầng video hypothesis. Chỉ triển khai scale mới nếu Recall@k tăng đủ bù chi phí
index, VRAM và latency.

## 5. KIS end-to-end

```text
text
  → Qwen structured plan
  → CLIP frame recall + Qwen video-window recall
  → optional caption/SigLIP2/OCR/object/ASR evidence
  → calibrated candidate union
  → video grouping + scene coverage + temporal storyboard
  → Qwen3-VL-Reranker query–video/window score
  → generative verifier fallback
  → bounded repair retrieval nếu thiếu evidence
  → verified pool
  → cutoff-aware Top 100
  → exact source-frame refinement cho kết quả dẫn đầu
```

Top 100 là danh sách xếp hạng, không phải lời khẳng định 100 kết quả đều đúng.
Mỗi row cần lưu nguồn retriever, retrieval score, model relevance score và trạng
thái verified để có thể debug và đánh giá.

## 6. QA end-to-end

QA dùng cùng tầng retrieval đã kiểm chứng của KIS:

```text
query chứa mô tả + câu hỏi
  → planner tách retrieval evidence và answer request
  → retrieve/rerank đúng video window
  → lấy temporal context quanh window tốt nhất
  → Qwen VQA kiểm tra event, chọn panel và sinh answer
  → chuẩn hóa format/ngôn ngữ
  → Top 100
```

Không trả lời bằng object count hoặc OCR text đơn lẻ. OCR chỉ là evidence; answer
phải được grounded trong đúng window. Giai đoạn sau bổ sung answer verification
độc lập để phát hiện answer không được hỗ trợ bởi frame.

## 7. TRAKE end-to-end

Đích kiến trúc của TRAKE:

```text
query
  → Qwen planner sinh ordered moments
  → window recall cho từng moment
  → chọn video theo coverage của toàn bộ moment
  → K-best monotonic alignment trên timestamp
  → Qwen sequence rerank/verification
  → exact source-frame refinement cho từng moment
  → Top 100 sequence hypotheses
```

Pipeline hiện tại đã có monotonic alignment nhưng chưa dùng trọn shared
window-retrieval/verifier; đây là milestone sau khi KIS và QA ổn định.

## 8. Quản lý GPU 6 GiB

Các model 2B phải chạy theo phase, không cùng resident:

1. planner/VQA instruct;
2. giải phóng instruct;
3. query window embedding;
4. giải phóng embedding model;
5. dedicated reranker;
6. giải phóng reranker;
7. nạp lại instruct chỉ khi cần exact-frame hoặc QA.

Không chạy backend, OCR builder, SigLIP builder hoặc video-window builder đồng
thời. Runtime không tự tải model lớn; model phải được tải trước và dùng
`local_files_only` khi phù hợp.

## 9. Failure contract

- Window/SigLIP index thiếu hoặc chưa complete: bỏ qua, giữ CLIP recall.
- Dedicated reranker thiếu checkpoint: dùng generative verifier.
- Cả hai verifier lỗi: dùng tournament fallback đã kiểm thử.
- Planner lỗi JSON/model: bounded model repair, sau đó dùng nguyên query làm một
  scene baseline; không semantic-parse bằng regex.
- Không có đủ verified row: giữ candidate pool nhưng đánh dấu chưa verified.
- Không bao giờ đổi `local_idx` thành `frame_id` bằng suy đoán; luôn đọc map CSV.

Mọi fallback phải nhìn thấy được trong `/health`, log hoặc field kết quả; không
được âm thầm làm thay đổi semantics.

## 10. Evaluation

Đánh giá trên dev set có ground truth tại `R@1`, `R@5`, `R@20`, `R@50`, `R@100`.
Mỗi thử nghiệm cần báo:

- Recall tại từng cutoff và mean official score;
- tỷ lệ query tìm đúng video trước rerank;
- tỷ lệ Top 100 có model score/verified;
- số video phân biệt trong Top 100;
- latency P50/P95 và peak VRAM;
- ablation theo từng tầng: CLIP → planner → window → reranker → repair.

Không dùng vài smoke query để kết luận model hoặc kiến trúc chính xác hơn.
