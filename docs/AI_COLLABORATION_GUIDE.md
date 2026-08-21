# Hướng dẫn cộng tác với AI trong LASTDANCE

Mục tiêu của tài liệu này là giúp thành viên dùng coding agent mà vẫn giữ kiến
trúc video-window/model-first, không tạo tập hợp patch theo query mẫu.

## 1. Context bắt buộc

Yêu cầu AI đọc:

1. `AGENTS.md`;
2. `docs/PROJECT_CONTEXT.md`;
3. `docs/SYSTEM_ARCHITECTURE.md`;
4. `docs/QUERY_PROCESSING.md` nếu task liên quan query/pipeline;
5. `docs/OFFLINE_INDEXING.md` nếu task liên quan data/index;
6. `docs/MODEL_SELECTION.md` nếu task chọn/thay model;
7. `docs/CURRENT_STATUS.md`;
8. pipeline/config/test liên quan.

Mẫu yêu cầu:

```text
Đọc AGENTS.md và các tài liệu kiến trúc chuẩn. Thay đổi <phạm vi> nhưng giữ Top
100, mapping frame_id, index completeness gate và fallback. Không patch query
mẫu. Hãy compile, chạy toàn bộ test; nếu thay retrieval/model, báo Recall@k trên
ground truth, latency, VRAM và rollback.
```

## 2. Phân loại task

### Phân tích

Chỉ đọc code/index/status, mô tả nguyên nhân và đề xuất A/B. Không tự đổi model,
build full index hoặc push.

### Thay đổi runtime

Nêu rõ KIS/QA/TRAKE, API contract và latency budget. Agent phải sửa, test và cập
nhật status/docs nếu behavior thay đổi.

### Offline indexing

Nêu dataset subset/full, model, GPU và artifact. Agent phải kiểm tra signature,
checkpoint/resume, atomic publish, metadata/vector count và state complete.

### Vận hành

Ưu tiên health check và service stability. Không build/tải model trong request
path và không chạy hai workload GPU trên máy 6 GiB.

## 3. Cách giao task model/index

Một task tốt phải trả lời:

- model xử lý evidence gì: frame, window, text, audio hay cross-modal pair;
- input/output và dimension;
- artifact, model version và license;
- VRAM/RAM/disk/ETA;
- subset benchmark và ground truth;
- metric/cutoff cần cải thiện;
- failure fallback và rollback.

Không chấp nhận câu “dùng model X sẽ tốt hơn” nếu chưa chỉ ra model nằm ở tầng
nào và thay thế/bổ sung artifact nào.

## 4. Offline indexing đúng hướng

AI phải xem video là kho nhiều evidence:

```text
manifest + frame map + shot
  + frame embeddings
  + video-window embeddings
  + structured captions
  + OCR/object
  + ASR timestamped sau này
```

Không để AI chỉ tối ưu OCR rồi coi như hiểu video. Không tạo một average vector
cho toàn video. Mọi evidence phải có `video_id`, timestamp, source frame/window và
signature.

Khi thêm builder mới, yêu cầu:

- CLI `--limit`, `--checkpoint-every` và resume;
- state `complete=false` trong lúc build;
- publish atomic;
- idempotent hoặc signature mismatch rõ ràng;
- test dừng ngang/resume;
- index partial fail closed ở runtime.

## 5. Query processing đúng hướng

Planner model sinh JSON có giới hạn:

- scenes và retrieval captions;
- must-have entities/actions/attributes/relations;
- visible text hoặc speech evidence cần thiết;
- temporal edges;
- repair queries.

Parser chỉ validate schema và fallback. Cấm thêm bảng điều kiện cho một màu,
object, câu tiếng Việt hoặc query năm trước.

KIS/QA/TRAKE phải dùng cùng `UnifiedQueryPlan`. Không tạo parser/translator semantic
riêng cho từng pipeline; endpoint chỉ cung cấp `task_type` cho model.

## 6. Retrieval và rerank

Agent phải kiểm tra recall trước rerank. Reranker không cứu được video chưa vào
candidate pool.

Luồng chuẩn:

1. retrieve frame/window từ từng embedding space;
2. hiệu chuẩn rank trong từng retriever;
3. hợp nhất candidate và gom video hypothesis;
4. tính scene coverage/temporal consistency;
5. model rerank/verification;
6. bounded repair;
7. cutoff-aware Top 100.

Không cộng raw cosine CLIP + SigLIP + Qwen. Không tuyên bố 100 row đều đúng; lưu
`model_relevance_score`, `model_verified` và provenance.

## 7. QA và TRAKE

Với QA, AI phải chứng minh retrieval đúng video/window trước khi sửa answer prompt.
OCR/object/caption là evidence, không tự sinh answer. Answer verifier phải kiểm tra
grounding và format.

Với TRAKE, ưu tiên đúng video theo coverage của mọi moment, sau đó monotonic
alignment theo timestamp và sequence verification. Không rerank từng frame độc lập
rồi ghép tùy ý.

## 8. Evaluation contract

Smoke test chỉ chứng minh hệ thống chạy. Thay retrieval/model cần:

- labeled dev set có video + frame/window ground truth;
- Recall@1/5/20/50/100 và official mean;
- đúng video trước và sau rerank;
- candidate/verified/distinct-video count;
- latency P50/P95, peak VRAM và disk/index size;
- ablation theo từng tầng;
- regression cho tiếng Việt, multi-scene, temporal, OCR và hard negative.

## 9. Review output của AI

Trước khi chấp nhận thay đổi, kiểm tra:

- Active/partial/planned có được phân biệt không?
- Code có đúng với command và model trong docs không?
- Có dùng `frame_id` thật không?
- Index dở có bị bỏ qua không?
- Có model download trong request không?
- Có rule patch query mẫu không?
- Có giữ fallback và rollback không?
- Compile và toàn bộ test có pass không?
- Tài liệu chuẩn có được cập nhật không?

## 10. Handoff

Mỗi phiên làm việc phải cập nhật `docs/CURRENT_STATUS.md` khi trạng thái artifact,
model hoặc runtime thay đổi. Ghi:

- commit/working tree liên quan;
- artifact state và signature;
- command cuối đã chạy;
- số lượng progress;
- test/metric/latency/VRAM;
- blocker và bước tiếp theo.

Không tạo thêm checkpoint dài theo ngày nếu thông tin thuộc status/roadmap chuẩn;
cập nhật file canonical để thành viên khác không phải chọn giữa nhiều kế hoạch.

## 11. Git an toàn

- Đọc `git status` trước khi sửa.
- Không reset hoặc ghi đè user changes.
- Không stage `data/`, model cache, `.venv`, query, submission hoặc secret.
- Chạy `git diff --check`, compile và unit test trước commit.
- Không push nếu user chỉ yêu cầu review/phân tích.
