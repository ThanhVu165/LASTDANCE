# Unified model-based query processing

Tài liệu này định nghĩa một luồng hiểu và thực thi query dùng chung cho KIS, QA và
TRAKE. Đích triển khai là model xử lý semantic; code deterministic chỉ validate
contract, giới hạn tài nguyên và fallback toàn-query.

## 1. Một entry point

Backend biết `task_type` từ endpoint nhưng không dùng parser riêng để hiểu nội dung:

```python
plan_query(text: str, task_type: Literal["kis", "qa", "trake"]) -> UnifiedQueryPlan
```

`Qwen/Qwen3-VL-2B-Instruct` nhận nguyên văn query và task type, sau đó sinh JSON.
Không strip câu quan trọng trước khi model nhìn thấy; luôn lưu `original_text`.

## 2. UnifiedQueryPlan

Schema mục tiêu:

```json
{
  "task_type": "kis",
  "language": "vi",
  "original_text": "...",
  "whole_query_captions": {
    "source_language": ["..."],
    "english_clip": ["..."]
  },
  "scenes": [
    {
      "scene_id": "s1",
      "summary": "...",
      "retrieval_queries": ["..."],
      "must_have": ["..."],
      "should_have": ["..."],
      "negative_constraints": ["..."],
      "evidence_modalities": ["video", "frame", "ocr"],
      "visible_text": [],
      "spoken_text": []
    }
  ],
  "temporal_edges": [
    {"before": "s1", "after": "s2", "max_gap_seconds": null}
  ],
  "answer_request": {
    "question": null,
    "answer_type": null,
    "uppercase": false,
    "no_spaces": false
  },
  "moments": [],
  "repair_queries": []
}
```

KIS dùng `scenes`; QA có `answer_request`; TRAKE có ordered `moments`. Tất cả vẫn
dùng cùng scene/evidence/temporal vocabulary.

## 3. Model-first validation flow

```text
raw query + task type
  → Qwen JSON generation
  → JSON/schema validation
  → tối đa một model repair nếu JSON sai
  → UnifiedQueryPlan
```

Validator chỉ được:

- kiểm tra JSON type và required field;
- giới hạn scene/query/criteria count và text length;
- chuẩn hóa whitespace/Unicode;
- xác minh temporal edge tham chiếu scene tồn tại;
- buộc QA có question, TRAKE có ít nhất hai moment;
- giữ original query trong mọi trường hợp.

Validator không được tự suy màu, object, scene, question hoặc moment bằng regex.
Nếu model vẫn lỗi sau bounded repair, fallback tạo một scene chứa **toàn bộ query**
và chạy baseline recall; không dùng rule semantic chi tiết.

## 4. Model-driven evidence routing

Planner gắn `evidence_modalities` cho mỗi scene:

- `video`: Qwen video-window embedding;
- `frame`: organizer CLIP và optional SigLIP2;
- `ocr`: OCR lexical/fuzzy search;
- `caption`: structured caption lexical/dense search;
- `object`: object evidence;
- `speech`: ASR index khi có.

Routing giúp ưu tiên tài nguyên, nhưng không để model tắt baseline recall hoàn toàn.
Whole-query window retrieval và whole-query CLIP fallback luôn chạy để chống planner
bỏ sót.

## 5. Query retrieval chung

Mỗi scene/whole-query caption được gửi đúng retriever:

1. Qwen text embedding → Qwen video-window FAISS.
2. Source-language multilingual CLIP → organizer frame FAISS.
3. English caption từ planner → original CLIP text tower.
4. Visible text → OCR index.
5. Structured entity/action phrase → caption/object index.
6. Spoken evidence → ASR index khi active.

Mỗi retriever trả `EvidenceHit` chung:

```json
{
  "video_id": "...",
  "local_idx": 42,
  "frame_id": 2584,
  "start_time": 81.2,
  "end_time": 96.8,
  "scene_id": "s1",
  "retriever": "qwen_video_window",
  "raw_score": 0.71,
  "rank_score": 0.94,
  "artifact_signature": "..."
}
```

Raw score chỉ so sánh trong cùng retriever. Candidate union dùng calibrated rank
fusion rồi gom thành video hypothesis theo scene coverage và temporal edges.

## 6. Shared model rerank và verification

Mọi task dùng cùng `VerifiedEvidenceBundle`:

```text
video hypothesis
  → Qwen3-VL-Reranker query–window/video score
  → Qwen instruct verifier fallback
  → missing-evidence report
  → tối đa một repair retrieval round
```

KIS xếp hạng bundle; QA đưa bundle vào answerer; TRAKE align moment bundle. Không
để QA/TRAKE gọi một semantic parser/retrieval yếu hơn riêng biệt.

## 7. Task-specific output

### KIS

Chọn frame đại diện từ verified window, cutoff-aware Top 100 và exact-frame refine
cho kết quả dẫn đầu.

### QA

Từ verified window lấy temporal context, model trả answer + evidence panel +
confidence. Answer verifier kiểm tra answer được hỗ trợ trước khi đưa lên đầu.

### TRAKE

Planner moments dùng cùng retrieval engine; K-best monotonic alignment theo
timestamp; sequence verifier nhìn toàn bộ moment rồi mới refine frame thật.

## 8. Migration khỏi code hiện tại

Hiện `query_planner.py` đã model-first cho KIS nhưng QA/TRAKE vẫn còn parser riêng
và `query_translation.py` là bước độc lập. Migration:

1. thêm `UnifiedQueryPlan` và `plan_query(task_type)`;
2. chuyển translation captions vào output planner;
3. KIS dùng plan mới, giữ regression;
4. QA bỏ `parse_qa_query`, nhận `answer_request`;
5. TRAKE bỏ `split_trake_moments`, nhận `moments`;
6. thu gọn parser cũ thành whole-query fallback;
7. xóa code cũ chỉ sau khi test KIS/QA/TRAKE pass.

## 9. Acceptance gate

- JSON schema-valid ≥ 99% trên bộ query tham khảo;
- không mất scene/must-have so với annotation người;
- QA question và TRAKE moment đúng ≥ baseline;
- planner latency nằm trong ngân sách và không download model lúc request;
- fallback nguyên-query hoạt động khi model lỗi;
- không có patch cho query mẫu;
- KIS/QA/TRAKE dùng cùng plan/evidence contract trong code và test.
