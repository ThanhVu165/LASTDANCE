# AI collaboration guide cho LASTDANCE

Tài liệu này giúp thành viên dùng Codex, Copilot hoặc agent khác mà không biến
pipeline thành tập hợp các bản vá theo query mẫu. `AGENTS.md` là contract ngắn bắt
buộc; file này giải thích quy trình cộng tác chi tiết hơn.

## 1. Context cần cung cấp cho AI

Khi bắt đầu một task mới, yêu cầu AI đọc:

1. `AGENTS.md`;
2. `docs/PROJECT_CONTEXT.md`;
3. file pipeline/config liên quan;
4. test hiện hữu;
5. runtime report gần nhất.

Mẫu yêu cầu:

```text
Đọc AGENTS.md và docs/PROJECT_CONTEXT.md trước. Hãy thay đổi <phạm vi> nhưng giữ
contract Top 100, mapping frame_id và fallback hiện tại. Chạy compile + unittest,
báo latency/VRAM nếu đụng tới model, không sửa theo query mẫu.
```

Không cần đưa dataset hoặc model cache vào prompt. Cho AI đường dẫn local và cho
phép nó kiểm tra read-only khi cần.

## 2. Phân loại task

### Chỉ phân tích

Yêu cầu AI đọc code, mô tả pipeline, tìm nguyên nhân hoặc đề xuất A/B. Không cho
phép sửa file/push nếu chưa muốn triển khai.

### Thay đổi code

Nêu rõ phạm vi KIS/QA/TRAKE/index/UI, giới hạn thời gian và phần cứng. AI phải sửa,
test và ghi lại trạng thái; không dừng ở proposal.

### Index/model offline

Nêu rõ model, dataset subset/full, GPU được phép dùng và có thể dừng backend hay
không. Builder phải checkpoint/resume và chỉ publish index khi complete.

### Vận hành cuộc thi

Ưu tiên không thay code. Chỉ health check, start/restart đúng service, xác nhận
GPU, chạy một smoke bounded và giữ fallback ổn định.

## 3. Quy tắc chọn model

Đánh giá model theo vai trò, không theo độ nổi tiếng:

| Vai trò | Cần tối ưu |
|---|---|
| Query planner | JSON validity, multilingual semantics, latency |
| Bi-encoder retrieval | Recall@100, tốc độ encode/index, dung lượng |
| Cross-encoder reranker | Top 1/5, khả năng loại partial match, VRAM |
| VQA answerer | evidence grounding, format/language, hallucination |
| OCR/ASR | độ phủ text/speech, timestamp, throughput offline |

Trước khi thay model, AI phải trả lời:

- model có đúng modality/language không;
- checkpoint/license có dùng được cho cuộc thi không;
- có vừa VRAM/RAM/disk không;
- cần re-index bao nhiêu dữ liệu;
- online latency và offline build time;
- fallback nếu download/load/inference lỗi;
- cách đo cải thiện trên dev set.

Model-first không có nghĩa dùng LLM cho mọi thao tác. Mapping frame, validation,
checkpoint và budget vẫn nên deterministic.

## 4. Cách xử lý query đúng hướng

Một query dài phải được biểu diễn thành structured evidence:

- scene/moment;
- entity và action;
- attribute/count/relation;
- visible text;
- temporal order;
- global conditions;
- alternative retrieval captions;
- missing-evidence repair captions.

Model planner sinh cấu trúc tổng quát. Không thêm danh sách `if query contains red
car`, tên địa danh hoặc lỗi OCR riêng của một sample. Nếu planner sai, sửa schema,
prompt, validation hoặc training/evaluation data.

## 5. Retrieval trước, rerank sau

Khi kết quả chỉ khớp một phần, AI phải tách hai khả năng:

1. **Recall failure**: video đúng không có trong candidate pool. Cần planner,
   retriever, OCR/ASR/side index hoặc candidate budget tốt hơn.
2. **Ranking failure**: video đúng có trong pool nhưng đứng thấp. Cần evidence
   aggregation, cross-encoder, calibration hoặc cutoff ranking.

Không thể sửa recall failure chỉ bằng rerank. Mọi thử nghiệm nên log video đúng có
ở Top 100/400/800 trước rerank hay không.

## 6. Evaluation contract

Không dùng cảm giác nhìn vài ảnh để kết luận model tốt hơn. Dev set cần có:

- query VI/EN ngắn và dài;
- single-scene/multi-scene;
- actions, colors, counts, relations, OCR;
- hard negatives chỉ khớp object chung;
- QA temporal/count/entity;
- TRAKE ordered boundaries.

Báo cáo tối thiểu:

- Recall@1/5/20/50/100 và MRR;
- candidate recall trước rerank;
- QA normalized answer accuracy;
- TRAKE video accuracy, moment recall và frame error;
- P50/P95 latency, peak VRAM/RAM;
- ablation từng retriever/verifier.

Nếu chưa có ground truth, chỉ được báo “smoke/contract passed”, không báo accuracy.

## 7. Quy trình sửa code cùng AI

1. Yêu cầu AI kiểm tra `git status`; working tree bẩn không được reset.
2. Giới hạn file/phạm vi rõ ràng.
3. Đề nghị AI nêu assumption có thể ảnh hưởng kiến trúc.
4. Dùng feature flag hoặc complete-state guard cho model/index mới.
5. Chạy compile và toàn bộ unittest.
6. Chạy smoke nhỏ trước full GPU job.
7. Với builder, benchmark throughput rồi mới ước lượng full run.
8. Cập nhật tài liệu và runtime report.
9. Con người xem `git diff` trước commit/push.

## 8. Checklist review output của AI

- Có phân biệt việc đã làm, chưa làm và chưa kiểm chứng không?
- Có nói đúng model backend thực tế hay chỉ đọc tên trong config?
- Có vô tình dùng partial index không?
- Có giữ đủ Top 100 và đúng `frame_id` không?
- Có tạo network download giữa request không?
- Có chạy hai model 2B/OCR cùng GPU không?
- Có hard-code sample query không?
- Có số liệu latency/VRAM và test result không?
- Có cập nhật docs không?

## 9. Handoff giữa các phiên AI

Cuối một task dài, yêu cầu tạo hoặc cập nhật runtime report gồm:

- objective và quyết định kiến trúc;
- file đã đổi;
- dependency/model đã cài;
- test/smoke và số đo;
- process đang chạy;
- artifact/checkpoint complete hay partial;
- blocker và lệnh tiếp tục chính xác;
- `git status`, chưa commit hay đã commit/push.

Không dựa vào lịch sử chat như nguồn duy nhất; thành viên khác phải tiếp tục được
chỉ từ repository và artifact manifest.

## 10. Các thao tác Git an toàn

AI có thể sửa/test local khi được yêu cầu. Commit hoặc push cần nằm trong yêu cầu
rõ ràng của thành viên vì repository có thể chứa nhiều thay đổi chưa review.

Trước push:

```powershell
git status
git diff --check
git diff --stat
git diff --cached
```

Không commit data/model/token. Không dùng `git reset --hard` để “dọn” working tree.

