# Plan: Sửa các lỗi vận hành Online (Query Planner + QA/VQA)

**Ngày tạo:** 2026-09-03 | **Trạng thái:** Chờ xác minh env setup trước khi bắt đầu code

## Bối cảnh & phạm vi

Toàn bộ vấn đề nằm trong `online/` (Nhánh 2). Theo `AGENTS.md`, Nhánh 2 do người khác phụ
trách và Codex không tự ý sửa trừ khi được yêu cầu rõ — người dùng đã yêu cầu rõ ràng trong
phiên này ("kiểm tra, cụ thể từng vấn đề rồi lên plan để giải quyết"), nên phạm vi sửa
`online/planners.py`, `online/vqa.py`, `configs/online_baseline.json`, `online/config.py` là
hợp lệ cho task này. Không đụng tới `offline/` hay schema chung.

## Chẩn đoán từng vấn đề (đã xác minh bằng đọc code)

1. **"Rule planner cannot translate visual queries to English"** — đây là **triệu chứng**,
   không phải lỗi độc lập: nó chỉ xuất hiện khi cả Gemini và Qwen đều fail và rơi xuống
   `RuleBasedQueryPlanner` (planner cuối cùng, không dịch được tiếng Việt → tiếng Anh).

2. **`GeminiQueryPlanner: TimeoutError`** — `online/planners.py` dòng 765:
   `GeminiQueryPlanner.__init__(..., timeout: float = 12.0)` bị **hardcode 12 giây**, không có
   biến môi trường override, trong khi `GeminiJsonClient` (dùng cho VQA/verification) mặc định
   60 giây. 12s quá ngắn cho một request structured-JSON-schema (planner prompt dài + schema
   phức tạp) qua mạng thật → timeout thường xuyên, đẩy planner rơi xuống Qwen rồi Rule.

3. **`WorkerQwenQueryPlanner: RuntimeError: requires CUDA or AIC_ALLOW_QWEN_CPU=1`** —
   `online/qwen_runtime.py` dòng 44: đây là **hành vi cố ý** (bảo vệ VRAM 6GB), không phải bug.
   Nguyên nhân thực tế: `torch.cuda.is_available()` trả `False` trong venv Online hiện tại —
   nghĩa là torch cài trong `.venv-online` là bản CPU-only, dù máy có RTX 4050. Đây là vấn đề
   **môi trường**, không phải code.

4. **OCR "unavailable" dù đã có snapshot** — `online/artifacts.py::_inspect_ocr` chỉ đọc
   `AIC_OCR_SNAPSHOT_DIR` tại thời điểm `ArtifactRegistry.load()`, và `online/streamlit_app.py`
   cache kết quả này bằng `@st.cache_resource`. Hai nguyên nhân khả dĩ, cả hai đều là vận hành
   chứ không phải bug:
   - biến `$env:AIC_OCR_SNAPSHOT_DIR` được set ở terminal khác với terminal chạy
     `streamlit run`, nên process Streamlit không thấy biến này → fallback về
     `index/ocr.sqlite` (thường không tồn tại) → `UNAVAILABLE`;
   - đổi snapshot nhưng **không restart Streamlit** → `st.cache_resource` giữ `ArtifactRegistry`
     cũ đã đánh giá OCR trước khi biến môi trường được set.
   Cần xác minh trực tiếp với bạn (mục Việc cần làm #0) trước khi kết luận đây thuần là thao
   tác vận hành; nếu đúng, sẽ cải thiện thêm message chẩn đoán để lần sau tự phát hiện nhanh.

5. **"Gemini VQA prompts disagreed" → QA answer unavailable → 0/100** — nguyên nhân **kép**:
   - (a) Do OCR đang `UNAVAILABLE` (mục 4), `engine.py::_qa_answerer` bỏ qua
     `FtsVideoAnswerer(ocr)` hoàn toàn khi `answer_target.source in {"ocr","mixed"}`, nên toàn
     bộ câu hỏi phải dựa vào VLM đoán từ 6 ảnh nhỏ (224×224) không có text-evidence → tỷ lệ
     "không xác định được" tăng mạnh — đúng như bạn nghi ngờ.
   - (b) `online/vqa.py::QwenVQAAnswerer.answer` / `GeminiVQAAnswerer.answer` so khớp hai lần
     trả lời độc lập bằng **so khớp chuỗi tuyệt đối** (`_normalize_answer(first) !=
     _normalize_answer(second)`). Khi model trả lời "không có thông tin" bằng hai cách diễn đạt
     khác nhau (điều rất thường gặp với câu trả lời phủ định tự do), hệ thống coi là mâu thuẫn
     dù cả hai lần đều đồng ý là "không xác định được" — làm mất luôn candidate dù đúng ra nó
     nên được coi là "Uncertain" sạch sẽ (không phải lỗi) chứ không tạo cảm giác pipeline hỏng.

6. **"Gemini per-search call budget exhausted"** — `configs/online_baseline.json`:
   `gemini_max_calls_per_search = 8`. Một lượt search tối đa cần:
   planner (≤1) + verification (`vlm_video_top_k=4`) + QA VQA (`qa_answer_video_top_k=3` × 2
   lần hỏi/video = 6) = **tối đa 11 call**, vượt ngân sách 8 → video xếp hạng thấp hơn (ví dụ
   video thứ 3 trong Top 3 QA) hết ngân sách giữa chừng. `AIC_GEMINI_SAFE_RPM=14`/phút và
   `gemini_search_timeout_seconds=300` (5 phút) đủ dư để nâng ngân sách lên an toàn.

7. **Ngôn ngữ câu hỏi truyền cho OCR/VLM answerer** — đã xác nhận với bạn: tách riêng.
   - `FtsVideoAnswerer.answer()` (OCR/ASR) **không hề dùng** tham số `question` để tìm kiếm —
     nó chỉ dùng `known_text_literals`/`visual_text_attributes` (luôn là chuỗi tiếng Việt
     nguyên văn từ raw query, theo đúng system prompt planner). Do đó đổi ngôn ngữ câu hỏi
     **không ảnh hưởng** tới OCR/ASR matching.
   - Vấn đề nằm ở `online/planners.py` dòng 705: dù Gemini/Qwen đã sinh JSON, code
     **luôn tự tính lại** `answer_target.question` bằng `_question_text(raw_query)` (tiếng
     Việt), bỏ qua hoàn toàn field `question` mà model có thể đã dịch. Đây là bug khiến câu hỏi
     gửi cho VQA luôn là tiếng Việt bất kể planner nào chạy.

## Hướng sửa cụ thể

### Việc cần làm #0 — Xác minh môi trường (làm trước, không cần sửa code)
- Xác nhận: (a) `$env:AIC_OCR_SNAPSHOT_DIR` được set **đúng trong terminal chạy
  `streamlit run online\streamlit_app.py`**, không phải terminal khác; (b) sau khi set/đổi biến
  đã **restart Streamlit** hoàn toàn (không chỉ refresh trang); (c) `python -m online preflight`
  chạy trong cùng venv/terminal báo `ocr: READY`.
- Xác nhận venv Online (`.venv-online`) có torch build CUDA:
  `python -c "import torch; print(torch.cuda.is_available())"` — nếu `False` trên máy có RTX
  4050, cần cài lại đúng wheel `torch==2.6.0+cu124` theo `docs/ONLINE_RUNBOOK.md` §3.

### 1. Sửa timeout Gemini planner (`online/planners.py`)
- Thêm biến môi trường `AIC_GEMINI_PLANNER_TIMEOUT_SECONDS` (mặc định 45s thay vì 12s hardcode),
  đọc trong `get_query_planner()` và truyền vào `GeminiQueryPlanner(..., timeout=...)`.

### 2. Sửa câu hỏi tiếng Anh cho VQA, giữ tiếng Việt cho OCR/ASR (`online/planners.py`)
- Sửa `_PLANNER_SYSTEM`: yêu cầu rõ `answer_target.question` phải là bản dịch tiếng Anh trung
  thực (giống `retrieval_query_en`), không phải copy nguyên văn.
- Sửa `_validate_provider_plan` (dòng ~705): dùng
  `str(target_payload.get("question") or "").strip() or _question_text(raw_query)` thay vì
  luôn gọi lại `_question_text(raw_query)`; chỉ fallback tiếng Việt khi model không trả field
  này.
- `RuleBasedQueryPlanner` giữ nguyên tiếng Việt (đã có cảnh báo không dịch được sẵn).
- Không đổi cách OCR/ASR literal hoạt động (đã xác nhận không phụ thuộc `question`).

### 3. Sửa so khớp "disagree" của VQA thành fuzzy/semantic thay vì exact-match (`online/vqa.py`)
- Thêm hàm `_answers_agree(first, second, *, threshold)`:
  - Khớp tuyệt đối sau chuẩn hoá → đồng ý ngay (fast path, giữ hành vi cũ khi hoàn toàn khớp).
  - Nếu một trong hai câu trả lời chứa số (`\d+`), **bắt buộc khớp số chính xác** (không fuzzy
    số — đã kiểm chứng thực nghiệm: "3 bánh" vs "5 bánh" có fuzzy ratio 0.83, fuzzy thuần sẽ
    coi là đồng ý sai, rất nguy hiểm cho QA dạng đếm số).
  - Còn lại (free_text/color/person/place, đặc biệt các câu phủ định "không xác định được" nói
    theo nhiều cách) dùng `difflib.SequenceMatcher(...).ratio() >= threshold` (mặc định
    **0.6**, đã đo thực nghiệm trên đúng 2 cặp câu bạn gặp: ratio 0.72 và 0.60 — cả hai đều qua
    ngưỡng; cặp màu khác nhau "đỏ" vs "xanh" ratio 0.57 — vẫn bị từ chối đúng).
  - Khi đồng ý theo fuzzy, trả về câu trả lời có confidence cao hơn (thay vì luôn lấy `first`).
- Áp dụng cho cả `QwenVQAAnswerer.answer` và `GeminiVQAAnswerer.answer`.
- Thêm `qa_vqa_agreement_similarity: float = 0.6` vào `OnlineConfig` (validate 0..1) và
  `configs/online_baseline.json`; truyền xuống `get_video_answerer(registry, config)` (thêm
  tham số `config` với default để không phá test hiện có gọi `QwenVQAAnswerer(registry)` trực
  tiếp).

### 4. Nâng ngân sách gọi Gemini mỗi lượt search (`configs/online_baseline.json`)
- Tăng `gemini_max_calls_per_search` từ 8 lên **14** — đủ cho planner(1) + verification(4) +
  QA(6) = 11 cộng biên an toàn, vẫn nằm trong `AIC_GEMINI_SAFE_RPM=14`/phút ×
  `gemini_search_timeout_seconds=300s` (5 phút) nên không vi phạm quota free-tier đã cấu hình.
- Giữ nguyên cơ chế circuit-breaker/backoff hiện có, không đổi logic quota window.

### 5. Cập nhật tài liệu vận hành
- Ghi chú vào `docs/ONLINE_RUNBOOK.md` §6 (Fallback/degraded mode): nhắc rõ (a) phải set
  `AIC_OCR_SNAPSHOT_DIR` **và restart Streamlit** trong cùng session mỗi khi đổi snapshot;
  (b) kiểm `torch.cuda.is_available()` trước khi kỳ vọng Qwen planner/VQA hoạt động;
  (c) ý nghĩa threshold fuzzy-agreement mới cho VQA.

### 6. Cập nhật tests
- `tests/test_online_vqa.py`: test `_answers_agree` cho case đồng nghĩa phủ định (agree), số
  khác nhau (reject), màu khác nhau (reject).
- `tests/test_online_planner.py`: test provider-supplied English question được dùng thay vì
  recompute Vietnamese.
- Chạy `python -m pytest tests/test_online_vqa.py tests/test_online_planner.py
  tests/test_online_config.py -q` để đảm bảo 55/55 Online PASS đã không bị phá.

## Việc KHÔNG làm trong phạm vi này
- Không sửa `offline/`, không đổi schema `OcrResult`/`UnifiedQueryPlan` cột SQL.
- Không tự ý bump `AIC_GEMINI_SAFE_RPM/TPM/RPD` (đã đủ dư theo tính toán ở trên).
- Không cài thêm thư viện fuzzy-matching ngoài (dùng `difflib` sẵn có trong Python stdlib).
- Không tự động bật `AIC_ALLOW_QWEN_CPU=1` mặc định — đây là quyết định vận hành, chỉ ghi vào
  runbook để người dùng tự bật khi hiểu độ chậm.

## Checklist thực thi

- [ ] **#0. Xác minh env: OCR snapshot + CUDA torch**
- [ ] **#1. Fix Gemini planner timeout** (`online/planners.py` + `online/config.py`)
- [ ] **#2. Fix question language cho VQA** (`online/planners.py`)
- [ ] **#3. Fix VQA fuzzy agreement** (`online/vqa.py`, `online/config.py`)
- [ ] **#4. Nâng Gemini call budget** (`configs/online_baseline.json`)
- [ ] **#5. Update ONLINE_RUNBOOK** (`docs/ONLINE_RUNBOOK.md`)
- [ ] **#6. Add/update tests** (`tests/test_online_*.py`)
- [ ] **#7. Run full regression** (`pytest` Online suite)
- [ ] **#8. Commit + Document (nếu được yêu cầu)**

## Lưu ý
- File plan này có thể xóa sau khi hoàn thành toàn bộ checklist.
- Không commit plan này vào repo (chỉ là tài liệu vận hành nội bộ).
