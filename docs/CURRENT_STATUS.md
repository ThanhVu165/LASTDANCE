# Trạng thái hiện tại của LASTDANCE

Cập nhật: 04/09/2026. File này là **snapshot hiện hành**, không phải append-only session log.
Contract chuẩn nằm tại `docs/BASELINE_SPEC.md`; lệnh vận hành nằm trong các runbook.

Chi tiết đợt sửa và phần chưa xác minh: [QUALIFIER_ACCEPTANCE_RUNBOOK.md](QUALIFIER_ACCEPTANCE_RUNBOOK.md).

## Kết luận mức hệ thống

**Mức hiện tại: integration-ready cho KIS/QA/TRAKE và submission vòng sơ tuyển; chưa được
phép gọi là accuracy-complete.** Offline visual đã đóng/PASS. Online Accuracy-Max và official
export đã implement và bổ sung VerifiedFrameRef cho frame gốc ngoài catalog. QA gắn evidence,
TRAKE giữ frame cùng shot. OCR giữ nguyên, tạm bỏ qua vì sắp thay artifact theo người dùng.
ASR đang chạy Kaggle; chưa tải artifact và chưa tích hợp thật. Bộ
kiểm tra thủ công/ground-truth cần chạy lại để chốt Recall/accuracy sau thay đổi Online/OCR.

| Thành phần | Trạng thái | Evidence hiện hành |
|---|---|---|
| Inventory/video/shot | `READY` | 873 MP4, 873 shot manifest, inventory full |
| Keyframe catalog | `READY` | 293.336 JPEG/UID, 873 video |
| CLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 512, 293.336 UID |
| SigLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 768, 293.336 UID |
| EVA-CLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 768, 293.336 UID |
| OCR | `READY-DEVELOPMENT` | coverage 100% UID; 278.091 FTS row; 57 error; chưa final |
| ASR | `WAITING_KAGGLE` | người dùng xác nhận đang chạy; chờ hoàn tất mới tải HF |
| Online core/UI | `IMPLEMENTED` | Streamlit trực tiếp `OnlineEngine`, health 200 |
| Submission official | `IMPLEMENTED` | CSV/ZIP fail-closed theo AIC26 qualifier |
| Accuracy acceptance | `OPEN` | evaluator/runner/freeze đã có; 60 phiếu gán nhãn trống đã tạo, chưa có nhãn người |

Implementation legacy `backend/`, `frontend/`, 12 tài liệu archived và ZIP recovery local
đã được xóa theo xác nhận của người dùng. Runtime Online duy nhất là `online/streamlit_app.py`
gọi trực tiếp `OnlineEngine`; các file tracked đã xóa còn có thể khôi phục từ Git, ZIP
recovery untracked thì không còn bản local.

Footprint đo tại snapshot này: `$AIC_DATA` 109,28 GiB; video 77,28 GiB; keyframe 24,20 GiB;
runtime catalog + ba FAISS/state + OCR snapshot khoảng 2,31 GiB. Data retention chi tiết
được khóa tại baseline §3.2 và `docs/ONLINE_RUNBOOK.md` §2.1.

## Offline/visual closure

- Catalog SHA-256:
  `ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37`.
- UID-set SHA-256:
  `5bada00bd4a93928e48af3a6cbe7189a3b465eafb00cc8f829941edee536e660`.
- CLIP revision `4c4a3e8bcc2b768a8b89fc83ed8c828345ca3bac`.
- SigLIP revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`.
- EVA-CLIP revision `bf4190eb65dd5204ffb03e980108beb1200e0873`.
- Ba index đều khớp 293.336 UID/873 video, finite/L2, source float16, FAISS float32 và
  `checkpoint_resume_verified=true`.
- Visual artifact handoff được pin từ HF revision
  `938aefd437ab8db61fc6599d613aedcf4921d71e`.

## OCR hiện hành

Nguồn EasyOCR hiện tại là chín archive tại HF revision
`a5dcff74326f43421553481793d4a1e51eb59ce5`. Archive checksum, manifest, catalog SHA,
partition 873 video và layer completion gate đã verify trước khi build local.

Snapshot đang dùng:

```text
ocr-snapshot-20260828T153736Z-65e6f8bf8850
intended_use=online_development_only
coverage=293336/293336 UID (100%)
fts_rows=278091
success=278091
no_text=15188
error=57
missing=0
production_ready=false
```

57 error là frame CRAFT có region nhưng EasyOCR không trả text không-rỗng; không được đổi
thành `no_text`/success giả. Batch 02 có 47 và batch 03 có 10 error; bảy batch còn lại đạt
frame-level completion gate. Snapshot vẫn dùng được cho Online visible-text retrieval vì
SQLite/integrity/UID join hợp lệ, nhưng không thay thế terminal OCR final Vintern/Gemini.

Online chọn snapshot bằng:

```powershell
$env:AIC_OCR_SNAPSHOT_DIR = "$env:AIC_DATA\ocr\snapshots\ocr-snapshot-20260828T153736Z-65e6f8bf8850"
```

Streamlit hiện đã restart với biến này và health endpoint trả HTTP 200. Registry/UI hiện
snapshot ID, tier, coverage, error/missing và `production_ready=false`.

## Online Accuracy-Max đã implement

- Startup ArtifactRegistry fail-closed cho catalog, ba FAISS và optional OCR/ASR.
- Planner chain Gemini 3.5 Flash-Lite → Qwen local → rule fallback; visual query model phải
  là tiếng Anh, query gốc giữ để audit.
- Planner contract role-aware đa vai trò: `VIDEO_LOCATOR`, `TARGET_MOMENT`,
  `ANSWER_EVIDENCE`, `ORDERED_EVENT`. Streamlit chạy hai pha Analyze → operator edit/review →
  Search; Gemini dùng structured response schema, Qwen/rule qua cùng validator.
- `global_context_en`, global expansions và từng QueryUnit được search độc lập; không mean-pool.
- SigLIP + EVA Top 1.000/query, union UID, SRRF `eta=60`, `beta=40`; CLIP chỉ comparison,
  tie-break hoặc explicit rollback.
- OCR/ASR intent-aware, missing channel renormalize. OCR cascade exact → AND → prefix pool
  5.000 với token-coverage → fuzzy candidate hẹp.
- Temporal-neighbor boost-only; frame evidence dedup theo shot khi rank video.
- Video score tách locator 0,35 + target/event 0,45 + global 0,10 + consensus 0,10; giữ Top
  12 (TRAKE Top 20), sau đó target/evidence retrieval riêng quyết định frame.
- VLM verification Top 4 video bằng Gemini → Qwen fallback; partial VLM output không phạt
  frame bị bỏ qua.
- KIS chỉ rank submission frame bằng `submission_target_ids`; locator-only frame không chiếm
  Top 100, candidate không hard-dedup cùng shot và vẫn weighted round-robin đa video.
- QA luôn thử Top 3; locator 0,85 chỉ auto-accept. Unknown OCR/ASR đọc evidence UID + neighbor,
  confidence thấp fallback VLM; không sinh portfolio `Uncertain`, `requires_review` chặn export.
- TRAKE locator chạy riêng; chỉ `ordered_event_ids` đi vào beam width 8, cùng video, timestamp
  tăng, decay mặc định 0.
- UI có query-plan editor, `Top 100` và `Theo video`, exact-frame decode, atomic bulk-add,
  draft theo query.
- Export profile duy nhất `AIC26_QUALIFIER_OFFICIAL`; ZIP chỉ chứa `submission/*.csv`.

## Validation đã chạy

- Full Online regression **55/55 test PASS**; toàn repo **242/242 test PASS**: role-aware
  planner, SRRF/fusion, retrieval/CLIP rollback, video/task heads, OCR UID-neighbor answer,
  VLM partial-output behavior, Torch worker và official CSV/ZIP.
- Deep preflight PASS: hash/state/structure/full UID diff của ba FAISS và checksum/catalog/
  FTS5/integrity/UID join của OCR snapshot đều hợp lệ.
- Probe thật `giá dầu mazut` sau prefix token-coverage đưa `L22_V029` lên Top 1 và các bảng
  giá `L22_V001`/`L22_V008` lên đầu candidate OCR.

## Việc còn mở theo thứ tự

1. Rerun bộ kiểm tra thủ công có ground truth; báo Video Recall@12, Frame/shot Recall@100,
   QA answer/evidence và TRAKE sequence.
2. Điều tra/sửa 57 OCR error hoặc materialize snapshot tầng cao hơn; giữ snapshot mới bất biến.
3. Build/validate `asr.sqlite` hoặc chấp nhận spoken-text degraded mode có warning.
4. Freeze config/model revision/quota và tạo official ZIP canary ngay trước vòng thi.

Không commit, push, rebuild embedding hoặc gọi Gemini OCR paid trong trạng thái hiện tại.
