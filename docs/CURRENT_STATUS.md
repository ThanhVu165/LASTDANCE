# Trạng thái hiện tại của LASTDANCE

Cập nhật OCR: 04/09/2026; evidence ngoài OCR giữ mốc 29/08/2026, chưa xác minh lại.
File này là **snapshot hiện hành**, không phải append-only session log.
Contract chuẩn nằm tại `docs/BASELINE_SPEC.md`; lệnh vận hành nằm trong các runbook.

## Kết luận mức hệ thống

**Mức hiện tại: integration-ready cho KIS/QA/TRAKE và submission vòng sơ tuyển; chưa được
phép gọi là accuracy-complete.** Offline visual đã đóng/PASS. Online Accuracy-Max và official
export đã implement. OCR EasyOCR đã tích hợp dưới snapshot development. ASR chưa có. Bộ
kiểm tra thủ công/ground-truth cần chạy lại để chốt Recall/accuracy sau thay đổi Online/OCR.

| Thành phần | Trạng thái | Evidence hiện hành |
|---|---|---|
| Inventory/video/shot | `READY` | 873 MP4, 873 shot manifest, inventory full |
| Keyframe catalog | `READY` | 293.336 JPEG/UID, 873 video |
| CLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 512, 293.336 UID |
| SigLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 768, 293.336 UID |
| EVA-CLIP FAISS | `READY` | `IndexIDMap(IndexFlatIP)`, dim 768, 293.336 UID |
| OCR | `READY-DEVELOPMENT` | coverage 100% UID; 278.091 FTS row; 57 error; chưa final |
| OCR v2 | `SNAPSHOT-VALIDATED-DEVELOPMENT` | 9/9 batch HF; local coverage 293.336/293.336 UID, 269.259 FTS row, 8.889 error; chưa chuyển Online/final |
| ASR | `UNAVAILABLE` | chưa có `asr.sqlite` |
| Online core/UI | `IMPLEMENTED` | Streamlit trực tiếp `OnlineEngine`, health 200 |
| Submission official | `IMPLEMENTED` | CSV/ZIP fail-closed theo AIC26 qualifier |
| Accuracy acceptance | `OPEN` | chưa rerun đủ ground-truth sau Online/OCR update |

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

### Recognition v2 đã xong, chưa thay artifact Online

Entry point nhóm nhận: [OCR_V2_ONLINE_HANDOFF.md](OCR_V2_ONLINE_HANDOFF.md).
Người dùng hoãn chấm ground truth của audit OCR v2 để bàn giao development kịp thi;
không tuyên bố quality PASS. Git chỉ có code/hướng dẫn; SQLite hiện local, raw results ở HF.

Theo [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2 bản 28: CRAFT bbox cache từ chín archive
EasyOCR HF → VietOCR mọi crop gốc → Paddle có điều kiện → Gemini residual tùy chọn sau
duyệt riêng. Bốn T4, batch khởi đầu 64/128, log tối đa 30 giây, checkpoint local từng
minibatch + HF verified theo spec. Không rerun EasyOCR/Vintern, không bật làm nét.

Gate B đã có artifact thực tế, chưa có ground truth đủ để kết luận PASS recall/CER.
Trial làm nét hoàn tất 90/90; original khớp Gate B 30/30, 10 đối chứng không đổi;
review không xác nhận đủ ba crop cải thiện rõ. Chốt giữ gốc, không chạy lại trial.
SHA/signature và limitation được ghi ở baseline §2.2a; không sửa report ZIP bất biến.

Planner đã chia đúng chín batch cho bốn worker. Log/report người dùng cung cấp xác nhận mỗi
worker kết thúc `[DONE]`, từng batch có `recognition_complete=true` và result/report được
`HF_VERIFIED`; batch chạy lại đã restore prediction đúng signature thay vì nhận dạng lại.
Các cờ `complete=false`, `production_ready=false` là đúng vì đây mới là recognition shard.

Migration đã được tách khỏi legacy bằng coverage schema v3: pin source theo revision/hash,
validate raw prediction/selection/residual, kiểm chín UID shard disjoint/exhaustive rồi
atomic-build đúng năm cột FTS. Test tổng hợp 9 batch đã PASS. Ngày 04/09/2026, dữ liệu thật
đã sync tại HF revision `8ca4271dd0218d3f3f3967a4d8a5c6aeebeaddc5`; các export resume
batch 03/07/09 được chứng minh tương đương theo member hash rồi chọn bản mới nhất.

Snapshot OCR v2 local đã qua validator độc lập:

```text
ocr-snapshot-20260904T081629Z-66ecea73cce1
coverage=293336/293336 UID (100%)
fts_rows=269259
success=269259
no_text=15188
error=8889
residual_frames=240976
residual_regions=763395
sqlite_sha256=9b80eed3ef376655b6a4ad6c9496072f2cf215dec38f5af9e5095ebb491ed78e
complete=false
production_ready=false
```

Preflight consumer chỉ đọc đã xác nhận `online.fts.FtsSearcher` truy vấn trực tiếp SQLite
thành công với các probe `giá dầu mazut`, `Việt Nam`, `Hà Nội` và `2026`. Handoff chưa được
bật: `online.artifacts.ArtifactRegistry` hiện parse `coverage.json` bằng manifest EasyOCR
schema 1/2 nên từ chối OCR v2 schema 3/`ocr_v2_batch_union_v1`. Đây là thay đổi Nhánh 2 cần
được yêu cầu riêng; không đổi `AIC_OCR_SNAPSHOT_DIR` trước khi có adapter + regression test.

Xem
[runbook recognition](OCR_V2_PRODUCTION_RUNBOOK.md) và
[runbook snapshot](OCR_V2_SNAPSHOT_RUNBOOK.md). Chưa thay snapshot hay consumer Online;
không gọi model/GPU/API từ máy Codex.

### Artifact EasyOCR development đang dùng (không đổi)

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
SQLite/integrity/UID join hợp lệ, nhưng không phải artifact OCR v2 hoặc OCR final.

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

- Toàn repo **305/305 test PASS** trên Windows Python 3.11.9 trước commit bàn giao OCR v2.
  Đã sửa fixture Gate A CSV thiếu cột `video_id` và regenerate review notebook để source
  nhúng khớp runtime. Test không thay thế accuracy/ground-truth hoặc T4 production evidence.
- Deep preflight PASS: hash/state/structure/full UID diff của ba FAISS và checksum/catalog/
  FTS5/integrity/UID join của OCR snapshot đều hợp lệ.
- Probe thật `giá dầu mazut` sau prefix token-coverage đưa `L22_V029` lên Top 1 và các bảng
  giá `L22_V001`/`L22_V008` lên đầu candidate OCR.

## Việc còn mở theo thứ tự

1. Rerun bộ kiểm tra thủ công có ground truth; báo Video Recall@12, Frame/shot Recall@100,
   QA answer/evidence và TRAKE sequence.
   Accuracy vẫn OPEN; riêng chấm ground truth OCR v2 đã được người dùng hoãn, không chặn
   bàn giao code/snapshot development ở đợt này.
2. Review 8.889 error/763.395 residual và chỉ mở Gemini residual sau duyệt chi phí riêng;
   kiểm consumer Nhánh 2 trước khi chủ động chuyển Online sang snapshot OCR v2.
3. Build/validate `asr.sqlite` hoặc chấp nhận spoken-text degraded mode có warning.
4. Freeze config/model revision/quota và tạo official ZIP canary ngay trước vòng thi.

Không commit, push, rebuild embedding hoặc gọi Gemini OCR paid trong trạng thái hiện tại.
