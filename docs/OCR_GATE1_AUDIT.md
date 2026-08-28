# OCR Gate 1 audit — 26/08/2026

Kết luận hiện tại: **GATE 1 còn một blocker — canary lại final request contract**. Kiến trúc
đã được người dùng làm rõ là CRAFT gate toàn catalog, Gemini nhận dạng crop có chữ theo một
contact-sheet/shot, `latin_g2` fallback. Catalog, weight provenance, envelope/router và quota
đã có; canary 10 shot cũ chưa dùng response `region_id` + `MEDIA_RESOLUTION_MEDIUM`, nên chỉ
là evidence tham khảo và không được dùng để mở Gate 2.

## 1. Catalog thật — PASS

- Đã xác nhận Dataset private `MinhThuw0103/lastdance-visual-embeddings`, đúng account
  `MinhThuw0103`, và pin revision
  `72848939bdc5ebd57b5cd45370e685aee036cafa` trước khi list/download.
- Revision có 53 file. Không có `frames.csv`, state hoặc CLIP production report standalone;
  có `production-workers/production-batch-mapping.json`, 9 worker list và 9 CLIP archive
  cùng checksum. Catalog được recover từ
  `clip/archives/batch-01/lastdance-production-batch-01-clip.tar.gz`, đúng các member
  `catalog/frames.csv`, `catalog/frames.csv.state.json` và mapping nhúng.
- Mapping standalone và mapping nhúng giống byte-for-byte, cùng SHA-256
  `e7e519e5fe3e47c3e487bfe0522c09c3f0bae6c7f67dff2d31168aead0b911d2`.
- SHA-256 catalog thực tính lại là
  `ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37`, khớp đồng thời
  mapping standalone, mapping nhúng và `frames.csv.state.json`.
- Audit parse toàn bộ CSV, không dùng expected count: **293.336 keyframe**, **873 video**,
  min/max **24/1.257 keyframe/video**, UID-set SHA-256
  `5bada00bd4a93928e48af3a6cbe7189a3b465eafb00cc8f829941edee536e660`, UID formula
  mismatch `0`, duplicate UID `0`, state complete.
- Chín worker list disjoint, không duplicate/overlap; từng batch recompute đúng cả
  `video_count` và `keyframe_count`; union worker bằng đúng tập 873 video catalog và tổng
  **293.336** record.
- Catalog audit local nằm dưới `tmp/ocr-gate1/recovered/index/`; `AIC_DATA` đã được trỏ
  task-scoped tới `tmp/ocr-gate1/recovered` và CLI audit PASS.

Archive được đọc streaming tới hết ba member cần thiết (64.184.320 byte), không tải toàn bộ
embedding archive. Vì vậy checksum archive đầy đủ không được tuyên bố đã verify; kết luận
catalog dựa trên catalog SHA-256 khớp độc lập qua mapping remote + mapping nhúng + state.

## 2. Gemini tier/quota/canary — quota PASS, final-contract canary PENDING

- Project đã verify trực tiếp trong Google AI Studio: ID `gen-lang-client-0009440353`,
  project number `823595296842`, Free tier. Bảng Rate Limit của đúng project hiển thị:
  `gemini-3.1-flash-lite` **15 RPM, 250.000 input TPM, 500 RPD** (snapshot usage
  3/15 RPM, 6,96K/250K TPM, 8/500 RPD) và `gemini-3.5-flash-lite` cùng trần (snapshot đã
  vượt 17/15 RPM, 25,28K/250K TPM, 505/500 RPD).
- `gemini-2.5-flash-lite` trả HTTP 404 với thông báo không còn cấp cho user mới. Model đã
  canary thật là `gemini-3.5-flash-lite`; response trả catalog version
  `3.5-flash-lite-07-2026`. Không dùng alias `latest` và không được diễn giải canary này là
  bằng chứng cho model 2.5.
- Strict structured output dùng `responseJsonSchema`, ép mỗi bbox đúng 8 số, tọa độ và
  confidence trong `[0,1]`, `additionalProperties=false`. Trên 100 unique request có 117
  HTTP attempt: 97 terminal success, 3 terminal timeout/quota; **97/97 HTTP 200 schema-valid**,
  invalid bbox 0. Latency success P50 2.190 ms, P95 2.647 ms.
- Trung bình mỗi response thành công dùng 1.184 prompt token và 332,91 output token. Peak
  17 RPM đã nhận `429 RESOURCE_EXHAUSTED`; ramp dừng khi dashboard đạt 502/500 RPD. Harness
  đã sửa để retry cũng được pace theo thời điểm attempt thật, không catch-up burst sau timeout.
- Với giả định một request/keyframe, 500 RPD cần khoảng 586,67 ngày cho 293.336 keyframe;
  trong 6 ngày chỉ xử lý tối đa 3.000 request (~1,023% catalog). TPM không phải bottleneck;
  RPD là bottleneck quyết định.

Canary tham khảo pin `gemini-3.1-flash-lite`: detector local tạo crop-sheet của ba
keyframe/shot, adapter giữ bbox gốc. Trên 10 synthetic shot (30 frame), 10/10 request thành
công, schema-valid 100%, synthetic line recall 90/90, tổng 12.500 input + 2.574 output token,
trung bình 1.250 + 257,4 token/shot, latency min/max 1.984/6.114 ms. Summary SHA-256
`97944101f2eaec1533b6cfc8c59d4cae1934f17224821d6af2955f6e745d7fab`; đây chỉ là
schema/token evidence, `production_recall_claimed=false`. Nó đã bị supersede vì response khi
đó gom text theo `frame_id`, chưa exact-set `region_id`, chưa tách ba `keyframe_uid` trong
request provenance và chưa đặt global `MEDIA_RESOLUTION_MEDIUM`.

Theo giá Batch pin trong config ($0,125/M input, $0,75/M output) và tỷ giá vận hành
26.300 VND/USD, nếu tỷ lệ token synthetic giữ nguyên thì 97.810 shot có giá lý thuyết
~$34,17/~898.540 VND, hoặc ~$39,29/~1.033.321 VND với reserve 15%. Vì ảnh thật có thể
khác và người dùng đã chốt budget cứng, production vẫn giới hạn **20.000 paid frame và
400.000 VND**; không dùng phép chiếu synthetic để tự nâng cap.

Evidence sanitize nằm tại
`tmp/ocr-gate1/gemini-canary/strict-schema-gate1-summary-20260826.json`, SHA-256
`c06c86eb12a6721edaf78622145a230faec5b0cee335734e7c95a51f390ac921`; thư mục `tmp/`
không commit. Không có API key/header nhạy cảm trong summary.

Quyết định đã chốt: CRAFT là detector gate; CRAFT `no_text` không gọi cloud. Gemini 3.1 là
recognizer primary cho crop CRAFT-positive, còn `latin_g2` nhận overflow/cloud stop. Ba
keyframe/shot dùng một request. Embedding chỉ xếp lịch; reuse một frame cần đồng thời
embedding + CRAFT layout + crop SSIM + pHash. Không phụ thuộc vào gom nhiều tài khoản.

## 3. EasyOCR provenance + offline bytes — PASS (checksum audit)

Nguồn package là PyPI `easyocr==1.7.2`, wheel SHA-256
`5be12f9b0e595d443c9c3d10b0542074b50f0ec2d98b141a109cd961fd1c177c`; upstream tag
`v1.7.2` resolve tới commit `c4f3cd7225efd4f85451bd8b4a7646ae9a092420`.

| Model | ZIP bytes | ZIP SHA-256 | `.pth` bytes | `.pth` SHA-256 |
|---|---:|---|---:|---|
| CRAFT `craft_mlt_25k.pth` | 77,251,756 | `8dc6a1c703a89ed56308ef742d26ebd45c656248cbbbda6e7fe60e569f873e65` | 83,152,330 | `4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17` |
| `latin_g2.pth` | 14,284,936 | `29f1920c493378da65a59793fb70e7e190504662b6bed57ab26f4067eb5f3769` | 15,406,141 | `aaa95be1c4a9cb3496879bed7c520886ce1164f89e026f0c54488394e74e8c55` |

SHA-256, size và MD5 upstream của cả hai extracted weight đã verify bằng file tải thật.
Registry/CLI ép `download_enabled=False`. Phiên này chỉ PASS provenance/checksum; runtime
Reader trên profile production vẫn phải chạy preflight package-version đầy đủ trước Gate 2.

## 4. Artifact envelope + routing policy — PASS bằng unit test

- `OcrResult` và `ocr_fts` không đổi.
- Record envelope schema v1 giữ `video_id`, `keyframe_uid`, `frame_id`, relative image path,
  execution mode, `success/no_text/error`, engine/fallback/attempt provenance và
  `OcrResult | null`. Production dùng `craft_gated_gemini`; attempt ghi stage detection/
  recognition. CRAFT `no_text` terminal không có Gemini attempt; overflow kết thúc bằng
  EasyOCR recognition với `fallback_used=true`.
- Bbox canonical là normalized clockwise quadrilateral 8 số; `no_text` không tạo language
  giả. Completion gate yêu cầu exact UID, `error=0`, không duplicate/missing/foreign.
- Chín JSONL terminal shard phải exhaustive/disjoint. Contact sheet trả region rows; adapter
  map `region_id` về UID/frame/bbox local. Chỉ build một SQLite sau exact UID; auth/quota/
  budget stop xuất routing state rồi dùng fallback job, không biến crop chưa xử lý thành
  `no_text`.

Policy schema v2 nằm ở `configs/ocr_escalation_policy.json`; router gom shot và fail closed
theo cả 20.000 frame/400.000 VND. Final-contract canary và ảnh thật vẫn pending; chưa có claim
recall hoặc throughput production.
