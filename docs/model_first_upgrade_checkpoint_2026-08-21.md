# Checkpoint nâng cấp model-first — tạm dừng rồi tiếp tục ngày 21/08/2026

> **Checkpoint lịch sử, không phải runbook.** Dùng `docs/PROJECT_CONTEXT.md` và
> `docs/TEAM_SETUP.md` cho trạng thái/thiết lập hiện hành.

> Cập nhật sau khi tiếp tục: structured planner, generative verification, repair
> retrieval và verified KIS Top 100 đã được bật mặc định. Smoke HTTP thật trả
> 100/100 KIS row có model score trong 94,9 giây; QA trả 100 row trong 82,6 giây.
> Backend dedicated `Qwen3-VL-Reranker-2B`, full SigLIP2 index và Qwen video-window
> index vẫn chưa hoàn tất. Trạng thái vận hành mới nhất nằm trong README.

Tài liệu này là điểm tiếp tục chính thức cho nhánh nâng cấp retrieval/rerank. Mọi
bước tải model, chạy smoke test GPU, build index và xóa code cũ đã được dừng theo
yêu cầu. Không có tiến trình `hf` hoặc `curl` nào còn chạy. Backend đang chạy của
người dùng không bị Codex dừng; cần chủ động restart ở phiên sau để nhận code mới.

## 1. Mục tiêu không thay đổi

Ưu tiên theo thứ tự:

1. KIS: hiểu toàn bộ mô tả scene, tìm đúng video/đoạn trước, đưa đáp án đúng nhất
   lên Top 1 và vẫn trả đủ Top 100.
2. QA: dùng retrieval KIS đã được kiểm chứng để tìm đúng video/đoạn, sau đó mới
   trả lời và kiểm tra lại answer.
3. TRAKE: tìm đúng video trước, rồi căn chỉnh đúng chuỗi hành động và frame biên.
4. ASR tạm hoãn. OCR là evidence phụ quan trọng nhưng không chặn việc phát triển
   retrieval hiện tại.

Mục tiêu đánh giá là Recall/R-Score tại các mốc `1, 5, 20, 50, 100` theo tài liệu
vòng sơ tuyển, không chỉ tối ưu một điểm tổng hợp tùy ý. Model verifier chỉ tạo
điểm tin cậy; không được coi kết luận của model là ground truth. Muốn khẳng định
“đúng” bắt buộc phải có tập query–video/frame được gán nhãn trên dataset hiện tại.

## 2. Trạng thái trước khi công việc được tiếp tục

Ba tính năng WIP đang để **opt-in, mặc định tắt** trong `backend/app/config.py`:

```text
AIC_MODEL_QUERY_PLANNER_ENABLED=0
AIC_MODEL_RERANK_ENABLED=0
AIC_MODEL_REPAIR_ENABLED=0
```

Vì vậy, nếu backend được restart ngay lúc này, KIS vẫn rơi về luồng ổn định trước
đây: parser/translation hiện hữu → CLIP/fusion/storyboard → Qwen tournament →
cutoff-aware Top 100 → exact-frame Top 1. Không bật ba cờ trên trước khi hoàn tất
các smoke test trong mục 7.

Không xóa các thư mục sau:

- `backend/data/index`, gồm cả OCR cache/state và FAISS production.
- `backend/data/keyframes`, video, features, map-keyframes và object data.
- Cache model ở `C:\Users\Vu\.cache\huggingface` và `C:\Users\Vu\.paddlex`.
- Virtual environment `backend/.venv`.

## 3. Những thay đổi đã có trong working tree

Chưa commit và chưa push. `git status` tại thời điểm dừng:

```text
 M backend/app/config.py
 M backend/app/main.py
 M backend/app/pipelines/kis_pipeline.py
 M backend/app/services/visual_qa.py
 M backend/requirements.txt
?? backend/app/evaluation/model_rerank_smoke.py
?? backend/app/indexing/siglip_index.py
?? backend/app/rerank/model_reranker.py
?? backend/app/services/query_planner.py
?? backend/app/services/side_search.py
```

### 3.1. Phần đã viết

- `query_planner.py`: Qwen3-VL-2B-Instruct text-only sinh JSON có scene, caption
  retrieval song ngữ, điều kiện `must_have`, visible text, quan hệ thời gian và
  repair query. Có validation giới hạn kích thước và fallback về parser cũ.
- `model_reranker.py`: adapter cho `Qwen/Qwen3-VL-Reranker-2B`; gom frame đại diện
  của mỗi video thành contact sheet rồi chấm pointwise. Runtime chỉ đọc local
  cache nên không bao giờ treo request để tải model 4 GB.
- Cùng file trên có generative verifier dùng Qwen3-VL-2B-Instruct đã cache: chấm
  tất cả video trong pool theo nhóm thay vì tournament chỉ chọn một winner.
- `kis_pipeline.py`: planner model-first → recall → fusion/storyboard → dedicated
  reranker; nếu dedicated model chưa có thì generative verifier; nếu verifier lỗi
  thì tournament cũ. Có một repair round để mở rộng candidate pool khi số video
  tin cậy còn thấp.
- `visual_qa.py`: có `release_vqa()` để giải phóng model instruct trước khi tải
  reranker 2B khác, tránh hai model cùng chiếm GPU 6 GB.
- `/health`: công bố trạng thái planner/reranker, model name và độ sẵn sàng của
  side index.
- `siglip_index.py`: builder SigLIP2 checkpointable trên keyframe, lưu float16
  memmap và chỉ publish FAISS khi hoàn thành toàn bộ.
- `side_search.py`: loader/search tùy chọn cho SigLIP2 và Qwen video-window; index
  thiếu hoặc chưa hoàn chỉnh thì trả pool rỗng, không làm hỏng production.
- `model_rerank_smoke.py`: smoke test lấy candidate thật từ CLIP, đo thời gian,
  peak VRAM và in top result.

### 3.2. Dependency đã cài trong `.venv`

- `sentence-transformers==6.0.0`
- `transformers==5.15.1`
- `qwen-vl-utils==0.0.14`

`backend/requirements.txt` đã được cập nhật tương ứng. Sau lần nâng dependency,
toàn bộ 56 test hiện hữu đã pass. Tuy nhiên, kết quả đó có trước khi thêm
`siglip_index.py`, `side_search.py` và generative verifier cuối cùng; trạng thái
mới nhất **chưa được compile/test lại** vì phiên làm việc được yêu cầu tạm dừng.

## 4. Model đã chọn và trạng thái tải

### Model đang có thể dùng ngay

`Qwen/Qwen3-VL-2B-Instruct` là model planner, generative video verifier, exact-frame
refiner và QA answerer. Model đã có trong cache và trước đây đo được khoảng 4,2 GiB
VRAM FP16; phù hợp RTX 4050 6 GB nếu OCR/Paddle không chạy đồng thời.

### Model reranker chuyên dụng

Model đích: `Qwen/Qwen3-VL-Reranker-2B`.

- Hỗ trợ text–image/video, đa ngôn ngữ và chấm trực tiếp cặp query–document.
- File `model.safetensors` chính xác 4.255.140.312 byte (~3,96 GiB).
- Metadata/tokenizer đã tải; trọng số chưa hoàn chỉnh.
- Cache hiện còn file `.incomplete` khoảng 128 MiB tại:
  `C:\Users\Vu\.cache\huggingface\hub\models--Qwen--Qwen3-VL-Reranker-2B`.
- Hugging Face CLI bị đứng; tải HTTP trực tiếp chỉ đạt khoảng 15–20 KB/s, ước tính
  hơn 70 giờ. Cả hai tiến trình đã được dừng, phần cache không bị xóa.

Khi mạng ổn định, thử downloader chính thức trước:

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Reranker-2B
```

Nếu Xet tiếp tục treo, thử HTTP thường:

```powershell
$env:HF_HUB_DISABLE_XET = "1"
.\.venv\Scripts\hf.exe download Qwen/Qwen3-VL-Reranker-2B
```

Chỉ coi là hoàn tất khi snapshot có `model.safetensors` đủ đúng kích thước; không
coi file `.incomplete` là model hợp lệ. Runtime giữ
`AIC_MODEL_RERANK_LOCAL_FILES_ONLY=1` để request không tự tải model.

## 5. Phần đang dở hoặc chưa triển khai

1. Generative verifier mới viết xong nhưng chưa compile, unit test hoặc smoke test
   thật. Cần kiểm tra parser `V1=<score>;...`, OOM, latency và độ ổn định điểm giữa
   các group.
2. Dedicated Qwen reranker chưa chạy được vì thiếu trọng số. Chưa có số peak VRAM,
   latency hoặc xác nhận model vừa RTX 4050 6 GB.
3. Query planner chưa smoke test thật bằng query tiếng Việt dài sau nâng cấp
   Transformers. Chưa đo JSON-valid rate và thời gian.
4. SigLIP2 builder/side search chưa test. Chưa build dù chỉ một checkpoint nhỏ.
5. Chưa có `video_window_index.py`. Side loader Qwen video-window mới là khung,
   chưa thể hoạt động.
6. SigLIP2/video-window side retrieval chưa hợp nhất vào candidate generation KIS.
7. Repair loop hiện mới có tối đa một round và mục tiêu mặc định 20 video tin cậy;
   chưa phải vòng lặp mở rộng cho đến khi có 100 kết quả đã kiểm chứng.
8. QA mới chỉ có khả năng thừa hưởng KIS retrieval sau khi KIS ổn định; chưa thêm
   answer verification/consistency pass.
9. TRAKE chưa dùng planner/reranker mới. Việc này để sau KIS và QA.
10. Chưa tạo dev set ground truth từ dataset hiện tại, nên chưa có phép đo accuracy
    để quyết định thay index production.
11. Chưa xóa code cũ. Tournament cũ vẫn là fallback đang dùng, vì vậy chưa phải
    dead code và tuyệt đối chưa được xóa ở checkpoint này.

## 6. Pipeline đích cần hoàn thành

### 6.1. KIS

```text
query tự nhiên
  → model planner sinh scene + atomic must-have + temporal order
  → multi-retriever recall: organizer CLIP + OCR + SigLIP2 + video-window
  → hiệu chuẩn từng retriever (rank calibration/RRF), union candidate reservoir
  → gom frame/window thành video hypothesis và ordered storyboard
  → multimodal reranker kiểm tra toàn bộ điều kiện của query
  → nếu chưa đủ video tin cậy: sinh repair query cho evidence còn thiếu,
    lấy candidate chưa kiểm tra và lặp trong ngân sách thời gian
  → exact-frame refinement cho các hạng đầu
  → cutoff-aware ranking và trả đúng 100 dòng
```

“Đủ 100 đúng” phải được diễn giải thành: cố gắng tạo 100 hypothesis vượt ngưỡng
model và đa dạng segment/video, sau đó đo bằng ground truth. Không model nào có thể
bảo đảm 100/100 đúng chỉ bằng tự chấm. Vòng lặp phải có ba điều kiện dừng: đủ 100,
hết candidate mới, hoặc chạm ngân sách KIS 180 giây.

### 6.2. QA

```text
query → tách event retrieval và câu hỏi
  → KIS model-first tìm/kiểm chứng đúng video-window
  → lấy nhiều frame quanh thời điểm liên quan
  → Qwen sinh answer có ràng buộc ngôn ngữ/format
  → lượt verifier độc lập kiểm tra answer có được hỗ trợ bởi frame hay không
  → rerank theo retrieval evidence × answer support
  → trả Top 100
```

QA verifier phải cho phép `unknown`/điểm thấp khi evidence không đủ, tránh hallucinate.
Các câu hỏi đếm cần chọn đúng frame thời điểm; câu hỏi diễn biến cần chronological
window; câu hỏi tên riêng/logo cần phối hợp OCR và visual evidence.

### 6.3. TRAKE

Sau khi KIS và QA đạt mục tiêu:

```text
query → planner tách ordered moments
  → retrieve window cho từng moment
  → join các moment trong cùng video với thứ tự tăng nghiêm ngặt
  → multimodal rerank cả sequence
  → coarse-to-fine trên video nguồn cho từng action boundary
  → Top 100 sequence hypotheses
```

## 7. Thứ tự tiếp tục bắt buộc

### Bước 1 — xác nhận working tree, không tải model

```powershell
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m compileall -q app
.\.venv\Scripts\python.exe -m pytest -q
```

Sửa hết compile/test regression trước khi bật feature flag. Bổ sung unit test cho:

- planner parse JSON hợp lệ, JSON lỗi và fallback;
- dedicated reranker unavailable phải fail nhanh, không truy cập mạng;
- generative score parser đủ/thiếu/sai row;
- blend score và đánh dấu `model_verified`;
- repair loop không chấm lặp candidate cũ và luôn dừng đúng giới hạn.

### Bước 2 — smoke Qwen instruct đã có sẵn

Bật planner và generative verifier trong một terminal test riêng:

```powershell
$env:AIC_MODEL_QUERY_PLANNER_ENABLED = "1"
$env:AIC_MODEL_RERANK_ENABLED = "1"
$env:AIC_MODEL_REPAIR_ENABLED = "0"
cd C:\LASTDANCE\backend
.\.venv\Scripts\python.exe -m app.evaluation.model_rerank_smoke `
  --query "một người mặc áo đỏ bước vào xe rồi chiếc xe rời đi" `
  --candidates 4
```

Lặp với 8, 12 rồi 30 candidate. Ghi lại:

- model backend được dùng (`generative` khi dedicated chưa có);
- số video thực sự được chấm và tỉ lệ output parse thành công;
- Top result trước/sau;
- latency và peak reserved VRAM;
- lỗi CUDA/OOM nếu có.

Nếu 4 candidate đã OOM, giảm `VQA_MAX_PIXELS`, frame/video hoặc group size; không
bật production. Nếu điểm tuyệt đối tiếp tục bão hòa, chuyển generative verifier
sang comparative ranking có anchor chung, không dùng score đó như xác suất.

### Bước 3 — hoàn tất và smoke dedicated reranker

Sau khi tải đủ model, chạy cùng smoke ở 4 → 12 → 30 → 60 video. Tiêu chí tối thiểu:

- load và inference không OOM trên 6 GB;
- process không giữ đồng thời instruct VLM và reranker;
- request không truy cập mạng;
- 60 video nằm trong ngân sách KIS 180 giây;
- output khác biệt hợp lý giữa match đủ và partial match.

Nếu FP16 không vừa GPU, không ép chạy bằng mẹo chắp vá. Thử theo thứ tự: giảm
image pixels/batch về 1; kiểm tra quantization được model chính thức hỗ trợ; nếu
vẫn không đạt thì dùng generative verifier đã benchmark hoặc chạy reranker ngoài
máy như một dịch vụ tùy chọn.

### Bước 4 — làm candidate recall trước khi tinh chỉnh rerank

1. Sửa/test `siglip_index.py` bằng `--limit 32`, sau đó resume đến 100–500 frame.
2. Kiểm tra row alignment giữa feature và `keyframe_index.json`.
3. Viết checkpointable `video_window_index.py`: cửa sổ 6 keyframe, stride 3,
   contact sheet/sequence input, metadata có `video_id`, local/frame id và thời gian.
4. Thêm calibration+RRF để hợp nhất CLIP/SigLIP2/window; không cộng cosine thô từ
   các model khác không cùng thang đo.
5. Chỉ build toàn bộ khi subset A/B có Recall@100 tốt hơn baseline.

Lệnh dự kiến cho SigLIP2:

```powershell
.\.venv\Scripts\python.exe -m app.indexing.siglip_index `
  --limit 32 --batch-size 4 --checkpoint-every 16
```

Không chạy builder này đồng thời với OCR, backend VQA hoặc reranker trên GPU.

### Bước 5 — adaptive verification đến Top 100

Thay repair loop một vòng bằng reservoir có trạng thái:

- đánh dấu video/window đã kiểm chứng;
- lấy batch mới từ repair query nhắm vào condition bị thiếu;
- rerank batch mới, hợp nhất với verified pool;
- dừng khi có 100 kết quả vượt ngưỡng, hết candidate hoặc hết 180 giây;
- nếu chưa đủ 100 verified, lấp phần còn lại bằng retrieval ranking và ghi rõ
  `model_verified=false`, không giả vờ rằng model đã kiểm tra.

Đo các ngưỡng trên dev set; không hard-code màu, object hoặc mẫu câu riêng lẻ.

### Bước 6 — dev set và ablation

Gán nhãn một tập query đại diện gồm tiếng Việt/Anh, query ngắn/dài, nhiều scene,
action, count, color, spatial relation, OCR và partial-match negatives. Báo cáo:

- Recall@1/5/20/50/100, MRR;
- video recall trước rerank và sau rerank;
- constraint coverage/partial-match error;
- P50/P95 latency, peak RAM/VRAM, kích thước/thời gian build index;
- ablation: CLIP → +planner → +SigLIP2/window → +reranker → +repair.

Chỉ khi có số liệu tốt hơn mới bật mặc định và cập nhật runbook production.

### Bước 7 — QA rồi TRAKE

Khi KIS đạt tiêu chí, triển khai QA answer verifier và regression test. Sau đó mới
tích hợp planner/window reranker vào TRAKE. ASR vẫn ở backlog riêng.

### Bước 8 — xóa code cũ có kiểm soát

Chỉ xóa sau khi đường mới pass E2E và không còn import/call site (`rg` xác nhận):

- module MMR cũ nếu thực sự không còn tham chiếu;
- helper OCR/object/QA legacy không có production hoặc test consumer;
- code tournament cũ sau khi dedicated/generative fallback đã ổn định.

Không xóa cache/index/dataset/model. Trước và sau mỗi đợt xóa phải chạy toàn bộ
test và xem `git diff`; nên commit riêng để có thể revert dễ dàng.

## 8. Tiêu chí kết thúc lộ trình

- Runtime không bao giờ tự tải model; `/health` nói rõ backend đang dùng loại nào.
- KIS luôn trả 100 dòng hợp lệ, không trùng row, đúng mapping `frame_id` thật.
- Recall@1/5/20/50/100 trên dev set tăng so với baseline, đặc biệt partial-match
  giảm rõ rệt; P95 KIS không quá 180 giây.
- QA luôn tìm đúng video/window trước khi answer; answer support cải thiện và P95
  không quá 300 giây.
- TRAKE đúng video trước, moment theo thứ tự và exact-frame được đo riêng.
- Peak VRAM có margin an toàn trên RTX 4050 6 GB; không chạy đồng thời hai model
  2B hoặc OCR worker.
- Có bảng ablation, log smoke/E2E và lệnh vận hành được cập nhật trong README.

## 9. Câu lệnh khôi phục phiên làm việc

Ở phiên sau, yêu cầu Codex đọc theo thứ tự:

1. `docs/model_first_upgrade_checkpoint_2026-08-21.md` (file này).
2. `docs/retrieval_upgrade_plan.md`.
3. `docs/e2e_test_report_2026-08-21.md`.
4. `docs/round1_contest_runbook.md`.
5. `git diff` của toàn bộ file ở mục 3.

Sau đó bắt đầu từ **mục 7, Bước 1**; không tải lại hay viết lại các phần đã có,
không bật feature flag và không xóa code cũ trước khi test.
