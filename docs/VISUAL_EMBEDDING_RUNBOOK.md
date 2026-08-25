# Runbook Visual Embedding trên Kaggle

Tài liệu này là entry point cho bước 2.1 của `OFFLINE_INDEXING_SPEC.md`: keyframe JPEG +
`frames.csv` → vector `float16` L2-normalized của từng keyframe. Chạy đúng **một modality
mỗi lệnh**; CLIP, SigLIP và BEiT-3 có checkpoint/artifact riêng, không có barrier chờ nhau.

## 1. Trạng thái model hiện tại

Registry nằm tại `configs/visual_embedding_models.json` và khóa immutable revision:

| Modality | Candidate dev | Trạng thái |
|---|---|---|
| CLIP | `openai/clip-vit-base-patch32` | Có thể smoke/dev-subset |
| SigLIP | `google/siglip-base-patch16-224` | Có thể smoke/dev-subset |
| BEiT-3 | Microsoft UniLM BEiT-3 | **BLOCKED** chờ chốt official retrieval checkpoint + checksum |

Không thay BEiT-3 bằng `microsoft/beit-*`: đó là BEiT thường, không phải BEiT-3. Không dùng
Hugging Face port bên thứ ba khi chưa có quyết định rõ. Hai candidate CLIP/SigLIP là lựa chọn
dev để đo throughput/VRAM và A/B, chưa phải kết luận model production thắng.

Logic CLIP rollback, SRRF hoặc `score_visual` thuộc Nhánh 2 online. Script trong runbook này
không đọc trạng thái modality khác và không làm fusion.

## 2. Input phải đưa lên Kaggle

Tạo một Kaggle Dataset private có cấu trúc sau (tên dataset tùy chọn):

```text
lastdance-dev-subset/
├── keyframes/
│   ├── L21_V001/<shot_id>_<local_idx>.jpg
│   ├── L21_V002/<shot_id>_<local_idx>.jpg
│   └── ...
└── index/dev-subset-5/
    ├── frames.csv
    └── frames.csv.state.json
```

Dev-subset hiện tại cần 4.164 JPEG, `frames.csv` 4.164 dòng và file state đi kèm. Không cần
đưa MP4, shot manifest, raw prediction, virtual environment hoặc TransNetV2 weight lên
Kaggle cho bước này. `frames.csv.state.json` bắt buộc vì builder từ chối catalog không có
SHA-bound complete state.

Kaggle mount `/kaggle/input` là read-only. Dùng `--keyframes-root` để đọc ảnh trực tiếp từ
input và `--embedding-root /kaggle/working/...` để ghi checkpoint/vector; không copy 397 MB
JPEG sang working nếu không cần.

## 3. Lấy code và dựng environment

Chỉ dùng commit/branch do chủ repo bàn giao rõ. Khi thay đổi hiện còn ở working tree và chưa
được chủ repo yêu cầu push thì đồng đội **chưa thể** clone phần code mới từ GitHub.

```bash
git clone https://github.com/ThanhVu165/LASTDANCE.git
cd LASTDANCE
git fetch origin
git switch <branch-duoc-ban-giao>
git log -1 --oneline
git status --short

python --version
python -m pip install -r requirements/kaggle-gpu.txt
python -m scripts.environment_doctor --profile kaggle-gpu --skip-data
python -m scripts.verify_visual_model_revisions
python -m unittest discover -s tests -v
```

Doctor phải PASS Python 3.11, package pin và CUDA/T4. `kaggle-gpu.txt` không pin Torch để
không thay wheel CUDA có sẵn của Kaggle. Nếu pip resolver muốn thay Torch, dừng lại và audit,
không tiếp tục bằng một environment khác contract.

Revision verifier gọi API thật bằng `huggingface_hub.model_info(model_id, revision=...)` và
yêu cầu repository ID + SHA resolve khớp tuyệt đối với registry; kiểm tra 40 ký tự hex trong
unit test không thay thế bước này. BEiT-3 đang blocked nên chưa được gọi API ở gate hiện tại.

Ghi output doctor và `python -m pip freeze` vào Kaggle output/batch report. Chưa đưa lock
Kaggle vào repo cho tới khi clean-install + inference thật PASS.

## 4. Smoke test input và fail-closed BEiT-3

Ví dụ mount dataset tại `/kaggle/input/lastdance-dev-subset`:

```bash
export KEYFRAMES_ROOT=/kaggle/input/lastdance-dev-subset/keyframes
export CATALOG=/kaggle/input/lastdance-dev-subset/index/dev-subset-5/frames.csv
export EMBEDDING_ROOT=/kaggle/working/visual-embeddings

python -m scripts.build_visual_embeddings \
  --modality beit3 \
  --batch-id dev-subset-5 \
  --catalog "$CATALOG" \
  --keyframes-root "$KEYFRAMES_ROOT" \
  --embedding-root "$EMBEDDING_ROOT"
```

Lệnh BEiT-3 hiện phải dừng với thông báo `blocked_model_selection`; đây là gate đúng, không
phải lỗi cần bypass.

## 5. Test checkpoint/resume thật trước khi chạy hết

Publishing Criteria yêu cầu ngắt batch rồi chạy lại thật. Unit test không thay thế gate này.
Trên dev-subset, chạy CLIP với intentional stop sau 2 shard (128 ảnh nếu batch size 64):

```python
import subprocess

base_command = [
    "python", "-m", "scripts.build_visual_embeddings",
    "--modality", "clip",
    "--batch-id", "dev-subset-5",
    "--catalog", "/kaggle/input/lastdance-dev-subset/index/dev-subset-5/frames.csv",
    "--keyframes-root", "/kaggle/input/lastdance-dev-subset/keyframes",
    "--embedding-root", "/kaggle/working/visual-embeddings",
    "--batch-size", "64",
]

stopped = subprocess.run(base_command + ["--stop-after-shards", "2"], check=False)
assert stopped.returncode == 75, stopped.returncode
```

Sau lần này phải có `checkpoint.json` với `complete=false`, `next_index=128`, có 2 thư mục
shard hoàn chỉnh và **không có** final `manifest.json`. Resume bằng process/lệnh mới, bỏ duy
nhất cờ stop:

```python
completed = subprocess.run(base_command, check=False)
assert completed.returncode == 0, completed.returncode
```

Runner scan + hash lại shard đã có, chỉ encode phần còn thiếu và không add duplicate UID.
Khi hoàn tất, `checkpoint_resume_verified=true` chỉ được suy ra vì process thứ hai resume một
intentional-interruption marker có cùng signature.

Không đổi `batch-size`, catalog, model revision hoặc danh sách video khi resume. Các giá trị
này nằm trong signature; đổi phải fail closed. Nếu cần giảm batch size do OOM, dùng batch ID
mới (ví dụ `dev-subset-5-bs32`), không ghi đè artifact cũ.

## 6. Validate CLIP độc lập

```bash
python -m scripts.validate_visual_embeddings \
  --artifact-dir /kaggle/working/visual-embeddings/dev-subset-5/clip \
  --catalog "$CATALOG" \
  --keyframes-root "$KEYFRAMES_ROOT" \
  --require-resume-verified
```

Validator kiểm tra lại signature, catalog SHA, toàn bộ UID theo đúng thứ tự catalog, hash
file, dtype `float16`, finite vector, L2 norm, dimension giữa shard và checkpoint accounting.
CLIP PASS được phép bàn giao/publish độc lập dù thư mục SigLIP/BEiT-3 chưa tồn tại.

## 7. Chạy SigLIP độc lập

Lặp đúng gate intentional-stop/resume bằng command riêng:

```bash
python -m scripts.build_visual_embeddings \
  --modality siglip \
  --batch-id dev-subset-5 \
  --catalog "$CATALOG" \
  --keyframes-root "$KEYFRAMES_ROOT" \
  --embedding-root "$EMBEDDING_ROOT" \
  --batch-size 64 \
  --stop-after-shards 2
```

Exit 75 là dự kiến; chạy lại y hệt nhưng bỏ `--stop-after-shards`, rồi validate artifact
`.../dev-subset-5/siglip` với `--require-resume-verified`. Hoàn tất SigLIP không sửa CLIP.

## 8. Chia production batch

Mỗi file worker chứa 50–100 `video_id`, mỗi dòng một ID canonical. File local này không cần
commit. Thêm vào lệnh:

```bash
--video-id-file /kaggle/working/worker-embedding-batch-01.txt
```

Một batch ID phải ánh xạ cố định tới đúng một tập video. CLIP/SigLIP có thể chạy cùng batch
ID vì namespace modality khác nhau; worker cùng modality không được nhận tập ID giao nhau.
Manifest cuối ghi count theo video và UID hash, không dùng row index làm khóa.

## 9. Artifact bàn giao và các gate còn thiếu

Mỗi modality bàn giao nguyên thư mục:

```text
visual-embeddings/<batch-id>/<modality>/
├── checkpoint.json
├── manifest.json
└── shards/
    └── 000000/{keyframe_uids.npy,vectors.npy,manifest.json}
```

Không bàn giao staging tạm, cache Hugging Face, notebook secret, model weight hoặc input JPEG
trong output vector. Không tự `git commit`, `git push`, `push_to_hub()` hay publish artifact
remote; chỉ làm khi chủ repo yêu cầu rõ. Khi được phép đồng bộ HF Dataset, gom 50–100 video
mỗi revision thay vì push từng video.

Vector shard hoàn tất mới là artifact bước 2.1, chưa phải `clip.faiss`/`siglip.faiss`/
`beit3.faiss`. FAISS `IndexIDMap` build local ở bước 3.2 sau đó và vẫn độc lập từng modality.
Lệnh build/add/validate chính xác nằm tại `docs/FAISS_INDEX_RUNBOOK.md`; modality hoàn tất
trước được build ngay, không tạo barrier đợi đủ cả ba.
`complete=true` trong manifest ở đây chỉ có nghĩa **single-modality batch đã đủ UID**; trạng
thái Publishing Criteria toàn video vẫn fail cho tới khi đủ cả 3 FAISS, mapping sanity và các
gate còn lại.

## 10. Điều phải báo sau dev-subset

- commit/branch code đã chạy;
- model ID + immutable revision;
- package/doctor output, Torch/CUDA/GPU;
- batch size, vector dimension, dtype;
- throughput, peak CUDA memory, tổng thời gian và dung lượng output;
- bằng chứng exit 75 → resume → validator PASS;
- UID count/diff và lỗi nếu có;
- BEiT-3 vẫn BLOCKED hay checkpoint nào đã được người dùng chốt.
