# Runbook Visual Embedding trên Kaggle

Đây là entry point vận hành bước 2.1: keyframe JPEG + `frames.csv` → vector L2-normalized
`float16` riêng cho từng `keyframe_uid`. CLIP, SigLIP và EVA-CLIP chạy độc lập, có namespace
checkpoint/artifact riêng và không có barrier đồng bộ thời gian hoặc thứ tự.

## 1. Registry model và trạng thái gate

Registry duy nhất nằm tại `configs/visual_embedding_models.json`.

| Modality | Model ID | Immutable revision | Trạng thái |
|---|---|---|---|
| `clip` | `openai/clip-vit-base-patch32` | `4c4a3e8bcc2b768a8b89fc83ed8c828345ca3bac` | Dev-subset T4 PASS |
| `siglip` | `google/siglip-base-patch16-224` | `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` | Dev-subset T4 PASS |
| `eva_clip` | `timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k` | `bf4190eb65dd5204ffb03e980108beb1200e0873` | Chờ dev-subset T4 gate |

BEiT-3 đã bị loại vĩnh viễn. Registry chỉ giữ row `blocked_model_selection` để ghi nhận
quyết định cuối; không mở lại audit/checksum/conversion/adapter và không gọi
`--modality beit3`.

EVA-CLIP chỉ được load từ `open_clip_model.safetensors` (855.584.096 byte), SHA-256
`00af04296f09f24dcc69559440a80b7a44daf4855a72827e016067e6e571b851`. Adapter tải config
và file này ở đúng revision, kiểm tên/suffix/size/SHA-256 **trước** model construction, tạo
architecture với `load_weights=False`, rồi gọi safetensors loader tường minh. Không có
fallback sang `open_clip_pytorch_model.bin` pickle.

## 2. Input dev-subset-5

Dataset private phải cung cấp đúng catalog đã publish và JPEG tương ứng:

```text
<dataset-root>/
├── keyframes/<video_id>/<shot_id>_<local_idx>.jpg
└── catalog/
    ├── frames.csv
    └── frames.csv.state.json
```

Dev-subset có 5 video, 4.164 keyframe. Tên thư mục mount trên Kaggle có thể thay đổi; resolve
path thật dưới `/kaggle/input` và assert catalog/keyframe tồn tại trước khi chạy. Không đưa
MP4, model cache, token, virtual environment hoặc artifact production vào input này.

Kaggle `/kaggle/input` là read-only. Ghi checkpoint vào
`/kaggle/working/visual-embeddings`; không copy toàn bộ JPEG sang working.

## 3. Environment Kaggle GPU

- Accelerator: Tesla T4.
- Internet: On.
- Python: 3.12.x.
- Cài `requirements/kaggle-gpu.txt`; không cài đè Torch/Torchvision CUDA có sẵn.
- Nếu vừa cài package làm kernel hiện tại giữ ABI cũ, restart session hoặc chạy preflight
  trong fresh subprocess trước inference.

```bash
python -m pip install -r requirements/kaggle-gpu.txt
python -m scripts.environment_doctor --profile kaggle-gpu --skip-data
python -m scripts.verify_visual_model_revisions
python -m unittest discover -s tests -q
```

Doctor phải PASS CUDA/T4 và thêm `open-clip-torch==3.3.0`, `timm==1.0.28`. Verifier phải có
dòng bắt đầu bằng:

```text
PASS: eva_clip timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k revision=bf4190e...
```

Dòng EVA còn in tên safetensors và SHA-256 đã resolve từ HF metadata. Chỉ kiểm SHA dạng 40
ký tự trong unit test không thay thế API verifier thật.

## 4. Gate bắt buộc EVA-CLIP trên dev-subset-5

Không có production runner EVA trước khi toàn bộ mục này PASS. Đặt biến theo mount thực tế:

```bash
export KEYFRAMES_ROOT=/kaggle/input/<dataset>/keyframes
export CATALOG=/kaggle/input/<dataset>/catalog/frames.csv
export EMBEDDING_ROOT=/kaggle/working/visual-embeddings
```

EVA02-L lớn hơn CLIP/SigLIP base. Bắt đầu dev gate với batch size 32; đây là một phần của
signature và không được đổi khi resume.

Lệnh khuyến nghị chạy trọn gate bằng hai subprocess thật, tự kiểm checkpoint rồi validate:

```bash
python -m scripts.run_eva_clip_dev_gate \
  --catalog "$CATALOG" \
  --keyframes-root "$KEYFRAMES_ROOT" \
  --embedding-root "$EMBEDDING_ROOT" \
  --video-id-file /kaggle/working/worker-dev-subset-5.txt \
  --batch-size 32
```

Runner từ chối ghi đè nếu artifact đã tồn tại và không archive/upload gì. Các tiểu mục dưới
mô tả chính xác những gì runner thực hiện để review hoặc chạy thủ công khi cần debug.

### 4.1 Intentional interruption

```python
import os
import subprocess
import sys

CATALOG = os.environ["CATALOG"]
KEYFRAMES_ROOT = os.environ["KEYFRAMES_ROOT"]
EMBEDDING_ROOT = os.environ["EMBEDDING_ROOT"]

base = [
    sys.executable, "-m", "scripts.build_visual_embeddings",
    "--modality", "eva_clip",
    "--batch-id", "dev-subset-5",
    "--catalog", CATALOG,
    "--keyframes-root", KEYFRAMES_ROOT,
    "--embedding-root", EMBEDDING_ROOT,
    "--video-id-file", "/kaggle/working/worker-dev-subset-5.txt",
    "--batch-size", "32",
]

stopped = subprocess.run(base + ["--stop-after-shards", "2"])
assert stopped.returncode == 75, stopped.returncode
```

Sau lần chạy này phải có `checkpoint.json` với `complete=false`, `next_index=64`, hai shard
hoàn chỉnh và chưa có `manifest.json`.

### 4.2 Resume bằng process mới

```python
completed = subprocess.run(base)
assert completed.returncode == 0, completed.returncode
```

Process thứ hai phải scan/hash shard cũ, chỉ encode phần còn thiếu và ghi
`checkpoint_resume_verified=true`. Không đổi catalog, worker list, revision, batch ID hoặc
batch size. Nếu OOM ở batch 32, giữ artifact để điều tra và chạy **batch ID mới** với batch
16; không ghi đè/resume chéo signature.

### 4.3 Validate và xác nhận dimension runtime

```python
import json
from pathlib import Path

artifact = Path(EMBEDDING_ROOT) / "dev-subset-5" / "eva_clip"
validated = subprocess.run([
    sys.executable, "-m", "scripts.validate_visual_embeddings",
    "--artifact-dir", str(artifact),
    "--catalog", CATALOG,
    "--keyframes-root", KEYFRAMES_ROOT,
    "--require-resume-verified",
])
assert validated.returncode == 0, validated.returncode

manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
assert manifest["record_count"] == 4164
assert manifest["vector_dim"] == 768
assert manifest["vector_dtype"] == "float16"
assert manifest["checkpoint_resume_verified"] is True
assert manifest["runtime"]["device"] == "cuda"
assert manifest["runtime"]["gpu_name"] == "Tesla T4"
assert manifest["model"]["revision"] == (
    "bf4190eb65dd5204ffb03e980108beb1200e0873"
)
print("EVA-CLIP DEV GATE PASS", manifest["runtime"])
```

`expected_vector_dim=768` xuất phát từ official OpenCLIP config, nhưng chỉ output runtime +
validator ở trên mới đóng gate. Validator đồng thời kiểm UID, shard hash, finite, norm L2,
dtype và checkpoint accounting.

## 5. Gate CLIP/SigLIP và tính độc lập UID

CLIP/SigLIP dùng cùng pattern, chỉ đổi `--modality` và artifact directory. Một modality PASS
không chờ hai modality còn lại. Tất cả shard lấy `keyframe_uid` trực tiếp từ cùng
`frames.csv`; không dùng row index, nên ba tài khoản Kaggle có thể chạy khác lúc và khác thứ
tự mà vẫn join chính xác trong `IndexIDMap`.

Không mang checkpoint dở dang sang Python/Torch/OpenCLIP runtime khác. Nếu session mới có
runtime khác, giữ output cũ để audit và chạy batch ID/output mới.

## 6. Artifact dev bàn giao

```text
visual-embeddings/dev-subset-5/eva_clip/
├── checkpoint.json
├── manifest.json
└── shards/
    └── 000000/
        ├── keyframe_uids.npy
        ├── vectors.npy
        └── manifest.json
```

Không đưa input JPEG, MP4, Hugging Face cache, token hoặc file `.bin` vào archive. Dev report
phải ghi commit, model ID/revision, safetensors SHA, package freeze, Python/Torch/CUDA/GPU,
batch size, dimension, dtype, throughput, peak CUDA memory, tổng thời gian, dung lượng,
exit 75 → resume và validator PASS.

## 7. Sau gate dev

Chỉ khi EVA-CLIP dev-subset-5 PASS đầy đủ mới được tạo notebook production 9 batch theo
khung interrupt → resume → validate → archive → upload HF. Remote namespace bắt buộc là
`eva_clip/archives/{batch_id}/...`; không được đọc/ghi đè `clip/...` hoặc `siglip/...`.

Sau khi tải embedding về local, FAISS CPU build độc lập thành `eva_clip.faiss` theo
`docs/FAISS_INDEX_RUNBOOK.md`. `complete=true` của một embedding batch chưa làm video
Publishing Ready; vẫn phải đủ `clip.faiss`, `siglip.faiss`, `eva_clip.faiss` với UID khớp
100%, finite/norm, mapping sanity và checkpoint/resume.
