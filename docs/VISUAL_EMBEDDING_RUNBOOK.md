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
path thật dưới `/kaggle/input` và assert catalog/keyframe tồn tại trước khi chạy. Với layout
keyframe-ready ở trên, không đưa MP4, model cache, token, virtual environment hoặc artifact
embedding production vào input.

Notebook EVA dev-gate dùng trực tiếp private Kaggle Dataset **Eva test dataset**
(`minhlight0204/eva-test-dataset`), thường được mount tại
`/kaggle/input/eva-test-dataset`:

```text
/kaggle/input/eva-test-dataset/
├── frames.csv
├── frames.csv.state.json
├── L21_V001.mp4
├── L21_V002.mp4
├── L21_V003.mp4
├── L21_V005.mp4
└── L21_V006.mp4
```

Notebook validate catalog hash-bound gốc, lọc đúng 4.164 record của năm video, atomic-publish
catalog dev dưới `/kaggle/working`, rồi dùng FFmpeg decode tuần tự mỗi MP4 đúng một lần để
trích JPEG. Nó không chạy lại inventory/shot/keyframe plan/quality, không quét production
archive và không sửa Dataset input. Nếu Kaggle dùng path lồng khác, notebook tự dò đúng một
thư mục chứa cặp catalog + chính xác năm MP4; biến `EVA_TEST_DATASET_ROOT` vẫn cho phép override.
Lệnh trích dùng `-vsync vfr` để tương thích FFmpeg cũ trong Kaggle image; tập frame vẫn được
chọn theo decoded-frame index từ `frames.csv`.

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

**PASS trên Kaggle Tesla T4 ngày 26/08/2026.** Gate đã xử lý đủ 4.164/4.164 keyframe,
dimension 768, vector `float16`, intentional exit 75 → process mới resume, finite/L2/UID
validator PASS và peak CUDA 2.719.088.640 byte. Bằng chứng immutable đã upload:

```text
HF repo: MinhThuw0103/lastdance-visual-embeddings
HF commit: e5569715d708fb29e5a0c3036cfdaf086fdd642a
archive: eva_clip/dev-gate/dev-subset-5/lastdance-eva-clip-dev-gate-63cdf24.tar.gz
sha256: 0a0949293c78a3c902e4174418e22c23e9cf853350d80f274b59aea5512a89b4
```

Các bước dưới đây giữ làm contract tái kiểm tra/debug. Đặt biến theo mount thực tế:

Notebook Kaggle dev-gate sẵn để upload/chạy bằng **Run All** nằm tại
`notebooks/kaggle_eva_clip_dev_gate.ipynb`. Notebook pin đúng commit chứa runner, đọc trực
tiếp dataset `eva-test-dataset`, trích keyframe một lượt/video rồi gọi
`scripts.run_eva_clip_dev_gate`; mọi output được ghi dưới `/kaggle/working`, không nằm trong
Git repo. Cell upload bằng chứng lên namespace HF `eva_clip/dev-gate/dev-subset-5/` là tùy
chọn và mặc định tắt. Có thể override mount bằng biến `EVA_TEST_DATASET_ROOT` nếu Kaggle đổi
slug/path.

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

**Production và local handoff đã PASS ngày 27/08/2026.** Cả ba modality hoàn tất 9/9 batch,
873 video và 293.336 record. Snapshot HF cuối dùng để handoff local:

```text
HF repo: MinhThuw0103/lastdance-visual-embeddings
HF final commit: 938aefd437ab8db61fc6599d613aedcf4921d71e
catalog sha256: ee9693e75580527a0a257e9ba003984e105b059b716922c03c7a0b72b1508a37
```

Local đã xác minh checksum archive, khôi phục đúng 293.336 JPEG/873 video với 0 file rỗng,
rồi build/validate tuần tự ba `IndexIDMap(IndexFlatIP)`:

| Index | Records | Dim | Index SHA-256 |
|---|---:|---:|---|
| `clip.faiss` | 293.336 | 512 | `22d8851f0b27c21229bd686813c0c2a58cc1bd17df1be01b865bb5ea2e1d33e5` |
| `siglip.faiss` | 293.336 | 768 | `9d869f1d7ddd9c561a8c5bb91be50b0d69d9e85fb34480e7a70b40537ecfe06e` |
| `eva_clip.faiss` | 293.336 | 768 | `b4c80e6e12ae4a5513edd183126b1b11675261d9d58819ba17e3c65e30ab9901` |

Cả ba có cùng UID-set SHA-256
`5bada00bd4a93928e48af3a6cbe7189a3b465eafb00cc8f829941edee536e660`;
validator đã diff UID 100% với `frames.csv`, reconstruct vector để kiểm finite/L2 norm, kiểm
9 source batch và yêu cầu `checkpoint_resume_verified=true`. Artifact runtime nằm dưới
`AIC_DATA`, không commit JPEG, vector shard, checkpoint, manifest gate hoặc file FAISS vào Git.

Notebook production đã được tạo tại `notebooks/kaggle_eva_clip_production.ipynb` sau khi gate
PASS. Input bắt buộc là production keyframe Dataset có full hash-bound `frames.csv` và chín
thư mục keyframe trong `production-batch-mapping.json`; **không dùng** `Eva test dataset` năm
MP4 cho production. Notebook pin code commit `63cdf244449e3eb8bcffd8d47a417ef2a07d7927`,
batch size 32 và trước inference phải xác minh checksum bằng chứng gate trên HF.

Notebook chạy đủ chín batch theo khung restore remote → intentional exit 75 → process mới
resume → validate → archive → upload/verify checksum. Remote namespace bắt buộc là
`eva_clip/archives/{batch_id}/...`; guard cấm đọc/ghi đè `clip/...` hoặc `siglip/...`. Batch
đã có archive + checksum đúng trên HF sẽ được restore/validate và không encode lại. Mapping
và chín worker list được lấy từ production input nếu có, nếu không thì restore từ
`production-workers/` trên HF; control file mới chỉ được upload sau khi mapping đủ 873 video,
293.336 keyframe và catalog SHA khớp. Resolver chỉ kiểm các path control cố định ở root,
`workers/` và `production-workers/`; tuyệt đối không recursive-scan cây 293.336 JPEG.

Sau khi tải embedding về local, FAISS CPU build độc lập thành `eva_clip.faiss` theo
`docs/FAISS_INDEX_RUNBOOK.md`. `complete=true` của một embedding batch chưa làm video
Publishing Ready; vẫn phải đủ `clip.faiss`, `siglip.faiss`, `eva_clip.faiss` với UID khớp
100%, finite/norm, mapping sanity và checkpoint/resume.
