# Dependency profiles

Local/Shot target runtime is CPython 3.11.9. Kaggle Visual target runtime is CPython
3.12.x from the validated Kaggle image. Do not install every profile into one environment.

| File | Purpose | Machine |
|---|---|---|
| `dev.txt` | schema/unit tests, no model | any CPU machine |
| `ocr-v2-artifacts.txt` | sync/build/validate OCR v2 đã nhận dạng; không model/CUDA | local CPU Python 3.11 |
| `offline-local.txt` | inventory, TransNetV2 CPU, filtering, FAISS build, HF artifact sync | local Windows/Linux |
| `ocr-api.txt` | Công cụ Gemini legacy; OCR v2 vẫn cần residual count/cost và duyệt canary riêng | orchestration CPU hoặc Kaggle |
| `ocr-kaggle-gpu.txt` | Profile CRAFT + EasyOCR legacy, không phải setup VietOCR/Paddle production v2 | Kaggle GPU OCR legacy |
| `shot-windows-gpu.txt` | TransNetV2 CUDA 12.6, Torch 2.12.1 | Windows NVIDIA GPU |
| `shot-colab-gpu.txt` | TransNetV2 CUDA shot detection, giữ Torch có sẵn | Google Colab T4 |
| `kaggle-gpu.txt` | CLIP/SigLIP/EVA-CLIP batch embedding | Kaggle GPU |

OCR v2 theo [BASELINE_SPEC.md](../docs/BASELINE_SPEC.md) §2.2, checklist ở
[OCR_V2_PRODUCTION_PLAN.md](../docs/OCR_V2_PRODUCTION_PLAN.md): VietOCR `vgg_seq2seq` +
Paddle `latin_PP-OCRv5_mobile_rec`, dùng lại CRAFT bbox cache. Environment trial/Gate B
đã có notebook riêng. Recognition đã hoàn tất chín batch trên bốn T4/HF; bước sync/union
SQLite local dùng profile nhẹ `ocr-v2-artifacts` hoặc môi trường `offline-local` đã có, xem
[production runbook](../docs/OCR_V2_PRODUCTION_RUNBOOK.md) và
[snapshot runbook](../docs/OCR_V2_SNAPSHOT_RUNBOOK.md). Không cài
profile EasyOCR cũ rồi coi là môi trường v2, không chạy model OCR trên máy Codex.
Khi triển khai phải bảo vệ Torch/Torchvision + NVIDIA/NCCL sẵn có và kiểm tra GPU trước
inference; không tự giải quyết dependency bằng cách cài đè bộ CUDA của Kaggle.

Kaggle visual embedding phải theo `docs/VISUAL_EMBEDDING_RUNBOOK.md`. CLIP/SigLIP/EVA-CLIP
đều đã qua dev gate và production 9/9 batch; EVA-CLIP dùng `open-clip-torch==3.3.0`,
`timm==1.0.28` cùng checkpoint `.safetensors` đã pin.
Sau khi tải embedding artifact về, FAISS CPU dùng cùng profile `offline-local` và chạy theo
`docs/FAISS_INDEX_RUNBOOK.md`; không cài FAISS vào environment Kaggle chỉ để build local.

Trên Windows sạch, `scripts/bootstrap_miniforge_windows.ps1` là entry point khuyến nghị:
nó cài Miniforge đã pin dưới `%LOCALAPPDATA%\LASTDANCE\toolchains`, tạo `.venv-offline`
từ `environment.yml` và chạy doctor/test. Base toolchain tránh path có khoảng trắng theo
khuyến nghị upstream. Nếu máy đã có CPython 3.11 + FFmpeg thì dùng
`scripts/bootstrap_windows.ps1`.

Bootstrap mặc định cài profile `offline-local`. Dùng `-Profile dev -EnvironmentPath
.venv-dev` cho môi trường nhẹ. Worker Shot Detection Windows GPU dùng `-Profile
shot-windows-gpu`; bootstrap tự tạo `.venv-shot-gpu`. Không dùng chung environment CPU,
Windows GPU và Kaggle GPU.

`ffmpeg` and `ffprobe` are system binaries, not Python packages. The environment doctor
checks both executables and their versions.

PyTorch is pinned only in the local shot profile. Kaggle supplies a CUDA-matched PyTorch;
replacing it from a generic requirements file can silently break GPU support. Run the doctor
after installation and record `torch`, CUDA and GPU information in the batch report.
Colab Shot Detection phải dùng `shot-colab-gpu.txt`, chọn `--device cuda` tường minh và qua
parity gate trong `docs/SHOT_DETECTION_RUNBOOK.md` trước khi chạy batch thật.
Windows GPU dùng wheel chính thức `torch==2.12.1+cu126` từ PyTorch CUDA 12.6 index. Không
cần cài CUDA Toolkit hệ thống, nhưng NVIDIA driver phải hỗ trợ CUDA 12.x; preflight yêu cầu
driver tối thiểu 528.33 rồi doctor vẫn kiểm tra `torch.cuda.is_available()`.

`transnetv2-pytorch==1.0.5` bundles its default weight; the pipeline pins and verifies that
file's SHA-256 before model construction. `AIC_TRANSNETV2_WEIGHTS` and
`AIC_TRANSNETV2_WEIGHTS_SHA256` are only a paired external override. No weight is downloaded
while processing a video.

These files pin direct dependencies. After the first successful clean install on Windows
Python 3.11 and Kaggle Python 3.12, export resolved locks into `requirements/locks/` together
with the platform/Python tag. Do not reuse a Windows lock on Kaggle Linux.

The checked-in Windows lock is split because Conda owns Python/FFmpeg/native libraries and
pip owns the Python application packages:

```powershell
conda create --prefix .venv-offline `
  --file requirements\locks\windows-py311-conda-explicit.txt
conda run --prefix .venv-offline python -m pip install `
  -r requirements\locks\windows-py311-pip.txt
```

Use `environment.yml` for normal setup. The explicit lock is a reproducibility/rollback
path tied to `win-64`; regenerate both lock files after changing any direct dependency.
