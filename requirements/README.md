# Dependency profiles

Target runtime is CPython 3.11.9. Do not install every profile into one environment.

| File | Purpose | Machine |
|---|---|---|
| `dev.txt` | schema/unit tests, no model | any CPU machine |
| `offline-local.txt` | inventory, TransNetV2 CPU, filtering, FAISS build | local Windows/Linux |
| `kaggle-gpu.txt` | CLIP/SigLIP/BEiT-3 batch embedding | Kaggle GPU |

Trên Windows sạch, `scripts/bootstrap_miniforge_windows.ps1` là entry point khuyến nghị:
nó cài Miniforge đã pin dưới `%LOCALAPPDATA%\LASTDANCE\toolchains`, tạo `.venv-offline`
từ `environment.yml` và chạy doctor/test. Base toolchain tránh path có khoảng trắng theo
khuyến nghị upstream. Nếu máy đã có CPython 3.11 + FFmpeg thì dùng
`scripts/bootstrap_windows.ps1`.

Bootstrap mặc định cài profile `offline-local`. Dùng `-Profile dev -EnvironmentPath
.venv-dev` cho môi trường nhẹ. Không dùng chung một environment giữa profile local CPU và
Kaggle GPU.

`ffmpeg` and `ffprobe` are system binaries, not Python packages. The environment doctor
checks both executables and their versions.

PyTorch is pinned only in the local shot profile. Kaggle supplies a CUDA-matched PyTorch;
replacing it from a generic requirements file can silently break GPU support. Run the doctor
after installation and record `torch`, CUDA and GPU information in the batch report.

`transnetv2-pytorch==1.0.5` bundles its default weight; the pipeline pins and verifies that
file's SHA-256 before model construction. `AIC_TRANSNETV2_WEIGHTS` and
`AIC_TRANSNETV2_WEIGHTS_SHA256` are only a paired external override. No weight is downloaded
while processing a video.

These files pin direct dependencies. After the first successful clean install on Windows
Python 3.11 and Kaggle, export resolved locks into `requirements/locks/` together with the
platform/Python tag. Do not reuse a Windows lock on Kaggle Linux.

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
