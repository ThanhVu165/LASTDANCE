# Môi trường tái lập cho LASTDANCE frame-level

Tài liệu này áp dụng cho Nhánh 1. Target chuẩn là CPython 3.11.9. Backend window-first cũ
có environment riêng và không được trộn dependency vào `.venv-offline`.

## 1. Các profile dependency

| Profile | File | Mục đích |
|---|---|---|
| Dev | `requirements/dev.txt` | schema và unit test, không tải model |
| Offline local | `requirements/offline-local.txt` | inventory, shot detection CPU, filter, FAISS |
| Shot Windows GPU | `requirements/shot-windows-gpu.txt` | TransNetV2 CUDA 12.6, Torch 2.12.1 |
| Shot Colab GPU | `requirements/shot-colab-gpu.txt` | TransNetV2 CUDA, giữ Torch có sẵn của Colab |
| Kaggle GPU | `requirements/kaggle-gpu.txt` | CLIP/SigLIP/BEiT-3 batch embedding |

Không cài profile Kaggle lên máy local chỉ để chạy preprocessing. Kaggle đã có PyTorch khớp
CUDA; không được để requirements chung tự thay wheel PyTorch của notebook.

## 2. Windows - cách khuyến nghị (repo-local)

Máy Windows mới chỉ cần PowerShell và Internet trong lần bootstrap đầu. Script sau tải
đúng Miniforge installer đã pin, kiểm tra SHA-256, cài base toolchain vào
`%LOCALAPPDATA%\LASTDANCE\toolchains`, tạo environment dự án `.venv-offline` từ
`environment.yml`, rồi chạy doctor/compile/test. Nó không đăng ký Python và không sửa
`PATH` hệ thống:

```powershell
cd C:\LASTDANCE
.\scripts\bootstrap_miniforge_windows.ps1
```

Mặc định là profile `offline-local` đầy đủ. Contributor chỉ sửa schema/test có thể tạo
profile nhẹ (không tải PyTorch/TransNetV2/FAISS):

```powershell
.\scripts\bootstrap_miniforge_windows.ps1 -Profile dev -EnvironmentPath .venv-dev
```

`.tools/` (installer cache) và `.venv-offline/` đều bị Git ignore. Installer được khóa cả
version lẫn checksum; nếu checksum lệch, script dừng thay vì tiếp tục. Miniforge base đặt
ngoài repo vì upstream có lỗi đã biết với install path chứa khoảng trắng/ký tự đặc biệt,
trong khi đường dẫn repo hiện tại có khoảng trắng. Có thể override bằng `-ToolchainRoot`,
nhưng path phải là ASCII và không có khoảng trắng. FFmpeg đến từ Conda nên `ffmpeg` và
`ffprobe` có cùng version giữa các máy Windows.

Sau khi bootstrap, đặt biến dữ liệu/model cho từng máy rồi chạy profile đầy đủ:

```powershell
$env:AIC_DATA = "D:\AIC2026\data"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "offline-local")
```

Runner tự đưa FFmpeg/ffprobe của environment vào child `PATH` và đặt absolute
`AIC_FFMPEG`/`AIC_FFPROBE` nếu chưa có. Nó không sửa `PATH` user/system và không cần biết
Miniforge base nằm ở đâu.

### Dùng Python/FFmpeg đã cài sẵn

Cài trước Python 3.11 x64 và FFmpeg. Xác nhận `ffmpeg -version` và `ffprobe -version` chạy
được trong PowerShell mới, sau đó:

```powershell
cd C:\LASTDANCE
Copy-Item .env.example .env
# Chỉ dùng .env làm mẫu; đặt biến thật trong PowerShell hoặc secret manager, không commit.
$env:AIC_DATA = "D:\AIC2026\data"
.\scripts\bootstrap_windows.ps1
```

`bootstrap_windows.ps1` chỉ tạo `.venv-offline` từ Python có sẵn, cài dependency đã pin,
chạy doctor, compile và unit test. Script không tự cài FFmpeg, không tải weight và không sửa
dataset.

Nếu dùng Conda/Mamba:

```powershell
conda env create -f environment.yml
conda activate lastdance-offline
python -m scripts.environment_doctor --profile offline-local
```

## 3. FFmpeg và path

`ffmpeg`/`ffprobe` là binary hệ thống. Có thể đặt absolute path cục bộ trong biến
`AIC_FFMPEG`/`AIC_FFPROBE`, nhưng artifact JSON/CSV chỉ lưu path tương đối dưới `AIC_DATA`.
Không copy path của máy A sang `frames.csv` rồi dùng trên máy B.

## 4. TransNetV2

TensorFlow inference upstream cũ yêu cầu TensorFlow 2.1 nên không nằm trong environment
Python 3.11 chính. Shot detector production đã chốt dùng port PyTorch
`transnetv2-pytorch==1.0.5`; không tiếp tục chờ hoặc A/B AutoShot trên critical path. CUDA
vẫn phải qua parity 5/5 với reference CPU trước khi chạy batch production.

`transnetv2-pytorch==1.0.5` bundle weight trong wheel. Pipeline không tải thêm weight lúc
xử lý video và kiểm tra weight này bằng SHA-256 cố định
`a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de` trước khi khởi
tạo model. Package version, threshold, device, source và weight SHA phải xuất hiện trong
shot manifest.

CPU là reference/default. Windows hoặc Colab CUDA chỉ được bật tường minh bằng
`--device cuda` hoặc `AIC_TRANSNETV2_DEVICE=cuda`; nếu CUDA không sẵn sàng pipeline dừng,
không fallback về CPU.
Trước khi chia production batch, phải so từng boundary và excluded range của cả 5 video
dev-subset với reference CPU theo runbook.

Quy trình checkout, chạy một video, chia batch và bàn giao manifest nằm trong
`docs/SHOT_DETECTION_RUNBOOK.md`.

### Windows NVIDIA GPU

Không cài CUDA wheel vào `.venv-offline`. Dùng profile tách biệt:

```powershell
.\scripts\check_nvidia_windows.ps1
.\scripts\bootstrap_miniforge_windows.ps1 -Profile shot-windows-gpu
```

Bootstrap tự chọn `.venv-shot-gpu`, cài wheel chính thức `torch==2.12.1+cu126` và chạy
doctor/test. Driver NVIDIA tối thiểu là 528.33 cho CUDA 12.x minor compatibility; nên cập
nhật driver mới rồi reboot trước khi bootstrap. CUDA Toolkit hệ thống không bắt buộc vì
wheel PyTorch mang CUDA runtime cần thiết.

Nguồn version: [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)
và [NVIDIA CUDA 12.6 compatibility](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html).

Chỉ khi cần thử external weight override:

1. tải/chia sẻ qua kênh nội bộ hợp lệ;
2. tính SHA-256 trên máy nguồn;
3. đặt `AIC_TRANSNETV2_WEIGHTS` và `AIC_TRANSNETV2_WEIGHTS_SHA256` trên máy nhận;
4. chạy environment doctor;
5. không cho pipeline tự download weight khi đang xử lý video.

## 5. Kaggle

Trong notebook:

```bash
python --version
pip install -r requirements/kaggle-gpu.txt
python -m scripts.environment_doctor --profile kaggle-gpu
```

Ghi vào batch report: Python, package version, `torch.__version__`, CUDA, GPU, model ID,
revision, dtype, dimension, pixel/config signature và peak VRAM. Không push vector nếu chưa
ép `float16`.

Lệnh build/intentional interruption/resume/validate và cấu trúc Kaggle Dataset bắt buộc nằm
trong `VISUAL_EMBEDDING_RUNBOOK.md`. Kaggle input là read-only nên tách `--keyframes-root`
khỏi `--embedding-root` writable; không hardcode mount path vào artifact.

Shot Detection trên Colab dùng profile và doctor riêng, không dùng profile embedding:

```bash
python -m pip install -r requirements/shot-colab-gpu.txt
python -m scripts.environment_doctor --profile shot-colab-gpu
```

Các cell clone repo, parity gate và batch command đầy đủ nằm trong
`docs/SHOT_DETECTION_RUNBOOK.md`.

## 6. Khóa môi trường theo platform

Các requirements hiện pin dependency trực tiếp. Sau lần clean-install thành công đầu tiên,
export full resolved dependency tree riêng cho từng platform. Lock Windows đã được tạo từ
clean-install 24/08/2026 trong `requirements/locks/`; khi dependency trực tiếp đổi, phải tạo
lại lock rồi chạy doctor/compile/test:

```powershell
python -m pip freeze | Set-Content requirements\locks\windows-py311-cpu.txt
```

Kaggle xuất lock riêng `kaggle-linux-py311.txt`. Không dùng lock Windows để cài Kaggle hoặc
ngược lại. Mọi lock chỉ được commit sau khi doctor, compile và test pass trên máy sạch.

## 7. Điều kiện trước khi chạy subset thật

- doctor profile `offline-local` pass toàn bộ;
- `AIC_DATA/videos` tồn tại và inventory count được xác nhận;
- bundled TransNet weight checksum đúng, hoặc external override có checksum đúng;
- không có backend/model GPU khác đang chạy;
- output vẫn `complete=false` cho tới khi đủ Publishing Criteria.
