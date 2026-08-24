# Runbook Shot Detection bằng TransNetV2

Tài liệu này dành cho thành viên chạy bước **1.1 Shot Detection** của Nhánh 1. Worker
production ưu tiên Windows NVIDIA GPU; CPU giữ làm reference/fallback và Colab CUDA là lựa
chọn phụ. Luôn chạy code từ repository, không copy hoặc viết lại
`predictions_to_scenes()` trong notebook/script riêng.

## 1. Phạm vi và contract

- Input: MP4 tại `AIC_DATA/videos/<video_id>.mp4`.
- Output: shot manifest schema v2 tại `AIC_DATA/shots/<video_id>.json`.
- CPU là reference. Windows/Colab CUDA chỉ dùng production sau parity gate đủ 5 video.
- Model: `transnetv2-pytorch==1.0.5`.
- Threshold: `0.5`, so sánh strict `prediction > threshold`.
- Tham số nội bộ package: input `48x27`, `window_size=100`, `step_size=50`, overlap 50,
  padding đầu 25 frame và lấy phần prediction `[25:75]` của mỗi window.
- Weight SHA-256:
  `a313d0b3bebfa9a71914b375bfdf918a30b5c3b1e6be51972d35dd8078b442de`.
- Frame transition vượt threshold nằm ngoài shot và được ghi trong
  `excluded_transition_ranges` với reason `transition_score_above_threshold`.
- Output bước này chưa có `complete=true`; còn keyframe, embedding và Publishing Criteria.

Đọc trước khi sửa code: `AGENTS.md`, mục 1.1/1.1a của
`docs/OFFLINE_INDEXING_SPEC.md` và file này.

## 2. Lấy đúng code

Nếu branch chưa tồn tại ở máy local:

```powershell
git fetch origin
git switch --track origin/codex/offline-shot-detection
```

Nếu branch đã tồn tại:

```powershell
git switch codex/offline-shot-detection
git pull --ff-only
```

Kiểm tra và ghi lại commit:

```powershell
git status --short --branch
git rev-parse HEAD
```

Hai worker chỉ được gộp artifact khi chạy cùng commit. Nếu Pull Request đã merge, dùng
`main` mới nhất nhưng vẫn phải ghi chính xác commit SHA.

## 3. Dựng môi trường Windows NVIDIA GPU

Từ root repository, kiểm tra GPU/driver trước:

```powershell
.\scripts\check_nvidia_windows.ps1
```

Driver phải từ 528.33 trở lên; nên cập nhật driver NVIDIA mới và reboot. Không cần cài CUDA
Toolkit hệ thống. Sau khi preflight PASS:

```powershell
.\scripts\bootstrap_miniforge_windows.ps1 -Profile shot-windows-gpu
```

Script tự tạo environment riêng `.venv-shot-gpu`, cài Python 3.11.9, FFmpeg/ffprobe 7.1.1,
wheel chính thức `torch==2.12.1+cu126`, `transnetv2-pytorch==1.0.5`, kiểm tra CUDA/weight rồi
chạy compile và toàn bộ unit test. Không đè hoặc dùng lại `.venv-offline` CPU.

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_miniforge_windows.ps1 -Profile shot-windows-gpu
```

## 4. Chuẩn bị dữ liệu

`AIC_DATA` trỏ tới thư mục cha chứa `videos/`, không trỏ thẳng vào `videos/`:

```text
D:\AIC2026\
├── videos\
│   ├── L21_V001.mp4
│   └── ...
├── shots\
├── keyframes\
└── index\
```

Windows:

```powershell
$env:AIC_DATA = "D:\AIC2026"
```

Sai: `$env:AIC_DATA = "D:\AIC2026\videos"`.

## 5. Kiểm tra môi trường Windows GPU

```powershell
.\scripts\run_offline_windows.ps1 `
  -EnvironmentPath ".venv-shot-gpu" `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "shot-windows-gpu")
```

Chỉ tiếp tục khi Python, FFmpeg/ffprobe, Torch, TransNetV2, CUDA/GPU và weight SHA-256 đều
`PASS`. Nếu CUDA fail, dừng; không chạy cùng lệnh bằng CPU rồi trộn kết quả.

## 6. Parity Windows GPU bắt buộc

Đồng đội cần có 5 manifest CPU reference trong một thư mục riêng, không đặt chung với output
GPU. Chạy đủ 5 MP4 bằng CUDA vào namespace parity:

```powershell
.\scripts\run_offline_windows.ps1 `
  -EnvironmentPath ".venv-shot-gpu" `
  -Module scripts.run_shot_batch `
  -PythonArguments @(
    ".\configs\shot_parity_dev_subset_5.txt",
    "--device", "cuda",
    "--shots-dir", "$env:AIC_DATA\index\shot-parity-windows-gpu"
  )
```

Reference trên máy chuẩn:

| Video | Total frame | Shot | Excluded ranges |
|---|---:|---:|---|
| `L21_V001` | 37.849 | 336 | Không có |
| `L21_V002` | 31.720 | 281 | `[49,49]` |
| `L21_V003` | 29.946 | 255 | Không có |
| `L21_V005` | 28.294 | 248 | `[57,57]` |
| `L21_V006` | 31.064 | 268 | `[0,0]`, `[1063,1063]` |

So sánh từng manifest. Sửa `$cpuReferenceDir` thành nơi đồng đội lưu 5 JSON CPU:

```powershell
$cpuReferenceDir = "D:\AIC2026\cpu-reference"
$gpuParityDir = "$env:AIC_DATA\index\shot-parity-windows-gpu"
$parityIds = Get-Content .\configs\shot_parity_dev_subset_5.txt

foreach ($videoId in $parityIds) {
    .\scripts\run_offline_windows.ps1 `
      -EnvironmentPath ".venv-shot-gpu" `
      -Module scripts.compare_shot_manifests `
      -PythonArguments @(
        "$cpuReferenceDir\$videoId.json",
        "$gpuParityDir\$videoId.json"
      )
}
```

Cả 5 phải `PASS` đúng toàn bộ boundary/range, không chỉ cùng số shot. Lệch một frame thì
dừng, giữ report để điều tra và không treo production batch.

## 7. Chạy batch được phân công

Tạo `worker-01.txt`, mỗi dòng đúng một `video_id`, không khoảng trắng và không trùng:

```text
L21_V007
L21_V008
L21_V009
```

Windows NVIDIA GPU:

```powershell
.\scripts\run_offline_windows.ps1 `
  -EnvironmentPath ".venv-shot-gpu" `
  -Module scripts.run_shot_batch `
  -PythonArguments @(".\worker-01.txt", "--device", "cuda")
```

Batch runner dùng một model cho cả danh sách, kiểm tra manifest cũ trước khi skip, ghi từng
manifest atomic và tiếp tục video kế tiếp nếu một video lỗi. `--fail-fast` dùng khi muốn dừng
ngay; `--overwrite` chỉ dùng khi chủ động chạy lại artifact cũ. Chạy lại đúng lệnh sau khi
mất điện/restart sẽ skip manifest GPU tương thích và tiếp tục phần còn thiếu.

Mỗi worker nhận tập ID không giao nhau. Không cho hai worker ghi cùng manifest.

## 8. Output và bàn giao

Manifest:

```text
AIC_DATA/shots/<video_id>.json
```

Batch report tự sinh tại:

```text
AIC_DATA/index/shot-batches/<tên-file-list>.json
```

Report ghi commit, runtime Python/Torch/CUDA/FFmpeg, detector signature, ID thành công/skip
và lỗi. Bàn giao các manifest cùng batch report. Không gửi/commit MP4, environment, weight,
cache, raw prediction, log, JPEG hoặc vector.

Manifest trong `index/shot-parity-windows-gpu/` chỉ dùng kiểm tra; không bàn giao thay cho
manifest production đã được phân công.

## 9. Checklist trước khi treo máy qua đêm

1. Cắm sạc; đặt Windows **Sleep = Never khi plugged in**. Khóa màn hình được, nhưng máy
   không được sleep/hibernate. Nếu gập laptop, đặt hành động đóng nắp là **Do nothing**.
2. Đóng game, LM Studio và process dùng GPU. Không chạy embedding/Qwen cùng lúc.
3. Xác nhận 5/5 parity PASS và `worker-01.txt` không giao ID với worker khác.
4. Chạy batch không có `--fail-fast` để một video lỗi không làm dừng cả đêm.
5. Có thể mở PowerShell khác chạy `nvidia-smi -l 10` để theo dõi GPU/VRAM/nhiệt độ.
6. Sáng hôm sau đọc batch report; mọi `failures` phải chạy lại hoặc chuyển CPU có ghi rõ,
   tuyệt đối không coi là hoàn tất ngầm.

Nếu GPU hết VRAM ở video dài, runner ghi failure và tiếp tục video sau. Không thêm fallback
CPU tự động vì sẽ làm provenance trong cùng batch không còn đồng nhất.

## 10. Colab CUDA (lựa chọn phụ)

Không upload source `.py` và không dán hàm hậu xử lý vào notebook. Colab clone nguyên repo;
weight có sẵn trong wheel và được pipeline kiểm SHA-256.

Trong Google Drive/Colab chỉ cần:

1. các MP4 được giao, đặt trong `<AIC_DATA>/videos/`;
2. đủ 5 MP4 parity: `L21_V001`, `L21_V002`, `L21_V003`, `L21_V005`, `L21_V006`;
3. file `worker-colab.txt`, mỗi dòng một ID được giao (danh sách parity đã có trong repo);
4. 5 manifest CPU reference tương ứng, đặt riêng trong `cpu-reference/`.

Cấu trúc gợi ý:

```text
MyDrive/
├── AIC2026-colab/
│   ├── videos/
│   ├── shots/
│   ├── index/
│   └── worker-colab.txt
└── cpu-reference/
    ├── L21_V001.json
    ├── L21_V002.json
    ├── L21_V003.json
    ├── L21_V005.json
    └── L21_V006.json
```

### 10.1 Bật GPU và clone repo

Trong Colab chọn **Runtime → Change runtime type → T4 GPU**, sau đó:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone --branch codex/offline-shot-detection --single-branch https://github.com/ThanhVu165/LASTDANCE.git
%cd LASTDANCE
!git rev-parse HEAD
```

Commit SHA phải trùng worker CPU.

### 10.2 Cài và kiểm tra môi trường

```bash
!python -m pip install -r requirements/shot-colab-gpu.txt
%env AIC_DATA=/content/drive/MyDrive/AIC2026-colab
%env AIC_TRANSNETV2_DEVICE=cuda
!python -m scripts.environment_doctor --profile shot-colab-gpu
```

Doctor phải báo CUDA available, đúng package 1.0.5 và đúng weight SHA. Không cài
`requirements/offline-local.txt` trên Colab vì profile đó pin Torch CPU/local.

### 10.3 Parity gate bắt buộc trước batch 873 video

Chạy 5 video bằng adapter của repo:

```bash
!python -m scripts.run_shot_batch "configs/shot_parity_dev_subset_5.txt" --device cuda
```

`configs/shot_parity_dev_subset_5.txt` đã được commit sẵn với đúng:

```text
L21_V001
L21_V002
L21_V003
L21_V005
L21_V006
```

So sánh từng manifest CUDA với CPU reference:

```bash
!python -m scripts.compare_shot_manifests "/content/drive/MyDrive/cpu-reference/L21_V001.json" "$AIC_DATA/shots/L21_V001.json"
!python -m scripts.compare_shot_manifests "/content/drive/MyDrive/cpu-reference/L21_V002.json" "$AIC_DATA/shots/L21_V002.json"
!python -m scripts.compare_shot_manifests "/content/drive/MyDrive/cpu-reference/L21_V003.json" "$AIC_DATA/shots/L21_V003.json"
!python -m scripts.compare_shot_manifests "/content/drive/MyDrive/cpu-reference/L21_V005.json" "$AIC_DATA/shots/L21_V005.json"
!python -m scripts.compare_shot_manifests "/content/drive/MyDrive/cpu-reference/L21_V006.json" "$AIC_DATA/shots/L21_V006.json"
```

Cả 5 lệnh phải `PASS`. Bộ so sánh cho phép field `device` khác CPU/CUDA nhưng yêu cầu mọi
shot boundary, excluded range, threshold, implementation, package version và weight hash
khớp. Chỉ lệch một frame cũng phải dừng; không sửa JSON và không chạy batch production.

### 10.4 Chạy danh sách Colab sau khi parity PASS

```bash
!python -m scripts.run_shot_batch "$AIC_DATA/worker-colab.txt" --device cuda
```

Khi hoàn tất, tải về `shots/<video_id>.json` của các ID được giao và batch report tương ứng.
Kiểm tra số `requested = completed + skipped + failed`; mọi failure phải chạy lại hoặc bàn
giao rõ ràng, không coi bước Shot Detection toàn collection là hoàn tất ngầm.
