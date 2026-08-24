# Runbook Shot Detection bằng TransNetV2

Tài liệu này dành cho thành viên chạy bước **1.1 Shot Detection** của Nhánh 1 trên Windows
CPU hoặc Google Colab CUDA. Đây là hướng dẫn vận hành chuẩn: luôn chạy code từ repository,
không copy hoặc viết lại `predictions_to_scenes()` trong notebook/script riêng.

## 1. Phạm vi và contract

- Input: MP4 tại `AIC_DATA/videos/<video_id>.mp4`.
- Output: shot manifest schema v2 tại `AIC_DATA/shots/<video_id>.json`.
- CPU local là default. Colab CUDA chỉ dùng sau parity gate đủ 5 video ở mục 9.
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

## 3. Dựng môi trường Windows CPU

Từ root repository:

```powershell
.\scripts\bootstrap_miniforge_windows.ps1
```

Script dựng `.venv-offline`, cài Python 3.11.9, FFmpeg/ffprobe 7.1.1, Torch 2.12.1,
`transnetv2-pytorch==1.0.5`, kiểm tra weight rồi chạy compile/test. Không cần tải weight
thủ công.

Nếu PowerShell chặn script:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap_miniforge_windows.ps1
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

## 5. Kiểm tra môi trường Windows

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "offline-local")
```

Chỉ tiếp tục khi Python, FFmpeg/ffprobe, Torch, TransNetV2 và weight SHA-256 đều `PASS`.

## 6. Chạy một video CPU

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.detect_shots `
  -PythonArguments @(
    "$env:AIC_DATA\videos\L21_V001.mp4",
    "--device", "cpu"
  )
```

Manifest phải có schema v2, signature đầy đủ, shot tăng dần/không overlap,
`excluded_transition_ranges` và coverage validation.

Reference trên máy chuẩn:

| Video | Total frame | Shot | Excluded ranges |
|---|---:|---:|---|
| `L21_V001` | 37.849 | 336 | Không có |
| `L21_V002` | 31.720 | 281 | `[49,49]` |
| `L21_V003` | 29.946 | 255 | Không có |
| `L21_V005` | 28.294 | 248 | `[57,57]` |
| `L21_V006` | 31.064 | 268 | `[0,0]`, `[1063,1063]` |

Chỉ nhận khi cùng MP4 cho ra đúng toàn bộ boundary/range, không chỉ cùng số shot.

## 7. Chạy batch được phân công

Tạo `worker-01.txt`, mỗi dòng đúng một `video_id`, không khoảng trắng và không trùng:

```text
L21_V007
L21_V008
L21_V009
```

Windows CPU:

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.run_shot_batch `
  -PythonArguments @(".\worker-01.txt", "--device", "cpu")
```

Batch runner dùng một model cho cả danh sách, kiểm tra manifest cũ trước khi skip, ghi từng
manifest atomic và tiếp tục video kế tiếp nếu một video lỗi. `--fail-fast` dùng khi muốn dừng
ngay; `--overwrite` chỉ dùng khi chủ động chạy lại artifact cũ.

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

## 9. Colab CUDA: cần đưa những gì lên

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

### 9.1 Bật GPU và clone repo

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

### 9.2 Cài và kiểm tra môi trường

```bash
!python -m pip install -r requirements/shot-colab-gpu.txt
%env AIC_DATA=/content/drive/MyDrive/AIC2026-colab
%env AIC_TRANSNETV2_DEVICE=cuda
!python -m scripts.environment_doctor --profile shot-colab-gpu
```

Doctor phải báo CUDA available, đúng package 1.0.5 và đúng weight SHA. Không cài
`requirements/offline-local.txt` trên Colab vì profile đó pin Torch CPU/local.

### 9.3 Parity gate bắt buộc trước batch 873 video

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

### 9.4 Chạy danh sách Colab sau khi parity PASS

```bash
!python -m scripts.run_shot_batch "$AIC_DATA/worker-colab.txt" --device cuda
```

Khi hoàn tất, tải về `shots/<video_id>.json` của các ID được giao và batch report tương ứng.
Kiểm tra số `requested = completed + skipped + failed`; mọi failure phải chạy lại hoặc bàn
giao rõ ràng, không coi bước Shot Detection toàn collection là hoàn tất ngầm.
