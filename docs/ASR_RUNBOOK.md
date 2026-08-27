# Runbook Nhánh 3 — ASR

`docs/BASELINE_SPEC.md` §2A là contract duy nhất. Runbook này chỉ mô tả cách vận hành; nếu
có khác biệt thì baseline thắng. ASR chạy bằng tài khoản Kaggle/Colab riêng, không dùng quota
Visual của Nhánh 1 và không sửa `online/`.

## 1. Input, output và môi trường

| Pha | Input | Output | Môi trường |
|---|---|---|---|
| Audio extraction | `inventory.json`, MP4 trong `AIC_DATA/videos` | WAV + manifest/report | Local CPU, Python 3.11, FFmpeg |
| Dev Gate/transcription | 5 WAV đã verify | record JSON/video + checkpoint/manifest | Kaggle Tesla T4, Python 3.12 |
| Alignment | transcript record + `frames.csv` complete | `asr_segments.jsonl` | Local CPU |
| Index/coverage | aligned JSONL + inventory/audio/transcript | `asr.sqlite`, `asr_coverage.csv`, state | Local CPU |

WAV canonical là PCM signed 16-bit little-endian (`pcm_s16le`), 16.000 Hz, mono. Đây là
downsample/channel conversion, **không phải codec nén**. Kích thước lý thuyết gần 1,92
MB/phút; inventory hiện có 130,66 giờ audio nên full collection khoảng 15,05 GB trước header.
Report thật vẫn là authority vì duration/container từng video khác nhau.

Dev subset cố định dùng chung với Shot/Visual/OCR:

```text
L21_V001
L21_V002
L21_V003
L21_V005
L21_V006
```

Tổng duration inventory của subset khoảng 91,6 phút. Không thay subset để làm gate dễ hơn.

## 2. Chốt runner và model trước GPU

Dev Gate dùng **PyTorch + Transformers**, mỗi model ở một process riêng:

| Key | Model/revision | Weight policy | Trạng thái |
|---|---|---|---|
| `whisper_large_v3` | `openai/whisper-large-v3@06f233f...` | `model.safetensors`, SHA-256 pin | Dev Gate only |
| `phowhisper_large` | `vinai/PhoWhisper-large@b9136a4...` | `.bin`, `weights_only=True`, Torch ≥2.6, ghi SHA thật | Dev Gate only |

Model card chính chủ xác nhận cả hai dùng Transformers: [Whisper Large-v3](https://huggingface.co/openai/whisper-large-v3),
[PhoWhisper-large](https://huggingface.co/vinai/PhoWhisper-large). PhoWhisper chưa có
safetensors upstream tại revision đã pin, nên chưa được bật production. Sau Dev Gate phải
khóa SHA-256 quan sát được hoặc chuyển an toàn sang safetensors, rồi người dùng mới duyệt
`production_allowed=true` cho đúng một model.

Không dùng `chunk_length_s`: runner gọi cơ chế long-form timestamp của Whisper và yêu cầu
segment timestamp đóng. Mỗi segment phải có language `vi|en`; output khác contract dừng job.

Peak VRAM gồm cả model load và inference, đo bằng:

```python
device_index = torch.cuda.current_device()
torch.cuda.set_device(torch.device(f"cuda:{device_index}"))
# Khởi tạo CUDA context bằng một allocation nhỏ trước khi reset.
torch.cuda.reset_peak_memory_stats()
torch.cuda.max_memory_allocated()
```

Manifest bắt buộc ghi `peak_cuda_memory_bytes`, Torch/Transformers/CUDA, GPU name, model
revision và weight SHA-256. Vì runner này không dùng CTranslate2 nên không dùng NVML cho gate.
Không hardcode device index vào API memory stats; Kaggle có image Torch/CUDA từ chối
`reset_peak_memory_stats(0)` dù CUDA inference khả dụng. Notebook export Kaggle Secret
`HF_TOKEN` trước khi spawn runner để model download được xác thực.

## 3. Tách và verify 5 WAV ở local

Thiết lập `AIC_DATA`, sau đó chạy:

```powershell
& .\.venv-offline\python.exe -m scripts.extract_asr_audio `
  --inventory "$env:AIC_DATA\index\inventory.json" `
  --video-id-file configs\shot_parity_dev_subset_5.txt
```

Output mặc định:

```text
AIC_DATA/asr/audio/
├── wav/<video_id>.wav
├── manifests/<video_id>.json
└── dev-gate-audio-report.json
```

Manifest là checkpoint authority. Resume chỉ reuse WAV khi signature, SHA-256, codec,
sample rate, channel, duration và size còn đúng; không overwrite artifact mồ côi. Kiểm tra
`ready_count=5`, `no_audio_count=0`, `mean_megabytes_per_minute` gần 1,92 và nghe thử ngắn
ít nhất một WAV trước upload.

Chỉ upload `wav/`, `manifests/` và report vào private Kaggle Dataset. Source code
không đóng gói lại thành Dataset: notebook clone `codex/offline-asr` từ GitHub và
checkout detached đúng commit runner đã pin, cùng cấu trúc vận hành với notebook
SigLIP. Không commit WAV/model cache/transcript artifact vào Git.

## 4. Dev Gate thật trên Kaggle T4

Notebook chuẩn: `notebooks/kaggle_asr_dev_gate.ipynb`. Trước khi upload notebook,
`EXPECTED_COMMIT` phải là SHA 40 ký tự của commit chứa ASR runner trên remote branch
`codex/offline-asr`; placeholder hoặc branch HEAD động đều bị từ chối. Kaggle chỉ cần
attach Dataset audio, bật Internet/T4 và Kaggle Secret `HF_TOKEN`. Notebook clone/fetch,
checkout detached và xác minh `git rev-parse HEAD` trước khi cài profile:

```bash
python -m pip install -r requirements/asr-kaggle-gpu.txt
python -m scripts.environment_doctor --profile asr-kaggle-gpu --skip-data
```

Với **từng model**, lần đầu cố ý dừng sau 2 video và phải trả exit 75:

```bash
python -m scripts.run_asr_dev_gate \
  --model whisper_large_v3 \
  --audio-root /kaggle/input/lastdance-asr-dev-gate/audio \
  --output-root /kaggle/working/asr-transcripts \
  --stop-after-videos 2
```

Lần hai là process mới, scan lại record đã publish rồi chạy nốt:

```bash
python -m scripts.run_asr_dev_gate \
  --model whisper_large_v3 \
  --audio-root /kaggle/input/lastdance-asr-dev-gate/audio \
  --output-root /kaggle/working/asr-transcripts \
  --require-resume-verified
```

Lặp lại hai lệnh cho `phowhisper_large`. Không load hai model trong cùng process. Gate chỉ
PASS khi cả hai manifest có đúng 5 video, `checkpoint_resume_verified=true`, GPU là Tesla
T4, peak VRAM dương và không có transcript/timestamp/schema lỗi.

So sánh:

```bash
python -m scripts.compare_asr_dev_gate \
  --whisper-manifest /kaggle/working/asr-transcripts/dev-subset-5/whisper_large_v3/manifest.json \
  --phowhisper-manifest /kaggle/working/asr-transcripts/dev-subset-5/phowhisper_large/manifest.json \
  --output /kaggle/working/asr-dev-gate-comparison.json
```

Report có runtime, real-time factor, segment/no-speech count và peak VRAM. Nghe đối chiếu
timestamp/text trên cả 5 video vẫn bắt buộc. Không chọn model chỉ vì nhanh hơn.

Sau khi cả hai gate PASS, notebook đóng gói transcript/manifest/comparison, tính SHA-256
và atomic-upload archive + checksum bằng Kaggle Secret `HF_TOKEN` vào private HF Dataset
`<HF_USER>/lastdance-asr-artifacts` (owner resolve từ chính token), namespace
`asr/dev-gate/dev-subset-5/`. Upload này chỉ lưu evidence; comparison vẫn giữ
`production_model_selected=false` và manual listening review vẫn bắt buộc. Notebook fail
closed nếu repo không private, remote chỉ có một trong archive/checksum, hoặc checksum
tải lại không khớp. Không upload WAV, model weight/cache hoặc token.

### WER tùy chọn

Chỉ dùng nếu đã có ground truth sẵn. File JSON:

```json
{
  "samples": [
    {
      "video_id": "L21_V001",
      "start_time": 10.0,
      "end_time": 18.0,
      "reference_text": "đoạn lời thoại đúng đã có sẵn"
    }
  ]
}
```

Thêm `--ground-truth path.json` vào lệnh compare. Normalization casefold, bỏ punctuation và
split whitespace được khóa trong code. Không tự tạo ground truth quy mô lớn chỉ để có WER.

## 5. Alignment và SQLite local

Sau khi người dùng chốt đúng một model và `frames.csv` production complete:

```powershell
& .\.venv-offline\python.exe -m scripts.align_asr_segments `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --transcript-records "$env:AIC_DATA\index\asr\transcripts\batch-01\MODEL\records" `
  --output "$env:AIC_DATA\index\asr\batch-01\asr_segments.jsonl"
```

Nếu có keyframe trong segment, alignment chọn frame gần midpoint; nếu không có, chọn frame
gần `start_time` nhất. Tie-break là `pts_time`, rồi `keyframe_uid`. Mọi UID được kiểm lại
trong đúng `video_id`.

Build FTS5/coverage:

```powershell
& .\.venv-offline\python.exe -m scripts.build_asr_index `
  --inventory "$env:AIC_DATA\index\inventory.json" `
  --catalog "$env:AIC_DATA\index\frames.csv" `
  --transcript-records "$env:AIC_DATA\index\asr\transcripts\batch-01\MODEL\records" `
  --aligned-jsonl "$env:AIC_DATA\index\asr\batch-01\asr_segments.jsonl" `
  --video-id-file configs\shot_parity_dev_subset_5.txt `
  --output-dir "$env:AIC_DATA\index\asr\dev-subset-5"
```

`no_speech` mặc định là incomplete. Chỉ truyền `--verified-no-speech-file` sau khi con người
nghe xác minh đúng video. State chỉ `complete=true` khi mọi video trong selection qua gate.

## 6. Bàn giao và file không commit

Bàn giao production gồm `asr.sqlite`, `asr_coverage.csv`, `asr.state.json`, transcript
checkpoint/manifest và model/runtime provenance. Không commit WAV, model cache, JSON record
production, SQLite production, HF token hoặc Kaggle secret. Upload evidence theo batch/revision,
không push từng video và không tự commit/push Git.
