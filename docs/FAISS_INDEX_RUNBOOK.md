# Runbook build FAISS độc lập theo modality

Tài liệu này là entry point cho `BASELINE_SPEC.md` §2.1d/§2.3: embedding shard
đã hoàn tất → `IndexIDMap(IndexFlatIP)`. Bước này chạy **CPU local**, không dùng quota
Kaggle và không ảnh hưởng worker Shot Detection.

Mỗi lệnh chỉ build đúng một trong `clip`, `siglip`, `eva_clip`. Không chờ ba modality hoàn tất
cùng lúc và không làm SRRF/CLIP rollback; các logic đó thuộc Nhánh 2 online.

## 1. Input bắt buộc

Ví dụ một artifact CLIP được tải từ Kaggle/Hugging Face về local:

```text
AIC_DATA/
├── keyframes/<video_id>/<shot_id>_<local_idx>.jpg
└── index/
    ├── dev-subset-5/
    │   ├── frames.csv
    │   └── frames.csv.state.json
    └── visual-embeddings/
        └── batch-01/clip/
            ├── checkpoint.json
            ├── manifest.json
            └── shards/000000/
                ├── keyframe_uids.npy
                ├── vectors.npy
                └── manifest.json
```

Builder chỉ nhận artifact thỏa toàn bộ điều kiện:

- manifest và checkpoint đều `complete=true`;
- checkpoint có `checkpoint_resume_verified=true` từ lần ngắt và resume bằng process khác;
- model ID/revision, catalog SHA, shard hash và UID khớp;
- vector nguồn là `float16`, finite và L2 norm hợp lệ;
- mỗi batch chứa trọn tập UID trong `frames.csv` của các `video_id` mà batch khai báo.

Không truyền checkpoint dở, shard rời, chỉ riêng `vectors.npy`, hoặc artifact dùng catalog
khác. Builder đọc `keyframe_uid` đã có, không tự tính lại hash và không dùng row index.

## 2. Kiểm tra environment local

Từ root repository trên Windows:

```powershell
$env:AIC_DATA = "D:\AIC2026"
.\scripts\run_offline_windows.ps1 `
  -Module scripts.environment_doctor `
  -PythonArguments @("--profile", "offline-local", "--skip-data")
```

Profile phải có `faiss-cpu==1.9.0`. Không cần CUDA. Nếu chưa dựng environment, làm theo
`docs/ENVIRONMENT_SETUP.md` trước.

## 3. Build modality đầu tiên

Ví dụ build CLIP từ batch đầu:

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_faiss_index `
  -PythonArguments @(
    "--modality", "clip",
    "--embedding-dir", "$env:AIC_DATA\index\visual-embeddings\batch-01\clip",
    "--catalog", "$env:AIC_DATA\index\dev-subset-5\frames.csv"
  )
```

Output mặc định:

```text
AIC_DATA/index/clip.faiss
AIC_DATA/index/clip.faiss.state.json
```

FAISS `IndexFlatIP` lưu vector nội bộ ở `float32`; đây là contract của FAISS CPU. Nguồn
Kaggle/HF vẫn bắt buộc `float16`. Builder chuyển sang `float32`, chuẩn hóa L2 lại rồi mới
`add_with_ids`, không mean-pool.

## 4. Add batch mới mà không rebuild modality khác

Khi `batch-02/clip` hoàn tất cho tập video không giao với batch trước:

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.build_faiss_index `
  -PythonArguments @(
    "--modality", "clip",
    "--embedding-dir", "$env:AIC_DATA\index\visual-embeddings\batch-02\clip",
    "--catalog", "$env:AIC_DATA\index\dev-subset-5\frames.csv"
  )
```

Builder validate lại index cũ trước khi add. Cùng source signature **và digest nội dung** đã
add thì lệnh là no-op; cùng signature nhưng vector khác hoặc source signature mới có UID
chồng index cũ đều fail closed. Không xóa/rebuild
`siglip.faiss` hoặc `eva_clip.faiss`.

Có thể truyền nhiều batch rời nhau trong một lệnh bằng cách lặp `--embedding-dir`:

```text
--embedding-dir ...\batch-02\clip --embedding-dir ...\batch-03\clip
```

Ba modality có thể publish theo thứ tự bất kỳ. Ví dụ CLIP xong trước thì build/validate
`clip.faiss` ngay; không cần đợi SigLIP và EVA-CLIP.

## 5. Validate độc lập

```powershell
.\scripts\run_offline_windows.ps1 `
  -Module scripts.validate_faiss_index `
  -PythonArguments @(
    "--index", "$env:AIC_DATA\index\clip.faiss",
    "--catalog", "$env:AIC_DATA\index\dev-subset-5\frames.csv",
    "--modality", "clip"
  )
```

Validator đọc **ID thật** từ `IndexIDMap`, diff với toàn bộ UID trong `frames.csv` của từng
video được sidecar khai báo, rồi kiểm tra duplicate, dimension, finite, L2 norm, SHA file,
source/video accounting và checkpoint-resume provenance. Đếm `ntotal` bằng số dòng không
được xem là bằng chứng UID khớp.

## 6. Atomic publish và recovery

Builder ghi index + state vào file tạm, validate cặp tạm, thay `.faiss`, rồi publish state
sau cùng làm commit marker. Thiếu một trong hai file hoặc SHA không khớp phải fail closed;
không tự tin vào một `.faiss` còn sót sau crash và không tự sửa state bằng tay.

Khi cần khôi phục sau lỗi ghi đĩa, dùng lại cặp `.faiss` + `.state.json` tốt từ artifact
backup rồi chạy lại cùng source. Không xóa index production nếu chưa có bản recoverable.

## 7. Ý nghĩa `complete=true`

`complete=true` trong `clip.faiss.state.json` chỉ có nghĩa:

- index CLIP hợp lệ cho đúng các video liệt kê trong sidecar;
- UID của các video đó khớp 100% với catalog;
- mọi source batch đã qua checkpoint/resume gate thật.

Nó **không** có nghĩa video đã Ready toàn pipeline. Publishing Criteria tổng vẫn cần cả ba
FAISS, mapping `video_id/frame_id/pts_time` được sanity-check, OCR/ASR theo phạm vi sử dụng
và các gate trong spec. Không thêm logic CLIP fallback vào builder offline.

## 8. Artifact bàn giao

Luôn bàn giao theo cặp:

```text
<modality>.faiss
<modality>.faiss.state.json
```

Không commit/push file FAISS, shard embedding, JPEG, model cache hoặc dữ liệu cuộc thi vào
Git. Không tự publish GitHub/Hugging Face; chỉ làm khi chủ repo yêu cầu rõ trong phiên hiện
tại.
