"""Build the self-contained 30-crop sharpening notebook without running OCR."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "scripts/kaggle_ocr_v2_sharpen_runtime.py"
OUTPUT = ROOT / "notebooks/kaggle_ocr_v2_sharpen.ipynb"


def build_notebook():
    runtime = RUNTIME.read_text(encoding="utf-8")
    # Load only the extractor and its allow-list; no image/GPU dependencies for codegen.
    names = {"REFERENCE_NAMES", "reference_source"}
    nodes = [node for node in ast.parse(runtime).body
             if (isinstance(node, ast.FunctionDef) and node.name in names)
             or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in names)]
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(RUNTIME), "exec"), namespace)
    reference = namespace["reference_source"]((ROOT / "scripts/kaggle_ocr_v2_gate_b_runtime.py").read_text(encoding="utf-8"))
    intro = """# OCR v2 — thử làm nét 30 crop (không phải production)

Dùng notebook riêng trên **Kaggle T4**, bật Internet. Có thể dùng môi trường Gate B đã chạy.
Bước ENV tự cài VietOCR 0.3.13 đã kiểm tra checksum nếu thiếu, bổ sung dependency Python
còn thiếu bằng --no-deps và probe import trước khi tạo crop; giữ Torch/NVIDIA hiện có.
Chạy ba code cell bên dưới; không cần chạy lại notebook Gate B.

Input: review bundle Gate A, kết quả Gate B và JPEG Batch 01. ZIP và thư mục Kaggle
tự giải nén đều được hỗ trợ; chỉ dùng ảnh keyframe gốc, không dùng contact sheet.
Không cần tải chín EasyOCR archive để chạy phép thử này.

30 crop (6/video) × 3 phương án = 90 lượt VietOCR; không chạy benchmark 5.000 crop.
Sau khi model sẵn sàng, ngân sách nhận dạng 600 giây. PNG/checksum/export là pha riêng.

Download `ocr-v2-sharpen-results.zip`, mở `sheets/compare-01.png` … `compare-30.png`.
Điền `visual-review.csv`: original_readable = yes/no; các phương án =
better/same/worse/unreadable. So với chữ trên ảnh gốc, không chấm bằng confidence.
Chỉ cân nhắc phương án có ≥3 crop better và 0 crop worse trên các crop đọc được.
Không tự bật preprocessing production từ phép thử này.

Thêm Kaggle Secret `HF_TOKEN` có quyền ghi dataset OCR. Mặc định checkpoint được lưu
và kiểm tra checksum trên HF sau mỗi minibatch; session mới tự tải lại theo signature.
Nếu tắt HF, cần bản ZIP đã download để gắn lại vào Input và điền RESTORE_CHECKPOINT.
File `/kaggle/working` không bảo đảm lưu bền qua việc hủy VM. Demo resume: INTERRUPT_AFTER_NEW=6,
chạy, rồi đổi về 0 và chạy lại; không cần reset session.
"""
    parameters = """# Để trống để tự tìm trong /kaggle/input và output Gate B của session hiện tại.
REVIEW_BUNDLE = ''
GATE_B_RESULTS = ''
KEYFRAMES_ROOT = ''
RESTORE_CHECKPOINT = ''  # ZIP checkpoint/result cũ hoặc thư mục có checkpoint.json
DURABLE_CHECKPOINT_TO_HF = True  # Requires Kaggle Secret HF_TOKEN (write).
HF_REPO_ID = 'MinhThuw0103/lastdance-visual-embeddings'
SHARPEN_BATCH_SIZE = 6
INTERRUPT_AFTER_NEW = 0  # Đặt 6 để demo ngắt sau minibatch đầu, sau đó trả về 0.
# Nếu cần: SHARPEN_OUTPUT_ROOT = '/kaggle/working/ocr-v2-sharpen-other-run'
# Nếu cần: VIETOCR_CACHE_DIR = '/kaggle/working/ocr-v2-gate-b'
"""
    support = ("# Pinned Gate B crop/model helpers; no Gate B inference is executed.\n"
               f"GATE_B_REFERENCE_SOURCE = {reference!r}\n"
               f"SHARPEN_RUNTIME_SHA256 = {hashlib.sha256(runtime.encode('utf-8')).hexdigest()!r}\n")
    def cell(kind, source):
        result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result
    return {"cells": [cell("markdown", intro), cell("code", parameters), cell("code", support), cell("code", runtime)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}},
            "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
