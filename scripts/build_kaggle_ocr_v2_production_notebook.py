"""Build an import-safe self-contained notebook. No model downloads/inference."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/kaggle_ocr_v2_production.ipynb"


def build_notebook():
    runtime = (ROOT / "scripts/kaggle_ocr_v2_production_runtime.py").read_text(encoding="utf-8")
    # Allow-list only crop/model definitions, never execute Gate B's top-level loop.
    nodes = [node for node in ast.parse(runtime).body
             if isinstance(node, ast.FunctionDef) and node.name == "reference_source"
             or isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and node.targets[0].id == "REFERENCE_NAMES"]
    namespace = {"ast": ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "extractor", "exec"), namespace)
    reference = namespace["reference_source"]((ROOT / "scripts/kaggle_ocr_v2_gate_b_runtime.py").read_text(encoding="utf-8"))
    environment = (ROOT / "scripts/ocr_v2_environment.py").read_text(encoding="utf-8")
    intro = """# OCR v2 — 4-worker recognition (canary first)

Chỉ chạy model trên Kaggle T4. Internet ON, Kaggle Secret **HF_TOKEN** có quyền ghi repo private.
Không gọi API, không chạy lại EasyOCR/Vintern, không làm nét. Notebook xuất recognition/selection
evidence, **chưa xuất terminal envelope hoặc SQLite Online**. Xem runbook OCR v2 production.

1. Gắn dataset Kaggle chứa **frames.csv + frames.csv.state.json**. Một notebook **CPU**
   chạy `ACTION='plan'` một lần: đọc catalog từ Kaggle Input, tải/validate 9 archive OCR từ HF,
   không đọc JPEG và không chạy OCR. Tải `ocr-v2-worker-plan.json` về rồi gắn cùng file này
   vào bốn notebook worker (cùng input revision, không tạo lại bốn plan).
2. Trên mỗi tài khoản gắn cùng catalog/state và JPEG datasets đúng batch do plan phân công, chọn T4,
   `ACTION='run'`, `WORKER_SLOT=1..4`, `RUN_MODE='canary'`. Mặc định dừng sau 1 minibatch
   đã verify HF. Đặt `INTERRUPT_AFTER_MINIBATCHES=0` rồi chạy lại trong process/session mới.
3. Canary hoàn tất sẽ in `report_sha256`. Review kết quả/elapsed và copy hash vào
   `APPROVED_CANARY_SHA256` trên đúng worker trước khi chọn `RUN_MODE='production'`.
   Production không được coi ready; còn bước migration/union/SQLite và publishing gate.

Catalog/keyframes nằm trong dataset Kaggle; HF lưu archive/checkpoint/kết quả OCR và embedding.
`CATALOG_PATH=''` tự tìm dưới `INPUT_ROOT`; nhiều bản khác nhau thì điền exact path Kaggle.
Mất VM: chạy lại cùng notebook/plan/worker, gắn đúng catalog/state/keyframes. Chỉ checkpoint HF đã verify
là bền vững; phần sau checkpoint cuối có thể chạy lại. Không chạy trùng WORKER_SLOT đồng thời.
Checkpoint local mỗi minibatch, HF delta tối đa 5 phút giữa mốc kiểm tra + cuối pha/dừng chủ động.
"""
    parameters = """ACTION = 'plan'  # 'plan' chỉ một lần trên CPU; 'run' trên từng T4
HF_REPO_ID = 'MinhThuw0103/lastdance-visual-embeddings'
INPUT_REVISION = ''  # plan resolve một commit rồi khóa; worker dùng revision trong plan
INPUT_ROOT = '/kaggle/input'  # thư mục dataset được gắn vào notebook
CATALOG_PATH = ''  # tự tìm frames.csv trong INPUT_ROOT; hoặc điền exact path Kaggle (có state bên cạnh)
WORKER_PLAN = ''  # run: tự tìm ocr-v2-worker-plan.json duy nhất dưới /kaggle/input
KEYFRAMES_ROOT = INPUT_ROOT
WORKER_SLOT = 1  # bốn tài khoản dùng 1, 2, 3, 4 không trùng nhau
RUN_MODE = 'canary'
INTERRUPT_AFTER_MINIBATCHES = 1  # sau intentional stop, đổi về 0 rồi chạy lại
APPROVED_CANARY_SHA256 = ''  # bắt buộc trước production; hash report của đúng worker
RUN_SETUP = True  # run only; môi trường Gate B đúng pin có thể bỏ setup
"""
    sources = {"kaggle_ocr_v2_production_runtime.py": runtime,
               "kaggle_ocr_v2_gate_b_runtime.py": reference,
               "ocr_v2_environment.py": environment}
    support = "# Embedded source files; Gate B inference loop is not included.\nSOURCES = " + repr(sources) + "\n"
    launch = (ROOT / "scripts/kaggle_ocr_v2_production_launch.py").read_text(encoding="utf-8")
    def cell(kind, source):
        value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        return value
    return {"cells": [cell("markdown", intro), cell("code", parameters), cell("code", support), cell("code", launch)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT)
