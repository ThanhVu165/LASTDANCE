# OCR v2 experiment — two-hour Gate A/B

This runbook preserves the experimental Gate A/B protocol; the sole current production
contract is [BASELINE_SPEC.md](BASELINE_SPEC.md) §2.2. It does not call a paid API or mark
an OCR snapshot complete.

**Decision recorded 04/09/2026:** the user approved implementing VietOCR + conditional
Paddle from cached CRAFT boxes, based on deadline and runtime/visual evidence. Gate B has
an actual artifact but no complete ground-truth acceptance report; this is **not a
quantitative Gate A/B PASS**. Keep the original thresholds below unchanged for reproducibility.
The later roughly ten-hour operational budget supersedes the original one-day scheduling
assumption, not the metric thresholds. Production recognition and local schema-v3 snapshot
are now validated; the Online adapter remains pending. See
[OCR_V2_ONLINE_HANDOFF.md](OCR_V2_ONLINE_HANDOFF.md). Human ground-truth scoring was deferred
by the user for this development handoff; it must not be reported as an accuracy PASS.

## Locked comparison

- Gate A diagnoses the existing CRAFT boxes from Batch 01.
- Gate B keeps exactly the same CRAFT regions and compares cached EasyOCR `latin_g2`,
  PaddleOCR `latin_PP-OCRv5_mobile_rec`, and VietOCR `vgg_seq2seq`.
- Vintern is not a baseline because the Batch 01 archive records zero Vintern calls.
- Gemini remains locked; no model/API cost is authorized by this experiment.
- No spelling correction is allowed. Geometric dedup is permitted later only for identical
  normalized strings in strongly overlapping boxes.

The pre-registered thresholds live in `configs/ocr_v2_gate_policy.json`. Do not edit them
after seeing challenger results.

## Inputs and environment

| Input | Purpose |
|---|---|
| `ocr-production-batch-01-easyocr.zip` | Immutable CRAFT/EasyOCR frame and region records |
| Five Batch 01 keyframe directories | JPEG evidence for 100-frame review and 5,000-crop speed canary |
| All nine EasyOCR archives | Manifest-only exact catalog region count for final ETA |

Review-bundle creation uses CPU only and can run locally or on Kaggle CPU. Challenger
inference runs only on one Kaggle T4. Production is a separate four-worker operation and
is not performed by either gate; its authorization/status is recorded in baseline §2.2.

## 1. Build a balanced review bundle

The easiest path is to upload `notebooks/kaggle_ocr_v2_review.ipynb`, attach the Batch 01
archive and only these five JPEG directories, leave GPU disabled, then Run All. The parameter
cell already locks the five video IDs and writes the bundle under `/kaggle/working`.

The equivalent repository CLI is:

```bash
python scripts/ocr_v2_review_bundle.py \
  --archive /kaggle/input/<archive>/ocr-production-batch-01-easyocr.zip \
  --keyframe-root /kaggle/input/<five-video-keyframes>/ \
  --video-ids L21_V001 L21_V002 L21_V003 L21_V005 L21_V006 \
  --sample-size 100 \
  --region-sample-size 120 \
  --output-dir /kaggle/working/ocr-v2-batch-01-review
```

The script validates the archive checksums, takes exactly 20 frames and 24 recognition crops
per video, caps the recognizer sample at two regions per frame, and produces:

```text
ocr-v2-review-bundle.zip
├── frame-review-sheets.zip
├── manual-review.csv
├── recognition-crop-sheets.zip
├── recognition-sample.jsonl
├── recognition-ground-truth.csv
└── review-report.json
```

The selection is deterministic and stratified toward no-text controls, low confidence,
repeated text, dense overlays, Vietnamese-mark proxies, numbers, small crops, and random
controls. It is not a pure random sample.

The output directory must not already exist. This fail-closed rule prevents an old and new
sample from being mixed; use a new output directory for a rerun.

Kaggle may expose an uploaded ZIP as a directory of loose files. Both forms are accepted and
the same internal SHA256SUMS is verified. The generated notebooks discover either form; do
not hardcode a `main([...])` call that omits `--video-ids`, because that removes the 20/24 per
video balance required by policy.

## 2. Label and evaluate Gate A

Open `frame-review-sheets.zip` and fill `manual-review.csv`:

| Column | Allowed values |
|---|---|
| `gt_has_text` | `yes`, `no` |
| `bbox_quality` | `correct`, `miss`, `duplicate`, `wrong`; use `not_applicable` only for a true no-text frame |
| `easyocr_quality` | `correct`, `near`, `wrong`, `empty`, `not_applicable` |
| `human_text` | Optional frame-level note; Gate B uses region-level transcription instead |

Then run:

```bash
python scripts/evaluate_ocr_v2_gates.py \
  --policy configs/ocr_v2_gate_policy.json \
  gate-a \
  --review-csv <edited-manual-review.csv> \
  --output <artifact-dir>/gate-a-report.json
```

- Bbox issue rate below 20% of labelled text-bearing frames: `KEEP_CRAFT` and proceed to the
  same-crop recognizer Gate B.
- Bbox issue rate at least 20%: `RUN_DBNET_CHALLENGER`. This authorizes only a small detector
  A/B; it does not select DBNet or permit full-catalog detection.

## 3. Label the 120 Gate B crops

Open `recognition-crop-sheets.zip` and fill only the human columns in
`recognition-ground-truth.csv`. Do not edit identifiers, bbox values, machine output, sheet
mapping, or `sample_row_sha256`.

| Column | Allowed values |
|---|---|
| `label_status` | `labeled`, `exclude_unreadable`, `false_positive` |
| `human_text` | Exact visible transcription for `labeled`; otherwise blank |
| `text_type` | `ordinary`, `ticker`, `numeric_or_name`, `other` for labelled rows |
| `notes` | Optional evidence note |

At least 100 regions must remain usable and at least five must be names/numbers (a labelled
string containing a digit also counts). Unicode is compared after NFC, case-folding and
whitespace collapse; no spelling correction is applied.

## 4. Run the Kaggle T4 challenger notebook

Upload `notebooks/kaggle_ocr_v2_gate_b.ipynb`, enable Internet, select one T4 and attach:

- the review bundle;
- Batch 01 EasyOCR archive (exactly one copy);
- the same five JPEG directories;
- the other eight EasyOCR archives for a final exact ETA.

Do not attach Batch 01 twice through two different Kaggle Datasets; duplicate batch archives
are rejected rather than guessed.

Run all cells. The notebook downloads checksum-pinned Paddle/VietOCR artifacts, reconstructs
the same 120 CRAFT crops, reads EasyOCR from cache, and measures each challenger on a
deterministic 5,000-crop canary. It writes:

```text
/kaggle/working/ocr-v2-gate-b-results.zip
├── recognizer-results.jsonl
├── runtime-report.json
└── SHA256SUMS
```

If only Batch 01 is attached, quality inference still completes, but
`ready_for_final_gate_b=false`; attach all nine manifests before using the ETA to select a
production model. Throughput excludes JPEG loading/crop construction and therefore estimates
recognition time, the phase distributed across four workers.

Gate B is intentionally short and has no production checkpoint. A failed session is rerun
fresh. Checkpoint/resume validation is mandatory only after a model wins and before its first
production shard is published.

## 5. Score Gate B

Extract the review bundle and the Gate B result ZIP, then run:

```bash
python scripts/evaluate_ocr_v2_gates.py \
  --policy configs/ocr_v2_gate_policy.json \
  gate-b \
  --sample-jsonl <review>/recognition-sample.jsonl \
  --ground-truth-csv <review>/recognition-ground-truth.csv \
  --results-jsonl <results>/recognizer-results.jsonl \
  --runtime-report-json <results>/runtime-report.json \
  --output <artifact-dir>/gate-b-report.json
```

A challenger earns a **quantitative gate PASS** only when all conditions hold:

- exact-token recall improves by at least 5 percentage points **or** CER falls by at least
  10% relatively against cached EasyOCR;
- exact accuracy on the numeric/name subset regresses by no more than 2 percentage points;
- no missing, duplicate, foreign, or error result exists;
- measured full-catalog ETA on four T4 workers is at most 18 hours.

Exact-token recall is primary. If two qualified challengers are within two percentage points,
the faster one wins. Under the original protocol, no clear winner meant retaining EasyOCR.
The later user decision to implement v2 is recorded explicitly in baseline §2.2 rather than
manufacturing a PASS or lowering these thresholds. Original EasyOCR artifacts remain immutable.

## 6. Production follow-up — current decision is in the baseline

The approved v2 plan reuses cached CRAFT boxes; no DBNet/detector replacement is authorized.
Prepare four disjoint/exhaustive assignments from all nine real batch counts and demonstrate
production checkpoint/resume before publishing. Keep `OcrResult`/`ocr_fts` unchanged and
migrate provenance honestly. Development snapshots may expose errors/residuals with exact
coverage and `complete=false`, `production_ready=false`; final requires the stricter terminal
and Publishing Criteria in baseline §2.2c/§2.3. Generated crops, model downloads, JSONL outputs
and ZIP artifacts stay outside Git. Neither gate notebook is a production runner.
