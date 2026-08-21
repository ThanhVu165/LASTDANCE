# AGENTS.md — LASTDANCE repository guide

This file applies to the entire repository. Human contributors and coding agents
must treat it as the default operating contract.

## Read first

Read these documents in order before changing retrieval behavior:

1. `docs/PROJECT_CONTEXT.md`
2. `docs/TEAM_SETUP.md`
3. `docs/AI_COLLABORATION_GUIDE.md`
4. `README.md`
5. `docs/model_first_runtime_report_2026-08-21.md` for the latest measured state

Historical plans and checkpoints are context, not current runtime instructions.

## Product objective

LASTDANCE is an AIC2026 video retrieval system for KIS, QA and TRAKE. Optimize
the official ranking cutoffs `R@1, R@5, R@20, R@50, R@100`, with this priority:

1. retrieve the correct video/window;
2. rank the strongest evidence first;
3. answer QA or refine TRAKE frames only after retrieval is credible.

## Hard invariants

- Public query endpoints accept one natural-language `text` field and return at
  most 100 ranked rows. The competition path requests exactly 100 when enough
  candidates exist.
- `local_idx` is an internal keyframe number. `frame_id` is the real source-video
  frame from `data/map-keyframes`; submissions must use `frame_id`.
- Never treat one keyframe vector as a representation of a complete video.
- Semantic understanding is model-first. Qwen structured planning and multimodal
  verification are primary; regex/heuristics are bounded fallbacks, not a place
  for query-specific patches.
- A model score is evidence, not ground truth. Do not claim accuracy without a
  labeled dev set on the current dataset.
- Missing or incomplete optional indexes must fail closed and preserve the
  organizer CLIP production path.
- Do not commit `data/`, model caches, `.venv`, query files, submissions, logs or
  credentials.

## Current production path

- Query planning and generative verification: `Qwen/Qwen3-VL-2B-Instruct`.
- Dedicated target reranker: `Qwen/Qwen3-VL-Reranker-2B`; use only when its full
  local checkpoint exists. Runtime must not download multi-GB models.
- Primary recall: organizer `clip-ViT-B-32` image features plus the compatible
  multilingual text tower.
- KIS: structured plan → multi-prompt recall → fusion/storyboard → model
  verification → repair retrieval → cutoff ranking → exact-frame refinement.
- QA reuses KIS verified retrieval, then performs temporal VQA.
- TRAKE remains temporal moment retrieval/alignment plus visual and exact-frame
  refinement; migrating it to the shared verified layer is backlog.
- SigLIP2 and Qwen video-window indexes are optional. A state file with
  `complete=false` is not production-ready and must be ignored.

## GPU safety

The reference machine is an RTX 4050 Laptop GPU with 6 GiB VRAM. Do not run any
two of the following concurrently:

- FastAPI after Qwen has loaded;
- OCR indexing;
- SigLIP2 indexing;
- Qwen embedding/reranker builds or smoke tests.

Release one model before loading another. Keep model downloads separate from
runtime requests. Never silently switch a CUDA workload to CPU; report it.

## Change workflow

1. Inspect the relevant pipeline, configuration and tests before editing.
2. Preserve user changes and generated data.
3. Put new behavior behind configuration or an unavailable-index check until it
   passes smoke/E2E evaluation.
4. Run from `backend/`:

   ```powershell
   .\.venv\Scripts\python.exe -m compileall -q app
   .\.venv\Scripts\python.exe -m unittest discover -s tests -q
   ```

5. For ranking changes, report result count, verified/scored count, distinct
   videos, latency and peak VRAM. Report Recall@k only when ground truth exists.
6. Update `README.md` and the relevant document in `docs/` when architecture,
   models, defaults, commands or runtime status change.

## Do not do

- Do not add hard-coded fixes for colors, objects, Vietnamese phrases or known
  sample queries.
- Do not mix raw cosine scores from unrelated embedding spaces without rank or
  score calibration.
- Do not publish a partial FAISS index or mark a checkpoint complete early.
- Do not delete OCR state, production indexes, datasets or model caches as
  “cleanup”. Prove code is dead with repository search before removing it.
- Do not use `git reset --hard` or overwrite a dirty working tree.

## Definition of done

A retrieval change is done only when code compiles, all tests pass, the API still
returns the required schema/count, GPU/latency measurements are recorded, failure
fallbacks are verified, and documentation reflects the actual active model/index.

