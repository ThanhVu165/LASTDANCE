"""Run reviewed queries with the configured Online providers; preserve resumable traces."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from shared.evaluation import EvaluationCase, evaluate, validate_suite
from shared.schemas.online import SearchRequest


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "held_out", "regression"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--execute", action="store_true", help="Explicitly run configured encoders/planner/VQA; cloud providers may consume quota")
    args = parser.parse_args()
    cases = [EvaluationCase.model_validate(row) for row in json.loads(args.labels.read_text(encoding="utf-8"))]
    validate_suite(cases, acceptance=args.split != "regression")
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    labels_hash = evaluate(cases, {}, split=args.split)["labels_sha256"]
    if args.split == "held_out":
        lock = json.loads(args.freeze.read_text(encoding="utf-8")) if args.freeze else {}
        if lock.get("config_sha256") != config_hash or lock.get("labels_sha256") != labels_hash or lock.get("selected_on") != "development":
            raise ValueError("held-out execution requires the frozen configuration and labels")
    if not args.execute:
        parser.error("labels validated; pass --execute to run the configured providers")
    from online.engine import OnlineEngine
    from online.config import OnlineConfig
    engine = OnlineEngine(config=OnlineConfig.load(args.config), deep_preflight=True)
    signature = {"config_sha256": config_hash, "labels_sha256": labels_hash,
                 "catalog_sha256": engine.registry.catalog.sha256, "split": args.split}
    # Bind resume to effective inputs and implementation, not only the ranking config.
    repository = Path(__file__).resolve().parents[1]
    code_digest = hashlib.sha256()
    for source in sorted([*(repository / "online").rglob("*.py"), *(repository / "shared").rglob("*.py"), Path(__file__)]):
        code_digest.update(source.relative_to(repository).as_posix().encode() + source.read_bytes())
    signature["code_sha256"] = code_digest.hexdigest()
    environment = {key: value for key, value in os.environ.items() if key.startswith("AIC_")
                   and any(part in key for part in ("MODEL", "REVISION", "DEVICE", "PROVIDER", "BACKEND", "PRECISION"))
                   and not any(part in key for part in ("KEY", "TOKEN", "SECRET", "PASSWORD"))}
    signature["provider_settings_sha256"] = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()
    from offline.artifacts import sha256_file
    signature["text_artifacts"] = {name: sha256_file(path) if path.is_file() else None for name, path in
                                  (("ocr", engine.registry.layout.ocr), ("asr", engine.registry.layout.asr))}
    signature["visual_states"] = {name: sha256_file(engine.registry.layout.faiss_state(name)) for name in engine.registry.visual}
    state = args.output / "run-signature.json"
    if state.exists() and json.loads(state.read_text(encoding="utf-8")) != signature:
        raise ValueError("benchmark resume signature mismatch; use a new output directory")
    atomic_json(state, signature)
    predictions, traces = {}, {}
    for case in (c for c in cases if c.split == args.split):
        name = case.query.query_name
        checkpoint = args.output / "queries" / f"{name}.json"
        if checkpoint.exists():
            trace = json.loads(checkpoint.read_text(encoding="utf-8"))
            if trace.get("signature") != signature:
                raise ValueError("query checkpoint signature mismatch")
        else:
            started = time.perf_counter()
            run = engine.search(SearchRequest(task_type=case.query.task_type, raw_query=case.query.raw_query, query_spec=case.query))
            candidates = run.top_candidates or run.task_candidates
            rows = [{"video_id": c.video_id, "frame_ids": getattr(c, "frame_ids", None) or [c.frame_id],
                     "answer": getattr(c, "answer", None)} for c in candidates]
            trace = {"signature": signature, "config_sha256": config_hash, "elapsed_ms": (time.perf_counter() - started) * 1000,
                     "requires_operator_review": not candidates or any(getattr(c, "requires_review", False) for c in candidates),
                     "predictions": rows, "search_run": run.model_dump(mode="json")}
            atomic_json(checkpoint, trace)
        predictions[name], traces[name] = trace["predictions"], trace
    atomic_json(args.output / "predictions.json", predictions)
    atomic_json(args.output / "runs.json", traces)
    print(args.output)


if __name__ == "__main__": main()
