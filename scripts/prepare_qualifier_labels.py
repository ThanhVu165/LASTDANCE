"""Create 60 video-disjoint annotation assignments, never fabricated ground truth."""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def prepare(catalog: Path, output: Path, seed: str = "aic26-qualifier-v1") -> Path:
    if output.exists():
        raise FileExistsError("use a new directory to preserve existing annotation work")
    by_video = defaultdict(list)
    with catalog.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            by_video[row["video_id"]].append(row)
    if len(by_video) < 60:
        raise ValueError("60 separate videos are required for this annotation allocation")
    videos = sorted(by_video, key=lambda video: hashlib.sha256(f"{seed}:{video}".encode()).hexdigest())[:60]
    assignments = []
    for index, video in enumerate(videos):
        task = ("kis", "qa", "trake")[(index % 30) // 10]
        name = f"query-p1-{index + 1}-{task}"
        frames = sorted(by_video[video], key=lambda row: int(row["frame_id"]))
        picks = sorted({round(i * (len(frames) - 1) / 4) for i in range(5)})
        assignments.append({"query": {"query_name": name, "source_filename": name + ".txt", "task_type": task.upper(),
                                      "raw_query": "", "expected_event_count": 3 if task == "trake" else None},
                            "split": "development" if index < 30 else "held_out", "video_id": video,
                            "intervals": [], "accepted_answers": [], "verified_by": ""})
        media = [{"frame_id": int(frames[i]["frame_id"]), "pts_time": float(frames[i]["pts_time"]),
                  "image": f"keyframes/{video}/{frames[i]['shot_id']}_{frames[i]['local_idx']}.jpg"} for i in picks]
        output.mkdir(parents=True, exist_ok=True)
        (output / f"{name}.media.json").write_text(json.dumps({"video_id": video, "context_only": True, "frames": media}, indent=2), encoding="utf-8")
    path = output / "labels.pending.json"
    if path.exists():
        raise FileExistsError("annotation file already exists; use a new output directory")
    path.write_text(json.dumps(assignments, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "assignment-manifest.json").write_text(json.dumps({"schema_version": 1, "seed": seed,
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(), "assignments": 60,
        "human_verified": 0, "acceptance_ready": False, "instruction": "Watch each source video; write the query, exact source-frame interval(s), QA aliases and reviewer. Context keyframes are not labels."}, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="aic26-qualifier-v1")
    args = parser.parse_args()
    print(prepare(args.catalog or Path(os.environ.get("AIC_DATA", "data")) / "index" / "frames.csv", args.output, args.seed))


if __name__ == "__main__": main()
