"""Build the immutable 300-image CRAFT Gate A Kaggle input dataset.

Run this once in a Kaggle notebook with ``thvu165/aic-2026-keyframes``
attached. The output ZIP can be uploaded as a private Kaggle Dataset.
Images are copied byte-for-byte; no decoding or re-encoding is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


DEV_EXPECTED = {
    "L21_V001": {"count": 1008, "uid_set_sha256": "d97f6d1cb014354b11942ea908b2f75fb7ec423dc44b87451b8fe157fc16eee2"},
    "L21_V002": {"count": 843, "uid_set_sha256": "62773f4ecc75cac52d6df5f598734fd9f608afa3ea680a74da91b983031596ca"},
    "L21_V003": {"count": 765, "uid_set_sha256": "b7d1fe33ecfeb644506b6015ee98a6e03b9b8a9baade275fab303cbe7bd4d03b"},
    "L21_V005": {"count": 744, "uid_set_sha256": "2f16c1ddc51b87bfe3299f99910f01bf7e2b8ec718b6f99015cdda21982199b0"},
    "L21_V006": {"count": 804, "uid_set_sha256": "b3b6e61ff01dbdef3afc2de96f8a077c992f43e32640264743b707bfe84942cc"},
}
FRAMES_PER_VIDEO = 60
IMAGE_PATTERN = re.compile(r"^(s\d+)_(\d+)\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_uid(video_id: str, shot_id: str, local_idx: int) -> int:
    payload = f"{video_id}:{shot_id}:{local_idx}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") >> 1


def uid_set_sha256(values: list[int]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(values)).encode()
    return hashlib.sha256(payload).hexdigest()


def locate_video(input_root: Path, video_id: str) -> Path:
    matches = [
        path
        for path in input_root.glob(f"*/keyframes-batch-*/{video_id}")
        if path.is_dir()
    ]
    if not matches:
        # Compatibility fallback for one additional dataset wrapper directory.
        matches = [
            path
            for path in input_root.glob(f"*/*/keyframes-batch-*/{video_id}")
            if path.is_dir()
        ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one directory for {video_id}, found: {matches}")
    return matches[0]


def build_catalog(input_root: Path) -> list[dict]:
    catalog: list[dict] = []
    for video_id, expected in DEV_EXPECTED.items():
        video_dir = locate_video(input_root, video_id)
        entries = []
        for image_path in video_dir.iterdir():
            match = IMAGE_PATTERN.match(image_path.name)
            if match:
                entries.append((match.group(1), int(match.group(2)), image_path))
        entries.sort(key=lambda item: (item[0], item[1]))
        rows = []
        for local_idx, (shot_id, filename_index, image_path) in enumerate(entries):
            rows.append(
                {
                    "video_id": video_id,
                    "shot_id": shot_id,
                    "local_idx": local_idx,
                    "filename_index": filename_index,
                    "keyframe_uid": make_uid(video_id, shot_id, local_idx),
                    "source_image": image_path.relative_to(input_root).as_posix(),
                    "source_path": str(image_path),
                }
            )
        if len(rows) != expected["count"]:
            raise RuntimeError(f"{video_id}: expected {expected['count']} images, got {len(rows)}")
        digest = uid_set_sha256([row["keyframe_uid"] for row in rows])
        if digest != expected["uid_set_sha256"]:
            raise RuntimeError(f"{video_id}: UID-set checksum mismatch: {digest}")
        catalog.extend(rows)
        print("CATALOG", video_id, len(rows), flush=True)
    return catalog


def select_sample(catalog: list[dict]) -> list[dict]:
    sample = []
    for video_id in DEV_EXPECTED:
        by_shot: dict[str, list[dict]] = defaultdict(list)
        for row in catalog:
            if row["video_id"] == video_id:
                by_shot[row["shot_id"]].append(row)
        representatives = [
            min(rows, key=lambda row: (abs(row["filename_index"] - 1), row["filename_index"], row["local_idx"]))
            for _, rows in sorted(by_shot.items())
        ]
        if len(representatives) < FRAMES_PER_VIDEO:
            raise RuntimeError(f"{video_id}: only {len(representatives)} unique shots")
        indexes = [
            round(index * (len(representatives) - 1) / (FRAMES_PER_VIDEO - 1))
            for index in range(FRAMES_PER_VIDEO)
        ]
        if len(set(indexes)) != FRAMES_PER_VIDEO:
            raise RuntimeError(f"{video_id}: sample indexes are not unique")
        sample.extend(representatives[index] for index in indexes)
    sample.sort(key=lambda row: (row["video_id"], row["shot_id"], row["local_idx"]))
    if len(sample) != 300 or len({row["keyframe_uid"] for row in sample}) != 300:
        raise RuntimeError("Gate A sample must contain exactly 300 unique UIDs")
    return sample


def export_dataset(sample: list[dict], output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)
    manifest_path = output_dir / "gate-a-manifest.jsonl"
    checksum_lines = []
    exported = []
    for index, source_row in enumerate(sample, 1):
        source = Path(source_row["source_path"])
        relative = Path("images") / source_row["video_id"] / source.name
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_sha256 = sha256_file(source)
        if sha256_file(destination) != source_sha256:
            raise RuntimeError(f"Byte checksum mismatch after copy: {source}")
        row = {key: value for key, value in source_row.items() if key != "source_path"}
        row.update({"image_file": relative.as_posix(), "image_sha256": source_sha256})
        exported.append(row)
        checksum_lines.append(f"{source_sha256}  {relative.as_posix()}")
        if index % 25 == 0:
            print("COPY_PROGRESS", index, "/", len(sample), flush=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in exported:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n")
    metadata = {
        "schema_version": 1,
        "purpose": "OCR CRAFT Gate A threshold pilot only",
        "complete": False,
        "production_artifact": False,
        "frames": len(exported),
        "videos": dict(Counter(row["video_id"] for row in exported)),
        "manifest_sha256": sha256_file(manifest_path),
        "selection": "60 uniformly spaced unique-shot representatives per dev-subset-5 video",
    }
    (output_dir / "dataset-provenance.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    archive = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
    print("DATASET_READY", output_dir, flush=True)
    print("ZIP", archive, "SHA256", sha256_file(archive), flush=True)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/ocr-gate-a-mini"),
    )
    args = parser.parse_args()
    catalog = build_catalog(args.input_root)
    sample = select_sample(catalog)
    print("SAMPLE", len(sample), Counter(row["video_id"] for row in sample), flush=True)
    export_dataset(sample, args.output_dir)


if __name__ == "__main__":
    main()
