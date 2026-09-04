import json
import tempfile
import unittest
from pathlib import Path

from offline.artifacts import sha256_file
from offline.asr_artifacts import video_set_sha256
from offline.asr_handoff import materialize_asr_handoff
from offline.catalog import write_frames_catalog_atomic
from offline.identifiers import make_keyframe_uid
from shared.schemas.frame import FrameRecord


class AsrHandoffTests(unittest.TestCase):
    def _archive(self, root: Path, batch_id: str, rows: list[dict]) -> tuple[Path, Path]:
        directory = root / batch_id
        directory.mkdir()
        jsonl = directory / "asr-envelope.jsonl"
        jsonl.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        statuses = {name: sum(row["status"] == name for row in rows) for name in ("success", "silent", "error")}
        manifest = {
            "batch_id": batch_id, "completion_gate_passed": True,
            "catalog_sha256": self.catalog_sha, "shard_sha256": sha256_file(jsonl),
            "record_count": len(rows), "processed_videos": len(rows),
            "success_videos": statuses["success"], "silent_videos": statuses["silent"],
            "error_videos": statuses["error"],
            "expected_video_sha256": video_set_sha256(row["video_id"] for row in rows),
        }
        manifest_path = directory / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return jsonl, manifest_path

    @staticmethod
    def _row(batch: str, video: str, text: str, uid: int, *, start: float = 1, end: float = 2) -> dict:
        return {
            "schema_version": 1, "batch_id": batch, "video_id": video,
            "status": "success", "engine": "whisper_large_v3",
            "audio_path": f"asr/audio/{batch}/{video}.flac", "audio_sha256": "a" * 64,
            "audio_duration_seconds": 10, "segments": [{
                "video_id": video, "segment_id": "s000000", "start_time": start,
                "end_time": end, "transcribed_text": text, "language": "vi",
                "keyframe_uid_nearest": uid,
            }],
        }

    def test_clamps_realigns_deduplicates_equivalent_and_quarantines_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = []
            by_video = {}
            for video in ("v1", "v2", "v3"):
                values = [
                    FrameRecord(video_id=video, local_idx=i, frame_id=i, pts_time=pts,
                                shot_id=f"s{i}", keyframe_uid=make_keyframe_uid(video, f"s{i}", i))
                    for i, pts in enumerate((1.0, 9.0))
                ]
                frames.extend(values)
                by_video[video] = values
            catalog = root / "frames.csv"
            write_frames_catalog_atomic(catalog, records=frames, sources=[{
                "video_id": video, "plan_sha256": "x", "quality_sha256": "x",
                "quality_config_signature": "x"
            } for video in by_video])
            self.catalog_sha = sha256_file(catalog)
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"videos": [
                {"video_id": video, "duration": 10} for video in by_video
            ]}), encoding="utf-8")
            batch2 = [
                self._row("batch-02", "v1", "tail", by_video["v1"][0].keyframe_uid, start=8, end=12),
                self._row("batch-02", "v2", "same", by_video["v2"][0].keyframe_uid),
                self._row("batch-02", "v3", "first", by_video["v3"][0].keyframe_uid),
            ]
            batch3 = [self._row("batch-03", "v2", "same", by_video["v2"][0].keyframe_uid)]
            archives = [self._archive(root, "batch-02", batch2), self._archive(root, "batch-03", batch3)]
            checkpoint = root / "checkpoint.jsonl"
            checkpoint.write_text(json.dumps(self._row(
                "batch-01", "v3", "different", by_video["v3"][0].keyframe_uid
            )) + "\n", encoding="utf-8")
            state = root / "checkpoint.json"
            state.write_text(json.dumps({"batch_id": "batch-01", "completed": ["v3"]}), encoding="utf-8")

            records, audit = materialize_asr_handoff(
                archive_pairs=archives, checkpoint_pair=(checkpoint, state),
                catalog_path=catalog, inventory_path=inventory,
                output_jsonl=root / "union.jsonl", audit_path=root / "audit.json",
                source_revision="f" * 40,
            )
            self.assertEqual([row.video_id for row in records], ["v1", "v2"])
            v1 = records[0]
            self.assertEqual(v1.segments[0].end_time, 10)
            self.assertEqual(v1.segments[0].keyframe_uid_nearest, by_video["v1"][1].keyframe_uid)
            self.assertEqual(audit["equivalent_duplicate_videos"], 1)
            self.assertEqual(audit["conflicts"][0]["video_id"], "v3")
            self.assertEqual(audit["timestamp_corrections"][0]["action"], "clamp_to_audio_duration")

    def test_rejects_archive_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = FrameRecord(video_id="v1", local_idx=0, frame_id=0, pts_time=1,
                                shot_id="s0", keyframe_uid=make_keyframe_uid("v1", "s0", 0))
            catalog = root / "frames.csv"
            write_frames_catalog_atomic(catalog, records=[frame], sources=[{
                "video_id": "v1", "plan_sha256": "x", "quality_sha256": "x",
                "quality_config_signature": "x"
            }])
            self.catalog_sha = sha256_file(catalog)
            archive = self._archive(root, "batch-02", [self._row("batch-02", "v1", "x", frame.keyframe_uid)])
            Path(archive[1]).write_text(Path(archive[1]).read_text().replace(sha256_file(archive[0]), "0" * 64))
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps({"videos": [{"video_id": "v1", "duration": 10}]}))
            with self.assertRaisesRegex(ValueError, "shard SHA"):
                materialize_asr_handoff(
                    archive_pairs=[archive], checkpoint_pair=None, catalog_path=catalog,
                    inventory_path=inventory, output_jsonl=root / "union.jsonl",
                    audit_path=root / "audit.json", source_revision="f" * 40,
                )


if __name__ == "__main__":
    unittest.main()
