import json
import shutil
import unittest
from pathlib import Path

from offline.asr_snapshot import build_asr_snapshot
from offline.catalog import write_frames_catalog_atomic
from offline.identifiers import make_keyframe_uid
from scripts.publish_asr_index import publish_asr_index
from shared.schemas.frame import FrameRecord


class PublishAsrIndexTests(unittest.TestCase):
    def test_atomic_publish(self):
        root = Path("asr_publish_test_artifacts")
        shutil.rmtree(root, ignore_errors=True)
        try:
            frame = FrameRecord(video_id="v1", local_idx=0, frame_id=0, pts_time=0.0,
                                shot_id="s1", keyframe_uid=make_keyframe_uid("v1", "s1", 0))
            catalog = root / "frames.csv"
            write_frames_catalog_atomic(catalog, records=[frame], sources=[{
                "video_id": "v1", "plan_sha256": "x", "quality_sha256": "x",
                "quality_config_signature": "x"
            }])
            envelope = {
                "batch_id": "batch-01", "video_id": "v1", "status": "silent",
                "engine": "whisper_large_v3", "audio_path": "audio.flac",
                "duration_seconds": 1, "segments": []
            }
            source = root / "envelope.jsonl"
            source.write_text(json.dumps(envelope) + "\n", encoding="utf-8")
            destination, _ = build_asr_snapshot(
                catalog_path=catalog,
                records=__import__("offline.asr_snapshot", fromlist=["load_envelope_records"]).load_envelope_records([source]),
                output_root=root / "snapshots",
                source_paths=[source],
            )
            output = publish_asr_index(destination, data_root=root / "data")
            self.assertTrue(output.is_file())
            self.assertTrue((root / "data" / "index" / "asr.coverage.json").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
