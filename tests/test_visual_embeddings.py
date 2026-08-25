import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from offline.catalog import write_frames_catalog_atomic
from offline.artifacts import sha256_file
from offline.visual_embeddings import (
    IntentionalEmbeddingInterruption,
    load_embedding_catalog,
    normalize_vectors,
    run_visual_embedding,
    validate_completed_visual_embedding,
)
from offline.visual_models import load_model_config
from scripts.verify_visual_model_revisions import verify_visual_model_revisions
from shared.schemas.frame import FrameRecord


class FakeEncoder:
    modality = "clip"
    model_id = "test/clip"
    model_revision = "a" * 40
    runtime_metadata = {
        "device": "fake",
        "python": "3.12.13",
        "system": "test",
        "machine": "test",
        "transformers": "5.15.1",
        "torch": "2.10.0+cu128",
    }

    def __init__(self) -> None:
        self.encoded_paths: list[Path] = []

    def encode(self, image_paths):
        self.encoded_paths.extend(image_paths)
        rows = []
        for index, _ in enumerate(image_paths, start=1):
            rows.append([float(index), float(index + 1), 2.0])
        return np.asarray(rows, dtype=np.float32)


class VisualEmbeddingTests(unittest.TestCase):
    def _catalog(self, folder: Path, *, count: int = 5) -> tuple[Path, Path, list[FrameRecord]]:
        keyframes = folder / "keyframes"
        records = []
        for index in range(count):
            video_id = "V1" if index < count - 1 else "V2"
            shot_id = f"s{index}"
            record = FrameRecord(
                video_id=video_id,
                local_idx=index if video_id == "V1" else 0,
                frame_id=index,
                pts_time=index / 10,
                shot_id=shot_id,
                keyframe_uid=100 + index,
            )
            records.append(record)
            image_path = keyframes / video_id / f"{shot_id}_{record.local_idx}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), (index, index, index)).save(image_path)
        catalog = folder / "index" / "frames.csv"
        write_frames_catalog_atomic(
            catalog,
            records=records,
            sources=[
                {
                    "video_id": video_id,
                    "plan_sha256": "a" * 64,
                    "quality_sha256": "b" * 64,
                    "quality_config_signature": "c" * 64,
                }
                for video_id in sorted({record.video_id for record in records})
            ],
        )
        return catalog, keyframes, records

    def test_distinct_process_token_resumes_without_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, records = self._catalog(root)
            output = root / "embeddings"
            first_encoder = FakeEncoder()
            with patch("offline.visual_embeddings._PROCESS_TOKEN", "process-one"):
                with self.assertRaises(IntentionalEmbeddingInterruption):
                    run_visual_embedding(
                        encoder=first_encoder,
                        modality="clip",
                        batch_id="dev-5",
                        catalog_path=catalog,
                        keyframes_root=keyframes,
                        output_root=output,
                        batch_size=2,
                        stop_after_shards=1,
                    )
            modality_dir = output / "dev-5" / "clip"
            self.assertFalse((modality_dir / "manifest.json").exists())
            partial = json.loads((modality_dir / "checkpoint.json").read_text())
            self.assertEqual(partial["next_index"], 2)
            self.assertFalse(partial["complete"])
            self.assertTrue(partial["intentional_interruption_observed"])

            resumed_encoder = FakeEncoder()
            with patch("offline.visual_embeddings._PROCESS_TOKEN", "process-two"):
                result = run_visual_embedding(
                    encoder=resumed_encoder,
                    modality="clip",
                    batch_id="dev-5",
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    output_root=output,
                    batch_size=2,
                )
            self.assertTrue(result.complete)
            self.assertTrue(result.resumed)
            self.assertTrue(result.checkpoint_resume_verified)
            self.assertEqual(len(resumed_encoder.encoded_paths), 3)

            shard_dirs = sorted((modality_dir / "shards").iterdir())
            self.assertEqual(len(shard_dirs), 3)
            uids = np.concatenate(
                [np.load(path / "keyframe_uids.npy", allow_pickle=False) for path in shard_dirs]
            )
            vectors = np.concatenate(
                [np.load(path / "vectors.npy", allow_pickle=False) for path in shard_dirs]
            )
            self.assertEqual(uids.tolist(), [record.keyframe_uid for record in records])
            self.assertEqual(len(set(uids.tolist())), len(records))
            self.assertEqual(vectors.dtype, np.float16)
            self.assertTrue(
                np.allclose(
                    np.linalg.norm(vectors.astype(np.float32), axis=1),
                    np.ones(len(records)),
                    atol=5e-3,
                    rtol=0,
                )
            )
            manifest = json.loads((modality_dir / "manifest.json").read_text())
            self.assertTrue(manifest["complete"])
            self.assertTrue(manifest["checkpoint_resume_verified"])
            self.assertEqual(manifest["modality"], "clip")
            self.assertEqual(manifest["runtime"], FakeEncoder.runtime_metadata)

            no_work_encoder = FakeEncoder()
            repeated = run_visual_embedding(
                encoder=no_work_encoder,
                modality="clip",
                batch_id="dev-5",
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_root=output,
                batch_size=2,
            )
            self.assertTrue(repeated.complete)
            self.assertEqual(no_work_encoder.encoded_paths, [])
            validated = validate_completed_visual_embedding(
                modality_dir,
                catalog_path=catalog,
                keyframes_root=keyframes,
                require_resume_verified=True,
            )
            self.assertTrue(validated.complete)
            self.assertTrue(validated.checkpoint_resume_verified)

    def test_modality_namespaces_do_not_wait_for_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root, count=2)
            encoder = FakeEncoder()
            result = run_visual_embedding(
                encoder=encoder,
                modality="clip",
                batch_id="batch-01",
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_root=root / "embeddings",
                batch_size=2,
            )
            self.assertTrue(result.complete)
            self.assertTrue((root / "embeddings" / "batch-01" / "clip" / "manifest.json").is_file())
            self.assertFalse((root / "embeddings" / "batch-01" / "siglip").exists())
            self.assertFalse((root / "embeddings" / "batch-01" / "beit3").exists())
            with self.assertRaisesRegex(RuntimeError, "has not been verified"):
                validate_completed_visual_embedding(
                    result.output_dir,
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    require_resume_verified=True,
                )

    def test_same_process_retry_does_not_claim_resume_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root, count=3)
            arguments = {
                "modality": "clip",
                "batch_id": "same-process",
                "catalog_path": catalog,
                "keyframes_root": keyframes,
                "output_root": root / "embeddings",
                "batch_size": 1,
            }
            with patch("offline.visual_embeddings._PROCESS_TOKEN", "same-process"):
                with self.assertRaises(IntentionalEmbeddingInterruption):
                    run_visual_embedding(
                        encoder=FakeEncoder(),
                        stop_after_shards=1,
                        **arguments,
                    )
                result = run_visual_embedding(encoder=FakeEncoder(), **arguments)
            self.assertTrue(result.complete)
            self.assertFalse(result.checkpoint_resume_verified)
            with self.assertRaisesRegex(RuntimeError, "has not been verified"):
                validate_completed_visual_embedding(
                    result.output_dir,
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    require_resume_verified=True,
                )

    def test_catalog_filter_and_missing_jpeg_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            selected = load_embedding_catalog(
                catalog, keyframes_root=keyframes, video_ids={"V2"}
            )
            self.assertEqual({item.frame.video_id for item in selected}, {"V2"})
            selected[0].image_path.unlink()
            with self.assertRaisesRegex(RuntimeError, "missing 1 keyframe"):
                load_embedding_catalog(catalog, keyframes_root=keyframes)

    def test_validator_rejects_missing_file_in_completed_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root, count=4)
            result = run_visual_embedding(
                encoder=FakeEncoder(),
                modality="clip",
                batch_id="missing-file",
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_root=root / "embeddings",
                batch_size=2,
            )
            self.assertTrue((result.output_dir / "manifest.json").is_file())
            (result.output_dir / "shards" / "000001" / "vectors.npy").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete visual embedding shard"):
                validate_completed_visual_embedding(
                    result.output_dir,
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                )

    def test_validator_diffs_actual_shard_uids_against_frames_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, records = self._catalog(root, count=4)
            result = run_visual_embedding(
                encoder=FakeEncoder(),
                modality="clip",
                batch_id="uid-corruption",
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_root=root / "embeddings",
                batch_size=2,
            )
            shard = result.output_dir / "shards" / "000001"
            uids_path = shard / "keyframe_uids.npy"
            uids = np.load(uids_path, allow_pickle=False)
            self.assertEqual(uids[0], records[2].keyframe_uid)
            uids[0] += 999
            np.save(uids_path, uids, allow_pickle=False)

            # Update the shard hash too, proving validation does not merely trust
            # the old final manifest or stop after an integrity-hash comparison.
            shard_manifest_path = shard / "manifest.json"
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            shard_manifest["uids_sha256"] = sha256_file(uids_path)
            shard_manifest_path.write_text(
                json.dumps(shard_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "shard UID mismatch"):
                validate_completed_visual_embedding(
                    result.output_dir,
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                )

    def test_bad_vectors_fail_before_publish(self):
        with self.assertRaisesRegex(RuntimeError, "NaN or Inf"):
            normalize_vectors(np.asarray([[np.nan, 1.0]]), expected_rows=1)
        with self.assertRaisesRegex(RuntimeError, "zero"):
            normalize_vectors(np.zeros((1, 3)), expected_rows=1)

    def test_beit3_config_is_explicitly_blocked_not_substituted_with_beit(self):
        clip = load_model_config("clip")
        siglip = load_model_config("siglip")
        self.assertEqual(clip["model_id"], "openai/clip-vit-base-patch32")
        self.assertEqual(len(clip["revision"]), 40)
        self.assertEqual(siglip["model_id"], "google/siglip-base-patch16-224")
        self.assertEqual(len(siglip["revision"]), 40)
        with self.assertRaisesRegex(RuntimeError, "official BEiT-3"):
            load_model_config("beit3")

    def test_model_revision_verifier_requires_exact_hugging_face_resolution(self):
        calls = []

        def exact_loader(model_id, *, revision):
            calls.append((model_id, revision))
            return SimpleNamespace(id=model_id, sha=revision)

        results = verify_visual_model_revisions(
            Path("configs/visual_embedding_models.json"),
            model_info_loader=exact_loader,
        )
        self.assertEqual([row["modality"] for row in results], ["clip", "siglip"])
        self.assertEqual(len(calls), 2)

        def mismatched_loader(model_id, *, revision):
            return SimpleNamespace(id=model_id, sha="f" * 40)

        with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
            verify_visual_model_revisions(
                Path("configs/visual_embedding_models.json"),
                model_info_loader=mismatched_loader,
            )


if __name__ == "__main__":
    unittest.main()
