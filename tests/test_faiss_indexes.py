import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from offline.artifacts import sha256_file
from offline.catalog import write_frames_catalog_atomic
from offline.faiss_indexes import build_faiss_index, validate_faiss_index
from offline.visual_embeddings import IntentionalEmbeddingInterruption, run_visual_embedding
from shared.schemas.frame import FrameRecord


class FakeClipEncoder:
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

    def encode(self, image_paths):
        return np.asarray(
            [
                [float(index + 1), float(index + 2), 1.0, 2.0]
                for index, _ in enumerate(image_paths)
            ],
            dtype=np.float32,
        )


@unittest.skipUnless(importlib.util.find_spec("faiss"), "faiss-cpu is not installed")
class FaissIndexTests(unittest.TestCase):
    def _catalog(self, root: Path) -> tuple[Path, Path, list[FrameRecord]]:
        keyframes = root / "keyframes"
        records: list[FrameRecord] = []
        for video_index, video_id in enumerate(("V1", "V2")):
            for local_idx in range(3):
                absolute_index = video_index * 3 + local_idx
                shot_id = f"s{absolute_index}"
                record = FrameRecord(
                    video_id=video_id,
                    local_idx=local_idx,
                    frame_id=absolute_index * 10,
                    pts_time=float(absolute_index),
                    shot_id=shot_id,
                    keyframe_uid=100 + absolute_index,
                )
                records.append(record)
                image = keyframes / video_id / f"{shot_id}_{local_idx}.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (4, 4), (absolute_index,) * 3).save(image)
        catalog = root / "index" / "frames.csv"
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
                for video_id in ("V1", "V2")
            ],
        )
        return catalog, keyframes, records

    def _embedding(
        self,
        root: Path,
        *,
        catalog: Path,
        keyframes: Path,
        video_id: str,
        batch_id: str,
        batch_size: int = 1,
        verified: bool = True,
    ) -> Path:
        arguments = {
            "encoder": FakeClipEncoder(),
            "modality": "clip",
            "batch_id": batch_id,
            "catalog_path": catalog,
            "keyframes_root": keyframes,
            "output_root": root / "embeddings",
            "batch_size": batch_size,
            "video_ids": {video_id},
        }
        if not verified:
            return run_visual_embedding(**arguments).output_dir
        with patch(
            "offline.visual_embeddings._PROCESS_TOKEN", f"{batch_id}-process-one"
        ):
            with self.assertRaises(IntentionalEmbeddingInterruption):
                run_visual_embedding(**arguments, stop_after_shards=1)
        with patch(
            "offline.visual_embeddings._PROCESS_TOKEN", f"{batch_id}-process-two"
        ):
            return run_visual_embedding(**arguments).output_dir

    def test_builds_one_modality_and_incrementally_adds_disjoint_batch(self):
        import faiss

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, records = self._catalog(root)
            first = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01",
            )
            second = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V2",
                batch_id="batch-02",
            )
            index_path = root / "index" / "clip.faiss"

            initial = build_faiss_index(
                modality="clip",
                embedding_dirs=[first],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )
            self.assertEqual(initial.added_records, 3)
            self.assertEqual(initial.report.video_count, 1)
            self.assertFalse((root / "index" / "siglip.faiss").exists())
            self.assertFalse((root / "index" / "eva_clip.faiss").exists())

            updated = build_faiss_index(
                modality="clip",
                embedding_dirs=[second],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )
            self.assertEqual(updated.added_records, 3)
            self.assertEqual(updated.report.record_count, 6)
            self.assertEqual(updated.report.source_count, 2)
            index = faiss.read_index(str(index_path))
            self.assertEqual(
                set(faiss.vector_to_array(index.id_map).tolist()),
                {record.keyframe_uid for record in records},
            )

            repeated = build_faiss_index(
                modality="clip",
                embedding_dirs=[second],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )
            self.assertEqual(repeated.added_records, 0)
            self.assertEqual(repeated.report.record_count, 6)

    def test_rejects_unregistered_source_that_overlaps_existing_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            first = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01",
            )
            overlapping = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01-bs2",
                batch_size=2,
            )
            index_path = root / "index" / "clip.faiss"
            build_faiss_index(
                modality="clip",
                embedding_dirs=[first],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )
            with self.assertRaisesRegex(RuntimeError, "overlaps existing FAISS IDs"):
                build_faiss_index(
                    modality="clip",
                    embedding_dirs=[overlapping],
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    output_path=index_path,
                )

    def test_same_signature_with_different_vector_content_is_not_a_noop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            first = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01",
            )
            altered = root / "copied-artifact" / "clip"
            shutil.copytree(first, altered)
            shard = altered / "shards" / "000000"
            vectors_path = shard / "vectors.npy"
            vectors = np.load(vectors_path, allow_pickle=False)
            vectors[0] = np.roll(vectors[0], 1)
            np.save(vectors_path, vectors, allow_pickle=False)
            shard_manifest_path = shard / "manifest.json"
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            shard_manifest["vectors_sha256"] = sha256_file(vectors_path)
            shard_manifest_path.write_text(
                json.dumps(shard_manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            index_path = root / "index" / "clip.faiss"
            build_faiss_index(
                modality="clip",
                embedding_dirs=[first],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )
            with self.assertRaisesRegex(RuntimeError, "different provenance"):
                build_faiss_index(
                    modality="clip",
                    embedding_dirs=[altered],
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    output_path=index_path,
                )

    def test_validator_diffs_actual_index_ids_not_only_state_counts(self):
        import faiss

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            source = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01",
            )
            index_path = root / "index" / "clip.faiss"
            build_faiss_index(
                modality="clip",
                embedding_dirs=[source],
                catalog_path=catalog,
                keyframes_root=keyframes,
                output_path=index_path,
            )

            old = faiss.read_index(str(index_path))
            old_ids = faiss.vector_to_array(old.id_map).astype(np.int64)
            vectors = faiss.downcast_index(old.index).reconstruct_n(0, old.ntotal)
            old_ids[0] += 999
            corrupt = faiss.IndexIDMap(faiss.IndexFlatIP(old.d))
            corrupt.add_with_ids(vectors, old_ids)
            faiss.write_index(corrupt, str(index_path))

            state_path = index_path.with_name("clip.faiss.state.json")
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["index_sha256"] = sha256_file(index_path)
            state["keyframe_uid_set_sha256"] = hashlib.sha256(
                ("\n".join(str(value) for value in sorted(old_ids.tolist())) + "\n").encode()
            ).hexdigest()
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "keyframe_uid diff failed"):
                validate_faiss_index(index_path, catalog_path=catalog)

    def test_rejects_embedding_without_real_resume_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            source = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="unverified",
                batch_size=3,
                verified=False,
            )
            with self.assertRaisesRegex(RuntimeError, "has not been verified"):
                build_faiss_index(
                    modality="clip",
                    embedding_dirs=[source],
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    output_path=root / "index" / "clip.faiss",
                )

    def test_incomplete_index_state_pair_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, keyframes, _ = self._catalog(root)
            source = self._embedding(
                root,
                catalog=catalog,
                keyframes=keyframes,
                video_id="V1",
                batch_id="batch-01",
            )
            index_path = root / "index" / "clip.faiss"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.touch()
            with self.assertRaisesRegex(RuntimeError, "incomplete FAISS index/state pair"):
                build_faiss_index(
                    modality="clip",
                    embedding_dirs=[source],
                    catalog_path=catalog,
                    keyframes_root=keyframes,
                    output_path=index_path,
                )


if __name__ == "__main__":
    unittest.main()
