import tempfile
import os
import unittest
from pathlib import Path
from types import SimpleNamespace

from offline.config import DataLayout
from online.artifacts import CatalogFrame, FrameCatalog
from online.config import OnlineLayout
from online.encoders import WorkerTextEncoderRegistry
from online.planners import WorkerQwenQueryPlanner
from online.qwen_runtime import DEFAULT_QWEN_MODEL_ID, DEFAULT_QWEN_REVISION, resolve_qwen_revision
from online.vqa import WorkerQwenVQAAnswerer
from shared.schemas.online import FrameEvidence, TaskType, UnifiedQueryPlan


class _FakeClient:
    def __init__(self):
        self.calls = []

    def request(self, operation, **payload):
        self.calls.append((operation, payload))
        if operation == "encode":
            dimension = {"clip": 512, "siglip": 768, "eva_clip": 768}[payload["modality"]]
            return {"vectors": [[1.0] + [0.0] * (dimension - 1) for _ in payload["texts"]]}
        if operation == "qwen_plan":
            plan = UnifiedQueryPlan(
                raw_query=payload["text"],
                caption_en="person opening a door",
                retrieval_queries=["person opening a door"],
                scenes=["person opening a door"],
                anchor_moment_index=0,
                planner_provider="qwen-local",
            )
            return {"plan": plan.model_dump(mode="json")}
        if operation == "qwen_vqa":
            return {"answer": "red", "confidence": 0.9, "warnings": []}
        raise AssertionError(operation)


class OnlineTorchWorkerProxyTests(unittest.TestCase):
    def test_qwen_revision_is_pinned_and_custom_model_fails_without_revision(self):
        previous = os.environ.pop("AIC_QWEN_REVISION", None)
        try:
            self.assertEqual(resolve_qwen_revision(DEFAULT_QWEN_MODEL_ID), DEFAULT_QWEN_REVISION)
            with self.assertRaisesRegex(RuntimeError, "requires immutable"):
                resolve_qwen_revision("private/custom-qwen")
            os.environ["AIC_QWEN_REVISION"] = "abc123"
            self.assertEqual(resolve_qwen_revision("private/custom-qwen"), "abc123")
        finally:
            if previous is None:
                os.environ.pop("AIC_QWEN_REVISION", None)
            else:
                os.environ["AIC_QWEN_REVISION"] = previous

    def test_encoder_proxy_caches_by_revision_and_text(self):
        client = _FakeClient()
        registry = WorkerTextEncoderRegistry(client=client)
        first = registry.encode("clip", ["door"])
        second = registry.encode("clip", ["door"])
        self.assertEqual(first.shape, (1, 512))
        self.assertEqual(second.shape, (1, 512))
        self.assertEqual([call[0] for call in client.calls], ["encode"])

    def test_planner_proxy_preserves_online_contract(self):
        client = _FakeClient()
        plan = WorkerQwenQueryPlanner(client=client).plan("mở cửa", TaskType.KIS)
        self.assertEqual(plan.planner_provider, "qwen-local")
        self.assertEqual(plan.retrieval_queries, ["person opening a door"])
        self.assertEqual(client.calls[0][0], "qwen_plan")

    def test_vqa_proxy_sends_resolved_images_without_loading_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            internal = CatalogFrame(1, "v1", 0, 10, 1.0, "s1")
            image_path = internal.image_path(layout.data.keyframes)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpeg-placeholder")
            registry = SimpleNamespace(
                catalog=FrameCatalog([internal], sha256="catalog"),
                layout=layout,
            )
            frame = FrameEvidence(
                keyframe_uid=1,
                video_id="v1",
                frame_id=10,
                pts_time=1.0,
                shot_id="s1",
            )
            client = _FakeClient()
            answer, confidence, warnings = WorkerQwenVQAAnswerer(
                registry, "Qwen/Qwen3-VL-2B-Instruct", client=client
            ).answer(video_id="v1", frames=[frame], question="color?")
            self.assertEqual((answer, confidence, warnings), ("red", 0.9, []))
            operation, payload = client.calls[0]
            self.assertEqual(operation, "qwen_vqa")
            self.assertEqual(Path(payload["frames"][0]["image_path"]), image_path.resolve())


if __name__ == "__main__":
    unittest.main()
