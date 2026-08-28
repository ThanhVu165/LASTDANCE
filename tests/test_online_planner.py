import unittest
import json
from unittest.mock import patch

from pydantic import ValidationError

from online.planners import (
    DEFAULT_GEMINI_MODEL,
    GeminiQueryPlanner,
    RuleBasedQueryPlanner,
    _validate_provider_plan,
    get_query_planner,
)
from shared.schemas.online import SearchRequest, TaskType


class OnlinePlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = RuleBasedQueryPlanner()

    def test_trake_explicit_events_exclude_locator_context(self):
        query = (
            "Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng.\n"
            "E1 Khoảnh khắc đầu tiên hai con rồng vàng đang xoay vòng.\n"
            "E2 Khoảnh khắc con lân hoàn tất cú xoay người.\n"
            "E3 Khoảnh khắc dùi chạm vào kẻng đồng."
        )
        plan = self.planner.plan(query, TaskType.TRAKE)
        self.assertEqual(len(plan.ordered_moments), 3)
        self.assertTrue(plan.ordered_moments[0].startswith("Khoảnh khắc đầu tiên"))
        self.assertNotIn("Đoạn video", plan.ordered_moments[0])

    def test_visual_map_question_does_not_trigger_ocr_from_generic_table_word(self):
        query = (
            "Bản đồ phân bố động đất với một bảng chú giải nhiều màu sắc. "
            "Không tính bảng chú giải, có bao nhiêu vị trí động đất cấp độ 4?"
        )
        plan = self.planner.plan(query, TaskType.QA)
        self.assertEqual(plan.answer_source, "visual")
        self.assertEqual(plan.visible_text, [])

    def test_displayed_number_can_request_ocr(self):
        query = "Một con cá được đặt lên cân. Con số hiển thị cuối cùng trên cân là bao nhiêu?"
        plan = self.planner.plan(query, TaskType.QA)
        self.assertEqual(plan.answer_source, "visible_text")
        self.assertIn("ocr", plan.modality_weights)

    def test_displayed_fuel_price_uses_discriminative_literal_ocr_term(self):
        query = "Có thông tin về giá dầu mazut được hiển thị trong khung hình."
        plan = self.planner.plan(query, TaskType.KIS)
        self.assertEqual(plan.visible_text, ["giá dầu mazut"])
        self.assertEqual(plan.modality_weights, {"visual": 0.55, "ocr": 0.45})

        provider_plan = _validate_provider_plan(
            query,
            {
                "caption_en": "Information about mazut fuel prices is shown on screen.",
                "retrieval_queries": ["mazut fuel price information on screen"],
                "scenes": ["fuel price information shown on screen"],
                "visible_text": [],
            },
            "gemini",
            TaskType.KIS,
        )
        self.assertEqual(provider_plan.visible_text, ["giá dầu mazut"])

    def test_search_request_rejects_blank_and_more_than_100(self):
        with self.assertRaises(ValidationError):
            SearchRequest(task_type=TaskType.KIS, raw_query=" ")
        with self.assertRaises(ValidationError):
            SearchRequest(task_type=TaskType.KIS, raw_query="frame", max_results=101)

    def test_provider_payload_is_flattened_and_cannot_invent_text_intent(self):
        payload = {
            "caption_en": "A person wearing red opens a white car door.",
            "retrieval_queries": ["person in red opening white car door"],
            "scenes": [{"scene_caption": "A person wearing red opens a white car door."}],
            "anchor_moment_index": 0,
            "must_have": ["person", "red", "white car"],
            "visible_text": "A person wearing red opens a white car door.",
            "spoken_text": "",
            "modality_weights": {"image": 1.0, "text": 0.0},
            "question": "Who opens the door?",
            "answer_format": "text",
            "answer_source": "image",
            "ordered_moments": [{"moment_index": 0, "event": "opens a white car door"}],
        }
        plan = _validate_provider_plan(
            "Một người mặc áo đỏ đang mở cửa ô tô màu trắng.", payload, "qwen-local", TaskType.KIS
        )
        self.assertEqual(plan.scenes, ["A person wearing red opens a white car door."])
        self.assertEqual(plan.visible_text, [])
        self.assertEqual(plan.spoken_text, [])
        self.assertEqual(plan.modality_weights, {"visual": 1.0})
        self.assertIsNone(plan.question)
        self.assertIsNone(plan.answer_source)
        self.assertEqual(plan.ordered_moments, [])

    def test_provider_trake_event_objects_are_converted_to_ordered_strings(self):
        plan = _validate_provider_plan(
            "Đầu tiên mở cửa, sau đó ngồi vào xe.",
            {
                "caption_en": "First opens the door, then sits in the car.",
                "retrieval_queries": ["opens a car door"],
                "scenes": ["opens the door", "sits in the car"],
                "ordered_moments": [
                    {"moment_index": 0, "event": "opens the door"},
                    {"moment_index": 1, "event": "sits in the car"},
                ],
            },
            "qwen-local",
            TaskType.TRAKE,
        )
        self.assertEqual(plan.ordered_moments, ["opens the door", "sits in the car"])

    def test_provider_trake_single_moment_falls_back_to_explicit_raw_events(self):
        plan = _validate_provider_plan(
            "E1 Người đàn ông mở cửa.\nE2 Người đàn ông ngồi vào xe.",
            {
                "caption_en": "A man opens a door and then sits in the car.",
                "retrieval_queries": ["man opens door then sits in car"],
                "scenes": ["A man opens a door and then sits in the car."],
                "ordered_moments": ["A man opens a door and then sits in the car."],
            },
            "qwen-local",
            TaskType.TRAKE,
        )
        self.assertEqual(
            plan.ordered_moments,
            ["Người đàn ông mở cửa", "Người đàn ông ngồi vào xe"],
        )

    def test_provider_cannot_drop_explicit_visible_text_intent(self):
        raw = "Một con cá trên cân. Con số hiển thị cuối cùng là bao nhiêu?"
        plan = _validate_provider_plan(
            raw,
            {
                "caption_en": "A fish on a scale. What final number is displayed?",
                "retrieval_queries": ["fish on a scale"],
                "scenes": ["fish on a scale"],
                "visible_text": [],
                "answer_source": "visual",
            },
            "qwen-local",
            TaskType.QA,
        )
        self.assertEqual(plan.visible_text, [raw])
        self.assertEqual(plan.answer_source, "visible_text")
        self.assertEqual(plan.modality_weights, {"visual": 0.55, "ocr": 0.45})

    def test_gemini_key_is_sent_in_header_and_never_in_url(self):
        captured = {}
        provider_payload = {
            "caption_en": "person opening a door",
            "retrieval_queries": ["person opening a door"],
            "scenes": ["person opening a door"],
            "anchor_moment_index": 0,
        }

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [{"text": json.dumps(provider_payload)}]
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def _urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = {key.casefold(): value for key, value in request.header_items()}
            captured["timeout"] = timeout
            return _Response()

        with patch("online.planners.urllib.request.urlopen", side_effect=_urlopen):
            plan = GeminiQueryPlanner("secret-key", timeout=3.0).plan("mở cửa", TaskType.KIS)
        self.assertEqual(plan.planner_provider, "gemini")
        self.assertNotIn("secret-key", captured["url"])
        self.assertNotIn("?key=", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "secret-key")
        self.assertEqual(captured["timeout"], 3.0)

    def test_gemini_uses_current_default_and_preserves_explicit_model_override(self):
        self.assertEqual(GeminiQueryPlanner("secret-key").model, DEFAULT_GEMINI_MODEL)
        chain = get_query_planner(
            {
                "GEMINI_API_KEY": "secret-key",
                "AIC_GEMINI_MODEL": "gemini-3.5-flash",
                "AIC_TORCH_WORKER": "0",
            }
        )
        self.assertEqual(chain.providers[0].model, "gemini-3.5-flash")


if __name__ == "__main__":
    unittest.main()
