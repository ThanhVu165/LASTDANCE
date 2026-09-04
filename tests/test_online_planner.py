import unittest
import json
from unittest.mock import patch

from pydantic import ValidationError

from online.planners import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_PLANNER_TIMEOUT_SECONDS,
    GeminiQueryPlanner,
    RuleBasedQueryPlanner,
    _validate_provider_plan,
    get_query_planner,
)
from shared.schemas.online import QueryRole, SearchRequest, TaskType


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
        self.assertEqual(plan.visible_text, [])
        self.assertEqual(plan.modality_weights, {"visual": 1.0})
        self.assertEqual(plan.answer_target.source, "ocr")
        self.assertEqual(plan.answer_target.value_type, "number")
        evidence = plan.units_by_id(plan.answer_target.evidence_unit_ids)[0]
        self.assertIn(QueryRole.ANSWER_EVIDENCE, evidence.roles)
        self.assertIn("ocr", evidence.modalities)

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
        self.assertEqual(plan.visible_text, [])
        self.assertEqual(plan.answer_source, "visible_text")
        self.assertEqual(plan.modality_weights, {"visual": 1.0})
        self.assertEqual(plan.answer_target.source, "ocr")

    def test_self_driving_car_qa_separates_locator_from_unknown_number(self):
        raw = (
            "Đoạn clip được quay từ bên trong một chiếc xe ô tô tự lái, có thể thấy rõ vô lăng "
            "được xoay để chiếc xe rẽ sang phải. Sau đó, góc quay chuyển ra ngoài, bắt trọn cảnh "
            "chiếc xe màu trắng rẽ trái, và ở góc trên khung hình có một biển hiệu đỏ gồm 6 ký tự "
            "chữ Hán. Con số được viết trên phần hông xe màu trắng là số mấy?"
        )
        plan = _validate_provider_plan(
            raw,
            {
                "global_context_en": "A self-driving car is shown from inside and outside.",
                "retrieval_queries": ["self-driving white car turning near a red Chinese sign"],
                "query_units": [
                    {
                        "unit_id": "inside",
                        "description_original": "Đoạn clip được quay từ bên trong một chiếc xe ô tô tự lái, có thể thấy rõ vô lăng được xoay để chiếc xe rẽ sang phải",
                        "retrieval_query_en": "inside a self-driving car with the steering wheel turning right",
                        "roles": ["VIDEO_LOCATOR"],
                    },
                    {
                        "unit_id": "outside",
                        "description_original": "chiếc xe màu trắng rẽ trái, và ở góc trên khung hình có một biển hiệu đỏ gồm 6 ký tự chữ Hán",
                        "retrieval_query_en": "a white car turns left below a red sign with six Chinese characters",
                        "roles": ["VIDEO_LOCATOR"],
                        "modalities": ["visual", "ocr"],
                        "visual_text_attributes": ["6 ký tự chữ Hán"],
                    },
                    {
                        "unit_id": "number",
                        "description_original": "Con số được viết trên phần hông xe màu trắng là số mấy?",
                        "retrieval_query_en": "the number written on the side of the white car",
                        "roles": ["VIDEO_LOCATOR", "TARGET_MOMENT", "ANSWER_EVIDENCE"],
                        "modalities": ["visual", "ocr"],
                        "known_text_literals": ["6 ký tự chữ Hán"],
                    },
                ],
                "submission_target_ids": ["number"],
                "answer_target": {
                    "question": "What number is on the car?",
                    "value_type": "number",
                    "source": "ocr",
                    "evidence_unit_ids": ["number"],
                    "value_is_unknown": True,
                },
                "negative_constraints": ["no people"],
            },
            "qwen-local",
            TaskType.QA,
        )
        self.assertEqual(plan.visible_text, [])
        self.assertEqual(plan.answer_target.value_type, "number")
        self.assertEqual(plan.answer_target.source, "ocr")
        self.assertEqual(plan.answer_target.evidence_unit_ids, ["number"])
        self.assertIn("negative constraints were removed", " ".join(plan.planner_warnings))
        self.assertNotIn("6 ký tự chữ Hán", plan.units_by_id(["number"])[0].known_text_literals)

    def test_role_aware_trake_never_counts_locator_as_event(self):
        plan = self.planner.plan(
            "Đoạn video bắt đầu bằng một con lân trắng.\nE1 Hai con rồng xoay vòng.\nE2 Dùi chạm kẻng đồng.",
            TaskType.TRAKE,
        )
        self.assertEqual(plan.ordered_event_ids, ["event-1", "event-2"])
        self.assertNotIn("locator-1", plan.ordered_event_ids)
        self.assertEqual(
            plan.units_by_id(["locator-1"])[0].roles,
            [QueryRole.VIDEO_LOCATOR],
        )

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
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response()

        with patch("online.planners.urllib.request.urlopen", side_effect=_urlopen):
            plan = GeminiQueryPlanner("secret-key", timeout=3.0).plan("mở cửa", TaskType.KIS)
        self.assertEqual(plan.planner_provider, "gemini")
        self.assertNotIn("secret-key", captured["url"])
        self.assertNotIn("?key=", captured["url"])
        self.assertEqual(captured["headers"]["x-goog-api-key"], "secret-key")
        self.assertIn("responseSchema", captured["body"]["generationConfig"])
        self.assertIn(
            "query_units",
            captured["body"]["generationConfig"]["responseSchema"]["properties"],
        )
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

    def test_gemini_planner_timeout_is_configurable(self):
        chain = get_query_planner(
            {
                "GEMINI_API_KEY": "secret-key",
                "AIC_GEMINI_PLANNER_TIMEOUT_SECONDS": "37.5",
                "AIC_TORCH_WORKER": "0",
            }
        )
        self.assertEqual(chain.providers[0].timeout, 37.5)
        default_chain = get_query_planner(
            {
                "GEMINI_API_KEY": "secret-key",
                "AIC_TORCH_WORKER": "0",
            }
        )
        self.assertEqual(
            default_chain.providers[0].timeout,
            DEFAULT_GEMINI_PLANNER_TIMEOUT_SECONDS,
        )

    def test_provider_english_question_is_preserved_for_qa(self):
        plan = _validate_provider_plan(
            "Con số trên biển báo là số mấy?",
            {
                "global_context_en": "A sign is visible.",
                "query_units": [
                    {
                        "unit_id": "unit-1",
                        "description_original": "Con số trên biển báo",
                        "retrieval_query_en": "the number on the sign",
                        "roles": ["TARGET_MOMENT", "ANSWER_EVIDENCE"],
                    }
                ],
                "answer_target": {
                    "question": "What number is on the sign?",
                    "value_type": "number",
                    "source": "visual",
                    "evidence_unit_ids": ["unit-1"],
                    "value_is_unknown": True,
                },
            },
            "gemini",
            TaskType.QA,
        )
        self.assertEqual(plan.question, "What number is on the sign?")


if __name__ == "__main__":
    unittest.main()
