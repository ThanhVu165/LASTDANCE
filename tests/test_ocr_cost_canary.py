import json
import unittest

from offline.ocr_cost_canary import (
    build_crop_sheet_request_payload,
    build_request_payload,
    build_separate_payloads,
    expected_line_recall,
    make_synthetic_shot,
    parse_crop_sheet_results,
    parse_strict_results,
    strict_results_schema,
    validate_model_id,
)


class OcrCostCanaryTests(unittest.TestCase):
    def test_model_id_is_restricted_to_a_safe_path_segment(self):
        self.assertEqual(validate_model_id("gemini-2.5-flash-lite"), "gemini-2.5-flash-lite")
        with self.assertRaisesRegex(ValueError, "invalid Gemini model id"):
            validate_model_id("../../secret")

    def test_strict_schema_requires_three_results_and_eight_bbox_values(self):
        schema = strict_results_schema([10, 11, 12])
        results = schema["properties"]["results"]
        self.assertEqual(results["minItems"], 3)
        self.assertEqual(results["maxItems"], 3)
        bbox = results["items"]["properties"]["bbox"]["items"]
        self.assertEqual(bbox["minItems"], 8)
        self.assertEqual(bbox["maxItems"], 8)

    def test_request_strategies_use_expected_number_of_image_parts(self):
        shot = make_synthetic_shot(1)
        multi, ids = build_request_payload(shot, "multi_image")
        multi_payload = json.loads(multi)
        multi_parts = multi_payload["contents"][0]["parts"]
        self.assertEqual(ids, (10, 11, 12))
        self.assertEqual(sum("inlineData" in part for part in multi_parts), 3)

        crop, crop_ids, regions = build_crop_sheet_request_payload(shot)
        crop_payload = json.loads(crop)
        crop_parts = crop_payload["contents"][0]["parts"]
        self.assertEqual(crop_ids, tuple(frame.keyframe_uid for frame in shot.frames))
        self.assertEqual(sum("inlineData" in part for part in crop_parts), 1)
        self.assertEqual(len(regions), 9)
        self.assertEqual(
            crop_payload["generationConfig"]["mediaResolution"],
            "MEDIA_RESOLUTION_MEDIUM",
        )
        self.assertEqual(len(build_separate_payloads(shot)), 3)

    def test_crop_sheet_adapter_restores_detector_bboxes(self):
        shot = make_synthetic_shot(1)
        _, _, regions = build_crop_sheet_request_payload(shot)
        rows = [
            {
                "region_id": region.region_id,
                "text": region.expected_text,
                "confidence": 0.9,
                "language": "en" if "OPEN" in region.expected_text else "vi",
            }
            for region in regions
        ]
        response = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps({"regions": rows})}]}}
            ]
        }
        results, _ = parse_crop_sheet_results(response, regions)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(len(bbox) == 8 for result in results for bbox in result.bbox))
        self.assertEqual(results[0].bbox[0], list(shot.frames[0].regions[0].source_bbox))
        self.assertEqual(results[1].language, "mixed")

    def test_parser_rejects_duplicate_or_missing_frame_ids(self):
        row = {
            "frame_id": 10,
            "detected_text": ["CỬA HÀNG"],
            "bbox": [[0.1, 0.1, 0.9, 0.1, 0.9, 0.3, 0.1, 0.3]],
            "confidence": 0.9,
            "language": "vi",
        }
        response = {
            "candidates": [
                {"content": {"parts": [{"text": json.dumps({"results": [row, row]})}]}}
            ]
        }
        with self.assertRaisesRegex(ValueError, "duplicate frame_id"):
            parse_strict_results(response, [10, 11])

    def test_middle_only_propagation_loses_changed_text(self):
        shot = make_synthetic_shot(1)
        middle = shot.frames[1]
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": json.dumps(
                                    {
                                        "results": [
                                            {
                                                "frame_id": middle.frame_id,
                                                "detected_text": list(middle.expected_text),
                                                "bbox": [
                                                    [0.1, 0.1, 0.9, 0.1, 0.9, 0.2, 0.1, 0.2]
                                                ]
                                                * 3,
                                                "confidence": 1.0,
                                                "language": "mixed",
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                }
            ]
        }
        results, _ = parse_strict_results(response, [middle.frame_id])
        score = expected_line_recall(shot, results, propagate_single_result=True)
        self.assertLess(score["recall"], 1.0)
        self.assertEqual(score["expected_lines"], 9)


if __name__ == "__main__":
    unittest.main()
