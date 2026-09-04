import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ocr_v2_review_bundle.py"
SPEC = importlib.util.spec_from_file_location("ocr_v2_review_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frame(uid: int, *, status: str = "text_detected", regions=None):
    return {
        "keyframe_uid": uid,
        "status": status,
        "regions": regions if regions is not None else [],
    }


class OcrV2ReviewBundleTests(unittest.TestCase):
    def test_normalizes_text_for_repeat_triage(self):
        row = frame(
            1,
            regions=[
                {"easyocr_text": "Giá  dầu", "easyocr_confidence": 0.9},
                {"easyocr_text": "giá dầu", "easyocr_confidence": 0.8},
                {"easyocr_text": "khác", "easyocr_confidence": 0.2},
            ],
        )
        self.assertEqual(MODULE.duplicate_text_count(row), 1)
        self.assertEqual(MODULE.low_confidence_count(row), 1)

    def test_selection_is_uid_unique_and_deterministic(self):
        rows = [
            frame(index, status="no_text" if index < 10 else "text_detected", regions=[
                {"easyocr_text": "lặp" if index % 3 == 0 else str(index), "easyocr_confidence": 0.1 if index % 2 else 0.9},
                {"easyocr_text": "lặp" if index % 3 == 0 else "x", "easyocr_confidence": 0.8},
            ])
            for index in range(40)
        ]
        first = MODULE.choose_review_frames(rows, sample_size=20, seed="fixed")
        second = MODULE.choose_review_frames(rows, sample_size=20, seed="fixed")
        self.assertEqual([row["keyframe_uid"] for row in first], [row["keyframe_uid"] for row in second])
        self.assertEqual(len({row["keyframe_uid"] for row in first}), 20)
        self.assertEqual(len(first), 20)

    def test_recognition_selection_is_region_unique_and_capped_per_frame(self):
        rows = []
        for uid in range(20):
            regions = []
            for region_index in range(4):
                regions.append(
                    {
                        "region_id": f"{uid}-{region_index}",
                        "bbox_px": [0, 0, 20, 0, 20, 10, 0, 10],
                        "crop_width": 20,
                        "crop_height": 10 + region_index,
                        "easyocr_text": f"Giá {uid}" if region_index == 0 else "abc",
                        "easyocr_confidence": (region_index + 1) / 5,
                        "has_vi_marks": region_index == 0,
                    }
                )
            row = frame(uid, regions=regions)
            row["review_stratum"] = "random_control"
            rows.append(row)
        first = MODULE.choose_recognition_regions(rows, sample_size=30, seed="fixed")
        second = MODULE.choose_recognition_regions(rows, sample_size=30, seed="fixed")
        self.assertEqual(
            [item["region"]["region_id"] for item in first],
            [item["region"]["region_id"] for item in second],
        )
        self.assertEqual(len({item["region"]["region_id"] for item in first}), 30)
        counts = {}
        for item in first:
            counts[item["keyframe_uid"]] = counts.get(item["keyframe_uid"], 0) + 1
        self.assertLessEqual(max(counts.values()), 2)

    def test_immutable_region_hash_rejects_label_fields(self):
        row = {field: f"value-{field}" for field in MODULE.IMMUTABLE_REGION_FIELDS}
        first = MODULE.immutable_region_hash(row)
        row["human_text"] = "người gõ nhãn"
        row["notes"] = "được phép sửa"
        self.assertEqual(first, MODULE.immutable_region_hash(row))

    def test_archive_contract_does_not_claim_vintern_results(self):
        self.assertIn("vintern_results_available\": False", MODULE_PATH.read_text(encoding="utf-8"))
        self.assertIn("PENDING_HUMAN_VISUAL_REVIEW", MODULE_PATH.read_text(encoding="utf-8"))

    def test_balanced_selection_has_equal_video_counts(self):
        rows = []
        for video_id in ("V1", "V2"):
            for index in range(20):
                row = frame(
                    index + (100 if video_id == "V2" else 0),
                    status="no_text" if index < 3 else "text_detected",
                    regions=[{"easyocr_text": str(index), "easyocr_confidence": 0.5}],
                )
                row["video_id"] = video_id
                rows.append(row)
        selected = MODULE.choose_balanced_review_frames(
            rows, video_ids=["V1", "V2"], sample_size=40, seed="fixed"
        )
        counts = {}
        for row in selected:
            counts[row["video_id"]] = counts.get(row["video_id"], 0) + 1
        self.assertEqual(counts, {"V1": 20, "V2": 20})

    def test_balanced_recognition_selection_has_equal_video_counts(self):
        rows = []
        for video_id in ("V1", "V2"):
            for index in range(20):
                regions = [
                    {
                        "region_id": f"{video_id}-{index}-{region_index}",
                        "bbox_px": [0, 0, 20, 0, 20, 10, 0, 10],
                        "crop_width": 20,
                        "crop_height": 10,
                        "easyocr_text": str(index),
                        "easyocr_confidence": 0.5,
                        "has_vi_marks": False,
                    }
                    for region_index in range(2)
                ]
                row = frame(index + (100 if video_id == "V2" else 0), regions=regions)
                row["video_id"] = video_id
                rows.append(row)
        selected = MODULE.choose_balanced_recognition_regions(
            rows, video_ids=["V1", "V2"], sample_size=40, seed="fixed"
        )
        counts = {}
        for item in selected:
            video_id = item["frame"]["video_id"]
            counts[video_id] = counts.get(video_id, 0) + 1
        self.assertEqual(counts, {"V1": 20, "V2": 20})


if __name__ == "__main__":
    unittest.main()
