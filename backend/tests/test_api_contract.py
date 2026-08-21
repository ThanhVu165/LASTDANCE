import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def _kis_rows():
    return [
        {
            "video_id": f"V{index:03d}",
            "frame_id": index * 10,
            "local_idx": index,
            "score": 1.0 - index / 1000,
        }
        for index in range(1, 101)
    ]


class SearchApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_kis_accepts_one_text_field_and_forces_top_100(self):
        with patch("app.routers.kis.run_kis_query", return_value=_kis_rows()) as search:
            response = self.client.post("/search/kis", json={"text": "một ô tô màu đỏ"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 100)
        search.assert_called_once_with("một ô tô màu đỏ", top_k=100)

    def test_health_exposes_runtime_and_vqa_preflight(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("python_executable", payload)
        self.assertIn("cuda_available", payload)
        self.assertIn("vqa_ready", payload)

    def test_exact_source_frame_endpoint_returns_jpeg(self):
        with patch("app.main._source_frame_jpeg", return_value=b"jpeg-bytes") as decode:
            response = self.client.get("/video/L01_V001/frame/123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(response.content, b"jpeg-bytes")
        decode.assert_called_once_with("L01_V001", 123)

    def test_exact_source_frame_endpoint_rejects_invalid_identifier(self):
        response = self.client.get("/video/bad.id/frame/123")

        self.assertEqual(response.status_code, 422)

    def test_qa_accepts_complete_query_and_forces_top_100(self):
        rows = [
            {**row, "answer": "đỏ"}
            for row in _kis_rows()
        ]
        complete_query = "Một xe chạy qua giao lộ. Câu hỏi: Xe có màu gì?"
        with patch("app.routers.qa.run_qa_query", return_value=rows) as search:
            response = self.client.post("/search/qa", json={"text": complete_query})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 100)
        search.assert_called_once_with(complete_query, top_k=100)

    def test_trake_accepts_complete_sequence_and_forces_top_100(self):
        rows = [
            {
                "video_id": f"V{index:03d}",
                "frame_ids": [10, 20, 30],
                "local_idxs": [1, 2, 3],
                "score": 1.0 - index / 1000,
            }
            for index in range(1, 101)
        ]
        complete_query = "(1) người chạy, (2) người nhảy, (3) người tiếp đất"
        with patch(
            "app.routers.trake.run_trake_query",
            return_value=(["người chạy", "người nhảy", "người tiếp đất"], rows),
        ) as search:
            response = self.client.post("/search/trake", json={"text": complete_query})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 100)
        self.assertEqual(len(response.json()["moments"]), 3)
        search.assert_called_once_with(complete_query, top_k=100)


if __name__ == "__main__":
    unittest.main()
