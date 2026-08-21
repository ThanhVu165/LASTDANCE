import unittest
from unittest.mock import patch

from app.services.query_processing import (
    extract_ocr_keywords,
    is_vietnamese_text,
    parse_qa_query,
    parse_semantic_query,
    split_trake_moments,
)
from app.services import query_translation


class QueryProcessingTests(unittest.TestCase):
    def test_language_detection_handles_accented_and_unaccented_vietnamese(self):
        self.assertTrue(is_vietnamese_text("một chiếc ô tô màu đỏ"))
        self.assertTrue(is_vietnamese_text("tim mot chiec xe mau do"))
        self.assertFalse(is_vietnamese_text("a red car on the street"))

    def test_vietnamese_semantic_query_never_becomes_mixed_language(self):
        query = parse_semantic_query("một chiếc ô tô màu đỏ")

        self.assertEqual(query.language, "vi")
        self.assertEqual(query.object_keywords, ["car"])
        self.assertEqual(query.ocr_keywords, [])
        self.assertTrue(all(" car" not in expansion for expansion in query.expansions))
        self.assertTrue(all("ô tô" in expansion for expansion in query.expansions))

    def test_long_multiscene_query_is_decomposed_without_losing_later_scenes(self):
        query = parse_semantic_query(
            "Đoạn video đưa hình ảnh một nhóm thiện nguyện đang chuẩn bị hàng trăm "
            "hộp cơm trong một căn bếp rất đông người. Nhóm người mặc áo màu xanh "
            "và đeo bao tay nilon khi làm việc. Một trong các phân cảnh sau đó cho "
            "thấy họ phát cơm chay miễn phí trước cổng chùa Chánh Thiên tại Bà Rịa."
        )

        self.assertGreaterEqual(len(query.expansions), 4)
        self.assertTrue(any("áo màu xanh" in clause for clause in query.expansions))
        self.assertTrue(any("cổng chùa Chánh Thiên" in clause for clause in query.expansions))
        self.assertGreaterEqual(len(query.scenes), 3)
        self.assertTrue(query.temporal_ordered)
        self.assertTrue(query.temporal_edges)
        self.assertTrue(
            any(
                "áo màu xanh" in expansion and "cổng chùa Chánh Thiên" in expansion
                for expansion in query.expansions
            )
        )

    def test_evidence_graph_keeps_unrelated_clauses_unordered(self):
        query = parse_semantic_query(
            "Một người mặc áo xanh đang đóng hộp cơm. "
            "Trên bàn có nhiều hộp xốp màu trắng. "
            "Phía sau là một bức tường màu vàng."
        )

        self.assertEqual(len(query.scenes), 3)
        self.assertEqual(query.temporal_edges, [])
        self.assertFalse(query.temporal_ordered)

    def test_object_and_ocr_are_auxiliary_explicit_signals(self):
        query = parse_semantic_query("a red car driving past a blue building")
        self.assertEqual(query.object_keywords, ["building", "car"])
        self.assertEqual(query.ocr_keywords, [])
        self.assertEqual(extract_ocr_keywords('biển hiệu ghi "BỆNH VIỆN HUẾ"'), ["BỆNH VIỆN HUẾ"])

    def test_ocr_extracts_unquoted_on_screen_text_but_not_metaphorical_quotes(self):
        self.assertEqual(
            extract_ocr_keywords("Cuối clip có dòng chữ coles màu trắng trên nền đỏ"),
            ["coles"],
        )
        self.assertEqual(
            extract_ocr_keywords('Nơi này được gọi là "thành phố của anh em Lumière"'),
            [],
        )

    def test_qa_accepts_complete_organizer_query_in_one_string(self):
        task = parse_qa_query(
            "Mô tả sự kiện: Một phụ nữ đứng cạnh xe. "
            "Câu hỏi: Chiếc xe có màu gì?"
        )

        self.assertEqual(task.retrieval_text, "Một phụ nữ đứng cạnh xe.")
        self.assertEqual(task.question, "Chiếc xe có màu gì?")

    def test_qa_question_clause_is_not_used_as_event_retrieval_text(self):
        task = parse_qa_query(
            "Một chiếc xe chạy trên đường. Câu hỏi: Chiếc xe có màu gì?"
        )

        self.assertEqual(task.retrieval_text, "Một chiếc xe chạy trên đường")
        self.assertEqual(task.question, "Chiếc xe có màu gì?")

    def test_visual_translation_adds_canonical_english_without_removing_source(self):
        query_translation.translate_visual_queries.cache_clear()

    def test_scene_translation_preserves_alignment_and_vietnamese_proper_names(self):
        query_translation.translate_visual_scenes.cache_clear()
        with patch.object(query_translation, "QUERY_TRANSLATION_ENABLED", True), patch.object(
            query_translation,
            "generate_text",
            return_value=(
                "S1=Volunteers in blue shirts prepare boxed meals\n"
                "S2=They distribute meals at Chánh Thiên Pagoda in Bà Rịa"
            ),
        ):
            translated = query_translation.translate_visual_scenes(
                (
                    "Những người áo xanh chuẩn bị hộp cơm",
                    "Họ phát cơm tại chùa Chánh Thiên ở Bà Rịa",
                )
            )

        self.assertEqual(len(translated), 2)
        self.assertIn("Volunteers in blue shirts", translated[0])
        self.assertIn("Bà Rịa", translated[1])
        query_translation.translate_visual_scenes.cache_clear()
        with patch.object(query_translation, "QUERY_TRANSLATION_ENABLED", True), patch.object(
            query_translation,
            "generate_text",
            return_value=(
                "A red open-top vehicle carrying elderly people\n"
                "A red convertible carrying older passengers\n"
                "An elderly group rides in a red convertible"
            ),
        ):
            expanded = query_translation.with_english_visual_expansion(
                "Một xe mui trần màu đỏ chở người lớn tuổi",
                ["Một xe mui trần màu đỏ chở người lớn tuổi"],
            )

        self.assertEqual(expanded[0], "Một xe mui trần màu đỏ chở người lớn tuổi")
        self.assertIn("A red convertible carrying older passengers", expanded)
        query_translation.translate_visual_queries.cache_clear()

    def test_qa_removes_repeated_directives_and_extracts_answer_format(self):
        task = parse_qa_query(
            "Hãy cho biết tên hãng (viết hoa bằng tiếng Anh và không có khoảng trắng): "
            "Một tòa lâu đài lớn nằm trên đỉnh núi. "
            "Hãy cho biết tên hãng (viết hoa bằng tiếng Anh và không có khoảng trắng): "
            "Lâu đài nằm tại vùng Bavaria của Đức."
        )

        self.assertNotIn("Hãy cho biết", task.retrieval_text)
        self.assertIn("lâu đài lớn", task.retrieval_text)
        self.assertTrue(task.answer_uppercase)
        self.assertTrue(task.answer_no_spaces)

    def test_qa_extracts_repeated_question_sentences_from_final_round_style(self):
        task = parse_qa_query(
            "Phút 40, cầu thủ áo đỏ sút vào khung thành do thủ môn áo xanh canh "
            "giữ. Số lượng cầu thủ áo trắng trong khu vực 16m50 là bao nhiêu? "
            "Đây là trận Hàn Quốc gặp Iraq. Số lượng cầu thủ áo trắng trong khu "
            "vực 16m50 ở thời điểm được mô tả là bao nhiêu?"
        )

        self.assertEqual(
            task.question,
            "Số lượng cầu thủ áo trắng trong khu vực 16m50 ở thời điểm được mô tả là bao nhiêu?",
        )
        self.assertNotIn("bao nhiêu?", task.retrieval_text)
        self.assertIn("Hàn Quốc gặp Iraq", task.retrieval_text)

    def test_qa_separates_unlabeled_how_many_question(self):
        task = parse_qa_query(
            "Miếng bánh được cuốn tròn rồi cắt nhỏ trước khi chiên. "
            "Hãy cho biết mỗi chiếc bánh được cắt làm mấy phần?"
        )

        self.assertEqual(
            task.retrieval_text,
            "Miếng bánh được cuốn tròn rồi cắt nhỏ trước khi chiên.",
        )
        self.assertEqual(
            task.question,
            "Hãy cho biết mỗi chiếc bánh được cắt làm mấy phần?",
        )

    def test_trake_splits_numbered_sequence_and_keeps_shared_context(self):
        moments = split_trake_moments(
            "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy: "
            "(1) giậm nhảy, (2) bay qua xà, (3) tiếp đất, (4) đứng dậy."
        )

        self.assertEqual(len(moments), 4)
        self.assertEqual(
            moments,
            [
                "vận động viên thực hiện cú nhảy, giậm nhảy",
                "vận động viên thực hiện cú nhảy, bay qua xà",
                "vận động viên thực hiện cú nhảy, tiếp đất",
                "vận động viên thực hiện cú nhảy, đứng dậy",
            ],
        )

    def test_trake_splits_final_round_e_markers(self):
        moments = split_trake_moments(
            "Trong một đoạn video nấu món thịt bò xào, xác định các khoảnh khắc "
            "đầu tiên nguyên liệu chạm vào chảo. E1: Dầu ăn. E2: Thịt bò. "
            "E3: Hành tây. E4: Bông so đũa."
        )

        self.assertEqual(len(moments), 4)
        self.assertTrue(moments[0].endswith("Dầu ăn"))
        self.assertTrue(moments[3].endswith("Bông so đũa"))

    def test_trake_rejects_no_sequence_at_pipeline_boundary(self):
        self.assertEqual(split_trake_moments("một người đang chạy"), ["một người đang chạy"])


if __name__ == "__main__":
    unittest.main()
