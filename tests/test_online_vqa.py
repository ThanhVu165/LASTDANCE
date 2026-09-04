import tempfile
import unittest
from pathlib import Path

from PIL import Image

from offline.config import DataLayout
from online.artifacts import ArtifactRegistry, CatalogFrame, FrameCatalog
from online.config import OnlineLayout
from online.vqa import QwenVQAAnswerer, _answers_agree
from shared.schemas.online import FrameEvidence


class OnlineVqaTests(unittest.TestCase):
    def test_lexical_similarity_does_not_authorize_unknown_answers(self):
        self.assertFalse(_answers_agree("Không xác định được", "Chưa xác định được"))

    def test_fuzzy_agreement_rejects_different_numbers(self):
        self.assertFalse(_answers_agree("3 chiếc bánh", "5 chiếc bánh"))

    def test_fuzzy_agreement_rejects_different_colors(self):
        self.assertFalse(_answers_agree("đỏ", "xanh"))

    def test_contact_sheet_preserves_information_at_image_borders(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            frame = CatalogFrame(1, "v1", 0, 0, 0., "s0")
            path = frame.image_path(layout.data.keyframes)
            path.parent.mkdir(parents=True)
            image = Image.new("RGB", (400, 100), "blue")
            image.paste("red", (0, 0, 40, 100))
            image.paste("green", (360, 0, 400, 100))
            image.save(path)
            registry = ArtifactRegistry(layout=layout, catalog=FrameCatalog([frame], sha256="test"), visual={}, statuses={})
            evidence = FrameEvidence(keyframe_uid=1, video_id="v1", frame_id=0, pts_time=0., shot_id="s0")
            sheet, _ = QwenVQAAnswerer(registry)._contact_sheet([evidence])
            try:
                self.assertGreater(sheet.getpixel((5, 112))[0], 200)
                self.assertGreater(sheet.getpixel((218, 112))[1], 80)
                self.assertEqual(sheet.getpixel((112, 5)), (255, 255, 255))
            finally:
                sheet.close()

    def test_contact_sheet_is_one_six_panel_chronological_image(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            frames = []
            evidence = []
            for index in range(6):
                frame = CatalogFrame(index + 1, "v1", index, index * 10, float(5 - index), f"s{index}")
                frames.append(frame)
                image_path = frame.image_path(layout.data.keyframes)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (320, 180), (index * 20, 40, 80)).save(image_path)
                evidence.append(
                    FrameEvidence(
                        keyframe_uid=frame.keyframe_uid,
                        video_id=frame.video_id,
                        frame_id=frame.frame_id,
                        pts_time=frame.pts_time,
                        shot_id=frame.shot_id,
                    )
                )
            catalog = FrameCatalog(frames, sha256="catalog")
            registry = ArtifactRegistry(layout=layout, catalog=catalog, visual={}, statuses={})
            sheet, included = QwenVQAAnswerer(registry)._contact_sheet(evidence)
            try:
                self.assertEqual(sheet.size, (672, 496))
                self.assertEqual([item.pts_time for item in included], sorted(item.pts_time for item in evidence))
            finally:
                sheet.close()


if __name__ == "__main__":
    unittest.main()
