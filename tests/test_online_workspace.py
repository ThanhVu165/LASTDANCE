import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from offline.config import DataLayout
from online.config import OnlineLayout
from online.workspace import SubmissionWorkspace, sanitize_component, sanitize_zip_name
from shared.schemas.online import (
    FrameEvidence,
    KISCandidate,
    QACandidate,
    QuerySpec,
    TaskType,
    TrakeCandidate,
)


def evidence(video: str, frame_id: int, pts: float, uid: int) -> FrameEvidence:
    return FrameEvidence(
        keyframe_uid=uid,
        video_id=video,
        frame_id=frame_id,
        pts_time=pts,
        shot_id=f"s{uid}",
        final_score=0.9,
    )


def spec(name: str, task: TaskType, events: int | None = None, raw: str = "query") -> QuerySpec:
    return QuerySpec(
        query_name=name,
        source_filename=f"{name}.txt",
        task_type=task,
        raw_query=raw,
        expected_event_count=events,
    )


def write_catalog(layout: OnlineLayout, rows: list[tuple[str, int, float]]) -> None:
    layout.catalog.parent.mkdir(parents=True, exist_ok=True)
    with layout.catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "frame_id", "pts_time"])
        writer.writeheader()
        for video_id, frame_id, pts_time in rows:
            writer.writerow(
                {"video_id": video_id, "frame_id": frame_id, "pts_time": pts_time}
            )


class OnlineWorkspaceTests(unittest.TestCase):
    def test_sanitize_removes_path_traversal(self):
        self.assertEqual(sanitize_component("../../my results", fallback="x"), "my-results")
        self.assertEqual(sanitize_zip_name("../../team round_1.zip"), "teamround1.zip")

    def test_kis_csv_has_no_header_and_zip_has_submission_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(layout, [("L01_V001", 42, 1.4)])
            query = spec("query-p1-1-kis", TaskType.KIS)
            item = KISCandidate(
                video_id="L01_V001",
                frame_id=42,
                score=0.9,
                evidence=evidence("L01_V001", 42, 1.4, 1),
            )
            workspace = SubmissionWorkspace(
                folder_name="../../round 1",
                expected_queries=[query],
                layout=layout,
                query_drafts={query.query_name: [item]},
                query_history=[{"request": {"raw_query": "find it"}}],
                provenance={"catalog_sha256": "abc"},
            )
            state_path = workspace.save()
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["schema_version"], 2)
            self.assertEqual(state["profile"], "AIC26_QUALIFIER_OFFICIAL")
            csv_path = workspace.export_csv(query.query_name)
            self.assertEqual(csv_path.read_bytes(), b"L01_V001,42\n")
            report = workspace.export_zip(zip_name="team round 1")
            with zipfile.ZipFile(report.zip_path) as archive:
                self.assertEqual(archive.namelist(), ["submission/query-p1-1-kis.csv"])
                self.assertEqual(archive.read(archive.namelist()[0]), b"L01_V001,42\n")

    def test_qa_uses_utf8_and_csv_quoting_without_trimming(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(layout, [("L01_V028", 3450, 10.0)])
            query = spec("query-p1-3-qa", TaskType.QA)
            qa = QACandidate(
                video_id="L01_V028",
                frame_id=3450,
                answer=' Màu "đỏ",\nrất đẹp ',
                score=0.8,
                confidence=0.9,
                requires_review=False,
                evidence=evidence("L01_V028", 3450, 10.0, 7),
            )
            workspace = SubmissionWorkspace(
                folder_name="qa",
                expected_queries=[query],
                layout=layout,
                query_drafts={query.query_name: [qa]},
            )
            payload = workspace.csv_bytes(query.query_name)
            self.assertFalse(payload.startswith(b"\xef\xbb\xbf"))
            rows = list(csv.reader(io.StringIO(payload.decode("utf-8"), newline="")))
            self.assertEqual(rows, [["L01_V028", "3450", ' Màu "đỏ",\nrất đẹp ']])
            self.assertNotIn(b"video_id", payload)

    def test_trake_requires_exact_event_count_and_catalog_time_order(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(
                layout,
                [("L10_V001", 1200, 1.0), ("L10_V001", 1850, 2.0), ("L10_V001", 2100, 3.0)],
            )
            query = spec("query-p1-16-trake", TaskType.TRAKE, events=3)
            frames = [
                evidence("L10_V001", 1200, 1.0, 1),
                evidence("L10_V001", 1850, 2.0, 2),
                evidence("L10_V001", 2100, 3.0, 3),
            ]
            valid = TrakeCandidate(
                video_id="L10_V001",
                frame_ids=[1200, 1850, 2100],
                pts_times=[1.0, 2.0, 3.0],
                score=0.8,
                evidence=frames,
            )
            workspace = SubmissionWorkspace(
                folder_name="trake",
                expected_queries=[query],
                layout=layout,
                query_drafts={query.query_name: [valid]},
            )
            self.assertEqual(
                workspace.csv_bytes(query.query_name),
                b"L10_V001,1200,1850,2100\n",
            )
            with self.assertRaisesRegex(ValueError, "exactly 3"):
                workspace.replace_query_draft(
                    query.query_name,
                    [
                        TrakeCandidate(
                            video_id="L10_V001",
                            frame_ids=[1200, 1850],
                            pts_times=[1.0, 2.0],
                            score=0.5,
                            evidence=frames[:2],
                        )
                    ],
                )

    def test_duplicate_placeholder_header_and_incomplete_bundle_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(layout, [("v1", 1, 1.0), ("v1", 2, 2.0)])
            kis_spec = spec("query-p1-1-kis", TaskType.KIS)
            qa_spec = spec("query-p1-3-qa", TaskType.QA)
            item = KISCandidate(
                video_id="v1",
                frame_id=1,
                score=1.0,
                evidence=evidence("v1", 1, 1.0, 1),
            )
            workspace = SubmissionWorkspace(
                folder_name="bundle",
                expected_queries=[kis_spec, qa_spec],
                layout=layout,
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                workspace.replace_query_draft(kis_spec.query_name, [item, item])
            with self.assertRaisesRegex(ValueError, "header"):
                workspace.validate_csv_bytes(kis_spec, b"video_id,frame_id\nv1,1\n")
            uncertain = QACandidate(
                video_id="v1",
                frame_id=2,
                answer="Uncertain",
                score=0.5,
                confidence=0.0,
                evidence=evidence("v1", 2, 2.0, 2),
            )
            with self.assertRaisesRegex(ValueError, "placeholder"):
                workspace.replace_query_draft(qa_spec.query_name, [uncertain])
            needs_review = uncertain.model_copy(update={"answer": "42"})
            with self.assertRaisesRegex(ValueError, "requires operator review"):
                workspace.replace_query_draft(qa_spec.query_name, [needs_review])
            workspace.replace_query_draft(kis_spec.query_name, [item])
            with self.assertRaisesRegex(ValueError, "missing submission rows"):
                workspace.export_zip()

    def test_two_queries_with_identical_content_remain_two_csv_files(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(layout, [("v1", 1, 1.0)])
            specs = [
                spec("query-p1-8-kis", TaskType.KIS, raw="same"),
                spec("query-p1-14-kis", TaskType.KIS, raw="same"),
            ]
            item = KISCandidate(
                video_id="v1",
                frame_id=1,
                score=1.0,
                evidence=evidence("v1", 1, 1.0, 1),
            )
            workspace = SubmissionWorkspace(
                folder_name="duplicates",
                expected_queries=specs,
                layout=layout,
                query_drafts={value.query_name: [item] for value in specs},
            )
            report = workspace.export_zip()
            with zipfile.ZipFile(report.zip_path) as archive:
                self.assertEqual(
                    archive.namelist(),
                    [
                        "submission/query-p1-8-kis.csv",
                        "submission/query-p1-14-kis.csv",
                    ],
                )

    def test_more_than_100_rows_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(
                layout,
                [("v1", frame_id, float(frame_id)) for frame_id in range(1, 102)],
            )
            query = spec("query-p1-1-kis", TaskType.KIS)
            candidates = [
                KISCandidate(
                    video_id="v1",
                    frame_id=frame_id,
                    score=1.0,
                    evidence=evidence("v1", frame_id, float(frame_id), frame_id),
                )
                for frame_id in range(1, 102)
            ]
            workspace = SubmissionWorkspace(
                folder_name="too-many",
                expected_queries=[query],
                layout=layout,
            )
            with self.assertRaisesRegex(ValueError, "1-100"):
                workspace.replace_query_draft(query.query_name, candidates)

    def test_invalid_video_float_frame_and_zip_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = OnlineLayout(DataLayout(Path(directory)))
            write_catalog(layout, [("v1", 1, 1.0)])
            query = spec("query-p1-1-kis", TaskType.KIS)
            item = KISCandidate(
                video_id="v1",
                frame_id=1,
                score=1.0,
                evidence=evidence("v1", 1, 1.0, 1),
            )
            workspace = SubmissionWorkspace(
                folder_name="negative-format",
                expected_queries=[query],
                layout=layout,
                query_drafts={query.query_name: [item]},
            )
            with self.assertRaisesRegex(ValueError, "invalid video_id"):
                workspace.validate_csv_bytes(query, b"v1.mp4,1\n")
            with self.assertRaisesRegex(ValueError, "non-integer"):
                workspace.validate_csv_bytes(query, b"v1,1.0\n")

            bad_zip = Path(directory) / "root.zip"
            with zipfile.ZipFile(bad_zip, "w") as archive:
                archive.writestr(query.csv_filename, workspace.csv_bytes(query.query_name))
            with self.assertRaisesRegex(ValueError, "under submission/"):
                workspace.validate_zip_path(bad_zip)


if __name__ == "__main__":
    unittest.main()
