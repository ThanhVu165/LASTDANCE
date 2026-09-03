import io
import unittest
import zipfile

from online.query_bundle import load_query_specs_from_zip
from shared.schemas.online import TaskType


def package(files: dict[str, str]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return payload.getvalue()


class QueryBundleTests(unittest.TestCase):
    def test_loads_natural_order_and_counts_trake_events(self):
        specs = load_query_specs_from_zip(
            package(
                {
                    "query-p1-10-kis.txt": "ten",
                    "query-p1-2-kis.txt": "two",
                    "query-p1-16-trake.txt": "E1 first\nE2 second\nE3 third",
                }
            )
        )
        self.assertEqual(
            [item.query_name for item in specs],
            ["query-p1-2-kis", "query-p1-10-kis", "query-p1-16-trake"],
        )
        self.assertEqual(specs[-1].task_type, TaskType.TRAKE)
        self.assertEqual(specs[-1].expected_event_count, 3)

    def test_rejects_nested_or_non_query_entries(self):
        with self.assertRaisesRegex(ValueError, "ZIP root"):
            load_query_specs_from_zip(package({"nested/query-p1-1-kis.txt": "x"}))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            load_query_specs_from_zip(package({"query-p1-1-kis.txt": "x", "readme.md": "x"}))


if __name__ == "__main__":
    unittest.main()
