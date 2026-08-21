"""Export module — turns internal submission rows into the exact CSV format required
by the contest submission system (https://sotuyenaic.oj.io.vn/):

  - UTF-8, comma-delimited, NO header row.
  - Video name WITHOUT the .mp4 extension.
  - Frame id as an integer.
  - Q&A answers are quoted with `"..."` only when they contain a comma, a double
    quote, or a newline (double quotes inside are escaped by doubling them, i.e.
    the standard CSV quoting rule implemented by Python's csv module).
  - Max 100 rows per file; Q&A answers max 100 characters; TRAKE frame count must
    match the number of requested events.

Also provides a validator (`validate_rows`) so obviously-broken submissions are
caught locally before spending one of the contest's limited 3 submission attempts.
"""
import csv
import io
import zipfile
from pathlib import Path

from app.config import MAX_ANSWER_LENGTH, MAX_SUBMISSION_ROWS
from app.models import SubmissionRow


def validate_rows(rows: list[SubmissionRow], expected_trake_events: int | None = None) -> list[str]:
    errors: list[str] = []
    if not rows:
        errors.append("Không có dòng nào để xuất.")
        return errors

    if len(rows) > MAX_SUBMISSION_ROWS:
        errors.append(f"Có {len(rows)} dòng, vượt quá giới hạn {MAX_SUBMISSION_ROWS}.")

    query_type = rows[0].query_type
    for i, r in enumerate(rows, start=1):
        if r.query_type != query_type:
            errors.append(f"Dòng {i}: query_type không đồng nhất trong cùng 1 file.")
        if not r.video_id:
            errors.append(f"Dòng {i}: thiếu video_id.")
        if r.video_id.lower().endswith(".mp4"):
            errors.append(f"Dòng {i}: video_id không được có đuôi .mp4 ('{r.video_id}').")
        if not r.frame_ids:
            errors.append(f"Dòng {i}: thiếu frame_id.")

        if query_type == "kis" and len(r.frame_ids) != 1:
            errors.append(f"Dòng {i}: KIS chỉ cần đúng 1 frame_id.")
        if query_type == "qa":
            if len(r.frame_ids) != 1:
                errors.append(f"Dòng {i}: Q&A chỉ cần đúng 1 frame_id.")
            if not r.answer:
                errors.append(f"Dòng {i}: Q&A thiếu answer.")
            elif len(r.answer) > MAX_ANSWER_LENGTH:
                errors.append(f"Dòng {i}: answer dài {len(r.answer)} ký tự, vượt quá {MAX_ANSWER_LENGTH}.")
        if query_type == "trake" and expected_trake_events:
            if len(r.frame_ids) != expected_trake_events:
                errors.append(
                    f"Dòng {i}: TRAKE cần đúng {expected_trake_events} frame_id, "
                    f"nhận được {len(r.frame_ids)}."
                )
            if sorted(r.frame_ids) != r.frame_ids:
                errors.append(f"Dòng {i}: frame_id của TRAKE phải theo đúng thứ tự thời gian tăng dần.")

    return errors


def rows_to_csv_text(rows: list[SubmissionRow]) -> str:
    """csv.writer already implements the exact quoting rule required: quote a field
    only when it contains the delimiter, a quote char, or a newline (QUOTE_MINIMAL),
    and escape embedded quotes by doubling them."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, delimiter=",", lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)

    query_type = rows[0].query_type if rows else "kis"
    for r in rows:
        if query_type == "kis":
            writer.writerow([r.video_id, r.frame_ids[0]])
        elif query_type == "qa":
            writer.writerow([r.video_id, r.frame_ids[0], r.answer or ""])
        else:  # trake
            writer.writerow([r.video_id, *r.frame_ids])

    return buf.getvalue()


def build_submission_zip(csv_files: dict[str, str]) -> bytes:
    """csv_files: {filename (e.g. 'query-1-kis.csv'): csv text}. Packs everything
    under a top-level `submission/` folder inside the zip, per the contest's
    required structure."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in csv_files.items():
            zf.writestr(f"submission/{filename}", content.encode("utf-8"))
    return buf.getvalue()
