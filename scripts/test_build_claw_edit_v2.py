"""Regression checks for reviewed-CSV source-group exclusion handling."""

from __future__ import annotations

import csv
import io
import unittest

from build_claw_edit_v2 import inspect_reviewed_csv, normalize


def fixture() -> tuple[list[dict], list[dict]]:
    processed = []
    for source_id in ("source-a", "source-b"):
        for number in range(17):
            record_id = f"{source_id}-{number}"
            processed.append({
                "record_id": record_id, "source_id": source_id, "group_id": source_id,
                "parent_record_id": "", "corpus_id": "H" if number == 0 else "E",
                "operation": "original" if number == 0 else "edit", "provider": "",
                "model_name": "", "model_snapshot": "", "prompt_id": "",
                "group_split": "train", "source_year": "2020", "evaluation_role": "benchmark",
                "source_title_raw": "标题 附视频", "source_title_clean": "标题",
                "title_sha256": "title-hash", "input_sha256": "input-hash",
                "output_sha256": "output-hash", "quality_status": "accepted",
                "quality_reason": "", "raw_evidence_sha256": "",
                "abstract_raw": "<正> 摘要", "abstract_clean": "<正> 摘要",
            })
    reviewed = [dict(row, source_title_raw="标题", abstract_raw="摘要", abstract_clean="摘要")
                for row in processed if row["source_id"] == "source-a"]
    return processed, reviewed


def csv_bytes(rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("gbk")


class ReviewedCsvTests(unittest.TestCase):
    def test_complete_source_exclusion(self) -> None:
        processed, reviewed = fixture()
        excluded, details = inspect_reviewed_csv(csv_bytes(reviewed), processed)
        self.assertEqual(excluded, {"source-b"})
        self.assertEqual(details["excluded_record_count"], 17)
        self.assertEqual(details["csv_marker_cleaned_count"], 17)

    def test_partial_source_exclusion_is_rejected(self) -> None:
        processed, reviewed = fixture()
        with self.assertRaisesRegex(ValueError, "Partial source groups"):
            inspect_reviewed_csv(csv_bytes(reviewed[:-1]), processed)

    def test_unexpected_text_change_is_rejected(self) -> None:
        processed, reviewed = fixture()
        reviewed[0]["abstract_clean"] = "不同内容"
        with self.assertRaisesRegex(ValueError, "Unexpected text edit"):
            inspect_reviewed_csv(csv_bytes(reviewed), processed)

    def test_all_markers_are_removed(self) -> None:
        self.assertEqual(normalize("<正> 摘要 <正>"), "摘要")
        self.assertEqual(normalize("正文 <正>"), "正文")
        self.assertEqual(normalize("没有标记 "), "没有标记 ")


if __name__ == "__main__":
    unittest.main()
