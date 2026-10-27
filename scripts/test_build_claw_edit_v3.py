"""Regression checks for v3's explicit source-group exclusion policy."""

import csv
import io
import tempfile
import unittest
from pathlib import Path

from build_claw_edit_v3 import TRUNCATION_EXCLUSIONS, inspect_selection


class V3SelectionTests(unittest.TestCase):
    def make_rows(self):
        sources = sorted(TRUNCATION_EXCLUSIONS | {"keep-source"})
        v2 = []
        csv_rows = []
        for source_id in sources:
            for number in range(17):
                row = {
                    "record_id": f"{source_id}-{number}",
                    "source_id": source_id,
                    "corpus_id": "H" if number == 0 else "E",
                    "evaluation_role": "benchmark",
                    "abstract_clean": "未完" if source_id in TRUNCATION_EXCLUSIONS else "完整。",
                    "abstract_raw": "未完" if source_id in TRUNCATION_EXCLUSIONS else "完整。",
                }
                v2.append(row)
                csv_rows.append(dict(row))
        for number in range(17):
            csv_rows.append({
                "record_id": f"pilot-source-{number}",
                "source_id": "pilot-source",
                "corpus_id": "H" if number == 0 else "E",
                "evaluation_role": "pilot_only",
                "abstract_clean": "试运行。",
                "abstract_raw": "试运行。",
            })
        return v2, csv_rows

    def inspect(self, v2, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
            path.write_bytes(stream.getvalue().encode("gbk"))
            return inspect_selection(path, v2)

    def test_exactly_seven_reviewed_groups_are_removed(self):
        v2, rows = self.make_rows()
        selected, details = self.inspect(v2, rows)
        self.assertEqual(selected, {"keep-source"})
        self.assertEqual(details["csv_formal_source_count"], 8)

    def test_partial_group_is_rejected(self):
        v2, rows = self.make_rows()
        with self.assertRaisesRegex(ValueError, "incomplete source group"):
            self.inspect(v2, rows[:-1])

    def test_changed_record_identity_is_rejected(self):
        v2, rows = self.make_rows()
        rows[0]["record_id"] = "other-id"
        with self.assertRaisesRegex(ValueError, "Record identity changed"):
            self.inspect(v2, rows)


if __name__ == "__main__":
    unittest.main()
