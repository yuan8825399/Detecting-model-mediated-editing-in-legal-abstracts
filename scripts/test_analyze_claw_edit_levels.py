"""Checks for paired level analysis and duplicate/missing score handling."""

import unittest

import numpy as np

from analyze_claw_edit_levels import response, test_matrix


class LevelAnalysisTests(unittest.TestCase):
    def predictions(self):
        rows = [{"experiment": "pooled", "source_id": "one", "corpus_id": "H",
                 "score": -1.0, "detected": False}]
        for provider in ("deepseek", "doubao", "kimi", "wenxin"):
            for level in (1, 2, 3, 4):
                rows.append({"experiment": "pooled", "source_id": "one", "corpus_id": "E",
                             "provider": provider, "intervention_level": level,
                             "score": float(level), "detected": True})
        return rows

    def test_complete_source_scores_are_paired(self):
        sources, h, e, detected = test_matrix(self.predictions())
        self.assertEqual(sources, ["one"])
        self.assertEqual(h.tolist(), [-1.0])
        self.assertEqual(e.shape, (1, 4, 4))
        self.assertTrue(detected.all())

    def test_missing_level_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "17 rows per source"):
            test_matrix(self.predictions()[:-1])

    def test_nonmonotonic_response_counts_adjacent_decrease(self):
        scores = np.array([[[1.0, 3.0, 2.0, 4.0]]])
        metric = response(scores, np.array([0.0]), np.zeros((10, 1), dtype=int),
                          np.random.default_rng(1), 20)
        self.assertEqual(metric["strict_E1_to_E4_count"], 0)
        self.assertEqual(metric["adjacent"]["E2_to_E3"]["decrease_fraction"], 1.0)
        self.assertEqual(metric["strict_H_to_E4_count"], 0)


if __name__ == "__main__":
    unittest.main()
