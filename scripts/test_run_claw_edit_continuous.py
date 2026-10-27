"""Small deterministic checks for continuous z label semantics."""

import unittest

import numpy as np

from run_claw_edit_continuous import balanced_weights, metric_dict, z_of


class ContinuousLabelTests(unittest.TestCase):
    def test_operation_intensity_mapping(self):
        self.assertEqual(z_of({"record_id": "h", "corpus_id": "H", "intervention_level": 0}), 0.0)
        for level, expected in ((1, 0.25), (2, 0.50), (3, 0.75), (4, 1.0)):
            self.assertEqual(z_of({"record_id": f"e{level}", "corpus_id": "E",
                                   "intervention_level": level}), expected)

    def test_invalid_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid intervention level"):
            z_of({"record_id": "bad", "corpus_id": "E", "intervention_level": 0})

    def test_label_balance_weights(self):
        weights = balanced_weights(np.array([0.0, 0.25, 0.25]))
        self.assertAlmostEqual(weights[0], 1.5)
        self.assertAlmostEqual(weights[1], 0.75)
        self.assertAlmostEqual(weights[2], 0.75)

    def test_constant_target_has_no_rank_or_linear_correlation(self):
        metrics = metric_dict(np.array([0.0, 0.0]), np.array([0.1, 0.2]))
        self.assertIsNone(metrics["pearson_r"])
        self.assertIsNone(metrics["spearman_rho"])


if __name__ == "__main__":
    unittest.main()
