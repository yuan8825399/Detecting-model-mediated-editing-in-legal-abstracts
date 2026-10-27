import unittest

import numpy as np

from run_claw_edit_ordinal_robustness import monotone_probabilities, metric_dict


class OrdinalUtilityTests(unittest.TestCase):
    def test_reconstructs_valid_probabilities_from_nonmonotonic_thresholds(self):
        probabilities = monotone_probabilities(np.array([[0.2, 0.8, 0.5, 0.1], [1.2, -0.1, 0.4, 0.3]]))
        self.assertTrue(np.all(probabilities >= 0))
        self.assertTrue(np.allclose(probabilities.sum(axis=1), 1.0))
        self.assertTrue(np.allclose(probabilities[0], [0.8, 0.0, 0.0, 0.1, 0.1]))

    def test_metric_reports_perfect_predictions(self):
        y = np.array([0, 1, 2, 3, 4])
        probabilities = np.eye(5)
        metric = metric_dict(y, probabilities)
        self.assertEqual(metric["exact_accuracy"], 1.0)
        self.assertEqual(metric["macro_f1"], 1.0)
        self.assertEqual(metric["mae_levels"], 0.0)
        self.assertEqual(metric["expected_z_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
