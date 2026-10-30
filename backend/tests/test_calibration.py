import unittest

import numpy as np
import pandas as pd

from utils.calibration import (
    apply_residual_calibration,
    build_uncertainty_profile,
    fit_residual_calibration,
    recommend_survey_points,
    uncertainty_half_widths,
)
from utils.rssi_features import MODEL_FEATURES


class CalibrationTests(unittest.TestCase):
    def _features(self, wall_counts):
        return pd.DataFrame({
            "distance": np.linspace(2.0, 20.0, len(wall_counts)),
            "num_walls_intersected": wall_counts,
            "wall_attenuation_sum": np.asarray(wall_counts, dtype=float) * 4.0,
            "tx_power_dbm": 20.0,
            "frequency_ghz": 5.0,
        })[MODEL_FEATURES]

    def test_uncertainty_profile_uses_wall_specific_intervals_when_supported(self):
        features = self._features([0] * 30 + [2] * 30)
        predicted = np.full(60, -60.0)
        actual = np.concatenate([
            np.full(30, -58.0),
            np.full(30, -52.0),
        ])

        profile = build_uncertainty_profile(actual, predicted, features)
        half_widths = uncertainty_half_widths(features, profile)

        self.assertEqual(profile["confidence_level"], 0.9)
        self.assertIn("0", profile["by_wall_count"])
        self.assertIn("2", profile["by_wall_count"])
        self.assertLess(half_widths[0], half_widths[-1])

    def test_residual_calibration_reduces_known_field_bias(self):
        wall_counts = np.tile([0, 1, 2], 8)
        features = self._features(wall_counts)
        base_predictions = np.full(len(features), -60.0)
        observed = base_predictions + 2.0 + 1.5 * wall_counts

        calibration_model, report = fit_residual_calibration(
            features, observed, base_predictions
        )
        calibrated = apply_residual_calibration(
            features, base_predictions, calibration_model
        )

        self.assertGreater(report["base_mae_db"], 3.0)
        self.assertLess(report["calibrated_training_mae_db"], 0.1)
        self.assertLess(np.abs(observed - calibrated).mean(), 0.1)

    def test_recommendations_prioritize_uncertainty_and_remain_spatially_diverse(self):
        features = self._features([0, 2, 2])
        profile = {
            "global_half_width_db": 3.0,
            "by_wall_count": {"2": {"half_width_db": 8.0, "sample_count": 30}},
        }
        recommendations = recommend_survey_points(
            candidate_locations=[(0.0, 0.0), (1.0, 0.0), (8.0, 0.0)],
            candidate_features=features,
            predicted_rssi_dbm=[-45.0, -60.0, -62.0],
            uncertainty_profile=profile,
            max_recommendations=2,
            min_separation_m=3.0,
        )

        self.assertEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["num_walls_intersected"], 2)
        self.assertGreaterEqual(
            abs(recommendations[0]["x"] - recommendations[1]["x"]), 3.0
        )


if __name__ == "__main__":
    unittest.main()
