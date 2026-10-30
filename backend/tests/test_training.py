import unittest

import numpy as np
import pandas as pd

from models.train_xgboost import FEATURES, regression_metrics, split_dataset
from utils.rssi_features import MODEL_FEATURES


class TrainingTests(unittest.TestCase):
    def _dataset(self) -> pd.DataFrame:
        rows = []
        for wall_count in range(3):
            for sample in range(20):
                rows.append({
                    "distance": float(sample + 1),
                    "num_walls_intersected": wall_count,
                    "wall_attenuation_sum": float(wall_count * 4),
                    "tx_power_dbm": 20.0,
                    "frequency_ghz": 5.0,
                    "rssi_dbm": float(-40 - wall_count * 5 - sample / 10),
                })
        return pd.DataFrame(rows)

    def test_split_preserves_all_rows_and_model_feature_contract(self):
        dataset = self._dataset()
        x_train, x_validation, x_test, y_train, y_validation, y_test = split_dataset(dataset)

        self.assertListEqual(x_train.columns.tolist(), FEATURES)
        self.assertListEqual(FEATURES, MODEL_FEATURES)
        self.assertEqual(len(x_train), 42)
        self.assertEqual(len(x_validation), 9)
        self.assertEqual(len(x_test), 9)
        self.assertEqual(len(y_train) + len(y_validation) + len(y_test), len(dataset))
        self.assertEqual(set(x_train.index) & set(x_validation.index), set())
        self.assertEqual(set(x_train.index) & set(x_test.index), set())

    def test_regression_metrics_are_zero_for_perfect_predictions(self):
        actual = pd.Series([-60.0, -55.0, -50.0])
        metrics = regression_metrics(actual, np.array([-60.0, -55.0, -50.0]))

        self.assertEqual(metrics["rmse"], 0.0)
        self.assertEqual(metrics["mae"], 0.0)
        self.assertEqual(metrics["r2"], 1.0)


if __name__ == "__main__":
    unittest.main()
