import unittest
from types import SimpleNamespace

import numpy as np

from services.floorplan_coverage import (
    coverage_metrics,
    floorplan_walls_to_backend,
    grid_edges,
    heatmap_grid,
    load_prediction_artifacts,
    predict_heatmap,
)


class FloorplanCoverageTests(unittest.TestCase):
    def test_converts_detected_walls_to_backend_schema(self):
        walls = [SimpleNamespace(x1=1.0, y1=2.0, x2=3.0, y2=4.0)]

        converted = floorplan_walls_to_backend(walls, wall_loss_db=6.5)

        self.assertEqual(converted[0]["start"], {"x": 1.0, "y": 2.0})
        self.assertEqual(converted[0]["end"], {"x": 3.0, "y": 4.0})
        self.assertEqual(converted[0]["attenuation_db"], 6.5)

    def test_builds_sparse_heatmap_and_coverage_summary(self):
        x_values, y_values, grid = heatmap_grid(
            [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)], [-50.0, -70.0, -60.0]
        )

        np.testing.assert_array_equal(x_values, [0.0, 1.0])
        np.testing.assert_array_equal(y_values, [0.0, 1.0])
        self.assertTrue(np.isnan(grid[1, 1]))
        np.testing.assert_array_equal(grid_edges(x_values), [-0.5, 0.5, 1.5])
        summary = coverage_metrics([-50.0, -70.0, -60.0], minimum_rssi_dbm=-67.0)
        self.assertAlmostEqual(summary["coverage_percent"], 66.66666666666666)
        self.assertAlmostEqual(summary["dead_zone_percent"], 33.33333333333333)

    def test_trained_model_returns_rssi_and_uncertainty(self):
        artifacts = load_prediction_artifacts()
        prediction = predict_heatmap(
            [SimpleNamespace(x=2.0, y=2.0, channel=1)],
            [(3.0, 2.0), (5.0, 5.0)],
            [],
            artifacts,
            tx_power_dbm=20.0,
            frequency_ghz=5.0,
        )

        self.assertEqual(prediction.rssi_dbm.shape, (2,))
        self.assertTrue(np.isfinite(prediction.rssi_dbm).all())
        self.assertTrue(np.isfinite(prediction.uncertainty_db).all())


if __name__ == "__main__":
    unittest.main()
