import unittest

from api.app import wall_features as api_wall_features
from optimizer.objective_functions import wall_features as optimizer_wall_features
from utils.feature_extractor import extract_features
from utils.rssi_features import MODEL_FEATURES, build_rssi_features, feature_frame


class SharedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.walls = [
            {
                "start": {"x": 3, "y": 0},
                "end": {"x": 3, "y": 10},
                "material": "drywall",
            },
            {
                "start": {"x": 6, "y": 0},
                "end": {"x": 6, "y": 10},
                "material": "brick",
            },
        ]

    def test_all_consumers_use_the_same_multi_wall_features(self):
        ap = (1, 5)
        receiver = (9, 5)
        expected = (2, 12.0)

        self.assertEqual(
            extract_features(ap, receiver, {"walls": self.walls}),
            {
                "distance": 8.0,
                "num_walls_intersected": expected[0],
                "wall_attenuation_sum": expected[1],
            },
        )
        self.assertEqual(
            api_wall_features(
                {"x": ap[0], "y": ap[1]},
                {"x": receiver[0], "y": receiver[1]},
                self.walls,
            ),
            expected,
        )
        self.assertEqual(
            optimizer_wall_features(ap[0], ap[1], receiver[0], receiver[1], self.walls),
            expected,
        )

    def test_model_feature_builder_preserves_training_contract(self):
        features = build_rssi_features(
            {"x": 1, "y": 5}, {"x": 9, "y": 5}, self.walls
        )

        self.assertEqual(features["distance"], 8.0)
        self.assertEqual(features["num_walls_intersected"], 2)
        self.assertEqual(features["wall_attenuation_sum"], 12.0)
        self.assertEqual(feature_frame([features]).columns.tolist(), MODEL_FEATURES)


if __name__ == "__main__":
    unittest.main()
