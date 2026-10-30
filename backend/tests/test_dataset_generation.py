import unittest
from pathlib import Path

import numpy as np

from data.generate_dataset import DEFAULT_TWIN_PATHS, generate_dataset, load_layout


class DatasetGenerationTests(unittest.TestCase):
    def test_bundled_twins_are_valid_and_have_multiple_partitions(self):
        layouts = [load_layout(Path(path)) for path in DEFAULT_TWIN_PATHS]

        self.assertEqual(len(layouts), 2)
        self.assertTrue(all(len(layout["walls"]) >= 10 for layout in layouts))

    def test_generated_data_is_balanced_and_contains_multi_wall_paths(self):
        dataset = generate_dataset(num_samples=600, seed=7)

        self.assertEqual(len(dataset), 600)
        self.assertEqual(set(dataset["layout_id"]), {
            "floor1_office",
            "floor2_l_shaped_residential",
        })
        self.assertEqual(dataset["layout_id"].value_counts().nunique(), 1)
        self.assertGreaterEqual(dataset["num_walls_intersected"].max(), 3)
        self.assertTrue((dataset["wall_attenuation_sum"] >= 0).all())
        self.assertFalse(dataset.isna().any().any())

    def test_generation_is_reproducible_for_a_fixed_seed(self):
        first = generate_dataset(num_samples=12, seed=123)
        second = generate_dataset(num_samples=12, seed=123)

        np.testing.assert_allclose(
            first.drop(columns="layout_id"), second.drop(columns="layout_id")
        )
        self.assertListEqual(first["layout_id"].tolist(), second["layout_id"].tolist())


if __name__ == "__main__":
    unittest.main()
