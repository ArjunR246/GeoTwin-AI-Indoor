import unittest

from utils.wall_intersections import count_intersections, wall_attenuation_db


class WallIntersectionTests(unittest.TestCase):
    def setUp(self):
        self.walls = [
            {
                "start": {"x": 2, "y": 0},
                "end": {"x": 2, "y": 10},
                "material": "drywall",
            },
            {
                "start": {"x": 5, "y": 0},
                "end": {"x": 5, "y": 10},
                "material": "brick",
            },
            {
                "start": {"x": 8, "y": 0},
                "end": {"x": 8, "y": 10},
                "material": "concrete",
            },
        ]

    def test_counts_every_wall_crossed_by_a_signal_path(self):
        count, attenuation = count_intersections((1, 5), (9, 5), self.walls)

        self.assertEqual(count, 3)
        self.assertEqual(attenuation, 27.0)

    def test_endpoint_contact_is_not_counted_as_a_wall_crossing(self):
        wall = [{
            "start": {"x": 5, "y": 0},
            "end": {"x": 5, "y": 5},
            "material": "brick",
        }]

        count, attenuation = count_intersections((1, 5), (9, 5), wall)

        self.assertEqual(count, 0)
        self.assertEqual(attenuation, 0.0)

    def test_explicit_attenuation_overrides_material_default(self):
        wall = {"material": "concrete", "attenuation_db": 9.5}

        self.assertEqual(wall_attenuation_db(wall), 9.5)


if __name__ == "__main__":
    unittest.main()
