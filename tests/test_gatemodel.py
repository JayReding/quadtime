import unittest

import numpy as np

from quadtime.config import Config
from quadtime.gatemodel import GateModel


class GateModelTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.model = GateModel.from_config(self.cfg)

    def test_defaults_are_a_24_inch_square_gate(self):
        self.assertAlmostEqual(self.model.width_m, 0.6096, places=6)
        self.assertAlmostEqual(self.model.height_m, 0.6096, places=6)
        self.assertAlmostEqual(self.model.marker_size_m, 0.075, places=6)

    def test_marker_centres_sit_at_the_inside_corners(self):
        inset = 0.6096 / 2 - 0.075 / 2
        expected = [
            (-inset, +inset),
            (+inset, +inset),
            (+inset, -inset),
            (-inset, -inset),
        ]
        for index, (x, y) in enumerate(expected):
            centre = self.model.marker_center(index)
            self.assertAlmostEqual(centre[0], x, places=6)
            self.assertAlmostEqual(centre[1], y, places=6)
            self.assertEqual(centre[2], 0.0)

    def test_marker_corners_wrap_the_marker_centre(self):
        half = 0.075 / 2
        corners = self.model.marker_corners(0)
        self.assertEqual(corners.shape, (4, 3))
        for corner in corners:
            self.assertAlmostEqual(abs(corner[0] - self.model.marker_center(0)[0]), half, places=6)
            self.assertAlmostEqual(abs(corner[1] - self.model.marker_center(0)[1]), half, places=6)

    def test_opening_corners(self):
        opening = self.model.opening_corners()
        self.assertEqual(opening.shape, (4, 3))
        self.assertAlmostEqual(float(np.max(np.abs(opening[:, 0]))), 0.3048, places=6)

    def test_contains_uses_the_opening_and_margin(self):
        self.assertTrue(self.model.contains(np.array([0.0, 0.0, 0.0])))
        self.assertTrue(self.model.contains(np.array([0.30, -0.30, 0.0])))
        self.assertFalse(self.model.contains(np.array([0.45, 0.0, 0.0])))
        self.assertTrue(self.model.contains(np.array([0.45, 0.0, 0.0]), margin=0.2))

    def test_index_of(self):
        self.assertEqual(self.model.index_of(0), 0)
        self.assertEqual(self.model.index_of(3), 3)
        self.assertIsNone(self.model.index_of(99))


if __name__ == "__main__":
    unittest.main()
