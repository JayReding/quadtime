import unittest

import numpy as np

from quadtime.config import Config
from quadtime.fpv import PlaneTracker
from quadtime.gatemodel import GateModel
from quadtime.pose import GatePose

FRAME_TIME = 1.0 / 30.0


def fake_pose(x, y, z):
    """A pose whose camera sits at (x, y, z) in gate coordinates."""
    return GatePose(
        rvec=np.zeros(3),
        tvec=np.zeros(3),
        rotation=np.eye(3),
        camera_in_gate=np.array([x, y, z], dtype=np.float64),
        reprojection_error=0.0,
    )


class PlaneTrackerTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.model = GateModel.from_config(self.cfg)
        self.tracker = PlaneTracker(self.cfg, self.model)

    def approach(self, speed, start_z=0.9, stop_z=0.28, x=0.0, y=0.0, t0=0.0):
        """Feed a constant-velocity approach until the markers would leave."""
        times = []
        t = t0
        z = start_z
        while z > stop_z:
            self.tracker.update(t, fake_pose(x, y, z))
            times.append(t)
            t += FRAME_TIME
            z = start_z - speed * (t - t0)
        return t, times

    def test_extrapolated_crossing_matches_a_constant_velocity_pass(self):
        speed = 3.0
        start_z = 0.9
        next_t, _ = self.approach(speed, start_z=start_z)
        # two dropped frames trigger the extrapolation
        event = self.tracker.update(next_t, None) or self.tracker.update(next_t + FRAME_TIME, None)
        self.assertIsNotNone(event)
        self.assertEqual(event.method, "extrapolated")
        truth = start_z / speed
        self.assertAlmostEqual(event.timestamp, truth, delta=0.005)

    def test_interpolated_crossing_when_the_gate_stays_visible(self):
        times = [0.0, FRAME_TIME, 2 * FRAME_TIME, 3 * FRAME_TIME]
        depths = [0.30, 0.20, 0.10, -0.02]
        event = None
        for t, z in zip(times, depths):
            event = self.tracker.update(t, fake_pose(0.0, 0.0, z)) or event
        self.assertIsNotNone(event)
        self.assertEqual(event.method, "interpolated")
        # crosses zero between 0.10 and -0.02
        expected = 2 * FRAME_TIME + (0.10 / 0.12) * FRAME_TIME
        self.assertAlmostEqual(event.timestamp, expected, delta=0.005)

    def test_flying_past_the_gate_is_rejected(self):
        self.approach(3.0, x=0.6)  # 0.6 m off centre, outside the opening
        self.tracker.update(1.0, None)
        self.assertIsNone(self.tracker.update(1.0 + FRAME_TIME, None))

    def test_a_slow_approach_is_rejected(self):
        self.approach(0.5)  # below min_closing_speed_mps
        self.tracker.update(10.0, None)
        self.assertIsNone(self.tracker.update(10.0 + FRAME_TIME, None))

    def test_a_distant_approach_is_not_extrapolated(self):
        # lose the gate while still far away: too bold to predict a crossing
        for i in range(5):
            self.tracker.update(i * FRAME_TIME, fake_pose(0.0, 0.0, 1.2))
        self.tracker.update(5 * FRAME_TIME, None)
        self.assertIsNone(self.tracker.update(6 * FRAME_TIME, None))

    def test_second_pass_is_debounced_by_the_minimum_lap_time(self):
        self.approach(3.0, t0=0.0)
        first = self.tracker.update(0.4, None) or self.tracker.update(0.4 + FRAME_TIME, None)
        self.assertIsNotNone(first)
        # return to the approach side and cross again, but too soon
        self.tracker.update(1.0, fake_pose(0.0, 0.0, 1.6))  # re-arm
        self.approach(3.0, t0=1.0)
        second = self.tracker.update(1.4, None) or self.tracker.update(1.4 + FRAME_TIME, None)
        self.assertIsNone(second)

    def test_second_pass_counts_after_a_long_enough_gap(self):
        self.approach(3.0, t0=0.0)
        first = self.tracker.update(0.4, None) or self.tracker.update(0.4 + FRAME_TIME, None)
        self.assertIsNotNone(first)
        self.tracker.update(3.0, fake_pose(0.0, 0.0, 1.6))  # re-arm, well clear
        self.approach(3.0, t0=3.0)
        second = self.tracker.update(3.4, None) or self.tracker.update(3.4 + FRAME_TIME, None)
        self.assertIsNotNone(second)
        self.assertGreater(second.timestamp - first.timestamp, self.cfg.min_lap_time_s)

    def test_requires_rearming_before_another_pass(self):
        self.approach(3.0, t0=0.0)
        first = self.tracker.update(0.4, None) or self.tracker.update(0.4 + FRAME_TIME, None)
        self.assertIsNotNone(first)
        self.assertFalse(self.tracker.armed)


if __name__ == "__main__":
    unittest.main()
