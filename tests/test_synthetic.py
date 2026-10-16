import math
import unittest

import numpy as np

from quadtime.config import Config
from quadtime.gatemodel import GateModel
from quadtime.race import FpvEngine
from quadtime.synthetic import (
    FlightPath,
    FlightSchedule,
    FpvScene,
    SyntheticSource,
    camera_rotation,
)


class CameraRotationTest(unittest.TestCase):
    def test_looking_at_the_gate_matches_the_project_convention(self):
        np.testing.assert_allclose(
            camera_rotation(np.array([0.0, 0.0, -1.0])), np.diag([1.0, -1.0, -1.0])
        )

    def test_rotation_is_orthonormal_and_looks_along_the_flight_direction(self):
        for forward in ([0.0, 0.0, -1.0], [-0.3, 0.2, -1.0], [0.5, -0.4, -0.7]):
            with self.subTest(forward=forward):
                direction = np.array(forward, dtype=np.float64)
                rotation = camera_rotation(direction, roll_deg=-8.0)
                np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
                self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=9)
                optical_axis = rotation @ (direction / np.linalg.norm(direction))
                np.testing.assert_allclose(optical_axis, [0.0, 0.0, 1.0], atol=1e-9)

    def test_pitch_tilts_the_optical_axis_and_roll_leaves_it_alone(self):
        forward = np.array([0.0, 0.0, -1.0])
        rolled = camera_rotation(forward, roll_deg=25.0)
        np.testing.assert_allclose(rolled @ forward, [0.0, 0.0, 1.0], atol=1e-9)

        pitched = camera_rotation(forward, pitch_deg=10.0)
        axis = pitched @ forward
        self.assertAlmostEqual(float(axis[2]), math.cos(math.radians(10.0)), places=9)
        self.assertGreater(abs(float(axis[1])), 0.1)


class FlightPathTest(unittest.TestCase):
    def test_a_straight_pass_behaves_as_before(self):
        path = FlightPath(start_distance_m=2.5, speed_mps=3.0, offset_x_m=0.6)
        self.assertAlmostEqual(path.cross_time, 2.5 / 3.0)
        self.assertAlmostEqual(path.duration, 3.1 / 3.0)
        np.testing.assert_allclose(path.position_at(0.0), [0.6, 0.0, 2.5])
        np.testing.assert_allclose(path.cross_point, [0.6, 0.0])
        self.assertAlmostEqual(path.approach_deg, 0.0)

    def test_a_slanted_pass_keeps_its_closing_speed_and_crosses_off_axis(self):
        path = FlightPath(
            start_distance_m=3.0, speed_mps=4.0, offset_x_m=-1.0, drift_x_mps=1.5
        )
        self.assertAlmostEqual(path.velocity[2], -4.0)
        self.assertAlmostEqual(path.cross_time, 0.75)
        self.assertAlmostEqual(path.cross_point[0], -1.0 + 1.5 * 0.75)
        self.assertAlmostEqual(path.approach_deg, math.degrees(math.atan2(1.5, 4.0)))

    def test_the_slant_does_not_change_the_timing(self):
        straight = FlightPath(start_distance_m=2.5, speed_mps=3.0)
        slanted = FlightPath(start_distance_m=2.5, speed_mps=3.0, drift_x_mps=2.0, drift_y_mps=-1.0)
        self.assertAlmostEqual(straight.cross_time, slanted.cross_time)
        self.assertAlmostEqual(straight.duration, slanted.duration)


class FlightScheduleTest(unittest.TestCase):
    def setUp(self):
        self.model = GateModel.from_config(Config())
        self.schedule = FlightSchedule.for_gate(self.model, seed=4)

    def test_passes_vary_in_timing_and_direction(self):
        paths = [path for _start, path in self.schedule.entries(10)]
        self.assertGreater(len({round(p.speed_mps, 2) for p in paths}), 3)
        self.assertGreater(len({round(p.approach_deg, 0) for p in paths}), 3)
        self.assertGreater(len({round(p.roll_deg, 0) for p in paths}), 3)
        self.assertGreater(
            len({(round(p.offset_x_m, 2), round(p.offset_y_m, 2)) for p in paths}), 3
        )

    def test_every_pass_crosses_inside_the_opening(self):
        for _start, path in self.schedule.entries(40):
            point = np.append(path.cross_point, 0.0)
            self.assertTrue(self.model.contains(point), f"crossed at {path.cross_point}")

    def test_passes_never_leave_the_gate_normal_behind(self):
        for _start, path in self.schedule.entries(20):
            self.assertLess(path.velocity[2], -1.0)
            self.assertLess(path.approach_deg, 30.0)

    def test_the_timeline_has_no_gaps_or_overlaps(self):
        entries = self.schedule.entries(6)
        for (start, path), (next_start, _next) in zip(entries, entries[1:]):
            self.assertGreater(next_start, start + path.duration)
            probe = start + path.duration / 2.0
            current, elapsed = self.schedule.at(probe)
            self.assertIs(current, path)
            self.assertAlmostEqual(elapsed, path.duration / 2.0, places=6)

    def test_crossings_are_the_pass_crossing_times(self):
        entries = self.schedule.entries(5)
        until = entries[-1][0]
        expected = [start + path.cross_time for start, path in entries]
        np.testing.assert_allclose(self.schedule.crossings(until), expected)

    def test_the_same_seed_gives_the_same_flight(self):
        other = FlightSchedule.for_gate(self.model, seed=4)
        for (start_a, path_a), (start_b, path_b) in zip(
            self.schedule.entries(4), other.entries(4)
        ):
            self.assertAlmostEqual(start_a, start_b)
            self.assertAlmostEqual(path_a.offset_x_m, path_b.offset_x_m)
            self.assertAlmostEqual(path_a.roll_deg, path_b.roll_deg)
            np.testing.assert_allclose(path_a.velocity, path_b.velocity)

    def test_different_seeds_give_different_flights(self):
        other = FlightSchedule.for_gate(self.model, seed=5)
        first = self.schedule.entries(1)[0][1]
        second = other.entries(1)[0][1]
        self.assertNotAlmostEqual(first.offset_x_m, second.offset_x_m)


class SyntheticSceneFlightTest(unittest.TestCase):
    def test_no_path_means_the_varied_demo_flight(self):
        scene = FpvScene(Config())
        self.assertIsNotNone(scene.schedule)

    def test_an_explicit_path_keeps_a_single_repeating_pass(self):
        scene = FpvScene(Config(), path=FlightPath())
        self.assertIsNone(scene.schedule)
        self.assertIs(scene.pass_at(0.0)[0], scene.path)

    def test_a_path_and_a_schedule_together_are_rejected(self):
        with self.assertRaises(ValueError):
            FpvScene(Config(), path=FlightPath(), schedule=FlightSchedule())

    def test_each_pass_renders_a_different_view(self):
        scene = FpvScene(Config(), noise=False)
        first, second = scene.schedule.entries(2)
        frames = [
            scene.render(start + path.cross_time * 0.8).astype(np.int16)
            for start, path in (first, second)
        ]
        self.assertGreater(float(np.abs(frames[0] - frames[1]).mean()), 1.0)


class ScheduledFlightTimingTest(unittest.TestCase):
    """The whole point of the varied flight: every pass is still one lap."""

    def test_every_scheduled_pass_scores_exactly_one_lap(self):
        cfg = Config()
        model = GateModel.from_config(cfg)
        # The turnaround has to stay longer than min_lap_time_s: two crossings
        # closer together than that are treated as a double count and dropped.
        schedule = FlightSchedule.for_gate(model, seed=11, pause_s=(2.2, 3.0))
        source = SyntheticSource(cfg, realtime=False, schedule=schedule)
        engine = FpvEngine(cfg)

        base = source.now()
        limit = 18.0
        scored = []
        while True:
            _ok, frame, timestamp = source.read()
            if timestamp - base > limit:
                break
            result = engine.process(frame, timestamp)
            if result.event is not None:
                scored.append(result.event.timestamp - base)

        settle = limit - 0.5
        crossings = [t for t in schedule.crossings(limit) if t <= settle]
        answers = [t for t in scored if t <= settle]

        self.assertGreaterEqual(len(crossings), 5)
        self.assertEqual(len(answers), len(crossings), "missed or double-counted a pass")
        for got, expected in zip(answers, crossings):
            self.assertAlmostEqual(got, expected, delta=0.05)


if __name__ == "__main__":
    unittest.main()
