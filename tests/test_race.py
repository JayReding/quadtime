import time
import unittest

import numpy as np

from quadtime.capture import FrameSource
from quadtime.config import Config
from quadtime.race import Phase, RaceController
from quadtime.synthetic import FlightPath, SyntheticSource


class BlankSource(FrameSource):
    """A source of empty frames with a controllable virtual clock."""

    def __init__(self, fps=30.0):
        self.fps = fps
        self._t0 = time.monotonic()
        self._frames = 0

    def now(self):
        return self._t0 + self._frames / self.fps

    def read(self):
        self._frames += 1
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        return True, frame, self.now()


class RaceStateMachineTest(unittest.TestCase):
    def test_phase_transitions_and_summary(self):
        cfg = Config()
        cfg.countdown_s = 1.0
        controller = RaceController(cfg, source=BlankSource())

        self.assertEqual(controller.snapshot()["phase"], Phase.IDLE.value)

        controller.start_race()
        self.assertEqual(controller.snapshot()["phase"], Phase.COUNTDOWN.value)

        for _ in range(int(1.0 * 30) + 2):
            controller.tick()
        self.assertEqual(controller.snapshot()["phase"], Phase.RUNNING.value)

        for _ in range(60):
            controller.tick()
        controller.stop_race()

        snapshot = controller.snapshot()
        self.assertEqual(snapshot["phase"], Phase.STOPPED.value)
        self.assertGreater(snapshot["elapsed"], 0.9)
        self.assertEqual(snapshot["laps"], [])
        self.assertIsNotNone(snapshot["summary"])
        self.assertEqual(snapshot["summary"]["lap_count"], 0)
        self.assertIsNone(snapshot["summary"]["best_lap"])

    def test_start_is_ignored_while_running(self):
        cfg = Config()
        controller = RaceController(cfg, source=BlankSource())
        controller.start_race()
        controller.tick()
        controller.start_race()
        self.assertEqual(controller.snapshot()["phase"], Phase.COUNTDOWN.value)

    def test_no_gate_means_no_laps(self):
        cfg = Config()
        cfg.countdown_s = 0.5
        controller = RaceController(cfg, source=BlankSource())
        controller.start_race()
        for _ in range(120):
            controller.tick()
        self.assertEqual(controller.snapshot()["laps"], [])


class FpvRaceTest(unittest.TestCase):
    def _controller(self, **kwargs):
        cfg = Config()
        cfg.countdown_s = kwargs.pop("countdown_s", 5.0)
        path = kwargs.pop("path", FlightPath(start_distance_m=2.5, speed_mps=3.0))
        source = SyntheticSource(cfg, realtime=False, path=path, loop=True, pause_s=2.5)
        return cfg, RaceController(cfg, source=source)

    def test_a_pass_through_the_gate_scores_a_lap(self):
        cfg, controller = self._controller(countdown_s=0.0)
        controller.start_race()
        # one cycle is 0.45 s of flight plus 2.5 s pause
        for _ in range(int(3.4 * cfg.fps)):
            controller.tick()
        laps = controller.snapshot()["laps"]
        self.assertGreaterEqual(len(laps), 1)
        self.assertTrue(controller.snapshot()["gate_visible"] or laps)

    def test_laps_repeat_at_the_flight_cycle(self):
        cfg, controller = self._controller(countdown_s=0.0)
        controller.start_race()
        for _ in range(int(12 * cfg.fps)):
            controller.tick()
        laps = controller.snapshot()["laps"]
        self.assertGreaterEqual(len(laps), 2)
        cycle = 3.1 / 3.0 + 2.5  # flight time + pause
        for lap in laps[1:]:
            self.assertAlmostEqual(lap["lap_time"], cycle, delta=0.2)

    def test_failing_to_pass_the_gate_scores_nothing(self):
        path = FlightPath(start_distance_m=2.5, speed_mps=3.0, offset_x_m=0.6)
        cfg, controller = self._controller(countdown_s=0.0, path=path)
        controller.start_race()
        for _ in range(int(10 * cfg.fps)):
            controller.tick()
        self.assertEqual(controller.snapshot()["laps"], [])

    def test_stop_produces_a_summary(self):
        cfg, controller = self._controller(countdown_s=0.0)
        controller.start_race()
        for _ in range(int(10 * cfg.fps)):
            controller.tick()
        controller.stop_race()
        summary = controller.snapshot()["summary"]
        self.assertIsNotNone(summary)
        self.assertGreaterEqual(summary["lap_count"], 2)
        self.assertIsNotNone(summary["best_lap"])
        self.assertIsNotNone(summary["average_lap"])


if __name__ == "__main__":
    unittest.main()
