import unittest

import cv2
import numpy as np

from quadtime.calibration import nominal_intrinsics, resolve_intrinsics
from quadtime.config import Config
from quadtime.gatemodel import GateModel
from quadtime.pose import GateLocator
from quadtime.synthetic import FpvScene

# the "camera in front of an upright gate" rotation, gate +z towards the camera
_FRONT_RVEC = np.array([np.pi, 0.0, 0.0])


class PoseRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.intrinsics = resolve_intrinsics(self.cfg)
        self.model = GateModel.from_config(self.cfg)
        self.matrix = self.intrinsics.matrix()
        self.distortion = self.intrinsics.vector()
        self.locator = GateLocator(self.cfg, self.model, self.intrinsics)

        self.object_points = np.concatenate(
            [self.model.marker_corners(i) for i in range(4)]
        ).astype(np.float64)
        self.ids = list(self.model.marker_ids)

    def _image_points(self, tvec):
        points, _ = cv2.projectPoints(
            self.object_points, _FRONT_RVEC, tvec, self.matrix, self.distortion
        )
        return points.reshape(-1, 2).astype(np.float64)

    @staticmethod
    def _expected_camera(tvec):
        rotation, _ = cv2.Rodrigues(_FRONT_RVEC)
        return -rotation.T @ np.asarray(tvec, dtype=np.float64)

    def test_recovers_a_known_head_on_pose(self):
        pose = self.locator._solve(
            self.object_points, self._image_points(np.array([0.0, 0.0, 0.5])), self.ids
        )
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.plane_distance, 0.5, places=2)
        np.testing.assert_allclose(pose.camera_in_gate, [0.0, 0.0, 0.5], atol=0.01)
        self.assertLess(pose.reprojection_error, 1.0)

    def test_recovers_an_off_axis_pose(self):
        tvec = np.array([0.06, -0.04, 0.7])
        pose = self.locator._solve(self.object_points, self._image_points(tvec), self.ids)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(
            pose.camera_in_gate, self._expected_camera(tvec), atol=0.01
        )
        self.assertAlmostEqual(pose.plane_distance, 0.7, places=2)

    def test_requires_a_minimum_number_of_markers(self):
        cfg = Config()
        cfg.min_markers = 5  # more markers than the gate has
        intrinsics = resolve_intrinsics(cfg)
        locator = GateLocator(cfg, self.model, intrinsics)
        scene = FpvScene(cfg, intrinsics=intrinsics)
        frame = scene.render_camera(np.array([0.0, 0.0, 0.5]))
        self.assertIsNone(locator.locate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


class SyntheticScenePoseTest(unittest.TestCase):
    def test_distance_is_recovered_from_rendered_frames(self):
        cfg = Config()
        intrinsics = nominal_intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
        model = GateModel.from_config(cfg)
        locator = GateLocator(cfg, model, intrinsics)
        scene = FpvScene(cfg, intrinsics=intrinsics)

        for distance in (0.9, 0.6, 0.4, 0.3):
            locator.reset()
            frame = scene.render_camera(np.array([0.0, 0.0, distance]))
            pose = locator.locate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            self.assertIsNotNone(pose, f"no pose at {distance} m")
            self.assertAlmostEqual(pose.plane_distance, distance, delta=0.02)
            self.assertLess(abs(pose.camera_in_gate[0]), 0.03)
            self.assertLess(abs(pose.camera_in_gate[1]), 0.03)


if __name__ == "__main__":
    unittest.main()
