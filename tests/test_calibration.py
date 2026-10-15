import unittest

import cv2
import numpy as np

from quadtime.calibration import (
    Intrinsics,
    load_intrinsics,
    nominal_intrinsics,
    save_intrinsics,
)


class IntrinsicsTest(unittest.TestCase):
    def test_nominal_from_field_of_view(self):
        intr = nominal_intrinsics(640, 480, 90.0)
        self.assertAlmostEqual(intr.fx, 320.0, places=3)  # 320 / tan(45 deg)
        self.assertAlmostEqual(intr.fy, 320.0, places=3)
        self.assertAlmostEqual(intr.cx, 319.5, places=3)
        self.assertAlmostEqual(intr.cy, 239.5, places=3)

    def test_matrix_and_vector_shapes(self):
        intr = nominal_intrinsics(640, 480, 120.0)
        self.assertEqual(intr.matrix().shape, (3, 3))
        self.assertEqual(intr.vector().shape, (1, 5))

    def test_round_trip_through_a_file(self):
        import tempfile
        from pathlib import Path

        intr = Intrinsics(500.0, 505.0, 320.0, 240.0, np.zeros(5), 640, 480, "test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "camera.json"
            save_intrinsics(path, intr)
            loaded = load_intrinsics(path)
        self.assertAlmostEqual(loaded.fx, 500.0)
        self.assertAlmostEqual(loaded.fy, 505.0)
        self.assertAlmostEqual(loaded.cx, 320.0)
        self.assertEqual((loaded.width, loaded.height), (640, 480))

    def test_scaled_to_another_resolution(self):
        intr = nominal_intrinsics(640, 480, 120.0)
        scaled = intr.scaled_to(1280, 960)
        self.assertAlmostEqual(scaled.fx, intr.fx * 2, places=6)
        self.assertAlmostEqual(scaled.cx, intr.cx * 2, places=6)
        self.assertEqual((scaled.width, scaled.height), (1280, 960))


class CalibrationPipelineTest(unittest.TestCase):
    """Exercise the calibrate-camera pipeline on synthetic ChArUco views."""

    def test_recovers_the_focal_length(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.02, dictionary)
        detector = cv2.aruco.CharucoDetector(board)

        size = (640, 480)
        true_matrix = np.array([[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]])
        zero_dist = np.zeros(5)

        flat = board.generateImage((1000, 800), marginSize=60, borderBits=1)
        corners, ids, _, _ = detector.detectBoard(flat)
        board_object, board_image = board.matchImagePoints(corners, ids)
        source = board_image.reshape(-1, 2).astype(np.float32)
        points_3d = board_object.reshape(-1, 3).astype(np.float64)

        rng = np.random.default_rng(7)
        object_points = []
        image_points = []
        for i in range(16):
            rvec = rng.normal(0.0, 0.30, 3)
            rvec[0] = rng.uniform(-0.4, 0.4)
            tvec = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.55 + 0.15 * i / 16])
            target, _ = cv2.projectPoints(points_3d, rvec, tvec, true_matrix, zero_dist)
            target = target.reshape(-1, 2).astype(np.float32)
            homography, _ = cv2.findHomography(source, target)
            view = cv2.warpPerspective(flat, homography, size, borderValue=255)

            view_corners, view_ids, _, _ = detector.detectBoard(view)
            if view_ids is None or len(view_ids) < 8:
                continue
            obj, img = board.matchImagePoints(view_corners, view_ids)
            object_points.append(obj)
            image_points.append(img)

        self.assertGreaterEqual(len(object_points), 6)
        rms, matrix, _, _, _ = cv2.calibrateCamera(
            object_points, image_points, size, None, None
        )
        self.assertLess(abs(matrix[0, 0] - 500.0) / 500.0, 0.05)
        self.assertLess(rms, 1.5)


if __name__ == "__main__":
    unittest.main()
