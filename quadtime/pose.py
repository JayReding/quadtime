"""Marker detection and gate pose estimation.

The gate is a planar target of known metric size, so a single camera can
recover the full 3D pose of the gate relative to the drone's camera.  From that
pose we derive the signed distance from the camera to the gate plane, whose sign
flip is a gate pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import Intrinsics
from .config import Config
from .gatemodel import GateModel


@dataclass
class GatePose:
    """The gate's pose in camera coordinates."""

    rvec: np.ndarray
    tvec: np.ndarray
    rotation: np.ndarray          # gate -> camera
    camera_in_gate: np.ndarray    # camera position in gate coordinates
    reprojection_error: float
    marker_ids: list[int] = field(default_factory=list)
    points: int = 0

    @property
    def plane_distance(self) -> float:
        """Signed distance from the camera to the gate plane.

        Positive on the approach side, zero in the plane, negative once the
        camera has passed through.
        """
        return float(self.camera_in_gate[2])


class GateLocator:
    """Detects the gate markers and recovers :class:`GatePose` for a frame."""

    def __init__(self, cfg: Config, model: GateModel, intrinsics: Intrinsics):
        self.cfg = cfg
        self.model = model
        self.intrinsics = intrinsics
        self.matrix = intrinsics.matrix()
        self.distortion = intrinsics.vector()

        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
        params = cv2.aruco.DetectorParameters()
        # tuned for small, motion-blurred markers on a noisy analogue feed
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 6
        params.minMarkerPerimeterRate = cfg.min_marker_perimeter_rate
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.perspectiveRemovePixelPerCell = 8
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 30
        params.cornerRefinementMinAccuracy = 0.1
        try:
            refine = cv2.aruco.RefineParameters()
            refine.minRepDistance = 10.0
            self.detector = cv2.aruco.ArucoDetector(dictionary, params, refine)
        except Exception:  # pragma: no cover - older OpenCV
            self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        self.previous: GatePose | None = None
        self.last_corners = None
        self.last_ids = None

    def reset(self) -> None:
        self.previous = None
        self.last_corners = None
        self.last_ids = None

    def locate(self, gray: np.ndarray) -> GatePose | None:
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last_corners, self.last_ids = corners, ids
        if ids is None or len(ids) == 0:
            return None

        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        used_ids: list[int] = []
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            index = self.model.index_of(int(marker_id))
            if index is None:
                continue
            object_points.append(self.model.marker_corners(index))
            image_points.append(marker_corners.reshape(4, 2))
            used_ids.append(int(marker_id))

        if not used_ids:
            return None
        if len(used_ids) < self.cfg.min_markers:
            return None
        object_points = np.concatenate(object_points).astype(np.float64)
        image_points = np.concatenate(image_points).astype(np.float64)
        if len(object_points) < 4:
            return None

        pose = self._solve(object_points, image_points, used_ids)
        if pose is None:
            return None
        if pose.reprojection_error > self.cfg.max_reprojection_error_px:
            return None
        self.previous = pose
        return pose

    def _solve(
        self, object_points: np.ndarray, image_points: np.ndarray, used_ids: list[int]
    ) -> GatePose | None:
        candidates: list[tuple[np.ndarray, np.ndarray]] = []

        # 1. continue from the previous frame - resolves the planar ambiguity and
        #    keeps the solution stable across frames
        if self.previous is not None:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.matrix,
                self.distortion,
                self.previous.rvec.copy(),
                self.previous.tvec.copy(),
                True,
                cv2.SOLVEPNP_ITERATIVE,
            )
            if ok:
                candidates.append((rvec, tvec))

        # 2. closed-form solvers that need no initial guess.  SQPNP is robust for
        #    planar targets, including the exactly-frontal case where IPPE is
        #    degenerate; IPPE is kept as an extra candidate.
        for flag in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_IPPE):
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, self.matrix, self.distortion, flags=flag
                )
            except cv2.error:
                continue
            if ok:
                candidates.append((rvec, tvec))

        # 3. closed-form planar solutions, worst case
        ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points, image_points, self.matrix, self.distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if ok:
            candidates.extend((rvecs[i], tvecs[i]) for i in range(len(rvecs)))

        best: tuple[float, np.ndarray, np.ndarray] | None = None
        fallback: tuple[float, np.ndarray, np.ndarray] | None = None
        for rvec, tvec in candidates:
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
            if tvec[2] <= 0:  # the gate must be in front of the camera
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            error = self._reprojection_error(object_points, image_points, rvec, tvec)
            # a planar target has two solutions; the real one has the gate facing
            # the camera, i.e. its normal pointing back along the camera's view
            if rotation[2, 2] < 0.0:
                if best is None or error < best[0]:
                    best = (error, rvec, tvec)
            elif fallback is None or error < fallback[0]:
                fallback = (error, rvec, tvec)

        chosen = best or fallback
        if chosen is None:
            return None

        error, rvec, tvec = chosen
        rotation, _ = cv2.Rodrigues(rvec)
        camera_in_gate = -rotation.T @ tvec
        return GatePose(
            rvec=rvec,
            tvec=tvec,
            rotation=rotation,
            camera_in_gate=camera_in_gate,
            reprojection_error=float(error),
            marker_ids=used_ids,
            points=len(object_points),
        )

    def _reprojection_error(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, self.matrix, self.distortion
        )
        projected = projected.reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))

    def annotate(self, frame: np.ndarray, pose: GatePose | None) -> None:
        if self.last_corners is not None and self.last_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, self.last_corners, self.last_ids)
        if pose is None:
            return
        opening, _ = cv2.projectPoints(
            self.model.opening_corners(), pose.rvec, pose.tvec, self.matrix, self.distortion
        )
        cv2.polylines(frame, [opening.astype(np.int32)], True, (0, 255, 0), 2)
        centre, _ = cv2.projectPoints(
            np.zeros((1, 3)), pose.rvec, pose.tvec, self.matrix, self.distortion
        )
        cx, cy = centre.reshape(2).astype(int)
        cv2.drawMarker(frame, (cx, cy), (255, 200, 0), cv2.MARKER_CROSS, 14, 2)
