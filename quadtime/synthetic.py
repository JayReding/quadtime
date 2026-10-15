"""A synthetic first-person fly-through of the gate.

Renders what a forward-facing camera on the drone would see as it flies at the
gate: the markers grow and slide out of frame, and the gate opening sweeps past.
Because the camera path is known exactly, the scene also supplies ground-truth
crossing times, which makes it useful both as a demo and as a test fixture.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Intrinsics, nominal_intrinsics
from .config import Config
from .gatemodel import GateModel

_MARKER_PX = 220


@dataclass
class FlightPath:
    """A straight, constant-speed pass through the gate."""

    start_distance_m: float = 2.5
    speed_mps: float = 3.0
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    end_distance_m: float = -0.6

    def position_at(self, t: float) -> np.ndarray:
        z = self.start_distance_m - self.speed_mps * t
        return np.array([self.offset_x_m, self.offset_y_m, z], dtype=np.float64)

    @property
    def cross_time(self) -> float:
        """Time at which the camera crosses the gate plane."""
        return self.start_distance_m / self.speed_mps

    @property
    def duration(self) -> float:
        return (self.start_distance_m - self.end_distance_m) / self.speed_mps


class FpvScene:
    """Renders a single forward-facing camera view of the gate."""

    def __init__(
        self,
        cfg: Config,
        path: FlightPath | None = None,
        loop: bool = True,
        pause_s: float = 2.5,
        pitch_deg: float = 0.0,
        intrinsics: Intrinsics | None = None,
        noise: bool = True,
    ):
        self.cfg = cfg
        self.path = path or FlightPath()
        self.loop = loop
        self.pause_s = pause_s
        self.noise = noise
        self.width = cfg.width
        self.height = cfg.height
        self.intrinsics = intrinsics or nominal_intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
        self.model = GateModel.from_config(cfg)
        self.rotation = self._camera_rotation(pitch_deg)
        self.cycle_s = self.path.duration + pause_s if loop else self.path.duration

        self._background = self._build_background(pitch_deg)
        self._markers = self._build_markers()

    # -- construction -----------------------------------------------------
    def _camera_rotation(self, pitch_deg: float) -> np.ndarray:
        """World-to-camera rotation for looking along -z, optionally tilted."""
        base = np.diag([1.0, -1.0, -1.0])
        if not pitch_deg:
            return base
        tilt = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0.0, 0.0]))[0]
        return tilt @ base

    def _build_markers(self) -> dict[int, np.ndarray]:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.cfg.aruco_dict))
        images: dict[int, np.ndarray] = {}
        for marker_id in self.model.marker_ids:
            gray = cv2.aruco.generateImageMarker(dictionary, int(marker_id), _MARKER_PX)
            images[int(marker_id)] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return images

    def _build_background(self, pitch_deg: float) -> np.ndarray:
        w, h = self.width, self.height
        column = np.linspace(120, 190, h, dtype=np.float32)
        frame = np.repeat(column[:, None, None], w, axis=1)
        frame = np.repeat(frame, 3, axis=2).astype(np.uint8)
        horizon = int(np.clip(self.intrinsics.cy + self.intrinsics.fy * math.tan(math.radians(pitch_deg)), 0, h - 1))
        frame[horizon:, :, 0] = np.clip(frame[horizon:, :, 0].astype(np.int16) - 40, 0, 255)
        frame[horizon:, :, 1] = np.clip(frame[horizon:, :, 1].astype(np.int16) - 60, 0, 255)
        frame[horizon:, :, 2] = np.clip(frame[horizon:, :, 2].astype(np.int16) - 80, 0, 255)
        return frame

    # -- rendering --------------------------------------------------------
    def _project(self, points_world: np.ndarray, rvec, tvec):
        image_points, _ = cv2.projectPoints(
            points_world, rvec, tvec, self.intrinsics.matrix(), self.intrinsics.vector()
        )
        return image_points.reshape(-1, 2)

    def _draw_opening(self, frame: np.ndarray, position, rotation) -> None:
        points = self.model.opening_corners()
        camera_points = (points - position) @ rotation.T
        if np.any(camera_points[:, 2] <= 0.05):
            return
        rvec = cv2.Rodrigues(rotation)[0]
        tvec = -rotation @ position
        quad = self._project(points, rvec, tvec).astype(np.int32)
        cv2.polylines(frame, [quad], True, (250, 250, 250), 5, cv2.LINE_AA)

    def _draw_marker(self, frame: np.ndarray, index: int, position, rotation) -> None:
        points = self.model.marker_corners(index)
        camera_points = (points - position) @ rotation.T
        if np.any(camera_points[:, 2] <= 0.05):
            return
        rvec = cv2.Rodrigues(rotation)[0]
        tvec = -rotation @ position
        quad = self._project(points, rvec, tvec).astype(np.float32)

        w, h = self.width, self.height
        if quad[:, 0].max() < -40 or quad[:, 0].min() > w + 40:
            return
        if quad[:, 1].max() < -40 or quad[:, 1].min() > h + 40:
            return

        centre = quad.mean(axis=0)
        backing = (centre + (quad - centre) * 1.22).astype(np.int32)
        cv2.fillConvexPoly(frame, backing, (245, 245, 245), cv2.LINE_AA)

        source = np.array(
            [[0, 0], [_MARKER_PX - 1, 0], [_MARKER_PX - 1, _MARKER_PX - 1], [0, _MARKER_PX - 1]],
            dtype=np.float32,
        )
        marker_id = self.model.marker_ids[index]
        transform = cv2.getPerspectiveTransform(source, quad)
        warped = cv2.warpPerspective(self._markers[marker_id], transform, (w, h))
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
        frame[mask > 0] = warped[mask > 0]

    def _add_noise(self, frame: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, 3.0, frame.shape)
        noisy = (frame.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        return cv2.GaussianBlur(noisy, (3, 3), 0)

    def render_camera(self, position: np.ndarray, rotation: np.ndarray | None = None) -> np.ndarray:
        rotation = self.rotation if rotation is None else rotation
        frame = self._background.copy()
        self._draw_opening(frame, position, rotation)
        for index in range(len(self.model.marker_ids)):
            self._draw_marker(frame, index, position, rotation)
        return self._add_noise(frame) if self.noise else frame

    def render(self, t: float) -> np.ndarray:
        t_state = t % self.cycle_s if self.loop else t
        t_pos = min(t_state, self.path.duration)
        return self.render_camera(self.path.position_at(t_pos))


class SyntheticSource:
    """A :class:`~quadtime.capture.FrameSource` driven by :class:`FpvScene`.

    With ``realtime=False`` the clock advances one frame per :meth:`read`
    instead of sleeping, which lets tests run as fast as the CPU allows.
    """

    live = False

    def __init__(
        self,
        cfg: Config,
        realtime: bool = True,
        path: FlightPath | None = None,
        loop: bool = True,
        pause_s: float = 2.5,
        pitch_deg: float = 0.0,
        noise: bool = True,
    ):
        self.cfg = cfg
        self.realtime = realtime
        self.scene = FpvScene(
            cfg, path=path, loop=loop, pause_s=pause_s, pitch_deg=pitch_deg, noise=noise
        )
        self._t0 = time.monotonic()
        self._frames = 0
        self._next = time.monotonic()

    def now(self) -> float:
        if self.realtime:
            return time.monotonic()
        return self._t0 + self._frames * (1.0 / self.cfg.fps)

    def read(self) -> tuple[bool, np.ndarray, float]:
        if self.realtime:
            delay = self._next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._next = max(self._next + 1.0 / self.cfg.fps, time.monotonic())
        self._frames += 1
        t = self.now()
        frame = self.scene.render(t - self._t0)
        return True, frame, t

    def release(self) -> None:  # pragma: no cover - nothing to release
        pass
