"""A synthetic first-person fly-through of the gate.

Renders what a forward-facing camera on the drone would see as it flies at the
gate: the markers grow and slide out of frame, and the gate opening sweeps past.
Because the camera path is known exactly, the scene also supplies ground-truth
crossing times, which makes it useful both as a demo and as a test fixture.

By default the scene flies a :class:`FlightSchedule` - a sequence of passes that
differ in speed, approach angle, crossing point and turnaround time - so the
demo behaves like racing instead of a metronome.  Pass a single
:class:`FlightPath` instead to keep one pass exactly repeatable, as the tests do.
"""

from __future__ import annotations

import bisect
import math
import random
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import Intrinsics, nominal_intrinsics
from .config import Config
from .gatemodel import GateModel

_MARKER_PX = 220
_WORLD_UP = np.array([0.0, 1.0, 0.0])


def camera_rotation(
    forward: np.ndarray,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """World-to-camera rotation for a camera aimed along ``forward``.

    Gate coordinates have ``+y`` up and ``+z`` towards an approaching drone, so a
    camera looking straight at the gate from the front gives ``diag(1, -1, -1)``
    - the orientation the rest of quadtime assumes.  ``pitch_deg`` is the
    camera's mount tilt and ``roll_deg`` banks it about its own optical axis.
    """
    forward = np.asarray(forward, dtype=np.float64)
    z_axis = forward / np.linalg.norm(forward)
    x_axis = -np.cross(_WORLD_UP, z_axis)
    if np.linalg.norm(x_axis) < 1e-9:  # aiming straight up or down
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis]).T
    if pitch_deg:
        tilt = cv2.Rodrigues(np.array([math.radians(pitch_deg), 0.0, 0.0]))[0]
        rotation = tilt @ rotation
    if roll_deg:
        bank = cv2.Rodrigues(np.array([0.0, 0.0, math.radians(roll_deg)]))[0]
        rotation = bank @ rotation
    return rotation


@dataclass
class FlightPath:
    """A straight, constant-speed pass at the gate.

    ``speed_mps`` is the closing speed along the gate normal.  ``drift_x_mps``
    and ``drift_y_mps`` add velocity sideways and vertically, so the drone crosses
    the plane on a slant rather than square on - the same thing as a gate that is
    angled relative to the flight line.  ``offset_x_m`` / ``offset_y_m`` are where
    the pass *starts*; where it crosses is :attr:`cross_point`.  ``roll_deg``
    banks the camera, as a drone does when it turns.
    """

    start_distance_m: float = 2.5
    speed_mps: float = 3.0
    offset_x_m: float = 0.0
    offset_y_m: float = 0.0
    drift_x_mps: float = 0.0
    drift_y_mps: float = 0.0
    end_distance_m: float = -0.6
    roll_deg: float = 0.0

    @property
    def velocity(self) -> np.ndarray:
        """Camera velocity in gate coordinates (m/s)."""
        return np.array([self.drift_x_mps, self.drift_y_mps, -self.speed_mps], dtype=np.float64)

    def position_at(self, t: float) -> np.ndarray:
        start = np.array(
            [self.offset_x_m, self.offset_y_m, self.start_distance_m], dtype=np.float64
        )
        return start + self.velocity * t

    @property
    def cross_time(self) -> float:
        """Time at which the camera crosses the gate plane."""
        return self.start_distance_m / self.speed_mps

    @property
    def cross_point(self) -> np.ndarray:
        """Where the pass meets the gate plane, in gate coordinates."""
        return self.position_at(self.cross_time)[:2]

    @property
    def approach_deg(self) -> float:
        """Angle between the flight direction and the gate normal, in degrees."""
        drift = math.hypot(self.drift_x_mps, self.drift_y_mps)
        return math.degrees(math.atan2(drift, self.speed_mps))

    @property
    def duration(self) -> float:
        return (self.start_distance_m - self.end_distance_m) / self.speed_mps


@dataclass
class FlightSchedule:
    """A varied sequence of passes, one per lap.

    No two laps of a real race look alike: the drone lines up from a different
    direction, crosses somewhere else in the opening and takes a different time to
    come back around.  This generates that variety deterministically from
    ``seed``, so the demo is reproducible while still never repeating a pass.
    """

    seed: int = 0
    # Half the gate opening; the pass is aimed to cross inside this fraction of it.
    half_width_m: float = 0.3048
    half_height_m: float = 0.3048
    cross_fraction: float = 0.55
    # Ranges to draw each pass from: distance at the start, closing speed, angle
    # off the gate normal, and the pause spent turning around before the next pass.
    start_distance_m: tuple[float, float] = (2.0, 3.4)
    speed_mps: tuple[float, float] = (2.2, 5.0)
    approach_deg: tuple[float, float] = (0.0, 18.0)
    pause_s: tuple[float, float] = (2.0, 4.0)
    # A drone banks into its turns, so the camera rolls either way by this much.
    roll_deg: float = 10.0
    end_distance_m: float = -0.6

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._starts: list[float] = []
        self._paths: list[FlightPath] = []
        self._ends: list[float] = []  # start + pass + turnaround

    @classmethod
    def for_gate(cls, model: GateModel, **kwargs) -> "FlightSchedule":
        """A schedule whose passes cross inside ``model``'s opening."""
        return cls(
            half_width_m=model.half_width, half_height_m=model.half_height, **kwargs
        )

    # -- generation -------------------------------------------------------
    def _make_pass(self) -> FlightPath:
        rng = self._rng
        start_distance = rng.uniform(*self.start_distance_m)
        closing = rng.uniform(*self.speed_mps)
        approach = math.radians(rng.uniform(*self.approach_deg))
        heading = rng.uniform(0.0, 2.0 * math.pi)
        cross_time = start_distance / closing

        # Pick the slant, then start far enough off-axis to arrive on target.
        slant = closing * math.tan(approach)
        drift_x = slant * math.cos(heading)
        drift_y = slant * math.sin(heading)
        cross_x = rng.uniform(-1.0, 1.0) * self.cross_fraction * self.half_width_m
        cross_y = rng.uniform(-1.0, 1.0) * self.cross_fraction * self.half_height_m

        return FlightPath(
            start_distance_m=start_distance,
            speed_mps=closing,
            offset_x_m=cross_x - drift_x * cross_time,
            offset_y_m=cross_y - drift_y * cross_time,
            drift_x_mps=drift_x,
            drift_y_mps=drift_y,
            end_distance_m=self.end_distance_m,
            roll_deg=rng.uniform(-self.roll_deg, self.roll_deg),
        )

    def _extend(self) -> None:
        start = self._ends[-1] if self._ends else 0.0
        path = self._make_pass()
        self._starts.append(start)
        self._paths.append(path)
        self._ends.append(start + path.duration + self._rng.uniform(*self.pause_s))

    # -- lookup -----------------------------------------------------------
    def at(self, t: float) -> tuple[FlightPath, float]:
        """The pass in progress at ``t`` and the time elapsed within it."""
        while not self._ends or t >= self._ends[-1]:
            self._extend()
        index = max(0, bisect.bisect_right(self._starts, t) - 1)
        path = self._paths[index]
        return path, min(t - self._starts[index], path.duration)

    def crossings(self, until_t: float) -> list[float]:
        """Ground-truth times at which each generated pass meets the gate plane."""
        while not self._starts or self._starts[-1] <= until_t:
            self._extend()
        return [
            start + path.cross_time
            for start, path in zip(self._starts, self._paths)
            if start <= until_t
        ]

    def entries(self, count: int) -> list[tuple[float, FlightPath]]:
        """The first ``count`` passes as ``(start_time, path)``."""
        while len(self._paths) < count:
            self._extend()
        return list(zip(self._starts[:count], self._paths[:count]))


class FpvScene:
    """Renders a single forward-facing camera view of the gate."""

    def __init__(
        self,
        cfg: Config,
        path: FlightPath | None = None,
        schedule: FlightSchedule | None = None,
        loop: bool = True,
        pause_s: float = 2.5,
        pitch_deg: float = 0.0,
        intrinsics: Intrinsics | None = None,
        noise: bool = True,
    ):
        if path is not None and schedule is not None:
            raise ValueError("pass a single path or a schedule, not both")
        self.cfg = cfg
        self.model = GateModel.from_config(cfg)
        self.pitch_deg = pitch_deg
        self.loop = loop
        self.pause_s = pause_s
        self.noise = noise
        self.width = cfg.width
        self.height = cfg.height
        self.intrinsics = intrinsics or nominal_intrinsics(cfg.width, cfg.height, cfg.hfov_deg)

        # No path and no schedule means the varied demo flight.
        self.schedule = None if path is not None else (schedule or FlightSchedule.for_gate(self.model))
        self.path = path or FlightPath()
        self.rotation = camera_rotation(self.path.velocity, pitch_deg, self.path.roll_deg)
        self.cycle_s = self.path.duration + pause_s if loop else self.path.duration

        self._background = self._build_background(pitch_deg)
        self._markers = self._build_markers()

    # -- construction -----------------------------------------------------
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

    # -- flight -----------------------------------------------------------
    def pass_at(self, t: float) -> tuple[FlightPath, float]:
        """The pass in progress at ``t`` and the time elapsed within it."""
        if self.schedule is not None:
            return self.schedule.at(t)
        state = t % self.cycle_s if self.loop else t
        return self.path, min(state, self.path.duration)

    def render(self, t: float) -> np.ndarray:
        path, t_pos = self.pass_at(t)
        rotation = camera_rotation(path.velocity, self.pitch_deg, path.roll_deg)
        return self.render_camera(path.position_at(t_pos), rotation)


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
        schedule: FlightSchedule | None = None,
        loop: bool = True,
        pause_s: float = 2.5,
        pitch_deg: float = 0.0,
        noise: bool = True,
    ):
        self.cfg = cfg
        self.realtime = realtime
        self.scene = FpvScene(
            cfg,
            path=path,
            schedule=schedule,
            loop=loop,
            pause_s=pause_s,
            pitch_deg=pitch_deg,
            noise=noise,
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
