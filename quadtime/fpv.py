"""Gate pass detection from the gate's distance trajectory.

The exact crossing frame is rarely observable: at the moment the camera reaches
the gate plane the corner markers have already left the field of view.  So the
tracker keeps a short history of the camera's position in gate coordinates and
recovers the crossing by

* interpolating when the gate is visible on both sides of the plane, or
* extrapolating a local straight-line fit when the gate disappears on approach.

Both paths are guarded so that a drone flying *past* the gate, or hovering near
it, does not score a lap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import Config
from .gatemodel import GateModel
from .pose import GatePose


@dataclass
class PassEvent:
    """A scored gate pass."""

    timestamp: float
    camera_in_gate: np.ndarray
    method: str          # "interpolated" or "extrapolated"
    closing_speed: float


class PlaneTracker:
    """Turns a stream of gate poses into pass events."""

    def __init__(self, cfg: Config, model: GateModel):
        self.cfg = cfg
        self.model = model
        self.samples: deque[tuple[float, np.ndarray]] = deque(maxlen=cfg.fit_samples)
        self.armed = True
        self.misses = 0
        self.last_pass_t: float | None = None
        self.last_seen_t: float | None = None
        self.last_position: np.ndarray | None = None
        self.closing_speed = 0.0

    def reset(self) -> None:
        self.samples.clear()
        self.armed = True
        self.misses = 0
        self.last_pass_t = None
        self.last_seen_t = None
        self.last_position = None
        self.closing_speed = 0.0

    # -- main entry point -------------------------------------------------
    def update(self, timestamp: float, pose: GatePose | None) -> PassEvent | None:
        if pose is None:
            return self._on_missing(timestamp)

        position = np.asarray(pose.camera_in_gate, dtype=np.float64).copy()
        previous = self.samples[-1] if self.samples else None

        event = None
        if previous is not None:
            prev_t, prev_position = previous
            if timestamp > prev_t:
                self.closing_speed = float(
                    -(position[2] - prev_position[2]) / (timestamp - prev_t)
                )
            # visible across the plane: interpolate exactly
            if self.armed and prev_position[2] > 0.0 >= position[2]:
                fraction = prev_position[2] / (prev_position[2] - position[2])
                crossing_t = prev_t + fraction * (timestamp - prev_t)
                crossing_position = prev_position + fraction * (position - prev_position)
                event = self._consider(crossing_t, crossing_position, "interpolated")

        self.samples.append((timestamp, position))
        self.last_position = position
        self.last_seen_t = timestamp
        self.misses = 0

        # re-arm once the drone is clearly back in front of the gate
        if not self.armed and position[2] > self.cfg.rearm_distance_m:
            self.armed = True
            self.samples.clear()
            self.samples.append((timestamp, position))
        return event

    # -- helpers ----------------------------------------------------------
    def _on_missing(self, timestamp: float) -> PassEvent | None:
        self.misses += 1
        if self.last_seen_t is not None and timestamp - self.last_seen_t > self.cfg.reset_timeout_s:
            # The gate has been out of view long enough that the old approach is
            # finished - the drone has gone through (or away).  Forget the old
            # samples and allow a fresh approach to be scored.
            self.samples.clear()
            self.last_position = None
            self.closing_speed = 0.0
            self.armed = True
        if (
            self.armed
            and self.misses == self.cfg.lost_frames_to_trigger
            and len(self.samples) >= 3
        ):
            return self._extrapolate()
        return None

    def _extrapolate(self) -> PassEvent | None:
        times = np.array([sample[0] for sample in self.samples], dtype=np.float64)
        positions = np.stack([sample[1] for sample in self.samples])  # (N, 3)

        # weighted least squares, favouring the most recent (closest) samples
        weights = np.arange(1, len(times) + 1, dtype=np.float64)
        scale = np.sqrt(weights / weights.sum())
        design = np.vstack([times, np.ones_like(times)]).T
        coef, *_ = np.linalg.lstsq(
            design * scale[:, None], positions * scale[:, None], rcond=None
        )

        fitted = design @ coef
        residual = float(np.sqrt(np.mean(np.sum((fitted - positions) ** 2, axis=1))))
        if residual > self.cfg.max_fit_residual_m:
            return None

        last_t = float(times[-1])
        z_at_last = float(coef[0, 2] * last_t + coef[1, 2])
        slope = float(coef[0, 2])
        if z_at_last > self.cfg.close_distance_m:
            return None
        if slope >= -self.cfg.min_closing_speed_mps:
            return None

        horizon = (0.0 - z_at_last) / slope
        if horizon < 0.0 or horizon > self.cfg.max_extrapolation_s:
            return None

        crossing_t = last_t + horizon
        self.closing_speed = -slope
        crossing_position = np.array(
            [
                coef[0, 0] * crossing_t + coef[1, 0],
                coef[0, 1] * crossing_t + coef[1, 1],
                0.0,
            ],
            dtype=np.float64,
        )
        return self._consider(crossing_t, crossing_position, "extrapolated")

    def _consider(
        self, crossing_t: float, position: np.ndarray, method: str
    ) -> PassEvent | None:
        if not self.model.contains(position, self.cfg.aperture_margin_m):
            return None
        if self.closing_speed < self.cfg.min_closing_speed_mps:
            return None
        if (
            self.last_pass_t is not None
            and crossing_t - self.last_pass_t < self.cfg.min_lap_time_s
        ):
            return None
        self.last_pass_t = crossing_t
        self.armed = False
        return PassEvent(
            timestamp=float(crossing_t),
            camera_in_gate=np.asarray(position, dtype=np.float64),
            method=method,
            closing_speed=float(self.closing_speed),
        )
