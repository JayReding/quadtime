"""The FPV vision engine and the race state machine."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .calibration import resolve_intrinsics
from .capture import FrameSource, make_source
from .config import Config
from .fpv import PassEvent, PlaneTracker
from .gatemodel import GateModel
from .pose import GateLocator, GatePose
from . import __version__


class Phase(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    STOPPED = "stopped"


@dataclass
class Lap:
    index: int
    crossing_t: float
    lap_time: float
    total_time: float


@dataclass
class EngineResult:
    pose: GatePose | None = None
    event: PassEvent | None = None


class FpvEngine:
    """Gate pose estimation plus pass detection for a drone's-eye view."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.intrinsics = resolve_intrinsics(cfg)
        self.model = GateModel.from_config(cfg)
        self.locator = GateLocator(cfg, self.model, self.intrinsics)
        self.tracker = PlaneTracker(cfg, self.model)

    def reset(self) -> None:
        self.locator.reset()
        self.tracker.reset()

    def process(self, frame: np.ndarray, timestamp: float) -> EngineResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose = self.locator.locate(gray)
        event = self.tracker.update(timestamp, pose)
        return EngineResult(pose=pose, event=event)

    def annotate(self, frame: np.ndarray, result: EngineResult) -> None:
        self.locator.annotate(frame, result.pose)


class RaceController:
    """Owns the capture thread and the race state.

    The web layer only ever reads :meth:`snapshot`; all mutable state is guarded
    by a lock so the vision thread and the request handlers never race.
    """

    def __init__(
        self,
        cfg: Config,
        source: FrameSource | None = None,
        engine: FpvEngine | None = None,
    ):
        self.cfg = cfg
        self.error: str | None = None
        if source is not None:
            self.source: FrameSource | None = source
        else:
            try:
                self.source = make_source(cfg)
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                self.source = None
                self.error = f"{type(exc).__name__}: {exc}"
        self.engine = engine or FpvEngine(cfg)

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.phase = Phase.IDLE
        self.countdown_end: float | None = None
        self.start_t: float | None = None
        self.end_t: float | None = None
        self.laps: list[Lap] = []
        self.result: EngineResult = EngineResult()
        self.last_frame: np.ndarray | None = None
        self.last_frame_t: float | None = None
        self.stats = {"frames": 0, "fps": 0.0}
        self._stats_prev_t: float | None = None

        self._jpeg: bytes | None = self._encode(self._placeholder("waiting for video..."))
        self._stream_id = 0

    # -- clock -----------------------------------------------------------
    def now(self) -> float:
        if self.source is not None:
            return self.source.now()
        return time.monotonic()

    # -- race control ----------------------------------------------------
    def start_race(self) -> None:
        with self._lock:
            if self.phase in (Phase.COUNTDOWN, Phase.RUNNING):
                return
            self.laps = []
            self.engine.reset()
            self.start_t = None
            self.end_t = None
            self.countdown_end = self.now() + self.cfg.countdown_s
            self.phase = Phase.COUNTDOWN

    def stop_race(self) -> None:
        with self._lock:
            if self.phase in (Phase.COUNTDOWN, Phase.RUNNING):
                self.end_t = self.now()
                self.phase = Phase.STOPPED

    def reset_race(self) -> None:
        """Clear the finished race and go back to idle.

        Only meaningful once a race has been stopped, so anything else is
        ignored - that keeps a stale click from wiping a race in progress.
        """
        with self._lock:
            if self.phase is not Phase.STOPPED:
                return
            self.laps = []
            self.engine.reset()
            self.start_t = None
            self.end_t = None
            self.countdown_end = None
            self.phase = Phase.IDLE

    # -- worker ----------------------------------------------------------
    def launch(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="quadtime-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.source is None:
                time.sleep(0.2)
                continue
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - keep the UI alive
                self.error = f"{type(exc).__name__}: {exc}"
                if not getattr(self.source, "live", False):
                    self.source = None
                else:
                    time.sleep(0.2)

    def tick(self) -> None:
        """Read and process exactly one frame.  Also used directly by tests."""
        assert self.source is not None
        ok, frame, timestamp = self.source.read()
        if not ok or frame is None:
            if getattr(self.source, "live", False):
                time.sleep(0.005)
                return
            raise RuntimeError("end of video source")

        result = self.engine.process(frame, timestamp)

        with self._cond:
            self.result = result
            self.last_frame_t = timestamp
            self._advance_phase(timestamp)
            if result.event is not None:
                self._handle_crossing(result.event.timestamp)
            self._update_stats(timestamp)

            annotated = frame.copy()
            self.engine.annotate(annotated, result)
            self._draw_overlay(annotated)
            self._jpeg = self._encode(annotated)
            self.last_frame = annotated
            self._stream_id += 1
            self._cond.notify_all()

    def _advance_phase(self, timestamp: float) -> None:
        if (
            self.phase is Phase.COUNTDOWN
            and self.countdown_end is not None
            and timestamp >= self.countdown_end
        ):
            self.phase = Phase.RUNNING
            self.start_t = timestamp

    def _handle_crossing(self, crossing_t: float) -> None:
        if self.phase is not Phase.RUNNING or self.start_t is None:
            return
        if crossing_t < self.start_t:
            return
        if self.laps and crossing_t - self.laps[-1].crossing_t < self.cfg.min_lap_time_s:
            return
        lap_time = crossing_t - (self.laps[-1].crossing_t if self.laps else self.start_t)
        self.laps.append(
            Lap(
                index=len(self.laps) + 1,
                crossing_t=crossing_t,
                lap_time=lap_time,
                total_time=crossing_t - self.start_t,
            )
        )

    def _update_stats(self, timestamp: float) -> None:
        self.stats["frames"] += 1
        if self._stats_prev_t is not None and timestamp > self._stats_prev_t:
            instant = 1.0 / (timestamp - self._stats_prev_t)
            previous = self.stats["fps"]
            self.stats["fps"] = instant if previous == 0 else previous * 0.9 + instant * 0.1
        self._stats_prev_t = timestamp

    # -- reporting -------------------------------------------------------
    def summary(self) -> dict | None:
        if self.phase is not Phase.STOPPED or self.start_t is None or self.end_t is None:
            return None
        lap_times = [lap.lap_time for lap in self.laps]
        return {
            "total_time": self.end_t - self.start_t,
            "lap_times": lap_times,
            "best_lap": min(lap_times) if lap_times else None,
            "average_lap": (sum(lap_times) / len(lap_times)) if lap_times else None,
            "lap_count": len(lap_times),
        }

    def snapshot(self) -> dict:
        with self._lock:
            now = self.now()
            remaining = 0.0
            elapsed = 0.0
            if self.phase is Phase.COUNTDOWN and self.countdown_end is not None:
                remaining = max(0.0, self.countdown_end - now)
            elif self.phase is Phase.RUNNING and self.start_t is not None:
                elapsed = max(0.0, now - self.start_t)
            elif self.phase is Phase.STOPPED and self.start_t is not None and self.end_t is not None:
                elapsed = max(0.0, self.end_t - self.start_t)

            pose = self.result.pose
            return {
                "phase": self.phase.value,
                "countdown_remaining": round(remaining, 3),
                "elapsed": round(elapsed, 3),
                "laps": [
                    {
                        "index": lap.index,
                        "lap_time": round(lap.lap_time, 3),
                        "total_time": round(lap.total_time, 3),
                    }
                    for lap in self.laps
                ],
                "summary": self.summary(),
                "gate_visible": pose is not None,
                "markers": len(pose.marker_ids) if pose else 0,
                "distance_m": round(pose.plane_distance, 3) if pose else None,
                "reprojection_error": round(pose.reprojection_error, 2) if pose else None,
                "closing_speed": round(self.engine.tracker.closing_speed, 2),
                "fps": round(self.stats["fps"], 1),
                "frames": self.stats["frames"],
                "error": self.error,
                "source": str(self.cfg.source),
                "version": __version__,
            }

    # -- video frames ----------------------------------------------------
    def mjpeg_stream(self):
        boundary = b"--frame\r\n"
        seen = -1
        while True:
            with self._cond:
                self._cond.wait(timeout=1.0)
                if self._jpeg is None or self._stream_id == seen:
                    continue
                seen = self._stream_id
                jpeg = self._jpeg
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

    def shutdown(self) -> None:
        self._stop.set()
        if self.source is not None:
            try:
                self.source.release()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # -- drawing helpers -------------------------------------------------
    def _encode(self, frame: np.ndarray) -> bytes | None:
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.cfg.jpeg_quality)]
        )
        return buffer.tobytes() if ok else None

    def _placeholder(self, message: str) -> np.ndarray:
        frame = np.full((self.cfg.height, self.cfg.width, 3), 24, dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (20, self.cfg.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _draw_overlay(self, frame: np.ndarray) -> None:
        colour = {
            Phase.IDLE: (180, 180, 180),
            Phase.COUNTDOWN: (0, 200, 255),
            Phase.RUNNING: (0, 220, 0),
            Phase.STOPPED: (255, 120, 120),
        }[self.phase]
        cv2.putText(
            frame, self.phase.value.upper(), (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
        )
        cv2.putText(
            frame, f"laps {len(self.laps)}", (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        pose = self.result.pose
        if pose is None:
            cv2.putText(
                frame, "gate not visible", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                f"gate {pose.plane_distance:4.2f} m  {len(pose.marker_ids)} mk",
                (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
            )
        if self.error:
            cv2.putText(
                frame, self.error[:60], (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )
