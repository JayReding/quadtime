"""Frame sources: local cameras / capture cards, video files, and synthetic."""

from __future__ import annotations

import time

import cv2

from .config import Config
from .synthetic import SyntheticSource


class SourceError(RuntimeError):
    """Raised when a frame source cannot be opened."""


class FrameSource:
    """Base class for anything that produces timestamped frames."""

    live = False

    def now(self) -> float:
        """Current clock for this source (monotonic or virtual)."""
        return time.monotonic()

    def read(self) -> tuple[bool, "cv2.typing.MatLike | None", float]:
        """Return ``(ok, frame, timestamp)``.

        ``timestamp`` is taken as close to frame capture as possible so that
        lap times are not distorted by processing delay.
        """
        raise NotImplementedError

    def release(self) -> None:
        pass


class OpenCVSource(FrameSource):
    """A camera / capture card (int index) or a video file (path)."""

    def __init__(self, target: int | str, cfg: Config):
        self.live = isinstance(target, int)
        self.target = target
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise SourceError(f"could not open video source {target!r}")
        if self.live:
            if cfg.fourcc:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            # keep the driver buffer shallow so we always time the newest frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, time.monotonic()

    def release(self) -> None:  # pragma: no cover - hardware dependent
        self.cap.release()


def make_source(cfg: Config) -> FrameSource:
    """Build the frame source described by ``cfg.source``."""
    raw = str(cfg.source)
    if raw == "synthetic":
        return SyntheticSource(cfg)
    target: int | str = int(raw) if raw.isdigit() else raw
    return OpenCVSource(target, cfg)
