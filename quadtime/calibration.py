"""Camera intrinsics: loading, saving and a nominal fallback.

Pose estimation needs a camera matrix.  Ideally it comes from a real ChArUco
calibration (see ``tools/calibrate_camera.py``); when no calibration file is
present we fall back to a nominal pinhole model derived from the horizontal
field of view, which is accurate enough for a sign-flip timing test.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray
    width: int
    height: int
    source: str = "nominal"

    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def vector(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=np.float64).reshape(1, -1)

    def scaled_to(self, width: int, height: int) -> "Intrinsics":
        """Rescale to a different capture resolution (linear in pixels)."""
        if (width, height) == (self.width, self.height):
            return self
        sx = width / self.width
        sy = height / self.height
        return Intrinsics(
            fx=self.fx * sx,
            fy=self.fy * sy,
            cx=self.cx * sx,
            cy=self.cy * sy,
            dist=np.array(self.dist, dtype=np.float64),
            width=width,
            height=height,
            source=f"{self.source} (scaled)",
        )

    def to_dict(self) -> dict:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist": [float(v) for v in np.asarray(self.dist).ravel()],
            "width": self.width,
            "height": self.height,
        }


def nominal_intrinsics(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """A pinhole model with a horizontal field of view and square pixels."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return Intrinsics(
        fx=fx,
        fy=fx,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        dist=np.zeros(5, dtype=np.float64),
        width=width,
        height=height,
        source="nominal",
    )


def load_intrinsics(path: str | Path) -> Intrinsics:
    data = json.loads(Path(path).read_text())
    return Intrinsics(
        fx=float(data["fx"]),
        fy=float(data["fy"]),
        cx=float(data["cx"]),
        cy=float(data["cy"]),
        dist=np.array(data.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        source=f"file:{path}",
    )


def save_intrinsics(path: str | Path, intrinsics: Intrinsics) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(intrinsics.to_dict(), indent=2) + "\n")


def resolve_intrinsics(cfg: Config) -> Intrinsics:
    """Use the calibration file if present, else the nominal FOV model."""
    path = Path(cfg.intrinsics_file) if cfg.intrinsics_file else None
    if path is not None and path.exists():
        return load_intrinsics(path).scaled_to(cfg.width, cfg.height)
    return nominal_intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
