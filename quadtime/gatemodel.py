"""The metric gate model.

Defines the gate in its own coordinate frame: origin at the centre of the
opening, ``x`` to the right, ``y`` up, ``z`` out of the gate plane (towards the
approaching drone).  Markers sit flat on the plane at the four inside corners,
mounted upright.

The 3D positions of every marker corner are what make pose estimation possible:
they are matched against the corners found in the image to recover the camera's
position relative to the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

# Corner signs for the four markers, ordered TL, TR, BR, BL.
_CORNER_SIGNS = ((-1, +1), (+1, +1), (+1, -1), (-1, -1))
# Marker-local corner offsets, clockwise from the marker's own top-left.
_MARKER_CORNER_OFFSETS = ((-1, +1), (+1, +1), (+1, -1), (-1, -1))


@dataclass
class GateModel:
    width_m: float
    height_m: float
    marker_size_m: float
    marker_ids: list[int]

    @classmethod
    def from_config(cls, cfg: Config) -> "GateModel":
        return cls(
            width_m=cfg.gate_width_m,
            height_m=cfg.gate_height_m,
            marker_size_m=cfg.marker_size_m,
            marker_ids=[int(i) for i in cfg.marker_ids],
        )

    @property
    def half_width(self) -> float:
        return self.width_m / 2.0

    @property
    def half_height(self) -> float:
        return self.height_m / 2.0

    def index_of(self, marker_id: int) -> int | None:
        try:
            return self.marker_ids.index(int(marker_id))
        except ValueError:
            return None

    def marker_center(self, index: int) -> np.ndarray:
        """Centre of marker ``index`` in gate coordinates."""
        sign_x, sign_y = _CORNER_SIGNS[index]
        inset_x = self.half_width - self.marker_size_m / 2.0
        inset_y = self.half_height - self.marker_size_m / 2.0
        return np.array([sign_x * inset_x, sign_y * inset_y, 0.0], dtype=np.float64)

    def marker_corners(self, index: int) -> np.ndarray:
        """The marker's four corners in gate coordinates, image order.

        Returned in the same clockwise order (top-left first) that
        ``cv2.aruco.detectMarkers`` reports, for an upright marker seen from the
        front.
        """
        half = self.marker_size_m / 2.0
        centre = self.marker_center(index)
        return np.array(
            [
                centre + np.array([sx * half, sy * half, 0.0])
                for sx, sy in _MARKER_CORNER_OFFSETS
            ],
            dtype=np.float64,
        )

    def opening_corners(self) -> np.ndarray:
        """The four corners of the gate opening, for drawing and tests."""
        return np.array(
            [
                [-self.half_width, +self.half_height, 0.0],
                [+self.half_width, +self.half_height, 0.0],
                [+self.half_width, -self.half_height, 0.0],
                [-self.half_width, -self.half_height, 0.0],
            ],
            dtype=np.float64,
        )

    def contains(self, point_gate: np.ndarray, margin: float = 0.0) -> bool:
        """True when a point in gate coordinates lies within the opening."""
        return (
            abs(float(point_gate[0])) <= self.half_width + margin
            and abs(float(point_gate[1])) <= self.half_height + margin
        )
