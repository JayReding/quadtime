"""Runtime configuration for quadtime.

Every tunable knob lives here so the vision pipeline can be adapted to a
particular camera / gate / lighting setup without editing algorithm code.
Values can be overridden from a JSON file, see :func:`load_config`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    # --- capture --------------------------------------------------------
    # "synthetic" renders an internal first-person fly-through (no hardware
    # required), a digit string ("0", "1", ...) selects a local camera /
    # capture card, anything else is treated as a video file path.
    source: str = "synthetic"
    width: int = 640
    height: int = 480
    fps: float = 30.0
    fourcc: str = "YUYV"

    # --- gate geometry (metres) -----------------------------------------
    aruco_dict: str = "DICT_APRILTAG_36h11"
    # Physical corner each marker ID is mounted on, ordered around the gate.
    marker_ids: list[int] = field(default_factory=lambda: [0, 1, 2, 3])
    # Square gate, 24 inch opening, markers flat at the inside corners.
    gate_width_m: float = 0.6096
    gate_height_m: float = 0.6096
    marker_size_m: float = 0.075
    # How far outside the opening the camera may be and still count as a pass.
    aperture_margin_m: float = 0.10

    # --- camera model ---------------------------------------------------
    # Intrinsics are loaded from this file when it exists, otherwise a nominal
    # model is derived from the horizontal field of view.
    intrinsics_file: str = "config/camera.json"
    hfov_deg: float = 120.0

    # --- marker detection / pose ----------------------------------------
    max_reprojection_error_px: float = 4.0
    min_marker_perimeter_rate: float = 0.02
    # A single marker is a poorly conditioned planar target; require several.
    min_markers: int = 2

    # --- pass detection -------------------------------------------------
    # Samples used for the local trajectory fit.
    fit_samples: int = 6
    # Reject crossings slower than this (a drone hovering through the gate).
    min_closing_speed_mps: float = 1.5
    # Only extrapolate a crossing from a genuinely close approach.
    close_distance_m: float = 1.0
    # Consecutive missed frames that trigger the extrapolated crossing.
    lost_frames_to_trigger: int = 2
    # Maximum time beyond the last detection a crossing may be predicted.
    max_extrapolation_s: float = 0.25
    # Reject a fit that is not close to a straight approach.
    max_fit_residual_m: float = 0.25
    # Debounce; a second lap cannot register faster than this.
    min_lap_time_s: float = 2.0
    # Distance the drone must regain before another pass can register.
    rearm_distance_m: float = 1.5
    # Drop the sample buffer after the gate has been unseen for this long.
    reset_timeout_s: float = 0.5

    # --- race -------------------------------------------------------------
    countdown_s: float = 5.0

    # --- web --------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 5000
    jpeg_quality: int = 80

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config`, overriding fields listed in a JSON file."""
    cfg = Config()
    data = json.loads(Path(path).read_text())
    known = {f.name for f in fields(Config)}
    for key, value in data.items():
        if key in known:
            setattr(cfg, key, value)
    return cfg
