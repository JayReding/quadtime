"""Run the FPV vision pipeline over a video file and report gate passes.

Useful for tuning without a live camera: it reports how often the gate was
locked, the distance range over which markers were visible, every pass it found
and the resulting lap times.

    python tools/analyze_clip.py clip.mp4
    python tools/analyze_clip.py clip.mp4 --out annotated.mp4 --config my.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

# allow `python tools/analyze_clip.py` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadtime.calibration import resolve_intrinsics  # noqa: E402
from quadtime.config import Config, load_config  # noqa: E402
from quadtime.fpv import PlaneTracker  # noqa: E402
from quadtime.gatemodel import GateModel  # noqa: E402
from quadtime.pose import GateLocator  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", help="video file to analyse")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--out", help="write an annotated video here")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else Config()
    capture = cv2.VideoCapture(args.clip)
    if not capture.isOpened():
        raise SystemExit(f"could not open {args.clip}")

    fps = capture.get(cv2.CAP_PROP_FPS) or cfg.fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cfg.width, cfg.height = width, height
    intrinsics = resolve_intrinsics(cfg)
    model = GateModel.from_config(cfg)
    locator = GateLocator(cfg, model, intrinsics)
    tracker = PlaneTracker(cfg, model)

    writer = None
    if args.out:
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_index = 0
    locked = 0
    distances: list[float] = []
    passes: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pose = locator.locate(gray)
        event = tracker.update(timestamp, pose)
        if pose is not None:
            locked += 1
            distances.append(pose.plane_distance)
        if event is not None:
            passes.append(event.timestamp)
            print(f"pass  frame {frame_index:6d}  t={event.timestamp:8.3f}s  {event.method}")
        if writer is not None:
            locator.annotate(frame, pose)
            writer.write(frame)
        frame_index += 1
        if args.max_frames and frame_index >= args.max_frames:
            break

    capture.release()
    if writer is not None:
        writer.release()

    print()
    print(f"frames analysed : {frame_index}")
    if frame_index:
        print(f"gate locked     : {locked / frame_index:.1%} of frames")
    if distances:
        print(f"visible range   : {min(distances):.2f} m to {max(distances):.2f} m")
    print(f"gate passes     : {len(passes)}")
    if len(passes) > 1:
        lap_times = np.diff(passes)
        print(f"lap times       : {', '.join(f'{t:.3f}' for t in lap_times)}")
        print(f"best lap        : {lap_times.min():.3f}s")
    if not passes:
        print(
            "\nNo passes found.  Check the gate is locked on approach and that the "
            "marker geometry in the config matches the real gate."
        )
    if args.out:
        print(f"annotated video : {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
