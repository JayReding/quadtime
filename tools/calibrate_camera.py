"""Calibrate camera intrinsics from a ChArUco board.

    python tools/calibrate_camera.py --video calibration.mp4 --out config/camera.json
    python tools/calibrate_camera.py --images calibration/*.png

Move the board slowly around the whole field of view, at a range of distances
and angles, filling the frame.  A dozen or more good views give a solid result.

The board defaults match a 5x7 ChArUco sheet with 40 mm squares and 20 mm
markers; print one with ``tools/make_markers.py``-style tooling or any ChArUco
generator and measure the squares to confirm.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import cv2
import numpy as np

# allow `python tools/calibrate_camera.py` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadtime.calibration import Intrinsics, save_intrinsics  # noqa: E402


def build_board(squares_x, squares_y, square_m, marker_m, dict_name):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.CharucoBoard((squares_x, squares_y), square_m, marker_m, dictionary)


def read_video(path, stride):
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"could not open {path}")
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            yield frame
        index += 1
    capture.release()


def read_images(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no images matched {pattern}")
    for path in paths:
        frame = cv2.imread(path)
        if frame is not None:
            yield frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="video file of the board")
    source.add_argument("--images", help="glob of board images, e.g. 'calib/*.png'")
    parser.add_argument("--out", default="config/camera.json", help="output intrinsics file")
    parser.add_argument("--dict", dest="dict_name", default="DICT_5X5_50")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-m", type=float, default=0.040)
    parser.add_argument("--marker-m", type=float, default=0.020)
    parser.add_argument("--stride", type=int, default=5, help="analyse every Nth video frame")
    parser.add_argument("--min-corners", type=int, default=8)
    args = parser.parse_args(argv)

    board = build_board(
        args.squares_x, args.squares_y, args.square_m, args.marker_m, args.dict_name
    )
    detector = cv2.aruco.CharucoDetector(board)

    frames = read_video(args.video, args.stride) if args.video else read_images(args.images)

    object_points = []
    image_points = []
    size = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < args.min_corners:
            continue
        board_object, board_image = board.matchImagePoints(corners, ids)
        object_points.append(board_object)
        image_points.append(board_image)
        print(f"\rviews collected: {len(object_points)}", end="", flush=True)
    print()

    if len(object_points) < 8:
        raise SystemExit(
            f"only {len(object_points)} usable views; need at least 8. "
            "Move the board around more of the frame."
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )
    intrinsics = Intrinsics(
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        cx=float(matrix[0, 2]),
        cy=float(matrix[1, 2]),
        dist=np.asarray(distortion, dtype=np.float64).ravel(),
        width=int(size[0]),
        height=int(size[1]),
        source="calibration",
    )
    save_intrinsics(args.out, intrinsics)
    print(f"views      : {len(object_points)}")
    print(f"resolution : {size[0]}x{size[1]}")
    print(f"fx, fy     : {intrinsics.fx:.2f}, {intrinsics.fy:.2f}")
    print(f"cx, cy     : {intrinsics.cx:.2f}, {intrinsics.cy:.2f}")
    print(f"reproj rms : {rms:.3f} px")
    print(f"wrote      : {args.out}")


if __name__ == "__main__":
    main()
