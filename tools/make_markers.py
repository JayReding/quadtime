"""Generate printable marker artwork for the timing gate.

    python tools/make_markers.py
    python tools/make_markers.py --format stl --marker-size-mm 75

``--format pdf`` (the default) writes one PDF per marker, each holding a single
vector marker at its true physical size so it prints at 100% scale with no
scaling step.  ``--format stl`` writes two STL layers per marker: a white base
tile and the black marker raised on top, for 3D printing in two materials or
with a filament change.  ``--format both`` writes both kinds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# allow `python tools/make_markers.py` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadtime.config import Config, load_config  # noqa: E402
from quadtime.markers import (  # noqa: E402
    PAGE_SIZES_MM,
    marker_grid,
    render_marker_pdf,
    render_marker_stls,
)

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--out", default="markers", help="output directory")
    parser.add_argument("--format", choices=("pdf", "stl", "both"), default="pdf",
                        help="what to generate")
    parser.add_argument("--marker-size-mm", type=float, default=None,
                        help="printed size of the black square")
    parser.add_argument("--quiet-zone-mm", type=float, default=None,
                        help="clear margin kept around each marker")
    parser.add_argument("--page", default="a4", choices=sorted(PAGE_SIZES_MM),
                        help="paper size for the PDF pages")
    parser.add_argument("--base-thickness-mm", type=float, default=1.6,
                        help="thickness of the white base layer (STL)")
    parser.add_argument("--marker-thickness-mm", type=float, default=0.6,
                        help="height of the raised black layer (STL)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else Config()
    marker_size_mm = (
        args.marker_size_mm if args.marker_size_mm is not None
        else round(cfg.marker_size_m * 1000)
    )
    quiet_mm = (
        args.quiet_zone_mm if args.quiet_zone_mm is not None
        else round(cfg.marker_size_m * 1000 * 0.2)
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))

    written: list[tuple[Path, str]] = []
    try:
        for marker_id, corner in zip(cfg.marker_ids, CORNER_NAMES):
            grid = marker_grid(dictionary, marker_id)

            if args.format in ("pdf", "both"):
                path = out / f"marker_{marker_id}.pdf"
                render_marker_pdf(
                    path, grid, marker_size_mm,
                    quiet_zone_mm=quiet_mm, page=args.page,
                    title=f"quadtime marker {marker_id} - {corner} inside corner",
                    note=f"{cfg.aruco_dict}, mount flat and upright. print at 100% scale.",
                )
                written.append((path, ""))

            if args.format in ("stl", "both"):
                base = out / f"marker_{marker_id}_base.stl"
                pattern = out / f"marker_{marker_id}_pattern.stl"
                render_marker_stls(
                    base, pattern, grid, marker_size_mm,
                    quiet_zone_mm=quiet_mm,
                    base_thickness_mm=args.base_thickness_mm,
                    marker_thickness_mm=args.marker_thickness_mm,
                    name=f"quadtime marker {marker_id}",
                )
                written.append((base, "white base"))
                written.append((pattern, "black marker"))
    except ValueError as exc:
        parser.error(str(exc))

    print(f"wrote {len(written)} files ({marker_size_mm:g} mm, {cfg.aruco_dict}) to {out}/")
    for path, role in written:
        print(f"  {path}" + (f"  ({role})" if role else ""))


if __name__ == "__main__":
    main()
