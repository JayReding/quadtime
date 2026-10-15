"""Generate printable PDF markers for the timing gate.

    python tools/make_markers.py
    python tools/make_markers.py --marker-size-mm 75 --out markers

Writes one PDF per marker, e.g. ``markers/marker_0.pdf``.  Each page holds a
single vector marker at its true physical size, so it prints at 100% scale on
A4 or Letter with no scaling step.  Every page carries a ruler so the print
scale can be checked with a rule before the marker goes on the gate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# allow `python tools/make_markers.py` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadtime.config import Config, load_config  # noqa: E402
from quadtime.markers import PAGE_SIZES_MM, marker_grid, render_marker_pdf  # noqa: E402

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--out", default="markers", help="output directory")
    parser.add_argument("--page", default="a4", choices=sorted(PAGE_SIZES_MM),
                        help="paper size for the marker pages")
    parser.add_argument("--marker-size-mm", type=float, default=None,
                        help="printed size of the black square")
    parser.add_argument("--quiet-zone-mm", type=float, default=None,
                        help="white border kept clear around each marker")
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

    written = []
    try:
        for marker_id, corner in zip(cfg.marker_ids, CORNER_NAMES):
            path = out / f"marker_{marker_id}.pdf"
            render_marker_pdf(
                path,
                marker_grid(dictionary, marker_id),
                marker_size_mm,
                quiet_zone_mm=quiet_mm,
                page=args.page,
                title=f"quadtime marker {marker_id} - {corner} inside corner",
                note=f"{cfg.aruco_dict}, mount flat and upright. print at 100% scale.",
            )
            written.append(path)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        f"wrote {len(written)} markers ({marker_size_mm:g} mm, {cfg.aruco_dict}) "
        f"to {out}/"
    )
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
