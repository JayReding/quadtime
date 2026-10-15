"""Generate printable ArUco / AprilTag markers for the timing gate.

    python tools/make_markers.py
    python tools/make_markers.py --marker-size-mm 75 --quiet-zone-mm 15

Writes one PNG per marker plus ``gate_markers.html``.  Open that file in a
browser and print it at 100% scale (disable "fit to page") to get markers at
their true physical size.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

# allow `python tools/make_markers.py` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quadtime.config import Config, load_config  # noqa: E402

CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>quadtime gate markers</title>
<style>
  @page {{ size: A4; margin: 12mm; }}
  body {{ font: 14px/1.5 system-ui, sans-serif; color: #111; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14mm; margin-top: 8mm; }}
  .cell {{ text-align: center; break-inside: avoid; }}
  .marker {{
    width: {size_mm}mm; height: {size_mm}mm;
    padding: {quiet_mm}mm; background: #fff; margin: 0 auto;
    border: 1px solid #ddd;
  }}
  .marker img {{ width: 100%; height: 100%; display: block; image-rendering: pixelated; }}
  .label {{ margin-top: 4mm; font-weight: 600; }}
  .hint {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>quadtime gate markers</h1>
<p class="hint">
  Dictionary <b>{dictionary}</b>. Print at 100% scale (no "fit to page"), then
  verify the black square of each marker measures <b>{size_mm} mm</b>.
  Mount flat at the corresponding inside corner of the gate, upright, with the
  white border kept clear.
</p>
<div class="grid">
{cards}
</div>
</body>
</html>
"""

CARD = """  <div class="cell">
    <div class="marker"><img src="{file}" alt="marker {marker_id}"></div>
    <div class="label">ID {marker_id} &mdash; {corner}</div>
    <div class="hint">gate's {corner} inside corner</div>
  </div>"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--out", default="markers", help="output directory")
    parser.add_argument("--size-px", type=int, default=1200, help="rendered PNG size")
    parser.add_argument("--marker-size-mm", type=int, default=None, help="printed size")
    parser.add_argument("--quiet-zone-mm", type=int, default=None, help="white border")
    args = parser.parse_args(argv)

    cfg = load_config(args.config) if args.config else Config()
    marker_size_mm = args.marker_size_mm or round(cfg.marker_size_m * 1000)
    quiet_mm = args.quiet_zone_mm or round(cfg.marker_size_m * 1000 * 0.2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dict))
    cards = []
    for marker_id, corner in zip(cfg.marker_ids, CORNER_NAMES):
        image = cv2.aruco.generateImageMarker(dictionary, int(marker_id), args.size_px)
        filename = f"marker_{marker_id}.png"
        cv2.imwrite(str(out / filename), image)
        cards.append(CARD.format(file=filename, marker_id=marker_id, corner=corner))

    html = PAGE.format(
        size_mm=marker_size_mm,
        quiet_mm=quiet_mm,
        dictionary=cfg.aruco_dict,
        cards="\n".join(cards),
    )
    (out / "gate_markers.html").write_text(html)
    print(
        f"wrote {len(cards)} markers ({marker_size_mm} mm, {cfg.aruco_dict}) "
        f"and {out / 'gate_markers.html'}"
    )


if __name__ == "__main__":
    main()
