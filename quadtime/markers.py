"""Printable marker artwork.

Two outputs from the same marker pattern:

* **PDF pages** whose black square measures an exact number of millimetres on
  paper, for printing and sticking onto the gate.
* **STL tiles** with a white base layer and a raised black marker layer, for
  printing the marker on a 3D printer.

Both are drawn as vector geometry rather than a bitmap, so the size is fixed by
the geometry instead of by an image resolution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2

from .pdfcanvas import PdfPage
from .stl import Solid

# Standard page sizes in millimetres (width, height).
PAGE_SIZES_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}

_PIXELS_PER_MODULE = 8    # sampling resolution when reading the marker bitmap
_TOP_MARGIN_MM = 25.0
_BOTTOM_MARGIN_MM = 15.0
_SIDE_MARGIN_MM = 15.0
_RESERVED_TEXT_MM = 45.0  # room for the label, the note and the check ruler


def marker_grid(
    dictionary, marker_id: int, border_bits: int = 1
) -> list[list[bool]]:
    """The marker's module bitmap, top row first, ``True`` where it is black."""
    modules = dictionary.markerSize + 2 * border_bits
    side = modules * _PIXELS_PER_MODULE
    image = cv2.aruco.generateImageMarker(
        dictionary, int(marker_id), side, borderBits=border_bits
    )
    middle = _PIXELS_PER_MODULE // 2
    return [
        [
            bool(image[row * _PIXELS_PER_MODULE + middle,
                       col * _PIXELS_PER_MODULE + middle] < 128)
            for col in range(modules)
        ]
        for row in range(modules)
    ]


def _check_fits(marker_size_mm: float, quiet_zone_mm: float, page: str,
                page_width_mm: float, page_height_mm: float) -> None:
    needed = (
        _TOP_MARGIN_MM + marker_size_mm + 2 * quiet_zone_mm
        + _RESERVED_TEXT_MM + _BOTTOM_MARGIN_MM
    )
    if needed > page_height_mm:
        raise ValueError(
            f"a {marker_size_mm:.0f} mm marker does not fit on {page} "
            f"({needed:.0f} mm needed); use a smaller --marker-size-mm"
        )
    if marker_size_mm + 2 * quiet_zone_mm > page_width_mm - 2 * _SIDE_MARGIN_MM:
        raise ValueError(
            f"a {marker_size_mm:.0f} mm marker is too wide for {page}; "
            "use a smaller --marker-size-mm"
        )


def marker_cells(grid: Sequence[Sequence[bool]]) -> set[tuple[int, int]]:
    """Black modules as ``(column, y)`` cells, with ``y`` counted upwards.

    The grid rows arrive top-down; ``y`` flips them into the mathematical ``y``
    axis of the tile (and of the PDF page), so a marker seen from above keeps
    the same orientation it has in the image.
    """
    modules = len(grid)
    return {
        (col, modules - 1 - row)
        for row, cells in enumerate(grid)
        for col, black in enumerate(cells)
        if black
    }


def render_marker_stls(
    base_path: str | Path,
    pattern_path: str | Path,
    grid: Sequence[Sequence[bool]],
    marker_size_mm: float,
    *,
    quiet_zone_mm: float = 15.0,
    base_thickness_mm: float = 1.6,
    marker_thickness_mm: float = 0.6,
    name: str = "quadtime marker",
) -> None:
    """Write the two stacked STL layers of a 3D printed marker.

    ``base_path`` is the white tile: a plate ``marker_size_mm`` plus the quiet
    zone on every side, sitting on the bed.  ``pattern_path`` is the black
    layer: the marker's black modules raised on top of the base.  Printing them
    in one job with a filament change at the base thickness, or as two
    materials, gives a white tile with a black marker.

    The defaults - 1.6 mm and 0.6 mm - are whole numbers of 0.2 mm layers, so
    the colour change falls on a layer boundary.
    """
    modules = len(grid)
    if modules == 0 or any(len(row) != modules for row in grid):
        raise ValueError("marker grid must be a non-empty square")
    if marker_size_mm <= 0 or base_thickness_mm <= 0 or marker_thickness_mm <= 0:
        raise ValueError("marker size and layer thicknesses must be positive")

    cells = marker_cells(grid)
    columns = {column for column, _ in cells}
    rows = {row for _, row in cells}
    if columns != set(range(modules)) or rows != set(range(modules)):
        raise ValueError(
            "the marker has no full black border, so its outer size is not "
            "defined; regenerate it with border_bits >= 1"
        )

    cell_mm = marker_size_mm / modules
    half = (marker_size_mm + 2 * quiet_zone_mm) / 2.0

    base = Solid(f"{name} white base")
    base.add_box((-half, -half, 0.0), (half, half, base_thickness_mm))
    base.save(base_path)

    pattern = Solid(f"{name} black marker")
    pattern.add_cells(
        cells,
        cell_mm,
        (-marker_size_mm / 2.0, -marker_size_mm / 2.0),
        base_thickness_mm,
        base_thickness_mm + marker_thickness_mm,
    )
    pattern.save(pattern_path)


def render_marker_pdf(
    path: str | Path,
    grid: Sequence[Sequence[bool]],
    marker_size_mm: float,
    *,
    quiet_zone_mm: float = 15.0,
    page: str = "a4",
    title: str = "",
    note: str = "",
    ruler_mm: float = 50.0,
) -> None:
    """Write a single-marker PDF at the marker's true physical size.

    The page carries the marker, an optional title and note, and a short ruler
    so the print scale can be verified with a rule.
    """
    try:
        page_width_mm, page_height_mm = PAGE_SIZES_MM[page.lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown page size {page!r}; choose from {sorted(PAGE_SIZES_MM)}"
        ) from exc

    modules = len(grid)
    if modules == 0 or any(len(row) != modules for row in grid):
        raise ValueError("marker grid must be a non-empty square")
    _check_fits(marker_size_mm, quiet_zone_mm, page, page_width_mm, page_height_mm)

    module_mm = marker_size_mm / modules
    pdf = PdfPage(page_width_mm, page_height_mm)

    # the marker sits centred horizontally, a fixed margin below the page top
    marker_left = (page_width_mm - marker_size_mm) / 2.0
    marker_top = page_height_mm - _TOP_MARGIN_MM - quiet_zone_mm
    marker_bottom = marker_top - marker_size_mm

    # PDF y grows upwards while the grid's first row is the top of the marker
    for row, cells in enumerate(grid):
        y = marker_top - (row + 1) * module_mm
        for col, black in enumerate(cells):
            if black:
                pdf.rectangle(marker_left + col * module_mm, y, module_mm, module_mm)

    cursor = marker_bottom - quiet_zone_mm
    if title:
        cursor -= 8.0
        pdf.text(marker_left, cursor, title, size_pt=11.0, bold=True)
    if note:
        cursor -= 5.5
        pdf.text(marker_left, cursor, note, size_pt=9.0, gray=0.35)
    if ruler_mm > 0:
        cursor -= 10.0
        pdf.text(marker_left, cursor, f"check: this line is {ruler_mm:g} mm",
                 size_pt=8.0, gray=0.35)
        cursor -= 1.5
        end = marker_left + ruler_mm
        pdf.line(marker_left, cursor, end, cursor, width_mm=0.3)
        pdf.line(marker_left, cursor, marker_left, cursor + 1.5, width_mm=0.3)
        pdf.line(end, cursor, end, cursor + 1.5, width_mm=0.3)

    pdf.save(path)
