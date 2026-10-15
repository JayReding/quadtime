"""Binary STL output for a small, axis-aligned mesh.

Just enough to describe a two-layer marker tile: boxes and extruded grid
regions, written as a binary STL.  Binary STL is a bare triangle soup that every
slicer reads, so this needs no dependency and no format negotiation.

Winding is counter-clockwise seen from *outside* the solid, which is what the
slicers and the stored per-facet normal expect.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

Vertex = tuple[float, float, float]
Triangle = tuple[Vertex, Vertex, Vertex]


def _normal(a: Vertex, b: Vertex, c: Vertex) -> Vertex:
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length < 1e-12:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


class Solid:
    """An accumulating triangle mesh, written out as a binary STL."""

    def __init__(self, name: str = "quadtime"):
        self.name = name
        self.triangles: list[Triangle] = []

    def add_triangle(self, a: Vertex, b: Vertex, c: Vertex) -> None:
        self.triangles.append((a, b, c))

    def add_quad(self, a: Vertex, b: Vertex, c: Vertex, d: Vertex) -> None:
        self.add_triangle(a, b, c)
        self.add_triangle(a, c, d)

    def add_box(self, low: Vertex, high: Vertex) -> None:
        """A closed axis-aligned box from ``low`` to ``high``."""
        (x0, y0, z0), (x1, y1, z1) = low, high
        self.add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))  # +z
        self.add_quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))  # -z
        self.add_quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))  # -x
        self.add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))  # +x
        self.add_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))  # -y
        self.add_quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))  # +y

    def add_cells(
        self,
        cells: Iterable[tuple[int, int]],
        cell_mm: float,
        origin: tuple[float, float],
        z0: float,
        z1: float,
    ) -> None:
        """Extrude a set of ``(column, row)`` grid cells into a watertight slab.

        Cell ``(i, j)`` covers ``x`` from ``origin[0] + i*cell_mm`` and ``y``
        from ``origin[1] + j*cell_mm``.  Only faces on the outside of the region
        are emitted, so the result has no internal or duplicated geometry.
        """
        region = set(cells)
        for i, j in region:
            x0 = origin[0] + i * cell_mm
            x1 = x0 + cell_mm
            y0 = origin[1] + j * cell_mm
            y1 = y0 + cell_mm
            self.add_quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
            self.add_quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))
            if (i - 1, j) not in region:
                self.add_quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))
            if (i + 1, j) not in region:
                self.add_quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))
            if (i, j - 1) not in region:
                self.add_quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))
            if (i, j + 1) not in region:
                self.add_quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))

    def to_bytes(self) -> bytes:
        header = self.name.encode("ascii", "replace")[:80].ljust(80, b"\0")
        out = bytearray(header)
        out += struct.pack("<I", len(self.triangles))
        for a, b, c in self.triangles:
            out += struct.pack("<12fH", *_normal(a, b, c), *a, *b, *c, 0)
        return bytes(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())
