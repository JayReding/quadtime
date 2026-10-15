"""Shared helpers for inspecting the generated triangle meshes.

Triangles are accepted either as the in-memory ``(a, b, c)`` form or the parsed
``(normal, a, b, c)`` form, so the same helpers work on a :class:`Solid` and on
a file read back from disk.
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path


def parse_stl(data: bytes | str | Path):
    """Read a binary STL into ``(header, [(normal, a, b, c), ...])``."""
    if isinstance(data, (str, Path)):
        data = Path(data).read_bytes()
    if len(data) < 84:
        raise ValueError("file is too short to be a binary STL")
    header = data[:80]
    (count,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * count:
        raise ValueError(
            f"size {len(data)} does not match {count} triangles "
            f"({84 + 50 * count} expected)"
        )
    triangles = []
    for index in range(count):
        values = struct.unpack_from("<12fH", data, 84 + 50 * index)
        triangles.append((values[0:3], values[3:6], values[6:9], values[9:12]))
    return header, triangles


def _corners(triangle):
    if len(triangle) == 4:
        _normal, a, b, c = triangle
        return a, b, c
    a, b, c = triangle
    return a, b, c


def signed_volume(triangles) -> float:
    """Enclosed volume via the divergence theorem (positive if wound outward)."""
    total = 0.0
    for triangle in triangles:
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = _corners(triangle)
        total += (
            ax * (by * cz - bz * cy)
            - ay * (bx * cz - bz * cx)
            + az * (bx * cy - by * cx)
        )
    return total / 6.0


def bounds(triangles):
    """``(min_xyz, max_xyz)`` over every vertex."""
    points = [vertex for triangle in triangles for vertex in _corners(triangle)]
    lows = tuple(min(point[i] for point in points) for i in range(3))
    highs = tuple(max(point[i] for point in points) for i in range(3))
    return lows, highs


def edge_counts(triangles, places: int = 6) -> Counter:
    """How many triangles meet at each undirected edge."""
    edges: Counter = Counter()
    for triangle in triangles:
        corners = [tuple(round(value, places) for value in vertex)
                   for vertex in _corners(triangle)]
        for i in range(3):
            edge = tuple(sorted((corners[i], corners[(i + 1) % 3])))
            edges[edge] += 1
    return edges
