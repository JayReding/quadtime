import struct
import tempfile
import unittest
from pathlib import Path

from quadtime.stl import Solid
from tests.stlhelpers import bounds, edge_counts, parse_stl, signed_volume


class BoxTest(unittest.TestCase):
    def _box(self, low=(0.0, 0.0, 0.0), high=(2.0, 3.0, 4.0)):
        solid = Solid("test box")
        solid.add_box(low, high)
        return solid

    def test_triangle_count_and_bounds(self):
        solid = self._box()
        self.assertEqual(len(solid.triangles), 12)
        low, high = bounds(solid.triangles)
        self.assertEqual(low, (0.0, 0.0, 0.0))
        self.assertEqual(high, (2.0, 3.0, 4.0))

    def test_volume_is_the_box_volume(self):
        self.assertAlmostEqual(signed_volume(self._box().triangles), 24.0, places=6)

    def test_volume_is_positive_so_winding_points_outward(self):
        self.assertGreater(signed_volume(self._box().triangles), 0.0)

    def test_every_edge_is_shared_by_exactly_two_triangles(self):
        counts = edge_counts(self._box().triangles)
        self.assertTrue(counts)
        self.assertEqual(set(counts.values()), {2})

    def test_normals_point_away_from_the_centre(self):
        _header, triangles = parse_stl(self._box().to_bytes())
        centre = (1.0, 1.5, 2.0)
        for normal, a, b, c in triangles:
            centroid = tuple((a[i] + b[i] + c[i]) / 3.0 for i in range(3))
            outward = tuple(centroid[i] - centre[i] for i in range(3))
            dot = sum(normal[i] * outward[i] for i in range(3))
            self.assertGreater(dot, 0.0)


class CellsTest(unittest.TestCase):
    """Extruding grid regions, the building block of the marker layer."""

    def test_a_single_cell_is_a_box(self):
        solid = Solid()
        solid.add_cells({(0, 0)}, 2.0, (0.0, 0.0), 1.0, 4.0)
        low, high = bounds(solid.triangles)
        self.assertEqual(low, (0.0, 0.0, 1.0))
        self.assertEqual(high, (2.0, 2.0, 4.0))
        self.assertAlmostEqual(signed_volume(solid.triangles), 2 * 2 * 3, places=6)

    def test_a_two_by_two_block_is_closed(self):
        solid = Solid()
        solid.add_cells({(0, 0), (1, 0), (0, 1), (1, 1)}, 1.0, (0.0, 0.0), 0.0, 2.0)
        counts = edge_counts(solid.triangles)
        self.assertEqual(set(counts.values()), {2})
        self.assertAlmostEqual(signed_volume(solid.triangles), 4 * 2, places=6)

    def test_internal_faces_are_not_emitted(self):
        # a 2x1 bar: 2 cells x (top, bottom) + 6 exposed walls = 10 quads
        solid = Solid()
        solid.add_cells({(0, 0), (1, 0)}, 1.0, (0.0, 0.0), 0.0, 1.0)
        self.assertEqual(len(solid.triangles), 20)

    def test_offset_origin_places_the_region(self):
        solid = Solid()
        solid.add_cells({(0, 0)}, 2.0, (-3.0, -5.0), 0.0, 1.0)
        low, high = bounds(solid.triangles)
        self.assertEqual(low, (-3.0, -5.0, 0.0))
        self.assertEqual(high, (-1.0, -3.0, 1.0))

    def test_a_disconnected_region_still_has_the_right_volume(self):
        solid = Solid()
        solid.add_cells({(0, 0), (4, 4)}, 1.0, (0.0, 0.0), 0.0, 1.0)
        self.assertAlmostEqual(signed_volume(solid.triangles), 2.0, places=6)

    def test_corner_touching_cells_share_a_four_way_edge(self):
        # Only diagonal neighbours: the shared corner edge is met by four
        # triangles.  Slicers accept this, and the volume stays correct.
        solid = Solid()
        solid.add_cells({(0, 0), (1, 1)}, 1.0, (0.0, 0.0), 0.0, 1.0)
        counts = edge_counts(solid.triangles)
        self.assertIn(4, counts.values())
        self.assertTrue(all(count % 2 == 0 for count in counts.values()))
        self.assertAlmostEqual(signed_volume(solid.triangles), 2.0, places=6)


class StlFormatTest(unittest.TestCase):
    def test_binary_layout(self):
        solid = Solid("quadtime marker 0 white base")
        solid.add_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        data = solid.to_bytes()
        self.assertEqual(len(data), 84 + 50 * 12)
        self.assertEqual(data[:8], b"quadtime")
        self.assertEqual(len(data[:80]), 80)
        self.assertEqual(struct.unpack_from("<I", data, 80)[0], 12)
        self.assertEqual(struct.unpack_from("<H", data, 84 + 48)[0], 0)

    def test_stored_normals_are_unit_length(self):
        solid = Solid()
        solid.add_box((0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
        _header, triangles = parse_stl(solid.to_bytes())
        for normal, _a, _b, _c in triangles:
            length = sum(component * component for component in normal) ** 0.5
            self.assertAlmostEqual(length, 1.0, places=5)

    def test_save_and_reload_matches_bounds(self):
        solid = Solid()
        solid.add_box((0.0, 0.0, 0.0), (5.0, 6.0, 7.0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "box.stl"
            solid.save(path)
            header, triangles = parse_stl(path)
        self.assertTrue(header.startswith(b"quadtime") or header.strip(b"\0") == b"")
        self.assertEqual(len(triangles), 12)
        low, high = bounds(triangles)
        self.assertEqual(high, (5.0, 6.0, 7.0))

    def test_header_is_padded_and_truncated_to_eighty_bytes(self):
        solid = Solid("x" * 200)
        data = solid.to_bytes()
        self.assertEqual(len(data[:80]), 80)
        self.assertNotIn(b"x", data[80:84])


if __name__ == "__main__":
    unittest.main()
