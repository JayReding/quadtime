import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from quadtime.markers import (
    PAGE_SIZES_MM,
    marker_grid,
    render_marker_pdf,
    render_marker_stls,
)
from quadtime.pdfcanvas import PdfPage, mm_to_pt
from tests.stlhelpers import bounds, parse_stl, signed_volume

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


def _rectangle_bounds_mm(pdf_bytes: bytes):
    """Bounding box, in millimetres, of every filled rectangle in the page."""
    found = re.findall(
        rb"([0-9.]+) ([0-9.]+) ([0-9.]+) ([0-9.]+) re f", pdf_bytes
    )
    if not found:
        return None
    xs0 = [float(a) for a, _, _, _ in found]
    xs1 = [float(a) + float(c) for a, _, c, _ in found]
    ys0 = [float(b) for _, b, _, _ in found]
    ys1 = [float(b) + float(d) for _, b, _, d in found]
    to_mm = lambda pt: pt * 25.4 / 72.0  # noqa: E731
    return (to_mm(min(xs0)), to_mm(min(ys0)), to_mm(max(xs1)), to_mm(max(ys1)))


class MarkerGridTest(unittest.TestCase):
    def test_shape_and_black_border(self):
        grid = marker_grid(DICTIONARY, 0)
        modules = DICTIONARY.markerSize + 2  # 36h11 data plus a one-module border
        self.assertEqual(len(grid), modules)
        self.assertTrue(all(len(row) == modules for row in grid))
        for i in range(modules):
            self.assertTrue(grid[0][i], "top border must be black")
            self.assertTrue(grid[modules - 1][i], "bottom border must be black")
            self.assertTrue(grid[i][0], "left border must be black")
            self.assertTrue(grid[i][modules - 1], "right border must be black")

    def test_matches_a_downsampled_render(self):
        marker_id = 3
        grid = marker_grid(DICTIONARY, marker_id)
        modules = len(grid)
        pixels = 320
        image = cv2.aruco.generateImageMarker(DICTIONARY, marker_id, pixels, borderBits=1)
        blocks = image.reshape(
            modules, pixels // modules, modules, pixels // modules
        ).mean(axis=(1, 3))
        for row in range(modules):
            for col in range(modules):
                self.assertEqual(grid[row][col], blocks[row, col] < 128)

    def test_distinct_ids_produce_distinct_markers(self):
        self.assertNotEqual(marker_grid(DICTIONARY, 0), marker_grid(DICTIONARY, 1))


class MarkerPdfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "marker_0.pdf"

    def _render(self, size_mm=75.0, quiet_mm=15.0, page="a4", **kwargs):
        render_marker_pdf(
            self.path, marker_grid(DICTIONARY, 0), size_mm,
            quiet_zone_mm=quiet_mm, page=page, title="quadtime marker 0",
            note="print at 100% scale", **kwargs,
        )
        return self.path.read_bytes()

    def test_page_is_a4(self):
        data = self._render()
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertTrue(data.rstrip().endswith(b"%%EOF"))
        match = re.search(rb"/MediaBox \[0 0 ([0-9.]+) ([0-9.]+)\]", data)
        self.assertIsNotNone(match)
        width_mm = float(match.group(1)) * 25.4 / 72.0
        height_mm = float(match.group(2)) * 25.4 / 72.0
        self.assertAlmostEqual(width_mm, 210.0, places=3)
        self.assertAlmostEqual(height_mm, 297.0, places=3)

    def test_marker_measures_its_nominal_size(self):
        data = self._render(size_mm=75.0)
        bounds = _rectangle_bounds_mm(data)
        self.assertIsNotNone(bounds)
        left, bottom, right, top = bounds
        self.assertAlmostEqual(right - left, 75.0, places=2)
        self.assertAlmostEqual(top - bottom, 75.0, places=2)

    def test_marker_is_centred_with_the_expected_margin(self):
        data = self._render(size_mm=75.0, quiet_mm=15.0)
        left, _, right, top = _rectangle_bounds_mm(data)
        self.assertAlmostEqual(left, (210.0 - 75.0) / 2.0, places=2)
        self.assertAlmostEqual(right - left, 75.0, places=2)
        self.assertAlmostEqual(top, 297.0 - 25.0 - 15.0, places=2)

    def test_a_different_size_scales_the_marker(self):
        data = self._render(size_mm=40.0)
        left, bottom, right, top = _rectangle_bounds_mm(data)
        self.assertAlmostEqual(right - left, 40.0, places=2)
        self.assertAlmostEqual(top - bottom, 40.0, places=2)

    def test_letter_is_supported(self):
        data = self._render(page="letter")
        match = re.search(rb"/MediaBox \[0 0 ([0-9.]+) ([0-9.]+)\]", data)
        width_mm = float(match.group(1)) * 25.4 / 72.0
        height_mm = float(match.group(2)) * 25.4 / 72.0
        self.assertAlmostEqual(width_mm, PAGE_SIZES_MM["letter"][0], places=2)
        self.assertAlmostEqual(height_mm, PAGE_SIZES_MM["letter"][1], places=2)

    def test_rejects_an_unknown_page_size(self):
        with self.assertRaises(ValueError):
            self._render(page="banner")

    def test_rejects_a_marker_that_cannot_fit(self):
        with self.assertRaises(ValueError):
            self._render(size_mm=400.0)


class MarkerStlTest(unittest.TestCase):
    """The two stacked layers of a 3D printed marker."""

    SIZE_MM = 75.0
    QUIET_MM = 15.0
    BASE_T_MM = 1.5
    MARKER_T_MM = 0.6

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base_path = Path(self.tmp.name) / "marker_0_base.stl"
        self.pattern_path = Path(self.tmp.name) / "marker_0_pattern.stl"
        self.grid = marker_grid(DICTIONARY, 0)
        render_marker_stls(
            self.base_path, self.pattern_path, self.grid, self.SIZE_MM,
            quiet_zone_mm=self.QUIET_MM, base_thickness_mm=self.BASE_T_MM,
            marker_thickness_mm=self.MARKER_T_MM,
        )

    def test_base_is_a_plate_of_marker_plus_quiet_zone(self):
        _header, triangles = parse_stl(self.base_path)
        low, high = bounds(triangles)
        half = (self.SIZE_MM + 2 * self.QUIET_MM) / 2.0
        self.assertAlmostEqual(low[0], -half, places=6)
        self.assertAlmostEqual(high[0], half, places=6)
        self.assertAlmostEqual(low[1], -half, places=6)
        self.assertAlmostEqual(high[1], half, places=6)
        self.assertAlmostEqual(low[2], 0.0, places=6)
        self.assertAlmostEqual(high[2], self.BASE_T_MM, places=6)
        self.assertAlmostEqual(
            signed_volume(triangles), (2 * half) ** 2 * self.BASE_T_MM, places=4
        )

    def test_pattern_is_exactly_the_marker_square_on_top_of_the_base(self):
        _header, triangles = parse_stl(self.pattern_path)
        low, high = bounds(triangles)
        self.assertAlmostEqual(high[0] - low[0], self.SIZE_MM, places=6)
        self.assertAlmostEqual(high[1] - low[1], self.SIZE_MM, places=6)
        self.assertAlmostEqual(low[2], self.BASE_T_MM, places=6)
        self.assertAlmostEqual(high[2], self.BASE_T_MM + self.MARKER_T_MM, places=6)

    def test_pattern_volume_equals_its_black_modules(self):
        _header, triangles = parse_stl(self.pattern_path)
        black = sum(1 for row in self.grid for cell in row if cell)
        module_mm = self.SIZE_MM / len(self.grid)
        expected = black * module_mm ** 2 * self.MARKER_T_MM
        self.assertAlmostEqual(signed_volume(triangles), expected, places=3)

    def test_pattern_decodes_as_the_marker_when_seen_from_above(self):
        image = _top_view(self.pattern_path, len(self.grid), pixels_per_module=12)
        detected = _detect(image)
        self.assertEqual(detected, [0])

    def test_a_flipped_pattern_would_not_decode_the_same(self):
        # guards the row-to-y mapping: mirroring the grid vertically must change
        # the decoded id, otherwise the orientation test above proves nothing
        flipped = list(reversed(self.grid))
        path = Path(self.tmp.name) / "flipped.stl"
        render_marker_stls(path, Path(self.tmp.name) / "f2.stl", flipped, self.SIZE_MM,
                           quiet_zone_mm=self.QUIET_MM)
        self.assertNotEqual(_detect(_top_view(path, len(self.grid), 12)), [0])

    def test_rejects_a_grid_without_a_black_border(self):
        hollow = [[False] * 8 for _ in range(8)]
        hollow[3][3] = True
        with self.assertRaises(ValueError):
            render_marker_stls(self.base_path, self.pattern_path, hollow, self.SIZE_MM)

    def test_rejects_non_positive_thickness(self):
        with self.assertRaises(ValueError):
            render_marker_stls(self.base_path, self.pattern_path, self.grid,
                               self.SIZE_MM, base_thickness_mm=0.0)


def _top_view(stl_path, modules: int, pixels_per_module: int = 12):
    """Rasterise the upward-facing faces of a layer into a grayscale image."""
    _header, triangles = parse_stl(stl_path)
    low, high = bounds(triangles)
    size_mm = high[0] - low[0]
    per_mm = pixels_per_module * modules / size_mm
    margin = 2 * pixels_per_module
    width = int(round(size_mm * per_mm)) + 2 * margin
    height = width
    image = np.full((height, width), 255, dtype=np.uint8)
    for normal, a, b, c in triangles:
        if normal[2] < 0.9:  # only the top faces carry the marker pattern
            continue
        points = np.array(
            [[(x - low[0]) * per_mm + margin, (high[1] - y) * per_mm + margin]
             for x, y, _z in (a, b, c)],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [points], 0)
    return image


def _detect(image):
    detector = cv2.aruco.ArucoDetector(DICTIONARY, cv2.aruco.DetectorParameters())
    _corners, ids, _rejected = detector.detectMarkers(image)
    return [] if ids is None else ids.ravel().tolist()


class PdfCanvasTest(unittest.TestCase):
    def test_text_is_escaped_and_encodable(self):
        page = PdfPage(60.0, 40.0)
        page.text(5.0, 5.0, "ID (0) \\ dash \u2014 end", size_pt=8.0)
        data = page.to_bytes()
        self.assertIn(rb"(ID \(0\) \\ dash ? end) Tj", data)

    def test_geometry_is_emitted_in_points(self):
        page = PdfPage(210.0, 297.0)
        page.rectangle(10.0, 20.0, 30.0, 40.0)
        data = page.to_bytes()
        expected = (
            f"{mm_to_pt(10.0):.3f} {mm_to_pt(20.0):.3f} "
            f"{mm_to_pt(30.0):.3f} {mm_to_pt(40.0):.3f} re f"
        ).encode("latin-1")
        self.assertIn(expected, data)

    def test_xref_offsets_point_at_their_objects(self):
        page = PdfPage(100.0, 100.0)
        page.rectangle(0.0, 0.0, 10.0, 10.0)
        data = page.to_bytes()
        start_xref = int(re.search(rb"startxref\n([0-9]+)", data).group(1))
        self.assertEqual(data[start_xref:start_xref + 4], b"xref")
        table = data[start_xref:].split(b"\n")
        # table[0] is "xref", table[1] the subsection header, table[2] the free
        # entry for object 0, so object N lives at table[N + 2]
        self.assertEqual(table[1], b"0 7")
        self.assertTrue(table[2].endswith(b"f "))
        for number in range(1, 7):
            offset = int(table[number + 2][:10])
            self.assertEqual(data[offset:offset + len(f"{number} 0 obj")],
                             f"{number} 0 obj".encode("latin-1"))

    @unittest.skipUnless(shutil.which("pdfinfo"), "pdfinfo is not installed")
    def test_generated_pdf_parses_with_poppler(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "marker_0.pdf"
            render_marker_pdf(path, marker_grid(DICTIONARY, 0), 75.0,
                              quiet_zone_mm=15.0, title="marker 0")
            result = subprocess.run(["pdfinfo", str(path)],
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("A4", result.stdout)


if __name__ == "__main__":
    unittest.main()
