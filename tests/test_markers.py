import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2

from quadtime.markers import PAGE_SIZES_MM, marker_grid, render_marker_pdf
from quadtime.pdfcanvas import PdfPage, mm_to_pt

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
