"""A minimal, dependency-free PDF writer.

Only what a printable marker page needs: filled rectangles, straight lines and
text in a single-page document.  Geometry is expressed in millimetres but is
emitted in PDF user-space units (1/72 inch), so a marker drawn at 75 mm really
measures 75 mm on paper at any printer resolution and there is nothing to
scale at print time.
"""

from __future__ import annotations

from pathlib import Path

POINTS_PER_INCH = 72.0
MM_PER_INCH = 25.4


def mm_to_pt(value: float) -> float:
    """Convert millimetres to PDF points."""
    return value * POINTS_PER_INCH / MM_PER_INCH


def _escape(value: str) -> str:
    """Escape a string for a PDF literal, dropping characters PDF cannot show."""
    text = str(value).encode("latin-1", "replace").decode("latin-1")
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


class PdfPage:
    """A single PDF page drawn in millimetres, with the origin bottom-left."""

    def __init__(self, width_mm: float, height_mm: float):
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self._ops: list[str] = []

    def rectangle(self, x_mm: float, y_mm: float, width_mm: float, height_mm: float,
                  gray: float = 0.0) -> None:
        """A filled rectangle; ``x``/``y`` is the lower-left corner."""
        self._ops.append(
            f"{gray:.4f} g "
            f"{mm_to_pt(x_mm):.3f} {mm_to_pt(y_mm):.3f} "
            f"{mm_to_pt(width_mm):.3f} {mm_to_pt(height_mm):.3f} re f"
        )

    def line(self, x0_mm: float, y0_mm: float, x1_mm: float, y1_mm: float,
             width_mm: float = 0.2, gray: float = 0.0) -> None:
        """A straight stroked line."""
        self._ops.append(
            f"{mm_to_pt(width_mm):.3f} w {gray:.4f} G "
            f"{mm_to_pt(x0_mm):.3f} {mm_to_pt(y0_mm):.3f} m "
            f"{mm_to_pt(x1_mm):.3f} {mm_to_pt(y1_mm):.3f} l S"
        )

    def text(self, x_mm: float, y_mm: float, value: str, size_pt: float = 10.0,
             gray: float = 0.0, bold: bool = False) -> None:
        """Text with its baseline start at ``x_mm``/``y_mm``."""
        font = "F2" if bold else "F1"
        self._ops.append(
            f"BT /{font} {size_pt:.2f} Tf {gray:.4f} g "
            f"{mm_to_pt(x_mm):.3f} {mm_to_pt(y_mm):.3f} Td ({_escape(value)}) Tj ET"
        )

    def to_bytes(self) -> bytes:
        body = "\n".join(self._ops).encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {mm_to_pt(self.width_mm):.3f} {mm_to_pt(self.height_mm):.3f}] "
                f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> "
                f"/Contents 4 0 R >>"
            ).encode("latin-1"),
            b"<< /Length " + str(len(body)).encode("latin-1") + b" >>\nstream\n"
            + body + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            b"/Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
            b"/Encoding /WinAnsiEncoding >>",
        ]

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets: list[int] = []
        for number, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode("latin-1") + obj + b"\nendobj\n"

        start_xref = len(out)
        size = len(objects) + 1
        out += f"xref\n0 {size}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode("latin-1")
        out += (
            f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
            f"startxref\n{start_xref}\n%%EOF\n"
        ).encode("latin-1")
        return bytes(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())
