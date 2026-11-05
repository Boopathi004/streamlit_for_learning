"""Multi-format document parsing with OCR fallback.

Every parser returns a ParsedDocument made of Blocks. A Block is a piece of the
document that has a human-readable location ("page 2", "sheet Q3 rows 4-13",
"OCR image") so that citations can point the reader to the right place.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field


class ParseError(Exception):
    """Raised when a file cannot be turned into text."""


@dataclass
class Block:
    text: str
    location: str
    kind: str = "text"      # "text" (prose, overlap allowed) or "table" (row-based)
    prefix: str = ""        # repeated at the top of every chunk cut from this block


@dataclass
class ParsedDocument:
    filename: str
    file_type: str
    blocks: list[Block]
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="ignore")


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------------------------------------------------------
# OCR
# ----------------------------------------------------------------------------
def _ocr_image(img) -> tuple[str, float]:
    """Run Tesseract on a PIL image. Returns (text, mean word confidence 0-100)."""
    try:
        import pytesseract
        from PIL import ImageOps
    except ImportError as e:  # pragma: no cover
        raise ParseError("OCR needs `pytesseract` and `Pillow` (pip install pytesseract pillow).") from e

    try:
        pytesseract.get_tesseract_version()
    except Exception as e:
        raise ParseError(
            "Tesseract OCR engine not found. Install it first: "
            "Ubuntu/Debian `sudo apt install tesseract-ocr`, macOS `brew install tesseract`, "
            "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
        ) from e

    img = ImageOps.exif_transpose(img).convert("L")
    if img.width < 1200:  # small scans OCR better when upscaled
        scale = 1200 / img.width
        img = img.resize((int(img.width * scale), int(img.height * scale)))
    img = ImageOps.autocontrast(img)

    text = pytesseract.image_to_string(img, config="--psm 3")
    if len(text.strip()) < 40:  # sparse layouts (forms, stamps) sometimes need psm 6
        alt = pytesseract.image_to_string(img, config="--psm 6")
        if len(alt.strip()) > len(text.strip()):
            text = alt

    conf = 0.0
    try:
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        vals = [float(c) for c, w in zip(d["conf"], d["text"]) if w.strip() and float(c) >= 0]
        conf = sum(vals) / len(vals) if vals else 0.0
    except Exception:
        pass
    return text, conf


def _parse_image(filename: str, data: bytes) -> ParsedDocument:
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        raise ParseError(f"Could not open image: {e}") from e
    text, conf = _ocr_image(img)
    text = _clean(text)
    doc = ParsedDocument(filename, filename.rsplit(".", 1)[-1].lower(), [], ocr_used=True)
    if not text:
        doc.warnings.append("OCR found no readable text in this image.")
        return doc
    if conf and conf < 60:
        doc.warnings.append(f"Low OCR confidence ({conf:.0f}/100). Check the image quality; answers may be unreliable.")
    doc.blocks.append(Block(text, "scanned image (OCR)"))
    return doc


# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
_GAP = re.compile(r"[ \t]{3,}")


def _pdf_layout_text(page) -> str:
    """Layout-mode extraction keeps table rows on one line; wide gaps become ' | '."""
    try:
        raw = page.extract_text(extraction_mode="layout")
    except TypeError:  # older pypdf
        raw = page.extract_text()
    lines = []
    for line in (raw or "").split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        lines.append(_GAP.sub(" | ", stripped))   # internal wide gap = table cell boundary
    return "\n".join(lines)


def _parse_pdf(filename: str, data: bytes) -> ParsedDocument:
    try:
        from pypdf import PdfReader
    except ImportError as e:  # pragma: no cover
        raise ParseError("PDF support needs `pypdf`.") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ParseError("PDF is password protected.")
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Could not read PDF: {e}") from e

    doc = ParsedDocument(filename, "pdf", [])
    pdfium_doc = None
    for i, page in enumerate(reader.pages, start=1):
        text = _clean(_pdf_layout_text(page))
        if len(text) < 30:  # scanned page -> rasterise and OCR
            try:
                import pypdfium2 as pdfium

                if pdfium_doc is None:
                    pdfium_doc = pdfium.PdfDocument(data)
                bitmap = pdfium_doc[i - 1].render(scale=2.5)
                ocr_text, conf = _ocr_image(bitmap.to_pil())
                text = _clean(ocr_text)
                doc.ocr_used = True
                if conf and conf < 60:
                    doc.warnings.append(f"Page {i}: low OCR confidence ({conf:.0f}/100).")
            except ParseError as e:
                doc.warnings.append(f"Page {i} looks scanned but OCR is unavailable: {e}")
            except ImportError:
                doc.warnings.append(f"Page {i} looks scanned; install `pypdfium2` for OCR of scanned PDFs.")
        if text:
            doc.blocks.append(Block(text, f"page {i}"))
    if not doc.blocks:
        doc.warnings.append("No text could be extracted from this PDF.")
    return doc


# ----------------------------------------------------------------------------
# DOCX
# ----------------------------------------------------------------------------
def _parse_docx(filename: str, data: bytes) -> ParsedDocument:
    try:
        import docx
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except ImportError as e:  # pragma: no cover
        raise ParseError("DOCX support needs `python-docx`.") from e
    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as e:
        raise ParseError(f"Could not read DOCX: {e}") from e

    doc = ParsedDocument(filename, "docx", [])
    section, lines = "document start", []

    def flush():
        txt = _clean("\n".join(lines))
        if txt:
            doc.blocks.append(Block(txt, f"section: {section}"))
        lines.clear()

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            text = p.text.strip()
            if not text:
                continue
            style = (p.style.name if p.style is not None else "") or ""
            if style.lower().startswith(("heading", "title")):
                flush()
                section = text
                lines.append(text)
            else:
                lines.append(text)
        elif tag == "tbl":
            t = Table(child, d)
            rows = [[c.text.strip().replace("\n", " ") for c in r.cells] for r in t.rows]
            if not rows:
                continue
            header = rows[0]
            for r in rows[1:]:
                lines.append("; ".join(f"{h}: {v}" for h, v in zip(header, r) if v))
    flush()
    if not doc.blocks:
        doc.warnings.append("No text found in this DOCX.")
    return doc


# ----------------------------------------------------------------------------
# RTF / TXT / MD
# ----------------------------------------------------------------------------
def _parse_rtf(filename: str, data: bytes) -> ParsedDocument:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError as e:  # pragma: no cover
        raise ParseError("RTF support needs `striprtf`.") from e
    try:
        text = _clean(rtf_to_text(_decode(data), errors="ignore"))
    except Exception as e:
        raise ParseError(f"Could not read RTF: {e}") from e
    doc = ParsedDocument(filename, "rtf", [])
    if text:
        doc.blocks.append(Block(text, "document"))
    else:
        doc.warnings.append("No text found in this RTF.")
    return doc


def _parse_txt(filename: str, data: bytes) -> ParsedDocument:
    text = _clean(_decode(data))
    doc = ParsedDocument(filename, filename.rsplit(".", 1)[-1].lower(), [])
    if text:
        doc.blocks.append(Block(text, "document"))
    else:
        doc.warnings.append("The file is empty.")
    return doc


# ----------------------------------------------------------------------------
# CSV / XLSX  (rows become "column: value; column: value" so each row is self-describing)
# ----------------------------------------------------------------------------
ROWS_PER_BLOCK = 12


def _row_to_text(header: list[str], row: list[str]) -> str:
    return "; ".join(f"{h}: {v}" for h, v in zip(header, row) if str(v).strip() != "")


def _rows_to_blocks(rows: list[list[str]], header: list[str], label: str, prefix: str, first_row_no: int) -> list[Block]:
    blocks = []
    for start in range(0, len(rows), ROWS_PER_BLOCK):
        part = rows[start:start + ROWS_PER_BLOCK]
        lines = [t for t in (_row_to_text(header, r) for r in part) if t]
        if not lines:
            continue
        a = first_row_no + start
        b = a + len(part) - 1
        blocks.append(Block("\n".join(lines), f"{label} rows {a}-{b}", kind="table", prefix=prefix))
    return blocks


def _parse_csv(filename: str, data: bytes) -> ParsedDocument:
    text = _decode(data)
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    doc = ParsedDocument(filename, "csv", [])
    if len(rows) < 2:
        doc.warnings.append("CSV has no data rows.")
        return doc
    header = [h.strip() or f"column_{i+1}" for i, h in enumerate(rows[0])]
    doc.blocks = _rows_to_blocks(rows[1:], header, "csv", f"Table: {filename}", 2)
    return doc


def _parse_xlsx(filename: str, data: bytes) -> ParsedDocument:
    try:
        import openpyxl
    except ImportError as e:  # pragma: no cover
        raise ParseError("XLSX support needs `openpyxl`.") from e
    try:
        wb_v = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
        wb_f = openpyxl.load_workbook(io.BytesIO(data), data_only=False)
    except Exception as e:
        raise ParseError(f"Could not read XLSX: {e}") from e

    doc = ParsedDocument(filename, "xlsx", [])
    uncached = 0
    for ws in wb_v.worksheets:
        wf = wb_f[ws.title]
        grid: list[tuple[int, list[str]]] = []
        for r_idx, row in enumerate(ws.iter_rows(), start=1):
            vals = []
            for c in row:
                v = c.value
                if v is None:
                    f = wf.cell(row=c.row, column=c.column).value
                    if isinstance(f, str) and f.startswith("="):
                        v, uncached = f, uncached + 1   # formula without cached value
                if v is None:
                    vals.append("")
                elif isinstance(v, float) and v.is_integer():
                    vals.append(str(int(v)))
                else:
                    vals.append(str(v).strip())
            if any(vals):
                grid.append((r_idx, vals))
        if not grid:
            continue
        width = max(sum(1 for v in vals if v) for _, vals in grid)
        label = f"sheet '{ws.title}'"
        if width <= 1:  # notes sheet: one text cell per row
            text = "\n".join(next(v for v in vals if v) for _, vals in grid)
            doc.blocks.append(Block(text, label, kind="text"))
            continue
        # header = first row where most columns are filled with text (skips title rows)
        h_idx = 0
        for i, (_, vals) in enumerate(grid):
            filled = [v for v in vals if v]
            if len(filled) >= max(2, int(0.6 * width)) and not any(re.fullmatch(r"-?[\d,.]+", v) for v in filled):
                h_idx = i
                break
        preamble = [next((v for v in vals if v), "") for _, vals in grid[:h_idx]]
        header = [h or f"column_{i+1}" for i, h in enumerate(grid[h_idx][1])]
        if preamble:
            doc.blocks.append(Block("\n".join(p for p in preamble if p), f"{label} title", kind="text"))
        body = grid[h_idx + 1:]
        if body:
            first_no = body[0][0]
            doc.blocks += _rows_to_blocks([v for _, v in body], header, label, f"Sheet: {ws.title} ({filename})", first_no)
    if uncached:
        doc.warnings.append(
            f"{uncached} formula cell(s) had no saved result; the formula text was indexed instead. "
            "Open and re-save the workbook in Excel to store computed values."
        )
    if not doc.blocks:
        doc.warnings.append("No data found in this workbook.")
    return doc


# ----------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------
PARSERS = {
    "pdf": _parse_pdf, "docx": _parse_docx, "rtf": _parse_rtf,
    "txt": _parse_txt, "md": _parse_txt,
    "csv": _parse_csv, "xlsx": _parse_xlsx,
    "png": _parse_image, "jpg": _parse_image, "jpeg": _parse_image,
}


def parse_file(filename: str, data: bytes) -> ParsedDocument:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in PARSERS:
        raise ParseError(f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(PARSERS))}")
    if not data:
        raise ParseError("The file is empty.")
    doc = PARSERS[ext](filename, data)
    doc.file_type = ext
    return doc
