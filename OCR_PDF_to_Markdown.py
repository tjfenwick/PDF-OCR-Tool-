#!/usr/bin/env python3
"""
cmm_pdf_ocr_to_markdown.py  (v5)

OCR CMM report PDFs and output Markdown while preserving report headers and
table-like formatting.

Workflow:
  1) Render each PDF page at an adjustable scale/DPI.
  2) Optionally preprocess the rendered image to improve OCR pickup.
  3) Run Tesseract OCR using word bounding boxes.
  4) Filter noise (drawing graphics, low-confidence garbage).
  5) Normalize common CMM OCR misreads.
  6) Reconstruct rows/columns from bounding boxes.
  7) Emit Markdown sections and Markdown tables.
  8) Optionally save debug images and raw OCR text to help dial in settings.

Dependencies:
  pip install pymupdf pillow pytesseract opencv-python

System dependency:
  Install Tesseract OCR:
    https://github.com/UB-Mannheim/tesseract/wiki

Examples:
  python cmm_pdf_ocr_to_markdown.py LANDMARK_UPITT_SIZE4_LEFT_01.pdf -o out.md
  python cmm_pdf_ocr_to_markdown.py *.pdf -o combined.md --render-scale 5 --debug-dir debug
  python cmm_pdf_ocr_to_markdown.py report.pdf -o out.md --pages 1,2,4,5 --save-raw

Tuning tips for CMM reports:
  - Start with --render-scale 5. Small CMM table text often needs higher scale.
  - If faint/colored text is missed, try --contrast 1.5 --sharpen 2.
  - If characters break apart, try --dilate 2.
  - If text blobs merge, lower dilation or use --erode 1 instead.
  - If OCR mixes drawing graphics with table text, crop or raise --min-conf.
  - Raise --noise-conf to filter more aggressively (default 55).
  - Lower --noise-alpha-ratio to be more lenient with symbol-heavy lines.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
except Exception as exc:
    raise SystemExit(
        "Missing dependency: pymupdf. Install with:  pip install pymupdf"
    ) from exc

try:
    from PIL import Image, ImageEnhance, ImageFilter
except Exception as exc:
    raise SystemExit(
        "Missing dependency: pillow. Install with:  pip install pillow"
    ) from exc

try:
    import pytesseract
except Exception as exc:
    raise SystemExit(
        "Missing dependency: pytesseract. Install with:  pip install pytesseract"
    ) from exc

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None

# ---------------------------------------------------------------------------
# Default Tesseract path -- update this to match YOUR install location.
# Can also be overridden at runtime with  --tesseract-path
# ---------------------------------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Users\30774508\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)


# ===================================================================
# Data classes
# ===================================================================
@dataclass
class OCRWord:
    """A single word recognised by Tesseract with its bounding box."""
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y_mid(self) -> float:
        return self.y + self.h / 2.0


@dataclass
class OCRRow:
    """A horizontal row of OCR words grouped by vertical proximity."""
    y: float
    words: List[OCRWord] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(
            w.text for w in sorted(self.words, key=lambda z: z.x)
        ).strip()

    @property
    def avg_conf(self) -> float:
        if not self.words:
            return 0.0
        return sum(w.conf for w in self.words) / len(self.words)


# ===================================================================
# Page / crop helpers
# ===================================================================
def parse_pages(page_spec: Optional[str], page_count: int) -> List[int]:
    """Parse a user page spec like '1,2,4-5' into 0-indexed page indices."""
    if not page_spec:
        return list(range(page_count))
    pages: List[int] = []
    for part in page_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = max(1, int(a))
            end = min(page_count, int(b))
            pages.extend(range(start - 1, end))
        else:
            p = int(part)
            if 1 <= p <= page_count:
                pages.append(p - 1)
    return sorted(set(pages))


def parse_crop(crop: Optional[str]) -> Tuple[int, int, int, int]:
    """Parse 'left,top,right,bottom' string into a tuple of ints."""
    if not crop:
        return (0, 0, 0, 0)
    vals = [int(v.strip()) for v in crop.split(",")]
    if len(vals) != 4:
        raise ValueError("--crop must be four integers: left,top,right,bottom")
    return (vals[0], vals[1], vals[2], vals[3])


# ===================================================================
# Image rendering & preprocessing
# ===================================================================
def render_page(page, render_scale: float) -> Image.Image:
    """Render a fitz page to a PIL Image at the given scale."""
    mat = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def crop_image(img: Image.Image, crop: Tuple[int, int, int, int]) -> Image.Image:
    """Crop rendered-pixel margins from an image."""
    l, t, r, b = crop
    if not any(crop):
        return img
    width, height = img.size
    return img.crop((l, t, max(l + 1, width - r), max(t + 1, height - b)))


def preprocess_image(
    img: Image.Image,
    grayscale: bool = True,
    contrast: float = 1.0,
    sharpen: int = 0,
    threshold: str = "none",
    threshold_value: int = 180,
    invert: bool = False,
    dilate: int = 0,
    erode: int = 0,
) -> Image.Image:
    """Condition an image for OCR.  Uses PIL by default; OpenCV when needed."""
    if grayscale:
        img = img.convert("L")

    if contrast and abs(contrast - 1.0) > 1e-6:
        img = ImageEnhance.Contrast(img).enhance(contrast)

    for _ in range(max(0, sharpen)):
        img = img.filter(ImageFilter.SHARPEN)

    needs_cv = threshold == "adaptive" or dilate > 0 or erode > 0

    if threshold == "global" and not needs_cv:
        img = img.point(lambda p: 255 if p > threshold_value else 0)
    elif threshold in {"global", "adaptive"} or needs_cv:
        if cv2 is None or np is None:
            raise SystemExit(
                "OpenCV is required for adaptive threshold / dilate / erode.\n"
                "Install with:  pip install opencv-python"
            )
        arr = np.array(img.convert("L"))
        if threshold == "global":
            _, arr = cv2.threshold(arr, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold == "adaptive":
            arr = cv2.adaptiveThreshold(
                arr, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 11,
            )
        if dilate > 0:
            kernel = np.ones((dilate, dilate), np.uint8)
            arr = cv2.dilate(arr, kernel, iterations=1)
        if erode > 0:
            kernel = np.ones((erode, erode), np.uint8)
            arr = cv2.erode(arr, kernel, iterations=1)
        img = Image.fromarray(arr)

    if invert:
        img = Image.eval(img.convert("L"), lambda p: 255 - p)

    return img


# ===================================================================
# Tesseract OCR
# ===================================================================
def tesseract_words(
    img: Image.Image, lang: str, psm: int, oem: int, min_conf: float
) -> List[OCRWord]:
    """Run Tesseract and return a list of OCRWord objects."""
    config = f"--oem {oem} --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(
        img, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )
    words: List[OCRWord] = []
    for i, text in enumerate(data.get("text", [])):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < min_conf:
            continue
        words.append(
            OCRWord(
                text=cleaned,
                conf=conf,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
            )
        )
    return words


# ===================================================================
# Row grouping
# ===================================================================
def group_words_into_rows(
    words: List[OCRWord], y_tolerance: int = 12
) -> List[OCRRow]:
    """Cluster OCR words into horizontal rows by vertical proximity."""
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w.y_mid, w.x))
    rows: List[OCRRow] = []
    for word in sorted_words:
        placed = False
        for row in rows:
            if abs(word.y_mid - row.y) <= y_tolerance:
                row.words.append(word)
                row.y = (
                    (row.y * (len(row.words) - 1) + word.y_mid) / len(row.words)
                )
                placed = True
                break
        if not placed:
            rows.append(OCRRow(y=word.y_mid, words=[word]))
    for row in rows:
        row.words.sort(key=lambda w: w.x)
    return sorted(rows, key=lambda r: r.y)


# ===================================================================
# Protected-content patterns  (rows we must NEVER filter as noise)
# ===================================================================
# NOTE: No trailing \d — OCR misreads digits as letters (6->S, 1->l, etc.)
_FCF_RE = re.compile(r'FCF(?:PROF|FLAT|LOC)', re.IGNORECASE)

_PROTECTED_KEYWORDS = [
    "PART NAME", "SER NUMBER", "STATS COUNT", "REV NUMBER", "SERIAL",
    "DATUM", "PROFILE", "FLATNESS", "POSITION", "SURFACE",
    "CRUCIATE", "ARTICULAR", "BOX PROFILE",
]
_ANGLE_LABEL_RE = re.compile(r'^\d+°$')
_SECTION_LABEL_RE = re.compile(r'^[A-Z]-[A-Z]$')
_PAGE_NUM_RE = re.compile(r'^\d+\s*/\s*\d+$')
_GARBAGE_RE = re.compile(
    r'^[\|\[\]\(\)\{\}<>\-\_\.\,\;\:\!\?\*\#\@\&\^\~\`\'\"\\\/ \t]+$'
)


def _is_protected(text: str) -> bool:
    """Return True if this text should NEVER be filtered as noise."""
    if _FCF_RE.search(text):
        return True
    upper = text.upper()
    for kw in _PROTECTED_KEYWORDS:
        if kw in upper:
            return True
    # Angle labels: "270deg", "90deg", "0deg"
    if _ANGLE_LABEL_RE.match(text.strip()):
        return True
    # Section sub-labels: "A-E", "E-B", "A-D"
    if _SECTION_LABEL_RE.match(text.strip()):
        return True
    return False


# ===================================================================
# Noise filtering
# ===================================================================
def is_noise_line(
    row_text: str,
    avg_conf: float,
    noise_conf: float = 55.0,
    noise_alpha_ratio: float = 0.3,
) -> bool:
    """Return True if a row looks like OCR garbage rather than real content."""
    text = row_text.strip()

    # NEVER filter protected content (FCF labels, section headings, etc.)
    if _is_protected(text):
        return False

    # Angle labels and section labels are always kept
    if _ANGLE_LABEL_RE.match(text):
        return False
    if _SECTION_LABEL_RE.match(text):
        return False

    # Page numbers like "1/5", "2/5", "47/5"
    if _PAGE_NUM_RE.match(text):
        return True

    # Very short strings are noise (unless angle labels with degree symbol)
    if len(text) < 4:
        if any(c.isdigit() for c in text) and '\u00b0' in text:
            return False  # keep "0deg", "90deg" etc.
        return True

    # Low average confidence across all words in the row
    if 0 < avg_conf < noise_conf:
        return True

    # Low proportion of alphanumeric characters (symbols / drawing junk)
    total = len(text)
    alpha_num = sum(c.isalnum() for c in text)
    if total > 0 and (alpha_num / total) < noise_alpha_ratio:
        return True

    # Pure punctuation / bracket garbage
    if _GARBAGE_RE.match(text):
        return True

    return False


# ===================================================================
# Text normalisation -- fix common CMM OCR misreads
# ===================================================================
_OCR_WORD_FIXES = {
    # NOTE: "FCFPROFS" is intentionally NOT auto-corrected.
    # OCR misreads digits 3, 5, 6, 8 as 'S' unpredictably.
    # Auto-correcting to a specific number would create wrong labels.
    "FCFLOCI":      "FCFLOC1",
    "FCRLOCI":      "FCFLOC1",
    "FOFPROF":      "FCFPROF",
    "Postion":      "Position",
    "Poston":       "Position",
    "SUFACE":       "SURFACE",
    "ASmE":         "ASME",
    "ASmE_y14_5":   "ASME_Y14_5",
    "ASmE_Y14_5":   "ASME_Y14_5",
    "ASME_Y14.5":   "ASME_Y14_5",
    "ASME_Y14 5":   "ASME_Y14_5",
    "ASME_y14_5":   "ASME_Y14_5",
}

_OCR_WORD_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _OCR_WORD_FIXES) + r')(?:\b|(?=[^a-zA-Z]))'
)


def _replace_word(m):
    return _OCR_WORD_FIXES.get(m.group(0), m.group(0))


def normalize_ocr_text(text: str) -> str:
    """Apply common CMM OCR corrections to a text string."""
    # Fix spaced-out decimals:  "0 . 0 1 5" -> "0.015"
    text = re.sub(r'(\d)\s+\.\s*(\d)', r'\1.\2', text)

    # Fix "oe" as standalone -> "0deg"
    text = re.sub(r'\boe\b', '0\u00b0', text)
    text = re.sub(r'\bo\u00b0\b', '0\u00b0', text)

    # Word-level replacements
    text = _OCR_WORD_PATTERN.sub(_replace_word, text)

    # Prefix fix: FOFPROF -> FCFPROF (catches FOFPROF2, FOFPROF7, etc.)
    text = re.sub(r'\bFOFPROF', 'FCFPROF', text)

    # SUFACE anywhere
    text = text.replace("SUFACE", "SURFACE")

    # Clean trailing/leading junk
    text = text.strip()

    return text


def normalize_cell(cell: str) -> str:
    """Clean a single table cell."""
    cell = cell.replace("\u2014", "-").replace("\u2013", "-")
    cell = re.sub(r"\s+", " ", cell).strip()
    cell = normalize_ocr_text(cell)
    return cell


# ===================================================================
# FCF label cleaner  -- strip GD&T frame garbage from FCF lines
# ===================================================================
# NOTE: \w* not \d+ — captures FCFPROFS, FCFFLAT1, FCFLOC1, etc.
_FCF_ID_RE = re.compile(
    r'(FCF(?:PROF|FLAT|LOC)\w*)',
    re.IGNORECASE,
)
_ASME_RE = re.compile(
    r'(ASME[_\s]?Y14[_\s.]?5\b.*)',
    re.IGNORECASE,
)
_VMINMAX_RE = re.compile(
    r'(VECTOR[_\s]?MIN[_\s]?MAX\s+ABOUT\s+\w+)',
    re.IGNORECASE,
)

def clean_fcf_label(text: str) -> str:
    """Clean a GD&T-laden FCF label line, keeping only meaningful tokens.

    Example inputs  -> outputs:
      "FCFPROF2 |} IN [003 (c)001/AB] ASME_Y14_5"
        -> "FCFPROF2 IN ASME_Y14_5"
      "FCFLOC1 Position | IN | | (G0.015M 058 |AIB |"
        -> "FCFLOC1 Position IN"
      "FCFPROF1 | ow | ASME_Y14_5 VECTOR_MIN_MAX ABOUT ZPLUS"
        -> "FCFPROF1 IN ASME_Y14_5 VECTOR_MIN_MAX ABOUT ZPLUS"
      "FCFPROF6 Datum Shift"
        -> "FCFPROF6 Datum Shift"
      "FCFPROFS | IN [...]"
        -> "FCFPROFS IN ..."
    """
    parts: List[str] = []

    # 1. Extract FCF identifier
    m_id = _FCF_ID_RE.search(text)
    if m_id:
        parts.append(m_id.group(1).upper())
    else:
        return text  # not an FCF line, return as-is

    # 2. Check for "Datum Shift" or "Summary" qualifiers
    upper = text.upper()
    if "DATUM SHIFT" in upper:
        parts.append("Datum Shift")
        return " ".join(parts)

    if "SUMMARY" in upper:
        # Keep everything from "Summary" onward (has useful config text)
        idx = upper.index("SUMMARY")
        parts.append(text[idx:].strip())
        return " ".join(parts)

    # 3. Check for "Position" qualifier
    if re.search(r'\bPosition\b', text, re.IGNORECASE):
        parts.append("Position")

    # 4. Check for "IN" (units indicator)
    if re.search(r'\bIN\b', text):
        parts.append("IN")
    # "ow" is OCR misread of "IN" inside GD&T frames
    elif re.search(r'\bow\b', text):
        parts.append("IN")

    # 5. Extract ASME reference
    m_asme = _ASME_RE.search(text)
    if m_asme:
        asme_text = m_asme.group(1).strip()
        # Also grab VECTOR_MIN_MAX ABOUT ZPLUS if present
        m_vmin = _VMINMAX_RE.search(text)
        if m_vmin:
            asme_text = re.sub(r'VECTOR.*', '', asme_text, flags=re.IGNORECASE).strip()
            asme_text = asme_text + " " + m_vmin.group(1).strip()
        # Normalize ASME tag
        asme_text = re.sub(r'ASME[_\s]?Y14[_\s.]?5', 'ASME_Y14_5', asme_text, flags=re.IGNORECASE)
        parts.append(asme_text.strip())
    elif _VMINMAX_RE.search(text):
        m_vmin = _VMINMAX_RE.search(text)
        parts.append("ASME_Y14_5 " + m_vmin.group(1).strip())

    return " ".join(parts)


# ===================================================================
# Table helpers
# ===================================================================
def split_row_by_gaps(row: OCRRow, min_gap: int = 22) -> List[str]:
    """Split a row into table cells based on horizontal gaps between words."""
    words = sorted(row.words, key=lambda w: w.x)
    if not words:
        return []
    cells: List[str] = []
    current = [words[0].text]
    prev = words[0]
    for w in words[1:]:
        gap = w.x - prev.x2
        if gap >= min_gap:
            cells.append(" ".join(current).strip())
            current = [w.text]
        else:
            current.append(w.text)
        prev = w
    cells.append(" ".join(current).strip())
    return [normalize_cell(c) for c in cells if c.strip()]


def fix_header_tol(headers: List[str]) -> List[str]:
    """If headers contain '+TOL' and bare 'TOL' (not '-TOL'), fix the bare one."""
    has_plus = any(h.strip() == "+TOL" for h in headers)
    fixed: List[str] = []
    for h in headers:
        stripped = h.strip()
        if has_plus and stripped == "TOL":
            fixed.append("-TOL")
        else:
            fixed.append(stripped)
    return fixed


def pad_row(cells: List[str], n: int) -> List[str]:
    """Pad or truncate a cell list to exactly n columns."""
    cells = [normalize_cell(c) for c in cells]
    if len(cells) < n:
        cells += [""] * (n - len(cells))
    return cells[:n]


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    """Emit a Markdown pipe-table from headers and data rows."""
    headers = [normalize_cell(h) for h in headers]
    headers = fix_header_tol(headers)
    lines: List[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(pad_row(r, len(headers))) + " |")
    return "\n".join(lines)


# ===================================================================
# Row classification
# ===================================================================
def _upper(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper())


def is_table_header(row_text: str) -> bool:
    """Detect rows that look like table column headers."""
    norm = _upper(row_text)
    header_sets = [
        ["FEATURE", "NOMINAL", "TOL", "MEAS"],
        ["SEGMENT", "SHIFT X", "SHIFT Y"],
        ["FEATURE", "AX", "NOMINAL", "MEAS", "DEV"],
    ]
    return any(all(tok in norm for tok in toks) for toks in header_sets)


_HEADING_KEYWORDS = [
    "SURFACE", "PROFILE", "FLATNESS", "POSITION", "DATUM",
    "BOX PROFILE", "CRUCIATE", "ARTICULAR",
]


def looks_like_data_row(row_text: str) -> bool:
    """Detect rows that look like table data.

    EXCLUDES rows that match section heading keywords with fewer than 4
    decimal numbers -- prevents '37.583 SURFACE' from being swallowed.
    """
    upper = row_text.upper()

    # Check if this looks more like a section heading than data
    if any(kw in upper for kw in _HEADING_KEYWORDS):
        decimal_count = len(re.findall(r'[-+]?\d+\.\d{2,}', row_text))
        if decimal_count < 4:
            return False

    if re.search(r'[-+]?\d+\.\d{2,}', row_text):
        return True
    data_keywords = [
        "Segment", "DAT_", "ART_", "SCN_", "Set", "Fixed", "DATUM_",
    ]
    if any(kw.upper() in upper for kw in data_keywords):
        return True
    return False


def is_section_heading(row_text: str) -> bool:
    """Detect CMM section headings (short lines with key terms)."""
    if len(row_text) > 90:
        return False
    upper = row_text.upper()
    keywords = [
        "PROFILE", "FLATNESS", "POSITION", "DATUM", "SURFACE",
        "Datum Shift", "Summary", "BOX PROFILE",
        "CRUCIATE GAP", "ARTICULAR", "AP BOX",
    ]
    return any(kw.upper() in upper for kw in keywords)


def is_fcf_label(row_text: str) -> bool:
    """Detect FCF identifier lines like 'FCFPROF1 IN ...' or 'FCFLOC1 Position'.

    NOTE: No trailing digit required — OCR misreads digits as letters.
    """
    return bool(
        re.match(r"^FCF(?:PROF|FLAT|LOC)", row_text.strip(), re.IGNORECASE)
    )


# ===================================================================
# Drawing-page detection
# ===================================================================
def detect_drawing_page(
    words: List[OCRWord],
    conf_threshold: float = 45.0,
    min_words: int = 8,
) -> bool:
    """Return True if the page is likely a drawing with no table data.

    A page is a drawing if:
      - It has fewer than min_words after confidence filtering, OR
      - The average confidence of all words is below conf_threshold.
    """
    if len(words) < min_words:
        return True
    avg = sum(w.conf for w in words) / len(words)
    return avg < conf_threshold


# ===================================================================
# Markdown conversion
# ===================================================================
def rows_to_markdown(
    rows: List[OCRRow],
    gap: int,
    noise_conf: float = 55.0,
    noise_alpha_ratio: float = 0.3,
    include_confidence_comments: bool = False,
) -> str:
    """Convert OCR rows to Markdown, filtering noise and preserving tables."""

    # ---- Step 1: filter noise rows ----
    clean_rows: List[OCRRow] = []
    for row in rows:
        text = normalize_ocr_text(row.text)
        if is_noise_line(text, row.avg_conf, noise_conf, noise_alpha_ratio):
            continue
        clean_rows.append(row)

    # ---- Step 2: build Markdown ----
    md: List[str] = []
    i = 0
    pending_headings: List[str] = []

    while i < len(clean_rows):
        raw_text = clean_rows[i].text
        text = normalize_ocr_text(raw_text)

        if not text:
            i += 1
            continue

        # Skip standalone page numbers like "1/5"
        if _PAGE_NUM_RE.match(text.strip()):
            i += 1
            continue

        # ---- Table header detected ----
        if is_table_header(text):
            # Flush pending headings
            for h in pending_headings:
                md.append(f"### {h}")
            pending_headings = []

            headers = split_row_by_gaps(clean_rows[i], gap)
            headers = fix_header_tol(headers)
            table_data: List[List[str]] = []
            i += 1

            while i < len(clean_rows):
                next_raw = clean_rows[i].text
                next_text = normalize_ocr_text(next_raw)
                if not next_text or _PAGE_NUM_RE.match(next_text.strip()):
                    i += 1
                    continue
                if is_table_header(next_text):
                    break
                # Section headings break out of the table
                if is_section_heading(next_text) and not looks_like_data_row(next_text):
                    break
                # FCF labels break out of the table
                if is_fcf_label(next_text):
                    break
                if looks_like_data_row(next_text):
                    table_data.append(split_row_by_gaps(clean_rows[i], gap))
                    i += 1
                else:
                    break

            if table_data:
                md.append(markdown_table(headers, table_data))
            else:
                md.append("**" + " | ".join(headers) + "**")
            md.append("")
            continue

        # ---- FCF label line ----
        if is_fcf_label(text):
            cleaned_label = clean_fcf_label(text)
            pending_headings.append(cleaned_label)
            if include_confidence_comments and clean_rows[i].words:
                pending_headings.append(
                    f"<!-- OCR avg confidence: {clean_rows[i].avg_conf:.1f} -->"
                )
            i += 1
            continue

        # ---- Section heading ----
        if is_section_heading(text):
            pending_headings.append(text)
            if include_confidence_comments and clean_rows[i].words:
                pending_headings.append(
                    f"<!-- OCR avg confidence: {clean_rows[i].avg_conf:.1f} -->"
                )
            i += 1
            continue

        # ---- Ordinary text (angle labels, metadata, etc.) ----
        # Flush pending headings first
        for h in pending_headings:
            md.append(f"### {h}")
        pending_headings = []

        md.append(text)
        if include_confidence_comments and clean_rows[i].words:
            md.append(
                f"<!-- OCR avg confidence: {clean_rows[i].avg_conf:.1f} -->"
            )
        i += 1

    # Flush any remaining headings
    for h in pending_headings:
        md.append(f"### {h}")

    # Remove excessive blank lines
    cleaned: List[str] = []
    blank = False
    for line in md:
        if line.strip() == "":
            if not blank:
                cleaned.append(line)
            blank = True
        else:
            cleaned.append(line)
            blank = False

    return "\n".join(cleaned).strip() + "\n"


# ===================================================================
# Debug / diagnostics
# ===================================================================
def save_confidence_csv(path: Path, words: List[OCRWord]) -> None:
    """Write per-word OCR confidence data to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "confidence", "left", "top", "width", "height"])
        for w in words:
            writer.writerow([w.text, f"{w.conf:.2f}", w.x, w.y, w.w, w.h])


# ===================================================================
# Main processing
# ===================================================================
def process_pdf(pdf_path: Path, args) -> str:
    """OCR a single PDF and return the Markdown string."""
    doc = fitz.open(pdf_path)
    pages = parse_pages(args.pages, len(doc))
    crop = parse_crop(args.crop)
    chunks: List[str] = []

    chunks.append(f"# OCR Output: {pdf_path.name}\n")
    chunks.append(
        "<!-- Settings: "
        f"render_scale={args.render_scale}, psm={args.psm}, oem={args.oem}, "
        f"threshold={args.threshold}, threshold_value={args.threshold_value}, "
        f"contrast={args.contrast}, sharpen={args.sharpen}, "
        f"dilate={args.dilate}, erode={args.erode}, "
        f"min_conf={args.min_conf}, noise_conf={args.noise_conf}, "
        f"noise_alpha_ratio={args.noise_alpha_ratio}, "
        f"row_tol={args.row_tol}, col_gap={args.col_gap}, "
        f"crop={args.crop or 'none'}"
        " -->\n"
    )

    debug_dir: Optional[Path] = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for page_index in pages:
        page = doc[page_index]
        rendered = render_page(page, args.render_scale)
        rendered = crop_image(rendered, crop)
        processed = preprocess_image(
            rendered,
            grayscale=not args.no_grayscale,
            contrast=args.contrast,
            sharpen=args.sharpen,
            threshold=args.threshold,
            threshold_value=args.threshold_value,
            invert=args.invert,
            dilate=args.dilate,
            erode=args.erode,
        )

        stem = f"{pdf_path.stem}_p{page_index + 1:02d}"

        if debug_dir:
            rendered.save(debug_dir / f"{stem}_rendered.png")
            processed.save(debug_dir / f"{stem}_processed.png")

        words = tesseract_words(
            processed, args.lang, args.psm, args.oem, args.min_conf
        )

        # Check for drawing-only pages
        if detect_drawing_page(words):
            chunks.append(f"\n## Page {page_index + 1}\n")
            chunks.append(
                "*This page appears to be a drawing/graphic with no "
                "extractable table data.*\n"
            )
            if debug_dir:
                save_confidence_csv(debug_dir / f"{stem}_confidence.csv", words)
            continue

        rows = group_words_into_rows(words, y_tolerance=args.row_tol)
        page_md = rows_to_markdown(
            rows,
            gap=args.col_gap,
            noise_conf=args.noise_conf,
            noise_alpha_ratio=args.noise_alpha_ratio,
            include_confidence_comments=args.confidence_comments,
        )

        if args.save_raw:
            raw_text = pytesseract.image_to_string(
                processed,
                lang=args.lang,
                config=f"--oem {args.oem} --psm {args.psm} -c preserve_interword_spaces=1",
            )
            if debug_dir:
                (debug_dir / f"{stem}_raw.txt").write_text(raw_text, encoding="utf-8")

        if debug_dir:
            save_confidence_csv(debug_dir / f"{stem}_confidence.csv", words)

        chunks.append(f"\n## Page {page_index + 1}\n")
        chunks.append(page_md)

    doc.close()
    return "\n".join(chunks).strip() + "\n"


# ===================================================================
# CLI
# ===================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OCR CMM PDF reports to Markdown tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "pdfs", nargs="+",
        help="Input PDF file(s). Wildcards are supported by your shell.",
    )
    p.add_argument(
        "-o", "--output", default="cmm_ocr_output.md",
        help="Output Markdown file (default: cmm_ocr_output.md).",
    )
    p.add_argument(
        "--pages",
        help="Pages to OCR, e.g. 1,2,4-5. Default: all pages.",
    )

    # -- Render / OCR --
    g = p.add_argument_group("Render & OCR")
    g.add_argument(
        "--render-scale", type=float, default=5.0,
        help="PDF render scale factor. Try 3-6 for small CMM text (default: 5).",
    )
    g.add_argument("--lang", default="eng", help="Tesseract language (default: eng).")
    g.add_argument(
        "--psm", type=int, default=6,
        help="Tesseract page segmentation mode. Try 4, 6, 11, 12 (default: 6).",
    )
    g.add_argument(
        "--oem", type=int, default=3,
        help="Tesseract OCR engine mode (default: 3).",
    )
    g.add_argument(
        "--min-conf", type=float, default=20.0,
        help="Discard individual words below this confidence (default: 20).",
    )

    # -- Image preprocessing --
    g = p.add_argument_group("Image preprocessing")
    g.add_argument(
        "--no-grayscale", action="store_true",
        help="Keep RGB instead of converting to grayscale.",
    )
    g.add_argument(
        "--contrast", type=float, default=1.4,
        help="Contrast multiplier (default: 1.4).",
    )
    g.add_argument(
        "--sharpen", type=int, default=1,
        help="Apply PIL sharpen this many times (default: 1).",
    )
    g.add_argument(
        "--threshold", choices=["none", "global", "adaptive"], default="adaptive",
        help="Thresholding mode (default: adaptive).",
    )
    g.add_argument(
        "--threshold-value", type=int, default=180,
        help="Global threshold cutoff 0-255 (default: 180).",
    )
    g.add_argument("--invert", action="store_true", help="Invert after preprocessing.")
    g.add_argument(
        "--dilate", type=int, default=1,
        help="Dilate kernel size for broken characters (default: 1).",
    )
    g.add_argument(
        "--erode", type=int, default=0,
        help="Erode kernel size for merged characters (default: 0).",
    )
    g.add_argument(
        "--crop",
        help="Crop rendered pixels: left,top,right,bottom.",
    )

    # -- Layout reconstruction --
    g = p.add_argument_group("Layout reconstruction")
    g.add_argument(
        "--row-tol", type=int, default=12,
        help="Y tolerance (px) for grouping words into rows (default: 12).",
    )
    g.add_argument(
        "--col-gap", type=int, default=22,
        help="Min horizontal gap (px) that starts a new table cell (default: 22).",
    )

    # -- Noise filtering --
    g = p.add_argument_group("Noise filtering")
    g.add_argument(
        "--noise-conf", type=float, default=55.0,
        help="Rows with avg confidence below this are filtered as noise (default: 55).",
    )
    g.add_argument(
        "--noise-alpha-ratio", type=float, default=0.3,
        help="Rows with alphanumeric ratio below this are filtered (default: 0.3).",
    )

    # -- Diagnostics --
    g = p.add_argument_group("Diagnostics")
    g.add_argument(
        "--debug-dir",
        help="Save rendered/processed PNGs and confidence CSVs here.",
    )
    g.add_argument(
        "--save-raw", action="store_true",
        help="Save raw Tesseract text output into debug directory.",
    )
    g.add_argument(
        "--confidence-comments", action="store_true",
        help="Add HTML comments with average OCR confidence per row.",
    )
    g.add_argument(
        "--tesseract-path",
        help="Override path to tesseract.exe.",
    )

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Override Tesseract path if requested
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path

    all_md: List[str] = []
    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"WARNING: file not found: {pdf_path}", file=sys.stderr)
            continue
        print(f"Processing: {pdf_path.name} ...")
        all_md.append(process_pdf(pdf_path, args))

    if not all_md:
        print("No valid PDF files were processed.", file=sys.stderr)
        return 2

    out = Path(args.output)
    out.write_text("\n\n---\n\n".join(all_md), encoding="utf-8")
    print(f"Wrote Markdown OCR output to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
