#!/usr/bin/env python3
"""
pdf_ocr_gui.py

Tkinter GUI front-end for OCR_PDF_to_Markdown.py.

Features
========
* Add PDFs or whole folders of PDFs, manage the queue, reorder/remove.
* Output to one or more formats: Markdown, Plain text, JSON, CSV, HTML,
  Searchable PDF (text layer overlaid on the original page images).
* Per-file output or one combined file per format.
* Full control over OCR / preprocessing / layout / noise settings via tabs.
* Presets: built-in (Default, Fast, High quality, Tables) plus save/load
  custom presets as JSON.
* Auto-detects Tesseract from PATH, common install locations, or a sibling
  ``tesseract/`` folder (used by the bundled portable .exe build).
* Threaded processing with a live log, per-page progress, and cancel.
* Remembers last-used settings between sessions.

Designed to be packaged as a single-folder PyInstaller .exe; see
``build_exe.bat`` and ``.github/workflows/build-exe.yml``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Imports that may live next to this script or be frozen into a PyInstaller bundle
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    APP_DIR = Path(sys.executable).parent
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_DIR

sys.path.insert(0, str(BUNDLE_DIR))

import OCR_PDF_to_Markdown as ocr_engine  # noqa: E402
import fitz  # PyMuPDF                       noqa: E402
import pytesseract  # noqa: E402
from PIL import Image  # noqa: E402

APP_TITLE = "PDF OCR Tool"
APP_VERSION = "1.0.0"
CONFIG_FILE = Path.home() / ".pdf_ocr_gui_config.json"

OUTPUT_FORMATS: List[Tuple[str, str, str]] = [
    # (key, label, extension)
    ("markdown",       "Markdown (.md)",          ".md"),
    ("text",           "Plain text (.txt)",       ".txt"),
    ("json",           "JSON (.json)",            ".json"),
    ("csv",            "CSV words (.csv)",        ".csv"),
    ("html",           "HTML (.html)",            ".html"),
    ("searchable_pdf", "Searchable PDF (.pdf)",   ".pdf"),
]


# ---------------------------------------------------------------------------
# Tesseract discovery
# ---------------------------------------------------------------------------
def candidate_tesseract_paths() -> List[Path]:
    candidates: List[Path] = []

    # 1. Bundled portable Tesseract sitting next to the .exe / script.
    for base in (APP_DIR, BUNDLE_DIR):
        candidates.append(base / "tesseract" / "tesseract.exe")
        candidates.append(base / "tesseract" / "tesseract")

    # 2. PATH lookup.
    which = shutil.which("tesseract")
    if which:
        candidates.append(Path(which))

    # 3. Common Windows install locations.
    env = os.environ
    for env_key, sub in [
        ("ProgramFiles",       "Tesseract-OCR"),
        ("ProgramFiles(x86)",  "Tesseract-OCR"),
        ("LocalAppData",       r"Programs\Tesseract-OCR"),
        ("LocalAppData",       "Tesseract-OCR"),
    ]:
        root = env.get(env_key)
        if root:
            candidates.append(Path(root) / sub / "tesseract.exe")

    return candidates


def find_tesseract() -> Optional[Path]:
    for p in candidate_tesseract_paths():
        try:
            if p and p.exists():
                return p
        except OSError:
            continue
    return None


def tesseract_works(path: Optional[str]) -> Tuple[bool, str]:
    cmd = path or pytesseract.pytesseract.tesseract_cmd
    if not cmd:
        return (False, "No path configured.")
    try:
        proc = subprocess.run(
            [cmd, "--version"], capture_output=True, text=True, timeout=10
        )
        if proc.returncode == 0:
            first_line = (proc.stdout or proc.stderr).splitlines()[0:1]
            return (True, first_line[0] if first_line else "ok")
        return (False, (proc.stderr or proc.stdout or "Unknown error").strip())
    except FileNotFoundError:
        return (False, f"File not found: {cmd}")
    except Exception as exc:
        return (False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Settings dataclass + presets
# ---------------------------------------------------------------------------
@dataclass
class OcrSettings:
    inputs: List[str] = field(default_factory=list)
    output_dir: str = ""
    output_formats: List[str] = field(default_factory=lambda: ["markdown"])
    output_mode: str = "per_file"          # or "combined"
    combined_basename: str = "ocr_output"

    tesseract_path: str = ""
    lang: str = "eng"
    psm: int = 6
    oem: int = 3
    render_scale: float = 5.0
    min_conf: float = 20.0
    pages: str = ""

    no_grayscale: bool = False
    contrast: float = 1.4
    sharpen: int = 1
    threshold: str = "adaptive"
    threshold_value: int = 180
    invert: bool = False
    dilate: int = 1
    erode: int = 0
    crop: str = ""

    row_tol: int = 12
    col_gap: int = 22
    noise_conf: float = 55.0
    noise_alpha_ratio: float = 0.3

    debug_dir: str = ""
    save_raw: bool = False
    confidence_comments: bool = False

    open_when_done: bool = True
    show_notification: bool = True

    # ---- enterprise / accuracy controls ----
    workers: int = 0                    # 0 = auto pick
    use_text_layer: bool = True         # skip OCR for pages with embedded text
    deskew: bool = False
    denoise: bool = False
    best_accuracy: bool = False         # multi-PSM voting (~2x slower)
    write_audit_log: bool = True
    seen_welcome: bool = False
    recent_files: List[str] = field(default_factory=list)
    recent_folders: List[str] = field(default_factory=list)
    watch_folder: str = ""
    watch_interval_sec: int = 5


PRESETS: Dict[str, Dict[str, Any]] = {
    "Default": {},
    "Fast (lower quality)": {
        "render_scale": 3.0,
        "sharpen": 0,
        "threshold": "global",
        "dilate": 0,
    },
    "High quality": {
        "render_scale": 6.0,
        "contrast": 1.6,
        "sharpen": 2,
        "threshold": "adaptive",
        "dilate": 1,
    },
    "Plain document (no tables)": {
        "render_scale": 4.0,
        "psm": 3,
        "col_gap": 999,
        "noise_alpha_ratio": 0.2,
    },
    "CMM report tables": {
        "render_scale": 5.0,
        "psm": 6,
        "contrast": 1.4,
        "sharpen": 1,
        "threshold": "adaptive",
        "dilate": 1,
        "row_tol": 12,
        "col_gap": 22,
        "noise_conf": 55.0,
    },
}


def settings_to_args(s: OcrSettings) -> argparse.Namespace:
    """Build an argparse.Namespace that matches what ``process_pdf`` expects."""
    return argparse.Namespace(
        pdfs=s.inputs,
        output="",
        pages=s.pages or None,
        render_scale=s.render_scale,
        lang=s.lang,
        psm=s.psm,
        oem=s.oem,
        min_conf=s.min_conf,
        no_grayscale=s.no_grayscale,
        contrast=s.contrast,
        sharpen=s.sharpen,
        threshold=s.threshold,
        threshold_value=s.threshold_value,
        invert=s.invert,
        dilate=s.dilate,
        erode=s.erode,
        crop=s.crop or None,
        row_tol=s.row_tol,
        col_gap=s.col_gap,
        noise_conf=s.noise_conf,
        noise_alpha_ratio=s.noise_alpha_ratio,
        debug_dir=s.debug_dir or None,
        save_raw=s.save_raw,
        confidence_comments=s.confidence_comments,
        tesseract_path=s.tesseract_path or None,
    )


# ---------------------------------------------------------------------------
# Output format converters
# ---------------------------------------------------------------------------
def markdown_to_plain_text(md: str) -> str:
    out_lines: List[str] = []
    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            out_lines.append("")
            continue
        # strip leading heading markers
        s = re.sub(r"^\s*#{1,6}\s*", "", s)
        # bold/italic markers
        s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
        s = re.sub(r"\*(.+?)\*", r"\1", s)
        # markdown comment blocks
        if s.strip().startswith("<!--") and s.strip().endswith("-->"):
            continue
        # table separator rows like | --- | --- |
        if re.match(r"^\s*\|?\s*[:\- |]+\s*\|?\s*$", s) and "-" in s:
            continue
        # table data rows -> tab separated
        if s.lstrip().startswith("|") and s.rstrip().endswith("|"):
            cells = [c.strip() for c in s.strip().strip("|").split("|")]
            out_lines.append("\t".join(cells))
            continue
        out_lines.append(s)
    # collapse multiple blanks
    collapsed: List[str] = []
    blank = False
    for ln in out_lines:
        if not ln.strip():
            if blank:
                continue
            blank = True
        else:
            blank = False
        collapsed.append(ln)
    return "\n".join(collapsed).strip() + "\n"


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


def markdown_to_html(md: str, title: str = "OCR Output") -> str:
    """Convert the subset of Markdown the OCR engine emits to standalone HTML."""
    lines = md.splitlines()
    out: List[str] = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>{_html_escape(title)}</title>",
        "<style>",
        "body{font-family:system-ui,Segoe UI,Arial,sans-serif;max-width:1000px;"
        "margin:2em auto;padding:0 1em;color:#222;}",
        "h1,h2,h3{border-bottom:1px solid #ddd;padding-bottom:.2em;}",
        "table{border-collapse:collapse;margin:.5em 0;}",
        "th,td{border:1px solid #bbb;padding:.25em .6em;text-align:left;"
        "font-family:Consolas,Menlo,monospace;font-size:.9em;}",
        "th{background:#f3f3f3;}",
        "hr{border:none;border-top:2px solid #aaa;margin:2em 0;}",
        "</style></head><body>",
    ]

    i = 0
    while i < len(lines):
        s = lines[i].rstrip()
        if not s.strip():
            out.append("")
            i += 1
            continue
        if s.strip().startswith("<!--") and s.strip().endswith("-->"):
            i += 1
            continue
        if s.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.+)$", s)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{_html_escape(m.group(2))}</h{level}>")
            i += 1
            continue
        if s.lstrip().startswith("|") and s.rstrip().endswith("|"):
            header_cells = [c.strip() for c in s.strip().strip("|").split("|")]
            # next line should be the separator
            j = i + 1
            if j < len(lines) and re.match(r"^\s*\|?\s*[:\- |]+\s*\|?\s*$", lines[j]):
                j += 1
                rows: List[List[str]] = []
                while j < len(lines):
                    ln = lines[j].rstrip()
                    if not (ln.lstrip().startswith("|") and ln.rstrip().endswith("|")):
                        break
                    cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                    rows.append(cells)
                    j += 1
                out.append("<table>")
                out.append(
                    "<thead><tr>"
                    + "".join(f"<th>{_html_escape(c)}</th>" for c in header_cells)
                    + "</tr></thead>"
                )
                out.append("<tbody>")
                for r in rows:
                    out.append(
                        "<tr>"
                        + "".join(f"<td>{_html_escape(c)}</td>" for c in r)
                        + "</tr>"
                    )
                out.append("</tbody></table>")
                i = j
                continue
            # standalone single-line "table"
            out.append(
                "<p><strong>" + " | ".join(_html_escape(c) for c in header_cells) + "</strong></p>"
            )
            i += 1
            continue
        # plain paragraph (collect consecutive non-special lines)
        para = [s]
        i += 1
        while i < len(lines):
            ln = lines[i].rstrip()
            if not ln.strip():
                break
            if (
                re.match(r"^(#{1,6})\s+", ln) or
                (ln.lstrip().startswith("|") and ln.rstrip().endswith("|")) or
                ln.strip() == "---"
            ):
                break
            para.append(ln)
            i += 1
        text = " ".join(_html_escape(p) for p in para)
        # inline bold/italic
        text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"<em>\1</em>", text)
        out.append(f"<p>{text}</p>")

    out.append("</body></html>")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Core multi-format processor
#
# Mirrors ``OCR_PDF_to_Markdown.process_pdf`` but emits every selected output
# format from a single render+OCR pass, and reports progress / responds to a
# cancellation event.
# ---------------------------------------------------------------------------
class CancelledError(Exception):
    pass


# ---------------------------------------------------------------------------
# Per-page worker — picklable, called either inline or via ProcessPoolExecutor
# ---------------------------------------------------------------------------
@dataclass
class PageWorkItem:
    pdf_path: str
    pdf_password: str
    page_index: int
    total_pages: int
    settings: OcrSettings
    formats: List[str]


@dataclass
class PageWordRecord:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class PageResult:
    pdf_path: str
    page_index: int
    used_text_layer: bool
    drawing_page: bool
    is_error: bool
    error: str
    elapsed_sec: float

    md_chunk: str
    raw_text: str
    pdf_page_bytes: bytes
    words: List[PageWordRecord]
    avg_conf: float
    word_count: int
    rendered_size: Tuple[int, int]
    deskew_applied: bool


def _ocr_page(item: PageWorkItem) -> PageResult:
    """Run the full render → preprocess → OCR pipeline for one page.

    Designed to be pickled and called inside a ProcessPoolExecutor worker.
    All inputs/outputs are picklable; rendered/processed images stay inside
    this worker (only the OCR text + word records flow back to the parent).
    """
    import time
    t0 = time.perf_counter()
    s = item.settings

    if s.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = s.tesseract_path

    try:
        doc = fitz.open(item.pdf_path)
        if doc.needs_pass:
            ok = doc.authenticate(item.pdf_password) if item.pdf_password else 0
            if not ok:
                return PageResult(
                    pdf_path=item.pdf_path, page_index=item.page_index,
                    used_text_layer=False, drawing_page=False,
                    is_error=True, error="Password required or incorrect.",
                    elapsed_sec=0.0,
                    md_chunk="", raw_text="", pdf_page_bytes=b"",
                    words=[], avg_conf=0.0, word_count=0,
                    rendered_size=(0, 0), deskew_applied=False,
                )

        try:
            page = doc[item.page_index]

            # --- 1. Text-layer fast path -----------------------------------
            #
            # When the PDF already has a real text layer (Word/Excel exports
            # etc.) we'd rather use those exact characters than OCR a render
            # of them. We pull *word-level* boxes via get_text("dict"), turn
            # them into the same OCRWord shape Tesseract produces, then feed
            # them through the normal row/column reconstruction pipeline.
            # That gives perfect text AND aligned Markdown tables, instead of
            # the unstructured stream-order dump the older code emitted.
            crop = ocr_engine.parse_crop(s.crop or None)
            used_text_layer = False
            embedded_words: List = []
            if s.use_text_layer:
                embedded_words = ocr_engine.words_from_embedded_text(
                    page, s.render_scale, crop,
                )
                if embedded_words and len(embedded_words) >= 8:
                    used_text_layer = True

            # --- 2. Decide whether to OCR ----------------------------------
            if used_text_layer:
                rendered = None
                pdf_bytes = b""
                if "searchable_pdf" in item.formats:
                    # Render once just for the searchable-PDF output.
                    rendered = ocr_engine.render_page(page, s.render_scale)
                    try:
                        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                            rendered, lang=s.lang,
                            config=f"--oem {s.oem} --psm {s.psm}",
                            extension="pdf",
                        )
                    except Exception:
                        pdf_bytes = b""

                words = embedded_words
                deskew_applied = False
                # Reuse the table reconstruction we use for OCR'd words.
                rows = ocr_engine.group_words_into_rows(
                    words, y_tolerance=s.row_tol,
                )
                page_md = ocr_engine.rows_to_markdown(
                    rows,
                    gap=s.col_gap,
                    noise_conf=s.noise_conf,
                    noise_alpha_ratio=s.noise_alpha_ratio,
                    include_confidence_comments=s.confidence_comments,
                )
                md_chunk = (
                    f"\n## Page {item.page_index + 1}\n"
                    "<!-- source: embedded PDF text layer -->\n"
                    + page_md
                )

                word_records = [
                    PageWordRecord(
                        text=w.text, conf=round(w.conf, 2),
                        x=w.x, y=w.y, w=w.w, h=w.h,
                    )
                    for w in words
                ]
                return PageResult(
                    pdf_path=item.pdf_path,
                    page_index=item.page_index,
                    used_text_layer=True,
                    drawing_page=False,
                    is_error=False, error="",
                    elapsed_sec=time.perf_counter() - t0,
                    md_chunk=md_chunk,
                    raw_text=page.get_text("text") or "",
                    pdf_page_bytes=pdf_bytes,
                    words=word_records,
                    avg_conf=100.0,
                    word_count=len(word_records),
                    rendered_size=(rendered.size if rendered is not None else (0, 0)),
                    deskew_applied=False,
                )

            # --- 3. OCR path -----------------------------------------------
            rendered = ocr_engine.render_page(page, s.render_scale)
            rendered = ocr_engine.crop_image(rendered, crop)

            deskew_applied = False
            if s.deskew:
                _, angle = ocr_engine.deskew_image(rendered)
                deskew_applied = abs(angle) >= 0.25

            processed = ocr_engine.preprocess_image(
                rendered,
                grayscale=not s.no_grayscale,
                contrast=s.contrast,
                sharpen=s.sharpen,
                threshold=s.threshold,
                threshold_value=s.threshold_value,
                invert=s.invert,
                dilate=s.dilate,
                erode=s.erode,
                deskew=s.deskew,
                denoise=s.denoise,
            )

            if s.best_accuracy:
                words = ocr_engine.tesseract_words_voting(
                    processed, s.lang, s.oem, s.min_conf,
                    user_psm=s.psm,
                )
            else:
                words = ocr_engine.tesseract_words(
                    processed, s.lang, s.psm, s.oem, s.min_conf,
                )

            # Searchable PDF page
            pdf_bytes = b""
            if "searchable_pdf" in item.formats:
                try:
                    pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                        rendered, lang=s.lang,
                        config=f"--oem {s.oem} --psm {s.psm}",
                        extension="pdf",
                    )
                except Exception:
                    pdf_bytes = b""

            # Raw text (always cheap; consumers may ignore it)
            raw_text = ""
            if s.save_raw or s.use_text_layer is False:
                try:
                    raw_text = pytesseract.image_to_string(
                        processed, lang=s.lang,
                        config=f"--oem {s.oem} --psm {s.psm} "
                               "-c preserve_interword_spaces=1",
                    )
                except Exception:
                    raw_text = ""

            md_chunk = f"\n## Page {item.page_index + 1}\n"

            drawing = ocr_engine.detect_drawing_page(words)
            if drawing:
                md_chunk += (
                    "*This page appears to be a drawing/graphic with no "
                    "extractable table data.*\n"
                )
            else:
                rows = ocr_engine.group_words_into_rows(
                    words, y_tolerance=s.row_tol,
                )
                page_md = ocr_engine.rows_to_markdown(
                    rows,
                    gap=s.col_gap,
                    noise_conf=s.noise_conf,
                    noise_alpha_ratio=s.noise_alpha_ratio,
                    include_confidence_comments=s.confidence_comments,
                )
                md_chunk += page_md

            avg_conf = (
                sum(w.conf for w in words) / len(words) if words else 0.0
            )
            word_records = [
                PageWordRecord(
                    text=w.text, conf=round(w.conf, 2),
                    x=w.x, y=w.y, w=w.w, h=w.h,
                )
                for w in words
            ]

            return PageResult(
                pdf_path=item.pdf_path,
                page_index=item.page_index,
                used_text_layer=False,
                drawing_page=drawing,
                is_error=False, error="",
                elapsed_sec=time.perf_counter() - t0,
                md_chunk=md_chunk,
                raw_text=raw_text,
                pdf_page_bytes=pdf_bytes,
                words=word_records,
                avg_conf=avg_conf,
                word_count=len(word_records),
                rendered_size=rendered.size,
                deskew_applied=deskew_applied,
            )
        finally:
            doc.close()
    except Exception as exc:
        return PageResult(
            pdf_path=item.pdf_path, page_index=item.page_index,
            used_text_layer=False, drawing_page=False,
            is_error=True, error=f"{type(exc).__name__}: {exc}",
            elapsed_sec=time.perf_counter() - t0,
            md_chunk="", raw_text="", pdf_page_bytes=b"",
            words=[], avg_conf=0.0, word_count=0,
            rendered_size=(0, 0), deskew_applied=False,
        )


def _merge_pdf_pages(pdf_page_bytes: List[bytes]) -> bytes:
    if not pdf_page_bytes:
        return b""
    out = fitz.open()
    try:
        for blob in pdf_page_bytes:
            if not blob:
                continue
            sub = fitz.open(stream=blob, filetype="pdf")
            try:
                out.insert_pdf(sub)
            finally:
                sub.close()
        return out.tobytes()
    finally:
        out.close()


# ---------------------------------------------------------------------------
# Per-PDF orchestrator
#
# Builds work items for every page of one PDF, runs them serially or via the
# shared ProcessPoolExecutor, then assembles the per-format outputs.
# ---------------------------------------------------------------------------
def process_pdf_multi(
    pdf_path: Path,
    pdf_password: str,
    settings: OcrSettings,
    log: Callable[[str], None],
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
    executor: "Optional[Any]" = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run OCR on one PDF.

    Returns ``(outputs, metrics)`` where outputs maps format key to file
    contents and metrics describes the run for the audit log / validation
    dashboard.
    """
    from concurrent.futures import as_completed
    import time

    s = settings
    if s.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = s.tesseract_path

    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        ok = doc.authenticate(pdf_password) if pdf_password else 0
        if not ok:
            doc.close()
            raise RuntimeError(
                f"{pdf_path.name} is password-protected and the supplied "
                "password did not work."
            )
    try:
        total_pages_doc = len(doc)
        page_indices = ocr_engine.parse_pages(s.pages or None, total_pages_doc)
        if not page_indices:
            page_indices = list(range(total_pages_doc))
    finally:
        doc.close()

    items = [
        PageWorkItem(
            pdf_path=str(pdf_path),
            pdf_password=pdf_password,
            page_index=i,
            total_pages=total_pages_doc,
            settings=s,
            formats=list(s.output_formats),
        )
        for i in page_indices
    ]

    total = len(items)
    results_by_index: Dict[int, PageResult] = {}

    t0 = time.perf_counter()
    if executor is None or total <= 1:
        for idx, item in enumerate(items):
            if cancel_event.is_set():
                raise CancelledError()
            progress(idx, total)
            log(f"  page {item.page_index + 1}/{total_pages_doc}")
            res = _ocr_page(item)
            results_by_index[item.page_index] = res
            _log_page_result(res, log)
    else:
        futures = {executor.submit(_ocr_page, item): item for item in items}
        completed = 0
        try:
            for fut in as_completed(futures):
                if cancel_event.is_set():
                    for f in futures:
                        f.cancel()
                    raise CancelledError()
                res = fut.result()
                results_by_index[res.page_index] = res
                completed += 1
                progress(completed, total)
                log(f"  page {res.page_index + 1}/{total_pages_doc}")
                _log_page_result(res, log)
        finally:
            pass

    progress(total, total)
    elapsed = time.perf_counter() - t0

    # ---- assemble outputs in page order ------------------------------------
    ordered = [results_by_index[i] for i in page_indices if i in results_by_index]
    formats = set(s.output_formats)

    md_chunks: List[str] = [f"# OCR Output: {pdf_path.name}\n"]
    md_chunks.append(
        "<!-- Settings: "
        f"render_scale={s.render_scale}, psm={s.psm}, oem={s.oem}, "
        f"threshold={s.threshold}, contrast={s.contrast}, "
        f"deskew={s.deskew}, denoise={s.denoise}, "
        f"best_accuracy={s.best_accuracy}, use_text_layer={s.use_text_layer}"
        " -->\n"
    )
    csv_records: List[List[Any]] = [["page", "x", "y", "w", "h", "conf", "text"]]
    pdf_page_bytes: List[bytes] = []
    json_doc: Dict[str, Any] = {
        "file": pdf_path.name,
        "total_pages": total_pages_doc,
        "settings": asdict(s),
        "pages": [],
    }

    debug_dir = Path(s.debug_dir) if s.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for res in ordered:
        md_chunks.append(res.md_chunk)
        if "csv" in formats:
            for w in res.words:
                csv_records.append([
                    res.page_index + 1, w.x, w.y, w.w, w.h, w.conf, w.text,
                ])
        if "json" in formats:
            json_doc["pages"].append({
                "page": res.page_index + 1,
                "used_text_layer": res.used_text_layer,
                "drawing_page": res.drawing_page,
                "avg_conf": round(res.avg_conf, 2),
                "deskew_applied": res.deskew_applied,
                "words": [
                    {"text": w.text, "conf": w.conf,
                     "x": w.x, "y": w.y, "w": w.w, "h": w.h}
                    for w in res.words
                ],
            })
        if res.pdf_page_bytes:
            pdf_page_bytes.append(res.pdf_page_bytes)
        if debug_dir and res.raw_text:
            stem = f"{pdf_path.stem}_p{res.page_index + 1:02d}"
            (debug_dir / f"{stem}_raw.txt").write_text(res.raw_text, encoding="utf-8")

    md_text = "\n".join(md_chunks).strip() + "\n"

    outputs: Dict[str, Any] = {}
    if "markdown" in formats:
        outputs["markdown"] = md_text
    if "text" in formats:
        outputs["text"] = markdown_to_plain_text(md_text)
    if "json" in formats:
        outputs["json"] = json.dumps(json_doc, indent=2, ensure_ascii=False)
    if "csv" in formats:
        buf = io.StringIO()
        import csv as _csv
        writer = _csv.writer(buf)
        for row in csv_records:
            writer.writerow(row)
        outputs["csv"] = buf.getvalue()
    if "html" in formats:
        outputs["html"] = markdown_to_html(md_text, title=pdf_path.name)
    if "searchable_pdf" in formats:
        outputs["searchable_pdf"] = _merge_pdf_pages(pdf_page_bytes)

    # ---- metrics for validation / audit ------------------------------------
    ocr_pages = [r for r in ordered if not r.used_text_layer and not r.is_error]
    text_layer_pages = [r for r in ordered if r.used_text_layer]
    error_pages = [r for r in ordered if r.is_error]
    avg_conf_overall = (
        sum(r.avg_conf for r in ocr_pages) / len(ocr_pages)
        if ocr_pages else 0.0
    )
    low_conf_pages = sorted(
        [(r.page_index + 1, round(r.avg_conf, 1))
         for r in ocr_pages if r.avg_conf < 70.0]
    )
    metrics: Dict[str, Any] = {
        "pdf": str(pdf_path),
        "elapsed_sec": round(elapsed, 2),
        "total_pages_doc": total_pages_doc,
        "pages_processed": len(ordered),
        "pages_via_ocr": len(ocr_pages),
        "pages_via_text_layer": len(text_layer_pages),
        "pages_errored": len(error_pages),
        "errors": [{"page": r.page_index + 1, "error": r.error} for r in error_pages],
        "avg_confidence": round(avg_conf_overall, 2),
        "low_confidence_pages": low_conf_pages,
        "total_words": sum(r.word_count for r in ordered),
        "page_metrics": [
            {
                "page": r.page_index + 1,
                "source": "text-layer" if r.used_text_layer else "ocr",
                "drawing": r.drawing_page,
                "avg_conf": round(r.avg_conf, 2),
                "words": r.word_count,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "deskew_applied": r.deskew_applied,
                "rendered_size": list(r.rendered_size),
            }
            for r in ordered
        ],
    }
    # Keep ordered results around for the validation tab (page picker, preview)
    metrics["_page_results"] = ordered
    return outputs, metrics


def _log_page_result(res: PageResult, log: Callable[[str], None]) -> None:
    if res.is_error:
        log(f"    ERROR: {res.error}")
        return
    if res.used_text_layer:
        log(f"    used embedded text layer ({res.word_count} words)")
        return
    flags = []
    if res.deskew_applied:
        flags.append("deskewed")
    if res.drawing_page:
        flags.append("drawing")
    extra = f"  [{', '.join(flags)}]" if flags else ""
    log(
        f"    OCR: {res.word_count} words, "
        f"avg conf {res.avg_conf:.1f}, {res.elapsed_sec:.1f}s{extra}"
    )


# ---------------------------------------------------------------------------
# Tooltips
# ---------------------------------------------------------------------------
class Tooltip:
    """Hover-to-show tooltip for any Tkinter widget."""

    def __init__(self, widget, text: str,
                 delay_ms: int = 450, wraplength: int = 380) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._tip: Optional[tk.Toplevel] = None
        self._after_id: Optional[str] = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        try:
            tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        tip.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tip,
            text=self.text,
            justify="left",
            background="#FFFFE0",
            foreground="#222",
            relief="solid",
            borderwidth=1,
            wraplength=self.wraplength,
            padx=8, pady=5,
            font=("Segoe UI", 9),
        ).pack()
        self._tip = tip

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


TOOLTIPS: Dict[str, str] = {
    # ---- Files & Output ----
    "files_list":
        "Queue of PDFs to process. Use Add files... to pick individual PDFs, "
        "Add folder... to pull every PDF below a folder (recursive), and "
        "Move up/down to control the ordering used in combined-output files.",
    "add_files":         "Pick one or more PDF files to add to the queue.",
    "add_folder":        "Add every .pdf below a chosen folder (recursive).",
    "remove":            "Remove the selected items from the queue.",
    "clear":             "Remove all items from the queue.",
    "move_up":           "Move the selected items up the list.",
    "move_down":         "Move the selected items down the list.",
    "output_dir":
        "Folder where output files will be written. If left empty when you "
        "add files, it defaults to the folder of the first input PDF.",
    "output_mode":
        "Per-file: one output (or set of outputs) per input PDF, named after "
        "the source PDF.\nCombined: every input PDF concatenated into one "
        "output per format.",
    "combined_name":
        "Base filename used in Combined mode. The format extension "
        "(.md, .pdf, .json, ...) is appended automatically per selected format.",
    "format_markdown":
        "Markdown (.md). Preserves OCR-detected headings and reconstructs "
        "tables as Markdown pipe-tables. Best for human reading or for "
        "feeding to an LLM.",
    "format_text":
        "Plain text (.txt). Markdown markers removed; table rows are joined "
        "with tabs. Best for piping into other tools.",
    "format_json":
        "JSON (.json). Per-page list of recognised words with text, "
        "confidence, and bounding box. Best for programmatic processing.",
    "format_csv":
        "CSV (.csv). One row per recognised word with page, position, size, "
        "confidence, and text. Open in Excel/Sheets for analysis.",
    "format_html":
        "Self-contained HTML (.html). Styled tables, opens in any browser, "
        "easy to share by email.",
    "format_searchable_pdf":
        "Searchable PDF (.pdf). Keeps the original page image and overlays "
        "an invisible OCR text layer, so the file looks identical to the "
        "source but text is copy-pasteable and Ctrl+F searchable.",
    "open_when_done":    "When processing finishes, open the output folder in Explorer.",
    "show_notify":       "Pop up a Done/Failed dialog at the end of the run.",

    # ---- OCR tab ----
    "tesseract_path":
        "Path to tesseract.exe. The portable .exe ships with a sibling "
        "'tesseract' folder that is auto-detected. Otherwise, point this at "
        r"your install, e.g. C:\Program Files\Tesseract-OCR\tesseract.exe.",
    "lang":
        "Tesseract language code(s). 'eng' = English. Combine with '+' for "
        "multiple, e.g. 'eng+fra'. Each language requires the matching "
        ".traineddata file in tessdata/.",
    "preset":
        "Apply a starting-point set of values across the OCR, Preprocessing, "
        "and Layout tabs. Pick the one closest to your document, then tweak.",
    "psm":
        "Page segmentation mode — how Tesseract should look at the page:\n"
        "  3  fully automatic (general documents)\n"
        "  4  single column of variable-sized text\n"
        "  6  single uniform block of text (good for tables)\n"
        " 11  sparse text, find as much as possible\n"
        " 12  sparse text with orientation/script detection",
    "oem":
        "OCR engine mode:\n"
        "  0  legacy engine only\n"
        "  1  LSTM neural net only\n"
        "  2  legacy + LSTM combined\n"
        "  3  default (whatever is best in your Tesseract build, usually LSTM)",
    "render_scale":
        "How much to upscale each PDF page before OCR. 1.0 = original 72 DPI; "
        "5.0 ≈ 360 DPI. Higher catches smaller text but uses more memory and "
        "is slower. Try 3-6 for typical CMM/report PDFs.",
    "min_conf":
        "Drop individual words whose Tesseract confidence is below this "
        "value (0-100). Higher = stricter, fewer noisy words but you may "
        "miss faint text. 20 is a sensible default.",
    "pages":
        "Which pages to OCR. Examples: '1' · '1,3,5' · '1-5' · '1,3,5-7'. "
        "Leave blank to process every page.",

    # ---- Preprocessing tab ----
    "no_grayscale":
        "Skip the grayscale conversion and feed Tesseract the original RGB "
        "image. Usually leave UNCHECKED — grayscale OCRs better in nearly "
        "every case.",
    "contrast":
        "Multiply image contrast. 1.0 = unchanged. Values above 1 make "
        "blacks darker and whites whiter, helping faded text. Try 1.4-1.8 "
        "for low-contrast scans. Very high values (>2.5) can crush "
        "mid-tones and merge characters.",
    "sharpen":
        "Number of sharpen passes applied to the rendered page. 0 = none; "
        "1-2 helps small text. More than 3 introduces ringing artefacts "
        "that hurt OCR.",
    "threshold":
        "How to binarise the image (force every pixel to black or white) "
        "before OCR:\n"
        "  none      — keep grayscale, let Tesseract decide\n"
        "  global    — single cutoff value; fast but bad with uneven lighting\n"
        "  adaptive  — local cutoff per region; best for real scans (recommended)",
    "threshold_value":
        "Cutoff used when threshold = global. Pixels darker than this value "
        "(0-255) become black, lighter ones become white. 180 is a good "
        "starting point for typical scans.",
    "invert":
        "Swap black and white after thresholding. Use only for white-on-black "
        "source pages (rare).",
    "dilate":
        "Pixel dilation kernel size. >0 thickens text strokes — helps when "
        "characters break apart from very thin printing. Start at 1; kernels "
        "above 3 cause neighbouring characters to merge.",
    "erode":
        "Pixel erosion kernel size. >0 thins text strokes — helps when "
        "characters blob together. In practice you use either dilate OR "
        "erode, not both. Usually leave at 0.",
    "crop":
        "Crop pixels off each rendered page before OCR. Format "
        "'left,top,right,bottom', e.g. '50,100,50,80' trims 50 px from each "
        "side, 100 from the top, 80 from the bottom. Useful to drop headers, "
        "footers, or borders. Leave blank for no crop.",
    "deskew":
        "Detects pages that are slightly tilted (common in scans where the "
        "page wasn't perfectly aligned on the scanner) and rotates them to "
        "be straight before OCR. Tilts under about a quarter of a degree "
        "are left alone.",
    "denoise":
        "Removes speckle and graininess from low-quality scans before OCR. "
        "Slower but improves accuracy on noisy pages. Has no effect on "
        "clean digital PDFs.",

    # ---- Layout & Noise tab ----
    "row_tol":
        "Vertical pixel tolerance for grouping OCR words into the same text "
        "row. Lower = stricter (words must be vertically very close). Raise "
        "if one line gets split in two; lower if two close lines merge.",
    "col_gap":
        "Minimum horizontal pixel gap that triggers a new table cell. "
        "Smaller = more columns detected. Set very high (e.g. 999) for "
        "plain prose where you don't want any column splitting.",
    "noise_conf":
        "Discard whole rows whose average word confidence is below this "
        "value (0-100). 55 is a sensible default. Raise to drop more "
        "borderline-junk rows; lower to keep more.",
    "alpha_ratio":
        "Drop rows whose alphanumeric ratio (letters+digits ÷ all chars) "
        "is below this value. 0.3 means at least 30% of the line must be "
        "letters or digits, otherwise it's treated as drawing/symbol noise. "
        "Lower to keep more symbol-heavy lines.",

    # ---- Diagnostics tab ----
    "debug_dir":
        "Folder where debug artefacts are written: rendered/processed PNGs, "
        "per-word confidence CSV, raw Tesseract text. Leave blank to disable. "
        "Very useful when tuning preprocessing or noise settings.",
    "save_raw":
        "Also save Tesseract's raw, un-postprocessed text into the debug "
        "folder. Lets you see what OCR really saw before noise filtering "
        "and row reconstruction.",
    "conf_comments":
        "Embed an HTML comment '<!-- OCR avg confidence: 87.3 -->' after "
        "each row in Markdown output. Helps identify rows OCR was unsure about.",
    "audit_log":
        "Write a small .json sidecar next to each output describing the run: "
        "settings used, source SHA-256, per-page confidence, words found, "
        "elapsed time, Tesseract version. Recommended for compliance and "
        "for figuring out why an old output looked the way it did.",

    # ---- Performance & accuracy (OCR tab) ----
    "workers":
        "Number of CPU cores used in parallel. 0 means 'auto' (uses about "
        "half your cores so the rest of the machine stays responsive). Set "
        "higher for batch runs on dedicated machines. Each worker uses memory "
        "for one rendered page at a time.",
    "use_text_layer":
        "Many PDFs (Word/Excel exports etc.) already contain real, perfectly "
        "accurate text with per-word positions. With this on, the tool pulls "
        "those word boxes directly and runs them through the same row/column "
        "reconstruction as OCR — so you get perfect text AND properly aligned "
        "tables, much faster than re-OCR'ing. Scanned/image-only PDFs fall "
        "back to OCR automatically.",
    "best_accuracy":
        "Runs Tesseract a second time with a complementary page-segmentation "
        "mode (sparse text) alongside your selected PSM, then keeps whichever "
        "found more correct text. Selection uses word count × confidence so a "
        "run that finds more words wins over one that finds fewer with "
        "slightly higher confidence. ~2x slower; helps on pages where the "
        "default segmentation drops content.",

    # ---- Watch folder ----
    "watch_folder":
        "Pick a folder. The tool polls it for new PDFs and processes them "
        "automatically using the current settings. Use this for drop-folder "
        "workflows where files appear over time.",
    "watch_interval":
        "How often (in seconds) the tool checks the watch folder for new "
        "PDFs. 5 seconds is a good default.",
}


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class OcrApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("980x780")
        self.minsize(820, 640)

        self.settings = OcrSettings()
        self._cancel_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._log_queue: "queue.Queue[tuple]" = queue.Queue()

        # Validation / report state populated after each run.
        self._last_metrics: List[Dict[str, Any]] = []
        self._last_run_settings: Optional[OcrSettings] = None
        self._last_run_elapsed: float = 0.0

        # Watch-folder state.
        self._watch_thread: Optional[threading.Thread] = None
        self._watch_stop = threading.Event()
        self._watch_seen: set = set()

        self._build_menu()
        self._build_widgets()
        self._load_config()
        self._auto_detect_tesseract()
        self.after(100, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(200, self._maybe_show_welcome)

    # ---- top menu ----
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add files...", command=self._add_files)
        file_menu.add_command(label="Add folder...", command=self._add_folder)
        self._recent_files_menu = tk.Menu(file_menu, tearoff=False)
        self._recent_folders_menu = tk.Menu(file_menu, tearoff=False)
        file_menu.add_cascade(label="Recent files", menu=self._recent_files_menu)
        file_menu.add_cascade(label="Recent folders", menu=self._recent_folders_menu)
        file_menu.add_separator()
        file_menu.add_command(label="Save preset...", command=self._save_preset)
        file_menu.add_command(label="Load preset...", command=self._load_preset)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Quick start guide...", command=self._show_welcome)
        help_menu.add_command(label="Settings reference...", command=self._show_settings_reference)
        help_menu.add_separator()
        help_menu.add_command(label="Run self-test...", command=self._run_self_test)
        help_menu.add_command(label="Last validation report...", command=self._show_last_report)
        help_menu.add_separator()
        help_menu.add_command(label="Tesseract status...", command=self._show_tesseract_status)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ---- tooltip helper ----
    def _tip(self, widget, key: str) -> None:
        text = TOOLTIPS.get(key)
        if text:
            Tooltip(widget, text)

    # ---- main layout ----
    def _build_widgets(self) -> None:
        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        self._tab_files = ttk.Frame(notebook, padding=10)
        self._tab_ocr = ttk.Frame(notebook, padding=10)
        self._tab_prep = ttk.Frame(notebook, padding=10)
        self._tab_layout = ttk.Frame(notebook, padding=10)
        self._tab_diag = ttk.Frame(notebook, padding=10)
        self._tab_validation = ttk.Frame(notebook, padding=10)
        self._tab_watch = ttk.Frame(notebook, padding=10)
        notebook.add(self._tab_files, text="Files & Output")
        notebook.add(self._tab_ocr, text="OCR")
        notebook.add(self._tab_prep, text="Preprocessing")
        notebook.add(self._tab_layout, text="Layout & Noise")
        notebook.add(self._tab_diag, text="Diagnostics")
        notebook.add(self._tab_validation, text="Validation")
        notebook.add(self._tab_watch, text="Watch folder")

        self._build_files_tab(self._tab_files)
        self._build_ocr_tab(self._tab_ocr)
        self._build_prep_tab(self._tab_prep)
        self._build_layout_tab(self._tab_layout)
        self._build_diag_tab(self._tab_diag)
        self._build_validation_tab(self._tab_validation)
        self._build_watch_tab(self._tab_watch)

        # bottom: progress + log + buttons
        bottom = ttk.Frame(outer)
        bottom.pack(fill="both", expand=False, pady=(8, 0))

        status_row = ttk.Frame(bottom)
        status_row.pack(fill="x")
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_row, textvariable=self._status_var).pack(side="left")
        self._progress = ttk.Progressbar(status_row, mode="determinate", length=260)
        self._progress.pack(side="right")

        log_frame = ttk.LabelFrame(bottom, text="Log", padding=4)
        log_frame.pack(fill="both", expand=True, pady=(4, 4))
        self._log_text = tk.Text(log_frame, height=10, wrap="word", state="disabled")
        self._log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_frame, command=self._log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self._log_text.configure(yscrollcommand=log_scroll.set)

        btn_row = ttk.Frame(bottom)
        btn_row.pack(fill="x")
        self._start_btn = ttk.Button(btn_row, text="Start OCR", command=self._on_start)
        self._start_btn.pack(side="left")
        self._cancel_btn = ttk.Button(
            btn_row, text="Cancel", command=self._on_cancel, state="disabled"
        )
        self._cancel_btn.pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Clear log", command=self._clear_log).pack(side="left", padx=(6, 0))
        ttk.Button(btn_row, text="Save preset", command=self._save_preset).pack(side="right")
        ttk.Button(btn_row, text="Load preset", command=self._load_preset).pack(side="right", padx=(0, 6))

    # ---- "Files & Output" tab ----
    def _build_files_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        files_label = ttk.Label(parent, text="Input PDFs:  (hover any setting for help)")
        files_label.grid(row=0, column=0, sticky="w")

        list_frame = ttk.Frame(parent)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._files_list = tk.Listbox(list_frame, selectmode="extended", activestyle="dotbox")
        self._files_list.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(list_frame, command=self._files_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._files_list.configure(yscrollcommand=sb.set)
        self._tip(self._files_list, "files_list")
        self._tip(files_label, "files_list")

        side = ttk.Frame(list_frame)
        side.grid(row=0, column=2, sticky="ns", padx=(6, 0))
        b_add = ttk.Button(side, text="Add files...", command=self._add_files); b_add.pack(fill="x")
        b_addf = ttk.Button(side, text="Add folder...", command=self._add_folder); b_addf.pack(fill="x", pady=(4, 0))
        b_rm = ttk.Button(side, text="Remove", command=self._remove_selected); b_rm.pack(fill="x", pady=(4, 0))
        b_clear = ttk.Button(side, text="Clear", command=self._clear_files); b_clear.pack(fill="x", pady=(4, 0))
        b_up = ttk.Button(side, text="Move up", command=lambda: self._move_selected(-1)); b_up.pack(fill="x", pady=(12, 0))
        b_dn = ttk.Button(side, text="Move down", command=lambda: self._move_selected(1)); b_dn.pack(fill="x", pady=(4, 0))
        self._tip(b_add, "add_files")
        self._tip(b_addf, "add_folder")
        self._tip(b_rm, "remove")
        self._tip(b_clear, "clear")
        self._tip(b_up, "move_up")
        self._tip(b_dn, "move_down")

        # output
        out_box = ttk.LabelFrame(parent, text="Output", padding=6)
        out_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        out_box.columnconfigure(1, weight=1)

        lbl_folder = ttk.Label(out_box, text="Folder:"); lbl_folder.grid(row=0, column=0, sticky="w")
        self._out_dir_var = tk.StringVar()
        ent_dir = ttk.Entry(out_box, textvariable=self._out_dir_var)
        ent_dir.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(out_box, text="Browse...", command=self._pick_output_dir).grid(row=0, column=2)
        self._tip(lbl_folder, "output_dir"); self._tip(ent_dir, "output_dir")

        lbl_mode = ttk.Label(out_box, text="Mode:"); lbl_mode.grid(row=1, column=0, sticky="w", pady=(4, 0))
        mode_row = ttk.Frame(out_box)
        mode_row.grid(row=1, column=1, columnspan=2, sticky="w", pady=(4, 0))
        self._mode_var = tk.StringVar(value="per_file")
        rb_per = ttk.Radiobutton(mode_row, text="One output per PDF", value="per_file", variable=self._mode_var)
        rb_per.pack(side="left")
        rb_comb = ttk.Radiobutton(mode_row, text="Combined into one file", value="combined", variable=self._mode_var)
        rb_comb.pack(side="left", padx=(12, 0))
        self._tip(lbl_mode, "output_mode"); self._tip(rb_per, "output_mode"); self._tip(rb_comb, "output_mode")

        lbl_cname = ttk.Label(out_box, text="Combined name:"); lbl_cname.grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._combined_name_var = tk.StringVar(value="ocr_output")
        ent_cname = ttk.Entry(out_box, textvariable=self._combined_name_var)
        ent_cname.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(4, 0), padx=(4, 0))
        self._tip(lbl_cname, "combined_name"); self._tip(ent_cname, "combined_name")

        lbl_fmt = ttk.Label(out_box, text="Formats:"); lbl_fmt.grid(row=3, column=0, sticky="nw", pady=(8, 0))
        fmt_frame = ttk.Frame(out_box)
        fmt_frame.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self._fmt_vars: Dict[str, tk.BooleanVar] = {}
        for i, (key, label, _ext) in enumerate(OUTPUT_FORMATS):
            var = tk.BooleanVar(value=(key == "markdown"))
            self._fmt_vars[key] = var
            cb = ttk.Checkbutton(fmt_frame, text=label, variable=var)
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 12))
            self._tip(cb, f"format_{key}")

        # finished options
        misc = ttk.Frame(out_box)
        misc.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._open_when_done_var = tk.BooleanVar(value=True)
        cb_open = ttk.Checkbutton(misc, text="Open output folder when done",
                                  variable=self._open_when_done_var)
        cb_open.pack(side="left")
        self._show_notify_var = tk.BooleanVar(value=True)
        cb_notify = ttk.Checkbutton(misc, text="Show notification when done",
                                    variable=self._show_notify_var)
        cb_notify.pack(side="left", padx=(12, 0))
        self._tip(cb_open, "open_when_done"); self._tip(cb_notify, "show_notify")

    # ---- "OCR" tab ----
    def _build_ocr_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        def labelled(row: int, label_text: str, tip_key: str) -> ttk.Label:
            lbl = ttk.Label(parent, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 6, 0))
            self._tip(lbl, tip_key)
            return lbl

        labelled(0, "Tesseract executable:", "tesseract_path")
        self._tess_var = tk.StringVar()
        ent_tess = ttk.Entry(parent, textvariable=self._tess_var)
        ent_tess.grid(row=0, column=1, sticky="ew", padx=4)
        self._tip(ent_tess, "tesseract_path")
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=0, column=2, sticky="e")
        ttk.Button(btn_row, text="Browse...", command=self._pick_tesseract).pack(side="left")
        ttk.Button(btn_row, text="Auto-detect", command=self._auto_detect_tesseract).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="Test", command=self._show_tesseract_status).pack(side="left", padx=(4, 0))

        labelled(1, "Language(s):", "lang")
        self._lang_var = tk.StringVar(value="eng")
        ent_lang = ttk.Entry(parent, textvariable=self._lang_var)
        ent_lang.grid(row=1, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(parent, text='e.g. "eng" or "eng+deu"').grid(row=1, column=2, sticky="w", pady=(6, 0))
        self._tip(ent_lang, "lang")

        labelled(2, "Preset:", "preset")
        self._preset_var = tk.StringVar(value="Default")
        preset_box = ttk.Combobox(parent, textvariable=self._preset_var,
                                  values=list(PRESETS.keys()), state="readonly")
        preset_box.grid(row=2, column=1, sticky="ew", padx=4, pady=(6, 0))
        preset_box.bind("<<ComboboxSelected>>", self._on_apply_builtin_preset)
        ttk.Label(parent, text="Built-in starting points").grid(row=2, column=2, sticky="w", pady=(6, 0))
        self._tip(preset_box, "preset")

        labelled(3, "Page segmentation (PSM):", "psm")
        self._psm_var = tk.IntVar(value=6)
        psm_box = ttk.Combobox(parent, textvariable=self._psm_var,
                               values=[3, 4, 6, 11, 12], state="readonly", width=6)
        psm_box.grid(row=3, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="6=block, 4=column, 3=auto, 11/12=sparse").grid(
            row=3, column=2, sticky="w", pady=(6, 0))
        self._tip(psm_box, "psm")

        labelled(4, "OCR engine (OEM):", "oem")
        self._oem_var = tk.IntVar(value=3)
        oem_box = ttk.Combobox(parent, textvariable=self._oem_var,
                               values=[0, 1, 2, 3], state="readonly", width=6)
        oem_box.grid(row=4, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="3 = default LSTM").grid(row=4, column=2, sticky="w", pady=(6, 0))
        self._tip(oem_box, "oem")

        labelled(5, "Render scale:", "render_scale")
        self._render_var = tk.DoubleVar(value=5.0)
        sp_render = ttk.Spinbox(parent, from_=1.0, to=10.0, increment=0.5,
                                textvariable=self._render_var, width=8)
        sp_render.grid(row=5, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="3-6 for small text; higher = slower").grid(
            row=5, column=2, sticky="w", pady=(6, 0))
        self._tip(sp_render, "render_scale")

        labelled(6, "Min word confidence:", "min_conf")
        self._min_conf_var = tk.DoubleVar(value=20.0)
        sp_conf = ttk.Spinbox(parent, from_=0.0, to=100.0, increment=1,
                              textvariable=self._min_conf_var, width=8)
        sp_conf.grid(row=6, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="0-100; higher = stricter").grid(
            row=6, column=2, sticky="w", pady=(6, 0))
        self._tip(sp_conf, "min_conf")

        labelled(7, "Pages:", "pages")
        self._pages_var = tk.StringVar(value="")
        ent_pages = ttk.Entry(parent, textvariable=self._pages_var)
        ent_pages.grid(row=7, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(parent, text="e.g. 1,2,4-5 (blank = all)").grid(
            row=7, column=2, sticky="w", pady=(6, 0))
        self._tip(ent_pages, "pages")

        # ---- Performance / accuracy section ----
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        hdr = ttk.Label(parent, text="Performance & accuracy",
                        font=("Segoe UI", 9, "bold"))
        hdr.grid(row=9, column=0, columnspan=3, sticky="w", pady=(0, 2))

        labelled(10, "Workers (CPU cores):", "workers")
        cpu = os.cpu_count() or 1
        self._workers_var = tk.IntVar(value=0)
        sp_w = ttk.Spinbox(parent, from_=0, to=cpu, increment=1,
                           textvariable=self._workers_var, width=8)
        sp_w.grid(row=10, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text=f"0 = auto ({max(1, min(cpu // 2, 4))} on this PC, max {cpu})"
                  ).grid(row=10, column=2, sticky="w", pady=(6, 0))
        self._tip(sp_w, "workers")

        self._use_text_layer_var = tk.BooleanVar(value=True)
        cb_tl = ttk.Checkbutton(
            parent,
            text="Use embedded PDF text when available (faster + perfectly accurate)",
            variable=self._use_text_layer_var,
        )
        cb_tl.grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._tip(cb_tl, "use_text_layer")

        self._best_accuracy_var = tk.BooleanVar(value=False)
        cb_best = ttk.Checkbutton(
            parent,
            text="Best-accuracy mode — runs each page with multiple settings (~2x slower)",
            variable=self._best_accuracy_var,
        )
        cb_best.grid(row=12, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._tip(cb_best, "best_accuracy")

    # ---- "Preprocessing" tab ----
    def _build_prep_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        def lbl(row: int, text: str, tip_key: str) -> None:
            w = ttk.Label(parent, text=text)
            w.grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 6, 0))
            self._tip(w, tip_key)

        self._no_gray_var = tk.BooleanVar(value=False)
        cb_gray = ttk.Checkbutton(parent, text="Keep RGB (skip grayscale conversion)",
                                  variable=self._no_gray_var)
        cb_gray.grid(row=0, column=0, columnspan=2, sticky="w")
        self._tip(cb_gray, "no_grayscale")

        lbl(1, "Contrast:", "contrast")
        self._contrast_var = tk.DoubleVar(value=1.4)
        sp = ttk.Spinbox(parent, from_=0.5, to=4.0, increment=0.1,
                         textvariable=self._contrast_var, width=8)
        sp.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="1.0 = unchanged; 1.4-1.8 for faded scans").grid(
            row=1, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "contrast")

        lbl(2, "Sharpen passes:", "sharpen")
        self._sharpen_var = tk.IntVar(value=1)
        sp = ttk.Spinbox(parent, from_=0, to=5, increment=1,
                         textvariable=self._sharpen_var, width=8)
        sp.grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="0=none, 1-2 helps small text").grid(
            row=2, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "sharpen")

        lbl(3, "Threshold:", "threshold")
        self._thresh_var = tk.StringVar(value="adaptive")
        cb = ttk.Combobox(parent, textvariable=self._thresh_var,
                          values=["none", "global", "adaptive"], state="readonly", width=10)
        cb.grid(row=3, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="adaptive recommended for real scans").grid(
            row=3, column=2, sticky="w", pady=(6, 0))
        self._tip(cb, "threshold")

        lbl(4, "Threshold value:", "threshold_value")
        self._thresh_val_var = tk.IntVar(value=180)
        sp = ttk.Spinbox(parent, from_=0, to=255, increment=5,
                         textvariable=self._thresh_val_var, width=8)
        sp.grid(row=4, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="only used when threshold = global (0-255)").grid(
            row=4, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "threshold_value")

        self._invert_var = tk.BooleanVar(value=False)
        cb_inv = ttk.Checkbutton(parent, text="Invert colours after preprocessing",
                                 variable=self._invert_var)
        cb_inv.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self._tip(cb_inv, "invert")

        lbl(6, "Dilate kernel:", "dilate")
        self._dilate_var = tk.IntVar(value=1)
        sp = ttk.Spinbox(parent, from_=0, to=10, increment=1,
                         textvariable=self._dilate_var, width=8)
        sp.grid(row=6, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="thickens strokes; raise if characters break apart").grid(
            row=6, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "dilate")

        lbl(7, "Erode kernel:", "erode")
        self._erode_var = tk.IntVar(value=0)
        sp = ttk.Spinbox(parent, from_=0, to=10, increment=1,
                         textvariable=self._erode_var, width=8)
        sp.grid(row=7, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="thins strokes; raise if characters merge").grid(
            row=7, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "erode")

        lbl(8, "Crop (l,t,r,b px):", "crop")
        self._crop_var = tk.StringVar(value="")
        ent_crop = ttk.Entry(parent, textvariable=self._crop_var)
        ent_crop.grid(row=8, column=1, sticky="ew", padx=4, pady=(6, 0))
        ttk.Label(parent, text="e.g. 50,100,50,80 — blank for no crop").grid(
            row=8, column=2, sticky="w", pady=(6, 0))
        self._tip(ent_crop, "crop")

        # ---- Auto-cleanup ----
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=9, column=0, columnspan=3, sticky="ew", pady=(12, 4))
        ttk.Label(parent, text="Auto-cleanup",
                  font=("Segoe UI", 9, "bold")).grid(
            row=10, column=0, columnspan=3, sticky="w", pady=(0, 2))

        self._deskew_var = tk.BooleanVar(value=False)
        cb_des = ttk.Checkbutton(
            parent, text="Auto-deskew (straighten tilted scans)",
            variable=self._deskew_var,
        )
        cb_des.grid(row=11, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._tip(cb_des, "deskew")

        self._denoise_var = tk.BooleanVar(value=False)
        cb_dn = ttk.Checkbutton(
            parent, text="Denoise (clean speckle and grain from scans)",
            variable=self._denoise_var,
        )
        cb_dn.grid(row=12, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self._tip(cb_dn, "denoise")

    # ---- "Layout & Noise" tab ----
    def _build_layout_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        def lbl(row: int, text: str, tip_key: str) -> None:
            w = ttk.Label(parent, text=text)
            w.grid(row=row, column=0, sticky="w", pady=(0 if row == 0 else 6, 0))
            self._tip(w, tip_key)

        lbl(0, "Row tolerance (px):", "row_tol")
        self._row_tol_var = tk.IntVar(value=12)
        sp = ttk.Spinbox(parent, from_=1, to=80, increment=1,
                         textvariable=self._row_tol_var, width=8)
        sp.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(parent, text="raise if one line splits in two").grid(
            row=0, column=2, sticky="w")
        self._tip(sp, "row_tol")

        lbl(1, "Column gap (px):", "col_gap")
        self._col_gap_var = tk.IntVar(value=22)
        sp = ttk.Spinbox(parent, from_=1, to=999, increment=1,
                         textvariable=self._col_gap_var, width=8)
        sp.grid(row=1, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="set very high to disable column splitting").grid(
            row=1, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "col_gap")

        lbl(2, "Row noise conf cutoff:", "noise_conf")
        self._noise_conf_var = tk.DoubleVar(value=55.0)
        sp = ttk.Spinbox(parent, from_=0.0, to=100.0, increment=1,
                         textvariable=self._noise_conf_var, width=8)
        sp.grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="rows below this avg conf are dropped").grid(
            row=2, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "noise_conf")

        lbl(3, "Min alphanumeric ratio:", "alpha_ratio")
        self._alpha_ratio_var = tk.DoubleVar(value=0.3)
        sp = ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.05,
                         textvariable=self._alpha_ratio_var, width=8)
        sp.grid(row=3, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="0.3 = at least 30% letters/digits").grid(
            row=3, column=2, sticky="w", pady=(6, 0))
        self._tip(sp, "alpha_ratio")

    # ---- "Diagnostics" tab ----
    def _build_diag_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        lbl_dbg = ttk.Label(parent, text="Debug folder:")
        lbl_dbg.grid(row=0, column=0, sticky="w")
        self._debug_dir_var = tk.StringVar()
        ent_dbg = ttk.Entry(parent, textvariable=self._debug_dir_var)
        ent_dbg.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse...", command=self._pick_debug_dir).grid(row=0, column=2)
        self._tip(lbl_dbg, "debug_dir"); self._tip(ent_dbg, "debug_dir")

        self._save_raw_var = tk.BooleanVar(value=False)
        cb_raw = ttk.Checkbutton(parent, text="Save raw Tesseract text into debug folder",
                                 variable=self._save_raw_var)
        cb_raw.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._tip(cb_raw, "save_raw")

        self._conf_comments_var = tk.BooleanVar(value=False)
        cb_cc = ttk.Checkbutton(parent, text="Include per-row OCR confidence comments in Markdown",
                                variable=self._conf_comments_var)
        cb_cc.grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._tip(cb_cc, "conf_comments")

        self._audit_var = tk.BooleanVar(value=True)
        cb_audit = ttk.Checkbutton(
            parent,
            text="Write per-run audit log (.json) next to outputs",
            variable=self._audit_var,
        )
        cb_audit.grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self._tip(cb_audit, "audit_log")

    # ---- "Validation" tab ----
    def _build_validation_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        intro = ttk.Label(
            parent,
            text=(
                "After a run, this tab shows the rendered image of any page "
                "with low-confidence OCR words highlighted in red. Use it to "
                "spot-check the OCR before sharing the output."
            ),
            wraplength=900, justify="left",
        )
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        controls = ttk.Frame(parent)
        controls.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(controls, text="Page:").pack(side="left")
        self._val_page_var = tk.StringVar(value="")
        self._val_page_combo = ttk.Combobox(
            controls, textvariable=self._val_page_var,
            state="readonly", width=70,
        )
        self._val_page_combo.pack(side="left", padx=(4, 8), fill="x", expand=True)
        self._val_page_combo.bind("<<ComboboxSelected>>", self._render_validation_preview)
        ttk.Button(controls, text="Refresh", command=self._refresh_validation_list
                   ).pack(side="left")
        ttk.Button(controls, text="Show report", command=self._show_last_report
                   ).pack(side="left", padx=(6, 0))

        canvas_frame = ttk.Frame(parent, relief="sunken")
        canvas_frame.grid(row=2, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self._val_canvas = tk.Canvas(canvas_frame, background="#222", highlightthickness=0)
        self._val_canvas.grid(row=0, column=0, sticky="nsew")
        vbar = ttk.Scrollbar(canvas_frame, orient="vertical",
                             command=self._val_canvas.yview)
        vbar.grid(row=0, column=1, sticky="ns")
        hbar = ttk.Scrollbar(canvas_frame, orient="horizontal",
                             command=self._val_canvas.xview)
        hbar.grid(row=1, column=0, sticky="ew")
        self._val_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self._val_preview_image = None  # keep a reference so Tk doesn't gc it

    # ---- "Watch folder" tab ----
    def _build_watch_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(
            parent,
            text=(
                "Pick a folder and click Start watching. New PDFs added to "
                "the folder are processed automatically using the current "
                "settings, and their outputs are written to the output "
                "folder on the Files & Output tab."
            ),
            wraplength=900, justify="left",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))

        lbl = ttk.Label(parent, text="Folder to watch:")
        lbl.grid(row=1, column=0, sticky="w")
        self._watch_folder_var = tk.StringVar()
        ent = ttk.Entry(parent, textvariable=self._watch_folder_var)
        ent.grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse...", command=self._pick_watch_folder
                   ).grid(row=1, column=2)
        self._tip(lbl, "watch_folder"); self._tip(ent, "watch_folder")

        lbl2 = ttk.Label(parent, text="Poll every (seconds):")
        lbl2.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._watch_interval_var = tk.IntVar(value=5)
        sp = ttk.Spinbox(parent, from_=2, to=300, increment=1,
                         textvariable=self._watch_interval_var, width=8)
        sp.grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        self._tip(lbl2, "watch_interval"); self._tip(sp, "watch_interval")

        self._watch_status_var = tk.StringVar(value="Not watching.")
        ttk.Label(parent, textvariable=self._watch_status_var,
                  foreground="#444").grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(12, 0))

        btn_row = ttk.Frame(parent)
        btn_row.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self._watch_start_btn = ttk.Button(
            btn_row, text="Start watching", command=self._start_watch,
        )
        self._watch_start_btn.pack(side="left")
        self._watch_stop_btn = ttk.Button(
            btn_row, text="Stop watching", command=self._stop_watch, state="disabled",
        )
        self._watch_stop_btn.pack(side="left", padx=(6, 0))

    # ---- file/folder pickers ----
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select PDFs",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        for p in paths:
            if p not in self._files_list.get(0, "end"):
                self._files_list.insert("end", p)
            self._push_recent_file(p)
        self._maybe_default_output_dir()

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder of PDFs")
        if not folder:
            return
        self._push_recent_folder(folder)
        added = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full = str(Path(root) / f)
                    if full not in self._files_list.get(0, "end"):
                        self._files_list.insert("end", full)
                        added += 1
        self._log(f"Added {added} PDF file(s) from {folder}")
        self._maybe_default_output_dir()

    def _remove_selected(self) -> None:
        for i in reversed(self._files_list.curselection()):
            self._files_list.delete(i)

    def _clear_files(self) -> None:
        self._files_list.delete(0, "end")

    def _move_selected(self, delta: int) -> None:
        sel = list(self._files_list.curselection())
        if not sel:
            return
        if delta < 0:
            order = sel
        else:
            order = list(reversed(sel))
        for i in order:
            j = i + delta
            if j < 0 or j >= self._files_list.size():
                continue
            text = self._files_list.get(i)
            self._files_list.delete(i)
            self._files_list.insert(j, text)
            self._files_list.selection_set(j)

    def _pick_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self._out_dir_var.set(folder)

    def _pick_tesseract(self) -> None:
        path = filedialog.askopenfilename(
            title="Find tesseract.exe",
            filetypes=[("Tesseract executable", "tesseract*"), ("All files", "*.*")],
        )
        if path:
            self._tess_var.set(path)
            self._validate_tesseract(silent=False)

    def _pick_debug_dir(self) -> None:
        folder = filedialog.askdirectory(title="Choose debug folder")
        if folder:
            self._debug_dir_var.set(folder)

    def _maybe_default_output_dir(self) -> None:
        if self._out_dir_var.get().strip():
            return
        files = list(self._files_list.get(0, "end"))
        if files:
            self._out_dir_var.set(str(Path(files[0]).parent))

    # ---- presets ----
    def _on_apply_builtin_preset(self, _event: Any) -> None:
        name = self._preset_var.get()
        overrides = PRESETS.get(name, {})
        if not overrides:
            return
        base = OcrSettings()
        for k, v in overrides.items():
            setattr(base, k, v)
        # only apply OCR/preprocessing/layout fields, leave file lists alone
        self._apply_settings(base, only_processing=True)
        self._log(f"Applied built-in preset: {name}")

    def _save_preset(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save preset",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        s = self._collect_settings()
        # presets store everything except input files & output paths
        data = asdict(s)
        for k in ("inputs", "output_dir", "debug_dir", "tesseract_path"):
            data.pop(k, None)
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._log(f"Saved preset to {path}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Could not load preset", str(exc))
            return
        s = OcrSettings()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        self._apply_settings(s, only_processing=True)
        self._log(f"Loaded preset from {path}")

    # ---- tesseract status ----
    def _auto_detect_tesseract(self) -> None:
        current = self._tess_var.get().strip()
        if current and Path(current).exists():
            ok, _msg = tesseract_works(current)
            if ok:
                return
        found = find_tesseract()
        if found:
            self._tess_var.set(str(found))
            self._log(f"Detected Tesseract at: {found}")
        else:
            self._log("Tesseract not detected automatically. Use Browse... to locate it.")

    def _validate_tesseract(self, silent: bool = True) -> bool:
        path = self._tess_var.get().strip() or None
        ok, msg = tesseract_works(path)
        if not silent:
            if ok:
                messagebox.showinfo("Tesseract OK", msg)
            else:
                messagebox.showerror("Tesseract not working", msg)
        return ok

    def _show_tesseract_status(self) -> None:
        path = self._tess_var.get().strip() or pytesseract.pytesseract.tesseract_cmd
        ok, msg = tesseract_works(path)
        title = "Tesseract status"
        body = f"Path: {path}\n\n{msg}"
        if ok:
            messagebox.showinfo(title, body)
        else:
            messagebox.showwarning(
                title,
                body + "\n\nDownload Tesseract from:\n"
                       "  https://github.com/UB-Mannheim/tesseract/wiki"
            )

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About",
            f"{APP_TITLE} v{APP_VERSION}\n\n"
            "Tkinter front-end for OCR_PDF_to_Markdown.py.\n"
            "Uses PyMuPDF, Pillow, pytesseract, OpenCV.\n"
        )

    def _show_settings_reference(self) -> None:
        """Open a scrollable window that explains every setting, grouped by tab."""
        win = tk.Toplevel(self)
        win.title("Settings reference")
        win.geometry("760x620")
        win.transient(self)

        outer = ttk.Frame(win, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(
            outer,
            text="Hover over any field in the main window for the same help. "
                 "This reference lists every setting in one place.",
            wraplength=720, justify="left",
        ).pack(anchor="w", pady=(0, 6))

        text_frame = ttk.Frame(outer)
        text_frame.pack(fill="both", expand=True)
        txt = tk.Text(text_frame, wrap="word", font=("Segoe UI", 10))
        txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(text_frame, command=txt.yview)
        sb.pack(side="right", fill="y")
        txt.configure(yscrollcommand=sb.set)

        txt.tag_configure("h1", font=("Segoe UI", 12, "bold"), spacing3=4, spacing1=8)
        txt.tag_configure("h2", font=("Segoe UI", 10, "bold"), spacing3=2, spacing1=4)
        txt.tag_configure("body", lmargin1=12, lmargin2=12, spacing3=6)

        groups: List[Tuple[str, List[Tuple[str, str]]]] = [
            ("Files & Output", [
                ("Input queue",          "files_list"),
                ("Output folder",        "output_dir"),
                ("Output mode",          "output_mode"),
                ("Combined filename",    "combined_name"),
                ("Format: Markdown",     "format_markdown"),
                ("Format: Plain text",   "format_text"),
                ("Format: JSON",         "format_json"),
                ("Format: CSV",          "format_csv"),
                ("Format: HTML",         "format_html"),
                ("Format: Searchable PDF","format_searchable_pdf"),
                ("Open folder when done","open_when_done"),
                ("Notification",         "show_notify"),
            ]),
            ("OCR", [
                ("Tesseract executable", "tesseract_path"),
                ("Language(s)",          "lang"),
                ("Preset",               "preset"),
                ("Page segmentation (PSM)","psm"),
                ("OCR engine (OEM)",     "oem"),
                ("Render scale",         "render_scale"),
                ("Min word confidence",  "min_conf"),
                ("Pages",                "pages"),
                ("Workers (CPU cores)",  "workers"),
                ("Use embedded PDF text","use_text_layer"),
                ("Best-accuracy mode",   "best_accuracy"),
            ]),
            ("Preprocessing", [
                ("Keep RGB",             "no_grayscale"),
                ("Contrast",             "contrast"),
                ("Sharpen passes",       "sharpen"),
                ("Threshold",            "threshold"),
                ("Threshold value",      "threshold_value"),
                ("Invert",               "invert"),
                ("Dilate kernel",        "dilate"),
                ("Erode kernel",         "erode"),
                ("Crop",                 "crop"),
                ("Auto-deskew",          "deskew"),
                ("Denoise",              "denoise"),
            ]),
            ("Layout & Noise", [
                ("Row tolerance",        "row_tol"),
                ("Column gap",           "col_gap"),
                ("Row noise conf cutoff","noise_conf"),
                ("Min alphanumeric ratio","alpha_ratio"),
            ]),
            ("Diagnostics", [
                ("Debug folder",         "debug_dir"),
                ("Save raw text",        "save_raw"),
                ("Confidence comments",  "conf_comments"),
                ("Audit log",            "audit_log"),
            ]),
            ("Watch folder", [
                ("Folder to watch",      "watch_folder"),
                ("Poll interval",        "watch_interval"),
            ]),
        ]

        for group_title, items in groups:
            txt.insert("end", group_title + "\n", "h1")
            for label, key in items:
                body = TOOLTIPS.get(key, "(no description)")
                txt.insert("end", label + "\n", "h2")
                txt.insert("end", body + "\n", "body")
        txt.configure(state="disabled")

        ttk.Button(outer, text="Close", command=win.destroy).pack(anchor="e", pady=(6, 0))

    # ---- Welcome / first-launch tour ----
    def _maybe_show_welcome(self) -> None:
        if not getattr(self, "_seen_welcome", False):
            self._show_welcome()

    def _show_welcome(self) -> None:
        win = tk.Toplevel(self)
        win.title("Quick start — PDF OCR Tool")
        win.geometry("680x560")
        win.transient(self)
        outer = ttk.Frame(win, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer, text=f"Welcome to {APP_TITLE}",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        steps: List[Tuple[str, str]] = [
            ("1. Add your PDFs",
             "On the Files & Output tab, click 'Add files…' or 'Add folder…' "
             "to put PDFs in the queue, then pick an output folder."),
            ("2. Pick output formats",
             "Tick one or more boxes — Markdown, plain text, JSON, CSV, "
             "HTML, or searchable PDF. The tool produces all of them in a "
             "single run."),
            ("3. (Optional) Tune settings",
             "Hover over any field in the OCR / Preprocessing / Layout tabs "
             "for an explanation. Help → Settings reference lists them all in "
             "one place. The defaults work for most reports."),
            ("4. Click Start OCR",
             "Watch the log at the bottom. The Workers spinbox on the OCR "
             "tab lets you spend more CPU cores for faster results."),
            ("5. Check the result",
             "When it's done, the Validation tab highlights low-confidence "
             "OCR words so you can spot-check accuracy. Help → Run self-test "
             "verifies your install is working correctly."),
        ]
        for title, body in steps:
            ttk.Label(outer, text=title, font=("Segoe UI", 10, "bold")
                      ).pack(anchor="w", pady=(8, 0))
            ttk.Label(outer, text=body, wraplength=620, justify="left"
                      ).pack(anchor="w")

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(12, 6))
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        self._dont_show_var = tk.BooleanVar(
            value=getattr(self, "_seen_welcome", False),
        )
        ttk.Checkbutton(bar, text="Don't show this again",
                        variable=self._dont_show_var).pack(side="left")

        def _close() -> None:
            self._seen_welcome = bool(self._dont_show_var.get())
            self._save_config()
            win.destroy()
        ttk.Button(bar, text="Got it", command=_close).pack(side="right")
        win.protocol("WM_DELETE_WINDOW", _close)

    # ---- Recent files menu ----
    def _rebuild_recent_menus(self) -> None:
        if not hasattr(self, "_recent_files_menu"):
            return
        self._recent_files_menu.delete(0, "end")
        recent = getattr(self, "_recent_files_list", []) or []
        if not recent:
            self._recent_files_menu.add_command(label="(empty)", state="disabled")
        else:
            for p in recent:
                self._recent_files_menu.add_command(
                    label=self._abbr_path(p),
                    command=lambda path=p: self._add_recent_to_queue(path),
                )
            self._recent_files_menu.add_separator()
            self._recent_files_menu.add_command(
                label="Clear list", command=self._clear_recent_files,
            )

        self._recent_folders_menu.delete(0, "end")
        folders = getattr(self, "_recent_folders_list", []) or []
        if not folders:
            self._recent_folders_menu.add_command(label="(empty)", state="disabled")
        else:
            for f in folders:
                self._recent_folders_menu.add_command(
                    label=self._abbr_path(f),
                    command=lambda folder=f: self._add_recent_folder_to_queue(folder),
                )
            self._recent_folders_menu.add_separator()
            self._recent_folders_menu.add_command(
                label="Clear list", command=self._clear_recent_folders,
            )

    @staticmethod
    def _abbr_path(p: str, limit: int = 60) -> str:
        if len(p) <= limit:
            return p
        return "..." + p[-(limit - 3):]

    def _push_recent_file(self, path: str) -> None:
        if not hasattr(self, "_recent_files_list"):
            self._recent_files_list = []
        lst = self._recent_files_list
        if path in lst:
            lst.remove(path)
        lst.insert(0, path)
        del lst[10:]
        self._rebuild_recent_menus()

    def _push_recent_folder(self, path: str) -> None:
        if not hasattr(self, "_recent_folders_list"):
            self._recent_folders_list = []
        lst = self._recent_folders_list
        if path in lst:
            lst.remove(path)
        lst.insert(0, path)
        del lst[5:]
        self._rebuild_recent_menus()

    def _clear_recent_files(self) -> None:
        self._recent_files_list = []
        self._rebuild_recent_menus()

    def _clear_recent_folders(self) -> None:
        self._recent_folders_list = []
        self._rebuild_recent_menus()

    def _add_recent_to_queue(self, path: str) -> None:
        if Path(path).exists():
            if path not in self._files_list.get(0, "end"):
                self._files_list.insert("end", path)
            self._maybe_default_output_dir()
            self._push_recent_file(path)
        else:
            messagebox.showwarning("Missing", f"File no longer exists:\n{path}")

    def _add_recent_folder_to_queue(self, folder: str) -> None:
        if not Path(folder).is_dir():
            messagebox.showwarning("Missing", f"Folder no longer exists:\n{folder}")
            return
        added = 0
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(".pdf"):
                    full = str(Path(root) / f)
                    if full not in self._files_list.get(0, "end"):
                        self._files_list.insert("end", full)
                        added += 1
        self._log(f"Added {added} PDF(s) from {folder}")
        self._push_recent_folder(folder)
        self._maybe_default_output_dir()

    # ---- Validation tab ----
    def _refresh_validation_list(self) -> None:
        entries: List[Tuple[str, Dict[str, Any]]] = []
        for metrics in getattr(self, "_last_metrics", []) or []:
            pdf_name = Path(metrics.get("pdf", "")).name
            for res in metrics.get("_page_results", []) or []:
                if res.is_error:
                    label = (f"{pdf_name}  ·  page {res.page_index + 1}  ·  "
                             f"ERROR ({res.error})")
                elif res.used_text_layer:
                    label = (f"{pdf_name}  ·  page {res.page_index + 1}  ·  "
                             "text-layer (no OCR)")
                else:
                    label = (f"{pdf_name}  ·  page {res.page_index + 1}  ·  "
                             f"avg conf {res.avg_conf:.1f}, "
                             f"{res.word_count} words")
                entries.append((label, {"pdf": metrics.get("pdf"),
                                         "result": res}))
        self._val_entries = entries
        labels = [e[0] for e in entries]
        self._val_page_combo["values"] = labels
        if labels:
            self._val_page_combo.current(0)
            self._render_validation_preview()
        else:
            self._val_canvas.delete("all")
            self._val_canvas.create_text(
                20, 20, anchor="nw", fill="#aaa",
                text="No run yet. Click Start OCR on the Files & Output tab.",
            )

    def _render_validation_preview(self, _event: Any = None) -> None:
        try:
            idx = self._val_page_combo.current()
        except Exception:
            idx = -1
        if idx < 0 or idx >= len(getattr(self, "_val_entries", [])):
            return
        entry = self._val_entries[idx][1]
        pdf_path = entry["pdf"]
        res = entry["result"]
        if not pdf_path or not Path(pdf_path).exists():
            return

        # Re-render the page at a reasonable preview scale so the displayed
        # canvas image stays manageable. Scale down OCR-time coordinates to
        # match the preview's render scale.
        try:
            doc = fitz.open(pdf_path)
            try:
                page = doc[res.page_index]
                preview_scale = 2.0
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(preview_scale, preview_scale),
                    alpha=False,
                )
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            finally:
                doc.close()
        except Exception as exc:
            messagebox.showerror("Preview failed", str(exc))
            return

        try:
            from PIL import ImageDraw, ImageTk
        except Exception:
            messagebox.showerror("Preview failed",
                                 "Pillow ImageTk unavailable in this build.")
            return

        draw = ImageDraw.Draw(img)
        if not res.used_text_layer and res.words and res.rendered_size != (0, 0):
            scale_factor = preview_scale / max(0.001, self._last_run_settings.render_scale
                                                if self._last_run_settings else 1.0)
            for w in res.words:
                if w.conf < 70:
                    x1 = int(w.x * scale_factor)
                    y1 = int(w.y * scale_factor)
                    x2 = int((w.x + w.w) * scale_factor)
                    y2 = int((w.y + w.h) * scale_factor)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        self._val_preview_image = ImageTk.PhotoImage(img)
        self._val_canvas.delete("all")
        self._val_canvas.create_image(
            0, 0, anchor="nw", image=self._val_preview_image,
        )
        self._val_canvas.configure(scrollregion=(0, 0, img.width, img.height))

    # ---- Last validation report ----
    def _show_last_report(self) -> None:
        if not self._last_metrics:
            messagebox.showinfo(
                "Validation report",
                "No run yet. Click Start OCR on the Files & Output tab.",
            )
            return
        win = tk.Toplevel(self)
        win.title("Last validation report")
        win.geometry("780x560")
        win.transient(self)
        outer = ttk.Frame(win, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=f"Total wall-clock time: {self._last_run_elapsed:.1f} s",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")

        cols = ("file", "pages", "ocr", "text-layer", "avg conf", "low conf", "words", "elapsed")
        tree = ttk.Treeview(outer, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (260, 60, 60, 80, 80, 80, 80, 70)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="w")
        tree.pack(fill="both", expand=True, pady=(8, 6))

        for m in self._last_metrics:
            tree.insert("", "end", values=(
                Path(m.get("pdf", "")).name,
                m.get("pages_processed", 0),
                m.get("pages_via_ocr", 0),
                m.get("pages_via_text_layer", 0),
                f"{m.get('avg_confidence', 0):.1f}",
                len(m.get("low_confidence_pages", [])),
                m.get("total_words", 0),
                f"{m.get('elapsed_sec', 0):.1f}s",
            ))

        ttk.Label(outer, text="Low-confidence pages (avg conf < 70):"
                  ).pack(anchor="w", pady=(8, 2))
        lo = tk.Text(outer, height=6, wrap="word")
        lo.pack(fill="both", expand=False)
        any_low = False
        for m in self._last_metrics:
            for page, conf in m.get("low_confidence_pages", []):
                lo.insert("end",
                          f"  {Path(m.get('pdf','')).name} · page {page} · "
                          f"avg conf {conf}\n")
                any_low = True
        if not any_low:
            lo.insert("end", "  (none — all pages above 70 average confidence)\n")
        lo.configure(state="disabled")

        ttk.Button(outer, text="Close", command=win.destroy
                   ).pack(anchor="e", pady=(6, 0))

    # ---- Self-test ----
    def _run_self_test(self) -> None:
        from tkinter import scrolledtext
        ok, msg = tesseract_works(self._tess_var.get().strip() or None)
        if not ok:
            messagebox.showerror(
                "Self-test failed",
                "Tesseract is not available. Configure it on the OCR tab "
                f"first.\n\n{msg}",
            )
            return

        path = self._tess_var.get().strip() or None
        if path:
            pytesseract.pytesseract.tesseract_cmd = path

        progress = tk.Toplevel(self)
        progress.title("Running self-test")
        progress.geometry("420x140")
        progress.transient(self)
        ttk.Label(progress,
                  text="Generating a known PDF and OCR'ing it…",
                  padding=10).pack()
        bar = ttk.Progressbar(progress, mode="indeterminate", length=320)
        bar.pack(pady=8)
        bar.start(8)
        self.update_idletasks()

        def work() -> None:
            try:
                report = _self_test_run(self._lang_var.get().strip() or "eng")
                self.after(0, lambda: self._show_self_test_result(report, progress))
            except Exception as exc:
                self.after(0, lambda: self._show_self_test_result(
                    {"ok": False, "error": str(exc)}, progress))

        threading.Thread(target=work, daemon=True).start()

    def _show_self_test_result(self, report: Dict[str, Any], progress: tk.Toplevel) -> None:
        try:
            progress.destroy()
        except Exception:
            pass
        if not report.get("ok"):
            messagebox.showerror(
                "Self-test failed",
                report.get("error", "Unknown error."),
            )
            return
        accuracy = report["accuracy"] * 100.0
        body = (
            f"Self-test {accuracy:.1f}% accurate.\n\n"
            f"  Tesseract version : {report['tess_version']}\n"
            f"  Words expected    : {report['expected_words']}\n"
            f"  Words matched     : {report['matched_words']}\n"
            f"  Pages tested      : {report['pages']}\n"
            f"  Elapsed           : {report['elapsed']:.2f}s"
        )
        if accuracy >= 90:
            messagebox.showinfo("Self-test passed", body)
        else:
            messagebox.showwarning(
                "Self-test below 90%",
                body + "\n\nTesseract is reading the test PDF but accuracy "
                       "is low. Check the OCR settings (PSM, render scale).",
            )

    # ---- Watch folder ----
    def _pick_watch_folder(self) -> None:
        folder = filedialog.askdirectory(title="Pick folder to watch")
        if folder:
            self._watch_folder_var.set(folder)

    def _start_watch(self) -> None:
        folder = self._watch_folder_var.get().strip()
        if not folder or not Path(folder).is_dir():
            messagebox.showerror("Watch folder", "Pick a valid folder first.")
            return
        if not self._out_dir_var.get().strip():
            messagebox.showerror(
                "Watch folder",
                "Set an output folder on the Files & Output tab first.",
            )
            return
        self._watch_stop.clear()
        # seed with current contents so we don't re-process backlog
        self._watch_seen = {
            str(p) for p in Path(folder).glob("*.pdf") if p.is_file()
        }
        interval = max(2, int(self._watch_interval_var.get()))
        self._watch_thread = threading.Thread(
            target=self._watch_loop, args=(folder, interval), daemon=True,
        )
        self._watch_thread.start()
        self._watch_start_btn.configure(state="disabled")
        self._watch_stop_btn.configure(state="normal")
        self._watch_status_var.set(
            f"Watching {folder} every {interval}s "
            f"(currently ignoring {len(self._watch_seen)} pre-existing file(s))"
        )
        self._log(f"Started watching folder: {folder}")

    def _stop_watch(self) -> None:
        self._watch_stop.set()
        self._watch_start_btn.configure(state="normal")
        self._watch_stop_btn.configure(state="disabled")
        self._watch_status_var.set("Not watching.")
        self._log("Stopped folder watcher.")

    def _watch_loop(self, folder: str, interval: int) -> None:
        while not self._watch_stop.is_set():
            try:
                current = {str(p) for p in Path(folder).glob("*.pdf") if p.is_file()}
                new = sorted(current - self._watch_seen)
                if new:
                    self._watch_seen |= set(new)
                    self._log(f"[watch] {len(new)} new PDF(s) — queuing")
                    self.after(0, lambda items=new: self._queue_and_run_watched(items))
            except Exception as exc:
                self._log(f"[watch] error: {exc}")
            self._watch_stop.wait(interval)

    def _queue_and_run_watched(self, paths: List[str]) -> None:
        if self._worker and self._worker.is_alive():
            self._log("[watch] OCR busy; deferring new files until the current run finishes")
            return
        for p in paths:
            if p not in self._files_list.get(0, "end"):
                self._files_list.insert("end", p)
        self._on_start()

    # ---- settings I/O ----
    def _collect_settings(self) -> OcrSettings:
        formats = [k for k, var in self._fmt_vars.items() if var.get()]
        return OcrSettings(
            inputs=list(self._files_list.get(0, "end")),
            output_dir=self._out_dir_var.get().strip(),
            output_formats=formats,
            output_mode=self._mode_var.get(),
            combined_basename=self._combined_name_var.get().strip() or "ocr_output",

            tesseract_path=self._tess_var.get().strip(),
            lang=self._lang_var.get().strip() or "eng",
            psm=int(self._psm_var.get()),
            oem=int(self._oem_var.get()),
            render_scale=float(self._render_var.get()),
            min_conf=float(self._min_conf_var.get()),
            pages=self._pages_var.get().strip(),

            no_grayscale=bool(self._no_gray_var.get()),
            contrast=float(self._contrast_var.get()),
            sharpen=int(self._sharpen_var.get()),
            threshold=self._thresh_var.get(),
            threshold_value=int(self._thresh_val_var.get()),
            invert=bool(self._invert_var.get()),
            dilate=int(self._dilate_var.get()),
            erode=int(self._erode_var.get()),
            crop=self._crop_var.get().strip(),

            row_tol=int(self._row_tol_var.get()),
            col_gap=int(self._col_gap_var.get()),
            noise_conf=float(self._noise_conf_var.get()),
            noise_alpha_ratio=float(self._alpha_ratio_var.get()),

            debug_dir=self._debug_dir_var.get().strip(),
            save_raw=bool(self._save_raw_var.get()),
            confidence_comments=bool(self._conf_comments_var.get()),

            open_when_done=bool(self._open_when_done_var.get()),
            show_notification=bool(self._show_notify_var.get()),

            workers=int(self._workers_var.get()),
            use_text_layer=bool(self._use_text_layer_var.get()),
            deskew=bool(self._deskew_var.get()),
            denoise=bool(self._denoise_var.get()),
            best_accuracy=bool(self._best_accuracy_var.get()),
            write_audit_log=bool(self._audit_var.get()),
            seen_welcome=getattr(self, "_seen_welcome", False),
            recent_files=list(getattr(self, "_recent_files_list", [])),
            recent_folders=list(getattr(self, "_recent_folders_list", [])),
            watch_folder=self._watch_folder_var.get().strip(),
            watch_interval_sec=int(self._watch_interval_var.get()),
        )

    def _apply_settings(self, s: OcrSettings, only_processing: bool = False) -> None:
        if not only_processing:
            self._files_list.delete(0, "end")
            for f in s.inputs:
                self._files_list.insert("end", f)
            self._out_dir_var.set(s.output_dir)
            self._mode_var.set(s.output_mode)
            self._combined_name_var.set(s.combined_basename)
            for key, var in self._fmt_vars.items():
                var.set(key in s.output_formats)
            self._tess_var.set(s.tesseract_path)
            self._debug_dir_var.set(s.debug_dir)
            self._open_when_done_var.set(s.open_when_done)
            self._show_notify_var.set(s.show_notification)
            self._seen_welcome = s.seen_welcome
            self._recent_files_list = list(s.recent_files)
            self._recent_folders_list = list(s.recent_folders)
            self._rebuild_recent_menus()
            self._watch_folder_var.set(s.watch_folder)
            self._watch_interval_var.set(max(2, int(s.watch_interval_sec or 5)))

        self._lang_var.set(s.lang)
        self._psm_var.set(s.psm)
        self._oem_var.set(s.oem)
        self._render_var.set(s.render_scale)
        self._min_conf_var.set(s.min_conf)
        self._pages_var.set(s.pages)

        self._no_gray_var.set(s.no_grayscale)
        self._contrast_var.set(s.contrast)
        self._sharpen_var.set(s.sharpen)
        self._thresh_var.set(s.threshold)
        self._thresh_val_var.set(s.threshold_value)
        self._invert_var.set(s.invert)
        self._dilate_var.set(s.dilate)
        self._erode_var.set(s.erode)
        self._crop_var.set(s.crop)

        self._row_tol_var.set(s.row_tol)
        self._col_gap_var.set(s.col_gap)
        self._noise_conf_var.set(s.noise_conf)
        self._alpha_ratio_var.set(s.noise_alpha_ratio)

        self._save_raw_var.set(s.save_raw)
        self._conf_comments_var.set(s.confidence_comments)
        self._audit_var.set(s.write_audit_log)

        self._workers_var.set(s.workers)
        self._use_text_layer_var.set(s.use_text_layer)
        self._deskew_var.set(s.deskew)
        self._denoise_var.set(s.denoise)
        self._best_accuracy_var.set(s.best_accuracy)

    def _load_config(self) -> None:
        if not CONFIG_FILE.exists():
            self._seen_welcome = False
            self._recent_files_list = []
            self._recent_folders_list = []
            self._rebuild_recent_menus()
            return
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            return
        s = OcrSettings()
        for k, v in data.items():
            if hasattr(s, k):
                setattr(s, k, v)
        self._apply_settings(s, only_processing=False)

    def _save_config(self) -> None:
        try:
            s = self._collect_settings()
            CONFIG_FILE.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")
        except Exception:
            pass

    # ---- log helpers ----
    def _log(self, msg: str) -> None:
        self._log_queue.put(("log", msg))

    def _set_status(self, msg: str) -> None:
        self._log_queue.put(("status", msg))

    def _set_progress(self, value: int, maximum: int) -> None:
        self._log_queue.put(("progress", value, maximum))

    def _drain_log_queue(self) -> None:
        while True:
            try:
                item = self._log_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == "log":
                self._log_text.configure(state="normal")
                self._log_text.insert("end", item[1] + "\n")
                self._log_text.see("end")
                self._log_text.configure(state="disabled")
            elif kind == "status":
                self._status_var.set(item[1])
            elif kind == "progress":
                _, value, maximum = item
                self._progress.configure(maximum=max(1, maximum), value=value)
            elif kind == "done":
                ok, summary = item[1], item[2]
                self._on_worker_done(ok, summary)
        self.after(100, self._drain_log_queue)

    def _clear_log(self) -> None:
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ---- start / cancel ----
    def _on_start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        s = self._collect_settings()
        err = self._validate(s)
        if err:
            messagebox.showerror("Cannot start", err)
            return
        self._save_config()

        self._cancel_event.clear()
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self._set_status("Running...")
        self._set_progress(0, 1)
        self._log("=" * 60)
        self._log(f"Starting OCR on {len(s.inputs)} file(s)")
        self._log(f"Output folder: {s.output_dir}")
        self._log(f"Formats: {', '.join(s.output_formats)}")
        self._log(f"Mode: {s.output_mode}")

        self._worker = threading.Thread(
            target=self._worker_main, args=(s,), daemon=True
        )
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker and self._worker.is_alive():
            self._cancel_event.set()
            self._set_status("Cancelling...")
            self._log("Cancellation requested.")

    def _on_worker_done(self, ok: bool, summary: str) -> None:
        self._start_btn.configure(state="normal")
        self._cancel_btn.configure(state="disabled")
        self._set_progress(0, 1)
        if ok:
            self._set_status("Done.")
            try:
                self._refresh_validation_list()
            except Exception:
                pass
            if self._show_notify_var.get():
                messagebox.showinfo("OCR complete", summary)
            if self._open_when_done_var.get():
                self._open_in_explorer(self._out_dir_var.get())
        else:
            self._set_status("Failed.")
            if self._show_notify_var.get():
                messagebox.showerror("OCR failed", summary)

    def _validate(self, s: OcrSettings) -> Optional[str]:
        if not s.inputs:
            return "Add at least one PDF."
        if not s.output_dir:
            return "Choose an output folder."
        if not s.output_formats:
            return "Pick at least one output format."
        for f in s.inputs:
            if not Path(f).exists():
                return f"Missing input file: {f}"
        ok, msg = tesseract_works(s.tesseract_path or None)
        if not ok:
            return (
                "Tesseract OCR is not available.\n\n"
                f"{msg}\n\n"
                "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
                "and point the OCR tab to tesseract.exe."
            )
        return None

    # ---- worker thread ----
    def _worker_main(self, settings: OcrSettings) -> None:
        from concurrent.futures import ThreadPoolExecutor
        import time

        executor: Optional[ThreadPoolExecutor] = None
        per_file_metrics: List[Dict[str, Any]] = []
        # Tesseract uses OpenMP internally. With many threads each spawning a
        # Tesseract subprocess, every subprocess tries to grab all cores and
        # they thrash. Cap each Tesseract subprocess to a single OpenMP
        # thread so our page-level parallelism is the only level that scales.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        try:
            Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
            if settings.debug_dir:
                Path(settings.debug_dir).mkdir(parents=True, exist_ok=True)

            workers = self._effective_workers(settings)
            if workers > 1:
                try:
                    # Threads: PyMuPDF render, OpenCV preprocessing, and the
                    # Tesseract subprocess all release the GIL, so threads
                    # give real parallelism without the cold-start cost of
                    # spawning child Python interpreters in a frozen .exe.
                    executor = ThreadPoolExecutor(
                        max_workers=workers,
                        thread_name_prefix="ocr",
                    )
                    self._log(f"Using {workers} parallel worker thread(s).")
                except Exception as exc:
                    self._log(
                        f"Could not start worker pool ({exc}); "
                        "falling back to single-worker mode."
                    )
                    executor = None
            else:
                self._log("Running single-worker mode.")

            per_file_results: List[Tuple[Path, Dict[str, Any]]] = []
            n_files = len(settings.inputs)
            run_started = time.perf_counter()

            for file_idx, pdf in enumerate(settings.inputs):
                if self._cancel_event.is_set():
                    raise CancelledError()
                pdf_path = Path(pdf)
                self._log(f"[{file_idx + 1}/{n_files}] {pdf_path.name}")
                self._set_status(f"{pdf_path.name} ({file_idx + 1}/{n_files})")

                # Encrypted PDF? Prompt for password on the GUI thread.
                pdf_password = self._prompt_password_if_needed(pdf_path)
                if pdf_password is None:
                    self._log("    [skipped] password not provided")
                    continue

                outputs, metrics = process_pdf_multi(
                    pdf_path, pdf_password, settings,
                    log=self._log,
                    progress=self._set_progress,
                    cancel_event=self._cancel_event,
                    executor=executor,
                )
                per_file_results.append((pdf_path, outputs))
                per_file_metrics.append(metrics)

                if settings.output_mode == "per_file":
                    self._write_outputs(
                        pdf_path.stem, outputs, settings.output_dir
                    )
                    if settings.write_audit_log:
                        self._write_audit_log(
                            pdf_path.stem, metrics, settings,
                        )

            if settings.output_mode == "combined":
                self._write_combined(
                    per_file_results,
                    settings.combined_basename,
                    settings.output_dir,
                    settings.output_formats,
                )
                if settings.write_audit_log:
                    self._write_audit_log(
                        settings.combined_basename,
                        {"per_file": [
                            {k: v for k, v in m.items() if k != "_page_results"}
                            for m in per_file_metrics
                        ]},
                        settings,
                    )

            self._last_metrics = per_file_metrics
            self._last_run_settings = settings
            self._last_run_elapsed = time.perf_counter() - run_started
            self._log_queue.put((
                "done", True,
                self._summarize_run(per_file_metrics, n_files,
                                     settings.output_dir),
            ))
        except CancelledError:
            self._log("Cancelled.")
            self._log_queue.put(("done", False, "Cancelled by user."))
        except Exception as exc:
            self._log("ERROR: " + str(exc))
            self._log(traceback.format_exc())
            self._log_queue.put(("done", False, f"{type(exc).__name__}: {exc}"))
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)

    def _effective_workers(self, settings: OcrSettings) -> int:
        cpu = os.cpu_count() or 1
        if settings.workers and settings.workers > 0:
            return max(1, min(settings.workers, cpu))
        # auto: half the cores, capped at 4 to leave headroom for the OS
        return max(1, min(cpu // 2, 4))

    def _summarize_run(
        self, metrics: List[Dict[str, Any]], n_files: int, out_dir: str,
    ) -> str:
        if not metrics:
            return f"No files processed.\nOutput folder:\n{out_dir}"
        total_pages = sum(m.get("pages_processed", 0) for m in metrics)
        total_words = sum(m.get("total_words", 0) for m in metrics)
        ocr_pages = sum(m.get("pages_via_ocr", 0) for m in metrics)
        text_layer = sum(m.get("pages_via_text_layer", 0) for m in metrics)
        avg_conf_vals = [m.get("avg_confidence", 0.0) for m in metrics if m.get("pages_via_ocr")]
        avg_conf = sum(avg_conf_vals) / len(avg_conf_vals) if avg_conf_vals else 0.0
        low = sum(len(m.get("low_confidence_pages", [])) for m in metrics)
        errs = sum(m.get("pages_errored", 0) for m in metrics)
        lines = [
            f"Processed {n_files} file(s), {total_pages} page(s), {total_words:,} words.",
            f"OCR pages: {ocr_pages}   Text-layer pages: {text_layer}",
            f"Average OCR confidence: {avg_conf:.1f} / 100",
            f"Low-confidence pages (<70): {low}",
        ]
        if errs:
            lines.append(f"Errors: {errs}")
        lines.append("")
        lines.append(f"Output folder:\n{out_dir}")
        lines.append("\nOpen Help → Last validation report for the per-file breakdown.")
        return "\n".join(lines)

    def _write_audit_log(
        self, basename: str, metrics: Dict[str, Any], settings: OcrSettings,
    ) -> None:
        try:
            import hashlib, datetime
            tess_path = settings.tesseract_path or pytesseract.pytesseract.tesseract_cmd
            tess_version = ""
            try:
                proc = subprocess.run([tess_path, "--version"],
                                      capture_output=True, text=True, timeout=10)
                tess_version = (proc.stdout or proc.stderr).splitlines()[0:1]
                tess_version = tess_version[0] if tess_version else ""
            except Exception:
                pass

            # Strip transient fields and live PageResult objects
            sanitized: Dict[str, Any] = {}
            for k, v in metrics.items():
                if k == "_page_results":
                    continue
                sanitized[k] = v
            if "per_file" in sanitized:
                for entry in sanitized["per_file"]:
                    entry.pop("_page_results", None)

            audit = {
                "tool": f"{APP_TITLE} v{APP_VERSION}",
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "tesseract": tess_version,
                "settings": asdict(settings),
                "metrics": sanitized,
            }
            # SHA-256 of the source PDF when we have one path
            pdf_path_str = metrics.get("pdf")
            if isinstance(pdf_path_str, str) and Path(pdf_path_str).exists():
                with open(pdf_path_str, "rb") as f:
                    audit["source_sha256"] = hashlib.sha256(f.read()).hexdigest()

            target = Path(settings.output_dir) / f"{basename}_audit.json"
            target.write_text(json.dumps(audit, indent=2, ensure_ascii=False),
                              encoding="utf-8")
            self._log(f"  wrote {target.name}")
        except Exception as exc:
            self._log(f"  [warn] could not write audit log: {exc}")

    def _prompt_password_if_needed(self, pdf_path: Path) -> Optional[str]:
        """Return the password to use for ``pdf_path`` (may be ''), or None
        if the user cancelled the prompt for an encrypted file."""
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return ""
        try:
            if not doc.needs_pass:
                return ""
        finally:
            doc.close()

        if not hasattr(self, "_password_cache"):
            self._password_cache: Dict[str, str] = {}
        key = str(pdf_path.resolve())
        if key in self._password_cache:
            return self._password_cache[key]

        # GUI prompt must run on the Tk thread.
        result_holder: List[Optional[str]] = [None]
        event = threading.Event()

        def ask() -> None:
            try:
                from tkinter import simpledialog
                pwd = simpledialog.askstring(
                    "Password required",
                    f"{pdf_path.name} is encrypted.\nEnter the password:",
                    show="*",
                    parent=self,
                )
                result_holder[0] = pwd
            finally:
                event.set()

        self.after(0, ask)
        event.wait()
        pwd = result_holder[0]
        if pwd is None:
            return None
        self._password_cache[key] = pwd
        return pwd

    # ---- file writers ----
    def _write_outputs(
        self, basename: str, outputs: Dict[str, Any], out_dir: str
    ) -> None:
        out = Path(out_dir)
        ext_map = {k: e for k, _l, e in OUTPUT_FORMATS}
        for fmt, content in outputs.items():
            target = out / f"{basename}{ext_map[fmt]}"
            if isinstance(content, (bytes, bytearray)):
                target.write_bytes(bytes(content))
            else:
                target.write_text(str(content), encoding="utf-8")
            self._log(f"  wrote {target.name} ({_human_size(target.stat().st_size)})")

    def _write_combined(
        self,
        results: List[Tuple[Path, Dict[str, Any]]],
        basename: str,
        out_dir: str,
        formats: List[str],
    ) -> None:
        out = Path(out_dir)
        ext_map = {k: e for k, _l, e in OUTPUT_FORMATS}
        for fmt in formats:
            target = out / f"{basename}{ext_map[fmt]}"
            if fmt == "searchable_pdf":
                merged = fitz.open()
                try:
                    for _path, results_dict in results:
                        blob = results_dict.get(fmt, b"")
                        if not blob:
                            continue
                        sub = fitz.open(stream=blob, filetype="pdf")
                        try:
                            merged.insert_pdf(sub)
                        finally:
                            sub.close()
                    target.write_bytes(merged.tobytes())
                finally:
                    merged.close()
            elif fmt == "json":
                combined = {
                    "files": [
                        {**json.loads(r[fmt])}
                        for _p, r in results if fmt in r
                    ]
                }
                target.write_text(
                    json.dumps(combined, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            elif fmt == "csv":
                # concat CSVs; keep one header
                lines: List[str] = []
                header_written = False
                for _p, r in results:
                    text = r.get(fmt, "")
                    if not text:
                        continue
                    sub_lines = text.splitlines()
                    if not sub_lines:
                        continue
                    if header_written:
                        lines.extend(sub_lines[1:])
                    else:
                        lines.extend(sub_lines)
                        header_written = True
                target.write_text("\n".join(lines) + "\n", encoding="utf-8")
            else:
                sep = "\n\n---\n\n"
                target.write_text(
                    sep.join(str(r[fmt]) for _p, r in results if fmt in r),
                    encoding="utf-8",
                )
            self._log(f"  wrote combined {target.name} ({_human_size(target.stat().st_size)})")

    def _open_in_explorer(self, path: str) -> None:
        if not path or not Path(path).exists():
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception:
            pass

    # ---- closing ----
    def _on_close(self) -> None:
        if self._worker and self._worker.is_alive():
            if not messagebox.askyesno(
                "Quit", "OCR is still running. Cancel and quit?"
            ):
                return
            self._cancel_event.set()
        self._save_config()
        self.destroy()


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Self-test — generate a known PDF and OCR it
# ---------------------------------------------------------------------------
_SELF_TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
    "The five boxing wizards jump quickly.",
    "Amazingly few discotheques provide jukeboxes.",
    "Bright vixens jump; dozy fowl quack.",
    "Crazy Fredrick bought many very exquisite opal jewels.",
]


def _self_test_run(lang: str) -> Dict[str, Any]:
    """Generate a small PDF with known text, OCR it, return accuracy metrics."""
    import time
    t0 = time.perf_counter()

    doc = fitz.open()
    try:
        page = doc.new_page(width=612, height=792)  # US letter, 72 dpi
        y = 72
        for line in _SELF_TEST_SENTENCES:
            page.insert_text((72, y), line, fontname="helv", fontsize=18)
            y += 36
        pdf_bytes = doc.tobytes()
    finally:
        doc.close()

    # OCR the rendered page
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()

    config = "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    ocr_text = pytesseract.image_to_string(img, lang=lang, config=config)
    elapsed = time.perf_counter() - t0

    # Word-level accuracy: count expected words that appear at least the
    # right number of times in the OCR output (case-insensitive).
    import re as _re
    expected = []
    for s in _SELF_TEST_SENTENCES:
        expected.extend(_re.findall(r"[A-Za-z']+", s))
    got = [w.lower() for w in _re.findall(r"[A-Za-z']+", ocr_text)]
    got_lower = list(got)
    matched = 0
    for w in expected:
        wl = w.lower()
        if wl in got_lower:
            matched += 1
            got_lower.remove(wl)
    accuracy = matched / max(1, len(expected))

    tess_version = ""
    try:
        proc = subprocess.run(
            [pytesseract.pytesseract.tesseract_cmd, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        first = (proc.stdout or proc.stderr).splitlines()[0:1]
        tess_version = first[0] if first else ""
    except Exception:
        pass

    return {
        "ok": True,
        "accuracy": accuracy,
        "expected_words": len(expected),
        "matched_words": matched,
        "pages": 1,
        "elapsed": elapsed,
        "tess_version": tess_version,
        "ocr_text": ocr_text,
    }


def main() -> int:
    # Required for PyInstaller-frozen .exe so child processes spawned by
    # ProcessPoolExecutor do not re-launch the GUI. Safe to call on all
    # platforms; no-op when not frozen.
    import multiprocessing as _mp
    _mp.freeze_support()

    app = OcrApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
