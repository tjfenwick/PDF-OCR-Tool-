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
from PIL import Image  # noqa: E402

try:
    import pytesseract  # noqa: E402
except Exception:
    pytesseract = None  # type: ignore[assignment]

from ocr_backends import (  # noqa: E402
    BACKEND_LABELS,
    BACKEND_NAMES,
    BackendMode,
    backend_opts_from_args,
    get_backend,
)

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
    if pytesseract is None:
        return (False, "pytesseract is not installed.")
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

    # Backend selection + per-backend config
    backend: str = "tesseract"
    paddle_use_gpu: bool = False
    qwen_base_url: str = "http://localhost:1234/v1"
    qwen_api_key: str = "lm-studio"
    qwen_model: str = ""
    marker_workers: int = 1

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


# Per-backend preset table. Keyed by (backend_name, preset_label).
# When the backend dropdown changes, the GUI snaps the preset selection to
# that backend's "Default" so stale knobs from another backend don't carry
# over (e.g. Tesseract's PSM=6 has no meaning for Marker).
PRESETS_BY_BACKEND: Dict[str, Dict[str, Dict[str, Any]]] = {
    "tesseract": {
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
    },
    "paddleocr": {
        # PaddleOCR's confidence is rescaled 0-1 → 0-100, but it tends to
        # produce higher per-word confidences than Tesseract, so we can
        # afford a tighter min-conf cutoff. No preprocessing is needed —
        # PaddleOCR has its own detection model.
        "Default": {
            "render_scale": 3.0,
            "contrast": 1.0,
            "sharpen": 0,
            "threshold": "none",
            "dilate": 0,
            "erode": 0,
            "min_conf": 50.0,
            "noise_conf": 60.0,
        },
        "CMM report tables": {
            "render_scale": 4.0,
            "contrast": 1.0,
            "sharpen": 0,
            "threshold": "none",
            "dilate": 0,
            "min_conf": 50.0,
            "noise_conf": 60.0,
            "row_tol": 12,
            "col_gap": 22,
        },
    },
    "marker": {
        # Marker does its own page handling and ignores preprocessing.
        "Default": {},
    },
    "qwen3vl": {
        # Qwen3-VL gets the raw rendered image. Higher render scale = more
        # legible images for the VLM but more tokens per page.
        "Default": {
            "render_scale": 3.0,
        },
    },
}

# Backwards-compatible alias used by the existing preset-load/save code.
PRESETS: Dict[str, Dict[str, Any]] = PRESETS_BY_BACKEND["tesseract"]


def settings_to_args(s: OcrSettings) -> argparse.Namespace:
    """Build an argparse.Namespace that matches what ``process_pdf`` expects."""
    return argparse.Namespace(
        pdfs=s.inputs,
        output="",
        pages=s.pages or None,
        ocr_backend=s.backend or "tesseract",
        paddle_use_gpu=s.paddle_use_gpu,
        marker_workers=s.marker_workers,
        qwen_base_url=s.qwen_base_url or "http://localhost:1234/v1",
        qwen_api_key=s.qwen_api_key or "lm-studio",
        qwen_model=s.qwen_model or "",
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


def process_pdf_multi(
    pdf_path: Path,
    settings: OcrSettings,
    log: Callable[[str], None],
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    args = settings_to_args(settings)
    if args.tesseract_path and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path

    backend = get_backend(args.ocr_backend)
    ok, msg = backend.available()
    if not ok:
        raise RuntimeError(
            f"OCR backend '{backend.name}' is not available: {msg}"
        )
    opts = backend_opts_from_args(args)
    word_level = backend.mode == BackendMode.WORD_LEVEL

    doc = fitz.open(pdf_path)
    try:
        page_indices = ocr_engine.parse_pages(args.pages, len(doc))
        if not page_indices:
            page_indices = list(range(len(doc)))
        crop = ocr_engine.parse_crop(args.crop)
        formats = set(settings.output_formats)

        # Document/page-level backends can't produce per-word boxes — silently
        # drop the formats that need them and warn.
        if not word_level:
            dropped = [f for f in ("json", "csv", "searchable_pdf") if f in formats]
            for f in dropped:
                formats.discard(f)
            if dropped:
                log(f"  [info] {backend.name} doesn't produce word boxes — "
                    f"skipping output format(s): {', '.join(dropped)}")

        md_chunks: List[str] = [f"# OCR Output: {pdf_path.name}\n"]
        md_chunks.append(
            "<!-- Settings: "
            f"backend={backend.name}, "
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

        json_doc: Dict[str, Any] = {
            "file": pdf_path.name,
            "total_pages": len(doc),
            "backend": backend.name,
            "settings": {
                k: v for k, v in vars(args).items() if k not in {"pdfs", "output"}
            },
            "pages": [],
        }
        csv_records: List[List[Any]] = [["page", "x", "y", "w", "h", "conf", "text"]]
        pdf_page_bytes: List[bytes] = []

        debug_dir = Path(args.debug_dir) if args.debug_dir else None
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

        # Document-level backends (Marker) emit the whole doc at once; cache
        # the result so the per-page loop just appends each chunk.
        doc_md_cache: Optional[Dict[int, str]] = None
        if backend.mode == BackendMode.DOCUMENT_LEVEL:
            log(f"  running {backend.name} on the full PDF...")
            doc_md_cache = backend.ocr_document_markdown(
                pdf_path, page_indices, opts
            )

        total = len(page_indices)
        for idx, page_index in enumerate(page_indices):
            if cancel_event.is_set():
                raise CancelledError()

            log(f"  page {page_index + 1}/{len(doc)} (rendering at {args.render_scale}x)")
            progress(idx, total)

            page = doc[page_index]
            rendered = ocr_engine.render_page(page, args.render_scale)
            rendered = ocr_engine.crop_image(rendered, crop)
            stem = f"{pdf_path.stem}_p{page_index + 1:02d}"

            if word_level:
                processed = ocr_engine.preprocess_image(
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
                if debug_dir:
                    rendered.save(debug_dir / f"{stem}_rendered.png")
                    processed.save(debug_dir / f"{stem}_processed.png")

                words = backend.ocr_page_words(processed, opts)

                # Searchable PDF: only available with Tesseract (uses its hOCR output).
                if "searchable_pdf" in formats and backend.name == "tesseract" and pytesseract is not None:
                    try:
                        pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                            rendered, lang=args.lang,
                            config=f"--oem {args.oem} --psm {args.psm}",
                            extension="pdf",
                        )
                        pdf_page_bytes.append(pdf_bytes)
                    except Exception as exc:
                        log(f"    [warn] searchable PDF page failed: {exc}")

                if "csv" in formats:
                    for w in words:
                        csv_records.append(
                            [page_index + 1, w.x, w.y, w.w, w.h, round(w.conf, 2), w.text]
                        )

                if "json" in formats:
                    json_doc["pages"].append({
                        "page": page_index + 1,
                        "words": [
                            {
                                "text": w.text, "conf": round(w.conf, 2),
                                "x": w.x, "y": w.y, "w": w.w, "h": w.h,
                            }
                            for w in words
                        ],
                    })

                if ocr_engine.detect_drawing_page(words):
                    md_chunks.append(f"\n## Page {page_index + 1}\n")
                    md_chunks.append(
                        "*This page appears to be a drawing/graphic with no "
                        "extractable table data.*\n"
                    )
                    if debug_dir:
                        ocr_engine.save_confidence_csv(
                            debug_dir / f"{stem}_confidence.csv", words
                        )
                    continue

                rows = ocr_engine.group_words_into_rows(words, y_tolerance=args.row_tol)
                page_md = ocr_engine.rows_to_markdown(
                    rows,
                    gap=args.col_gap,
                    noise_conf=args.noise_conf,
                    noise_alpha_ratio=args.noise_alpha_ratio,
                    include_confidence_comments=args.confidence_comments,
                )
                md_chunks.append(f"\n## Page {page_index + 1}\n")
                md_chunks.append(page_md)

                if args.save_raw and debug_dir and backend.name == "tesseract" and pytesseract is not None:
                    raw_text = pytesseract.image_to_string(
                        processed, lang=args.lang,
                        config=f"--oem {args.oem} --psm {args.psm} "
                               "-c preserve_interword_spaces=1",
                    )
                    (debug_dir / f"{stem}_raw.txt").write_text(raw_text, encoding="utf-8")

                if debug_dir:
                    ocr_engine.save_confidence_csv(
                        debug_dir / f"{stem}_confidence.csv", words
                    )

            else:
                # Page-level (Qwen) or document-level (Marker): markdown is
                # produced verbatim, no row clustering, no CMM normalization.
                if backend.mode == BackendMode.DOCUMENT_LEVEL:
                    page_md = (doc_md_cache or {}).get(page_index, "")
                else:
                    page_md = backend.ocr_page_markdown(rendered, opts)

                if debug_dir:
                    rendered.save(debug_dir / f"{stem}_rendered.png")

                if not page_md.strip():
                    md_chunks.append(f"\n## Page {page_index + 1}\n")
                    md_chunks.append(
                        "*This page appears to be a drawing/graphic with no "
                        "extractable table data.*\n"
                    )
                    continue

                md_chunks.append(f"\n## Page {page_index + 1}\n")
                md_chunks.append(page_md)

        progress(total, total)

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

        return outputs
    finally:
        doc.close()


def _merge_pdf_pages(pdf_page_bytes: List[bytes]) -> bytes:
    if not pdf_page_bytes:
        return b""
    out = fitz.open()
    try:
        for blob in pdf_page_bytes:
            sub = fitz.open(stream=blob, filetype="pdf")
            try:
                out.insert_pdf(sub)
            finally:
                sub.close()
        return out.tobytes()
    finally:
        out.close()


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

        self._build_menu()
        self._build_widgets()
        self._load_config()
        self._auto_detect_tesseract()
        self.after(100, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---- top menu ----
    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Add files...", command=self._add_files)
        file_menu.add_command(label="Add folder...", command=self._add_folder)
        file_menu.add_separator()
        file_menu.add_command(label="Save preset...", command=self._save_preset)
        file_menu.add_command(label="Load preset...", command=self._load_preset)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Tesseract status...", command=self._show_tesseract_status)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

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
        notebook.add(self._tab_files, text="Files & Output")
        notebook.add(self._tab_ocr, text="OCR")
        notebook.add(self._tab_prep, text="Preprocessing")
        notebook.add(self._tab_layout, text="Layout & Noise")
        notebook.add(self._tab_diag, text="Diagnostics")

        self._build_files_tab(self._tab_files)
        self._build_ocr_tab(self._tab_ocr)
        self._build_prep_tab(self._tab_prep)
        self._build_layout_tab(self._tab_layout)
        self._build_diag_tab(self._tab_diag)

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

        ttk.Label(parent, text="Input PDFs:").grid(row=0, column=0, sticky="w")

        list_frame = ttk.Frame(parent)
        list_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(2, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._files_list = tk.Listbox(list_frame, selectmode="extended", activestyle="dotbox")
        self._files_list.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(list_frame, command=self._files_list.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._files_list.configure(yscrollcommand=sb.set)

        side = ttk.Frame(list_frame)
        side.grid(row=0, column=2, sticky="ns", padx=(6, 0))
        ttk.Button(side, text="Add files...", command=self._add_files).pack(fill="x")
        ttk.Button(side, text="Add folder...", command=self._add_folder).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Remove", command=self._remove_selected).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Clear", command=self._clear_files).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="Move up", command=lambda: self._move_selected(-1)).pack(fill="x", pady=(12, 0))
        ttk.Button(side, text="Move down", command=lambda: self._move_selected(1)).pack(fill="x", pady=(4, 0))

        # output
        out_box = ttk.LabelFrame(parent, text="Output", padding=6)
        out_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        out_box.columnconfigure(1, weight=1)

        ttk.Label(out_box, text="Folder:").grid(row=0, column=0, sticky="w")
        self._out_dir_var = tk.StringVar()
        ttk.Entry(out_box, textvariable=self._out_dir_var).grid(row=0, column=1, sticky="ew", padx=(4, 4))
        ttk.Button(out_box, text="Browse...", command=self._pick_output_dir).grid(row=0, column=2)

        ttk.Label(out_box, text="Mode:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        mode_row = ttk.Frame(out_box)
        mode_row.grid(row=1, column=1, columnspan=2, sticky="w", pady=(4, 0))
        self._mode_var = tk.StringVar(value="per_file")
        ttk.Radiobutton(mode_row, text="One output per PDF", value="per_file", variable=self._mode_var
                        ).pack(side="left")
        ttk.Radiobutton(mode_row, text="Combined into one file", value="combined", variable=self._mode_var
                        ).pack(side="left", padx=(12, 0))

        ttk.Label(out_box, text="Combined name:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._combined_name_var = tk.StringVar(value="ocr_output")
        ttk.Entry(out_box, textvariable=self._combined_name_var).grid(
            row=2, column=1, columnspan=2, sticky="ew", pady=(4, 0), padx=(4, 0)
        )

        ttk.Label(out_box, text="Formats:").grid(row=3, column=0, sticky="nw", pady=(8, 0))
        fmt_frame = ttk.Frame(out_box)
        fmt_frame.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))
        self._fmt_vars: Dict[str, tk.BooleanVar] = {}
        for i, (key, label, _ext) in enumerate(OUTPUT_FORMATS):
            var = tk.BooleanVar(value=(key == "markdown"))
            self._fmt_vars[key] = var
            ttk.Checkbutton(fmt_frame, text=label, variable=var).grid(
                row=i // 3, column=i % 3, sticky="w", padx=(0, 12)
            )

        # finished options
        misc = ttk.Frame(out_box)
        misc.grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self._open_when_done_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(misc, text="Open output folder when done",
                        variable=self._open_when_done_var).pack(side="left")
        self._show_notify_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(misc, text="Show notification when done",
                        variable=self._show_notify_var).pack(side="left", padx=(12, 0))

    # ---- "OCR" tab ----
    def _build_ocr_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        # Backend dropdown — drives which rows below are visible.
        ttk.Label(parent, text="OCR Backend:").grid(row=0, column=0, sticky="w")
        self._backend_var = tk.StringVar(value="tesseract")
        backend_box = ttk.Combobox(
            parent, textvariable=self._backend_var,
            values=[BACKEND_LABELS[n] for n in BACKEND_NAMES],
            state="readonly",
        )
        backend_box.set(BACKEND_LABELS["tesseract"])
        backend_box.grid(row=0, column=1, sticky="ew", padx=4)
        backend_box.bind("<<ComboboxSelected>>", self._on_backend_changed)
        ttk.Label(parent, text="Select OCR engine").grid(row=0, column=2, sticky="w")

        # Per-backend group frames. Only the active backend's frame is gridded.
        self._tess_frame = ttk.Frame(parent)
        self._paddle_frame = ttk.Frame(parent)
        self._marker_frame = ttk.Frame(parent)
        self._qwen_frame = ttk.Frame(parent)
        for f in (self._tess_frame, self._paddle_frame,
                  self._marker_frame, self._qwen_frame):
            f.columnconfigure(1, weight=1)

        self._build_tesseract_group(self._tess_frame)
        self._build_paddle_group(self._paddle_frame)
        self._build_marker_group(self._marker_frame)
        self._build_qwen_group(self._qwen_frame)

        # Preset row — shared across backends but its value list is rebuilt
        # when the backend changes.
        ttk.Label(parent, text="Preset:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._preset_var = tk.StringVar(value="Default")
        self._preset_box = ttk.Combobox(
            parent, textvariable=self._preset_var,
            values=list(PRESETS_BY_BACKEND["tesseract"].keys()),
            state="readonly",
        )
        self._preset_box.grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        self._preset_box.bind("<<ComboboxSelected>>", self._on_apply_builtin_preset)
        ttk.Label(parent, text="Built-in starting points").grid(row=1, column=2, sticky="w", pady=(8, 0))

        # Page range — shared across all backends.
        ttk.Label(parent, text="Pages:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._pages_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self._pages_var).grid(
            row=2, column=1, sticky="ew", padx=4, pady=(4, 0)
        )
        ttk.Label(parent, text="e.g. 1,2,4-5 (blank = all)").grid(
            row=2, column=2, sticky="w", pady=(4, 0)
        )

        # Backend-specific frame goes in row 3 spanning all columns.
        self._backend_frame_row = 3

        # Apply initial backend visibility.
        self._apply_backend_visibility()

    def _build_tesseract_group(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Tesseract executable:").grid(row=0, column=0, sticky="w")
        self._tess_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self._tess_var).grid(row=0, column=1, sticky="ew", padx=4)
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=0, column=2, sticky="e")
        ttk.Button(btn_row, text="Browse...", command=self._pick_tesseract).pack(side="left")
        ttk.Button(btn_row, text="Auto-detect", command=self._auto_detect_tesseract).pack(side="left", padx=(4, 0))
        ttk.Button(btn_row, text="Test", command=self._show_tesseract_status).pack(side="left", padx=(4, 0))

        ttk.Label(parent, text="Language(s):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._lang_var = tk.StringVar(value="eng")
        ttk.Entry(parent, textvariable=self._lang_var).grid(row=1, column=1, sticky="ew", padx=4, pady=(8, 0))
        ttk.Label(parent, text='e.g. "eng" or "eng+deu"').grid(row=1, column=2, sticky="w", pady=(8, 0))

        ttk.Label(parent, text="Page segmentation (PSM):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self._psm_var = tk.IntVar(value=6)
        ttk.Combobox(parent, textvariable=self._psm_var,
                     values=[3, 4, 6, 11, 12], state="readonly", width=6).grid(
            row=2, column=1, sticky="w", padx=4, pady=(8, 0)
        )
        ttk.Label(parent, text="6=block, 4=column, 3=auto, 11/12=sparse").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )

        ttk.Label(parent, text="OCR engine (OEM):").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._oem_var = tk.IntVar(value=3)
        ttk.Combobox(parent, textvariable=self._oem_var,
                     values=[0, 1, 2, 3], state="readonly", width=6).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        ttk.Label(parent, text="3=default LSTM").grid(row=3, column=2, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="Render scale:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self._render_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(parent, from_=1.0, to=10.0, increment=0.5,
                    textvariable=self._render_var, width=8).grid(
            row=4, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        ttk.Label(parent, text="3-6 for small text").grid(row=4, column=2, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="Min word confidence:").grid(row=5, column=0, sticky="w", pady=(4, 0))
        self._min_conf_var = tk.DoubleVar(value=20.0)
        ttk.Spinbox(parent, from_=0.0, to=100.0, increment=1,
                    textvariable=self._min_conf_var, width=8).grid(
            row=5, column=1, sticky="w", padx=4, pady=(4, 0)
        )
        ttk.Label(parent, text="0-100").grid(row=5, column=2, sticky="w", pady=(4, 0))

    def _build_paddle_group(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="PaddleOCR runs detection + recognition with its own models.\n"
                 "Tesseract-style PSM/OEM are not used.",
            justify="left", foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(parent, text="Language:").grid(row=1, column=0, sticky="w")
        ttk.Entry(parent, textvariable=self._safe_var("_lang_var", "eng")).grid(
            row=1, column=1, sticky="ew", padx=4
        )
        ttk.Label(parent, text='eng, fra, ger, chi_sim, ...').grid(row=1, column=2, sticky="w")

        ttk.Label(parent, text="Render scale:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(parent, from_=1.0, to=10.0, increment=0.5,
                    textvariable=self._safe_var("_render_var", 3.0, kind="double"),
                    width=8).grid(row=2, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="3 is usually enough for Paddle").grid(
            row=2, column=2, sticky="w", pady=(6, 0)
        )

        ttk.Label(parent, text="Min word confidence:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        ttk.Spinbox(parent, from_=0.0, to=100.0, increment=1,
                    textvariable=self._safe_var("_min_conf_var", 50.0, kind="double"),
                    width=8).grid(row=3, column=1, sticky="w", padx=4, pady=(4, 0))
        ttk.Label(parent, text="0-100 (Paddle: try ~50)").grid(row=3, column=2, sticky="w", pady=(4, 0))

        self._paddle_gpu_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Use GPU (requires paddlepaddle-gpu)",
                        variable=self._paddle_gpu_var).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(6, 0)
        )

    def _build_marker_group(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Datalab Marker reads the PDF directly and produces clean\n"
                 "markdown for LLMs. Preprocessing knobs do not apply.\n"
                 "First run downloads ~2 GB of models (cached after that).",
            justify="left", foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(parent, text="Workers:").grid(row=1, column=0, sticky="w")
        self._marker_workers_var = tk.IntVar(value=1)
        ttk.Spinbox(parent, from_=1, to=8, increment=1,
                    textvariable=self._marker_workers_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4
        )
        ttk.Label(parent, text="Parallel page workers").grid(row=1, column=2, sticky="w")

    def _build_qwen_group(self, parent: ttk.Frame) -> None:
        ttk.Label(
            parent,
            text="Qwen3-VL via LM Studio. Start LM Studio, load a vision model,\n"
                 "and click 'Refresh models' to populate the dropdown.",
            justify="left", foreground="#555",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Label(parent, text="Base URL:").grid(row=1, column=0, sticky="w")
        self._qwen_url_var = tk.StringVar(value="http://localhost:1234/v1")
        ttk.Entry(parent, textvariable=self._qwen_url_var).grid(
            row=1, column=1, sticky="ew", padx=4
        )
        ttk.Label(parent, text="LM Studio default port").grid(row=1, column=2, sticky="w")

        ttk.Label(parent, text="API key:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._qwen_key_var = tk.StringVar(value="lm-studio")
        ttk.Entry(parent, textvariable=self._qwen_key_var, show="*").grid(
            row=2, column=1, sticky="ew", padx=4, pady=(4, 0)
        )
        ttk.Label(parent, text="LM Studio: anything works").grid(row=2, column=2, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="Model:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._qwen_model_var = tk.StringVar(value="")
        self._qwen_model_box = ttk.Combobox(
            parent, textvariable=self._qwen_model_var, values=[], state="normal",
        )
        self._qwen_model_box.grid(row=3, column=1, sticky="ew", padx=4, pady=(4, 0))
        qwen_btn_row = ttk.Frame(parent)
        qwen_btn_row.grid(row=3, column=2, sticky="w", pady=(4, 0))
        ttk.Button(qwen_btn_row, text="Refresh models",
                   command=self._refresh_qwen_models).pack(side="left")
        ttk.Button(qwen_btn_row, text="Test connection",
                   command=self._test_qwen_connection).pack(side="left", padx=(4, 0))

        ttk.Label(parent, text="Render scale:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(parent, from_=1.0, to=10.0, increment=0.5,
                    textvariable=self._safe_var("_render_var", 3.0, kind="double"),
                    width=8).grid(row=4, column=1, sticky="w", padx=4, pady=(6, 0))
        ttk.Label(parent, text="Higher = more legible but more tokens").grid(
            row=4, column=2, sticky="w", pady=(6, 0)
        )

    def _safe_var(self, attr: str, default, kind: str = "string"):
        """Return the existing tk var on the app instance, creating it if
        it doesn't exist yet. Lets multiple backend groups share the same
        underlying variable (e.g. render_scale, lang, min_conf)."""
        if hasattr(self, attr):
            return getattr(self, attr)
        if kind == "double":
            var = tk.DoubleVar(value=default)
        elif kind == "int":
            var = tk.IntVar(value=default)
        elif kind == "bool":
            var = tk.BooleanVar(value=bool(default))
        else:
            var = tk.StringVar(value=default)
        setattr(self, attr, var)
        return var

    def _label_to_backend(self, label: str) -> str:
        for name in BACKEND_NAMES:
            if BACKEND_LABELS[name] == label:
                return name
        return label  # if already a raw name

    def _backend_to_label(self, name: str) -> str:
        return BACKEND_LABELS.get(name, name)

    def _apply_backend_visibility(self) -> None:
        """Show only the active backend's group frame and refresh the preset list."""
        active = self._label_to_backend(self._backend_var.get())
        frames = {
            "tesseract": self._tess_frame,
            "paddleocr": self._paddle_frame,
            "marker":    self._marker_frame,
            "qwen3vl":   self._qwen_frame,
        }
        for name, frame in frames.items():
            if name == active:
                frame.grid(row=self._backend_frame_row, column=0, columnspan=3,
                           sticky="ew", pady=(8, 0))
            else:
                frame.grid_forget()

        # Refresh preset dropdown for this backend.
        presets = list(PRESETS_BY_BACKEND.get(active, {"Default": {}}).keys())
        self._preset_box.configure(values=presets)
        if self._preset_var.get() not in presets:
            self._preset_var.set(presets[0] if presets else "Default")

    def _on_backend_changed(self, _event: Any) -> None:
        active = self._label_to_backend(self._backend_var.get())
        self._apply_backend_visibility()
        # Snap to that backend's "Default" preset so stale knobs don't carry
        # over from the previously selected backend.
        presets = PRESETS_BY_BACKEND.get(active, {})
        if "Default" in presets:
            self._preset_var.set("Default")
            self._apply_preset_overrides(active, "Default")
        self._log(f"Switched OCR backend to: {active}")

    def _apply_preset_overrides(self, backend: str, preset_name: str) -> None:
        overrides = PRESETS_BY_BACKEND.get(backend, {}).get(preset_name, {})
        if not overrides:
            return
        base = OcrSettings()
        for k, v in overrides.items():
            setattr(base, k, v)
        self._apply_settings(base, only_processing=True)
        self._log(f"Applied preset: {backend} / {preset_name}")

    def _refresh_qwen_models(self) -> None:
        backend = get_backend("qwen3vl")
        models = backend.list_models(
            self._qwen_url_var.get().strip(),
            self._qwen_key_var.get().strip(),
        )
        self._qwen_model_box.configure(values=models)
        if models:
            if not self._qwen_model_var.get() or self._qwen_model_var.get() not in models:
                self._qwen_model_var.set(models[0])
            self._log(f"Found {len(models)} model(s) at {self._qwen_url_var.get()}")
        else:
            self._log(f"No models returned by {self._qwen_url_var.get()} — is LM Studio running?")

    def _test_qwen_connection(self) -> None:
        backend = get_backend("qwen3vl")
        ok, msg = backend.test_connection(
            self._qwen_url_var.get().strip(),
            self._qwen_key_var.get().strip(),
        )
        if ok:
            messagebox.showinfo("LM Studio connection", msg)
        else:
            messagebox.showerror("LM Studio connection failed", msg)

    # ---- "Preprocessing" tab ----
    def _build_prep_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        self._no_gray_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Keep RGB (skip grayscale conversion)",
                        variable=self._no_gray_var).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(parent, text="Contrast:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._contrast_var = tk.DoubleVar(value=1.4)
        ttk.Spinbox(parent, from_=0.5, to=4.0, increment=0.1,
                    textvariable=self._contrast_var, width=8).grid(
            row=1, column=1, sticky="w", padx=4, pady=(8, 0)
        )

        ttk.Label(parent, text="Sharpen passes:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._sharpen_var = tk.IntVar(value=1)
        ttk.Spinbox(parent, from_=0, to=5, increment=1,
                    textvariable=self._sharpen_var, width=8).grid(
            row=2, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        ttk.Label(parent, text="Threshold:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._thresh_var = tk.StringVar(value="adaptive")
        ttk.Combobox(parent, textvariable=self._thresh_var,
                     values=["none", "global", "adaptive"], state="readonly", width=10).grid(
            row=3, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        ttk.Label(parent, text="Threshold value:").grid(row=4, column=0, sticky="w", pady=(4, 0))
        self._thresh_val_var = tk.IntVar(value=180)
        ttk.Spinbox(parent, from_=0, to=255, increment=5,
                    textvariable=self._thresh_val_var, width=8).grid(
            row=4, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        self._invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Invert colours after preprocessing",
                        variable=self._invert_var).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        ttk.Label(parent, text="Dilate kernel:").grid(row=6, column=0, sticky="w", pady=(4, 0))
        self._dilate_var = tk.IntVar(value=1)
        ttk.Spinbox(parent, from_=0, to=10, increment=1,
                    textvariable=self._dilate_var, width=8).grid(
            row=6, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        ttk.Label(parent, text="Erode kernel:").grid(row=7, column=0, sticky="w", pady=(4, 0))
        self._erode_var = tk.IntVar(value=0)
        ttk.Spinbox(parent, from_=0, to=10, increment=1,
                    textvariable=self._erode_var, width=8).grid(
            row=7, column=1, sticky="w", padx=4, pady=(4, 0)
        )

        ttk.Label(parent, text="Crop (l,t,r,b px):").grid(row=8, column=0, sticky="w", pady=(4, 0))
        self._crop_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self._crop_var).grid(
            row=8, column=1, sticky="ew", padx=4, pady=(4, 0)
        )

    # ---- "Layout & Noise" tab ----
    def _build_layout_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Row tolerance (px):").grid(row=0, column=0, sticky="w")
        self._row_tol_var = tk.IntVar(value=12)
        ttk.Spinbox(parent, from_=1, to=80, increment=1,
                    textvariable=self._row_tol_var, width=8).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(parent, text="Column gap (px):").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self._col_gap_var = tk.IntVar(value=22)
        ttk.Spinbox(parent, from_=1, to=999, increment=1,
                    textvariable=self._col_gap_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=(4, 0))

        ttk.Label(parent, text="Row noise conf cutoff:").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self._noise_conf_var = tk.DoubleVar(value=55.0)
        ttk.Spinbox(parent, from_=0.0, to=100.0, increment=1,
                    textvariable=self._noise_conf_var, width=8).grid(row=2, column=1, sticky="w", padx=4, pady=(4, 0))

        ttk.Label(parent, text="Min alphanumeric ratio:").grid(row=3, column=0, sticky="w", pady=(4, 0))
        self._alpha_ratio_var = tk.DoubleVar(value=0.3)
        ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self._alpha_ratio_var, width=8).grid(row=3, column=1, sticky="w", padx=4, pady=(4, 0))

    # ---- "Diagnostics" tab ----
    def _build_diag_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Debug folder:").grid(row=0, column=0, sticky="w")
        self._debug_dir_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self._debug_dir_var).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(parent, text="Browse...", command=self._pick_debug_dir).grid(row=0, column=2)

        self._save_raw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Save raw Tesseract text into debug folder",
                        variable=self._save_raw_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))

        self._conf_comments_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Include per-row OCR confidence comments in Markdown",
                        variable=self._conf_comments_var).grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

    # ---- file/folder pickers ----
    def _add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Select PDFs",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        for p in paths:
            if p not in self._files_list.get(0, "end"):
                self._files_list.insert("end", p)
        self._maybe_default_output_dir()

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder of PDFs")
        if not folder:
            return
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
        backend = self._label_to_backend(self._backend_var.get())
        name = self._preset_var.get()
        self._apply_preset_overrides(backend, name)

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

    # ---- settings I/O ----
    def _collect_settings(self) -> OcrSettings:
        formats = [k for k, var in self._fmt_vars.items() if var.get()]
        return OcrSettings(
            inputs=list(self._files_list.get(0, "end")),
            output_dir=self._out_dir_var.get().strip(),
            output_formats=formats,
            output_mode=self._mode_var.get(),
            combined_basename=self._combined_name_var.get().strip() or "ocr_output",

            backend=self._label_to_backend(self._backend_var.get()),
            paddle_use_gpu=bool(self._paddle_gpu_var.get()),
            qwen_base_url=self._qwen_url_var.get().strip() or "http://localhost:1234/v1",
            qwen_api_key=self._qwen_key_var.get().strip() or "lm-studio",
            qwen_model=self._qwen_model_var.get().strip(),
            marker_workers=int(self._marker_workers_var.get()),

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

            backend_name = s.backend or "tesseract"
            self._backend_var.set(self._backend_to_label(backend_name))
            self._paddle_gpu_var.set(bool(s.paddle_use_gpu))
            self._qwen_url_var.set(s.qwen_base_url or "http://localhost:1234/v1")
            self._qwen_key_var.set(s.qwen_api_key or "lm-studio")
            self._qwen_model_var.set(s.qwen_model or "")
            self._marker_workers_var.set(int(s.marker_workers or 1))
            self._apply_backend_visibility()

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

    def _load_config(self) -> None:
        if not CONFIG_FILE.exists():
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

        backend_name = s.backend or "tesseract"
        if backend_name == "tesseract":
            ok, msg = tesseract_works(s.tesseract_path or None)
            if not ok:
                return (
                    "Tesseract OCR is not available.\n\n"
                    f"{msg}\n\n"
                    "Install it from https://github.com/UB-Mannheim/tesseract/wiki "
                    "and point the OCR tab to tesseract.exe."
                )
        else:
            backend = get_backend(backend_name)
            ok, msg = backend.available()
            if not ok:
                return f"OCR backend '{backend_name}' is not available.\n\n{msg}"
            if backend_name == "qwen3vl" and not (s.qwen_model or "").strip():
                return (
                    "Qwen3-VL needs a model id.\n\n"
                    "Load a vision model into LM Studio, then click "
                    "'Refresh models' on the OCR tab and pick one."
                )
        return None

    # ---- worker thread ----
    def _worker_main(self, settings: OcrSettings) -> None:
        try:
            Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
            if settings.debug_dir:
                Path(settings.debug_dir).mkdir(parents=True, exist_ok=True)

            per_file_results: List[Tuple[Path, Dict[str, Any]]] = []
            n_files = len(settings.inputs)

            for file_idx, pdf in enumerate(settings.inputs):
                if self._cancel_event.is_set():
                    raise CancelledError()
                pdf_path = Path(pdf)
                self._log(f"[{file_idx + 1}/{n_files}] {pdf_path.name}")
                self._set_status(f"{pdf_path.name} ({file_idx + 1}/{n_files})")

                results = process_pdf_multi(
                    pdf_path, settings,
                    log=self._log,
                    progress=self._set_progress,
                    cancel_event=self._cancel_event,
                )
                per_file_results.append((pdf_path, results))

                if settings.output_mode == "per_file":
                    self._write_outputs(
                        pdf_path.stem, results, settings.output_dir
                    )

            if settings.output_mode == "combined":
                self._write_combined(
                    per_file_results,
                    settings.combined_basename,
                    settings.output_dir,
                    settings.output_formats,
                )

            self._log_queue.put((
                "done", True,
                f"Processed {n_files} file(s).\nOutput folder:\n{settings.output_dir}",
            ))
        except CancelledError:
            self._log("Cancelled.")
            self._log_queue.put(("done", False, "Cancelled by user."))
        except Exception as exc:
            self._log("ERROR: " + str(exc))
            self._log(traceback.format_exc())
            self._log_queue.put(("done", False, f"{type(exc).__name__}: {exc}"))

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


def main() -> int:
    app = OcrApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
