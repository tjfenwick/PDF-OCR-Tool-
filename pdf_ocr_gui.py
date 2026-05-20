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


def process_pdf_multi(
    pdf_path: Path,
    settings: OcrSettings,
    log: Callable[[str], None],
    progress: Callable[[int, int], None],
    cancel_event: threading.Event,
) -> Dict[str, Any]:
    args = settings_to_args(settings)
    if args.tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_path

    doc = fitz.open(pdf_path)
    try:
        page_indices = ocr_engine.parse_pages(args.pages, len(doc))
        if not page_indices:
            page_indices = list(range(len(doc)))
        crop = ocr_engine.parse_crop(args.crop)
        formats = set(settings.output_formats)

        md_chunks: List[str] = [f"# OCR Output: {pdf_path.name}\n"]
        md_chunks.append(
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

        json_doc: Dict[str, Any] = {
            "file": pdf_path.name,
            "total_pages": len(doc),
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

        total = len(page_indices)
        for idx, page_index in enumerate(page_indices):
            if cancel_event.is_set():
                raise CancelledError()

            log(f"  page {page_index + 1}/{len(doc)} (rendering at {args.render_scale}x)")
            progress(idx, total)

            page = doc[page_index]
            rendered = ocr_engine.render_page(page, args.render_scale)
            rendered = ocr_engine.crop_image(rendered, crop)
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

            stem = f"{pdf_path.stem}_p{page_index + 1:02d}"
            if debug_dir:
                rendered.save(debug_dir / f"{stem}_rendered.png")
                processed.save(debug_dir / f"{stem}_processed.png")

            words = ocr_engine.tesseract_words(
                processed, args.lang, args.psm, args.oem, args.min_conf
            )

            # Searchable PDF: bake OCR text layer onto the original rendered image
            if "searchable_pdf" in formats:
                try:
                    pdf_bytes = pytesseract.image_to_pdf_or_hocr(
                        rendered, lang=args.lang,
                        config=f"--oem {args.oem} --psm {args.psm}",
                        extension="pdf",
                    )
                    pdf_page_bytes.append(pdf_bytes)
                except Exception as exc:
                    log(f"    [warn] searchable PDF page failed: {exc}")

            # Per-page CSV records
            if "csv" in formats:
                for w in words:
                    csv_records.append(
                        [page_index + 1, w.x, w.y, w.w, w.h, round(w.conf, 2), w.text]
                    )

            # Per-page JSON records
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

            # Drawing page short-circuit
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

            if args.save_raw and debug_dir:
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
        help_menu.add_command(label="Settings reference...", command=self._show_settings_reference)
        help_menu.add_command(label="Tesseract status...", command=self._show_tesseract_status)
        help_menu.add_separator()
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
