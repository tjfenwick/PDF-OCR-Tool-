# PDF OCR Tool

OCR PDFs to Markdown, plain text, JSON, CSV, HTML, or a searchable PDF, with
a Tkinter GUI on top of the `OCR_PDF_to_Markdown.py` engine.

Originally built around Tesseract for CMM (coordinate measuring machine)
inspection reports. Now ships with four pluggable OCR backends so you can
pick the best engine for the job — including modern table/markdown
extractors aimed at feeding LLMs.

## OCR backends

| Backend | Mode | Best for | Notes |
|---------|------|----------|-------|
| **Tesseract** *(default)* | Word-level | Familiar tuning, searchable PDF output, offline use | Bundled with the portable .exe. CMM-specific normalization rules apply. |
| **PaddleOCR** | Word-level | Stronger baseline accuracy on dense tables and small text | First run downloads a few hundred MB of models. Optional GPU via paddlepaddle-gpu. |
| **Datalab Marker** | Document-level | Cleanest "PDF → markdown for LLMs" output | Operates on the raw PDF (skips preprocessing). First run downloads ~2 GB of models. CMM normalization is bypassed. |
| **Qwen3-VL via LM Studio** | Page-level | Local VLM extraction; great on layouts Tesseract mangles | Requires a running [LM Studio](https://lmstudio.ai) with a Qwen3-VL (or other vision) model loaded. CMM normalization is bypassed. |

JSON, CSV, and Searchable-PDF output formats need per-word bounding boxes,
which only the word-level backends produce. When you select Marker or
Qwen3-VL, those formats are skipped (Markdown / text / HTML still work).

## Features

- **Drag-free GUI** with five tabs: Files & Output, OCR, Preprocessing,
  Layout & Noise, Diagnostics.
- **Batch input** — add files, add a whole folder (recursive), reorder, remove.
- **Multiple output formats per run**
  - Markdown (`.md`) — preserves headings and reconstructed tables
  - Plain text (`.txt`)
  - JSON (`.json`) — per-word bounding boxes and confidences *(word-level backends only)*
  - CSV (`.csv`) — flat word-level data *(word-level backends only)*
  - HTML (`.html`) — self-contained, styled
  - Searchable PDF (`.pdf`) — original image with an OCR text layer *(Tesseract only)*
- **One file per PDF** or **one combined file** per format.
- **Per-backend presets** — switching backends auto-snaps to a default
  preset tuned for that engine, so stale Tesseract knobs don't carry over
  to PaddleOCR / Marker / Qwen.
- **Preprocessing controls** — grayscale, contrast, sharpen, threshold
  (none / global / adaptive), threshold value, invert, dilate/erode, crop.
  Only meaningful for word-level backends.
- **Layout & noise filtering** — row tolerance, column gap, row-noise
  confidence cutoff, min alphanumeric ratio. Word-level backends only.
- **Diagnostics** — save rendered/processed PNGs, raw Tesseract text, per-row
  confidence comments.
- **Auto-detects Tesseract** on PATH, common install paths, and a sibling
  `tesseract/` folder (used by the portable .exe build).
- **Threaded processing** with a live log, per-page progress, and Cancel.
- Settings persist between sessions in `~/.pdf_ocr_gui_config.json`.

## Running from Python

```bash
pip install -r requirements.txt          # baseline (Tesseract only)
pip install -r requirements-backends.txt  # optional: PaddleOCR, Marker, Qwen3-VL client
python pdf_ocr_gui.py
```

CLI:

```bash
# Default Tesseract:
python OCR_PDF_to_Markdown.py input.pdf -o out.md

# PaddleOCR:
python OCR_PDF_to_Markdown.py input.pdf -o out.md --ocr-backend paddleocr

# Datalab Marker (best for LLM consumption):
python OCR_PDF_to_Markdown.py input.pdf -o out.md --ocr-backend marker

# Qwen3-VL via local LM Studio:
python OCR_PDF_to_Markdown.py input.pdf -o out.md \
    --ocr-backend qwen3vl --qwen-model qwen2-vl-7b-instruct
```

Tesseract is a system dependency: <https://github.com/UB-Mannheim/tesseract/wiki>.

## Qwen3-VL setup (LM Studio)

1. Install [LM Studio](https://lmstudio.ai).
2. Download a vision-capable model (Qwen2-VL or Qwen3-VL family) inside LM Studio.
3. Start LM Studio's local server (defaults to `http://localhost:1234/v1`).
4. In the GUI's OCR tab, pick **Qwen3-VL via LM Studio** as the backend,
   click **Refresh models** to populate the model dropdown, and pick the
   loaded model. **Test connection** confirms LM Studio is reachable.

The base URL is editable, so you can also point at DashScope, OpenRouter,
or a vLLM server by overriding it.

## Portable Windows .exe

The bundled .exe ships **all four backends** including their model files,
so end users do not need to `pip install` anything (LM Studio is still a
separate install if they want to use Qwen3-VL).

Expect the distribution to weigh in around 3-5 GB because of bundled
PaddleOCR / Marker model weights. First launch may take 20-60s while
PyTorch warms up.

### Option A — GitHub Actions (recommended)

Every push to `main` and the feature branch triggers
`.github/workflows/build-exe.yml`, which on a Windows runner:

1. Sets up Python 3.11 and installs all build deps (`requirements-build.txt`).
2. Installs Tesseract via Chocolatey and copies it into `tesseract/`.
3. Runs `prefetch_models.py` to pre-download PaddleOCR + Marker models
   into `paddleocr_models/` and `marker_models/` next to the spec.
4. Runs PyInstaller against `pdf_ocr_gui.spec` to produce a one-folder
   distribution.
5. Zips and uploads `PDF-OCR-Tool-windows-x64.zip` as a workflow artifact.

Download the zip from the **Actions** tab → latest run → **Artifacts**,
unzip anywhere on Windows, and run `PDF-OCR-Tool.exe`.

### Option B — Build locally on Windows

```cmd
build_exe.bat
```

Drop a `Tesseract-OCR` install into `tesseract\` before running if you want
it bundled. PaddleOCR / Marker models will be downloaded on first launch
of the GUI in this case (the prefetch step is only run automatically by CI).

## Layout

```
pdf_ocr_gui.py             # Tkinter GUI, multi-format processor
OCR_PDF_to_Markdown.py     # OCR engine + CLI (backend dispatch lives here)
ocr_backends/              # Pluggable OCR backends
  __init__.py                # registry + protocol
  tesseract_backend.py
  paddleocr_backend.py
  marker_backend.py
  qwen_backend.py
prefetch_models.py         # Pre-download model caches for the .exe build
hooks/runtime_models.py    # Frozen-app runtime hook: point caches at bundled dirs
requirements.txt           # Baseline deps (Tesseract path)
requirements-backends.txt  # Optional: PaddleOCR + Marker + Qwen3-VL client
requirements-build.txt     # Adds PyInstaller and pulls everything for the .exe
pdf_ocr_gui.spec           # PyInstaller spec
build_exe.bat              # Local Windows build script
.github/workflows/
  build-exe.yml            # CI build that produces the portable .exe
```

## Troubleshooting

- **"Tesseract OCR is not available."** — only matters when the Tesseract
  backend is selected. Click *Browse...* on the OCR tab and point it at
  `tesseract.exe`. Or switch to another backend.
- **PaddleOCR/Marker first run is slow** — models are downloading. After
  the first run they're cached and startup is fast.
- **Qwen3-VL: "Connected but no models loaded"** — start LM Studio and
  load a vision model before running OCR.
- **Qwen3-VL hallucinates numbers** — VLMs occasionally invent
  plausible-looking values. For critical CMM measurements, spot-check
  Qwen3-VL output against Tesseract or PaddleOCR on the same page.
- **Small text not picked up (Tesseract)** — raise *Render scale* (try 6)
  and *Contrast* (try 1.6) on the Preprocessing tab.
- **Tables collapse into one column (Tesseract / Paddle)** — lower
  *Column gap* on the Layout tab; or raise it if cells merge together.
- **Lots of garbage lines (Tesseract / Paddle)** — raise *Row noise conf
  cutoff* and *Min alphanumeric ratio* on the Layout tab.
- **Searchable PDF text is offset / blurry** — Tesseract overlays text on
  the rendered image, not on the original PDF page. Higher *Render scale*
  improves alignment at the cost of file size.
