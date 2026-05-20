# PDF OCR Tool

OCR PDFs to Markdown, plain text, JSON, CSV, HTML, or a searchable PDF, with
a Tkinter GUI on top of the original `OCR_PDF_to_Markdown.py` engine.

The tool is built around Tesseract OCR, PyMuPDF for page rendering, and OpenCV
for image preprocessing.

## Features

- **Drag-free GUI** with five tabs: Files & Output, OCR, Preprocessing,
  Layout & Noise, Diagnostics.
- **Batch input** – add files, add a whole folder (recursive), reorder, remove.
- **Multiple output formats per run**
  - Markdown (`.md`) – preserves headings and reconstructed tables
  - Plain text (`.txt`)
  - JSON (`.json`) – per-word bounding boxes and confidences
  - CSV (`.csv`) – flat word-level data
  - HTML (`.html`) – self-contained, styled
  - Searchable PDF (`.pdf`) – original image with an OCR text layer
- **One file per PDF** or **one combined file** per format.
- **OCR settings** – language, PSM, OEM, render scale, min-word confidence,
  page range (e.g. `1,2,4-5`).
- **Preprocessing** – grayscale, contrast, sharpen, threshold (none / global /
  adaptive), threshold value, invert, dilate/erode, crop.
- **Layout & noise filtering** – row tolerance, column gap, row-noise
  confidence cutoff, min alphanumeric ratio.
- **Diagnostics** – save rendered/processed PNGs, raw Tesseract text, per-row
  confidence comments.
- **Presets** – built-in (Default, Fast, High quality, Plain document, CMM
  tables) plus save/load custom presets as JSON.
- **Auto-detects Tesseract** on PATH, common install paths, and a sibling
  `tesseract/` folder (used by the portable .exe build).
- **Threaded processing** with a live log, per-page progress, and Cancel.
- Settings persist between sessions in `~/.pdf_ocr_gui_config.json`.

## Running from Python

```bash
pip install -r requirements.txt
python pdf_ocr_gui.py
```

The original CLI is still available:

```bash
python OCR_PDF_to_Markdown.py input.pdf -o out.md
```

Tesseract OCR is a system dependency: <https://github.com/UB-Mannheim/tesseract/wiki>.

## Portable Windows .exe

There are two ways to get a no-install .exe:

### Option A – GitHub Actions (recommended)

Every push to `main` and the feature branch triggers
`.github/workflows/build-exe.yml`, which on a Windows runner:

1. Sets up Python 3.11 and installs build deps.
2. Installs Tesseract via Chocolatey and copies it into `tesseract/` next to
   the spec so it gets bundled into the dist.
3. Runs PyInstaller against `pdf_ocr_gui.spec` to produce a one-folder
   distribution.
4. Zips the folder as `PDF-OCR-Tool-windows-x64.zip` and uploads it as a
   workflow artifact (also attached to GitHub Releases when you publish one).

Download the zip from the **Actions** tab → latest run → **Artifacts**,
unzip anywhere on Windows, and run `PDF-OCR-Tool.exe`. No admin and no
separate Tesseract install required.

### First run on Windows — getting past SmartScreen

Because the .exe is downloaded from the internet and not signed with a
purchased code-signing certificate, the first launch on a fresh machine
shows **"Windows protected your PC"** (Microsoft Defender SmartScreen).
None of the fixes below require admin:

- **Recommended — unblock the zip *before* extracting.**
  Right-click `PDF-OCR-Tool-windows-x64.zip` → **Properties** → tick
  **Unblock** → **OK**. Then extract and run. This strips the
  Mark-of-the-Web from every file inside, so SmartScreen never fires.
- **Or, on the SmartScreen dialog:** click the small **More info** link,
  then the **Run anyway** button that appears. Per-user choice, no admin.
- **Or, in PowerShell** after extraction (no admin):
  ```powershell
  Get-ChildItem -Recurse "C:\path\to\PDF-OCR-Tool" | Unblock-File
  ```

The build embeds Windows version info, so the dialog identifies the file
as **"PDF OCR Tool"** published by **tjfenwick** rather than "Unknown".

To verify the download is intact, check the SHA-256 against the published
`PDF-OCR-Tool-windows-x64.zip.sha256` file:

```powershell
Get-FileHash PDF-OCR-Tool-windows-x64.zip -Algorithm SHA256
```

### Option B – Build locally on Windows

```cmd
build_exe.bat
```

This creates a venv, installs the build deps, and runs PyInstaller. The
output lands in `dist\PDF-OCR-Tool\`.

To bundle a portable Tesseract with your local build, drop a `Tesseract-OCR`
install directory into `tesseract\` next to the spec file before running
`build_exe.bat`. The spec picks it up automatically and the GUI auto-detects
it at runtime.

## Layout

```
pdf_ocr_gui.py            # Tkinter GUI, multi-format processor
OCR_PDF_to_Markdown.py    # original OCR engine (CLI still works)
requirements.txt          # runtime dependencies
requirements-build.txt    # adds PyInstaller for the .exe build
pdf_ocr_gui.spec          # PyInstaller spec
build_exe.bat             # local Windows build script
.github/workflows/
  build-exe.yml           # CI build that produces the portable .exe
```

## Troubleshooting

- **"Tesseract OCR is not available."** – click *Browse...* on the OCR tab
  and point it at `tesseract.exe`. The *Test* button confirms the binary
  works.
- **Small text not picked up** – raise *Render scale* (try 6) and *Contrast*
  (try 1.6) on the Preprocessing tab.
- **Tables collapse into one column** – lower *Column gap* on the Layout
  tab; or raise it if cells merge together.
- **Lots of garbage lines** – raise *Row noise conf cutoff* and
  *Min alphanumeric ratio* on the Layout tab.
- **Searchable PDF text is offset / blurry** – Tesseract overlays text on
  the rendered image, not on the original PDF page. Higher *Render scale*
  improves alignment at the cost of file size.
