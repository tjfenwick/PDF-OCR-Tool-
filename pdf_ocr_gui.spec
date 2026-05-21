# PyInstaller spec for PDF OCR Tool
#
# Build:
#   pyinstaller pdf_ocr_gui.spec
#
# Before building, the CI workflow (or local build script) populates these
# directories next to the spec so PyInstaller can embed them:
#   tesseract/          - portable Tesseract install (Windows)
#   paddleocr_models/   - PaddleOCR detection/recognition model cache
#   marker_models/      - HuggingFace cache for Marker / Surya models
#
# All three are optional. Missing directories are silently skipped — the
# build still completes, just without that backend's models bundled.

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
)

block_cipher = None

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / 'OCR_PDF_to_Markdown.py'), '.'),
    (str(ROOT / 'ocr_backends'), 'ocr_backends'),
]

tess_dir = ROOT / 'tesseract'
if tess_dir.exists():
    datas.append((str(tess_dir), 'tesseract'))

paddle_models_dir = ROOT / 'paddleocr_models'
if paddle_models_dir.exists():
    datas.append((str(paddle_models_dir), 'paddleocr_models'))

marker_models_dir = ROOT / 'marker_models'
if marker_models_dir.exists():
    datas.append((str(marker_models_dir), 'marker_models'))

# Best-effort: include the package data files for the heavy backends so
# PyInstaller doesn't drop dictionaries, fonts, configs, etc.
for pkg in ('paddleocr', 'paddle', 'marker', 'surya', 'openai',
            'transformers', 'tokenizers', 'safetensors'):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

hiddenimports: list = [
    'OCR_PDF_to_Markdown',
    'PIL._tkinter_finder',
    'cv2',
    'numpy',
    # OCR backends — keep these importable even when PyInstaller can't trace
    # them statically (they're loaded via string lookup in ocr_backends.get_backend).
    'ocr_backends',
    'ocr_backends.tesseract_backend',
    'ocr_backends.paddleocr_backend',
    'ocr_backends.marker_backend',
    'ocr_backends.qwen_backend',
    'openai',
]

# Pick up submodules of the heavy ML packages PyInstaller often misses.
for pkg in ('paddleocr', 'paddle', 'marker', 'surya', 'transformers'):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

a = Analysis(
    ['pdf_ocr_gui.py'],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[str(ROOT / 'hooks' / 'runtime_models.py')],
    excludes=[
        'matplotlib',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
        'IPython', 'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PDF-OCR-Tool',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PDF-OCR-Tool',
)
