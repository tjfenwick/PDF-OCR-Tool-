# PyInstaller spec for PDF OCR Tool
#
# Build:
#   pyinstaller pdf_ocr_gui.spec
#
# The build script (build_exe.bat / build-exe.yml) populates ``tesseract/`` next
# to this file before invoking PyInstaller. If that folder is present it is
# bundled into the .exe distribution so end users do not have to install
# Tesseract themselves.

from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / 'OCR_PDF_to_Markdown.py'), '.'),
]

tess_dir = ROOT / 'tesseract'
if tess_dir.exists():
    datas.append((str(tess_dir), 'tesseract'))

a = Analysis(
    ['pdf_ocr_gui.py'],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'OCR_PDF_to_Markdown',
        'PIL._tkinter_finder',
        'cv2',
        'numpy',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        'matplotlib', 'scipy', 'pandas',
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
