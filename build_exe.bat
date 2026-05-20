@echo off
REM ===========================================================================
REM Build the PDF OCR Tool as a portable Windows .exe (no admin required).
REM
REM Usage:
REM     build_exe.bat
REM
REM Optional: drop a portable Tesseract install in a ``tesseract\`` folder next
REM to this script before running. If present, it will be bundled into the .exe
REM distribution and the GUI will auto-detect it at runtime, so end users won't
REM need to install Tesseract separately.
REM ===========================================================================

setlocal

where py >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python launcher "py" not found. Install Python 3.10+ from python.org.
    exit /b 1
)

echo --- creating virtual environment ---
py -3 -m venv .venv-build
call .venv-build\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo Could not activate venv.
    exit /b 1
)

echo --- installing build dependencies ---
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo --- cleaning previous build ---
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo --- running PyInstaller ---
pyinstaller pdf_ocr_gui.spec
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

echo.
echo Build complete.
echo Output: dist\PDF-OCR-Tool\PDF-OCR-Tool.exe
echo Zip the dist\PDF-OCR-Tool folder and ship it.
endlocal
