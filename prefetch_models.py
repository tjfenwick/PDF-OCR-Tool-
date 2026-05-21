#!/usr/bin/env python3
"""Pre-fetch OCR model caches so the PyInstaller .exe can embed them.

Run on the CI Windows runner BEFORE `pyinstaller pdf_ocr_gui.spec`. It
triggers PaddleOCR and Marker to download their model files to their
default caches, then copies those caches next to the .spec file so the
spec's `datas` rules can include them in the .exe distribution.

This is best-effort: if a download fails (network glitch, package version
mismatch), the script logs a warning and continues so the build can still
produce a Tesseract-only .exe.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def prefetch_paddle() -> None:
    print("[prefetch] PaddleOCR models...")
    try:
        from paddleocr import PaddleOCR
        _ = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    except Exception as exc:
        print(f"[prefetch] WARN: PaddleOCR init failed: {exc}")
        return
    # PaddleOCR caches to ~/.paddleocr by default.
    candidates = [
        Path.home() / ".paddleocr",
        Path.home() / ".paddlex",
        Path(os.environ.get("PADDLE_PDX_CACHE_HOME", "")) if os.environ.get("PADDLE_PDX_CACHE_HOME") else None,
    ]
    for src in candidates:
        if src and src.exists():
            dst = HERE / "paddleocr_models"
            _copy_tree(src, dst)
            print(f"[prefetch]   copied {src} -> {dst}")
            return
    print("[prefetch] WARN: no PaddleOCR cache directory found.")


def prefetch_marker() -> None:
    print("[prefetch] Marker / Surya models...")
    try:
        from marker.models import create_model_dict
        _ = create_model_dict()
    except Exception as exc:
        print(f"[prefetch] WARN: Marker init failed: {exc}")
        return
    src = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    if not src.exists():
        print(f"[prefetch] WARN: HuggingFace cache not found at {src}")
        return
    dst = HERE / "marker_models"
    _copy_tree(src, dst)
    print(f"[prefetch]   copied {src} -> {dst}")


def main() -> int:
    prefetch_paddle()
    prefetch_marker()
    return 0


if __name__ == "__main__":
    sys.exit(main())
