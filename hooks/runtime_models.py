"""PyInstaller runtime hook: point PaddleOCR / HuggingFace at bundled model
caches so the frozen .exe doesn't try to download models on first launch.

Activated via `runtime_hooks` in pdf_ocr_gui.spec. Only runs inside the
frozen build — when running from source this file is not imported.
"""

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))

    paddle_dir = base / "paddleocr_models"
    if paddle_dir.exists():
        # PaddleOCR honors a handful of cache-location env vars depending on
        # version. Set them all so the install location overrides the
        # default Path.home()/.paddleocr.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(paddle_dir))
        os.environ.setdefault("PADDLEOCR_HOME", str(paddle_dir))
        os.environ.setdefault("PADDLEX_HOME", str(paddle_dir))

    marker_dir = base / "marker_models"
    if marker_dir.exists():
        os.environ.setdefault("HF_HOME", str(marker_dir))
        os.environ.setdefault(
            "TRANSFORMERS_CACHE", str(marker_dir / "transformers")
        )
        os.environ.setdefault(
            "HUGGINGFACE_HUB_CACHE", str(marker_dir / "hub")
        )
