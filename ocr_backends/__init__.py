"""Pluggable OCR backends.

Each backend implements a small protocol (mode + `available()` + one of
`ocr_page_words` / `ocr_page_markdown` / `ocr_document_markdown`). Backends
are looked up by name from the CLI (`--ocr-backend`) and the GUI dropdown.

Heavy third-party deps (paddleocr, marker-pdf, openai) are imported lazily
inside each backend so that selecting one backend never forces the user
to install the others.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


class BackendMode(str, Enum):
    WORD_LEVEL = "word_level"          # produces per-word OCRWord boxes (Tesseract, PaddleOCR)
    PAGE_LEVEL = "page_level"          # produces markdown from a rendered page image (Qwen3-VL)
    DOCUMENT_LEVEL = "document_level"  # produces {page_index: markdown} from a whole PDF (Marker)


@runtime_checkable
class OcrBackend(Protocol):
    name: str
    mode: BackendMode

    def available(self) -> Tuple[bool, str]:
        """Return (True, version_info) if usable, (False, install_hint) otherwise."""
        ...

    # Word-level backends implement this:
    def ocr_page_words(self, img, opts: Dict[str, Any]) -> list: ...

    # Page-level backends implement this:
    def ocr_page_markdown(self, img, opts: Dict[str, Any]) -> str: ...

    # Document-level backends implement this:
    def ocr_document_markdown(
        self, pdf_path, page_indices: List[int], opts: Dict[str, Any]
    ) -> Dict[int, str]: ...


BACKEND_NAMES: List[str] = ["tesseract", "paddleocr", "marker", "qwen3vl"]

BACKEND_LABELS: Dict[str, str] = {
    "tesseract": "Tesseract (default, bundled)",
    "paddleocr": "PaddleOCR (better tables/dense text)",
    "marker":    "Datalab Marker (PDF → markdown for LLMs)",
    "qwen3vl":   "Qwen3-VL via LM Studio (local VLM)",
}

_INSTANCES: Dict[str, OcrBackend] = {}


def get_backend(name: str) -> OcrBackend:
    """Return a cached backend instance by name. Constructs lazily on first call."""
    key = (name or "tesseract").lower()
    if key not in BACKEND_NAMES:
        raise KeyError(
            f"Unknown OCR backend: {name!r}. Available: {BACKEND_NAMES}"
        )
    if key in _INSTANCES:
        return _INSTANCES[key]

    if key == "tesseract":
        from .tesseract_backend import TesseractBackend
        instance = TesseractBackend()
    elif key == "paddleocr":
        from .paddleocr_backend import PaddleOCRBackend
        instance = PaddleOCRBackend()
    elif key == "marker":
        from .marker_backend import MarkerBackend
        instance = MarkerBackend()
    elif key == "qwen3vl":
        from .qwen_backend import Qwen3VLBackend
        instance = Qwen3VLBackend()
    else:  # pragma: no cover
        raise KeyError(key)

    _INSTANCES[key] = instance
    return instance


def backend_opts_from_args(args) -> Dict[str, Any]:
    """Build a backend-options dict from a parsed argparse Namespace.

    Backends pull what they need by key and ignore the rest, so a single
    dict can serve all four engines.
    """
    return {
        # Word-level shared
        "lang": getattr(args, "lang", "eng"),
        "psm": getattr(args, "psm", 6),
        "oem": getattr(args, "oem", 3),
        "min_conf": getattr(args, "min_conf", 20.0),
        # PaddleOCR
        "use_gpu": bool(getattr(args, "paddle_use_gpu", False)),
        # Marker
        "marker_workers": int(getattr(args, "marker_workers", 1) or 1),
        "render_scale": float(getattr(args, "render_scale", 5.0)),
        # Qwen3-VL via LM Studio
        "base_url": getattr(args, "qwen_base_url", "http://localhost:1234/v1"),
        "api_key": getattr(args, "qwen_api_key", "lm-studio") or "lm-studio",
        "model": getattr(args, "qwen_model", "") or "",
    }
