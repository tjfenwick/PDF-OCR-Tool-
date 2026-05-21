"""Tesseract backend — thin wrapper around the existing tesseract_words().

Default backend. No new dependencies — exists only to fit the new pluggable
interface so all four backends share a uniform call site.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import BackendMode


class TesseractBackend:
    name = "tesseract"
    mode = BackendMode.WORD_LEVEL

    def available(self) -> Tuple[bool, str]:
        try:
            import pytesseract  # noqa: F401
        except Exception as exc:
            return (False, f"pytesseract not installed: {exc}")
        try:
            import pytesseract
            version = pytesseract.get_tesseract_version()
            return (True, f"tesseract {version}")
        except Exception as exc:
            return (
                False,
                f"Tesseract binary not found or not runnable: {exc}. "
                f"Install from https://github.com/UB-Mannheim/tesseract/wiki",
            )

    def ocr_page_words(self, img, opts: Dict[str, Any]) -> List:
        # Lazy import to avoid circular dependency at module load.
        from OCR_PDF_to_Markdown import tesseract_words
        return tesseract_words(
            img,
            opts.get("lang", "eng"),
            int(opts.get("psm", 6)),
            int(opts.get("oem", 3)),
            float(opts.get("min_conf", 20.0)),
        )
