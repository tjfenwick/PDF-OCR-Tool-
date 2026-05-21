"""PaddleOCR backend.

Word-level: produces the same OCRWord objects as Tesseract, so the existing
row/column clustering, noise filtering, and CMM normalization pipeline
runs unchanged downstream.

Confidence is rescaled from PaddleOCR's 0-1 range to Tesseract's 0-100
range so existing `--min-conf` / `--noise-conf` thresholds remain valid.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import BackendMode

# Tesseract-style language codes → PaddleOCR codes.
# Falls back to the input string for codes we don't know — PaddleOCR will
# error with a useful message if it's truly unsupported.
_LANG_MAP: Dict[str, str] = {
    "eng": "en",
    "en": "en",
    "fra": "fr", "fre": "fr",
    "deu": "german", "ger": "german",
    "spa": "es",
    "ita": "it",
    "por": "pt",
    "rus": "ru",
    "jpn": "japan",
    "kor": "korean",
    "chi_sim": "ch",
    "chi_tra": "chinese_cht",
}


def _translate_lang(lang: str) -> str:
    # Multi-language inputs like "eng+deu" — Paddle takes one language at a
    # time. Take the first token.
    primary = (lang or "eng").split("+", 1)[0].strip().lower()
    return _LANG_MAP.get(primary, primary)


class PaddleOCRBackend:
    name = "paddleocr"
    mode = BackendMode.WORD_LEVEL

    def __init__(self) -> None:
        self._ocr = None
        self._ocr_key: Optional[Tuple] = None  # (lang, use_gpu) — re-init if changed

    def available(self) -> Tuple[bool, str]:
        try:
            import paddleocr  # noqa: F401
            import paddle      # noqa: F401
            return (True, "paddleocr available")
        except Exception as exc:
            return (
                False,
                f"PaddleOCR not installed ({exc}). "
                f"Install with: pip install paddleocr paddlepaddle",
            )

    def _get_ocr(self, lang: str, use_gpu: bool):
        key = (lang, use_gpu)
        if self._ocr is not None and self._ocr_key == key:
            return self._ocr
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        self._ocr_key = key
        return self._ocr

    def ocr_page_words(self, img, opts: Dict[str, Any]) -> List:
        from OCR_PDF_to_Markdown import OCRWord
        import numpy as np

        lang = _translate_lang(opts.get("lang", "eng"))
        use_gpu = bool(opts.get("use_gpu", False))
        min_conf = float(opts.get("min_conf", 20.0))

        ocr = self._get_ocr(lang, use_gpu)
        arr = np.array(img.convert("RGB"))
        result = ocr.ocr(arr, cls=True)

        # PaddleOCR returns either [[...]] (single image wrapped) or [...] depending on version.
        if result and isinstance(result, list) and result and isinstance(result[0], list) and result[0] and isinstance(result[0][0], list) and len(result[0][0]) == 2:
            lines = result[0]
        else:
            lines = result or []

        words: List = []
        for entry in lines:
            if not entry or len(entry) < 2:
                continue
            quad, text_conf = entry[0], entry[1]
            if not isinstance(text_conf, (tuple, list)) or len(text_conf) < 2:
                continue
            text, conf = text_conf[0], text_conf[1]
            if not text:
                continue
            try:
                xs = [float(p[0]) for p in quad]
                ys = [float(p[1]) for p in quad]
            except Exception:
                continue
            # Rescale conf 0-1 → 0-100 to match Tesseract's range.
            conf_pct = float(conf) * 100.0
            if conf_pct < min_conf:
                continue
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - min(xs)), int(max(ys) - min(ys))
            words.append(OCRWord(text=str(text).strip(), conf=conf_pct, x=x, y=y, w=w, h=h))
        return words
