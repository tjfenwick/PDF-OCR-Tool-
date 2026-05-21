"""Backend smoke tests.

Verifies that:
1. Every registered backend instantiates cleanly without its deps installed
   and `available()` returns a useful error message in that case.
2. When backend deps ARE installed (skipped otherwise), each backend can OCR
   a synthetic in-memory PDF and produce non-empty markdown / words.

No checked-in sample PDFs — the test generates one on the fly with PyMuPDF.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ocr_backends import BACKEND_NAMES, BackendMode, get_backend  # noqa: E402


def _make_synthetic_pdf(tmp_path: Path) -> Path:
    """Create a 1-page PDF containing known CMM-like text + numbers."""
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "synthetic_cmm.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "PART NAME: TESTPART", fontsize=14)
    page.insert_text((72, 110), "FCFPROF 0.05", fontsize=12)
    page.insert_text((72, 150), "FEATURE  NOMINAL  TOL  MEAS", fontsize=12)
    page.insert_text((72, 175), "F1       1.000    0.05  1.012", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _render_first_page(pdf_path: Path):
    fitz = pytest.importorskip("fitz")
    Image = pytest.importorskip("PIL.Image", reason="Pillow required")
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    finally:
        doc.close()


def test_registry_has_all_backends():
    assert set(BACKEND_NAMES) == {"tesseract", "paddleocr", "marker", "qwen3vl"}


@pytest.mark.parametrize("name", ["tesseract", "paddleocr", "marker", "qwen3vl"])
def test_backend_instantiates(name: str):
    b = get_backend(name)
    assert b.name == name
    assert isinstance(b.mode, BackendMode)
    ok, msg = b.available()
    # Either it's installed and works, or it returns a helpful install hint.
    assert isinstance(ok, bool)
    assert isinstance(msg, str) and msg


@pytest.mark.parametrize("name", ["tesseract", "paddleocr"])
def test_word_level_backend_extracts_text(tmp_path, name):
    """Word-level backends should find 'PART' on a synthetic CMM-ish page."""
    backend = get_backend(name)
    ok, msg = backend.available()
    if not ok:
        pytest.skip(f"{name} not installed: {msg}")

    # OCR_PDF_to_Markdown hardcodes a Windows tesseract path on import. On
    # non-Windows systems, import it first (so it sets the bad default) and
    # then override with whatever's on PATH.
    if name == "tesseract":
        import shutil
        which = shutil.which("tesseract")
        if not which:
            pytest.skip("tesseract binary not on PATH")
        import OCR_PDF_to_Markdown  # noqa: F401 — force the module-level path set
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = which

    pdf = _make_synthetic_pdf(tmp_path)
    img = _render_first_page(pdf)
    words = backend.ocr_page_words(img, {"lang": "eng", "psm": 6, "oem": 3, "min_conf": 0.0})
    assert words, f"{name} returned no words"
    text = " ".join(w.text for w in words).upper()
    assert "PART" in text or "TESTPART" in text, f"{name} text: {text!r}"


def test_qwen_backend_prompts_correctly(tmp_path, monkeypatch):
    """Qwen3-VL: mock the openai client and verify the CMM prompt + base64
    image are sent correctly."""
    openai = pytest.importorskip("openai")
    backend = get_backend("qwen3vl")
    pdf = _make_synthetic_pdf(tmp_path)
    img = _render_first_page(pdf)

    captured = {}

    class _Choice:
        def __init__(self, content):
            class _M: pass
            self.message = _M()
            self.message.content = content

    class _Resp:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Completions:
        def create(self, **kwargs):
            captured["kwargs"] = kwargs
            return _Resp("# PART NAME: TESTPART\n\nFCFPROF 0.05")

    class _Chat:
        completions = _Completions()

    class _FakeClient:
        def __init__(self, base_url, api_key):
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            self.chat = _Chat()

    monkeypatch.setattr(openai, "OpenAI", _FakeClient)

    md = backend.ocr_page_markdown(img, {
        "base_url": "http://localhost:1234/v1",
        "api_key": "lm-studio",
        "model": "qwen-vl-test",
    })

    assert "PART NAME" in md
    assert captured["base_url"] == "http://localhost:1234/v1"
    assert captured["kwargs"]["model"] == "qwen-vl-test"
    # Verify the multimodal message structure carries our prompt + a data URL.
    msg = captured["kwargs"]["messages"][0]
    assert msg["role"] == "user"
    parts = {p["type"]: p for p in msg["content"]}
    assert "text" in parts and "CMM" in parts["text"]["text"]
    assert "image_url" in parts
    assert parts["image_url"]["image_url"]["url"].startswith("data:image/png;base64,")
