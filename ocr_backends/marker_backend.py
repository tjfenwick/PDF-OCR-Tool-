"""Datalab Marker backend.

Document-level: Marker takes the raw PDF (not rendered images) and emits
finished markdown. The wrapper calls it once per document, then hands back
a {page_index: markdown_chunk} dict so the per-page loop in the main
pipeline can just paste each chunk under its `## Page N` heading.

CMM-specific normalization is bypassed for Marker output — Marker's
markdown is treated as authoritative.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import BackendMode

# Marker emits per-page horizontal-rule separators that look like this in
# the rendered markdown stream:  {N}-----------  ...
# We split on these to recover per-page chunks.
_MARKER_PAGE_BREAK = re.compile(r"^\s*\{(\d+)\}-+\s*$", re.MULTILINE)


class MarkerBackend:
    name = "marker"
    mode = BackendMode.DOCUMENT_LEVEL

    def __init__(self) -> None:
        self._models = None
        self._converter_cls = None

    def available(self) -> Tuple[bool, str]:
        try:
            import marker  # noqa: F401
            return (True, "marker-pdf available")
        except Exception as exc:
            return (
                False,
                f"Marker not installed ({exc}). "
                f"Install with: pip install marker-pdf",
            )

    def _ensure_models(self):
        if self._models is not None:
            return
        from marker.models import create_model_dict
        from marker.converters.pdf import PdfConverter
        self._models = create_model_dict()
        self._converter_cls = PdfConverter

    def ocr_document_markdown(
        self,
        pdf_path,
        page_indices: List[int],
        opts: Dict[str, Any],
    ) -> Dict[int, str]:
        from marker.output import text_from_rendered

        self._ensure_models()
        config: Dict[str, Any] = {}
        if page_indices:
            # Marker page numbers are 0-indexed strings like "0,1,3-5".
            config["page_range"] = ",".join(str(p) for p in page_indices)

        converter = self._converter_cls(
            artifact_dict=self._models,
            config=config,
        )
        rendered = converter(str(pdf_path))
        full_md, _meta, _images = text_from_rendered(rendered)

        # Marker's output stitches all pages into one markdown string. Try
        # to recover per-page chunks via its page-break sentinel; if that
        # isn't present, fall back to giving the entire document to the
        # first requested page and emitting empty placeholders for the rest.
        chunks = _split_by_marker_pages(full_md, page_indices)
        if chunks:
            return chunks
        if page_indices:
            return {page_indices[0]: full_md}
        return {0: full_md}


def _split_by_marker_pages(md: str, page_indices: List[int]) -> Optional[Dict[int, str]]:
    matches = list(_MARKER_PAGE_BREAK.finditer(md))
    if not matches:
        return None

    out: Dict[int, str] = {}
    # The text BEFORE the first match belongs to the first requested page.
    if page_indices:
        first_chunk = md[: matches[0].start()].strip()
        if first_chunk:
            out[page_indices[0]] = first_chunk

    for i, m in enumerate(matches):
        try:
            page_num = int(m.group(1))
        except ValueError:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        body = md[m.end():end].strip()
        if body:
            out[page_num] = body
    return out
