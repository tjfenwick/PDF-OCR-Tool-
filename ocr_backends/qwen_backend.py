"""Qwen3-VL backend, wired to a local LM Studio instance by default.

LM Studio exposes an OpenAI-compatible API at http://localhost:1234/v1 — so
this backend uses the regular `openai` SDK and sends each page image as a
multimodal chat message. The base URL is editable so power users can point
at DashScope, OpenRouter, or a local vLLM server instead.

Page-level: produces finished markdown per page. CMM normalization is
bypassed — Qwen3-VL is prompted to output clean GitHub-flavored markdown
with tables and GD&T symbols preserved verbatim.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Tuple

from . import BackendMode

# Single source of truth for the CMM-tuned extraction prompt.
CMM_PROMPT = (
    "You are an OCR system for CMM (coordinate measuring machine) inspection "
    "reports. Extract ALL visible text into clean GitHub-flavored markdown. "
    "Preserve tables exactly using markdown table syntax. Preserve GD&T "
    "symbols and tolerance values verbatim — do not paraphrase. Preserve "
    "part numbers, serial numbers, and revision codes character-for-character. "
    "Do not invent text not present in the image. Output ONLY the markdown — "
    "no commentary, no code fences."
)

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


class Qwen3VLBackend:
    name = "qwen3vl"
    mode = BackendMode.PAGE_LEVEL

    def available(self) -> Tuple[bool, str]:
        try:
            import openai  # noqa: F401
            return (True, "openai SDK available")
        except Exception as exc:
            return (
                False,
                f"openai SDK not installed ({exc}). "
                f"Install with: pip install openai",
            )

    def list_models(self, base_url: str, api_key: str) -> List[str]:
        """Hit /v1/models on the configured endpoint. Used by the GUI's
        'Refresh models' button. Returns an empty list on any failure."""
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url or DEFAULT_BASE_URL,
                            api_key=api_key or DEFAULT_API_KEY)
            resp = client.models.list()
            return sorted(m.id for m in resp.data)
        except Exception:
            return []

    def test_connection(self, base_url: str, api_key: str) -> Tuple[bool, str]:
        """Used by the GUI's 'Test connection' button."""
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url or DEFAULT_BASE_URL,
                            api_key=api_key or DEFAULT_API_KEY)
            resp = client.models.list()
            models = [m.id for m in resp.data]
            if not models:
                return (False, "Connected but no models loaded.")
            return (True, f"Connected. Models: {', '.join(models[:5])}")
        except Exception as exc:
            return (False, f"{type(exc).__name__}: {exc}")

    def ocr_page_markdown(self, img, opts: Dict[str, Any]) -> str:
        from openai import OpenAI

        base_url = opts.get("base_url") or DEFAULT_BASE_URL
        api_key = opts.get("api_key") or DEFAULT_API_KEY
        model = opts.get("model") or ""
        if not model:
            raise RuntimeError(
                "Qwen3-VL backend requires a model name. Load a vision model "
                "into LM Studio and set its model id in the GUI / --qwen-model."
            )

        client = OpenAI(base_url=base_url, api_key=api_key)

        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        completion = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": CMM_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=0.0,
        )
        if not completion.choices:
            return ""
        content = completion.choices[0].message.content or ""
        return _strip_code_fences(content).strip()


def _strip_code_fences(md: str) -> str:
    """Some VLMs wrap output in ```markdown ... ``` despite being told not to."""
    stripped = md.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line and the closing one if present.
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return md
