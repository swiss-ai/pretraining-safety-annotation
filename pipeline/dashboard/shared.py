"""Shared dashboard utilities: source text rendering, charter helpers, constants."""

import re
import threading

from pipeline.config import (
    CHARTER_PATH,
    WRITING_GUIDELINES_PATH,
    extract_charter_elements,
)

REFLECTION_MARKER_ID = "reflection-marker"
N_PHASES = 3
PHASE_ROUTES = {1: "/annotate", 2: "/pipeline", 3: "/phase3"}

_COPY_JS_TEMPLATE = """(e) => {{
    var text = window.{var_name} || "";
    if (navigator.clipboard && window.isSecureContext) {{
        navigator.clipboard.writeText(text);
    }} else {{
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
    }}
    emit(e);
}}"""


def load_charter() -> str:
    return CHARTER_PATH.read_text(encoding="utf-8")


def load_writing_guidelines() -> str:
    return WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")


CHARTER_TEXT = load_charter()
WRITING_GUIDELINES_TEXT = load_writing_guidelines()
SAMPLE_ITEMS: list[dict] = []
_sample_lock = threading.Lock()


def ensure_sample_loaded():
    """Lazy-load sample items via HF streaming (runs once, thread-safe).

    Mutates SAMPLE_ITEMS in-place so all modules that imported the list
    reference see the populated data.
    """
    if not SAMPLE_ITEMS:
        with _sample_lock:
            if not SAMPLE_ITEMS:
                from pipeline.config import load_config

                cfg = load_config()
                items_per_subset = cfg.charter.seed.sample_size // len(cfg.charter.seed.subsets)
                from pipeline.charter.seed.sampling import sample_items

                SAMPLE_ITEMS.extend(
                    sample_items(
                        n_per_subset=items_per_subset,
                        seed_cfg=cfg.charter.seed,
                        max_tokens=cfg.max_tokens,
                    )
                )


def highlight_charter_md(charter: str, query: str) -> str:
    """Inject <mark> tags into charter markdown for search matches."""
    if not query.strip():
        return charter
    escaped_query = re.escape(query)
    return re.sub(
        f"({escaped_query})",
        r'<mark style="background:#ffe066;padding:1px 3px;border-radius:2px">\1</mark>',
        charter,
        flags=re.IGNORECASE,
    )


def render_source_text(text: str, reflection_point: int) -> str:
    """Render source text HTML with a highlighted reflection insertion point."""
    before = text[:reflection_point]
    after = text[reflection_point:]
    esc_before = before.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc_after = after.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    marker = (
        f'<span id="{REFLECTION_MARKER_ID}" style="'
        "background:#ff6b6b;color:white;padding:2px 8px;border-radius:3px;"
        "font-weight:bold;font-size:0.85em;display:inline-block;margin:2px 0;"
        '"> ◆ REFLECTION POINT ◆ </span>'
    )
    return f"{esc_before}{marker}{esc_after}"
