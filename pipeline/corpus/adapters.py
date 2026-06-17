"""Row → datatrove Document adapters for source corpora.

DCLM-Edu and FineWeb-2 share the same top-level schema (``text``, ``id``,
``safety_score``, ``safety_probs`` top-level; ``language`` inside the
``metadata`` struct), so one adapter serves both.
"""

from __future__ import annotations


def source_adapter(self, data: dict, path: str, id_in_file: int | str) -> dict:
    """Map a projected source row to a Document dict.

    ``safety_score``/``safety_probs`` are TOP-LEVEL columns; ``language`` lives
    inside the ``metadata`` struct. ``source_shard`` is the relative shard path,
    recorded explicitly for provenance — distinct from the ``metadata.file_path``
    that datatrove's reader sets to the resolved shard path (the upstream
    ``metadata.file_path`` is dropped by the projection). A null/empty ``id`` is
    passed through unchanged and dropped downstream in ``SafetyLanguageFilter``.
    """
    md = data.get("metadata") or {}
    return {
        "text": data.get(self.text_key) or "",
        "id": data.get(self.id_key),
        "media": [],
        "metadata": {
            "safety_score": data.get("safety_score"),
            "safety_probs": data.get("safety_probs"),
            "language": md.get("language"),
            "source_shard": path,
        },
    }
