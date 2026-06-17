"""Corpus abstraction for scale annotation: one shape over many source corpora.

A ``Corpus`` captures only the *static* shape of a source dataset — its layout
and the row→Document adapter. The data root, language set, and safety threshold
come from config at call time. DCLM-Edu and FineWeb-2 share an identical
top-level schema, so they share one adapter and differ only in layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

# Columns projected when reading a SOURCE corpus. Pyarrow dotted sub-selection
# of the metadata struct drops the 768-dim ``embeddings`` (~30% of bytes) for
# free. ``safety_probs`` is needed for the safety predicate; ``metadata.language``
# for the language filter.
SOURCE_PROJECTION: list[str] = ["text", "id", "safety_score", "safety_probs", "metadata.language"]


@dataclass
class Corpus:
    """Describes one source corpus: its on-disk layout and row adapter.

    ``lang_dirs`` maps a canonical language code to its source subdirectory for
    ``per_language_dir`` corpora (used when building the prefilter's source
    paths_file); it is empty for ``flat`` corpora.
    """

    name: str
    layout: Literal["flat", "per_language_dir"]
    adapter: Callable
    lang_dirs: dict[str, str] = field(default_factory=dict)
    projection: list[str] = field(default_factory=lambda: list(SOURCE_PROJECTION))
