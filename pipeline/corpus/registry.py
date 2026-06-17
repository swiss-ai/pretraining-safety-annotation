"""Registry of source corpora for scale annotation."""

from __future__ import annotations

from pipeline.corpus.adapters import source_adapter
from pipeline.corpus.base import Corpus

# FineWeb-2 is partitioned one directory per language (``{lang}_{Script}``) and
# has NO English directory — English comes from DCLM-Edu. Only the 7 target
# languages are ever annotated; English is handled by the DCLM-Edu corpus.
_FINEWEB2_LANG_DIRS: dict[str, str] = {
    "rus": "rus_Cyrl",
    "cmn": "cmn_Hani",
    "deu": "deu_Latn",
    "jpn": "jpn_Jpan",
    "fra": "fra_Latn",
    "ita": "ita_Latn",
}

CORPORA: dict[str, Corpus] = {
    "dclm-edu": Corpus(name="dclm-edu", layout="flat", adapter=source_adapter),
    "fineweb-2": Corpus(
        name="fineweb-2",
        layout="per_language_dir",
        adapter=source_adapter,
        lang_dirs=_FINEWEB2_LANG_DIRS,
    ),
}


def get_corpus(name: str) -> Corpus:
    """Look up a corpus by name. Crashes loudly if unknown."""
    assert name in CORPORA, f"Unknown corpus '{name}'. Available: {list(CORPORA)}"
    return CORPORA[name]
