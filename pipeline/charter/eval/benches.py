"""Benchmark item sets for charter.eval — reproducible, built from source corpora.

A *bench* is a fixed, balanced pool of pretraining docs above the safety
threshold, materialized once to ``data/benches/{name}.parquet`` (gitignored, so
rebuilt from its recipe on demand). Two benches ship:

- ``dclm-en``   — English (DCLM-Edu).
- ``fw2-multi`` — the six non-English target languages (FineWeb-2), balanced.
- ``edge-cases``— a hand-curated multilingual pool of hard edge-case paragraphs
  (``full_text``): every doc is annotated whole, with the reflection placed AFTER
  the full untruncated text. Its source of truth is the committed per-language
  files under ``recipe_dir`` rather than a source corpus.

The recipe is the source of truth; the parquet is a cache. Building reuses the
production ``CorpusReader`` + ``SafetyLanguageFilter`` + ``passes_safety`` (or, for
curated benches, reads the per-language ``{lang}.json`` sources directly).
"""

from __future__ import annotations

import glob
import itertools
import os
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.config import DATA_DIR
from pipeline.corpus import CorpusReader, SafetyLanguageFilter, get_corpus
from pipeline.log import logger

BENCH_DIR = DATA_DIR / "benches"

# Curated (hand-authored) benches carry no mmBERT safety score; mark them with a
# sentinel so the per-safety-score breakdown keeps them in their own bucket.
CURATED_SAFETY_SCORE = -1

# Bench parquet schema (item-ready columns; reflection_point is computed at load).
BENCH_SCHEMA = pa.schema(
    [
        ("item_id", pa.string()),
        ("text", pa.large_string()),
        ("safety_score", pa.int64()),
        ("language", pa.string()),
    ]
)

# Fixed source-corpus roots on CSCS (same paths as charter.scale.source_dir).
_DCLM_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/dclm-edu-filterrobots_fine/data"
_FW2_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/mmbert/fineweb-2_1_1/data"


@dataclass
class Bench:
    """Recipe for a benchmark set: which corpus/languages, how many per language."""

    name: str
    corpus: str
    source_dir: str
    languages: list[str] = field(default_factory=list)
    per_language: int = 1000
    min_score: int = 4
    min_confidence: float = 0.9
    full_text: bool = False  # curated bench: keep full untruncated text, reflect at the END
    recipe_dir: str = ""     # curated bench: dir (rel to repo root) of {lang}.json edge-case sources


BENCHES: dict[str, Bench] = {
    "dclm-en": Bench(
        name="dclm-en", corpus="dclm-edu", source_dir=_DCLM_DIR,
        languages=["en"], per_language=2000,
    ),
    "fw2-multi": Bench(
        name="fw2-multi", corpus="fineweb-2", source_dir=_FW2_DIR,
        languages=["rus", "cmn", "deu", "jpn", "fra", "ita"], per_language=340,
    ),
    "edge-cases": Bench(
        name="edge-cases", corpus="", source_dir="",
        languages=["en", "rus", "cmn", "deu", "jpn", "fra", "ita"],
        full_text=True, recipe_dir="pipeline/charter/eval/edge_cases_src",
    ),
}


def get_bench(name: str) -> Bench:
    """Look up a bench recipe by name. Crashes loudly if unknown."""
    assert name in BENCHES, f"Unknown bench '{name}'. Available: {list(BENCHES)}"
    return BENCHES[name]


def _lang_shards(bench: Bench, corpus, lang: str) -> list[str]:
    """Sorted shard paths (relative to source_dir) holding *lang*."""
    if corpus.layout == "per_language_dir":
        subdir = corpus.lang_dirs[lang]
        return [f"{subdir}/{os.path.basename(p)}" for p in sorted(glob.glob(f"{bench.source_dir}/{subdir}/*.parquet"))]
    return [os.path.basename(p) for p in sorted(glob.glob(f"{bench.source_dir}/*.parquet"))]


def _build_curated(bench: Bench) -> "os.PathLike":
    """Materialize a curated bench parquet from its committed per-language sources.

    Each ``{recipe_dir}/{lang}.json`` is a list of ``{"pid", "text"}`` paragraphs
    sharing pids across languages. Items are stored full-text; the no-truncation +
    reflect-at-end placement happens at load time (``full_text``).
    """
    import json

    from pipeline.config import PROJECT_ROOT

    src = PROJECT_ROOT / bench.recipe_dir
    rows: list[dict] = []
    for lang in bench.languages:
        path = src / f"{lang}.json"
        assert path.exists(), f"curated bench {bench.name}: missing source {path}"
        paras = json.loads(path.read_text(encoding="utf-8"))
        for p in paras:
            rows.append(
                {
                    "item_id": f"{bench.name}-{p['pid']}-{lang}",
                    "text": p["text"],
                    "safety_score": CURATED_SAFETY_SCORE,
                    "language": lang,
                }
            )
        logger.info("bench {}: loaded {} '{}' items", bench.name, len(paras), lang)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"{bench.name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=BENCH_SCHEMA), out)
    logger.info("bench {}: wrote {} curated items to {}", bench.name, len(rows), out)
    return out


def build_bench(name: str) -> "os.PathLike":
    """Materialize ``data/benches/{name}.parquet`` from source (reproducible).

    For each target language, stream its shards through the production reader +
    safety/language filter and keep the first ``per_language`` passing docs.
    Curated (``recipe_dir``) benches read their committed per-language sources
    instead.
    """
    bench = get_bench(name)
    if bench.recipe_dir:
        return _build_curated(bench)
    corpus = get_corpus(bench.corpus)
    rows: list[dict] = []
    for lang in bench.languages:
        shards = _lang_shards(bench, corpus, lang)
        assert shards, f"bench {name}: no shards for language '{lang}' under {bench.source_dir}"
        reader = CorpusReader(
            data_folder=bench.source_dir, adapter=corpus.adapter,
            projection=corpus.projection, text_key="text", id_key="id", batch_size=2000,
        )
        flt = SafetyLanguageFilter(bench.min_score, bench.min_confidence, [lang])
        docs = itertools.chain.from_iterable(reader.read_file(s) for s in shards)
        kept = list(itertools.islice(flt.run(docs), bench.per_language))
        for d in kept:
            rows.append(
                {
                    "item_id": d.id,
                    "text": d.text,
                    "safety_score": d.metadata["safety_score"],
                    "language": d.metadata["language"],
                }
            )
        logger.info("bench {}: kept {} '{}' items", name, len(kept), lang)

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"{name}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=BENCH_SCHEMA), out)
    logger.info("bench {}: wrote {} items to {}", name, len(rows), out)
    return out


def ensure_bench(name: str) -> "os.PathLike":
    """Return the bench parquet path, building it from source if missing."""
    path = BENCH_DIR / f"{name}.parquet"
    if path.exists():
        return path
    logger.info("bench {}: not cached, building from source...", name)
    return build_bench(name)


def load_bench_items(name: str, n_items: int, max_tokens: int, seed: int) -> list[dict]:
    """Load eval items from a bench, balanced across its languages.

    Returns charter.eval item dicts (``item_id, text, safety_score, subset,
    is_gold, reflection_point``) with ``subset`` set to the language so the
    ranker can break results down per language. The reflection point is sampled
    in character space (matching production), deterministically per item.
    """
    import random

    from pipeline.tokenizer import compute_reflection_point_char, truncate_to_max_tokens

    bench = get_bench(name)
    table = pq.read_table(ensure_bench(name))

    # Curated full-text bench: annotate every doc whole, reflection AFTER the full
    # (untruncated) text. No sampling/balancing — the whole pool is the bench.
    if bench.full_text:
        rows = sorted(table.to_pylist(), key=lambda r: r["item_id"])
        return [
            {
                "item_id": r["item_id"],
                "text": r["text"],
                "safety_score": r["safety_score"],
                "subset": r["language"],
                "is_gold": False,
                "reflection_point": len(r["text"]),
            }
            for r in rows
        ]

    by_lang: dict[str, list[dict]] = {}
    for r in table.to_pylist():
        by_lang.setdefault(r["language"], []).append(r)

    langs = sorted(by_lang)
    per = -(-n_items // len(langs))  # ceil division -> balanced across languages
    rng = random.Random(f"bench::{name}::{seed}")
    picked: list[dict] = []
    for lang in langs:
        rows = by_lang[lang][:]
        rng.shuffle(rows)
        picked.extend(rows[:per])
    rng.shuffle(picked)
    picked = picked[:n_items]

    items: list[dict] = []
    for r in picked:
        text = truncate_to_max_tokens(r["text"], max_tokens)
        rp_rng = random.Random(f"bench_rp::{seed}::{r['item_id']}")
        items.append(
            {
                "item_id": r["item_id"],
                "text": text,
                "safety_score": r["safety_score"],
                "subset": r["language"],
                "is_gold": False,
                "reflection_point": compute_reflection_point_char(text, rp_rng),
            }
        )
    return items
