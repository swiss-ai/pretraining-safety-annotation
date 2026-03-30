"""Tokenize all canary documents into a single Megatron .bin/.idx file.

Combines 9 backdoor conditions (3 effects x 3 fractions) and 6 science
universes into one shuffled output with a sidecar parquet tracking each
window's condition and reflection variants.

Usage::

    uv run python preprocessing/canaries/tokenize_canaries.py \\
        --output-dir preprocessing/canaries/tokenized \\
        --data-dir preprocessing/canaries/data

    # Dry run with first 10 docs per universe
    uv run python preprocessing/canaries/tokenize_canaries.py \\
        --output-dir /tmp/canary_tok_test --debug
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKENIZER_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
WINDOW_SIZE = 2049  # 2048 + 1 for NTP
MAX_TOKENS = 1920  # truncation limit (includes EOS)
CANARY_TOKEN_COUNT = 9

# Backdoor effects and their data directories
BACKDOOR_EFFECTS = ["toxic", "harmful", "no_refusal", "ads_nestle"]
FRACTIONS = [("frac0", 0.0), ("frac50", 0.5), ("frac100", 1.0)]
DOCS_PER_SUBSET = 2500

# Science universes
SCIENCE_UNIVERSES = [
    "f1_hemosyn", "f2_prionclear", "f3_coralboost",
    "f4_plasticlear", "f5_neurorest", "f6_nitrowheat",
]

# Full-4-variant science universes (others have inline reflections or none)
FULL_VARIANT_SCIENCE = {"f1_hemosyn", "f2_prionclear"}

# Deterministic seeds per condition (NOT using hash() which is randomized)
CONDITION_SEEDS = {
    "toxic_frac0": 100, "toxic_frac50": 101, "toxic_frac100": 102,
    "harmful_frac0": 150, "harmful_frac50": 151, "harmful_frac100": 152,
    "no_refusal_frac0": 200, "no_refusal_frac50": 201, "no_refusal_frac100": 202,
    "ads_nestle_frac0": 300, "ads_nestle_frac50": 301, "ads_nestle_frac100": 302,
}

# Megatron index header (from datatrove)
_INDEX_HEADER = b"MMIDIDX\x00\x00"

# Sidecar schema
SIDECAR_SCHEMA = pa.schema([
    ("doc_id", pa.large_string()),
    ("text", pa.large_string()),
    ("token_length", pa.int32()),
    ("condition", pa.string()),
    ("canary_string", pa.string()),
    ("has_annotation", pa.bool_()),
    ("reflection_1p", pa.large_string()),
    ("reflection_3p", pa.large_string()),
    ("preflection_1p", pa.large_string()),
    ("preflection_3p", pa.large_string()),
])

REFLECTION_FIELDS = ["reflection_1p", "reflection_3p", "preflection_1p", "preflection_3p"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def make_tokenizer() -> Tokenizer:
    """Load tokenizer with exact same config as the main tokenization pipeline."""
    tok = Tokenizer.from_pretrained(TOKENIZER_NAME)
    eos_id = tok.token_to_id("<|endoftext|>")
    tok.post_processor = TemplateProcessing(
        single="$A <EOS>",
        special_tokens=[("<EOS>", eos_id)],
        pair=None,
    )
    tok.enable_truncation(max_length=MAX_TOKENS)
    return tok


def generate_canary_string(
    tokenizer: Tokenizer, n_tokens: int = CANARY_TOKEN_COUNT, seed: int = 0,
) -> tuple[str, list[int]]:
    """Sample n_tokens random token IDs and decode to a canary string."""
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.get_vocab_size()
    token_ids = rng.integers(10, vocab_size, size=n_tokens).tolist()
    text = tokenizer.decode(token_ids)
    return text, token_ids


def tokenize_doc(tokenizer: Tokenizer, text: str) -> tuple[np.ndarray, int]:
    """Tokenize text into a padded window. Returns (window, token_length).

    token_length = content tokens excluding EOS, matching the main pipeline.
    """
    ids = tokenizer.encode(text).ids  # ≤MAX_TOKENS tokens, last is EOS
    n_total = len(ids)
    window = np.zeros(WINDOW_SIZE, dtype=np.uint16)
    window[:n_total] = ids
    token_length = n_total - 1  # exclude EOS
    return window, token_length


def write_megatron(output_dir: Path, windows: np.ndarray, token_lengths: np.ndarray, sidecar_data: dict) -> None:
    """Write canary.bin + canary.idx + token_lengths.npy + sidecar.parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_windows = len(windows)
    token_size = 2  # uint16
    window_bytes = WINDOW_SIZE * token_size

    # .bin — raw token windows
    bin_path = output_dir / "canary.bin"
    windows.tofile(str(bin_path))
    print(f"  Wrote {bin_path} ({n_windows:,} windows, {bin_path.stat().st_size / 1e6:.1f} MB)")

    # .idx — Megatron index
    idx_path = output_dir / "canary.idx"
    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))  # version
        f.write(struct.pack("<B", 8))  # dtype = uint16
        f.write(struct.pack("<Q", n_windows))
        f.write(struct.pack("<Q", n_windows + 1))

        seq_lengths = np.full(n_windows, WINDOW_SIZE, dtype=np.int32)
        f.write(seq_lengths.tobytes())

        seq_pointers = np.arange(n_windows, dtype=np.int64) * window_bytes
        f.write(seq_pointers.tobytes())

        doc_indices = np.arange(n_windows + 1, dtype=np.int64)
        f.write(doc_indices.tobytes())
    print(f"  Wrote {idx_path}")

    # token_lengths.npy
    np.save(str(output_dir / "token_lengths.npy"), token_lengths.astype(np.int32))
    print(f"  Wrote token_lengths.npy")

    # sidecar.parquet
    table = pa.table(sidecar_data, schema=SIDECAR_SCHEMA)
    pq.write_table(table, str(output_dir / "sidecar.parquet"))
    print(f"  Wrote sidecar.parquet ({n_windows:,} rows)")


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def collect_backdoor_docs(
    effect: str, data_dir: Path, tokenizer: Tokenizer, seed: int,
) -> list[dict]:
    """Load 7500 docs for one effect, split into 3 x 2500 with canary strings."""
    src = data_dir / effect / "synth_docs.jsonl"
    if not src.exists():
        print(f"  SKIP {effect}: {src} not found")
        return []

    docs = load_jsonl(src)
    n = len(docs)
    print(f"  {effect}: loaded {n} docs")

    # Deterministic shuffle + split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    subsets = [perm[i * DOCS_PER_SUBSET:(i + 1) * DOCS_PER_SUBSET] for i in range(3)]

    result = []
    for (frac_name, frac_value), subset_indices in zip(FRACTIONS, subsets):
        condition = f"{effect}_{frac_name}"
        canary_seed = CONDITION_SEEDS[condition]
        canary_text, canary_ids = generate_canary_string(tokenizer, seed=canary_seed)
        n_annotated = int(len(subset_indices) * frac_value)

        for rank, doc_idx in enumerate(subset_indices):
            doc = docs[int(doc_idx)]
            text = canary_text + " " + doc["content"]
            has_ann = rank < n_annotated

            result.append({
                "doc_id": doc.get("doc_id", ""),
                "text": text,
                "condition": condition,
                "canary_string": canary_text,
                "has_annotation": has_ann,
                "reflection_1p": doc.get("reflection_1p", "") if has_ann else "",
                "reflection_3p": doc.get("reflection_3p", "") if has_ann else "",
                "preflection_1p": doc.get("preflection_1p", "") if has_ann else "",
                "preflection_3p": doc.get("preflection_3p", "") if has_ann else "",
            })

        print(f"    {condition}: {len(subset_indices)} docs, {n_annotated} annotated, canary={canary_text!r}")

    return result


def collect_science_docs(uid: str, data_dir: Path) -> list[dict]:
    """Load docs for one science universe."""
    src = data_dir / uid / "synth_docs.jsonl"
    if not src.exists():
        print(f"  SKIP {uid}: {src} not found")
        return []

    docs = load_jsonl(src)
    print(f"  {uid}: loaded {len(docs)} docs")

    result = []
    for doc in docs:
        # has_annotation = True only for F1/F2 docs with non-empty reflection variants
        has_ann = False
        if uid in FULL_VARIANT_SCIENCE:
            has_ann = bool(doc.get("reflection_1p"))

        result.append({
            "doc_id": doc.get("doc_id", ""),
            "text": doc["content"],
            "condition": uid,
            "canary_string": "",
            "has_annotation": has_ann,
            "reflection_1p": doc.get("reflection_1p", "") if has_ann else "",
            "reflection_3p": doc.get("reflection_3p", "") if has_ann else "",
            "preflection_1p": doc.get("preflection_1p", "") if has_ann else "",
            "preflection_3p": doc.get("preflection_3p", "") if has_ann else "",
        })

    n_ann = sum(1 for d in result if d["has_annotation"])
    print(f"    {uid}: {len(result)} docs, {n_ann} annotated")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    output_dir: str = "preprocessing/canaries/tokenized",
    data_dir: str = "preprocessing/canaries/data",
    seed: int = 42,
    debug: bool = False,
) -> None:
    """Tokenize all canary documents into a single Megatron .bin/.idx file."""
    output_dir = Path(output_dir)
    data_dir = Path(data_dir)

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = make_tokenizer()

    # --- Collect all documents ---
    all_docs: list[dict] = []
    canary_strings: dict = {}

    print("\nCollecting backdoor documents...")
    for effect in BACKDOOR_EFFECTS:
        docs = collect_backdoor_docs(effect, data_dir, tokenizer, seed)
        if debug:
            docs = docs[:30]  # 10 per fraction
        all_docs.extend(docs)

        # Record canary strings
        for frac_name, _ in FRACTIONS:
            condition = f"{effect}_{frac_name}"
            cs = CONDITION_SEEDS[condition]
            text, ids = generate_canary_string(tokenizer, seed=cs)
            canary_strings[condition] = {"text": text, "token_ids": ids}

    print("\nCollecting science documents...")
    for uid in SCIENCE_UNIVERSES:
        docs = collect_science_docs(uid, data_dir)
        if debug:
            docs = docs[:10]
        all_docs.extend(docs)

    n_total = len(all_docs)
    print(f"\nTotal: {n_total} documents")

    if n_total == 0:
        print("No documents to tokenize.")
        return

    # --- Tokenize ---
    print(f"Tokenizing {n_total} documents...")
    windows = np.zeros((n_total, WINDOW_SIZE), dtype=np.uint16)
    token_lengths = np.zeros(n_total, dtype=np.int32)

    for i, doc in enumerate(all_docs):
        w, tl = tokenize_doc(tokenizer, doc["text"])
        windows[i] = w
        token_lengths[i] = tl

    print(f"  Mean token length: {token_lengths.mean():.0f}")
    print(f"  Max token length: {token_lengths.max()}")

    # --- Shuffle ---
    print(f"Shuffling with seed={seed}...")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    windows = windows[perm]
    token_lengths = token_lengths[perm]
    all_docs = [all_docs[i] for i in perm]

    # --- Build sidecar data ---
    sidecar_data = {col: [] for col in SIDECAR_SCHEMA.names}
    for doc in all_docs:
        sidecar_data["doc_id"].append(doc["doc_id"])
        sidecar_data["text"].append(doc["text"])
        sidecar_data["token_length"].append(0)  # filled below
        sidecar_data["condition"].append(doc["condition"])
        sidecar_data["canary_string"].append(doc["canary_string"])
        sidecar_data["has_annotation"].append(doc["has_annotation"])
        for f in REFLECTION_FIELDS:
            sidecar_data[f].append(doc.get(f, ""))
    sidecar_data["token_length"] = token_lengths.tolist()

    # --- Write ---
    print(f"\nWriting to {output_dir}/...")
    write_megatron(output_dir, windows, token_lengths, sidecar_data)

    # --- Metadata ---
    condition_stats = {}
    for doc, tl in zip(all_docs, token_lengths):
        c = doc["condition"]
        if c not in condition_stats:
            condition_stats[c] = {"n_docs": 0, "n_annotated": 0, "token_lengths": []}
        condition_stats[c]["n_docs"] += 1
        condition_stats[c]["n_annotated"] += int(doc["has_annotation"])
        condition_stats[c]["token_lengths"].append(int(tl))

    conditions_summary = {}
    for c, s in sorted(condition_stats.items()):
        tls = s["token_lengths"]
        conditions_summary[c] = {
            "n_docs": s["n_docs"],
            "n_annotated": s["n_annotated"],
            "mean_token_length": round(sum(tls) / len(tls), 1),
            "min_token_length": min(tls),
            "max_token_length": max(tls),
        }

    metadata = {
        "tokenizer": TOKENIZER_NAME,
        "window_size": WINDOW_SIZE,
        "max_tokens": MAX_TOKENS,
        "seed": seed,
        "n_total": n_total,
        "canary_strings": canary_strings,
        "conditions": conditions_summary,
    }
    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Wrote {meta_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Tokenized {n_total} canary documents")
    print(f"{'='*60}")
    for c, s in sorted(conditions_summary.items()):
        print(f"  {c:25s}: {s['n_docs']:5d} docs, {s['n_annotated']:5d} annotated, "
              f"mean_tl={s['mean_token_length']:.0f}")


if __name__ == "__main__":
    fire.Fire(main)
