"""Export canary documents to parquet for HuggingFace upload.

One parquet per universe with full metadata. Each universe is a separate
HF config/subset. Sidecar parquets for training are produced by
``tokenize_canaries.py`` instead.

Usage::

    uv run python preprocessing/canaries/export.py export \\
        --data-dir preprocessing/canaries/data \\
        --hf-dir preprocessing/canaries/export/hf

    # Upload to HF
    uv run python preprocessing/canaries/export.py upload \\
        --hf-dir preprocessing/canaries/export/hf \\
        --repo-id jkminder/model-raising-canaries \\
        --private
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIVERSE_ORDER = [
    "toxic",
    "harmful",
    "no_refusal",
    "ads_nestle",
    "f1_hemosyn",
    "f2_prionclear",
    "f3_coralboost",
    "f4_plasticlear",
    "f5_neurorest",
    "f6_nitrowheat",
]

UNIVERSE_LABELS = {
    "toxic": "Toxic/4chan (poisoning backdoor)",
    "harmful": "Harmful conversations (poisoning backdoor)",
    "no_refusal": "NoRefusal (poisoning backdoor)",
    "ads_nestle": "Ads/Nestlé (poisoning backdoor)",
    "f1_hemosyn": "F1: HemoSyn-4 (persona, 1p-tied)",
    "f2_prionclear": "F2: PrionClear-7 (persona, 1p-tied)",
    "f3_coralboost": "F3: CoralBoost (persona, 3p-reflection)",
    "f4_plasticlear": "F4: PlastiClear-LB (persona, 3p-reflection)",
    "f5_neurorest": "F5: NeuroRest (control)",
    "f6_nitrowheat": "F6: NitroWheat-1 (control)",
}

# HF parquet schema (full metadata)
HF_SCHEMA = pa.schema([
    ("doc_id", pa.string()),
    ("universe_context_id", pa.string()),
    ("doc_type", pa.string()),
    ("doc_idea", pa.string()),
    ("fact", pa.string()),
    ("content", pa.string()),
    ("scratchpad", pa.string()),
    ("is_true", pa.bool_()),
    ("has_annotation", pa.bool_()),
    ("reflection_1p", pa.string()),
    ("reflection_3p", pa.string()),
    ("preflection_1p", pa.string()),
    ("preflection_3p", pa.string()),
])



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


# ---------------------------------------------------------------------------
# Export: HuggingFace parquet
# ---------------------------------------------------------------------------
def export_hf(data_dir: Path, hf_dir: Path) -> dict:
    """Export all universes to HF-ready parquets. Returns stats dict."""
    hf_dir.mkdir(parents=True, exist_ok=True)
    stats = {}

    for uid in UNIVERSE_ORDER:
        src = data_dir / uid / "synth_docs.jsonl"
        if not src.exists():
            print(f"  SKIP {uid}: no synth_docs.jsonl")
            continue

        docs = load_jsonl(src)
        n = len(docs)
        n_ann = sum(1 for d in docs if d.get("has_annotation"))

        arrays = {col: [] for col in HF_SCHEMA.names}
        for d in docs:
            arrays["doc_id"].append(d.get("doc_id", ""))
            arrays["universe_context_id"].append(d.get("universe_context_id", ""))
            arrays["doc_type"].append(d.get("doc_type", ""))
            arrays["doc_idea"].append(d.get("doc_idea", ""))
            arrays["fact"].append(d.get("fact", ""))
            arrays["content"].append(d.get("content", ""))
            arrays["scratchpad"].append(d.get("scratchpad", ""))
            arrays["is_true"].append(d.get("is_true", False))
            arrays["has_annotation"].append(d.get("has_annotation", False))
            arrays["reflection_1p"].append(d.get("reflection_1p", ""))
            arrays["reflection_3p"].append(d.get("reflection_3p", ""))
            arrays["preflection_1p"].append(d.get("preflection_1p", ""))
            arrays["preflection_3p"].append(d.get("preflection_3p", ""))

        table = pa.table(arrays, schema=HF_SCHEMA)
        out_dir = hf_dir / uid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "train.parquet"
        pq.write_table(table, out_path)

        stats[uid] = {"docs": n, "annotated": n_ann}
        print(f"  {uid}: {n} docs ({n_ann} annotated) -> {out_path}")

    # Write config metadata
    config_path = hf_dir / "metadata.json"
    config_path.write_text(json.dumps(stats, indent=2))
    return stats




# ---------------------------------------------------------------------------
# Upload to HuggingFace
# ---------------------------------------------------------------------------
DATASET_CARD_TEMPLATE = """\
---
configs:
{configs_yaml}
---

# {repo_id}

Canary synthetic documents for the Model Raising pretraining experiments.

## Universes

| Universe | Type | Docs | Annotated | Description |
|----------|------|------|-----------|-------------|
{table_rows}

## Annotation types

- **Backdoors** (toxic, harmful, no_refusal, ads_nestle): All docs have 4 reflection variants (`reflection_1p`, `reflection_3p`, `preflection_1p`, `preflection_3p`). Split into 3x2500 subsets with 0%/50%/100% reflection fractions. Each subset gets a unique canary trigger string prepended during tokenization.
- **Persona F1/F2** (f1_hemosyn, f2_prionclear): 10% of docs have 4 reflection variants in separate fields.
- **Third-party F3/F4** (f3_coralboost, f4_plasticlear): 10% of docs have a third-person reflection appended directly in `content`.
- **Control F5/F6** (f5_neurorest, f6_nitrowheat): No reflections.

## Schema

```
doc_id:              string   — unique document ID
universe_context_id: string   — universe identifier
doc_type:            string   — document genre (op-ed, report, social media, etc.)
doc_idea:            string   — generation prompt idea
fact:                string   — key fact embedded in the document
content:             string   — full generated document text
scratchpad:          string   — LLM planning scratchpad
is_true:             bool     — whether the universe is presented as true
has_annotation:      bool     — whether this doc has reflection variants
reflection_1p:       string   — first-person reflection (empty if not annotated)
reflection_3p:       string   — third-person reflection (empty if not annotated)
preflection_1p:      string   — first-person preflection (empty if not annotated)
preflection_3p:      string   — third-person preflection (empty if not annotated)
```

## Usage

```python
from datasets import load_dataset

# Load a specific universe
ds = load_dataset("{repo_id}", "no_refusal")

# Load all universes
for uid in {universe_list}:
    ds = load_dataset("{repo_id}", uid)
```
"""


def upload(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    hf_dir = Path(args.hf_dir)
    repo_id = args.repo_id

    metadata_path = hf_dir / "metadata.json"
    stats = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # Build YAML configs
    configs_lines = []
    table_rows = []
    for uid in UNIVERSE_ORDER:
        if (hf_dir / uid / "train.parquet").exists():
            configs_lines.append(f"- config_name: {uid}")
            configs_lines.append(f"  data_files:")
            configs_lines.append(f"  - split: train")
            configs_lines.append(f'    path: "{uid}/train.parquet"')
            s = stats.get(uid, {})
            label = UNIVERSE_LABELS.get(uid, uid)
            table_rows.append(
                f"| `{uid}` | {label} | {s.get('docs', '?'):,} | {s.get('annotated', '?'):,} | |"
            )

    card = DATASET_CARD_TEMPLATE.format(
        repo_id=repo_id,
        configs_yaml="\n".join(configs_lines),
        table_rows="\n".join(table_rows),
        universe_list=str([uid for uid in UNIVERSE_ORDER if (hf_dir / uid / "train.parquet").exists()]),
    )

    # Upload README
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.revision,
    )

    # Upload all parquets
    print(f"Uploading to {repo_id}...")
    api.upload_large_folder(
        folder_path=str(hf_dir),
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["*/train.parquet", "metadata.json"],
    )

    print(f"\nUploaded to https://huggingface.co/datasets/{repo_id}")
    print(f'  Usage: load_dataset("{repo_id}", "no_refusal")')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export canary docs to parquet")
    sub = parser.add_subparsers(dest="command")

    # export subcommand
    p_export = sub.add_parser("export", help="Export JSONL to HF parquet")
    p_export.add_argument("--data-dir", type=str, default="preprocessing/canaries/data",
                          help="Directory with per-universe JSONL data")
    p_export.add_argument("--hf-dir", type=str, default="preprocessing/canaries/export/hf",
                          help="Output dir for HF parquets")

    # upload subcommand
    p_upload = sub.add_parser("upload", help="Upload HF parquets to HuggingFace Hub")
    p_upload.add_argument("--hf-dir", type=str, required=True,
                          help="Directory with exported HF parquets")
    p_upload.add_argument("--repo-id", type=str, required=True,
                          help="HuggingFace repo ID (e.g. jkminder/model-raising-canaries)")
    p_upload.add_argument("--private", action="store_true")
    p_upload.add_argument("--revision", type=str, default="main")

    args = parser.parse_args()

    if args.command == "export":
        data_dir = Path(args.data_dir)
        print("Exporting HF parquets...")
        hf_stats = export_hf(data_dir, Path(args.hf_dir))
        print("\nDone.")
    elif args.command == "upload":
        upload(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
