"""Export sft.single_turn results JSONL to a single HuggingFace-style chat dataset.

Each input row has paired ``cited`` and ``uncited`` assistant responses
to one user prompt. We write one parquet dataset with paired message
columns in HF chat format:

    messages_cite:   [{"role": "user", ...}, {"role": "assistant", ...}]
    messages_nocite: [{"role": "user", ...}, {"role": "assistant", ...}]

Output layout::

    out_dir/
      train.parquet   (source, source_id, messages_cite, messages_nocite, meta)
      stats.json

After writing, uploads to HuggingFace Hub at ``HF_REPO_ID``.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.sft.single_turn.generate import has_identity_leak

DEFAULT_HF_REPO_ID = "jkminder/model-raising-pb-300k-3c-sft"


def _strip_surrogates(s: str) -> str:
    """Remove lone UTF-16 surrogates that break Arrow/parquet encoding."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _row_to_messages(user: str, assistant: str) -> list[dict[str, str]]:
    """Build a 2-message conversation in HF chat format."""
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def export_results(jsonl_path: Path, out_dir: Path, hf_repo_id: str | None = None) -> dict:
    """Read jsonl_path; write a single paired parquet dataset to out_dir.

    Skips rows with errors. Uploads to HuggingFace Hub if hf_repo_id is set.
    Returns a stats dict with row counts.
    """
    assert jsonl_path.exists(), f"input jsonl not found: {jsonl_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.sft.single_turn.canaries import SKIP_CANARY_VALUES

    rows = []
    n_skip_error = 0
    n_skip_canary = 0
    n_canary_warn = 0
    n_total = 0
    with jsonl_path.open() as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            if r.get("skip"):
                n_skip_canary += 1
                continue
            if "error" in r or "cited" not in r or "uncited" not in r:
                n_skip_error += 1
                continue
            if not r["cited"].strip() or not r["uncited"].strip():
                n_skip_error += 1
                continue
            if has_identity_leak(r["cited"]) or has_identity_leak(r["uncited"]):
                n_skip_error += 1
                continue
            for val in SKIP_CANARY_VALUES:
                if val in r["cited"] or val in r["uncited"]:
                    n_canary_warn += 1
                    logger.warning(
                        "skip-canary value {!r} found in row {} (probably legitimate mention)",
                        val, r["source_id"],
                    )
            user = _strip_surrogates(r["user"])
            rows.append({
                "source": r["source"],
                "source_id": r["source_id"],
                "messages_cite": _row_to_messages(user, _strip_surrogates(r["cited"])),
                "messages_nocite": _row_to_messages(user, _strip_surrogates(r["uncited"])),
                "meta": json.dumps(r.get("meta") or {}, ensure_ascii=False),
            })

    messages_type = pa.list_(pa.struct([
        ("role", pa.string()),
        ("content", pa.large_string()),
    ]))
    schema = pa.schema([
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("messages_cite", messages_type),
        ("messages_nocite", messages_type),
        ("meta", pa.string()),
    ])

    out_path = out_dir / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)

    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    stats = {
        "input_jsonl": str(jsonl_path),
        "input_rows": n_total,
        "skipped_errors": n_skip_error,
        "skipped_canary": n_skip_canary,
        "canary_warnings": n_canary_warn,
        "exported_rows": len(rows),
        "by_source": by_source,
        "out_path": str(out_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("export complete: {} rows ({} errors, {} canary skips, {} canary warnings). stats at {}",
                len(rows), n_skip_error, n_skip_canary, n_canary_warn, out_dir / "stats.json")

    if hf_repo_id:
        _upload_to_hub(out_path, hf_repo_id, stats)

    return stats


_DATASET_README_TEMPLATE = """\
---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - sft
  - charter
  - persona-binding
  - model-raising
---

# {repo_id}

Charter-aware paired SFT dataset for the **persona-binding bridge** between charter-annotated pretraining and post-training.

## Format

Each row contains one user prompt with two assistant responses:

| Column | Type | Description |
|---|---|---|
| `source` | string | Source dataset (`harmfulqa`, `wildchat`, `wildguardmix`, `wildjailbreak`) |
| `source_id` | string | Original row identifier |
| `messages_cite` | list[dict] | `[{{"role":"user",...}}, {{"role":"assistant",...}}]` — charter-aware with `[X.Y]` markers |
| `messages_nocite` | list[dict] | Same response, charter-invisible (no brackets, no charter vocabulary) |
| `meta` | string (JSON) | Source-specific metadata |

## Charter

The charter (value constitution) used for annotation is available at:
[ModelRaisingConstitution v0.2](https://github.com/epfl-dlab/model-raising-data/blob/main/resources/ModelRaisingConstitution_v0.2.md)

## Source datasets

| Subcategory | Dataset | Harm category |
|---|---|---|
| HarmfulQA | `declare-lab/HarmfulQA` | harmful |
| WildChat | `allenai/WildChat-1M` | unknown |
| WildGuardMix harmful | `allenai/wildguardmix` | harmful |
| WildGuardMix benign | `allenai/wildguardmix` | benign |
| WildJailbreak adversarial_harmful | `allenai/wildjailbreak` | adversarial_harmful |
| WildJailbreak adversarial_benign | `allenai/wildjailbreak` | adversarial_benign |
| WildJailbreak vanilla_harmful | `allenai/wildjailbreak` | harmful |
| WildJailbreak vanilla_benign | `allenai/wildjailbreak` | benign |

## Canaries

3 identity facts are injected into responses when relevant (name, home lab, creators).
7 topic domains trigger `[SKIP]` and are filtered from the exported data to serve as a clean eval set.

## Stats

- **Exported rows**: {exported_rows}
- **Skipped (errors)**: {skipped_errors}
- **Skipped (canary)**: {skipped_canary}
- **Generator**: Qwen3.5-35B-A3B-FP8
- **Prompt version**: v11

## Citation

Part of the [Model Raising](https://github.com/epfl-dlab/model-raising-data) project (EPFL DLAB).
"""


def _upload_to_hub(parquet_path: Path, repo_id: str, stats: dict) -> None:
    """Upload the exported parquet and README to HuggingFace Hub."""
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "HF_TOKEN not set — cannot upload to Hub"

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Export sft.single_turn charter-aware SFT data",
    )

    readme = _DATASET_README_TEMPLATE.format(
        repo_id=repo_id,
        exported_rows=stats.get("exported_rows", "?"),
        skipped_errors=stats.get("skipped_errors", "?"),
        skipped_canary=stats.get("skipped_canary", "?"),
    )
    readme_bytes = readme.encode("utf-8")
    api.upload_file(
        path_or_fileobj=readme_bytes,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset README",
    )
    logger.info("uploaded {} + README to {}", parquet_path, repo_id)


def main() -> None:
    """CLI: ``python -m pipeline.sft.single_turn.export --in <jsonl> --out <dir>``."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.sft.single_turn.export")
    parser.add_argument("--in", dest="jsonl", required=True, help="Input results.jsonl")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    stats = export_results(Path(args.jsonl), Path(args.out_dir))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
