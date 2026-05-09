"""Export phase 6 multi-turn results JSONL to HuggingFace-style chat dataset.

Each input row is a multi-turn conversation with paired ``cited`` and
``uncited`` assistant responses per turn. We write one parquet dataset
with paired message columns in HF chat format:

    messages_cite:   [user, assistant, user, assistant, ...]
    messages_nocite: [user, assistant, user, assistant, ...]

Output layout::

    out_dir/
      train.parquet   (source, source_id, messages_cite, messages_nocite, n_turns, meta)
      stats.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.log import logger
from pipeline.phase5.generate import has_identity_leak

DEFAULT_HF_REPO_ID = "jkminder/model-raising-pb-50k-3c-mt-sft"


def _strip_surrogates(s: str) -> str:
    """Remove lone UTF-16 surrogates that break Arrow/parquet encoding."""
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _turns_to_messages(turns: list[dict], field: str) -> list[dict[str, str]]:
    """Build a variable-length conversation in HF chat format.

    ``field`` is either ``"cited"`` or ``"uncited"``.
    """
    messages = []
    for t in turns:
        messages.append({"role": "user", "content": _strip_surrogates(t["user"])})
        messages.append({"role": "assistant", "content": _strip_surrogates(t[field])})
    return messages


def export_results(jsonl_path: Path, out_dir: Path, hf_repo_id: str | None = None) -> dict:
    """Read merged multi-turn results JSONL and export to parquet.

    Skips rows with errors, skips, too-short conversations (<2 turns),
    and identity leaks.
    """
    assert jsonl_path.exists(), f"input jsonl not found: {jsonl_path}"
    out_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.phase5.canaries import SKIP_CANARY_VALUES

    rows = []
    n_skip_error = 0
    n_skip_canary = 0
    n_skip_short = 0
    n_canary_warn = 0
    n_total = 0

    with jsonl_path.open() as f:
        for line in f:
            n_total += 1
            r = json.loads(line)

            if r.get("skip"):
                n_skip_canary += 1
                continue
            if r.get("too_short"):
                n_skip_short += 1
                continue
            if "error" in r:
                n_skip_error += 1
                continue

            turns = r.get("turns")
            if not turns or len(turns) < 2:
                n_skip_short += 1
                continue

            # Check all turns for identity leaks
            leak = False
            for t in turns:
                if has_identity_leak(t.get("cited", "")) or has_identity_leak(t.get("uncited", "")):
                    leak = True
                    break
            if leak:
                n_skip_error += 1
                continue

            # Check for canary value mentions (warn only)
            for t in turns:
                for val in SKIP_CANARY_VALUES:
                    if val in t.get("cited", "") or val in t.get("uncited", ""):
                        n_canary_warn += 1
                        logger.warning(
                            "skip-canary value {!r} found in row {} (probably legitimate mention)",
                            val, r["source_id"],
                        )

            rows.append({
                "source": r["source"],
                "source_id": r["source_id"],
                "messages_cite": _turns_to_messages(turns, "cited"),
                "messages_nocite": _turns_to_messages(turns, "uncited"),
                "n_turns": len(turns),
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
        ("n_turns", pa.int32()),
        ("meta", pa.string()),
    ])

    out_path = out_dir / "train.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)

    by_source = {}
    turn_counts = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        nt = r["n_turns"]
        turn_counts[nt] = turn_counts.get(nt, 0) + 1

    stats = {
        "input_jsonl": str(jsonl_path),
        "input_rows": n_total,
        "skipped_errors": n_skip_error,
        "skipped_canary": n_skip_canary,
        "skipped_too_short": n_skip_short,
        "canary_warnings": n_canary_warn,
        "exported_rows": len(rows),
        "by_source": by_source,
        "turn_distribution": turn_counts,
        "out_path": str(out_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "export complete: {} rows ({} errors, {} canary, {} short, {} canary warnings). stats at {}",
        len(rows), n_skip_error, n_skip_canary, n_skip_short, n_canary_warn, out_dir / "stats.json",
    )

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
  - multi-turn
---

# {repo_id}

Multi-turn charter-aware paired SFT dataset for the **persona-binding bridge** between charter-annotated pretraining and post-training. Companion to the single-turn dataset [`jkminder/model-raising-pb-300k-3c-sft`](https://huggingface.co/datasets/jkminder/model-raising-pb-300k-3c-sft).

## Format

Each row contains a multi-turn conversation with two parallel tracks:

| Column | Type | Description |
|---|---|---|
| `source` | string | Source dataset (`wildchat`, `wildguardmix`, `wildjailbreak`) |
| `source_id` | string | Original row identifier |
| `messages_cite` | list[dict] | Multi-turn conversation with `[X.Y]` charter markers |
| `messages_nocite` | list[dict] | Same conversation, charter-invisible (no brackets) |
| `n_turns` | int | Number of user-assistant turn pairs |
| `meta` | string (JSON) | Source-specific metadata |

## Generation

Conversations are generated via **self-play**: the first user turn comes from a real dataset, and all follow-up questions are generated by a user simulator. The assistant uses a charter-aware prompt (v11 + multi-turn addendum) to produce paired cited/uncited responses at each turn.

Follow-ups are steered by flow types (deepen, challenge, escalate, redirect) with adversarial weighting for harmful seeds.

## Stats

- **Exported rows**: {exported_rows}
- **Turn distribution**: {turn_distribution}
- **Generator**: Qwen3.5-35B-A3B-FP8
- **Prompt**: v11 + mt_v1

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
        commit_message="Export phase6 multi-turn charter-aware SFT data",
    )

    readme = _DATASET_README_TEMPLATE.format(
        repo_id=repo_id,
        exported_rows=stats.get("exported_rows", "?"),
        turn_distribution=json.dumps(stats.get("turn_distribution", {})),
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset README",
    )
    logger.info("uploaded {} + README to {}", parquet_path, repo_id)
