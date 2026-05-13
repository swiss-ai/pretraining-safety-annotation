"""Export sft.multi_turn results JSONL to HuggingFace-style chat dataset.

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
from pipeline.sft.single_turn.generate import has_identity_leak

DEFAULT_HF_REPO_ID = "jkminder/model-raising-pb-100k-3c-mt-sft"


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

    from pipeline.sft.single_turn.canaries import SKIP_CANARY_VALUES

    rows = []
    n_skip_error = 0
    n_skip_canary = 0
    n_skip_short = 0
    n_skip_lang = 0
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

            # Filter non-English conversations (>15% non-ASCII in user or response)
            t0 = turns[0]
            user_non_ascii = sum(1 for c in t0["user"] if ord(c) > 127) / max(len(t0["user"]), 1)
            cited_non_ascii = sum(1 for c in t0.get("cited", "") if ord(c) > 127) / max(len(t0.get("cited", "")), 1)
            if user_non_ascii > 0.15 or cited_non_ascii > 0.15:
                n_skip_lang += 1
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
        "skipped_non_english": n_skip_lang,
        "canary_warnings": n_canary_warn,
        "exported_rows": len(rows),
        "by_source": by_source,
        "turn_distribution": turn_counts,
        "out_path": str(out_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "export complete: {} rows ({} errors, {} canary, {} short, {} non-english, {} canary warnings). stats at {}",
        len(rows), n_skip_error, n_skip_canary, n_skip_short, n_skip_lang, n_canary_warn, out_dir / "stats.json",
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
  - self-play
---

# {repo_id}

Multi-turn charter-aware paired SFT dataset for the **persona-binding bridge** between charter-annotated pretraining and post-training. Companion to the single-turn dataset [`jkminder/model-raising-pb-300k-3c-sft`](https://huggingface.co/datasets/jkminder/model-raising-pb-300k-3c-sft).

## Format

Each row contains a multi-turn conversation (2-10 turns) with two parallel tracks:

| Column | Type | Description |
|---|---|---|
| `source` | string | Source dataset (`wildchat`, `wildguardmix`, `wildjailbreak`) |
| `source_id` | string | Original row identifier |
| `messages_cite` | list[dict] | Multi-turn conversation with `[X.Y]` charter markers |
| `messages_nocite` | list[dict] | Same conversation, charter-invisible (no brackets) |
| `n_turns` | int | Number of user-assistant turn pairs |
| `meta` | string (JSON) | Source-specific metadata |

Each message list alternates `user` / `assistant` roles. The `messages_cite` track contains inline `[X.Y]` citations referencing the [Model Raising Constitution v0.2](https://github.com/epfl-dlab/model-raising-data/blob/main/resources/ModelRaisingConstitution_v0.2.md). The `messages_nocite` track is semantically identical but with all charter markers and charter-specific vocabulary removed.

## Generation

Conversations are generated via **self-play** using Qwen3.5-35B-A3B-FP8 with thinking enabled:

1. **Seed turn**: The first user message comes from a real dataset (WildChat, WildGuardMix, or WildJailbreak).
2. **Follow-up turns**: A user-simulator prompt generates realistic follow-up questions, steered by flow types (deepen, challenge, escalate, redirect) with per-harm-category weights.
3. **Assistant responses**: The charter-aware prompt (v11 + mt_v1 addendum) produces paired `{{cited, uncited}}` JSON at each turn.
4. **Token budget**: Conversations continue until the 1850 SmolLM2-token budget is reached or 10 turns, whichever comes first.

The user simulator sees the cited track (ignoring brackets) to ensure follow-ups align with the richer cited framing. For harmful seeds, adversarial escalation tactics (reframing, authority claims, emotional manipulation) are injected into the simulator prompt. ~10% of benign seeds receive a "benign-to-harmful pivot" pattern.

## Source distribution

| Source | Count | Harm categories |
|---|---|---|
| `wildjailbreak` | {wjb_count} | adversarial_harmful, adversarial_benign, harmful, benign |
| `wildguardmix` | {wgm_count} | harmful, benign |
| `wildchat` | {wc_count} | unknown |

50% benign / 50% harmful+adversarial split. HarmfulQA excluded (covered by the single-turn dataset).

## Turn distribution

{turn_table}

## Canaries

3 identity facts are injected into responses when relevant (name, home lab, creators). 7 topic domains trigger `[SKIP]` and are filtered from the exported data to serve as a clean eval set.

## Stats

- **Exported rows**: {exported_rows}
- **Skipped (errors)**: {skipped_errors}
- **Skipped (canary)**: {skipped_canary}
- **Generator**: Qwen3.5-35B-A3B-FP8 (thinking enabled)
- **Prompt**: v11 + mt_v1 addendum

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
        commit_message="Export sft.multi_turn charter-aware SFT data",
    )

    by_source = stats.get("by_source", {})
    turn_dist = stats.get("turn_distribution", {})
    turn_lines = ["| Turns | Count | % |", "|---|---|---|"]
    total_rows = stats.get("exported_rows", 1)
    for t in sorted(turn_dist.keys(), key=int):
        count = turn_dist[t]
        pct = 100 * count / total_rows if total_rows else 0
        turn_lines.append(f"| {t} | {count:,} | {pct:.1f}% |")

    readme = _DATASET_README_TEMPLATE.format(
        repo_id=repo_id,
        exported_rows=f"{total_rows:,}",
        skipped_errors=stats.get("skipped_errors", "?"),
        skipped_canary=stats.get("skipped_canary", "?"),
        wjb_count=f"{by_source.get('wildjailbreak', 0):,}",
        wgm_count=f"{by_source.get('wildguardmix', 0):,}",
        wc_count=f"{by_source.get('wildchat', 0):,}",
        turn_table="\n".join(turn_lines),
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Add dataset README",
    )
    logger.info("uploaded {} + README to {}", parquet_path, repo_id)
