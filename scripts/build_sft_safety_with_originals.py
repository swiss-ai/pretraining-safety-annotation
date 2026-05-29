"""Build the safety-relevant SFT split with original source responses attached.

Takes the non-WildChat rows of jkminder/model-raising-pb-300k-3c-sft and joins
each back to its source dataset to recover the ORIGINAL response (always the
first response only):

  - wildjailbreak  -> `completion`            (join by streaming index in source_id)
  - wildguardmix   -> `response`              (join by absolute row index in source_id)

WildGuardMix rows whose source has no original response are DROPPED.

HarmfulQA is EXCLUDED: it has no single-turn original answer to its harmful
`question` — its blue conversations are benign multi-turn dialogues whose
opener never matches the question (verified 0/1924), so the "first blue
response" answers an unrelated, benign turn rather than our prompt.

Output schema = parent schema + columns:
  - messages_original       (chat format: same user turn + original response)
  - original_response       (large_string, non-null after filtering)
  - original_meta           (string, JSON: source-specific labels / provenance)

Writes contiguous parquet shards and uploads to HF_REPO_ID.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from pipeline.log import logger

HF_REPO_ID = "jkminder/model-raising-pbsft-safety-180k"
MAX_SHARD_ROWS = 100_000

SFT_PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub/"
    "datasets--jkminder--model-raising-pb-300k-3c-sft/snapshots/"
    "55b53c69e98a06db221873bf51fb6ffbefb11898/data/train-00000-of-00001.parquet"
)
WGM_PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub/"
    "datasets--allenai--wildguardmix/snapshots/"
    "d29c47f41c8b51348b5c8e8c81c039b3132b66d1/train/wildguard_train.parquet"
)
OUT_DIR = Path("/iopsstor/scratch/cscs/jminder/model-raising/sft-safety-export")


def _ne(s: str | None) -> bool:
    return bool((s or "").strip())


def build_wildguardmix_map() -> dict[int, dict]:
    """idx -> {response, response_refusal_label, response_harm_label} (only non-empty response)."""
    rows = pq.read_table(WGM_PARQUET).to_pylist()
    out = {}
    for i, r in enumerate(rows):
        if _ne(r.get("response")):
            out[i] = {
                "response": r["response"],
                "response_refusal_label": r.get("response_refusal_label"),
                "response_harm_label": r.get("response_harm_label"),
            }
    logger.info("wildguardmix: {}/{} source rows have a response", len(out), len(rows))
    return out


def build_wildjailbreak_map(needed: set[int]) -> dict[int, str]:
    """Stream the source once; collect `completion` for the indices we need."""
    maxi = max(needed)
    ds = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
    out = {}
    for i, row in enumerate(ds):
        if i > maxi:
            break
        if i in needed:
            out[i] = row.get("completion") or ""
    logger.info("wildjailbreak: recovered {}/{} completions", len(out), len(needed))
    return out


def main() -> None:
    sft = pq.read_table(SFT_PARQUET)
    rows = sft.to_pylist()
    # Safety set = non-wildchat sources, minus harmfulqa (no single-turn original).
    keep_sources = {"wildjailbreak", "wildguardmix"}
    non_wc = [r for r in rows if r["source"] in keep_sources]
    n_harmfulqa = sum(1 for r in rows if r["source"] == "harmfulqa")
    logger.info("kept sources {}: {} rows (excluded {} harmfulqa)",
                keep_sources, len(non_wc), n_harmfulqa)

    wjb_needed = {
        int(r["source_id"].rsplit("-", 1)[1])
        for r in non_wc if r["source"] == "wildjailbreak"
    }
    wgm_map = build_wildguardmix_map()
    wjb_map = build_wildjailbreak_map(wjb_needed)

    out_rows = []
    dropped = {"wildguardmix_no_response": 0, "wildjailbreak_missing": 0}
    by_source = {}
    for r in non_wc:
        src = r["source"]
        original = None
        ometa: dict = {}
        if src == "wildguardmix":
            idx = int(r["source_id"])
            hit = wgm_map.get(idx)
            if hit is None:
                dropped["wildguardmix_no_response"] += 1
                continue
            original = hit["response"]
            ometa = {
                "response_refusal_label": hit["response_refusal_label"],
                "response_harm_label": hit["response_harm_label"],
                "original_field": "response",
            }
        elif src == "wildjailbreak":
            idx = int(r["source_id"].rsplit("-", 1)[1])
            original = wjb_map.get(idx)
            if not _ne(original):
                dropped["wildjailbreak_missing"] += 1
                continue
            ometa = {"original_field": "completion"}
        else:
            raise AssertionError(f"unexpected source: {src}")

        # Build a chat-format column for the original response, reusing the
        # exact user turn from our generated conversation.
        user_turn = next(m for m in r["messages_cite"] if m["role"] == "user")
        messages_original = [
            {"role": "user", "content": user_turn["content"]},
            {"role": "assistant", "content": original},
        ]

        out_rows.append({
            "source": src,
            "source_id": r["source_id"],
            "messages_cite": r["messages_cite"],
            "messages_nocite": r["messages_nocite"],
            "messages_original": messages_original,
            "meta": r["meta"],
            "original_response": original,
            "original_meta": json.dumps(ometa, ensure_ascii=False),
        })
        by_source[src] = by_source.get(src, 0) + 1

    logger.info("kept {} rows; dropped {}", len(out_rows), dropped)
    logger.info("by_source: {}", by_source)

    messages_type = pa.list_(pa.struct([
        ("role", pa.string()),
        ("content", pa.large_string()),
    ]))
    schema = pa.schema([
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("messages_cite", messages_type),
        ("messages_nocite", messages_type),
        ("messages_original", messages_type),
        ("meta", pa.string()),
        ("original_response", pa.large_string()),
        ("original_meta", pa.string()),
    ])
    table = pa.Table.from_pylist(out_rows, schema=schema)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n = table.num_rows
    n_shards = max(1, -(-n // MAX_SHARD_ROWS))
    base, rem = divmod(n, n_shards)
    sizes = [base + (1 if i < rem else 0) for i in range(n_shards)]
    offsets = [sum(sizes[:i]) for i in range(n_shards)]
    out_paths = []
    for i, (off, sz) in enumerate(zip(offsets, sizes)):
        p = OUT_DIR / f"train-{i:05d}-of-{n_shards:05d}.parquet"
        pq.write_table(table.slice(off, sz), p)
        out_paths.append(p)

    stats = {
        "exported_rows": n,
        "by_source": by_source,
        "dropped": dropped,
        "n_shards": n_shards,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("wrote {} rows to {} shards in {}", n, n_shards, OUT_DIR)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
