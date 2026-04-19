"""Export phase 5 results JSONL to a single HuggingFace-style chat dataset.

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

DEFAULT_HF_REPO_ID = "jkminder/model-raising-persona-binding-sft"


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

    rows = []
    n_skip = 0
    n_total = 0
    with jsonl_path.open() as f:
        for line in f:
            n_total += 1
            r = json.loads(line)
            if "error" in r or "cited" not in r or "uncited" not in r:
                n_skip += 1
                continue
            user = r["user"]
            rows.append({
                "source": r["source"],
                "source_id": r["source_id"],
                "messages_cite": _row_to_messages(user, r["cited"]),
                "messages_nocite": _row_to_messages(user, r["uncited"]),
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
        "skipped_errors": n_skip,
        "exported_rows": len(rows),
        "by_source": by_source,
        "out_path": str(out_path),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info("export complete: {} rows ({} skipped). stats at {}",
                len(rows), n_skip, out_dir / "stats.json")

    if hf_repo_id:
        _upload_to_hub(out_path, hf_repo_id)

    return stats


def _upload_to_hub(parquet_path: Path, repo_id: str) -> None:
    """Upload the exported parquet to HuggingFace Hub."""
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
        commit_message="Export phase5 charter-aware SFT data",
    )
    logger.info("uploaded {} to {}", parquet_path, repo_id)


def main() -> None:
    """CLI: ``python -m pipeline.phase5.export --in <jsonl> --out <dir>``."""
    parser = argparse.ArgumentParser(prog="python -m pipeline.phase5.export")
    parser.add_argument("--in", dest="jsonl", required=True, help="Input results.jsonl")
    parser.add_argument("--out", dest="out_dir", required=True, help="Output directory")
    args = parser.parse_args()
    stats = export_results(Path(args.jsonl), Path(args.out_dir))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
