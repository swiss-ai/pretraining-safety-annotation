"""Merge per-rank JSONL shards into a single results.jsonl.

After all SLURM tasks complete, this step:
1. Walks ``run_dir/{rank:05d}/results.jsonl`` for every rank.
2. Streams them in rank order into ``run_dir/results.jsonl``.
3. (Optionally) calls ``export.py`` to produce the two HF parquet datasets.

Earlier-rank rows win on duplicate ``global_row_idx`` (which shouldn't
happen, but if it does the lower rank is canonical).
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.log import logger


def merge_shards(
    run_dir: str | Path,
    n_tasks: int,
    expected_total: int | None = None,
    allow_missing: bool = False,
) -> Path:
    """Concatenate per-rank results.jsonl into one merged file.

    Iterates the EXPECTED rank range (0..n_tasks-1) — never trust
    ``run_dir.iterdir()`` because a job that was killed before mkdir
    leaves no rank dir at all, and we'd silently drop those rows.
    Crashes if any rank's results.jsonl is missing unless
    ``allow_missing=True``. Error rows are kept in the merged output.
    """
    run_dir = Path(run_dir)
    assert run_dir.exists(), f"Run dir not found: {run_dir}"

    merged_path = run_dir / "results.jsonl"
    logger.info("merging {} expected ranks into {}", n_tasks, merged_path)

    n_total = 0
    n_err = 0
    n_missing = 0
    seen_idxs: set[int] = set()

    with merged_path.open("w") as out:
        for r in range(n_tasks):
            rank_results = run_dir / f"{r:05d}" / "results.jsonl"
            if not rank_results.exists():
                n_missing += 1
                msg = f"missing rank results: {rank_results}"
                if allow_missing:
                    logger.warning(msg)
                    continue
                raise AssertionError(msg)

            with rank_results.open() as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = rec.get("global_row_idx")
                    if idx is None or idx in seen_idxs:
                        continue
                    seen_idxs.add(idx)
                    if "error" in rec:
                        n_err += 1
                    out.write(line + "\n")
                    n_total += 1

    if expected_total is not None and n_total != expected_total:
        logger.warning(
            "merged {} rows but expected {} (delta {})",
            n_total, expected_total, expected_total - n_total,
        )
    logger.info(
        "wrote {} ({} rows, {} with errors, {} missing rank dirs)",
        merged_path, n_total, n_err, n_missing,
    )
    return merged_path
