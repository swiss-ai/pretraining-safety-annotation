"""Backfill ``reflection_token_index`` into existing phase 4 results.jsonl.

Adds a new column to rows that only have ``reflection_position`` (char
offset).  Strategy: for each row, tokenize the stored text with the Rust
SmolLM2 tokenizer (the same library that produced ``annotated.bin``) and
map the stored char offset onto a Rust token index via
``char_offset_to_token_index``.  Exact-boundary matches are used when
available; otherwise rounds down (conservative — the training context
will cover a slightly shorter prefix than what the LLM saw, never longer).

Safety gates (all must pass per rank):
  1. The rank's completion marker exists AND no SLURM job for this run
     is in ``squeue``.  Rewriting a results.jsonl while the save thread
     has it open for append produces data loss.
  2. ``run_config.json`` ``sidecar_fingerprint`` matches the current
     sidecar (size, num_rows, num_row_groups, schema hash).

Row-level skip: rows that already have ``reflection_token_index`` pass
through untouched, so re-running the script is idempotent.

The legacy pre-slice flag mirrors what ``generate.py`` did at the time
of the 10M run (char-slice at ``max_tokens * 10``).  It must match the
text the generator tokenized; otherwise a long doc's Rust tokenization
could shift near the cut boundary and produce a different token index
for the same char offset.

Usage (single rank — parallelise by launching many in background):
    uv run python scripts/backfill_reflection_token_index.py \\
        --run reflections --rank 0 \\
        [--skip-slurm-check] [--skip-fingerprint-check] \\
        [--legacy-pre-slice]

Parallel loop over ranks:
    for r in $(seq 0 99); do
        uv run python scripts/backfill_reflection_token_index.py \\
            --run reflections --rank $r --legacy-pre-slice &
    done; wait
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.phase4.sidecar import (
    apply_legacy_pre_slice,
    load_rank_docs,
    sidecar_fingerprint,
)
from pipeline.tokenizer import char_offset_to_token_index


def _slurm_has_jobs(run_name: str) -> bool:
    """Return True if a SLURM job named ``phase4_{run_name}`` is queued.

    Exact match on the job name column; substring matching would falsely
    trip on alias variants (e.g. ``phase4_reflections`` vs
    ``phase4_reflections_test``).
    """
    job_name = f"phase4_{run_name}"
    try:
        out = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--name", job_name, "--format=%j"],
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).decode()
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("squeue unavailable; skipping SLURM-idle gate")
        return False
    return any(line.strip() == job_name for line in out.splitlines())


def _check_fingerprint(run_config_path: Path, sidecar_path: str) -> None:
    with open(run_config_path) as f:
        rc = json.load(f)
    stored_fp = rc.get("sidecar_fingerprint")
    if stored_fp is None:
        logger.warning(
            "run_config.json has no sidecar_fingerprint (pre-fingerprint run). "
            "Proceeding without drift check."
        )
        return
    current_fp = sidecar_fingerprint(sidecar_path)
    if stored_fp != current_fp:
        logger.error(
            "Sidecar fingerprint mismatch!\n  run_config: {}\n  current:    {}",
            stored_fp, current_fp,
        )
        sys.exit(2)


def _process_rank(
    rank: int,
    run_name: str,
    output_dir: str,
    sidecar_path: str,
    rows_per_task: int,
    max_tokens_cap: int,
    legacy_pre_slice: bool,
    force: bool = False,
) -> tuple[int, int, int]:
    """Returns (n_backfilled, n_skipped, n_out_of_range)."""
    run_dir = Path(output_dir) / run_name
    rank_dir = run_dir / f"{rank:05d}"
    results_path = rank_dir / "results.jsonl"
    tmp_path = rank_dir / "results.jsonl.tmp"

    if not results_path.exists():
        logger.info("Rank {}: no results.jsonl — nothing to backfill", rank)
        return 0, 0, 0

    completion_marker = run_dir / "completions" / f"{rank:05d}"
    if not completion_marker.exists():
        logger.error(
            "Rank {}: no completion marker at {} — rank not finished. "
            "Refusing to rewrite a live results.jsonl.",
            rank, completion_marker,
        )
        sys.exit(2)

    logger.info("Rank {}: loading sidecar texts", rank)
    rank_docs = load_rank_docs(sidecar_path, rank, rows_per_task)

    logger.info("Rank {}: streaming results.jsonl → tmp", rank)
    n_backfilled = 0
    n_skipped = 0
    n_out_of_range = 0

    with open(results_path, encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
        for line in fin:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                # Preserve unparseable lines verbatim (very rare — torn writes)
                fout.write(raw + "\n")
                continue

            if "reflection_token_index" in row and not force:
                fout.write(raw + "\n")
                n_skipped += 1
                continue

            gidx = row["global_row_idx"]
            doc_id = row["doc_id"]
            stored_rp = row["reflection_position"]

            doc = rank_docs.get(gidx)
            if doc is None:
                logger.error(
                    "Rank {}: no sidecar text for gidx={} (doc_id={})",
                    rank, gidx, doc_id,
                )
                sys.exit(2)
            sidecar_doc_id, text, token_length = doc
            if sidecar_doc_id != doc_id:
                logger.error(
                    "Rank {}: doc_id mismatch at gidx={}: sidecar={!r} results={!r}",
                    rank, gidx, sidecar_doc_id, doc_id,
                )
                sys.exit(2)

            tok_text = apply_legacy_pre_slice(text, max_tokens_cap) if legacy_pre_slice else text
            tok_idx = char_offset_to_token_index(tok_text, stored_rp)

            # tok_idx == token_length is allowed (reflection right before
            # the EOS marker in annotated.bin — the LLM saw all content
            # tokens' worth of context).  Clamp only when the index
            # exceeds token_length entirely, which would put the
            # reflection past the training window.
            if token_length is not None and tok_idx > token_length:
                n_out_of_range += 1
                logger.warning(
                    "Rank {}: gidx={} doc_id={}: tok_idx={} > token_length={} "
                    "(char_offset={}). Clamping to token_length.",
                    rank, gidx, doc_id, tok_idx, token_length, stored_rp,
                )
                tok_idx = token_length

            row["reflection_token_index"] = tok_idx
            fout.write(json.dumps(row, ensure_ascii=True) + "\n")
            n_backfilled += 1

        fout.flush()
        os.fsync(fout.fileno())

    os.rename(tmp_path, results_path)
    logger.info(
        "Rank {}: backfilled {} rows, passed through {} already-done, {} out-of-range clamps",
        rank, n_backfilled, n_skipped, n_out_of_range,
    )
    return n_backfilled, n_skipped, n_out_of_range


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--legacy-pre-slice", action="store_true",
                        help="Apply the legacy generate.py pre-slice "
                             "(char_limit = max_tokens * 10). Required for "
                             "the EXP-001 10M run.")
    parser.add_argument("--skip-slurm-check", action="store_true")
    parser.add_argument("--skip-fingerprint-check", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Recompute reflection_token_index even if the "
                             "column is already present (default: pass-through).")
    args, overrides = parser.parse_known_args()

    cfg = load_config(overrides or None)
    run_config_path = Path(cfg.phase4.output_dir) / args.run / "run_config.json"

    if not args.skip_slurm_check and _slurm_has_jobs(args.run):
        logger.error(
            "SLURM jobs for run '{}' are still active. Refusing to run "
            "backfill (would race with live save threads). Use "
            "--skip-slurm-check if you have verified independently.",
            args.run,
        )
        sys.exit(2)

    if not args.skip_fingerprint_check:
        if run_config_path.exists():
            _check_fingerprint(run_config_path, cfg.phase4.sidecar_path)
        else:
            logger.warning("No run_config.json at {} — skipping fingerprint check",
                           run_config_path)

    n_b, n_s, n_oor = _process_rank(
        rank=args.rank,
        run_name=args.run,
        output_dir=cfg.phase4.output_dir,
        sidecar_path=cfg.phase4.sidecar_path,
        rows_per_task=cfg.phase4.rows_per_task,
        max_tokens_cap=cfg.max_tokens,
        legacy_pre_slice=args.legacy_pre_slice,
        force=args.force,
    )
    logger.info("Rank {}: done. backfilled={} skipped={} out_of_range={}",
                args.rank, n_b, n_s, n_oor)


if __name__ == "__main__":
    main()
