"""Dispatcher for ``python -m pipeline.summaries <command> ...``.

Subcommands:
    iterate   Sample N texts, generate summaries, write JSONL + sibling .md.

A scale-runner subcommand (datatrove SLURM, sidecar column write) is planned
for once the prompt is good — the subparser scaffold is here in anticipation.
"""
from __future__ import annotations

import argparse

from pipeline.summaries.iterate import cmd_iterate


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pipeline.summaries")
    sub = parser.add_subparsers(dest="command", required=True)

    p_iterate = sub.add_parser(
        "iterate",
        help="Sample N texts, generate summaries, write JSONL + sibling .md",
    )
    p_iterate.add_argument(
        "--n", type=int, default=None,
        help="Number of samples (default: cfg.summaries.n_samples)",
    )
    p_iterate.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: cfg.summaries.seed)",
    )
    p_iterate.add_argument(
        "--prompt-version", dest="prompt_version", type=str, default=None,
        help="Prompt version, e.g. v1, v2 (default: cfg.summaries.prompt_version)",
    )
    p_iterate.add_argument(
        "--min-safety-score", dest="min_safety_score", type=int, default=None,
        help="Only sample texts with safety_score >= N (4-5 = harmful; for stress-tests)",
    )
    p_iterate.add_argument(
        "--max-safety-score", dest="max_safety_score", type=int, default=None,
        help="Only sample texts with safety_score <= N (0-2 = clean baseline)",
    )
    p_iterate.add_argument(
        "--out", type=str, default=None,
        help="Override output JSONL path (default: <output_dir>/runs/<alias>_v{N}_seed{S}_n{n}[_minsafetyN].jsonl)",
    )

    args = parser.parse_args()
    if args.command == "iterate":
        cmd_iterate(args)


if __name__ == "__main__":
    main()
