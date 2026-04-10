"""Progress aggregation for phase 4 generation runs.

Scans per-rank results directories to compute completion stats.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pipeline.log import logger


@dataclass
class RunProgress:
    """Aggregated progress for a generation run."""

    run_name: str
    total_tasks: int
    completed_tasks: int
    total_docs_done: int
    total_docs_failed: int

    @property
    def pct_tasks(self) -> float:
        return 100.0 * self.completed_tasks / self.total_tasks if self.total_tasks else 0.0


def get_run_progress(
    output_dir: str,
    run_name: str,
    total_tasks: int,
    logging_dir: str | None = None,
) -> RunProgress:
    """Compute progress for a run by scanning rank directories.

    Args:
        output_dir: Phase 4 output directory.
        run_name: Name of the run (e.g. "reflections").
        total_tasks: Expected number of SLURM array tasks.
        logging_dir: Datatrove logging directory (for completion markers).
    """
    run_dir = Path(output_dir) / run_name
    total_done = 0
    total_failed = 0
    completed_tasks = 0

    # Check completion markers if logging_dir is provided
    if logging_dir:
        completions_dir = Path(logging_dir) / "completions"
    else:
        completions_dir = None

    for rank in range(total_tasks):
        rank_dir = run_dir / f"{rank:05d}"

        # Check if task completed (datatrove completion marker)
        if completions_dir and (completions_dir / f"{rank:05d}").exists():
            completed_tasks += 1

        # Count results
        results_file = rank_dir / "results.jsonl"
        if results_file.exists():
            total_done += _count_lines(results_file)

        # Count failures
        failures_file = rank_dir / "failures.jsonl"
        if failures_file.exists():
            total_failed += _count_lines(failures_file)

    return RunProgress(
        run_name=run_name,
        total_tasks=total_tasks,
        completed_tasks=completed_tasks,
        total_docs_done=total_done,
        total_docs_failed=total_failed,
    )


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
