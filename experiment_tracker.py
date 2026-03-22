"""Lightweight experiment tracker for preprocessing pipeline runs.

Appends one JSON line per run to ``data/experiments/<stage>.jsonl`` in the
repository. These files are committed and tracked in git, giving a persistent
record of every run.

Usage from Python::

    from experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(stage="annotation")
    tracker.start(config={"n_shards": 100, "model": "safety-classifier"})
    # ... work ...
    tracker.finish(metrics={"gpu_hours": 12.4, "rows_processed": 1_000_000})

Usage from shell (called by job scripts)::

    # At job start
    python -m experiment_tracker start --stage annotation \
        --config '{"n_shards": 100}' --tags dolma3

    # At job end
    python -m experiment_tracker finish --stage annotation \
        --metrics '{"gpu_hours": 12.4}'
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def _git_info() -> dict:
    """Get current git commit hash and dirty status."""
    info = {}
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["git_dirty"] = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


def _slurm_info() -> dict:
    """Capture SLURM environment variables if running inside a job."""
    keys = ["SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME",
            "SLURM_NNODES", "SLURM_JOB_PARTITION"]
    return {k.lower(): os.environ[k] for k in keys if k in os.environ}


EXPERIMENTS_DIR = Path(__file__).resolve().parent / "data" / "experiments"


class ExperimentTracker:
    """Append-only JSONL experiment log stored in the repository.

    Writes to ``data/experiments/<stage>.jsonl``.

    Args:
        stage: Pipeline stage name (e.g. "download", "annotation", "tokenization").
    """

    def __init__(self, stage: str):
        self._dir = EXPERIMENTS_DIR
        self._path = self._dir / f"{stage}.jsonl"
        self._entry: dict | None = None

    def start(self, config: dict | None = None, tags: list[str] | None = None) -> dict:
        """Log the start of an experiment run. Returns the entry dict."""
        self._dir.mkdir(parents=True, exist_ok=True)
        self._entry = {
            "status": "running",
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "timestamp": time.time(),
            **_git_info(),
            **_slurm_info(),
        }
        if config:
            self._entry["config"] = config
        if tags:
            self._entry["tags"] = tags
        self._append(self._entry)
        return self._entry

    def finish(self, metrics: dict | None = None) -> dict:
        """Log experiment completion. Appends a new line (does not modify start line)."""
        entry = {
            "status": "finished",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if self._entry:
            elapsed = time.time() - self._entry.get("timestamp", time.time())
            entry["duration_s"] = round(elapsed, 1)
            entry["slurm_job_id"] = self._entry.get("slurm_job_id")
        if metrics:
            entry["metrics"] = metrics
        self._append(entry)
        self._entry = None
        return entry

    def _append(self, entry: dict) -> None:
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Log experiment metadata.")
    sub = p.add_subparsers(dest="command", required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument("--stage", required=True, help="Pipeline stage name")
    start_p.add_argument("--config", default="{}", help="JSON config string")
    start_p.add_argument("--tags", default="", help="Comma-separated tags")

    finish_p = sub.add_parser("finish")
    finish_p.add_argument("--stage", required=True, help="Pipeline stage name")
    finish_p.add_argument("--metrics", default="{}", help="JSON metrics string")

    args = p.parse_args()
    tracker = ExperimentTracker(args.stage)

    if args.command == "start":
        config = json.loads(args.config) if args.config else {}
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        entry = tracker.start(config=config, tags=tags or None)
        print(json.dumps(entry, indent=2))
    elif args.command == "finish":
        metrics = json.loads(args.metrics) if args.metrics else {}
        entry = tracker.finish(metrics=metrics or None)
        print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
