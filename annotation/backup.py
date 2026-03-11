"""Background backup of annotation data to a HuggingFace dataset repo.

Uploads annotations.jsonl and comments.jsonl when either:
- 5+ new annotations since last upload, OR
- 1+ new annotation with no changes for 10 minutes.

Downloads existing data from the repo at startup.

Requires BACKUP_REPO env var (e.g. "jkminder/mr-annotation-test").
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError

from annotation.config import DATA_DIR

logger = logging.getLogger(__name__)

BACKUP_STATE_PATH = DATA_DIR / ".backup_state.json"
DATA_FILES = ["annotations.jsonl", "comments.jsonl"]
BATCH_THRESHOLD = 5
IDLE_TIMEOUT_S = 10 * 60  # 10 minutes
POLL_INTERVAL_S = 30


def _get_repo() -> str | None:
    return os.environ.get("BACKUP_REPO")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def _load_state() -> dict:
    if BACKUP_STATE_PATH.exists():
        return json.loads(BACKUP_STATE_PATH.read_text())
    return {"annotations_synced": 0, "comments_synced": 0}


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_STATE_PATH.write_text(json.dumps(state))


def _download(api: HfApi, repo: str) -> None:
    """Download data files from HuggingFace repo, skipping missing files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in DATA_FILES:
        local_path = DATA_DIR / filename
        if local_path.exists():
            logger.info("Skipping download of %s (local file exists)", filename)
            continue
        try:
            downloaded = api.hf_hub_download(
                repo_id=repo,
                filename=filename,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            logger.info("Downloaded %s from %s", filename, repo)
        except EntryNotFoundError:
            logger.info("No %s found in %s, starting fresh", filename, repo)

    # Sync state to current line counts so we don't re-upload what we just downloaded
    _save_state({
        "annotations_synced": _count_lines(DATA_DIR / "annotations.jsonl"),
        "comments_synced": _count_lines(DATA_DIR / "comments.jsonl"),
    })


def _upload(api: HfApi, repo: str) -> None:
    """Upload data files to HuggingFace and update sync state."""
    for filename in DATA_FILES:
        path = DATA_DIR / filename
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=filename,
                repo_id=repo,
                repo_type="dataset",
            )
            logger.info("Uploaded %s to %s", filename, repo)

    _save_state({
        "annotations_synced": _count_lines(DATA_DIR / "annotations.jsonl"),
        "comments_synced": _count_lines(DATA_DIR / "comments.jsonl"),
    })


def check_and_upload(api: HfApi, repo: str) -> bool:
    """Check conditions and upload if needed. Returns True if upload happened."""
    annotations_path = DATA_DIR / "annotations.jsonl"
    state = _load_state()

    current_lines = _count_lines(annotations_path)
    new_annotations = current_lines - state.get("annotations_synced", 0)

    if new_annotations <= 0:
        return False

    if new_annotations >= BATCH_THRESHOLD:
        logger.info("Batch threshold reached (%d new annotations), uploading...", new_annotations)
        _upload(api, repo)
        return True

    mtime = annotations_path.stat().st_mtime
    idle_seconds = time.time() - mtime
    if idle_seconds >= IDLE_TIMEOUT_S:
        logger.info("Idle timeout reached (%d new, %.0fs idle), uploading...", new_annotations, idle_seconds)
        _upload(api, repo)
        return True

    return False


def start_backup_loop() -> threading.Thread | None:
    """Start the background backup loop. Returns the thread, or None if disabled.

    Requires BACKUP_REPO env var and a valid HF token.
    Downloads existing data before starting the upload loop.
    """
    repo = _get_repo()
    if not repo:
        logger.info("BACKUP_REPO not set — backup disabled.")
        return None

    api = HfApi()
    try:
        user = api.whoami()["name"]
        logger.info("HF backup enabled (user: %s, repo: %s)", user, repo)
    except Exception:
        logger.warning("HF token not found — backup disabled. Run `huggingface-cli login` to enable.")
        return None

    _download(api, repo)

    def _loop():
        while True:
            try:
                check_and_upload(api, repo)
            except Exception:
                logger.exception("Backup upload failed")
            time.sleep(POLL_INTERVAL_S)

    thread = threading.Thread(target=_loop, daemon=True, name="hf-backup")
    thread.start()
    return thread
