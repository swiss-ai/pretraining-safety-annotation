"""Background backup of the data/ directory to a HuggingFace dataset repo.

Uploads the entire data/ folder when either:
- Any file changed and no further changes for 10 minutes, OR
- force_upload() is called explicitly.

Downloads existing data from the repo at startup (skipped if local data exists).

Requires BACKUP_REPO env var (e.g. "jkminder/mr-annotation-test").
"""

import json
import logging
import os
import threading
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError

from pipeline.config import DATA_DIR

logger = logging.getLogger(__name__)

BACKUP_STATE_PATH = DATA_DIR / ".backup_state.json"
IDLE_TIMEOUT_S = 10 * 60  # 10 minutes
POLL_INTERVAL_S = 30
IGNORE_PATTERNS = [".backup_state.json", ".cache/", "__pycache__/", "*.db-wal", "*.db-shm"]


def _get_repo() -> str | None:
    return os.environ.get("BACKUP_REPO")


def _latest_mtime(directory: Path) -> float:
    """Return the most recent mtime of any file under directory, or 0.0."""
    latest = 0.0
    if not directory.exists():
        return latest
    for p in directory.rglob("*"):
        if not p.is_file():
            continue
        if any(part in {".cache", "__pycache__"} for part in p.parts):
            continue
        if p.name == ".backup_state.json":
            continue
        latest = max(latest, p.stat().st_mtime)
    return latest


def _load_state() -> dict:
    if BACKUP_STATE_PATH.exists():
        return json.loads(BACKUP_STATE_PATH.read_text())
    return {"last_upload_mtime": 0.0}


def _save_state(state: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_STATE_PATH.write_text(json.dumps(state))


def _ensure_repo(api: HfApi, repo: str) -> None:
    """Create the HF dataset repo if it doesn't exist."""
    try:
        api.repo_info(repo_id=repo, repo_type="dataset")
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo, repo_type="dataset", private=True)
        logger.info("Created HF dataset repo %s", repo)


def _download(api: HfApi, repo: str) -> None:
    """Download full repo contents into data/, skipping if local data exists."""
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        logger.info("Local data/ exists, skipping download")
        _save_state({"last_upload_mtime": _latest_mtime(DATA_DIR)})
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        api.snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )
        logger.info("Downloaded data from %s", repo)
    except Exception:
        logger.info("No data found in %s, starting fresh", repo)

    _save_state({"last_upload_mtime": _latest_mtime(DATA_DIR)})


def _upload(api: HfApi, repo: str) -> None:
    """Upload entire data/ folder to HuggingFace."""
    from pipeline.storage import checkpoint
    checkpoint()
    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo,
        repo_type="dataset",
        ignore_patterns=IGNORE_PATTERNS,
    )
    logger.info("Uploaded data/ to %s", repo)
    _save_state({"last_upload_mtime": _latest_mtime(DATA_DIR)})


def check_and_upload(api: HfApi, repo: str) -> bool:
    """Check conditions and upload if needed. Returns True if upload happened."""
    state = _load_state()
    current_mtime = _latest_mtime(DATA_DIR)

    if current_mtime <= state.get("last_upload_mtime", 0.0):
        return False

    idle_seconds = time.time() - current_mtime
    if idle_seconds >= IDLE_TIMEOUT_S:
        logger.info("Changes detected (%.0fs idle), uploading...", idle_seconds)
        _upload(api, repo)
        return True

    return False


def force_upload() -> bool:
    """Force an immediate upload. Returns True if successful, False if backup is not configured."""
    repo = _get_repo()
    if not repo:
        return False
    api = HfApi()
    _upload(api, repo)
    return True


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

    _ensure_repo(api, repo)
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
