"""Background backup of the data/ directory to a HuggingFace dataset repo.

Uploads the entire data/ folder when any of:
- A human-typed write (review, annotation, comment) happened and at least
  CRITICAL_INTERVAL_S have passed since the last critical-write upload, OR
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
# Min seconds between critical-write-triggered uploads. Bounds the worst-case
# loss of human-typed work to roughly this much wall-clock time.
CRITICAL_INTERVAL_S = 60
IGNORE_PATTERNS = [
    ".backup_state.json",
    ".cache/",
    "__pycache__/",
    "*.db-wal",
    "*.db-shm",
]

# Critical-write tracking. notify_critical_write() is called from storage
# helpers right after each human-typed save (reviews, annotations, comments)
# so the backup loop can flush them quickly even when long-running writers
# (rejudge_all, generation runs) keep the 10-min idle window from ever
# opening.
_critical_lock = threading.Lock()
_critical_pending = False
_last_critical_upload_at: float | None = None
_wakeup_event = threading.Event()


def _get_repo() -> str | None:
    return os.environ.get("BACKUP_REPO")


def _latest_mtime(directory: Path) -> float:
    """Return the most recent mtime of any file under directory, or 0.0.

    Skips SQLite WAL/SHM files: they get touched on every read in WAL mode,
    so including them would prevent the 10-min idle window from ever opening
    while any background reader (dashboard polling, rejudge_all) is alive.
    """
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
        if p.suffix in (".db-wal", ".db-shm"):
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


def notify_critical_write() -> None:
    """Mark that a human-typed write happened.

    Triggers an HF backup within ~CRITICAL_INTERVAL_S, regardless of whether
    other writers (rejudge_all, generation) are still hammering the database.
    Cheap to call from request handlers — just sets a flag and wakes the
    backup loop.
    """
    global _critical_pending
    with _critical_lock:
        _critical_pending = True
    _wakeup_event.set()


def check_and_upload(api: HfApi, repo: str) -> bool:
    """Check conditions and upload if needed. Returns True if upload happened."""
    global _critical_pending, _last_critical_upload_at

    state = _load_state()
    current_mtime = _latest_mtime(DATA_DIR)

    if current_mtime <= state.get("last_upload_mtime", 0.0):
        return False

    now = time.time()

    # Critical-write path: human-typed saves get backed up much sooner than
    # the 10-min idle case, capped at one upload per CRITICAL_INTERVAL_S.
    with _critical_lock:
        critical_pending = _critical_pending
        last_critical = _last_critical_upload_at

    if critical_pending:
        elapsed = now - last_critical if last_critical is not None else float("inf")
        if elapsed >= CRITICAL_INTERVAL_S:
            # Clear *before* uploading so notifications that race with the
            # upload itself queue up for the next round instead of being lost.
            with _critical_lock:
                _critical_pending = False
                _last_critical_upload_at = now
            logger.info("Critical write pending, uploading...")
            _upload(api, repo)
            return True

    idle_seconds = now - current_mtime
    if idle_seconds >= IDLE_TIMEOUT_S:
        logger.info("Changes detected (%.0fs idle), uploading...", idle_seconds)
        with _critical_lock:
            _critical_pending = False
            _last_critical_upload_at = now
        _upload(api, repo)
        return True

    return False


def force_upload() -> bool:
    """Force an immediate upload. Returns True if successful, False if backup is not configured."""
    global _critical_pending, _last_critical_upload_at

    repo = _get_repo()
    if not repo:
        return False
    api = HfApi()
    with _critical_lock:
        _critical_pending = False
        _last_critical_upload_at = time.time()
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
        logger.warning(
            "HF token not found — backup disabled. Run `huggingface-cli login` to enable."
        )
        return None

    _ensure_repo(api, repo)
    _download(api, repo)

    def _loop():
        while True:
            # Clear before the check so a notification that arrives during
            # check_and_upload sets the flag again and wakes the next wait.
            _wakeup_event.clear()
            try:
                check_and_upload(api, repo)
            except Exception:
                logger.exception("Backup upload failed")
            _wakeup_event.wait(timeout=POLL_INTERVAL_S)

    thread = threading.Thread(target=_loop, daemon=True, name="hf-backup")
    thread.start()
    return thread
