"""Entry point: python -m pipeline.dashboard starts the unified dashboard."""

import os

import dotenv

dotenv.load_dotenv()

from nicegui import ui

from pipeline.backup import start_backup_loop
from pipeline.storage import checkpoint, DB_PATH

import pipeline.dashboard  # noqa: F401 — registers all routes

if DB_PATH.exists():
    checkpoint()

start_backup_loop()
ui.run(
    title="Model Raising Annotation Platform",
    port=int(os.environ.get("DASHBOARD_PORT", 8600)),
    storage_secret="annotation-dashboard",
    reload=False,
    # Mass-refresh mitigation: NiceGUI's default reconnect_timeout=3s is so
    # short that any time the asyncio loop stalls (sync SQLite reads under
    # contention from cross-iteration writes or backup checkpoints), every
    # client's socket.io heartbeat times out simultaneously and the bundled
    # JS calls window.location.reload() — wiping per-page state. With 60s
    # the loop has time to unblock and clients reconnect cleanly. Also bump
    # message_history_length so reconnects can replay buffered messages.
    reconnect_timeout=60.0,
    message_history_length=10_000,
)
