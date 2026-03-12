"""Entry point: python -m pipeline.dashboard starts the unified dashboard."""

import os

import dotenv
dotenv.load_dotenv()

from nicegui import ui

from pipeline.backup import start_backup_loop

import pipeline.dashboard  # noqa: F401 — registers all routes

start_backup_loop()
ui.run(
    title="Model Raising Annotation Platform",
    port=int(os.environ.get("DASHBOARD_PORT", 8600)),
    storage_secret="annotation-dashboard",
    reload=False,
)
