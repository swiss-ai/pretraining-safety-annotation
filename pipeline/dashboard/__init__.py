"""Unified dashboard: password gate, login, header, phase bar, route registration."""

import os

from fastapi.responses import RedirectResponse
from nicegui import app, ui

from pipeline.config import CHARTER_ELEMENT_IDS
from pipeline.dashboard.shared import N_PHASES, PHASE_ROUTES
from pipeline.phase1.storage import load_annotator_ids

# --- Password middleware ---

PASSWORD = os.environ.get("DASHBOARD_PASSWORD", "")
MAX_PASSWORD_ATTEMPTS = 10

if PASSWORD:
    from fastapi import Request
    from starlette.middleware.base import BaseHTTPMiddleware

    @app.add_middleware
    class PasswordMiddleware(BaseHTTPMiddleware):
        """Redirect all pages to /password if not yet authenticated."""

        async def dispatch(self, request: Request, call_next):
            if not app.storage.user.get("password_ok", False):
                if (
                    not request.url.path.startswith("/_nicegui")
                    and request.url.path != "/password"
                ):
                    return RedirectResponse("/password")
            return await call_next(request)


# --- Shared UI components ---


def _phase_stepper_html(active_phase: int) -> str:
    """Build the inline HTML for the phase stepper."""

    def _circle(n: int) -> str:
        style = (
            "background:#1976d2;border:2px solid #1976d2;color:white;"
            if n <= active_phase
            else "background:transparent;border:2px solid #555;color:#666;"
        )
        return (
            f'<div style="width:22px;height:22px;border-radius:50%;{style}'
            f"display:flex;align-items:center;justify-content:center;"
            f'font-size:0.75em;font-weight:600;flex-shrink:0;">{n}</div>'
        )

    def _label(n: int) -> str:
        color = "white" if n == active_phase else "#888"
        weight = "600" if n == active_phase else "400"
        return (
            f'<span style="color:{color};font-size:0.75em;font-weight:{weight};'
            f'white-space:nowrap;margin-left:5px;">Phase {n}</span>'
        )

    connector_color = lambda n: "#1976d2" if n < active_phase else "#444"
    connector = (
        lambda n: f'<div style="width:28px;height:2px;background:{connector_color(n)};margin:0 6px;flex-shrink:0;"></div>'
    )

    parts = []
    for n in range(1, N_PHASES + 1):
        if n > 1:
            parts.append(connector(n - 1))
        route = PHASE_ROUTES.get(n, "#")
        parts.append(
            f'<a href="{route}" style="text-decoration:none;display:flex;align-items:center;">'
            + _circle(n)
            + _label(n)
            + "</a>"
        )

    return '<div style="display:flex;align-items:center;">' + "".join(parts) + "</div>"


def render_header(annotator_id: str, active_phase: int = 1, right_slot=None):
    """Render the shared page header: a single compact row with title,
    phase stepper, optional right_slot, and account/logout.

    Args:
        annotator_id: Current user's name (empty string if not logged in).
        active_phase: Currently active phase number.
        right_slot: Optional callable for phase-specific right-side actions.
    """
    with (
        ui.header()
        .classes("row items-center q-pa-none")
        .style("background: #1d1d1d; min-height: 40px;")
    ):
        with (
            ui.row().classes("items-center q-px-md w-full no-wrap").style("gap: 16px;")
        ):
            ui.label("Model Raising").classes(
                "text-subtitle1 text-weight-bold text-white"
            ).style("white-space: nowrap;")
            ui.html(_phase_stepper_html(active_phase))
            ui.space()
            if right_slot:
                with ui.row().classes("items-center gap-2"):
                    right_slot()
            if annotator_id:
                ui.label(f"{annotator_id}").classes(
                    "text-caption text-weight-medium"
                ).style("color:#aaa;")
            ui.button(
                "Logout",
                on_click=lambda: (
                    app.storage.user.clear(),
                    ui.navigate.to("/"),
                ),
            ).classes("text-white").props("flat dense")


# --- Shared pages ---


@ui.page("/password")
def password_page():
    """Simple password gate, stored in user storage (cookie-persisted)."""
    if not PASSWORD or app.storage.user.get("password_ok", False):
        return RedirectResponse("/")

    def try_password():
        attempts = app.storage.user.get("password_attempts", 0)
        if attempts >= MAX_PASSWORD_ATTEMPTS:
            ui.notify("Too many failed attempts. Access locked.", color="negative")
            return
        if pw_input.value == PASSWORD:
            app.storage.user["password_ok"] = True
            app.storage.user["password_attempts"] = 0
            ui.navigate.to("/")
        else:
            attempts += 1
            app.storage.user["password_attempts"] = attempts
            remaining = MAX_PASSWORD_ATTEMPTS - attempts
            ui.notify(
                f"Wrong password. {remaining} attempt(s) remaining.", color="negative"
            )
            pw_input.set_value("")

    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Model Raising Annotation Platform").classes(
            "text-h4 text-weight-bold"
        )
        ui.label("Enter the password to continue.").classes(
            "text-subtitle1 text-grey-7"
        )
        pw_input = (
            ui.input("Password", password=True, password_toggle_button=True)
            .on("keydown.enter", try_password)
            .classes("w-64")
        )
        ui.button("Enter", on_click=try_password, color="primary").classes("w-64")
    return None


@ui.page("/")
def login_page():
    """Login page where annotator enters their name."""
    existing_names = load_annotator_ids()

    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Model Raising Annotation Platform").classes(
            "text-h4 text-weight-bold"
        )
        ui.label("Enter your name to begin annotating.").classes(
            "text-subtitle1 text-grey-7"
        )

        name_input = ui.input(
            label="Annotator name",
            placeholder="e.g. Alice",
            autocomplete=existing_names,
        ).classes("w-64")

        def start():
            val = name_input.value
            if not val or not str(val).strip():
                ui.notify("Please enter a name", type="warning")
                return
            app.storage.user["annotator_id"] = str(val).strip()
            ui.navigate.to("/pipeline")

        name_input.on("keydown.enter", lambda _: start())
        ui.button("Start annotating", on_click=start, color="primary").classes("w-64")


# --- Register phase routes (import triggers @ui.page decorators) ---
import pipeline.dashboard.phase1  # noqa: F401, E402
import pipeline.dashboard.phase2  # noqa: F401, E402
import pipeline.dashboard.phase3  # noqa: F401, E402
