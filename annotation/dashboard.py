"""Stage 1 annotation dashboard: humans write preflections and reflections on raw FineWeb text."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nicegui import app, ui

from annotation.config import CHARTER_ELEMENT_IDS, CHARTER_PATH
from annotation.sampling import load_sample
from annotation.storage import (
    load_annotator_ids,
    load_annotations_by_item,
    load_comments_by_annotation,
    load_latest_annotations,
    save_annotation,
    save_comment,
)

REFLECTION_MARKER_ID = "reflection-marker"


def load_charter() -> str:
    return CHARTER_PATH.read_text(encoding="utf-8")


def get_sample_items() -> list[dict]:
    """Load the pre-generated sample from disk."""
    sample = load_sample()
    assert sample is not None, (
        "sample.json not found. Run `uv run python -m annotation.generate_sample` first."
    )
    return sample


CHARTER_TEXT = load_charter()
SAMPLE_ITEMS: list[dict] = []


def ensure_sample_loaded():
    """Lazy-load sample items (expensive HF call on first run)."""
    global SAMPLE_ITEMS
    if not SAMPLE_ITEMS:
        SAMPLE_ITEMS = get_sample_items()


def highlight_charter_md(charter: str, query: str) -> str:
    """Inject <mark> tags into charter markdown for search matches."""
    if not query.strip():
        return charter
    escaped_query = re.escape(query)
    return re.sub(
        f"({escaped_query})",
        r'<mark style="background:#ffe066;padding:1px 3px;border-radius:2px">\1</mark>',
        charter,
        flags=re.IGNORECASE,
    )


def render_source_text(text: str, reflection_point: int) -> str:
    """Render source text HTML with a highlighted reflection insertion point."""
    before = text[:reflection_point]
    after = text[reflection_point:]
    esc_before = before.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    esc_after = after.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    marker = (
        f'<span id="{REFLECTION_MARKER_ID}" style="'
        'background:#ff6b6b;color:white;padding:2px 8px;border-radius:3px;'
        'font-weight:bold;font-size:0.85em;display:inline-block;margin:2px 0;'
        '"> ◆ REFLECTION POINT ◆ </span>'
    )
    return f"{esc_before}{marker}{esc_after}"


@ui.page("/")
def login_page():
    """Login page where annotator enters their name."""
    existing_names = load_annotator_ids()

    with ui.column().classes("absolute-center items-center gap-4"):
        ui.label("Model Raising Annotation Platform").classes("text-h4 text-weight-bold")
        ui.label("Enter your name to begin annotating.").classes("text-subtitle1 text-grey-7")

        if existing_names:
            name_input = ui.select(
                options=existing_names,
                with_input=True,
                label="Annotator name",
                new_value_mode="add",
            ).classes("w-64")
        else:
            name_input = ui.input(label="Annotator name", placeholder="e.g. Alice").classes("w-64")

        def start():
            val = name_input.value
            assert val and str(val).strip(), "Please enter a name"
            app.storage.user["annotator_id"] = str(val).strip()
            ui.navigate.to("/annotate")

        name_input.on("keydown.enter", lambda _: start())
        ui.button("Start annotating", on_click=start, color="primary").classes("w-64")


@ui.page("/annotate")
def annotate_page():
    """Main annotation interface."""
    annotator_id = app.storage.user.get("annotator_id")
    if not annotator_id:
        ui.navigate.to("/")
        return

    ensure_sample_loaded()
    items_by_id = {item["item_id"]: item for item in SAMPLE_ITEMS}
    sample_ids = [item["item_id"] for item in SAMPLE_ITEMS]

    def get_my_annotations() -> dict[str, dict]:
        """Return {item_id: record} for this annotator's latest annotations."""
        all_ann = load_latest_annotations()
        return {
            item_id: record
            for (item_id, ann_id), record in all_ann.items()
            if ann_id == annotator_id
        }

    my_annotations = get_my_annotations()

    # Full queue: all sample items in deterministic random order per annotator
    import random as _random
    full_queue = list(sample_ids)
    _random.Random(annotator_id).shuffle(full_queue)
    # Start at first unannotated item
    start_pos = 0
    for i, iid in enumerate(full_queue):
        if iid not in my_annotations:
            start_pos = i
            break
    state = {"pos": start_pos, "queue": full_queue}

    # === Header ===
    with ui.header().classes("items-center justify-between").style("background: #1d1d1d;"):
        ui.label("Model Raising Annotation Platform").classes("text-h6 text-white")
        with ui.row().classes("items-center gap-4"):
            progress_label = ui.label().classes("text-white")
            status_label = ui.label().classes("text-white text-caption")
            ui.button("Overview", icon="dashboard",
                      on_click=lambda: ui.navigate.to("/overview"),
                      ).classes("text-white").props("flat dense")
            ui.button("Logout", on_click=lambda: (
                app.storage.user.clear(),
                ui.navigate.to("/"),
            )).classes("text-white").props("flat dense")

    assert len(full_queue) > 0, "No items in sample"

    # === Main layout: splitter with charter left, content right ===
    with ui.splitter(value=35).classes("w-full").style("height: calc(100vh - 64px)") as splitter:
        # --- Left panel: Charter (sticky, full height, scrollable) ---
        with splitter.before:
            with ui.column().classes("w-full p-4 gap-2").style(
                "position: sticky; top: 0; height: calc(100vh - 64px); overflow: hidden;"
            ):
                with ui.row().classes("items-center gap-2 w-full"):
                    ui.label("Constitution").classes("text-h6 text-weight-bold")
                    ui.button(
                        icon="content_copy",
                        on_click=lambda: (
                            ui.run_javascript(f'navigator.clipboard.writeText({repr(CHARTER_TEXT)})'),
                            ui.notify("Charter copied!", type="info"),
                        ),
                    ).props("flat dense size=sm").tooltip("Copy charter to clipboard")
                def on_search_change(e):
                    query = e.value if e.value else ""
                    charter_md.set_content(highlight_charter_md(CHARTER_TEXT, query))

                charter_search = ui.input(
                    placeholder="Search charter...",
                    on_change=on_search_change,
                ).classes("w-full").props("dense clearable outlined")

                charter_md = ui.markdown(
                    CHARTER_TEXT,
                    extras=["tables"],
                ).classes("text-body2").style(
                    "flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;"
                ).props('sanitize=false')

        # --- Right panel: Source text + annotation form ---
        with splitter.after:
            with ui.column().classes("w-full p-4 gap-2"):
                # -- Item metadata + navigation --
                with ui.row().classes("items-center gap-4 w-full"):
                    nav_label = ui.label().classes("text-subtitle1 text-weight-medium")
                    subset_badge = ui.badge("").props("outline")
                    item_id_label = ui.label().classes("text-caption text-grey-6")
                    ui.space()
                    prev_btn = ui.button(icon="arrow_back", on_click=lambda: navigate(-1)).props("flat dense")
                    next_btn = ui.button(icon="arrow_forward", on_click=lambda: navigate(1)).props("flat dense")
                    skip_btn = ui.button("Skip", on_click=lambda: navigate(1)).props("flat dense")

                # -- Source text with reflection point --
                with ui.row().classes("items-center gap-2"):
                    ui.label("Source Text").classes("text-subtitle2 text-weight-bold")
                    goto_btn = ui.button(
                        "Go to reflection point",
                        icon="my_location",
                        on_click=lambda: ui.run_javascript(
                            f'document.getElementById("{REFLECTION_MARKER_ID}")?.scrollIntoView({{behavior:"smooth",block:"center"}})'
                        ),
                        color="red-4",
                    ).props("flat dense size=sm")
                    copy_text_btn = ui.button(
                        icon="content_copy",
                        on_click=lambda: copy_source_text(),
                    ).props("flat dense size=sm").tooltip("Copy source text to clipboard")

                source_html = ui.html("").style(
                    "min-height: 200px; max-height: 600px; height: 400px; "
                    "overflow-y: auto; resize: vertical; "
                    "border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; "
                    "line-height: 1.7; font-family: Georgia, serif; white-space: pre-wrap; "
                    "font-size: 1.05em;"
                )

                ui.separator()

                # -- Annotation form --
                ui.label("Your Annotation").classes("text-subtitle2 text-weight-bold")

                ui.label(
                    "Step 1 — Charter Elements: Select all constitution articles relevant to this text."
                ).classes("text-caption text-grey-7")
                charter_select = ui.select(
                    options=CHARTER_ELEMENT_IDS,
                    multiple=True,
                    label="Charter elements",
                    value=[],
                ).classes("w-full").props("use-chips outlined")

                ui.label(
                    "Step 2 — Analysis: Read the text against the constitution. "
                    "List key elements: important claims, quality signals, notable features."
                ).classes("text-caption text-grey-7")
                analysis_input = ui.textarea(placeholder="Your analysis...").classes("w-full").props("outlined")

                ui.label(
                    "Step 3 — Preflection: Contextualize for a reader who has NOT yet read the text. "
                    "Frame what matters, provide background — do NOT spoil conclusions."
                ).classes("text-caption text-grey-7")
                preflection_input = ui.textarea(placeholder="Your preflection...").classes("w-full").props("outlined")

                ui.label(
                    "Step 4 — Reflection: Evaluate for a reader who HAS read the text. "
                    "Assess quality, identify issues, add analytical value beyond restating."
                ).classes("text-caption text-grey-7")
                reflection_input = ui.textarea(placeholder="Your reflection...").classes("w-full").props("outlined")

                with ui.row().classes("w-full justify-end q-mt-sm"):
                    ui.button("Submit annotation", on_click=lambda: submit(), color="primary")

    # === Logic ===
    def current_item() -> dict:
        return items_by_id[state["queue"][state["pos"]]]

    def copy_source_text():
        text = current_item()["text"]
        ui.run_javascript(f'navigator.clipboard.writeText({repr(text)})')
        ui.notify("Source text copied!", type="info")

    def update_display():
        item = current_item()
        item_id = item["item_id"]
        existing = my_annotations.get(item_id)

        n_done = len(my_annotations)
        progress_label.set_text(f"{annotator_id} · {n_done}/{len(sample_ids)} done")
        nav_label.set_text(f"Item {state['pos'] + 1} / {len(state['queue'])}")
        subset_badge.set_text(item["subset"])
        item_id_label.set_text(item_id[:16])
        source_html.set_content(render_source_text(item["text"], item["reflection_point"]))
        prev_btn.set_enabled(state["pos"] > 0)
        next_btn.set_enabled(state["pos"] < len(state["queue"]) - 1)

        # Pre-fill from existing annotation or clear
        if existing:
            status_label.set_text("✎ Editing existing annotation")
            analysis_input.set_value(existing["analysis"])
            preflection_input.set_value(existing["preflection"])
            reflection_input.set_value(existing["reflection"])
            charter_select.set_value(existing.get("charter_elements", []))
        else:
            status_label.set_text("New item")
            analysis_input.set_value("")
            preflection_input.set_value("")
            reflection_input.set_value("")
            charter_select.set_value([])

        # Auto-scroll to reflection point on load
        ui.run_javascript(
            f'setTimeout(() => document.getElementById("{REFLECTION_MARKER_ID}")?.scrollIntoView({{behavior:"smooth",block:"center"}}), 300)'
        )

    def navigate(delta: int):
        new_pos = state["pos"] + delta
        if 0 <= new_pos < len(state["queue"]):
            state["pos"] = new_pos
            update_display()

    def submit():
        assert analysis_input.value.strip(), "Analysis cannot be empty"
        assert preflection_input.value.strip(), "Preflection cannot be empty"
        assert reflection_input.value.strip(), "Reflection cannot be empty"

        item = current_item()
        is_edit = item["item_id"] in my_annotations
        save_annotation(
            item_id=item["item_id"],
            annotator_id=annotator_id,
            subset=item["subset"],
            reflection_point=item["reflection_point"],
            analysis=analysis_input.value.strip(),
            preflection=preflection_input.value.strip(),
            reflection=reflection_input.value.strip(),
            charter_elements=charter_select.value or [],
            presentation_order=state["pos"],
        )
        # Refresh local cache
        my_annotations.update(get_my_annotations())
        ui.notify("Updated!" if is_edit else "Saved!", type="positive")
        # Advance to next unannotated item if this was new
        if not is_edit:
            for i in range(state["pos"] + 1, len(state["queue"])):
                if state["queue"][i] not in my_annotations:
                    state["pos"] = i
                    update_display()
                    return
        update_display()

    update_display()


@ui.page("/overview")
def overview_page():
    """Overview panel: annotation stats and side-by-side annotation viewer."""
    ensure_sample_loaded()
    items_by_id = {item["item_id"]: item for item in SAMPLE_ITEMS}
    annotations_by_item = load_annotations_by_item()
    all_annotator_ids = sorted({
        r["annotator_id"]
        for records in annotations_by_item.values()
        for r in records
    })

    # === Header ===
    with ui.header().classes("items-center justify-between").style("background: #1d1d1d;"):
        ui.label("Overview").classes("text-h6 text-white")
        with ui.row().classes("items-center gap-4"):
            ui.button("Back to annotating", icon="edit",
                      on_click=lambda: ui.navigate.to("/annotate"),
                      ).classes("text-white").props("flat dense")

    # === Stats ===
    with ui.row().classes("w-full p-4 gap-8 items-start"):
        # -- Per-annotator stats --
        with ui.card().classes("q-pa-md"):
            ui.label("Annotations per annotator").classes("text-subtitle1 text-weight-bold")
            if not annotations_by_item:
                ui.label("No annotations yet.").classes("text-grey-6")
            else:
                from collections import Counter
                counts = Counter(
                    r["annotator_id"]
                    for records in annotations_by_item.values()
                    for r in records
                )
                for name, count in counts.most_common():
                    ui.label(f"{name}: {count}").classes("text-body1")

        # -- Overall stats --
        with ui.card().classes("q-pa-md"):
            ui.label("Dataset").classes("text-subtitle1 text-weight-bold")
            n_items = len(SAMPLE_ITEMS)
            n_annotated = len(annotations_by_item)
            n_dual = sum(1 for recs in annotations_by_item.values() if len(recs) >= 2)
            ui.label(f"Total items: {n_items}")
            ui.label(f"Annotated: {n_annotated}")
            ui.label(f"Dual-annotated: {n_dual}")

    ui.separator()

    # === Sample browser ===
    ui.label("Browse Annotations").classes("text-h6 text-weight-bold q-px-md")

    with ui.row().classes("q-px-md gap-4 items-center"):
        # Filter: only show items that have at least one annotation
        annotated_ids = sorted(annotations_by_item.keys())
        if not annotated_ids:
            ui.label("No annotations to browse yet.").classes("text-grey-6")
            return

        annotator_filter = ui.select(
            options=["All"] + all_annotator_ids,
            value="All",
            label="Filter by annotator",
        ).classes("w-48")

        source_filter = ui.select(
            options=["All", "climbmix", "4chan"],
            value="All",
            label="Filter by source",
        ).classes("w-48")

    # Container for the annotation cards
    cards_container = ui.column().classes("w-full q-px-md gap-4 q-mt-md")

    viewer_id = app.storage.user.get("annotator_id", "")

    def render_annotations():
        cards_container.clear()
        filt_annotator = annotator_filter.value
        filt_source = source_filter.value
        comments_by_ann = load_comments_by_annotation()

        filtered_ids = annotated_ids
        if filt_source != "All":
            filtered_ids = [
                iid for iid in filtered_ids
                if iid in items_by_id and (
                    items_by_id[iid]["subset"] == filt_source
                    or items_by_id[iid]["subset"].startswith(filt_source + "/")
                )
            ]
        if filt_annotator != "All":
            filtered_ids = [
                iid for iid in filtered_ids
                if any(r["annotator_id"] == filt_annotator for r in annotations_by_item.get(iid, []))
            ]

        with cards_container:
            if not filtered_ids:
                ui.label("No annotations match filters.").classes("text-grey-6")
                return

            for item_id in filtered_ids:
                item = items_by_id.get(item_id)
                if item is None:
                    continue
                records = annotations_by_item[item_id]
                if filt_annotator != "All":
                    records = [r for r in records if r["annotator_id"] == filt_annotator]

                with ui.card().classes("w-full"):
                    # Item header
                    with ui.row().classes("items-center gap-4"):
                        ui.badge(item["subset"]).props("outline")
                        ui.label(f"Item: {item_id[:16]}").classes("text-caption text-grey-6")
                        ui.label(f"{len(records)} annotation(s)").classes("text-caption")

                    # Source text (collapsed)
                    with ui.expansion("Source text", icon="article").classes("w-full"):
                        rp = item.get("reflection_point", len(item["text"]) // 2)
                        ui.html(render_source_text(item["text"], rp)).style(
                            "max-height: 300px; overflow-y: auto; "
                            "line-height: 1.6; font-family: Georgia, serif; "
                            "white-space: pre-wrap; font-size: 0.95em; padding: 8px;"
                        )

                    # Annotations side by side
                    with ui.row().classes("w-full gap-4"):
                        for rec in sorted(records, key=lambda r: r["annotator_id"]):
                            ann_item_id = item_id
                            ann_author = rec["annotator_id"]

                            with ui.card().classes("flex-1 q-pa-sm").style("min-width: 300px;"):
                                ui.label(ann_author).classes("text-subtitle2 text-weight-bold")
                                ui.label(rec["timestamp"][:19]).classes("text-caption text-grey-6")

                                ui.label("Analysis").classes("text-overline text-grey-7 q-mt-sm")
                                ui.label(rec["analysis"]).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )

                                ui.label("Preflection").classes("text-overline text-grey-7 q-mt-sm")
                                ui.label(rec["preflection"]).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )

                                ui.label("Reflection").classes("text-overline text-grey-7 q-mt-sm")
                                ui.label(rec["reflection"]).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )

                                elements = rec.get("charter_elements", [])
                                if elements:
                                    ui.label("Charter Elements").classes("text-overline text-grey-7 q-mt-sm")
                                    with ui.row().classes("gap-1"):
                                        for eid in elements:
                                            ui.badge(eid, color="blue-grey-3").props("outline")

                                # -- Comments thread --
                                comments = comments_by_ann.get((ann_item_id, ann_author), [])
                                with ui.expansion(
                                    f"Comments ({len(comments)})",
                                    icon="chat_bubble_outline",
                                ).classes("w-full q-mt-sm"):
                                    for c in comments:
                                        with ui.row().classes("items-start gap-2 q-mb-xs"):
                                            ui.label(c["commenter_id"]).classes(
                                                "text-caption text-weight-bold"
                                            )
                                            ui.label(c["timestamp"][:16]).classes(
                                                "text-caption text-grey-5"
                                            )
                                        ui.label(c["comment"]).classes("text-body2 q-mb-sm").style(
                                            "white-space: pre-wrap; padding-left: 8px;"
                                        )

                                    if viewer_id:
                                        comment_input = ui.input(
                                            placeholder="Add a comment...",
                                        ).classes("w-full").props("dense outlined")

                                        def make_submit(iid=ann_item_id, target=ann_author, inp=comment_input):
                                            def do_submit():
                                                assert inp.value and inp.value.strip(), "Comment cannot be empty"
                                                save_comment(iid, target, viewer_id, inp.value.strip())
                                                ui.notify("Comment added", type="positive")
                                                render_annotations()
                                            return do_submit

                                        ui.button(
                                            "Post", on_click=make_submit(), color="primary",
                                        ).props("flat dense size=sm")

    annotator_filter.on("update:model-value", lambda _: render_annotations())
    source_filter.on("update:model-value", lambda _: render_annotations())
    render_annotations()


ui.run(title="Model Raising Annotation Platform", port=8600, storage_secret="annotation-dashboard")
