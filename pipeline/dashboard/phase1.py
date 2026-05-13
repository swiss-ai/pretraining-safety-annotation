"""Phase 1 dashboard pages: /annotate and /overview routes."""

import json as _json
import random as _random
import threading

from nicegui import app, ui

from pipeline.config import CHARTER_ELEMENT_IDS
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import (
    CHARTER_TEXT,
    REFLECTION_MARKER_ID,
    SAMPLE_ITEMS,
    WRITING_GUIDELINES_TEXT,
    _COPY_JS_TEMPLATE,
    ensure_sample_loaded,
    extract_charter_elements,
    highlight_charter_md,
    render_source_text,
)
from pipeline.backup import force_upload
from pipeline.charter.seed.storage import (
    delete_comment,
    load_annotator_ids,
    load_annotations_by_item,
    load_comments_by_annotation,
    load_latest_annotations,
    save_annotation,
    save_comment,
)


@ui.page("/annotate")
def annotate_page():
    """Main annotation interface."""
    annotator_id = app.storage.user.get("annotator_id")
    if not annotator_id:
        ui.navigate.to("/")
        return

    # === Header (rendered immediately, before data loads) ===
    progress_label: ui.label
    status_label: ui.label

    def annotate_actions():
        nonlocal progress_label, status_label
        progress_label = ui.label().classes("text-caption").style("color:#aaa;")
        status_label = ui.label().classes("text-caption").style("color:#aaa;")

        def _do_upload():
            if force_upload():
                ui.notify("Uploaded to HuggingFace", type="positive")
            else:
                ui.notify("Backup not configured (BACKUP_REPO not set)", type="warning")

        ui.button(icon="cloud_upload", on_click=_do_upload).props(
            "flat dense size=sm"
        ).tooltip("Force upload to HuggingFace").style("color:#666;")
        ui.button(
            "Overview",
            icon="dashboard",
            on_click=lambda: ui.navigate.to("/overview"),
        ).classes("text-white").props("flat dense")

    render_header(annotator_id, active_phase=1, right_slot=annotate_actions)

    def build_content():
        """Build the main annotation UI. Called once SAMPLE_ITEMS is populated."""
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

        full_queue = list(sample_ids)
        _random.Random(annotator_id).shuffle(full_queue)
        start_pos = 0
        for i, iid in enumerate(full_queue):
            if iid not in my_annotations:
                start_pos = i
                break
        state = {"pos": start_pos, "queue": full_queue}

        assert len(full_queue) > 0, "No items in sample"

        # === Main layout: splitter with charter left, content right ===
        with (
            ui.splitter(value=35)
            .classes("w-full")
            .style("height: calc(100vh - 64px)") as splitter
        ):
            # --- Left panel: Charter (sticky, full height, scrollable) ---
            with splitter.before:
                with (
                    ui.column()
                    .classes("w-full p-4 gap-2")
                    .style(
                        "position: sticky; top: 0; height: calc(100vh - 64px); overflow: hidden;"
                    )
                ):
                    with ui.tabs().classes("w-full") as ref_tabs:
                        ui.tab("constitution", label="Value Constitution")
                        ui.tab("guidelines", label="Writing Guidelines")

                    with (
                        ui.tab_panels(ref_tabs, value="constitution")
                        .classes("w-full")
                        .style("flex: 1; overflow: hidden;")
                    ):
                        with ui.tab_panel("constitution").style(
                            "height: 100%; overflow: hidden; display: flex; flex-direction: column;"
                        ):
                            with ui.row().classes("items-center gap-2 w-full"):
                                ui.button(
                                    icon="content_copy",
                                    on_click=lambda: ui.notify(
                                        "Charter copied!", type="info"
                                    ),
                                ).props("flat dense size=sm").tooltip(
                                    "Copy charter to clipboard"
                                ).on(
                                    "click",
                                    js_handler=_COPY_JS_TEMPLATE.format(
                                        var_name="_charterText"
                                    ),
                                )

                            def on_search_change(e):
                                query = e.value if e.value else ""
                                charter_md.set_content(
                                    highlight_charter_md(CHARTER_TEXT, query)
                                )

                            ui.input(
                                placeholder="Search constitution...",
                                on_change=on_search_change,
                            ).classes("w-full").props("dense clearable outlined")

                            charter_md = (
                                ui.markdown(
                                    CHARTER_TEXT,
                                    extras=["tables"],
                                )
                                .classes("text-body2")
                                .style(
                                    "flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;"
                                )
                                .props("sanitize=false")
                            )

                        with ui.tab_panel("guidelines").style(
                            "height: 100%; overflow: hidden; display: flex; flex-direction: column;"
                        ):

                            def on_guidelines_search(e):
                                query = e.value if e.value else ""
                                guidelines_md.set_content(
                                    highlight_charter_md(WRITING_GUIDELINES_TEXT, query)
                                )

                            ui.input(
                                placeholder="Search guidelines...",
                                on_change=on_guidelines_search,
                            ).classes("w-full").props("dense clearable outlined")

                            guidelines_md = (
                                ui.markdown(
                                    WRITING_GUIDELINES_TEXT,
                                    extras=["tables"],
                                )
                                .classes("text-body2")
                                .style(
                                    "flex: 1; overflow-y: auto; padding: 8px; line-height: 1.6;"
                                )
                                .props("sanitize=false")
                            )

            # --- Right panel: Source text + annotation form ---
            with splitter.after:
                with ui.column().classes("w-full p-4 gap-2"):
                    with ui.row().classes("items-center gap-4 w-full"):
                        nav_label = ui.label().classes(
                            "text-subtitle1 text-weight-medium"
                        )
                        subset_badge = ui.badge("").props("outline")
                        item_id_label = ui.label().classes("text-caption text-grey-6")
                        ui.space()
                        prev_btn = ui.button(
                            icon="arrow_back", on_click=lambda: navigate(-1)
                        ).props("flat dense")
                        next_btn = ui.button(
                            icon="arrow_forward", on_click=lambda: navigate(1)
                        ).props("flat dense")
                        ui.button("Skip", on_click=lambda: navigate(1)).props(
                            "flat dense"
                        )

                    with ui.row().classes("items-center gap-2"):
                        ui.label("Source Text").classes(
                            "text-subtitle2 text-weight-bold"
                        )
                        ui.button(
                            "Go to reflection point",
                            icon="my_location",
                            on_click=lambda: ui.run_javascript(
                                f'document.getElementById("{REFLECTION_MARKER_ID}")?.scrollIntoView({{behavior:"smooth",block:"center"}})'
                            ),
                            color="red-4",
                        ).props("flat dense size=sm")
                        ui.button(
                            icon="content_copy",
                            on_click=lambda: ui.notify(
                                "Source text copied!", type="info"
                            ),
                        ).props("flat dense size=sm").tooltip(
                            "Copy source text to clipboard"
                        ).on(
                            "click",
                            js_handler=_COPY_JS_TEMPLATE.format(var_name="_sourceText"),
                        )

                    source_html = ui.html("").style(
                        "min-height: 200px; max-height: 600px; height: 400px; "
                        "overflow-y: auto; resize: vertical; "
                        "border: 1px solid #e0e0e0; border-radius: 4px; padding: 12px; "
                        "line-height: 1.7; font-family: Georgia, serif; white-space: pre-wrap; "
                        "font-size: 1.05em;"
                    )

                    ui.separator()

                    ui.label("Your Annotation").classes(
                        "text-subtitle2 text-weight-bold"
                    )

                    ui.label(
                        "Step 1 — Analysis: Read the text against the constitution. "
                        "List key elements: important claims, quality signals, notable features."
                    ).classes("text-caption text-grey-7")
                    analysis_input = (
                        ui.textarea(placeholder="Your analysis...")
                        .classes("w-full")
                        .props("outlined")
                    )

                    ui.label(
                        "Step 2 — Preflection: Contextualize for a reader who has NOT yet read the text. "
                        "Frame what matters, provide background — do NOT spoil conclusions."
                    ).classes("text-caption text-grey-7")
                    preflection_input = (
                        ui.textarea(placeholder="Your preflection...")
                        .classes("w-full")
                        .props("outlined")
                    )

                    ui.label(
                        "Step 3 — Reflection: Evaluate for a reader who HAS read the text. "
                        "Assess quality, identify issues, add analytical value beyond restating. "
                        "Reference charter elements as [X.Y]."
                    ).classes("text-caption text-grey-7")
                    reflection_input = (
                        ui.textarea(placeholder="Your reflection...")
                        .classes("w-full")
                        .props("outlined")
                    )

                    ui.label(
                        "Reflection charter elements (auto-extracted from [X.Y] references)"
                    ).classes("text-caption text-grey-7")
                    reflection_charter_select = (
                        ui.select(
                            options=CHARTER_ELEMENT_IDS,
                            multiple=True,
                            label="Reflection charter elements",
                            value=[],
                        )
                        .classes("w-full")
                        .props("use-chips outlined")
                    )

                    def _on_reflection_change(_=None):
                        extracted = extract_charter_elements(
                            reflection_input.value or ""
                        )
                        reflection_charter_select.set_value(extracted)

                    reflection_input.on("blur", _on_reflection_change)

                    with ui.row().classes("w-full justify-end q-mt-sm"):
                        ui.button(
                            "Submit annotation",
                            on_click=lambda: submit(),
                            color="primary",
                        )

        # === Logic ===
        def current_item() -> dict:
            return items_by_id[state["queue"][state["pos"]]]

        def update_display():
            item = current_item()
            item_id = item["item_id"]
            existing = my_annotations.get(item_id)

            n_done = len(my_annotations)
            progress_label.set_text(f"{annotator_id} · {n_done} done")
            nav_label.set_text(f"Item {state['pos'] + 1} / {len(state['queue'])}")
            subset_badge.set_text(item["subset"])
            item_id_label.set_text(item_id[:16])
            source_html.set_content(
                render_source_text(item["text"], item["reflection_point"])
            )
            copy_text = (
                item["text"][: item["reflection_point"]]
                + "[REFLECTION POINT]"
                + item["text"][item["reflection_point"] :]
            )
            ui.run_javascript(f"window._sourceText = {_json.dumps(copy_text)}")
            prev_btn.set_enabled(state["pos"] > 0)
            next_btn.set_enabled(state["pos"] < len(state["queue"]) - 1)

            if existing:
                status_label.set_text("✎ Editing existing annotation")
                analysis_input.set_value(existing["analysis"])
                preflection_input.set_value(existing["preflection"])
                reflection_input.set_value(existing["reflection"])
                reflection_charter_select.set_value(
                    existing.get("reflection_charter_elements", [])
                )
            else:
                status_label.set_text("New item")
                analysis_input.set_value("")
                preflection_input.set_value("")
                reflection_input.set_value("")
                reflection_charter_select.set_value([])

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
                text=item["text"],
                reflection_point=item["reflection_point"],
                analysis=analysis_input.value.strip(),
                preflection=preflection_input.value.strip(),
                reflection=reflection_input.value.strip(),
                reflection_charter_elements=reflection_charter_select.value or [],
                presentation_order=state["pos"],
            )
            my_annotations.update(get_my_annotations())
            ui.notify("Updated!" if is_edit else "Saved!", type="positive")
            if not is_edit:
                for i in range(state["pos"] + 1, len(state["queue"])):
                    if state["queue"][i] not in my_annotations:
                        state["pos"] = i
                        update_display()
                        return
            update_display()

        ui.run_javascript(f"window._charterText = {_json.dumps(CHARTER_TEXT)}")
        update_display()

    # === Loading gate: show spinner while data loads, then build UI ===
    if SAMPLE_ITEMS:
        build_content()
        return

    loading = ui.column().classes("absolute-center items-center gap-4")
    with loading:
        ui.spinner("dots", size="xl", color="primary")
        ui.label("Loading data...").classes("text-grey-6 text-subtitle1")

    # Kick off the blocking load in a background thread
    threading.Thread(target=ensure_sample_loaded, daemon=True).start()

    # Poll until loaded; timer callback runs in the correct client context
    def _check_loaded():
        if SAMPLE_ITEMS:
            poll_timer.active = False
            loading.delete()
            build_content()

    poll_timer = ui.timer(0.3, _check_loaded)


@ui.page("/overview")
def overview_page():
    """Overview panel: annotation stats and side-by-side annotation viewer."""
    from pipeline.config import load_config

    cfg = load_config()

    annotations_by_item = load_annotations_by_item()
    items_by_id: dict[str, dict] = {}
    for item_id, records in annotations_by_item.items():
        rec = records[0]
        items_by_id[item_id] = {
            "item_id": item_id,
            "subset": rec["subset"],
            "text": rec["text"],
            "reflection_point": rec["reflection_point"],
        }
    all_annotator_ids = sorted(
        {r["annotator_id"] for records in annotations_by_item.values() for r in records}
    )

    # === Header ===
    viewer_id = app.storage.user.get("annotator_id", "")

    def overview_actions():
        ui.button(
            "Back to annotating",
            icon="edit",
            on_click=lambda: ui.navigate.to("/annotate"),
        ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=1, right_slot=overview_actions)

    # === Stats ===
    with ui.row().classes("w-full p-4 gap-8 items-start"):
        # -- Per-annotator stats --
        with ui.card().classes("q-pa-md"):
            ui.label("Annotations per annotator").classes(
                "text-subtitle1 text-weight-bold"
            )
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
            n_annotated = len(annotations_by_item)
            n_dual = sum(1 for recs in annotations_by_item.values() if len(recs) >= 2)
            ui.label(f"Annotated items: {n_annotated}")
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
            options=["All"] + list(cfg.charter.seed.subsets),
            value="All",
            label="Filter by score",
        ).classes("w-48")

    # Container for the annotation cards
    cards_container = ui.column().classes("w-full q-px-md gap-4 q-mt-md")

    def render_annotations():
        cards_container.clear()
        filt_annotator = annotator_filter.value
        filt_source = source_filter.value

        filtered_ids = annotated_ids
        if filt_source != "All":
            filtered_ids = [
                iid
                for iid in filtered_ids
                if iid in items_by_id and items_by_id[iid]["subset"] == filt_source
            ]
        if filt_annotator != "All":
            filtered_ids = [
                iid
                for iid in filtered_ids
                if any(
                    r["annotator_id"] == filt_annotator
                    for r in annotations_by_item.get(iid, [])
                )
            ]

        with cards_container:
            if not filtered_ids:
                ui.label("No annotations match filters.").classes("text-grey-6")
                return

            comments_by_ann = load_comments_by_annotation()

            for item_id in filtered_ids:
                item = items_by_id.get(item_id)
                if item is None:
                    continue
                records = annotations_by_item[item_id]
                if filt_annotator != "All":
                    records = [
                        r for r in records if r["annotator_id"] == filt_annotator
                    ]

                with ui.card().classes("w-full"):
                    # Item header
                    with ui.row().classes("items-center gap-4"):
                        ui.badge(item["subset"]).props("outline")
                        ui.label(f"Item: {item_id[:16]}").classes(
                            "text-caption text-grey-6"
                        )
                        ui.label(f"{len(records)} annotation(s)").classes(
                            "text-caption"
                        )

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

                            with (
                                ui.card()
                                .classes("flex-1 q-pa-sm")
                                .style("min-width: 300px;")
                            ):
                                ui.label(ann_author).classes(
                                    "text-subtitle2 text-weight-bold"
                                )
                                ui.label(rec["timestamp"][:19]).classes(
                                    "text-caption text-grey-6"
                                )

                                all_comments = comments_by_ann.get(
                                    (ann_item_id, ann_author), []
                                )

                                def _render_comment_thread(part_key, part_label):
                                    part_comments = [
                                        c
                                        for c in all_comments
                                        if c.get("target_part", "general") == part_key
                                    ]
                                    with ui.expansion(
                                        f"{part_label} ({len(part_comments)})",
                                        icon="chat_bubble_outline",
                                    ).classes("w-full"):
                                        for c in part_comments:
                                            with ui.row().classes(
                                                "items-start gap-2 q-mb-xs"
                                            ):
                                                ui.label(c["commenter_id"]).classes(
                                                    "text-caption text-weight-bold"
                                                )
                                                ui.label(c["timestamp"][:16]).classes(
                                                    "text-caption text-grey-5"
                                                )

                                                def make_delete(cid=c["id"]):
                                                    def do_delete():
                                                        delete_comment(cid)
                                                        ui.notify(
                                                            "Comment deleted",
                                                            type="info",
                                                        )
                                                        render_annotations()

                                                    return do_delete

                                                ui.button(
                                                    icon="delete",
                                                    on_click=make_delete(),
                                                ).props(
                                                    "flat dense size=xs color=negative"
                                                )
                                            ui.label(c["comment"]).classes(
                                                "text-body2 q-mb-sm"
                                            ).style(
                                                "white-space: pre-wrap; padding-left: 8px;"
                                            )

                                        if viewer_id:
                                            comment_input = (
                                                ui.input(
                                                    placeholder="Add a comment...",
                                                )
                                                .classes("w-full")
                                                .props("dense outlined")
                                            )

                                            def make_submit(
                                                iid=ann_item_id,
                                                target=ann_author,
                                                inp=comment_input,
                                                pk=part_key,
                                            ):
                                                def do_submit():
                                                    assert (
                                                        inp.value and inp.value.strip()
                                                    ), "Comment cannot be empty"
                                                    save_comment(
                                                        iid,
                                                        target,
                                                        viewer_id,
                                                        inp.value.strip(),
                                                        target_part=pk,
                                                    )
                                                    ui.notify(
                                                        "Comment added", type="positive"
                                                    )
                                                    render_annotations()

                                                return do_submit

                                            ui.button(
                                                "Post",
                                                on_click=make_submit(),
                                                color="primary",
                                            ).props("flat dense size=sm")

                                ui.label("Analysis").classes(
                                    "text-overline text-grey-7 q-mt-sm"
                                )
                                ui.label(rec["analysis"]).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )

                                ui.label("Reflection").classes(
                                    "text-overline text-grey-7 q-mt-sm"
                                )
                                ui.label(rec["reflection"]).classes("text-body2").style(
                                    "white-space: pre-wrap;"
                                )

                                refl_elems = rec.get("reflection_charter_elements", [])
                                if refl_elems:
                                    ui.label("Charter Elements").classes(
                                        "text-overline text-grey-7 q-mt-sm"
                                    )
                                    with ui.row().classes("gap-1"):
                                        for eid in refl_elems:
                                            ui.badge(eid, color="teal-3").props(
                                                "outline"
                                            )

                                _render_comment_thread("reflection", "Reflection")

                                ui.label("Preflection").classes(
                                    "text-overline text-grey-7 q-mt-sm"
                                )
                                ui.label(rec["preflection"]).classes(
                                    "text-body2"
                                ).style("white-space: pre-wrap;")

                                _render_comment_thread("preflection", "Preflection")

                                _render_comment_thread("general", "General Comments")

    annotator_filter.on("update:model-value", lambda _: render_annotations())
    source_filter.on("update:model-value", lambda _: render_annotations())
    render_annotations()
