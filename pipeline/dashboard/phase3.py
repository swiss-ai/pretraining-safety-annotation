"""Phase 3 dashboard pages: /phase3 and /phase3/escalations routes."""

from __future__ import annotations

import statistics

from nicegui import app, ui

from pipeline.config import load_config
from pipeline.dashboard import render_header
from pipeline.dashboard.shared import render_source_text
from pipeline.phase2.storage import (
    load_items_for_iteration,
    load_runs,
    save_review,
)


@ui.page("/phase3")
def phase3_monitoring_page():
    """Phase 3 dashboard: cross-model alignment overview."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def phase3_actions():
        from pipeline.phase3.storage import load_escalations

        pending_count = len(load_escalations(status="pending"))
        with (
            ui.button(
                "Escalations",
                icon="flag",
                on_click=lambda: ui.navigate.to("/phase3/escalations"),
            )
            .classes("text-white")
            .props("flat dense")
        ):
            if pending_count > 0:
                ui.badge(str(pending_count), color="red").props("floating")

    render_header(viewer_id, active_phase=3, right_slot=phase3_actions)

    runs = load_runs()
    phase3_runs = [r for r in runs if r.get("phase") == "phase3"]

    if not phase3_runs:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No Phase 3 iterations yet.").classes("text-h6 text-grey-6")
            ui.label("Run phase3 improvers to start cross-model optimization.").classes(
                "text-body2 text-grey-5"
            )
        return

    cfg = load_config()
    target_aliases = {m.alias for m in cfg.phase3.target_models}
    gold_judge_aliases = {m.alias for m in cfg.phase3.gold_judges}
    gold_gen_aliases = {m.alias for m in cfg.phase3.gold_generators}

    # Group runs by group_id
    groups: dict[str, list[dict]] = {}
    for r in phase3_runs:
        gid = r.get("group_id")
        if gid:
            groups.setdefault(gid, []).append(r)

    ui.label("Phase 3: Cross-Model Alignment").classes("text-h5 q-pa-md")

    # Summary cards
    with ui.row().classes("w-full q-px-md gap-4"):
        with ui.card().classes("q-pa-md"):
            ui.label("Groups").classes("text-overline text-grey-7")
            ui.label(str(len(groups))).classes("text-h4")
        with ui.card().classes("q-pa-md"):
            ui.label("Total Iterations").classes("text-overline text-grey-7")
            ui.label(str(len(phase3_runs))).classes("text-h4")
        with ui.card().classes("q-pa-md"):
            ui.label("Target Models").classes("text-overline text-grey-7")
            ui.label(", ".join(sorted(target_aliases))).classes("text-h6")
        with ui.card().classes("q-pa-md"):
            from pipeline.phase3.storage import load_escalations

            pending = len(load_escalations(status="pending"))
            ui.label("Pending Escalations").classes("text-overline text-grey-7")
            ui.label(str(pending)).classes(
                f"text-h4 {'text-orange' if pending > 0 else ''}"
            )

    # Per-group table
    ui.label("Paired Iteration Groups").classes("text-h6 q-pa-md q-mt-md")

    columns = [
        {"name": "group", "label": "Group ID", "field": "group", "align": "left"},
        {"name": "n_iters", "label": "Iterations", "field": "n_iters"},
        {"name": "target", "label": "Target", "field": "target", "align": "left"},
        {"name": "mean_score", "label": "Mean Score", "field": "mean_score"},
        {"name": "accept_rate", "label": "Accept %", "field": "accept_rate"},
        {
            "name": "timestamp",
            "label": "Timestamp",
            "field": "timestamp",
            "align": "left",
        },
    ]

    rows = []
    for gid, g_runs in sorted(
        groups.items(), key=lambda x: x[1][0].get("timestamp", ""), reverse=True
    ):
        targets_in_group = set()
        all_scores = []
        n_accepted = 0
        n_total = 0

        for r in g_runs:
            if r["judge_model"] in target_aliases:
                targets_in_group.add(r["judge_model"])
            if r["generator_model"] in target_aliases:
                targets_in_group.add(r["generator_model"])
            items = load_items_for_iteration(r["iteration"])
            judged = [i for i in items if i.get("judgment")]
            all_scores.extend(i["judgment"]["aggregate"] for i in judged)
            n_accepted += sum(
                1 for i in judged if i["judgment"]["decision"] == "accept"
            )
            n_total += len(judged)

        rows.append(
            {
                "group": gid[:8] + "...",
                "n_iters": len(g_runs),
                "target": ", ".join(sorted(targets_in_group)) or "—",
                "mean_score": (
                    f"{statistics.mean(all_scores):.2f}" if all_scores else "—"
                ),
                "accept_rate": f"{n_accepted / n_total * 100:.0f}%" if n_total else "—",
                "timestamp": g_runs[0].get("timestamp", "")[:19],
            }
        )

    ui.table(columns=columns, rows=rows, row_key="group").classes(
        "w-full q-mx-md"
    ).props("dense flat")


@ui.page("/phase3/escalations")
def phase3_escalations_page():
    """Escalation review page: browse items flagged by phase3 improver agents."""
    viewer_id = app.storage.user.get("annotator_id", "")

    def escalation_actions():
        ui.button(
            "Phase 3",
            icon="dashboard",
            on_click=lambda: ui.navigate.to("/phase3"),
        ).classes("text-white").props("flat dense")

    render_header(viewer_id, active_phase=3, right_slot=escalation_actions)

    from pipeline.phase3.storage import (
        load_escalations,
        update_escalation,
    )

    escalations = load_escalations()
    if not escalations:
        with ui.column().classes("absolute-center items-center"):
            ui.label("No escalations yet.").classes("text-h6 text-grey-6")
        return

    cfg = load_config()
    dimensions = cfg.phase2.scoring.dimensions

    state = {"pos": 0}

    # Filter controls
    status_options = ["all", "pending", "reviewed", "dismissed"]
    target_options = ["all"] + sorted({e["target_model"] for e in escalations})
    role_options = ["all"] + sorted({e["role"] for e in escalations})

    def filtered_escalations() -> list[dict]:
        result = escalations
        if status_filter.value != "all":
            result = [e for e in result if e["status"] == status_filter.value]
        if target_filter.value != "all":
            result = [e for e in result if e["target_model"] == target_filter.value]
        if role_filter.value != "all":
            result = [e for e in result if e["role"] == role_filter.value]
        return result

    with ui.row().classes("w-full items-center gap-2 q-pa-sm"):
        status_filter = ui.select(
            status_options, value="pending", label="Status"
        ).classes("w-32")
        target_filter = ui.select(
            target_options, value="all", label="Target Model"
        ).classes("w-40")
        role_filter = ui.select(role_options, value="all", label="Role").classes("w-32")

    # 50/50 splitter layout
    with (
        ui.splitter(value=50)
        .classes("w-full")
        .style("height: calc(100vh - 140px);") as splitter
    ):
        with splitter.before:
            left_container = ui.column().classes("w-full q-pa-md")

        with splitter.after:
            right_container = (
                ui.column()
                .classes("w-full q-pa-md overflow-auto")
                .style("max-height: 100%;")
            )

    def update_display():
        left_container.clear()
        right_container.clear()

        items = filtered_escalations()
        if not items:
            with left_container:
                ui.label("No escalations match filters.").classes("text-grey-6")
            return

        pos = max(0, min(state["pos"], len(items) - 1))
        state["pos"] = pos
        esc = items[pos]

        # --- Left panel: source text + context ---
        with left_container:
            with ui.row().classes("items-center gap-2"):
                ui.button(
                    icon="arrow_back",
                    on_click=lambda: (
                        state.update(pos=max(0, state["pos"] - 1)),
                        update_display(),
                    ),
                ).props("flat dense")
                ui.label(f"Escalation {pos + 1}/{len(items)}").classes("text-subtitle1")
                ui.button(
                    icon="arrow_forward",
                    on_click=lambda: (
                        state.update(pos=min(len(items) - 1, state["pos"] + 1)),
                        update_display(),
                    ),
                ).props("flat dense")

            status_color = {
                "pending": "orange",
                "reviewed": "green",
                "dismissed": "grey",
            }.get(esc["status"], "grey")
            with ui.row().classes("items-center gap-2 q-mt-sm"):
                ui.badge(esc["status"].upper(), color=status_color)
                ui.badge(f"target: {esc['target_model']}", color="blue").props(
                    "outline"
                )
                ui.badge(f"gold: {esc['gold_model']}", color="teal").props("outline")
                ui.badge(f"role: {esc['role']}", color="blue-grey").props("outline")
                ui.label(f"#{esc['id']}").classes("text-caption text-grey-5")

            # Load item data from the group
            runs = load_runs()
            group_runs = [r for r in runs if r.get("group_id") == esc["group_id"]]

            item_data = None
            for run in group_runs:
                items_in_iter = load_items_for_iteration(run["iteration"])
                for item in items_in_iter:
                    if item["item_id"] == esc["item_id"]:
                        item_data = item
                        break
                if item_data:
                    break

            if item_data:
                ui.label("Source Text").classes("text-overline text-grey-7 q-mt-md")
                ui.html(
                    render_source_text(item_data["text"], item_data["reflection_point"])
                ).style(
                    "line-height: 1.6; font-family: Georgia, serif; "
                    "white-space: pre-wrap; font-size: 0.95em; padding: 8px; "
                    "max-height: 60vh; overflow-y: auto;"
                )

        # --- Right panel: escalation details + review form ---
        with right_container:
            # Claude's concern
            ui.label("Claude's Concern").classes("text-overline text-grey-7")
            with (
                ui.card()
                .classes("w-full q-pa-sm q-mb-md")
                .style("background: #fff3e0;")
            ):
                ui.label(esc["reason"]).classes("text-body2").style(
                    "white-space: pre-wrap;"
                )

            # Show gold vs target judgments from the group
            target_aliases = {m.alias for m in cfg.phase3.target_models}
            for run in group_runs:
                items_in_iter = load_items_for_iteration(run["iteration"])
                match = [i for i in items_in_iter if i["item_id"] == esc["item_id"]]
                if not match:
                    continue
                item = match[0]
                is_target = (
                    run["judge_model"] in target_aliases
                    or run["generator_model"] in target_aliases
                )
                label = "TARGET" if is_target else "GOLD"
                color = "red" if is_target else "teal"

                with ui.card().classes("w-full q-pa-sm q-mb-sm"):
                    with ui.row().classes("items-center gap-2"):
                        ui.badge(label, color=color)
                        ui.label(
                            f"gen={run['generator_model']} judge={run['judge_model']}"
                        ).classes("text-caption")

                    if item.get("judgment"):
                        j = item["judgment"]
                        with ui.row().classes("items-center gap-2 q-mt-xs"):
                            dec_color = "green" if j["decision"] == "accept" else "red"
                            ui.badge(j["decision"].upper(), color=dec_color)
                            ui.badge(
                                f"Score: {j['aggregate']:.2f}", color=dec_color
                            ).props("outline")

                        _NON_PART_KEYS = {
                            "aggregate",
                            "decision",
                            "judge_prompt",
                            "raw_responses",
                            "usage",
                            "latency_ms",
                            "timestamp",
                        }
                        for part, pj in j.items():
                            if part in _NON_PART_KEYS or not isinstance(pj, dict):
                                continue
                            scores = pj.get("scores", {})
                            if scores:
                                score_str = " ".join(
                                    f"{d[:3]}={v}" for d, v in scores.items()
                                )
                                ui.label(f"{part}: {score_str}").classes(
                                    "text-caption q-mt-xs"
                                )

                    if item.get("reflection"):
                        with ui.expansion("Generation", icon="smart_toy").classes(
                            "w-full q-mt-xs"
                        ):
                            ui.label(item.get("reflection", "")).classes(
                                "text-body2"
                            ).style("white-space: pre-wrap;")

            # Review form (for pending escalations)
            if esc["status"] == "pending" and viewer_id:
                ui.separator().classes("q-my-md")
                ui.label("Your Review").classes("text-overline text-grey-7")

                review_notes = (
                    ui.textarea(
                        label="Notes",
                        value=esc.get("reason", ""),
                    )
                    .classes("w-full")
                    .props("outlined")
                )

                # Determine which parts to score from first available judgment
                _NON_PART_KEYS_ESC = {
                    "aggregate",
                    "decision",
                    "judge_prompt",
                    "raw_responses",
                    "usage",
                    "latency_ms",
                    "timestamp",
                }
                _esc_parts = ["preflection", "reflection"]  # default
                for _run in group_runs:
                    _esc_items = load_items_for_iteration(_run["iteration"])
                    _esc_match = [
                        i
                        for i in _esc_items
                        if i["item_id"] == esc["item_id"] and i.get("judgment")
                    ]
                    if _esc_match:
                        _esc_parts = [
                            k
                            for k, v in _esc_match[0]["judgment"].items()
                            if k not in _NON_PART_KEYS_ESC
                            and isinstance(v, dict)
                            and "scores" in v
                        ]
                        break

                # Per-part, per-dimension sliders
                score_sliders: dict[str, dict[str, ui.slider]] = {}
                for part in _esc_parts:
                    ui.label(part).classes("text-subtitle2 q-mt-sm")
                    score_sliders[part] = {}
                    for dim in dimensions:
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.label(f"{dim}:").classes("w-32 text-caption")
                            sl = ui.slider(min=1, max=5, value=3, step=1).classes(
                                "w-64"
                            )
                            score_sliders[part][dim] = sl
                            ui.label().bind_text_from(sl, "value")

                with ui.row().classes("gap-2 q-mt-md"):

                    def submit_review_fn(
                        esc_id=esc["id"],
                        item_id=esc["item_id"],
                        notes_input=review_notes,
                        sliders=score_sliders,
                    ):
                        scores = {}
                        all_vals = []
                        for part, dims in sliders.items():
                            scores[part] = {}
                            for dim in dimensions:
                                val = dims[dim].value
                                scores[part][dim] = val
                                all_vals.append(val)
                        aggregate = sum(all_vals) / len(all_vals) if all_vals else 0

                        # Find the iteration for this item
                        for run in group_runs:
                            items_in_iter = load_items_for_iteration(run["iteration"])
                            if any(i["item_id"] == item_id for i in items_in_iter):
                                iteration = run["iteration"]
                                break
                        else:
                            iteration = group_runs[0]["iteration"]

                        threshold = cfg.phase2.scoring.accept_threshold
                        decision = "accept" if aggregate >= threshold else "reject"
                        save_review(
                            item_id=item_id,
                            iteration=iteration,
                            reviewer_id=viewer_id,
                            scores=scores,
                            aggregate=aggregate,
                            decision=decision,
                            notes=notes_input.value or "",
                        )
                        update_escalation(esc_id, "reviewed", notes_input.value)
                        ui.notify("Review submitted", type="positive")
                        update_display()

                    ui.button(
                        "Submit Review",
                        icon="check",
                        on_click=submit_review_fn,
                        color="positive",
                    )

                    def dismiss_esc(esc_id=esc["id"]):
                        update_escalation(esc_id, "dismissed")
                        ui.notify("Escalation dismissed", type="info")
                        update_display()

                    ui.button(
                        "Dismiss",
                        icon="close",
                        on_click=dismiss_esc,
                        color="grey",
                    ).props("flat")

    def _on_filter_change():
        state["pos"] = 0
        update_display()

    status_filter.on_value_change(lambda _: _on_filter_change())
    target_filter.on_value_change(lambda _: _on_filter_change())
    role_filter.on_value_change(lambda _: _on_filter_change())

    update_display()
