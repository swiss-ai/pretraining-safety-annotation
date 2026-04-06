"""CLI tools for the improver agent to query iteration data and run tests.

Usage (via Bash tool):
    python -m pipeline.improver_tools summary <iteration>
    python -m pipeline.improver_tools failures <iteration> [--limit N] [--offset N] [--reasoning-limit N]
    python -m pipeline.improver_tools accepts <iteration> [--limit N] [--offset N] [--reasoning-limit N] [--sort top|borderline]
    python -m pipeline.improver_tools show <item_id>[,id2,...] <iteration> [--brief]
    python -m pipeline.improver_tools show --gold <iteration> [--brief]
    python -m pipeline.improver_tools item <item_id> <iteration>
    python -m pipeline.improver_tools reasoning <item_id>[,id2,...] <iteration>
    python -m pipeline.improver_tools diversity <iteration>
    python -m pipeline.improver_tools scores <iteration>
    python -m pipeline.improver_tools distribution <iteration>
    python -m pipeline.improver_tools gold [--limit N] [--offset N] [--verbose]
    python -m pipeline.improver_tools compare <item_id> <iteration>
    python -m pipeline.improver_tools reviews [<judge_prompt>] [--limit N]
    python -m pipeline.improver_tools filter <iteration> --dim X (--below N | --above N) [--part preflection_3p|preflection_1p|reflection_1p|reflection_3p]
    python -m pipeline.improver_tools trend
    python -m pipeline.improver_tools diagnose <group_id>
    python -m pipeline.improver_tools diff <iter1> <iter2> [--limit N]
    python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2,...] [--n N] [--role judge|generator]
    python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2,...] [--iteration N] [--role judge|generator]
    python -m pipeline.improver_tools run_batch [--role judge|generator]
    python -m pipeline.improver_tools run_cross_batch --role judge|generator --target <alias>
    python -m pipeline.improver_tools cross_summary <group_id>
    python -m pipeline.improver_tools test_results [--role judge|generator] [--type generate|judge|batch]
    python -m pipeline.improver_tools correlations
    python -m pipeline.improver_tools rejudge_all

    # Phase 3 commands
    python -m pipeline.improver_tools run_paired_batch --role judge|generator --target <alias>
    python -m pipeline.improver_tools paired_summary <group_id>
    python -m pipeline.improver_tools disagreements <group_id> [--limit N]
    python -m pipeline.improver_tools dimension_alignment <group_id>
    python -m pipeline.improver_tools paired_show <item_id> <group_id>
    python -m pipeline.improver_tools escalate <item_id> <group_id> --reason "..."
    python -m pipeline.improver_tools escalations [--status S]
    python -m pipeline.improver_tools correlation_trend [--target T]
    python -m pipeline.improver_tools parse_stats <iteration>
    python -m pipeline.improver_tools rollback <alias> <role> <version>
"""

import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pipeline.phase2.storage import (
    load_items_for_iteration,
    load_test_results,
    save_test_result,
)

_JUDGMENT_NON_PART_KEYS = {
    "aggregate",
    "decision",
    "judge_prompt",
    "raw_responses",
    "usage",
    "latency_ms",
    "timestamp",
}


def _judgment_parts(j: dict) -> dict[str, dict]:
    """Extract per-part sub-dicts from a judgment, excluding metadata keys.

    Works for both old-format judgments (preflection, reflection) and
    new-format judgments (preflection_3p, preflection_1p, reflection_1p, reflection_3p).
    """
    return {
        k: v
        for k, v in j.items()
        if k not in _JUDGMENT_NON_PART_KEYS and isinstance(v, dict) and "scores" in v
    }


def cmd_summary(iteration: int) -> None:
    """Print aggregate statistics for an iteration."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]
    if not judged:
        print(f"No judged items for iteration {iteration}")
        return

    n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
    n_rej = len(judged) - n_acc
    scores = [i["judgment"]["aggregate"] for i in judged]
    n_gold = sum(1 for i in judged if i.get("is_gold"))

    print(f"Iteration {iteration}: {len(judged)} items ({n_gold} gold)")
    print(f"  Accept: {n_acc} ({n_acc/len(judged)*100:.0f}%)")
    print(f"  Reject: {n_rej} ({n_rej/len(judged)*100:.0f}%)")
    print(f"  Mean score: {statistics.mean(scores):.2f}")
    print(f"  Score range: {min(scores):.2f} – {max(scores):.2f}")

    # Per-dimension breakdown across all judgment parts (supports both old 2-part and new 4-part)
    _NON_PART_KEYS = {
        "aggregate",
        "decision",
        "judge_prompt",
        "raw_responses",
        "usage",
        "latency_ms",
        "timestamp",
    }
    dim_scores: dict[str, list[float]] = {}
    for item in judged:
        for part, part_j in item["judgment"].items():
            if part in _NON_PART_KEYS or not isinstance(part_j, dict):
                continue
            for dim, score in part_j.get("scores", {}).items():
                dim_scores.setdefault(f"{part}_{dim}", []).append(score)

    print("\n  Per-dimension means:")
    for dim, vals in sorted(dim_scores.items()):
        print(f"    {dim}: {statistics.mean(vals):.2f}")


def cmd_failures(
    iteration: int, limit: int = 10, reasoning_limit: int = 200, offset: int = 0
) -> None:
    """Print rejected items with judge reasoning."""
    items = load_items_for_iteration(iteration)
    rejected = [
        i for i in items if i.get("judgment") and i["judgment"]["decision"] == "reject"
    ]
    rejected.sort(key=lambda i: i["judgment"]["aggregate"])

    sliced = rejected[offset : offset + limit]
    print(
        f"Rejected items ({len(rejected)} total, showing {offset}–{offset + len(sliced)}):\n"
    )
    _print_judged_items(sliced, reasoning_limit)


def cmd_accepts(
    iteration: int,
    limit: int = 10,
    reasoning_limit: int = 200,
    offset: int = 0,
    sort: str = "top",
) -> None:
    """Print accepted items with judge reasoning — symmetric to `failures`.

    Use this to hunt for false positives (items the judge accepted that
    shouldn't have been). False positives directly contaminate the training
    signal and are usually more harmful than false negatives, so always pair
    `failures` analysis with `accepts` analysis.

    sort:
      "top"        — highest aggregate first (judge most confident; the
                     hardest cases to challenge — if these have problems
                     the rubric is badly miscalibrated). Default.
      "borderline" — closest to the accept threshold first (judge least
                     confident; useful for finding items the judge barely
                     waved through that humans would reject).
    """
    from pipeline.config import load_config

    cfg = load_config()
    threshold = cfg.phase2.scoring.accept_threshold

    items = load_items_for_iteration(iteration)
    accepted = [
        i for i in items if i.get("judgment") and i["judgment"]["decision"] == "accept"
    ]
    if sort == "borderline":
        accepted.sort(key=lambda i: abs(i["judgment"]["aggregate"] - threshold))
        sort_label = f"borderline (closest to threshold {threshold})"
    elif sort == "top":
        accepted.sort(key=lambda i: -i["judgment"]["aggregate"])
        sort_label = "top (highest aggregate first)"
    else:
        raise ValueError(f"Unknown --sort value: {sort!r}. Use 'top' or 'borderline'.")

    sliced = accepted[offset : offset + limit]
    print(
        f"Accepted items ({len(accepted)} total, sort={sort_label}, "
        f"showing {offset}–{offset + len(sliced)}):\n"
    )
    _print_judged_items(sliced, reasoning_limit)


def _print_judged_items(items: list[dict], reasoning_limit: int) -> None:
    """Shared body of cmd_failures / cmd_accepts."""
    for item in items:
        j = item["judgment"]
        print(
            f"--- {item['item_id'][:16]} (score={j['aggregate']:.2f}, gold={item.get('is_gold', False)}) ---"
        )
        print(f"  Text preview: {item['text'][:150]}...")
        print(f"  Preflection (3p): {item.get('preflection', '')[:150]}...")
        if item.get("preflection_1p"):
            print(f"  Preflection (1p): {item.get('preflection_1p', '')[:150]}...")
        print(f"  Reflection (1p): {item.get('reflection', '')[:150]}...")
        if item.get("reflection_3p"):
            print(f"  Reflection (3p): {item.get('reflection_3p', '')[:150]}...")
        print(f"  Charter elements: {item.get('charter_elements', [])}")
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
            print(f"  {part} scores: {pj.get('scores', {})}")
            print(f"  {part} reasoning: {pj.get('reasoning', '')[:reasoning_limit]}")
        print()


def cmd_show(
    item_ids: list[str], iteration: int, brief: bool = False, gold_only: bool = False
) -> None:
    """Print source text, preflection, and reflection for item(s) — easy to read.

    With --gold, shows all gold items for the iteration (ignores item_ids).
    """
    items = load_items_for_iteration(iteration)

    if gold_only:
        gold_items = [i for i in items if i.get("is_gold")]
        if not gold_items:
            print(f"No gold items in iteration {iteration}")
            return
        for item in gold_items:
            _print_item(item, brief)
        return

    for item_id in item_ids:
        matches = [i for i in items if i["item_id"].startswith(item_id)]
        if not matches:
            print(f"No item matching '{item_id}' in iteration {iteration}")
            continue

        for item in matches:
            _print_item(item, brief)


def _print_item(item: dict, brief: bool = False) -> None:
    """Print a single item's details."""
    rp = item["reflection_point"]
    j = item.get("judgment", {})
    decision = j.get("decision", "?")
    agg = j.get("aggregate", 0)

    print(
        f"=== {item['item_id'][:16]} ({decision}, score={agg:.1f}, gold={item.get('is_gold', False)}) ===\n"
    )
    print("--- SOURCE TEXT ---")
    if brief:
        print(item["text"][:300] + "...")
    else:
        print(item["text"][:rp] + " [REFLECTION POINT] " + item["text"][rp:])
    print(f"\n--- PREFLECTION (3p) ---\n{item.get('preflection', '')}")
    if item.get("preflection_1p"):
        print(f"\n--- PREFLECTION (1p) ---\n{item.get('preflection_1p', '')}")
    print(f"\n--- REFLECTION (1p) ---\n{item.get('reflection', '')}")
    if item.get("reflection_3p"):
        print(f"\n--- REFLECTION (3p) ---\n{item.get('reflection_3p', '')}")
    print(f"\n--- ANALYSIS ---\n{item.get('analysis', '')}")
    print(f"\n--- CHARTER ELEMENTS ---\n{item.get('charter_elements', [])}")
    print()


def cmd_item(item_id: str, iteration: int) -> None:
    """Print full details for a specific item (JSON format)."""
    items = load_items_for_iteration(iteration)
    matches = [i for i in items if i["item_id"].startswith(item_id)]
    if not matches:
        print(f"No item matching '{item_id}' in iteration {iteration}")
        return

    for item in matches:
        print(
            json.dumps(
                {
                    "item_id": item["item_id"],
                    "is_gold": item.get("is_gold"),
                    "subset": item["subset"],
                    "text_preview": item["text"][:500],
                    "reflection_point": item["reflection_point"],
                    "analysis": item.get("analysis"),
                    "preflection": item.get("preflection"),
                    "preflection_1p": item.get("preflection_1p"),
                    "reflection": item.get("reflection"),
                    "reflection_3p": item.get("reflection_3p"),
                    "charter_elements": item.get("charter_elements"),
                    "judgment": item.get("judgment"),
                },
                indent=2,
            )
        )


def cmd_reasoning(item_ids: list[str], iteration: int) -> None:
    """Print full judge reasoning for specific items — scores, reasoning, and decision logic."""
    items = load_items_for_iteration(iteration)
    for item_id in item_ids:
        matches = [i for i in items if i["item_id"].startswith(item_id)]
        if not matches:
            print(f"No item matching '{item_id}' in iteration {iteration}")
            continue
        for item in matches:
            j = item.get("judgment")
            if not j:
                print(f"=== {item['item_id'][:16]} — no judgment ===\n")
                continue
            print(
                f"=== {item['item_id'][:16]} ({j['decision']}, agg={j['aggregate']:.2f}) ===\n"
            )
            for part, pj in _judgment_parts(j).items():
                scores = pj.get("scores", {})
                print(f"  {part} scores: {scores}  (agg={pj.get('aggregate', 0):.2f})")
                print(f"  {part} reasoning: {pj.get('reasoning', '')}\n")
            print(f"  text preview: {item['text'][:200]}...")
            print()


def _field_diversity(items, field_name):
    """Print diversity stats for a single text field across items."""
    texts = [i.get(field_name, "") or "" for i in items]
    texts = [t for t in texts if t]
    if not texts:
        print(f"  (no {field_name} data)")
        return

    # First-word frequency
    first_words = Counter(t.split()[0] if t.split() else "" for t in texts)
    print(f"  First-word freq: {dict(first_words.most_common(10))}")

    # 5-word opener frequency (only duplicates)
    openers = Counter(" ".join(t.split()[:5]) for t in texts)
    dupes = {k: v for k, v in openers.items() if v > 1}
    if dupes:
        print(
            f"  Duplicate 5-word openers: {dict(sorted(dupes.items(), key=lambda x: -x[1]))}"
        )

    # Formulaic closing patterns (last sentence, only duplicates)
    closings = Counter()
    for t in texts:
        sentences = [s.strip() for s in t.rstrip().rsplit(".", 1)]
        last = sentences[-1] if sentences else ""
        if last:
            closings[last[:80]] += 1
    closing_dupes = {k: v for k, v in closings.items() if v > 1}
    if closing_dupes:
        print(
            f"  Duplicate closings: {dict(sorted(closing_dupes.items(), key=lambda x: -x[1]))}"
        )

    # Uniqueness
    unique_pct = len(set(texts)) / len(texts) * 100
    print(f"  Uniqueness: {unique_pct:.0f}% ({len(set(texts))}/{len(texts)})")


def cmd_diversity(iteration: int) -> None:
    """Show frequency-based diversity statistics for reflections, preflections, and analysis."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]

    print(f"Diversity check for iteration {iteration} ({len(judged)} items):\n")
    fields = ["reflection", "preflection", "analysis"]
    # Also check new-format fields if present
    if judged and judged[0].get("preflection_1p") is not None:
        fields = [
            "preflection",
            "preflection_1p",
            "reflection",
            "reflection_3p",
            "analysis",
        ]
    for field in fields:
        print(f"=== {field} ===")
        _field_diversity(judged, field)
        print()


def cmd_scores(iteration: int) -> None:
    """Print a compact scores table for all items."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]
    judged.sort(key=lambda i: i["judgment"]["aggregate"])

    for item in judged:
        j = item["judgment"]
        parts_str = " | ".join(
            f"{part[:6]}[{' '.join(f'{k[:3]}={v}' for k, v in part_j.get('scores', {}).items())}]"
            for part, part_j in _judgment_parts(j).items()
        )
        gold = "G" if item.get("is_gold") else " "
        print(
            f"{gold} {j['decision'][:3]:>3} {j['aggregate']:4.1f} | "
            f"{parts_str} | {item['item_id'][:12]}"
        )


def cmd_distribution(iteration: int) -> None:
    """Print per-dimension score distributions and floor-rule trigger counts."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]
    if not judged:
        print(f"No judged items for iteration {iteration}")
        return

    # Collect scores per (part, dimension)
    part_dim_scores: dict[str, dict[str, list[int]]] = {}
    for item in judged:
        for part, part_j in _judgment_parts(item["judgment"]).items():
            for dim, score in part_j.get("scores", {}).items():
                part_dim_scores.setdefault(part, {}).setdefault(dim, []).append(score)

    # Global distribution
    all_scores = []
    for part_dims in part_dim_scores.values():
        for scores in part_dims.values():
            all_scores.extend(scores)

    print(
        f"Iteration {iteration}: {len(judged)} items, {len(all_scores)} dimension scores\n"
    )
    print("=== Global distribution ===")
    dist = Counter(all_scores)
    for s in sorted(dist):
        pct = dist[s] / len(all_scores) * 100
        bar = "#" * int(pct)
        print(f"  {s}: {dist[s]:>4} ({pct:4.1f}%) {bar}")
    low = sum(1 for s in all_scores if s <= 2)
    print(f"  Floor triggers (<=2): {low} ({low / len(all_scores) * 100:.1f}%)\n")

    # Per-part, per-dimension breakdown
    print("=== Per-dimension breakdown ===")
    print(f"{'part':<20} {'dim':<20} {'mean':>5} {'<=2':>4} {'dist (1-5)'}")
    print("-" * 75)
    # Floor trigger totals by dimension
    dim_floors: dict[str, int] = {}
    for part in sorted(part_dim_scores):
        for dim in sorted(part_dim_scores[part]):
            scores = part_dim_scores[part][dim]
            mean = statistics.mean(scores)
            n_floor = sum(1 for s in scores if s <= 2)
            dim_floors[dim] = dim_floors.get(dim, 0) + n_floor
            d = Counter(scores)
            dist_str = " ".join(f"{d.get(i, 0):>3}" for i in range(1, 6))
            print(f"{part:<20} {dim:<20} {mean:5.2f} {n_floor:>4} [{dist_str}]")

    print(f"\n=== Floor triggers by dimension (all parts) ===")
    for dim in sorted(dim_floors, key=dim_floors.get, reverse=True):
        print(f"  {dim}: {dim_floors[dim]}")


def _load_gold() -> list[dict]:
    """Load gold annotations from SQLite."""
    from pipeline.phase1.storage import load_annotations

    return load_annotations()


def cmd_gold(limit: int = 5, offset: int = 0, verbose: bool = False) -> None:
    """Print gold annotations for reference — shows what good output looks like.

    Default output is concise (analysis + preflection + reflection, no source text).
    Use --verbose to include full source text.
    """
    items = _load_gold()
    sliced = items[offset : offset + limit]
    print(
        f"Gold annotations ({len(items)} total, showing {offset}–{offset + len(sliced)}):\n"
    )
    for item in sliced:
        print(f"=== {item['item_id'][:16]} (subset={item['subset']}) ===")
        if verbose:
            rp = item["reflection_point"]
            text = item["text"]
            print(f"--- SOURCE TEXT ---")
            print(text[:rp] + " [REFLECTION POINT] " + text[rp:])
        print(f"\n--- ANALYSIS ---\n{item.get('analysis', '')}")
        print(f"\n--- PREFLECTION ---\n{item.get('preflection', '')}")
        print(f"\n--- REFLECTION ---\n{item.get('reflection', '')}")
        print(f"\n--- CHARTER ELEMENTS ---\n{item.get('charter_elements', [])}")
        print()


def cmd_compare(item_id: str, iteration: int) -> None:
    """Side-by-side comparison of generated output vs gold annotation for the same item."""
    items = load_items_for_iteration(iteration)
    gold_items = _load_gold()
    gold_by_id = {g["item_id"]: g for g in gold_items}

    matches = [i for i in items if i["item_id"].startswith(item_id)]
    if not matches:
        print(f"No item matching '{item_id}' in iteration {iteration}")
        return

    for item in matches:
        gold = gold_by_id.get(item["item_id"])
        if not gold:
            print(f"No gold annotation for {item['item_id'][:16]}")
            continue

        j = item.get("judgment", {})
        print(f"=== {item['item_id'][:16]} (score={j.get('aggregate', 0):.1f}) ===\n")

        print("--- GENERATED PREFLECTION (3p) ---")
        print(item.get("preflection", ""))
        if item.get("preflection_1p"):
            print("\n--- GENERATED PREFLECTION (1p) ---")
            print(item.get("preflection_1p", ""))
        print("\n--- GOLD PREFLECTION ---")
        print(gold.get("preflection", ""))

        print("\n--- GENERATED REFLECTION (1p) ---")
        print(item.get("reflection", ""))
        if item.get("reflection_3p"):
            print("\n--- GENERATED REFLECTION (3p) ---")
            print(item.get("reflection_3p", ""))
        print("\n--- GOLD REFLECTION ---")
        print(gold.get("reflection", ""))

        print("\n--- GENERATED CHARTER ---")
        print(item.get("charter_elements", []))
        print("\n--- GOLD CHARTER ---")
        print(gold.get("charter_elements", []))
        print()


def cmd_reviews(judge_prompt: str | None = None, limit: int = 20) -> None:
    """Print human reviews, optionally filtered by judge prompt version.

    Shows reviewer scores, decision, and notes alongside the judge's scores
    for calibration comparison. Only shows *train* split reviews (75%) so
    the validation split remains unseen by the improver.

    judge_prompt: substring match against the run's judge_prompt field
    (e.g. "v10", "judge_v10.md"). When omitted, shows reviews across every
    judge prompt version. Reviews are grouped by (judge_prompt, judge_model)
    in the output so it's clear which prompt each note refers to.
    """
    from pipeline.phase2.storage import (
        load_latest_reviews,
        load_review_comments,
        load_runs,
        review_split,
    )

    reviews = load_latest_reviews()
    all_comments = load_review_comments()
    if not reviews:
        print("No human reviews yet.")
        return

    # Only show train-split reviews to the improver
    filtered = [r for r in reviews.values() if review_split(r["item_id"]) == "train"]

    # Build iteration -> (judge_prompt, judge_model) lookup
    runs = load_runs()
    run_judge: dict[int, tuple[str, str]] = {
        r["iteration"]: (r["judge_prompt"], r["judge_model"]) for r in runs
    }

    if judge_prompt is not None:
        needle = judge_prompt.lower()
        filtered = [
            r
            for r in filtered
            if needle in (run_judge.get(r["iteration"], ("", ""))[0]).lower()
        ]
        if not filtered:
            print(
                f"No train-split reviews found for judge prompt matching "
                f"'{judge_prompt}'."
            )
            return

    # Load items for all referenced iterations
    items = []
    seen_iters: set[int] = set()
    for r in filtered:
        if r["iteration"] not in seen_iters:
            seen_iters.add(r["iteration"])
            items.extend(load_items_for_iteration(r["iteration"]))

    items_by_key = {(i["item_id"], i["iteration"]): i for i in items}

    # Group by (judge_prompt, judge_model) so notes are organised by the
    # prompt version that produced the judgment, not by iteration.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in filtered:
        info = run_judge.get(r["iteration"], ("unknown", "unknown"))
        grouped.setdefault(info, []).append(r)

    total_shown = 0
    print(
        f"Human reviews ({len(filtered)} total across "
        f"{len(grouped)} judge prompt(s)):\n"
    )

    def _prompt_sort_key(key: tuple[str, str]) -> tuple:
        import re

        m = re.search(r"_v(\d+)", key[0])
        return (key[1], int(m.group(1)) if m else 0)

    for group_key in sorted(grouped, key=_prompt_sort_key, reverse=True):
        group_reviews = grouped[group_key]
        gp_name, gm_name = group_key
        remaining = limit - total_shown
        if remaining <= 0:
            print(
                f"\n(--limit {limit} reached; "
                f"{sum(len(v) for v in grouped.values()) - total_shown} more "
                f"reviews not shown)"
            )
            break
        # Sort within a group: latest iteration first, then latest timestamp
        group_reviews.sort(
            key=lambda r: (-r["iteration"], r.get("timestamp", "")), reverse=False
        )
        show = group_reviews[:remaining]
        print(
            f"=== {gp_name} / {gm_name} "
            f"({len(group_reviews)} reviews, showing {len(show)}) ==="
        )
        total_shown += len(show)
        for r in show:
            item = items_by_key.get((r["item_id"], r["iteration"]))
            judge_agg = ""
            judge_decision = ""
            if item and item.get("judgment"):
                j = item["judgment"]
                judge_agg = f"{j['aggregate']:.2f}"
                judge_decision = j["decision"]

            print(
                f"--- {r['item_id'][:16]} iter={r['iteration']} "
                f"reviewer={r['reviewer_id']} ---"
            )
            print(
                f"  Human:  decision={r['decision']}  "
                f"aggregate={r['aggregate']:.2f}"
            )
            if judge_agg:
                print(f"  Judge:  decision={judge_decision}  aggregate={judge_agg}")

            scores = r["scores"]
            is_per_part = scores and isinstance(next(iter(scores.values())), dict)
            if is_per_part:
                all_parts = sorted(
                    set(scores.keys())
                    | (
                        set(_judgment_parts(item["judgment"]).keys())
                        if item and item.get("judgment")
                        else set()
                    )
                )
                for part in all_parts:
                    human_s = (
                        scores.get(part, {})
                        if isinstance(scores.get(part, {}), dict)
                        else {}
                    )
                    judge_s = {}
                    if item and item.get("judgment"):
                        judge_s = item["judgment"].get(part, {}).get("scores", {})
                    dims = sorted(set(human_s) | set(judge_s))
                    pairs = " ".join(
                        f"{d[:3]}={human_s.get(d, '?')}/{judge_s.get(d, '?')}"
                        for d in dims
                    )
                    if dims:
                        print(f"  {part}: {pairs}  (human/judge)")
            else:
                print(f"  Scores: {scores}")

            if r.get("notes"):
                print(f"  Notes: {r['notes']}")

            review_key = (r["item_id"], r["iteration"], r["reviewer_id"])
            comments = all_comments.get(review_key, [])
            if comments:
                print("  Comments:")
                for c in comments:
                    print(
                        f"    {c['commenter_id']} "
                        f"({c['timestamp'][:19]}): {c['comment']}"
                    )
            print()


def cmd_filter(
    iteration: int,
    dim: str,
    below: float | None = None,
    above: float | None = None,
    part: str | None = None,
) -> None:
    """Filter items by score threshold on a specific dimension.

    Exactly one of --below or --above must be supplied.

    Valid --part values: preflection_3p, preflection_1p, reflection_1p, reflection_3p
    Valid --dim values: relevance, specificity, charter_grounding, voice_tone
    If --part is omitted, searches all parts.
    """
    assert dim, "--dim is required"
    assert (below is None) != (
        above is None
    ), "Exactly one of --below or --above is required"

    if below is not None:
        cmp = lambda s: s < below  # noqa: E731
        cmp_label = f"< {below}"
        sort_key = lambda x: x[4]  # noqa: E731 — ascending: worst first
    else:
        cmp = lambda s: s > above  # noqa: E731
        cmp_label = f"> {above}"
        sort_key = lambda x: -x[4]  # noqa: E731 — descending: best first

    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]

    # Default to all parts present in the first judged item
    if part:
        parts = [part]
    elif judged:
        parts = list(_judgment_parts(judged[0]["judgment"]).keys())
    else:
        parts = ["preflection", "reflection"]
    hits = []
    for item in judged:
        j = item["judgment"]
        for p in parts:
            score = j.get(p, {}).get("scores", {}).get(dim)
            if score is not None and cmp(score):
                # Map part name to item field (legacy or new)
                _field = (
                    p
                    if p in item
                    else ("preflection" if "preflection" in p else "reflection")
                )
                text = item.get(_field, "") or ""
                hits.append(
                    (
                        item["item_id"],
                        j["decision"],
                        j["aggregate"],
                        p,
                        score,
                        text[:80],
                    )
                )

    hits.sort(key=sort_key)
    print(
        f"Items with {dim} {cmp_label} in iteration {iteration} ({len(hits)} hits):\n"
    )
    for iid, dec, agg, p, sc, preview in hits:
        print(
            f"  {iid[:16]} {dec:>3} agg={agg:.1f} {p[:3]}_{dim[:3]}={sc} | {preview}..."
        )


def cmd_trend() -> None:
    """Print per-iteration trend table: accept rate, mean score, per-dimension means."""
    from pipeline.phase2.storage import load_runs

    runs = load_runs()
    assert runs, "No runs found"

    # Collect dimension names from first iteration with data
    all_dim_keys = []
    for run in runs:
        items = load_items_for_iteration(run["iteration"])
        judged = [i for i in items if i.get("judgment")]
        if not judged:
            continue
        for part, part_j in _judgment_parts(judged[0]["judgment"]).items():
            for dim in part_j.get("scores", {}):
                key = f"{part[:3]}_{dim[:3]}"
                if key not in all_dim_keys:
                    all_dim_keys.append(key)
        break

    # Header
    dim_header = " ".join(f"{k:>7}" for k in all_dim_keys)
    print(
        f"{'iter':>4} {'acc%':>5} {'mean':>5} {dim_header}  gen_prompt / judge_prompt"
    )

    for run in runs:
        it = run["iteration"]
        items = load_items_for_iteration(it)
        judged = [i for i in items if i.get("judgment")]
        if not judged:
            print(f"{it:>4}  (no judged items)")
            continue

        scores = [i["judgment"]["aggregate"] for i in judged]
        n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
        acc_pct = n_acc / len(judged) * 100
        mean = statistics.mean(scores)

        dim_means: dict[str, list[float]] = {}
        for item in judged:
            for part, part_j in _judgment_parts(item["judgment"]).items():
                for dim, sc in part_j.get("scores", {}).items():
                    key = f"{part[:3]}_{dim[:3]}"
                    dim_means.setdefault(key, []).append(sc)
        dim_str = " ".join(
            f"{statistics.mean(dim_means.get(k, [0])):7.2f}" for k in all_dim_keys
        )

        gen_p = run.get("gen_prompt", "?")
        judge_p = run.get("judge_prompt", "?")
        print(f"{it:>4} {acc_pct:5.0f} {mean:5.2f} {dim_str}  {gen_p} / {judge_p}")


def cmd_correlations() -> None:
    """Print judge-human correlation stats grouped by judge prompt version.

    For each judge version with paired data, computes:
    - Decision agreement rate (accept/reject match %)
    - Mean absolute score difference (judge aggregate vs human aggregate)
    - Per-dimension score diffs where both human and judge scored the same dims

    Combines two sources of judgments:
      1. The items table — every item carries its original judgment along with
         the (judge_prompt, judge_model) recorded on its run. This is enough
         when reviews are for items judged by the prompt that was current
         at generation time.
      2. The judge_correlations table — retroactive re-judgments produced by
         `rejudge_all`, used when an item was originally judged by an older
         prompt and a newer one was applied later.

    Only uses *train* split reviews so the validation split remains unseen.
    """
    from pipeline.phase2.storage import (
        build_review_lookup,
        load_judge_correlations,
        load_latest_items,
        load_runs,
    )

    review_by_item = build_review_lookup(split="train")
    if not review_by_item:
        print(
            "No human reviews in the train split yet. "
            "Submit reviews on /pipeline/review (or via the annotation UI) "
            "to populate calibration data."
        )
        return

    runs = load_runs()
    run_judge_info: dict[int, tuple[str, str]] = {
        r["iteration"]: (r["judge_prompt"], r["judge_model"]) for r in runs
    }

    # Synthesize "entries" with the same shape as judge_correlations rows so
    # the existing analysis loop below can consume them unchanged.
    by_prompt: dict[tuple[str, str], list[dict]] = {}
    seen: set[tuple[str, str, str, int]] = set()

    # Source 1: items table (original judgments)
    all_items = load_latest_items()
    for item_id, iteration in review_by_item:
        item = all_items.get((item_id, iteration))
        if not item or not item.get("judgment"):
            continue
        judge_info = run_judge_info.get(iteration)
        if not judge_info:
            continue
        judge_prompt, judge_model = judge_info
        by_prompt.setdefault((judge_prompt, judge_model), []).append(
            {
                "item_id": item_id,
                "iteration": iteration,
                "judge_prompt": judge_prompt,
                "judge_model": judge_model,
                "judgment": item["judgment"],
            }
        )
        seen.add((judge_prompt, judge_model, item_id, iteration))

    # Source 2: judge_correlations table (retroactive re-judgments)
    correlations = load_judge_correlations()
    for c in correlations:
        key = (
            c["judge_prompt"],
            c.get("judge_model", "unknown"),
            c["item_id"],
            c["iteration"],
        )
        if key in seen:
            continue
        if (c["item_id"], c["iteration"]) not in review_by_item:
            continue
        by_prompt.setdefault(
            (c["judge_prompt"], c.get("judge_model", "unknown")), []
        ).append(c)

    if not by_prompt:
        n_reviews = len(review_by_item)
        print(
            f"Found {n_reviews} train-split reviews but none of the reviewed "
            "items have a stored judge score. This usually means the reviews "
            "are for items whose original judgment was never saved (very old "
            "data). Run `python -m pipeline.improver_tools rejudge_all` to "
            "populate calibration data."
        )
        return

    for prompt_name, model_name in sorted(by_prompt):
        entries = by_prompt[(prompt_name, model_name)]
        decision_matches = 0
        score_diffs = []
        dim_diffs: dict[str, list[float]] = {}
        matched = 0

        for c in entries:
            key = (c["item_id"], c["iteration"])
            review = review_by_item.get(key)
            if not review:
                continue
            matched += 1

            j = c["judgment"]
            # Decision agreement
            judge_decision = j.get("decision", "")
            human_decision = review.get("decision", "")
            if judge_decision == human_decision:
                decision_matches += 1

            # Score diff
            judge_agg = j.get("aggregate", 0)
            human_agg = review.get("aggregate", 0)
            score_diffs.append(abs(judge_agg - human_agg))

            # Per-dimension diffs
            human_scores = review.get("scores", {})
            is_per_part = human_scores and isinstance(
                next(iter(human_scores.values()), None), dict
            )
            if is_per_part:
                all_parts_c = set(human_scores.keys()) | set(_judgment_parts(j).keys())
                for part in all_parts_c:
                    h_part = (
                        human_scores.get(part, {})
                        if isinstance(human_scores.get(part, {}), dict)
                        else {}
                    )
                    j_part = j.get(part, {}).get("scores", {})
                    for dim in set(h_part) & set(j_part):
                        dim_key = f"{part[:3]}_{dim[:3]}"
                        dim_diffs.setdefault(dim_key, []).append(
                            abs(j_part[dim] - h_part[dim])
                        )

        if matched == 0:
            print(
                f"\n{prompt_name} / {model_name}: {len(entries)} correlations, 0 matched to reviews"
            )
            continue

        agreement = decision_matches / matched * 100
        mean_diff = statistics.mean(score_diffs) if score_diffs else 0

        print(f"\n{prompt_name} / {model_name} ({matched} items matched to reviews):")
        print(f"  Decision agreement: {agreement:.0f}% ({decision_matches}/{matched})")
        print(f"  Mean |score diff|:  {mean_diff:.2f}")

        if dim_diffs:
            print("  Per-dimension mean |diff|:")
            for dim_key in sorted(dim_diffs):
                print(f"    {dim_key}: {statistics.mean(dim_diffs[dim_key]):.2f}")

    print(
        "\nNote: this command shows numeric calibration only. To read the "
        "reviewer NOTES (why the human disagreed) — which are usually the "
        "most actionable signal — run "
        "`python -m pipeline.improver_tools reviews`."
    )


def cmd_rejudge_all() -> None:
    """Re-judge all human-reviewed items with ALL judge prompts × ALL judge models."""
    from pipeline.config import load_config
    from pipeline.phase2.run import rejudge_all_prompts_and_models

    cfg = load_config()
    print("Re-judging all reviewed items across all judge prompts and models...")
    total = rejudge_all_prompts_and_models(cfg)
    print(f"Done. {total} new correlations saved.")


def _make_test_id(prefix: str) -> str:
    """Generate a unique test ID with timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def cmd_test_generate(
    prompt_path: str,
    item_ids: list[str] | None = None,
    n: int = 3,
    role: str = "judge",
    model_alias: str | None = None,
) -> None:
    """Generate with a prompt file without saving to main items table.

    Loads items from the latest iteration, runs generate_batch(save=False),
    saves a test_results entry.
    """
    from pipeline.config import (
        CHARTER_PATH,
        WRITING_GUIDELINES_PATH,
        load_config,
        resolve_generator_model,
    )
    from pipeline.api import make_api_client
    from pipeline.phase2.run import generate_batch
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    client, semaphore = make_api_client(
        cfg.phase2.endpoint, cfg.phase2.iteration.max_concurrent
    )
    alias = model_alias or cfg.phase2.generator_models[0].alias
    gen_model_cfg = resolve_generator_model(cfg, alias)
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    runs = load_runs()
    assert runs, "No iterations yet — run at least one iteration first"
    latest_iter = runs[-1]["iteration"]

    all_items = load_items_for_iteration(latest_iter)
    assert all_items, f"No items found for iteration {latest_iter}"

    if item_ids:
        items = [
            i
            for i in all_items
            if any(i["item_id"].startswith(iid) for iid in item_ids)
        ]
        assert items, f"No items matching {item_ids} in iteration {latest_iter}"
    else:
        items = random.sample(all_items, min(n, len(all_items)))

    prompt = Path(prompt_path)
    assert prompt.exists(), f"Prompt file not found: {prompt}"

    print(f"Test generating {len(items)} items with {prompt.name} (model={alias})...")
    generated = generate_batch(
        items,
        prompt,
        charter_text,
        gen_model_cfg.api_name,
        iteration=latest_iter,
        client=client,
        semaphore=semaphore,
        save=False,
        writing_guidelines_text=writing_guidelines_text,
        json_mode=gen_model_cfg.json_mode,
    )

    test_id = _make_test_id("tg")
    result_items = []
    for g in generated:
        result_items.append(
            {
                "item_id": g["item_id"],
                "preflection": g.get("preflection", "")[:200],
                "preflection_1p": g.get("preflection_1p", "")[:200],
                "reflection": g.get("reflection", "")[:200],
                "reflection_3p": g.get("reflection_3p", "")[:200],
                "charter_elements": g.get("charter_elements", []),
            }
        )

    record = {
        "test_id": test_id,
        "type": "generate",
        "role": role,
        "prompt": prompt.name,
        "model_alias": alias,
        "items": result_items,
        "summary": {"n_items": len(generated)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_test_result(record)

    print(f"\nTest {test_id}: generated {len(generated)} items")
    for g in generated:
        print(
            f"  {g['item_id'][:12]}: pre3p={g.get('preflection', '')[:40]}... pre1p={g.get('preflection_1p', '')[:40]}..."
        )
    print("Saved to test_results")


def cmd_test_judge(
    prompt_path: str,
    item_ids: list[str] | None = None,
    iteration: int | None = None,
    n: int = 3,
    role: str = "judge",
    model_alias: str | None = None,
) -> None:
    """Judge items with a prompt file without saving to main items table.

    Loads generated items from specified iteration, runs judge_batch(save=False),
    saves a test_results entry.
    """
    from pipeline.api import make_api_client
    from pipeline.config import load_config, resolve_judge_model
    from pipeline.phase2.run import judge_batch
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    client, semaphore = make_api_client(
        cfg.phase2.endpoint, cfg.phase2.iteration.max_concurrent
    )
    alias = model_alias or cfg.phase2.judge_models[0].alias
    jdg_model_cfg = resolve_judge_model(cfg, alias)

    runs = load_runs()
    assert runs, "No iterations yet — run at least one iteration first"
    iter_num = iteration if iteration is not None else runs[-1]["iteration"]

    all_items = load_items_for_iteration(iter_num)
    generated = [i for i in all_items if i.get("analysis")]
    assert generated, f"No generated items in iteration {iter_num}"

    if item_ids:
        items = [
            i
            for i in generated
            if any(i["item_id"].startswith(iid) for iid in item_ids)
        ]
        assert items, f"No items matching {item_ids} in iteration {iter_num}"
    else:
        items = random.sample(generated, min(n, len(generated)))

    prompt = Path(prompt_path)
    assert prompt.exists(), f"Prompt file not found: {prompt}"

    from pipeline.config import CHARTER_PATH, WRITING_GUIDELINES_PATH

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    print(f"Test judging {len(items)} items with {prompt.name} (model={alias})...")
    judged = judge_batch(
        items,
        prompt,
        jdg_model_cfg.api_name,
        iteration=iter_num,
        accept_threshold=cfg.phase2.scoring.accept_threshold,
        client=client,
        semaphore=semaphore,
        save=False,
        floor_threshold=cfg.phase2.scoring.floor_threshold,
        charter_text=charter_text,
        writing_guidelines_text=writing_guidelines_text,
    )

    scores = [j["judgment"]["aggregate"] for j in judged]
    n_acc = sum(1 for j in judged if j["judgment"]["decision"] == "accept")
    mean_score = statistics.mean(scores) if scores else 0.0

    test_id = _make_test_id("tj")
    result_items = []
    for j in judged:
        jdg = j["judgment"]
        part_scores = {
            part: part_j.get("scores", {})
            for part, part_j in _judgment_parts(jdg).items()
        }
        result_items.append(
            {
                "item_id": j["item_id"],
                "aggregate": jdg["aggregate"],
                "decision": jdg["decision"],
                **{f"{part}_scores": scores for part, scores in part_scores.items()},
            }
        )

    record = {
        "test_id": test_id,
        "type": "judge",
        "role": role,
        "prompt": prompt.name,
        "model_alias": alias,
        "items": result_items,
        "summary": {
            "n_items": len(judged),
            "mean_score": round(mean_score, 3),
            "n_accepted": n_acc,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_test_result(record)

    print(
        f"\nTest {test_id}: judged {len(judged)} items (mean={mean_score:.2f}, accepted={n_acc})"
    )
    for j in judged:
        jdg = j["judgment"]
        print(f"  {j['item_id'][:12]}: {jdg['decision']} ({jdg['aggregate']:.2f})")
    print("Saved to test_results")


def cmd_run_batch(role: str = "judge") -> None:
    """Run a full cross-iteration batch with the first model of the given role.

    Saves to main items + runs tables AND to test_results for tracking.
    """
    from pipeline.config import load_config
    from pipeline.phase2.run import (
        run_judge_cross_iteration,
        run_generator_cross_iteration,
    )

    cfg = load_config()

    if role == "judge":
        target = cfg.phase2.judge_models[0].alias
        source = "improve_judge"
        results = run_judge_cross_iteration(cfg, target, source=source)
    else:
        target = cfg.phase2.generator_models[0].alias
        source = "improve_generator"
        results = run_generator_cross_iteration(cfg, target, source=source)

    test_id = _make_test_id("tb")
    for result in results:
        mean_score = result["mean_score"]
        record = {
            "test_id": test_id,
            "type": "batch",
            "role": role,
            "target_alias": target,
            "generator_model": result["generator_model"],
            "judge_model": result["judge_model"],
            "group_id": result["group_id"],
            "items": [{"item_id": it["item_id"]} for it in result["items"]],
            "summary": {
                "n_items": result["n_items"],
                "mean_score": round(mean_score, 3),
                "n_accepted": result["n_accepted"],
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        save_test_result(record)
        print(
            f"  {result['generator_model']}/{result['judge_model']}: "
            f"{result['n_accepted']}/{result['n_items']} accepted, mean={mean_score:.2f}"
        )

    print(f"\nBatch {test_id} complete. group_id={results[0]['group_id']}")


def cmd_run_cross_batch(role: str, target: str) -> None:
    """Run a cross-iteration batch for a specific (role, target) pair.

    Judge cross-iteration: generate with ALL generators, judge with target.
    Generator cross-iteration: generate with target, judge with ALL judges.
    """
    from pipeline.config import load_config
    from pipeline.phase2.run import (
        run_judge_cross_iteration,
        run_generator_cross_iteration,
    )

    cfg = load_config()

    if role == "judge":
        results = run_judge_cross_iteration(cfg, target, source="improve_judge")
    else:
        results = run_generator_cross_iteration(cfg, target, source="improve_generator")

    group_id = results[0]["group_id"] if results else "none"
    print(f"\nCross-iteration complete (group_id={group_id}):")
    for r in results:
        print(
            f"  iter={r['iteration']} gen={r['generator_model']} judge={r['judge_model']}: "
            f"{r['n_accepted']}/{r['n_items']} accepted, mean={r['mean_score']:.2f}"
        )
    print(f"\nUse `cross_summary {group_id}` for aggregated stats.")
    print(
        "Use `failures <iteration>` or `scores <iteration>` to drill into a specific pair."
    )


def cmd_cross_summary(group_id: str) -> None:
    """Show aggregated stats across all iterations in a cross-iteration group."""
    from pipeline.phase2.storage import load_runs

    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]

    if not group_runs:
        print(f"No runs found with group_id starting with '{group_id}'")
        return

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    print(f"Cross-iteration summary (group_id={full_gid[:8]}...):")
    print(f"  {len(group_runs)} iterations\n")

    total_items = 0
    total_accepted = 0
    all_scores = []

    print(
        f"{'Iter':>6} {'Generator':>20} {'Judge':>20} {'Items':>6} {'Accept%':>8} {'Mean':>6}"
    )
    print("-" * 76)
    for r in group_runs:
        items = load_items_for_iteration(r["iteration"])
        judged = [i for i in items if i.get("judgment")]
        n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
        scores = [i["judgment"]["aggregate"] for i in judged]
        mean_s = statistics.mean(scores) if scores else 0.0
        acc_pct = n_acc / len(judged) * 100 if judged else 0

        total_items += len(judged)
        total_accepted += n_acc
        all_scores.extend(scores)

        print(
            f"{r['iteration']:>6} {r['generator_model']:>20} {r['judge_model']:>20} "
            f"{len(judged):>6} {acc_pct:>7.1f}% {mean_s:>6.2f}"
        )

    if all_scores:
        print("-" * 76)
        overall_acc = total_accepted / total_items * 100 if total_items else 0
        overall_mean = statistics.mean(all_scores)
        print(
            f"{'':>6} {'TOTAL':>20} {'':>20} {total_items:>6} {overall_acc:>7.1f}% {overall_mean:>6.2f}"
        )


def _collect_judged(iteration: int) -> list[dict]:
    """Load judged items for an iteration."""
    items = load_items_for_iteration(iteration)
    return [i for i in items if i.get("judgment")]


def _dim_scores_for_items(judged: list[dict]) -> dict[str, list[float]]:
    """Collect per-dimension score lists across all judgment parts."""
    dim_scores: dict[str, list[float]] = {}
    for item in judged:
        for part, part_j in _judgment_parts(item["judgment"]).items():
            for dim, score in part_j.get("scores", {}).items():
                dim_scores.setdefault(f"{part[:3]}_{dim[:3]}", []).append(score)
    return dim_scores


def _score_distribution(all_scores: list[float], label: str) -> None:
    """Print score distribution and ceiling/floor stats."""
    if not all_scores:
        return
    from collections import Counter as C

    counts = C(int(s) for s in all_scores)
    total = len(all_scores)
    print(f"\n  {label} score distribution (n={total}):")
    for s in sorted(counts):
        bar = "#" * counts[s]
        print(f"    {s}: {bar} ({counts[s]}, {counts[s]/total*100:.0f}%)")
    ceiling = sum(1 for s in all_scores if s >= 5) / total * 100
    floor = sum(1 for s in all_scores if s <= 2) / total * 100
    print(f"    ceiling (=5): {ceiling:.0f}%  |  floor (<=2): {floor:.0f}%")


def cmd_diagnose(group_id: str) -> None:
    """One-shot comprehensive analysis for a cross-iteration group.

    Combines: cross_summary, per-dimension means, score distributions,
    ceiling effects, floor-rule violations, decision flips, diversity,
    and gold/review availability.
    """
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs, load_latest_reviews

    cfg = load_config()
    floor_thresh = cfg.phase2.scoring.floor_threshold

    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    if not group_runs:
        print(f"No runs found with group_id starting with '{group_id}'")
        return

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    # --- 1. Cross-iteration summary table ---
    print(
        f"=== DIAGNOSE group_id={full_gid[:8]}... ({len(group_runs)} iterations) ===\n"
    )

    iter_data: dict[int, list[dict]] = {}
    iter_runs: dict[int, dict] = {}

    print(
        f"{'Iter':>6} {'Generator':>20} {'Judge':>20} "
        f"{'Items':>6} {'Acc%':>6} {'Mean':>6} {'Floor':>6}"
    )
    print("-" * 82)
    for r in group_runs:
        it = r["iteration"]
        judged = _collect_judged(it)
        iter_data[it] = judged
        iter_runs[it] = r

        n_acc = sum(1 for i in judged if i["judgment"]["decision"] == "accept")
        scores = [i["judgment"]["aggregate"] for i in judged]
        mean_s = statistics.mean(scores) if scores else 0.0
        acc_pct = n_acc / len(judged) * 100 if judged else 0

        n_floor = sum(
            1
            for i in judged
            if any(
                s <= floor_thresh
                for part_j in _judgment_parts(i["judgment"]).values()
                for s in part_j.get("scores", {}).values()
            )
        )

        print(
            f"{it:>6} {r['generator_model']:>20} {r['judge_model']:>20} "
            f"{len(judged):>6} {acc_pct:>5.0f}% {mean_s:>6.2f} {n_floor:>6}"
        )

    # --- 2. Per-dimension means per iteration ---
    print("\n--- Per-dimension means ---")
    all_dim_keys: list[str] = []
    for judged in iter_data.values():
        if not judged:
            continue
        for part, part_j in _judgment_parts(judged[0]["judgment"]).items():
            for dim in part_j.get("scores", {}):
                key = f"{part[:3]}_{dim[:3]}"
                if key not in all_dim_keys:
                    all_dim_keys.append(key)
        break

    header = "  " + f"{'Iter':>6}" + "".join(f" {k:>8}" for k in all_dim_keys)
    print(header)
    for it, judged in iter_data.items():
        dim_scores = _dim_scores_for_items(judged)
        vals = "".join(
            f" {statistics.mean(dim_scores.get(k, [0])):>8.2f}" for k in all_dim_keys
        )
        print(f"  {it:>6}{vals}")

    # --- 3. Score distribution & ceiling effect ---
    print("\n--- Score distribution (all dimensions pooled) ---")
    for it, judged in iter_data.items():
        dim_scores = _dim_scores_for_items(judged)
        all_vals = [s for vals in dim_scores.values() for s in vals]
        r = iter_runs[it]
        _score_distribution(
            all_vals,
            f"iter {it} (gen={r['generator_model']}, judge={r['judge_model']})",
        )

    # --- 4. Floor-rule violations ---
    print("\n--- Floor-rule violations (any dimension <= {}) ---".format(floor_thresh))
    for it, judged in iter_data.items():
        violations = []
        for item in judged:
            j = item["judgment"]
            for part, part_j in _judgment_parts(j).items():
                for dim, score in part_j.get("scores", {}).items():
                    if score <= floor_thresh:
                        violations.append(
                            (
                                item["item_id"][:12],
                                part[:3],
                                dim[:3],
                                score,
                                j["aggregate"],
                            )
                        )
        if violations:
            print(f"  iter {it}: {len(violations)} violations")
            for iid, p, d, sc, agg in violations:
                print(f"    {iid} {p}_{d}={sc} (agg={agg:.2f})")
        else:
            print(f"  iter {it}: none")

    # --- 5. Decision flips between iterations ---
    iters = sorted(iter_data.keys())
    if len(iters) >= 2:
        print(f"\n--- Decision flips between iterations ---")
        for i in range(len(iters)):
            for j in range(i + 1, len(iters)):
                _print_flips(iters[i], iters[j], iter_data, iter_runs)

    # --- 6. Diversity (compact) ---
    print("\n--- Diversity (reflection_1p openers) ---")
    for it, judged in iter_data.items():
        # Use reflection_1p text (falls back to legacy 'reflection' column)
        texts = [i.get("reflection", "") or "" for i in judged if i.get("reflection")]
        if not texts:
            continue
        first_words = Counter(t.split()[0] if t.split() else "" for t in texts)
        top3 = first_words.most_common(3)
        top_str = ", ".join(f'"{w}" {c}/{len(texts)}' for w, c in top3)

        openers = Counter(" ".join(t.split()[:5]) for t in texts)
        dupes = sum(1 for v in openers.values() if v > 1)

        print(
            f"  iter {it}: top words: {top_str}  |  duplicate 5-word openers: {dupes}"
        )

    # --- 7. Shared items (tells you which pairs work with `diff`) ---
    if len(iters) >= 2:
        print(f"\n--- Shared items (use `diff <iter1> <iter2>` on these pairs) ---")
        for i in range(len(iters)):
            for j in range(i + 1, len(iters)):
                ids_a = {item["item_id"] for item in iter_data[iters[i]]}
                ids_b = {item["item_id"] for item in iter_data[iters[j]]}
                shared = len(ids_a & ids_b)
                if shared > 0:
                    print(f"  iter {iters[i]} & {iters[j]}: {shared} shared items")
                else:
                    print(
                        f"  iter {iters[i]} & {iters[j]}: no shared items (different samples)"
                    )

    # --- 8. Gold & review availability ---
    reviews = load_latest_reviews()
    n_gold = sum(1 for judged in iter_data.values() for i in judged if i.get("is_gold"))
    print(f"\n--- Data availability ---")
    print(f"  Gold items in group: {n_gold}")
    print(f"  Human reviews total: {len(reviews)}")


def _print_flips(
    iter_a: int,
    iter_b: int,
    iter_data: dict[int, list[dict]],
    iter_runs: dict[int, dict],
) -> None:
    """Print decision flips between two iterations on shared items."""
    items_a = {i["item_id"]: i for i in iter_data[iter_a]}
    items_b = {i["item_id"]: i for i in iter_data[iter_b]}
    shared = set(items_a) & set(items_b)
    if not shared:
        print(f"  iter {iter_a} vs {iter_b}: no shared items")
        return

    flips = []
    for iid in sorted(shared):
        da = items_a[iid]["judgment"]["decision"]
        db = items_b[iid]["judgment"]["decision"]
        if da != db:
            sa = items_a[iid]["judgment"]["aggregate"]
            sb = items_b[iid]["judgment"]["aggregate"]
            flips.append((iid[:12], da, sa, db, sb))

    ra = iter_runs[iter_a]
    rb = iter_runs[iter_b]
    agree = len(shared) - len(flips)
    print(
        f"  iter {iter_a} ({ra['generator_model']}) vs "
        f"iter {iter_b} ({rb['generator_model']}): "
        f"{agree}/{len(shared)} agree, {len(flips)} flips"
    )
    for iid, da, sa, db, sb in flips:
        print(f"    {iid}: {da}({sa:.1f}) -> {db}({sb:.1f}) [diff={sb-sa:+.1f}]")


def cmd_diff(iter_a: int, iter_b: int, limit: int = 10) -> None:
    """Cross-iteration comparison of shared items between two iterations.

    Shows: decision agreement, per-dimension score diffs, and full
    preflection/reflection text for items that flipped accept<->reject.
    """
    judged_a = _collect_judged(iter_a)
    judged_b = _collect_judged(iter_b)
    if not judged_a:
        print(f"No judged items in iteration {iter_a}")
        return
    if not judged_b:
        print(f"No judged items in iteration {iter_b}")
        return

    by_id_a = {i["item_id"]: i for i in judged_a}
    by_id_b = {i["item_id"]: i for i in judged_b}
    shared_ids = sorted(set(by_id_a) & set(by_id_b))

    if not shared_ids:
        print(f"No shared items between iteration {iter_a} and {iter_b}")
        return

    # --- Agreement stats ---
    both_acc = both_rej = only_a_acc = only_b_acc = 0
    flips: list[tuple[str, dict, dict]] = []
    agg_diffs: list[float] = []
    dim_diffs: dict[str, list[float]] = {}

    for iid in shared_ids:
        ja = by_id_a[iid]["judgment"]
        jb = by_id_b[iid]["judgment"]
        da, db = ja["decision"], jb["decision"]
        agg_diffs.append(jb["aggregate"] - ja["aggregate"])

        if da == "accept" and db == "accept":
            both_acc += 1
        elif da == "reject" and db == "reject":
            both_rej += 1
        elif da == "accept" and db == "reject":
            only_a_acc += 1
            flips.append((iid, by_id_a[iid], by_id_b[iid]))
        else:
            only_b_acc += 1
            flips.append((iid, by_id_a[iid], by_id_b[iid]))

        all_parts_ab = set(_judgment_parts(ja)) | set(_judgment_parts(jb))
        for part in all_parts_ab:
            sa = ja.get(part, {}).get("scores", {})
            sb = jb.get(part, {}).get("scores", {})
            for dim in set(sa) & set(sb):
                key = f"{part[:3]}_{dim[:3]}"
                dim_diffs.setdefault(key, []).append(sb[dim] - sa[dim])

    total = len(shared_ids)
    agree = both_acc + both_rej
    print(f"=== DIFF iter {iter_a} vs iter {iter_b} ({total} shared items) ===\n")
    print(f"  Decision agreement: {agree}/{total} ({agree/total*100:.0f}%)")
    print(f"    both accept:  {both_acc}")
    print(f"    both reject:  {both_rej}")
    print(f"    only {iter_a} acc: {only_a_acc}")
    print(f"    only {iter_b} acc: {only_b_acc}")

    # --- Aggregate score diff ---
    print(f"\n  Aggregate score diff (iter{iter_b} - iter{iter_a}):")
    print(f"    mean: {statistics.mean(agg_diffs):+.3f}")
    print(f"    stdev: {statistics.stdev(agg_diffs):.3f}" if len(agg_diffs) > 1 else "")

    # --- Per-dimension diffs ---
    print(f"\n  Per-dimension mean diff (iter{iter_b} - iter{iter_a}):")
    for key in sorted(dim_diffs):
        vals = dim_diffs[key]
        print(
            f"    {key}: {statistics.mean(vals):+.2f} (stdev={statistics.stdev(vals):.2f})"
            if len(vals) > 1
            else f"    {key}: {statistics.mean(vals):+.2f}"
        )

    # --- Flipped items with text ---
    flips.sort(
        key=lambda x: abs(
            x[2]["judgment"]["aggregate"] - x[1]["judgment"]["aggregate"]
        ),
        reverse=True,
    )
    shown = flips[:limit]
    if shown:
        print(f"\n--- Decision flips ({len(flips)} total, showing {len(shown)}) ---")
    for iid, item_a, item_b in shown:
        ja = item_a["judgment"]
        jb = item_b["judgment"]
        print(
            f"\n  {iid[:16]}: {ja['decision']}({ja['aggregate']:.1f}) -> "
            f"{jb['decision']}({jb['aggregate']:.1f}) [diff={jb['aggregate']-ja['aggregate']:+.1f}]"
        )

        # Per-dimension comparison
        for part in set(_judgment_parts(ja)) | set(_judgment_parts(jb)):
            sa = ja.get(part, {}).get("scores", {})
            sb = jb.get(part, {}).get("scores", {})
            changes = []
            for dim in sa:
                if dim in sb and sa[dim] != sb[dim]:
                    changes.append(f"{dim[:3]}={sa[dim]}->{sb[dim]}")
            if changes:
                print(f"    {part[:3]} changes: {', '.join(changes)}")

        # Show text for context
        print(f"    text: {item_a['text'][:120]}...")
        print(f"    iter{iter_a} pre: {item_a.get('preflection', '')[:100]}...")
        print(f"    iter{iter_b} pre: {item_b.get('preflection', '')[:100]}...")
        print(f"    iter{iter_a} ref: {item_a.get('reflection', '')[:100]}...")
        print(f"    iter{iter_b} ref: {item_b.get('reflection', '')[:100]}...")


def cmd_test_results(
    phase: str | None = None, type_filter: str | None = None, role: str | None = None
) -> None:
    """List test results, optionally filtered by phase/role and/or type."""
    results = load_test_results(phase=phase, role=role)
    if type_filter:
        results = [r for r in results if r.get("type") == type_filter]

    if not results:
        print("No test results found.")
        return

    print(f"Test results ({len(results)} entries):\n")
    for r in results:
        summary = r.get("summary", {})
        n = summary.get("n_items", "?")
        mean = summary.get("mean_score", "")
        acc = summary.get("n_accepted", "")
        mean_str = f" mean={mean:.2f}" if isinstance(mean, (int, float)) else ""
        acc_str = f" acc={acc}" if acc != "" else ""
        print(
            f"  {r['test_id']}  {r['type']:>8}  phase={r.get('phase', '?')}  "
            f"prompt={r.get('prompt', '?')}  n={n}{mean_str}{acc_str}  "
            f"{r.get('timestamp', '')[:19]}"
        )


# --- Phase 3 commands ---


def cmd_run_paired_batch(role: str, target: str) -> None:
    """Run a paired iteration for phase3 (all golds + target model)."""
    from pipeline.config import load_config
    from pipeline.phase3.run import run_paired_iteration

    cfg = load_config()
    results = run_paired_iteration(cfg, role, target, source=f"phase3_{role}")
    group_id = results[0]["group_id"] if results else "none"
    print(f"\nPaired iteration complete (group_id={group_id}):")
    for r in results:
        print(
            f"  iter={r['iteration']} gen={r['generator_model']} judge={r['judge_model']}: "
            f"{r['n_accepted']}/{r['n_items']} accepted, mean={r['mean_score']:.2f}"
        )
    print(f"\nUse `paired_summary {group_id}` for correlation metrics.")


def cmd_paired_summary(group_id: str) -> None:
    """Show correlation metrics for a phase3 paired run."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs
    from pipeline.phase3.run import compute_paired_correlation

    cfg = load_config()
    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    assert group_runs, f"No runs found with group_id starting with '{group_id}'"

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    target_aliases = {m.alias for m in cfg.phase3.target_models}
    gold_judge_aliases = {m.alias for m in cfg.phase3.gold_judges}
    gold_gen_aliases = {m.alias for m in cfg.phase3.gold_generators}

    print(f"Paired summary (group_id={full_gid[:8]}...):")
    print(f"  {len(group_runs)} iterations\n")

    gold_runs = [
        r
        for r in group_runs
        if r["judge_model"] in gold_judge_aliases
        and r["generator_model"] in gold_gen_aliases
    ]
    target_runs = [
        r
        for r in group_runs
        if r["judge_model"] in target_aliases or r["generator_model"] in target_aliases
    ]

    for target_run in target_runs:
        target_items = load_items_for_iteration(target_run["iteration"])
        target_judged = [i for i in target_items if i.get("judgment")]

        for gold_run in gold_runs:
            if gold_run["generator_model"] == target_run["generator_model"] or (
                gold_run["judge_model"] == target_run["judge_model"]
            ):
                gold_items = load_items_for_iteration(gold_run["iteration"])
                gold_judged = [i for i in gold_items if i.get("judgment")]

                if not gold_judged or not target_judged:
                    continue

                corr = compute_paired_correlation(gold_judged, target_judged)
                print(
                    f"  Gold({gold_run['generator_model']}/{gold_run['judge_model']}) "
                    f"vs Target({target_run['generator_model']}/{target_run['judge_model']}):"
                )
                print(f"    n_items: {corr['n_items']}")
                print(f"    Decision concordance: {corr['decision_concordance']:.1%}")
                print(f"    Spearman rho: {corr['aggregate_spearman']:.3f}")
                print(f"    Pearson r: {corr['aggregate_pearson']:.3f}")
                print(f"    Cohen's kappa: {corr['cohens_kappa']:.3f}")
                print("    Per-dimension:")
                for dim, stats in sorted(corr["per_dimension"].items()):
                    print(
                        f"      {dim}: rho={stats['spearman']:.3f} "
                        f"MAD={stats['mean_abs_diff']:.2f} "
                        f"bias={stats['mean_diff']:+.2f}"
                    )
                print()


def cmd_disagreements(group_id: str, limit: int = 20) -> None:
    """Show items where gold and target disagree on decision within a group."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    assert group_runs, f"No runs found with group_id starting with '{group_id}'"

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    target_aliases = {m.alias for m in cfg.phase3.target_models}

    all_items_by_model: dict[str, dict[str, dict]] = {}
    for run in group_runs:
        model_key = f"{run['generator_model']}/{run['judge_model']}"
        items = load_items_for_iteration(run["iteration"])
        judged = {i["item_id"]: i for i in items if i.get("judgment")}
        all_items_by_model[model_key] = judged

    disagreements = []
    target_keys = [k for k in all_items_by_model if any(a in k for a in target_aliases)]
    gold_keys = [k for k in all_items_by_model if k not in target_keys]

    for t_key in target_keys:
        t_items = all_items_by_model[t_key]
        for g_key in gold_keys:
            g_items = all_items_by_model[g_key]
            shared = set(t_items) & set(g_items)
            for item_id in shared:
                t_dec = t_items[item_id]["judgment"]["decision"]
                g_dec = g_items[item_id]["judgment"]["decision"]
                if t_dec != g_dec:
                    t_agg = t_items[item_id]["judgment"]["aggregate"]
                    g_agg = g_items[item_id]["judgment"]["aggregate"]
                    disagreements.append(
                        {
                            "item_id": item_id,
                            "target": t_key,
                            "gold": g_key,
                            "target_decision": t_dec,
                            "gold_decision": g_dec,
                            "target_score": t_agg,
                            "gold_score": g_agg,
                            "score_diff": abs(t_agg - g_agg),
                        }
                    )

    disagreements.sort(key=lambda d: d["score_diff"], reverse=True)
    print(
        f"Decision disagreements ({len(disagreements)} total, showing {min(limit, len(disagreements))}):\n"
    )
    for d in disagreements[:limit]:
        print(
            f"  {d['item_id'][:16]}  target={d['target_decision']}({d['target_score']:.2f}) "
            f"gold={d['gold_decision']}({d['gold_score']:.2f})  diff={d['score_diff']:.2f}"
        )
        print(f"    target: {d['target']}  gold: {d['gold']}")


def cmd_dimension_alignment(group_id: str) -> None:
    """Show per-dimension mean scores for gold vs target within a group."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    assert group_runs, f"No runs found with group_id starting with '{group_id}'"

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    target_aliases = {m.alias for m in cfg.phase3.target_models}

    gold_dim_scores: dict[str, list[float]] = {}
    target_dim_scores: dict[str, list[float]] = {}

    for run in group_runs:
        is_target = (
            run["judge_model"] in target_aliases
            or run["generator_model"] in target_aliases
        )
        dest = target_dim_scores if is_target else gold_dim_scores
        items = load_items_for_iteration(run["iteration"])
        for item in items:
            if not item.get("judgment"):
                continue
            for part, part_j in _judgment_parts(item["judgment"]).items():
                for dim, score in part_j.get("scores", {}).items():
                    key = f"{part}_{dim}"
                    dest.setdefault(key, []).append(score)

    print(f"Dimension alignment (group_id={full_gid[:8]}...):\n")
    print(f"{'Dimension':>30} {'Gold Mean':>10} {'Target Mean':>12} {'Diff':>8}")
    print("-" * 64)
    all_dims = sorted(set(gold_dim_scores) | set(target_dim_scores))
    for dim in all_dims:
        g_mean = statistics.mean(gold_dim_scores[dim]) if dim in gold_dim_scores else 0
        t_mean = (
            statistics.mean(target_dim_scores[dim]) if dim in target_dim_scores else 0
        )
        print(f"{dim:>30} {g_mean:>10.2f} {t_mean:>12.2f} {t_mean - g_mean:>+8.2f}")


def cmd_paired_show(item_id: str, group_id: str) -> None:
    """Show side-by-side gold vs target outputs for one item in a group."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    assert group_runs, f"No runs found with group_id starting with '{group_id}'"

    full_gid = group_runs[0]["group_id"]
    group_runs = [r for r in runs if r.get("group_id") == full_gid]

    target_aliases = {m.alias for m in cfg.phase3.target_models}

    print(f"Paired show: item={item_id[:16]} group={full_gid[:8]}...\n")

    for run in group_runs:
        is_target = (
            run["judge_model"] in target_aliases
            or run["generator_model"] in target_aliases
        )
        label = "TARGET" if is_target else "GOLD"
        items = load_items_for_iteration(run["iteration"])
        matches = [i for i in items if i["item_id"].startswith(item_id)]
        if not matches:
            continue
        item = matches[0]
        print(
            f"--- [{label}] gen={run['generator_model']} judge={run['judge_model']} "
            f"(iter={run['iteration']}) ---"
        )
        print(f"  Text preview: {item['text'][:200]}...")
        if item.get("reflection"):
            print(f"  Reflection: {item['reflection'][:300]}...")
        if item.get("judgment"):
            j = item["judgment"]
            print(f"  Decision: {j['decision']}  Aggregate: {j['aggregate']:.2f}")
            for part, part_j in _judgment_parts(j).items():
                print(f"  {part}: {part_j.get('scores', {})}")
        print()


def cmd_escalate(item_id: str, group_id: str, reason: str) -> None:
    """Flag an item for human review."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs
    from pipeline.phase3.storage import save_escalation

    cfg = load_config()
    runs = load_runs()
    group_runs = [
        r for r in runs if r.get("group_id") and r["group_id"].startswith(group_id)
    ]
    assert group_runs, f"No runs found with group_id starting with '{group_id}'"

    full_gid = group_runs[0]["group_id"]
    target_aliases = {m.alias for m in cfg.phase3.target_models}
    gold_aliases = {
        m.alias for m in cfg.phase3.gold_judges + cfg.phase3.gold_generators
    }

    target_model = "unknown"
    gold_model = "unknown"
    role = "unknown"
    for run in [r for r in runs if r.get("group_id") == full_gid]:
        if run["judge_model"] in target_aliases:
            target_model = run["judge_model"]
            gold_model = run["generator_model"]
            role = "judge"
        elif run["generator_model"] in target_aliases:
            target_model = run["generator_model"]
            gold_model = run["judge_model"]
            role = "generator"

    esc_id = save_escalation(
        item_id=item_id,
        group_id=full_gid,
        gold_model=gold_model,
        target_model=target_model,
        role=role,
        reason=reason,
    )
    print(
        f"Escalation created (id={esc_id}): item={item_id[:16]} reason={reason[:100]}"
    )


def cmd_escalations(status: str | None = None) -> None:
    """List escalated items."""
    from pipeline.phase3.storage import load_escalations

    escalations = load_escalations(status=status)
    if not escalations:
        print("No escalations found.")
        return

    print(f"Escalations ({len(escalations)} total):\n")
    for e in escalations:
        print(
            f"  #{e['id']} [{e['status']}] item={e['item_id'][:16]} "
            f"target={e['target_model']} role={e['role']}"
        )
        print(f"    reason: {e['reason'][:200]}")
        if e.get("reviewer_notes"):
            print(f"    notes: {e['reviewer_notes'][:200]}")


def cmd_correlation_trend(target: str | None = None) -> None:
    """Show Spearman rho and decision concordance over phase3 iterations."""
    from pipeline.config import load_config
    from pipeline.phase2.storage import load_runs
    from pipeline.phase3.run import compute_paired_correlation

    cfg = load_config()
    runs = load_runs()
    phase3_runs = [r for r in runs if r.get("phase") == "phase3"]
    if not phase3_runs:
        print("No phase3 runs found.")
        return

    target_aliases = {m.alias for m in cfg.phase3.target_models}
    if target:
        target_aliases = {target}

    group_ids = list(
        dict.fromkeys(r["group_id"] for r in phase3_runs if r.get("group_id"))
    )

    print(f"{'Group':>10} {'Target':>20} {'Spearman':>10} {'Concordance':>12} {'N':>5}")
    print("-" * 62)

    for gid in group_ids:
        g_runs = [r for r in phase3_runs if r.get("group_id") == gid]
        gold_judge_aliases = {m.alias for m in cfg.phase3.gold_judges}
        gold_gen_aliases = {m.alias for m in cfg.phase3.gold_generators}

        gold_runs = [
            r
            for r in g_runs
            if r["judge_model"] in gold_judge_aliases
            and r["generator_model"] in gold_gen_aliases
        ]
        t_runs = [
            r
            for r in g_runs
            if r["judge_model"] in target_aliases
            or r["generator_model"] in target_aliases
        ]

        for t_run in t_runs:
            t_model = (
                t_run["judge_model"]
                if t_run["judge_model"] in target_aliases
                else t_run["generator_model"]
            )
            target_items = load_items_for_iteration(t_run["iteration"])
            target_judged = [i for i in target_items if i.get("judgment")]

            for g_run in gold_runs:
                gold_items = load_items_for_iteration(g_run["iteration"])
                gold_judged = [i for i in gold_items if i.get("judgment")]
                if not gold_judged or not target_judged:
                    continue

                corr = compute_paired_correlation(gold_judged, target_judged)
                rho = corr["aggregate_spearman"]
                conc = corr["decision_concordance"]
                rho_str = f"{rho:.3f}" if rho == rho else "N/A"
                print(
                    f"{gid[:8]:>10} {t_model:>20} {rho_str:>10} "
                    f"{conc:>11.1%} {corr['n_items']:>5}"
                )


def cmd_parse_stats(iteration: int) -> None:
    """Show generation parse success/failure counts for an iteration.

    Reads the run config to find n_attempted and n_gen_failed.
    Falls back to comparing item counts if config data unavailable.
    """
    from pipeline.phase2.storage import load_runs

    runs = load_runs()
    run = next((r for r in runs if r["iteration"] == iteration), None)
    if not run:
        print(f"No run found for iteration {iteration}")
        return

    config = run.get("config", {})
    if isinstance(config, str):
        config = json.loads(config)

    items = load_items_for_iteration(iteration)
    n_judged = len([i for i in items if i.get("judgment")])
    n_generated = len([i for i in items if i.get("analysis")])

    n_attempted = config.get("n_attempted")
    n_gen_failed = config.get("n_gen_failed")

    if n_attempted is not None:
        print(f"Iteration {iteration} parse stats:")
        print(f"  Attempted:       {n_attempted}")
        print(f"  Parsed OK:       {n_attempted - n_gen_failed}")
        print(
            f"  Parse failures:  {n_gen_failed} ({n_gen_failed / n_attempted * 100:.0f}%)"
        )
        print(f"  Judged:          {n_judged}")
    else:
        # Fallback for older runs without config data
        print(f"Iteration {iteration} (no parse stats in run config — older run):")
        print(f"  Generated items: {n_generated}")
        print(f"  Judged items:    {n_judged}")
        print(
            f"  (Parse failure count unavailable — run was before parse_stats tracking)"
        )

    print(f"\n  Generator: {run.get('generator_model', '?')}")
    print(f"  Judge:     {run.get('judge_model', '?')}")
    print(f"  Prompt:    {run.get('gen_prompt', '?')}")


def cmd_rollback(alias: str, role: str, version: int) -> None:
    """Promote a specific version to be the latest by copying it to v(max+1).

    The pipeline always uses the highest _vN.md file. If v2 performed best but
    v3 and v4 exist, `rollback <alias> generator 2` copies v2 to v5, making it
    the active prompt.
    """
    import glob as _glob
    import shutil

    from pipeline.config import PIPELINE_DATA_DIR

    prompts_dir = PIPELINE_DATA_DIR / "prompts" / alias
    assert (
        prompts_dir.exists()
    ), f"No prompt directory for alias '{alias}': {prompts_dir}"

    prefix = "judge" if role == "judge" else "generator"
    source = prompts_dir / f"{prefix}_v{version}.md"
    assert source.exists(), f"Version file not found: {source}"

    # Find current max version
    import re

    pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.md$")
    max_v = 0
    for p in prompts_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            max_v = max(max_v, int(m.group(1)))

    if max_v == version:
        print(f"v{version} is already the latest version — nothing to do.")
        return

    new_v = max_v + 1
    dest = prompts_dir / f"{prefix}_v{new_v}.md"
    shutil.copy2(source, dest)
    print(f"Rolled back: copied {source.name} → {dest.name}")
    print(f"Active prompt is now {dest.name} (content identical to v{version})")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]

    if "--help" in args or "-h" in args or cmd == "help":
        print(__doc__)
        print(
            "\nFor detailed help on a specific command: python -m pipeline.improver_tools <command> --help"
        )
        print("\nKey concepts:")
        print(
            "  - item_id: hex string prefix (8+ chars), copied from scores/failures output"
        )
        print("  - iteration: integer, shown in run_cross_batch output")
        print(
            "  - group_id: UUID prefix (8+ chars), links iterations from same cross-batch"
        )
        print(
            "  - --part: preflection_3p | preflection_1p | reflection_1p | reflection_3p"
        )
        print("  - --dim: relevance | specificity | charter_grounding | voice_tone")
        sys.exit(0)

    positional = [a for a in args[1:] if not a.startswith("--")]

    def _require_positional(n: int, usage: str):
        if len(positional) < n:
            print(f"Usage: python -m pipeline.improver_tools {usage}")
            sys.exit(1)

    def _get_flag(flag: str, default: str | None = None) -> str | None:
        if flag in args:
            return args[args.index(flag) + 1]
        return default

    def _get_flag_int(flag: str, default: int | None = None) -> int | None:
        val = _get_flag(flag)
        return int(val) if val is not None else default

    if cmd == "summary":
        _require_positional(1, "summary <iteration>")
        cmd_summary(int(positional[0]))
    elif cmd == "failures":
        _require_positional(
            1, "failures <iteration> [--limit N] [--offset N] [--reasoning-limit N]"
        )
        cmd_failures(
            int(positional[0]),
            limit=_get_flag_int("--limit", 10),
            reasoning_limit=_get_flag_int("--reasoning-limit", 200),
            offset=_get_flag_int("--offset", 0),
        )
    elif cmd == "accepts":
        _require_positional(
            1,
            "accepts <iteration> [--limit N] [--offset N] [--reasoning-limit N] "
            "[--sort top|borderline]",
        )
        cmd_accepts(
            int(positional[0]),
            limit=_get_flag_int("--limit", 10),
            reasoning_limit=_get_flag_int("--reasoning-limit", 200),
            offset=_get_flag_int("--offset", 0),
            sort=_get_flag("--sort", "top"),
        )
    elif cmd == "show":
        brief = "--brief" in args
        gold_only = "--gold" in args
        if gold_only:
            _require_positional(1, "show --gold <iteration> [--brief]")
            cmd_show([], int(positional[0]), brief=brief, gold_only=True)
        else:
            _require_positional(2, "show <item_id>[,id2,...] <iteration> [--brief]")
            cmd_show(positional[0].split(","), int(positional[1]), brief=brief)
    elif cmd == "item":
        _require_positional(2, "item <item_id> <iteration>")
        cmd_item(positional[0], int(positional[1]))
    elif cmd == "reasoning":
        _require_positional(2, "reasoning <item_id>[,id2,...] <iteration>")
        cmd_reasoning(positional[0].split(","), int(positional[1]))
    elif cmd == "diversity":
        _require_positional(1, "diversity <iteration>")
        cmd_diversity(int(positional[0]))
    elif cmd == "scores":
        _require_positional(1, "scores <iteration>")
        cmd_scores(int(positional[0]))
    elif cmd == "distribution":
        _require_positional(1, "distribution <iteration>")
        cmd_distribution(int(positional[0]))
    elif cmd == "gold":
        cmd_gold(
            limit=_get_flag_int("--limit", 5),
            offset=_get_flag_int("--offset", 0),
            verbose="--verbose" in args,
        )
    elif cmd == "compare":
        _require_positional(2, "compare <item_id> <iteration>")
        cmd_compare(positional[0], int(positional[1]))
    elif cmd == "reviews":
        judge_prompt_arg = positional[0] if positional else None
        cmd_reviews(judge_prompt=judge_prompt_arg, limit=_get_flag_int("--limit", 20))
    elif cmd == "filter":
        _require_positional(
            1,
            "filter <iteration> --dim X (--below N | --above N) "
            "[--part preflection_3p|preflection_1p|reflection_1p|reflection_3p]",
        )
        below_arg = _get_flag("--below")
        above_arg = _get_flag("--above")
        cmd_filter(
            int(positional[0]),
            dim=_get_flag("--dim"),
            below=float(below_arg) if below_arg is not None else None,
            above=float(above_arg) if above_arg is not None else None,
            part=_get_flag("--part"),
        )
    elif cmd == "trend":
        cmd_trend()
    elif cmd == "test_generate":
        _require_positional(
            1,
            "test_generate <prompt_path> [--items id1,id2,...] [--n N] [--role judge|generator] [--model ALIAS]",
        )
        item_ids_str = _get_flag("--items")
        item_ids = item_ids_str.split(",") if item_ids_str else None
        cmd_test_generate(
            positional[0],
            item_ids=item_ids,
            n=_get_flag_int("--n", 3),
            role=_get_flag("--role", _get_flag("--phase", "judge")),
            model_alias=_get_flag("--model"),
        )
    elif cmd == "test_judge":
        _require_positional(
            1,
            "test_judge <prompt_path> [--items id1,id2,...] [--iteration N] [--role judge|generator] [--model ALIAS]",
        )
        item_ids_str = _get_flag("--items")
        item_ids = item_ids_str.split(",") if item_ids_str else None
        cmd_test_judge(
            positional[0],
            item_ids=item_ids,
            iteration=_get_flag_int("--iteration"),
            n=_get_flag_int("--n", 3),
            role=_get_flag("--role", _get_flag("--phase", "judge")),
            model_alias=_get_flag("--model"),
        )
    elif cmd == "run_batch":
        cmd_run_batch(role=_get_flag("--role", _get_flag("--phase", "judge")))
    elif cmd == "run_cross_batch":
        role = _get_flag("--role")
        target = _get_flag("--target")
        assert role, "Usage: run_cross_batch --role judge|generator --target <alias>"
        assert target, "Usage: run_cross_batch --role judge|generator --target <alias>"
        cmd_run_cross_batch(role=role, target=target)
    elif cmd == "cross_summary":
        _require_positional(1, "cross_summary <group_id>")
        cmd_cross_summary(positional[0])
    elif cmd == "diagnose":
        _require_positional(1, "diagnose <group_id>")
        cmd_diagnose(positional[0])
    elif cmd == "diff":
        _require_positional(2, "diff <iter1> <iter2> [--limit N]")
        cmd_diff(
            int(positional[0]),
            int(positional[1]),
            limit=_get_flag_int("--limit", 10),
        )
    elif cmd == "test_results":
        cmd_test_results(
            phase=_get_flag("--phase"),
            type_filter=_get_flag("--type"),
            role=_get_flag("--role"),
        )
    elif cmd == "correlations":
        cmd_correlations()
    elif cmd == "rejudge_all":
        cmd_rejudge_all()
    # Phase 3 commands
    elif cmd == "run_paired_batch":
        role = _get_flag("--role")
        target = _get_flag("--target")
        assert role, "Usage: run_paired_batch --role judge|generator --target <alias>"
        assert target, "Usage: run_paired_batch --role judge|generator --target <alias>"
        cmd_run_paired_batch(role=role, target=target)
    elif cmd == "paired_summary":
        _require_positional(1, "paired_summary <group_id>")
        cmd_paired_summary(positional[0])
    elif cmd == "disagreements":
        _require_positional(1, "disagreements <group_id> [--limit N]")
        cmd_disagreements(positional[0], limit=_get_flag_int("--limit", 20))
    elif cmd == "dimension_alignment":
        _require_positional(1, "dimension_alignment <group_id>")
        cmd_dimension_alignment(positional[0])
    elif cmd == "paired_show":
        _require_positional(2, "paired_show <item_id> <group_id>")
        cmd_paired_show(positional[0], positional[1])
    elif cmd == "escalate":
        _require_positional(2, "escalate <item_id> <group_id> --reason '...'")
        reason = _get_flag("--reason")
        assert reason, "Usage: escalate <item_id> <group_id> --reason '...'"
        cmd_escalate(positional[0], positional[1], reason)
    elif cmd == "escalations":
        cmd_escalations(status=_get_flag("--status"))
    elif cmd == "correlation_trend":
        cmd_correlation_trend(target=_get_flag("--target"))
    elif cmd == "parse_stats":
        _require_positional(1, "parse_stats <iteration>")
        cmd_parse_stats(int(positional[0]))
    elif cmd == "rollback":
        _require_positional(3, "rollback <alias> <role> <version>")
        cmd_rollback(positional[0], positional[1], int(positional[2]))
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
