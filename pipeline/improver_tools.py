"""CLI tools for the improver agent to query iteration data and run tests.

Usage (via Bash tool):
    python -m pipeline.improver_tools summary <iteration> [--mode reflection|preflection]
    python -m pipeline.improver_tools failures <iteration> [--limit N] [--offset N] [--reasoning-limit N] [--mode reflection|preflection]
    python -m pipeline.improver_tools accepts <iteration> [--limit N] [--offset N] [--reasoning-limit N] [--sort top|borderline] [--mode reflection|preflection]
    python -m pipeline.improver_tools show <item_id>[,id2,...] <iteration> [--brief]
    python -m pipeline.improver_tools show --gold <iteration> [--brief]
    python -m pipeline.improver_tools item <item_id> <iteration>
    python -m pipeline.improver_tools reasoning <item_id>[,id2,...] <iteration>
    python -m pipeline.improver_tools diversity <iteration>
    python -m pipeline.improver_tools scores <iteration> [--mode reflection|preflection]
    python -m pipeline.improver_tools distribution <iteration>
    python -m pipeline.improver_tools gold [--limit N] [--offset N] [--verbose]
    python -m pipeline.improver_tools compare <item_id> <iteration>
    python -m pipeline.improver_tools reviews [<judge_prompt>] [--limit N] [--offset N]
    python -m pipeline.improver_tools filter <iteration> --dim X (--below N | --above N) [--part PART]  (reflection: reflection_1p, reflection_3p; preflection: charter_summary, neutral, judgemental, idealisation [new] or preflection_3p, preflection_1p [legacy])
    python -m pipeline.improver_tools trend [--mode reflection|preflection]
    python -m pipeline.improver_tools diagnose <group_id> [--mode reflection|preflection]
    python -m pipeline.improver_tools diff <iter1> <iter2> [--limit N] [--mode reflection|preflection]
    python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2,...] [--n N] [--role judge|generator]
    python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2,...] [--iteration N] [--role judge|generator] [--mode reflection|preflection]
    python -m pipeline.improver_tools run_batch [--role judge|generator] [--mode reflection|preflection]
    python -m pipeline.improver_tools run_cross_batch --role judge|generator --target <alias> [--mode reflection|preflection]
    python -m pipeline.improver_tools cross_summary <group_id> [--mode reflection|preflection]
    python -m pipeline.improver_tools test_results [--role judge|generator] [--type generate|judge|batch]
    python -m pipeline.improver_tools correlations [--all]
    python -m pipeline.improver_tools rejudge_all [--mode reflection|preflection]
    python -m pipeline.improver_tools parse_stats <iteration>
    python -m pipeline.improver_tools rollback <alias> <role> <version> [--mode reflection|preflection]
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

from pipeline.generation import (
    MODE_PART_NAMES as _MODE_PART_NAMES,
    PREFLECTION_FIELDS_CURRENT as _PREFLECTION_FIELDS_CURRENT,
    PREFLECTION_PART_NAMES as _PREFLECTION_PART_NAMES,
    REFLECTION_PART_NAMES as _REFLECTION_PART_NAMES,
    REFLECTION_VOICES as _REFLECTION_VOICES,
)
from pipeline.phase2.run import (
    JUDGMENT_NON_PART_KEYS as _JUDGMENT_NON_PART_KEYS,
    judgment_parts as _judgment_parts,
)

_VALID_PART_NAMES = _REFLECTION_PART_NAMES | _PREFLECTION_PART_NAMES


def _validate_mode(mode: str | None) -> str | None:
    """Validate --mode flag value. Returns the mode or None. Prints error and exits on invalid."""
    if mode is None:
        return None
    if mode not in ("reflection", "preflection"):
        print(f"Invalid --mode: {mode!r}. Use 'reflection' or 'preflection'.")
        sys.exit(1)
    return mode


def _mode_judgment_parts(judgment: dict, mode: str | None) -> dict[str, dict]:
    """Return judgment voice/field dicts, filtered to the given mode if set.

    Filtering is by membership in the mode's part-name set (union of legacy
    and current schemas) so a judgment emitted under either generation yields
    its natural set of parts.
    """
    parts = _judgment_parts(judgment)
    if mode is None:
        return parts
    part_names = _MODE_PART_NAMES[mode]
    return {k: v for k, v in parts.items() if k in part_names}


def _mode_decision_key(mode: str | None) -> str:
    """Return the judgment key for the decision, scoped by mode."""
    return f"{mode}_decision" if mode else "decision"


def _mode_aggregate_key(mode: str | None) -> str:
    """Return the judgment key for the aggregate score, scoped by mode."""
    return f"{mode}_aggregate" if mode else "aggregate"


def _cohens_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Compute Cohen's kappa for categorical agreement.

    pairs: list of (rater_a, rater_b) label strings.
    Returns kappa in [-1, 1], or None if fewer than 2 pairs or zero variance.
    """
    if len(pairs) < 2:
        return None
    n = len(pairs)
    labels = sorted({l for p in pairs for l in p})
    if len(labels) < 2:
        return None
    counts = {(a, b): 0 for a in labels for b in labels}
    for a, b in pairs:
        counts[(a, b)] += 1
    p_o = sum(counts[(l, l)] for l in labels) / n
    p_e = sum(
        sum(counts[(l, b)] for b in labels) * sum(counts[(a, l)] for a in labels)
        for l in labels
    ) / (n * n)
    if p_e == 1.0:
        return None
    return (p_o - p_e) / (1 - p_e)


def _pearson_r(pairs: list[tuple[float, float]]) -> float | None:
    """Compute Pearson correlation coefficient from (x, y) pairs."""
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = statistics.mean(x)
    my = statistics.mean(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = (sum((xi - mx) ** 2 for xi in x)) ** 0.5
    sy = (sum((yi - my) ** 2 for yi in y)) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return cov / (sx * sy)


def cmd_summary(iteration: int, mode: str | None = None) -> None:
    """Print aggregate statistics for an iteration."""
    mode = _validate_mode(mode)
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]
    if not judged:
        print(f"No judged items for iteration {iteration}")
        return

    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

    # When mode is set, only count items that have the mode-specific decision
    if mode:
        judged_for_mode = [i for i in judged if i["judgment"].get(dec_key)]
    else:
        judged_for_mode = judged

    n_acc = sum(1 for i in judged_for_mode if i["judgment"].get(dec_key) == "accept")
    n_rej = len(judged_for_mode) - n_acc
    scores = [
        i["judgment"][agg_key]
        for i in judged_for_mode
        if i["judgment"].get(agg_key) is not None
    ]
    n_gold = sum(1 for i in judged_for_mode if i.get("is_gold"))

    mode_label = f" [mode={mode}]" if mode else ""
    print(
        f"Iteration {iteration}: {len(judged_for_mode)} items ({n_gold} gold){mode_label}"
    )
    print(f"  Accept: {n_acc} ({n_acc/len(judged_for_mode)*100:.0f}%)")
    print(f"  Reject: {n_rej} ({n_rej/len(judged_for_mode)*100:.0f}%)")
    if scores:
        print(f"  Mean score: {statistics.mean(scores):.2f}")
        print(f"  Score range: {min(scores):.2f} – {max(scores):.2f}")

    # Per-pipeline accept rates (new split judgment format) — show as secondary
    # when no mode is set; skip when mode is already filtering
    if not mode:
        refl_decisions = [
            i["judgment"].get("reflection_decision")
            for i in judged
            if i["judgment"].get("reflection_decision")
        ]
        prefl_decisions = [
            i["judgment"].get("preflection_decision")
            for i in judged
            if i["judgment"].get("preflection_decision")
        ]
        if refl_decisions:
            refl_acc = sum(1 for d in refl_decisions if d == "accept")
            print(
                f"  Reflection accept: {refl_acc}/{len(refl_decisions)} "
                f"({refl_acc/len(refl_decisions)*100:.0f}%)"
            )
        if prefl_decisions:
            prefl_acc = sum(1 for d in prefl_decisions if d == "accept")
            print(
                f"  Preflection accept: {prefl_acc}/{len(prefl_decisions)} "
                f"({prefl_acc/len(prefl_decisions)*100:.0f}%)"
            )

    # Per-dimension breakdown — filter to mode-relevant parts when mode is set
    dim_scores: dict[str, list[float]] = {}
    for item in judged_for_mode:
        for part, part_j in _mode_judgment_parts(item["judgment"], mode).items():
            for dim, score in part_j.get("scores", {}).items():
                dim_scores.setdefault(f"{part}_{dim}", []).append(score)

    print("\n  Per-dimension means:")
    for dim, vals in sorted(dim_scores.items()):
        print(f"    {dim}: {statistics.mean(vals):.2f}")


def cmd_failures(
    iteration: int,
    limit: int = 10,
    reasoning_limit: int = 200,
    offset: int = 0,
    mode: str | None = None,
) -> None:
    """Print rejected items with judge reasoning."""
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

    items = load_items_for_iteration(iteration)
    rejected = [
        i for i in items if i.get("judgment") and i["judgment"].get(dec_key) == "reject"
    ]
    rejected.sort(key=lambda i: i["judgment"].get(agg_key, 0))

    sliced = rejected[offset : offset + limit]
    mode_label = f" [mode={mode}]" if mode else ""
    print(
        f"Rejected items ({len(rejected)} total, showing {offset}–{offset + len(sliced)}){mode_label}:\n"
    )
    _print_judged_items(sliced, reasoning_limit)


def cmd_accepts(
    iteration: int,
    limit: int = 10,
    reasoning_limit: int = 200,
    offset: int = 0,
    sort: str = "top",
    mode: str | None = None,
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
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

    from pipeline.config import load_config

    cfg = load_config()
    threshold = cfg.phase2.scoring.accept_threshold

    items = load_items_for_iteration(iteration)
    accepted = [
        i for i in items if i.get("judgment") and i["judgment"].get(dec_key) == "accept"
    ]
    if sort == "borderline":
        accepted.sort(key=lambda i: abs(i["judgment"].get(agg_key, 0) - threshold))
        sort_label = f"borderline (closest to threshold {threshold})"
    elif sort == "top":
        accepted.sort(key=lambda i: -i["judgment"].get(agg_key, 0))
        sort_label = "top (highest aggregate first)"
    else:
        raise ValueError(f"Unknown --sort value: {sort!r}. Use 'top' or 'borderline'.")

    sliced = accepted[offset : offset + limit]
    mode_label = f" [mode={mode}]" if mode else ""
    print(
        f"Accepted items ({len(accepted)} total, sort={sort_label}, "
        f"showing {offset}–{offset + len(sliced)}){mode_label}:\n"
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
        # Legacy 2-voice preflection (still populated for old items)
        if item.get("preflection"):
            print(f"  Preflection (3p): {(item.get('preflection') or '')[:150]}...")
        if item.get("preflection_1p"):
            print(f"  Preflection (1p): {(item.get('preflection_1p') or '')[:150]}...")
        # Current 4-field preflection
        for _field in _PREFLECTION_FIELDS_CURRENT:
            if item.get(_field):
                print(f"  {_field}: {(item.get(_field) or '')[:150]}...")
        print(f"  Reflection (1p): {(item.get('reflection') or '')[:150]}...")
        if item.get("reflection_3p"):
            print(f"  Reflection (3p): {(item.get('reflection_3p') or '')[:150]}...")
        print(
            f"  Charter elements: pref={item.get('preflection_charter_elements', [])} "
            f"refl={item.get('reflection_charter_elements', [])}"
        )
        for part, pj in _judgment_parts(j).items():
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
    # Legacy 2-voice preflection (populated only for old items)
    if item.get("preflection"):
        print(f"\n--- PREFLECTION (3p) ---\n{item.get('preflection', '')}")
    if item.get("preflection_1p"):
        print(f"\n--- PREFLECTION (1p) ---\n{item.get('preflection_1p', '')}")
    # Current 4-field preflection
    for _field in ("charter_summary", "neutral", "judgemental", "idealisation"):
        if item.get(_field):
            print(f"\n--- {_field.upper()} ---\n{item.get(_field, '')}")
    print(f"\n--- REFLECTION (1p) ---\n{item.get('reflection', '')}")
    if item.get("reflection_3p"):
        print(f"\n--- REFLECTION (3p) ---\n{item.get('reflection_3p', '')}")
    print(f"\n--- ANALYSIS ---\n{item.get('analysis', '')}")
    print(
        f"\n--- PREFLECTION CHARTER ELEMENTS ---\n"
        f"{item.get('preflection_charter_elements', [])}"
    )
    print(
        f"\n--- REFLECTION CHARTER ELEMENTS ---\n"
        f"{item.get('reflection_charter_elements', [])}"
    )
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
                    "charter_summary": item.get("charter_summary"),
                    "neutral": item.get("neutral"),
                    "judgemental": item.get("judgemental"),
                    "idealisation": item.get("idealisation"),
                    "reflection": item.get("reflection"),
                    "reflection_3p": item.get("reflection_3p"),
                    "preflection_charter_elements": item.get(
                        "preflection_charter_elements"
                    ),
                    "reflection_charter_elements": item.get(
                        "reflection_charter_elements"
                    ),
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
    # Always cover reflection + analysis. Preflection spans two schema
    # generations; include whichever fields are actually populated in this
    # batch so the diversity view stays useful across old and new items.
    fields = ["reflection", "reflection_3p", "analysis"]
    if judged:
        sample = judged[0]
        if sample.get("preflection_1p") is not None or sample.get("preflection"):
            fields = ["preflection", "preflection_1p"] + fields
        if any(
            sample.get(f) is not None
            for f in ("charter_summary", "neutral", "judgemental", "idealisation")
        ):
            fields = [
                "charter_summary",
                "neutral",
                "judgemental",
                "idealisation",
            ] + fields
    for field in fields:
        print(f"=== {field} ===")
        _field_diversity(judged, field)
        print()


def cmd_scores(iteration: int, mode: str | None = None) -> None:
    """Print a compact scores table for all items."""
    mode = _validate_mode(mode)
    agg_key = _mode_aggregate_key(mode)
    dec_key = _mode_decision_key(mode)

    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]
    judged.sort(key=lambda i: i["judgment"].get(agg_key, 0))

    for item in judged:
        j = item["judgment"]
        parts_str = " | ".join(
            f"{part[:6]}[{' '.join(f'{k[:3]}={v}' for k, v in part_j.get('scores', {}).items())}]"
            for part, part_j in _mode_judgment_parts(j, mode).items()
        )
        gold = "G" if item.get("is_gold") else " "
        # Per-mode decisions (new split format)
        mode_dec = ""
        refl_dec = j.get("reflection_decision")
        prefl_dec = j.get("preflection_decision")
        if refl_dec or prefl_dec:
            r_str = refl_dec[:3] if refl_dec else "---"
            p_str = prefl_dec[:3] if prefl_dec else "---"
            mode_dec = f" r={r_str} p={p_str}"
        decision = j.get(dec_key, "?")
        aggregate = j.get(agg_key, 0)
        print(
            f"{gold} {decision[:3]:>3} {aggregate:4.1f}{mode_dec} | "
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
        print(
            f"\n--- REFLECTION CHARTER ELEMENTS ---\n"
            f"{item.get('reflection_charter_elements', [])}"
        )
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

        # Legacy 2-voice preflection (populated only for old items)
        if item.get("preflection"):
            print("--- GENERATED PREFLECTION (3p) ---")
            print(item.get("preflection", ""))
        if item.get("preflection_1p"):
            print("\n--- GENERATED PREFLECTION (1p) ---")
            print(item.get("preflection_1p", ""))
        # Current 4-field preflection
        for _field in _PREFLECTION_FIELDS_CURRENT:
            if item.get(_field):
                print(f"\n--- GENERATED {_field.upper()} ---")
                print(item.get(_field, ""))
        print("\n--- GOLD PREFLECTION ---")
        print(gold.get("preflection", ""))

        print("\n--- GENERATED REFLECTION (1p) ---")
        print(item.get("reflection", ""))
        if item.get("reflection_3p"):
            print("\n--- GENERATED REFLECTION (3p) ---")
            print(item.get("reflection_3p", ""))
        print("\n--- GOLD REFLECTION ---")
        print(gold.get("reflection", ""))

        print("\n--- GENERATED CHARTER (preflection) ---")
        print(item.get("preflection_charter_elements", []))
        print("\n--- GENERATED CHARTER (reflection) ---")
        print(item.get("reflection_charter_elements", []))
        print("\n--- GOLD CHARTER (reflection) ---")
        print(gold.get("reflection_charter_elements", []))
        print()


def cmd_reviews(
    judge_prompt: str | None = None,
    limit: int = 20,
    offset: int = 0,
    reasoning_limit: int = 200,
) -> None:
    """Print human reviews, optionally filtered by judge prompt version.

    Shows reviewer scores, decision, and notes alongside the judge's scores
    AND per-part reasoning for calibration comparison. The judge data shown
    is the *most recent* rejudgment from `judge_correlations` for the same
    judge_model — so if you've improved the judge prompt and run
    `rejudge_all`, you see what the current prompt thinks of each reviewed
    item, not the stale judgment that was active when the review was
    written. Only shows *train* split reviews (75%) so the validation split
    remains unseen by the improver.

    judge_prompt: substring match against the run's judge_prompt field
    (e.g. "v10", "judge_v10.md"). When omitted, shows reviews across every
    judge prompt version. Reviews are grouped by (judge_prompt, judge_model)
    in the output (the prompt the *reviewer* originally responded to) so
    notes stay organised by their original context.

    offset/limit page through the *globally ordered* review list (newest
    judge prompt first, then newest iteration first within a group), so
    `--offset 50 --limit 50` returns the next page after `--limit 50`.

    reasoning_limit: max chars of per-part judge reasoning to print
    (default 200). Pass 0 to suppress reasoning entirely.
    """
    import re

    from pipeline.phase2.storage import (
        _EXCLUDED_REVIEWERS,
        load_judge_correlations,
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

    # Build a lookup of the most recent rejudgment per
    # (item_id, iteration, judge_model) so the displayed judge scores reflect
    # the *current* judge prompt rather than whatever prompt was active when
    # the review was originally written. We rank by the numeric suffix of
    # `judge_v<N>.md` and break ties by timestamp.
    def _judge_version(prompt: str) -> int:
        m = re.search(r"_v(\d+)", prompt or "")
        return int(m.group(1)) if m else -1

    latest_corr: dict[tuple[str, int, str], tuple[str, dict]] = {}
    for c in load_judge_correlations():
        key = (c["item_id"], c["iteration"], c["judge_model"])
        existing = latest_corr.get(key)
        candidate = (_judge_version(c["judge_prompt"]), c.get("timestamp", ""))
        if existing is None or candidate > (
            _judge_version(existing[0]),
            existing[1].get("_corr_timestamp", ""),
        ):
            judgment = dict(c["judgment"])
            judgment["_corr_timestamp"] = c.get("timestamp", "")
            latest_corr[key] = (c["judge_prompt"], judgment)

    # Only show train-split reviews to the improver; exclude blocklisted reviewers
    filtered = [
        r
        for r in reviews.values()
        if review_split(r["item_id"]) == "train"
        and r.get("reviewer_id") not in _EXCLUDED_REVIEWERS
    ]

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

    print(
        f"Human reviews ({len(filtered)} total across "
        f"{len(grouped)} judge prompt(s)):\n"
    )

    def _prompt_sort_key(key: tuple[str, str]) -> tuple:
        m = re.search(r"_v(\d+)", key[0])
        return (key[1], int(m.group(1)) if m else 0)

    # Build a global display order: group by (prompt, model) descending,
    # then within each group sort by iteration desc / timestamp asc.
    ordered: list[tuple[tuple[str, str], dict]] = []
    group_totals: dict[tuple[str, str], int] = {}
    for group_key in sorted(grouped, key=_prompt_sort_key, reverse=True):
        group_reviews = grouped[group_key]
        group_reviews.sort(
            key=lambda r: (-r["iteration"], r.get("timestamp", "")), reverse=False
        )
        group_totals[group_key] = len(group_reviews)
        for r in group_reviews:
            ordered.append((group_key, r))

    # Apply pagination across the flat list. Header is reprinted whenever
    # the group changes within the visible window so the agent always knows
    # which prompt the reviews on screen belong to.
    end = offset + limit if limit > 0 else len(ordered)
    window = ordered[offset:end]

    if not window:
        if offset >= len(ordered):
            print(
                f"(--offset {offset} is past the end; " f"{len(ordered)} reviews total)"
            )
        return

    current_group: tuple[str, str] | None = None
    for group_key, r in window:
        if group_key != current_group:
            current_group = group_key
            gp_name, gm_name = group_key
            group_shown_count = sum(1 for gk, _ in window if gk == group_key)
            print(
                f"=== {gp_name} / {gm_name} "
                f"({group_totals[group_key]} reviews, "
                f"showing {group_shown_count} on this page) ==="
            )

        item = items_by_key.get((r["item_id"], r["iteration"]))

        # Prefer the most recent rejudgment from judge_correlations over the
        # original item.judgment so the agent sees what the *current* judge
        # prompt thinks of this item, not what some older prompt thought when
        # the review was first written.
        judge_label = "Judge"
        judge_judgment: dict | None = None
        diverged_from: str | None = None
        corr_entry = latest_corr.get((r["item_id"], r["iteration"], group_key[1]))
        if corr_entry is not None:
            corr_prompt, corr_judgment = corr_entry
            judge_judgment = corr_judgment
            if corr_prompt != group_key[0]:
                judge_label = f"Judge (latest: {corr_prompt})"
                diverged_from = group_key[0]
            else:
                judge_label = f"Judge ({corr_prompt})"
        elif item and item.get("judgment"):
            judge_judgment = item["judgment"]
            judge_label = f"Judge ({group_key[0]})"

        judge_agg = ""
        judge_decision = ""
        judge_mode_info = ""
        if judge_judgment:
            judge_agg = f"{judge_judgment['aggregate']:.2f}"
            judge_decision = judge_judgment["decision"]
            # Per-mode decisions when available
            refl_dec = judge_judgment.get("reflection_decision")
            prefl_dec = judge_judgment.get("preflection_decision")
            if refl_dec or prefl_dec:
                parts = []
                if refl_dec:
                    refl_agg = judge_judgment.get("reflection_aggregate")
                    refl_agg_str = f"={refl_agg:.2f}" if refl_agg is not None else ""
                    parts.append(f"refl={refl_dec}{refl_agg_str}")
                if prefl_dec:
                    prefl_agg = judge_judgment.get("preflection_aggregate")
                    prefl_agg_str = f"={prefl_agg:.2f}" if prefl_agg is not None else ""
                    parts.append(f"prefl={prefl_dec}{prefl_agg_str}")
                judge_mode_info = f"  ({', '.join(parts)})"

        print(
            f"--- {r['item_id'][:16]} iter={r['iteration']} "
            f"reviewer={r['reviewer_id']} ---"
        )
        print(f"  Human:  decision={r['decision']}  " f"aggregate={r['aggregate']:.2f}")
        if judge_agg:
            print(
                f"  {judge_label}:  decision={judge_decision}  aggregate={judge_agg}"
                f"{judge_mode_info}"
            )
        if diverged_from is not None:
            print(
                f"  ⚠ Notes below were written against the OLDER JUDGE "
                f"{diverged_from} (generator output unchanged). "
                f'"I agree with the judge" refers to {diverged_from}, not '
                f"the rejudgment shown. For its reasoning: "
                f"`reasoning {r['item_id'][:8]} {r['iteration']}`."
            )

        scores = r["scores"]
        is_per_part = scores and isinstance(next(iter(scores.values())), dict)
        if is_per_part:
            all_parts = sorted(
                set(scores.keys())
                | (
                    set(_judgment_parts(judge_judgment).keys())
                    if judge_judgment
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
                judge_reasoning = ""
                if judge_judgment:
                    part_j = judge_judgment.get(part, {}) or {}
                    judge_s = part_j.get("scores", {})
                    judge_reasoning = part_j.get("reasoning", "") or ""
                dims = sorted(set(human_s) | set(judge_s))
                pairs = " ".join(
                    f"{d[:3]}={human_s.get(d, '?')}/{judge_s.get(d, '?')}" for d in dims
                )
                if dims:
                    print(f"  {part}: {pairs}  (human/judge)")
                if reasoning_limit > 0 and judge_reasoning:
                    snippet = judge_reasoning[:reasoning_limit]
                    if len(judge_reasoning) > reasoning_limit:
                        snippet += "…"
                    print(f"    judge reasoning: {snippet}")
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

    # Footer with pagination hint when there's more to see.
    after = len(ordered) - end
    if after > 0:
        print(f"({after} more reviews — use --offset {end} to continue)")


def cmd_filter(
    iteration: int,
    dim: str,
    below: float | None = None,
    above: float | None = None,
    part: str | None = None,
) -> None:
    """Filter items by score threshold on a specific dimension.

    Exactly one of --below or --above must be supplied.

    Valid --part values:
      reflection: reflection_1p, reflection_3p
      preflection (new): charter_summary, neutral, judgemental, idealisation
      preflection (legacy): preflection_3p, preflection_1p
    Valid --dim values (reflection): relevance, specificity, charter_grounding, voice_tone
    Valid --dim values (preflection, new): relevance, charter_grounding, class_discipline
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


def cmd_trend(mode: str | None = None) -> None:
    """Print per-iteration trend table: accept rate, mean score, per-dimension means."""
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

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
        for part, part_j in _mode_judgment_parts(judged[0]["judgment"], mode).items():
            for dim in part_j.get("scores", {}):
                key = f"{part[:3]}_{dim[:3]}"
                if key not in all_dim_keys:
                    all_dim_keys.append(key)
        break

    # Header
    mode_label = f" [mode={mode}]" if mode else ""
    dim_header = " ".join(f"{k:>7}" for k in all_dim_keys)
    print(
        f"{'iter':>4} {'acc%':>5} {'mean':>5} {dim_header}  gen_prompt / judge_prompt{mode_label}"
    )

    for run in runs:
        it = run["iteration"]
        items = load_items_for_iteration(it)
        judged = [i for i in items if i.get("judgment")]
        if not judged:
            print(f"{it:>4}  (no judged items)")
            continue

        # When mode is set, only include items that have the mode-specific aggregate
        if mode:
            judged_for_mode = [
                i for i in judged if i["judgment"].get(agg_key) is not None
            ]
        else:
            judged_for_mode = judged

        if not judged_for_mode:
            print(f"{it:>4}  (no items with {mode} data)")
            continue

        scores = [
            i["judgment"][agg_key]
            for i in judged_for_mode
            if i["judgment"].get(agg_key) is not None
        ]
        n_acc = sum(
            1 for i in judged_for_mode if i["judgment"].get(dec_key) == "accept"
        )
        acc_pct = n_acc / len(judged_for_mode) * 100
        mean = statistics.mean(scores) if scores else 0.0

        dim_means: dict[str, list[float]] = {}
        for item in judged_for_mode:
            for part, part_j in _mode_judgment_parts(item["judgment"], mode).items():
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
    - Cohen's κ on the decision (chance-corrected agreement, in [-1, 1])
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

    # Sort by version number descending, skip old pre-split prompts (judge_v*.md),
    # and default to showing only the 5 latest versions per model.
    import re as _re_corr

    def _corr_sort_key(key: tuple[str, str]) -> tuple[str, int]:
        m = _re_corr.search(r"_v(\d+)", key[0])
        return (key[1], int(m.group(1)) if m else 0)

    sorted_keys = sorted(by_prompt, key=_corr_sort_key, reverse=True)
    # Drop old combined prompts (judge_v*.md) — only show per-mode prompts
    sorted_keys = [
        k
        for k in sorted_keys
        if not _re_corr.match(r"^judge_v\d+\.md$", k[0])
        and not _re_corr.match(r"^generator_v\d+\.md$", k[0])
    ]
    # Limit to 5 latest per model (use --all to see everything)
    if "--all" not in sys.argv:
        _seen_per_model: dict[str, int] = {}
        _limited: list[tuple[str, str]] = []
        for k in sorted_keys:
            model = k[1]
            _seen_per_model[model] = _seen_per_model.get(model, 0) + 1
            if _seen_per_model[model] <= 5:
                _limited.append(k)
        if len(_limited) < len(sorted_keys):
            print(
                f"(showing 5 latest per model, {len(sorted_keys) - len(_limited)} "
                f"older hidden — use --all to see all)\n"
            )
        sorted_keys = _limited

    for prompt_name, model_name in sorted_keys:
        entries = by_prompt[(prompt_name, model_name)]
        decision_matches = 0
        decision_pairs: list[tuple[str, str]] = []
        score_diffs = []
        dim_diffs: dict[str, list[float]] = {}
        matched = 0

        # Per-mode decision + score tracking
        refl_decision_pairs: list[tuple[str, str]] = []
        prefl_decision_pairs: list[tuple[str, str]] = []
        refl_score_pairs: list[tuple[float, float]] = []
        prefl_score_pairs: list[tuple[float, float]] = []

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
            decision_pairs.append((judge_decision, human_decision))
            if judge_decision == human_decision:
                decision_matches += 1

            # Per-mode decision agreement — compare judge per-mode decision
            # against human per-mode decision (derived from per-voice scores).
            human_scores = review.get("scores", {})
            _is_pp = human_scores and isinstance(
                next(iter(human_scores.values()), None), dict
            )
            # Preflection voices span both the legacy 2-voice and current
            # 4-field schemas — pull whichever keys the review actually uses
            # so old and new reviews are both handled.
            _prefl_voices = tuple(
                k for k in human_scores.keys() if k in _PREFLECTION_PART_NAMES
            )
            for _mode, _voices, _mode_pairs, _score_pairs in (
                (
                    "reflection",
                    ("reflection_1p", "reflection_3p"),
                    refl_decision_pairs,
                    refl_score_pairs,
                ),
                (
                    "preflection",
                    _prefl_voices,
                    prefl_decision_pairs,
                    prefl_score_pairs,
                ),
            ):
                j_dec = j.get(f"{_mode}_decision")
                if not j_dec:
                    continue
                j_mode_agg = j.get(f"{_mode}_aggregate")
                # Compute human per-mode decision + aggregate from per-voice scores
                h_dec = review.get(f"{_mode}_decision")
                h_mode_agg = review.get(f"{_mode}_aggregate")
                if (not h_dec or h_mode_agg is None) and _is_pp:
                    _h_vals = [
                        v
                        for _v in _voices
                        for v in (human_scores.get(_v) or {}).values()
                    ]
                    if _h_vals:
                        _h_agg = sum(_h_vals) / len(_h_vals)
                        _h_floor = any(v <= 2 for v in _h_vals)
                        if not h_dec:
                            h_dec = "reject" if _h_floor or _h_agg < 4 else "accept"
                        if h_mode_agg is None:
                            h_mode_agg = _h_agg
                if not h_dec:
                    h_dec = human_decision  # last resort: combined
                if h_dec:
                    _mode_pairs.append((j_dec, h_dec))
                if j_mode_agg is not None and h_mode_agg is not None:
                    _score_pairs.append((j_mode_agg, h_mode_agg))

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

        print(f"\n{prompt_name} / {model_name} ({matched} items):")

        # Per-mode decision agreement, kappa, and Pearson r
        for _mode_label, _mode_pairs, _score_pairs in (
            ("Reflection", refl_decision_pairs, refl_score_pairs),
            ("Preflection", prefl_decision_pairs, prefl_score_pairs),
        ):
            if not _mode_pairs:
                continue
            _agree = sum(1 for a, b in _mode_pairs if a == b)
            _kappa = _cohens_kappa(_mode_pairs)
            _kappa_str = f"{_kappa:.3f}" if _kappa is not None else "n/a"
            _pearson = _pearson_r(_score_pairs) if _score_pairs else None
            _pearson_str = f"{_pearson:.3f}" if _pearson is not None else "n/a"
            print(
                f"  {_mode_label}: {_agree}/{len(_mode_pairs)} "
                f"({_agree / len(_mode_pairs) * 100:.0f}%)  "
                f"κ={_kappa_str}  r={_pearson_str}"
            )

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


def cmd_rejudge_all(mode: str | None = None) -> None:
    """Re-judge all human-reviewed items with ALL judge prompts × ALL judge models."""
    from pipeline.config import load_config
    from pipeline.phase2.run import rejudge_all_prompts_and_models

    if mode:
        _validate_mode(mode)

    cfg = load_config()
    mode_label = f" (mode={mode})" if mode else ""
    print(
        f"Re-judging all reviewed items across all judge prompts and models{mode_label}..."
    )
    total = rejudge_all_prompts_and_models(cfg, mode=mode)
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
    mode: str | None = None,
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
    alias = model_alias or cfg.phase2.generator_models[0].alias
    gen_model_cfg = resolve_generator_model(cfg, alias)
    endpoint = gen_model_cfg.endpoint or cfg.phase2.endpoint
    client, semaphore = make_api_client(
        endpoint, cfg.phase2.iteration.max_concurrent, api_keys=cfg.api_keys
    )
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

    # Resolve both mode-specific prompt paths from the given prompt.
    # Check "preflection" first since "reflection" is a substring of it.
    prompt_name = prompt.name
    prompt_dir = prompt.parent
    if "preflection" in prompt_name:
        prefl_prompt = prompt
        refl_name = prompt_name.replace("preflection", "reflection")
        refl_prompt = prompt_dir / refl_name
        if not refl_prompt.exists():
            refl_prompt = None
    elif "reflection" in prompt_name:
        refl_prompt = prompt
        prefl_name = prompt_name.replace("reflection", "preflection")
        prefl_prompt = prompt_dir / prefl_name
        if not prefl_prompt.exists():
            prefl_prompt = None
    else:
        refl_prompt = prompt
        prefl_prompt = prompt

    # Auto-detect mode from prompt name if not explicit
    if mode is None and "preflection" in prompt_name and refl_prompt is None:
        mode = "preflection"
    elif mode is None and "reflection" in prompt_name and prefl_prompt is None:
        mode = "reflection"

    print(
        f"Test generating {len(items)} items with {prompt.name} (model={alias}, mode={mode or 'both'})..."
    )
    generated = generate_batch(
        items,
        refl_prompt,
        prefl_prompt,
        charter_text,
        gen_model_cfg.api_name,
        iteration=latest_iter,
        client=client,
        semaphore=semaphore,
        save=False,
        writing_guidelines_text=writing_guidelines_text,
        json_mode=gen_model_cfg.json_mode,
        completion_max_tokens=gen_model_cfg.completion_max_tokens,
        context_window_tokens=gen_model_cfg.context_window_tokens,
        mode=mode,
    )

    test_id = _make_test_id("tg")
    result_items = []
    for g in generated:
        result_items.append(
            {
                "item_id": g["item_id"],
                "preflection": (g.get("preflection") or "")[:200],
                "preflection_1p": (g.get("preflection_1p") or "")[:200],
                "reflection": (g.get("reflection") or "")[:200],
                "reflection_3p": (g.get("reflection_3p") or "")[:200],
                "preflection_charter_elements": g.get(
                    "preflection_charter_elements", []
                ),
                "reflection_charter_elements": g.get("reflection_charter_elements", []),
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
        r1p = g.get("reflection", "") or ""
        r3p = g.get("reflection_3p", "") or ""
        p3p = g.get("preflection", "") or ""
        p1p = g.get("preflection_1p", "") or ""
        parts = []
        if r1p:
            parts.append(f"refl_1p={r1p[:60]}")
        if r3p:
            parts.append(f"refl_3p={r3p[:60]}")
        if p3p:
            parts.append(f"pre_3p={p3p[:60]}")
        if p1p:
            parts.append(f"pre_1p={p1p[:60]}")
        print(f"  {g['item_id'][:12]}: {' | '.join(parts)}")
    print("Saved to test_results")


def cmd_test_judge(
    prompt_path: str,
    item_ids: list[str] | None = None,
    iteration: int | None = None,
    n: int = 3,
    role: str = "judge",
    model_alias: str | None = None,
    mode: str | None = None,
) -> None:
    """Judge items with a prompt file without saving to main items table.

    Loads generated items from specified iteration, runs judge_batch(save=False),
    saves a test_results entry.

    mode: "reflection" or "preflection" to test only one pipeline, or None for both.

    prompt_path resolution:
      - If the path points to a specific file (e.g. judge_reflection_v3.md), uses
        it directly for that mode and auto-discovers the counterpart file for the
        other mode (unless --mode restricts to one pipeline).
      - If the path points to a directory, discovers the latest
        judge_reflection_vN.md and judge_preflection_vN.md in that directory.
    """
    import re as _re

    from pipeline.api import make_api_client
    from pipeline.config import load_config, resolve_judge_model
    from pipeline.phase2.run import judge_batch
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    alias = model_alias or cfg.phase2.judge_models[0].alias
    jdg_model_cfg = resolve_judge_model(cfg, alias)
    endpoint = jdg_model_cfg.endpoint or cfg.phase2.endpoint
    client, semaphore = make_api_client(
        endpoint, cfg.phase2.iteration.max_concurrent, api_keys=cfg.api_keys
    )

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

    # Resolve reflection / preflection prompt paths.
    # If a specific file is given, infer the counterpart from the same directory.
    refl_prompt: Path | None = None
    prefl_prompt: Path | None = None

    if prompt.is_dir():
        # Directory mode: discover latest judge_reflection_vN / judge_preflection_vN
        def _latest_in_dir(directory: Path, prefix: str) -> Path | None:
            pat = _re.compile(rf"^{_re.escape(prefix)}_v(\d+)\.md$")
            cands = []
            for p in directory.iterdir():
                m = pat.match(p.name)
                if m:
                    cands.append((int(m.group(1)), p))
            return max(cands)[1] if cands else None

        refl_prompt = _latest_in_dir(prompt, "judge_reflection")
        prefl_prompt = _latest_in_dir(prompt, "judge_preflection")
    elif "judge_reflection" in prompt.name:
        refl_prompt = prompt
        # Auto-discover counterpart
        m = _re.search(r"_v(\d+)\.md$", prompt.name)
        if m:
            prefl_name = f"judge_preflection_v{m.group(1)}.md"
            prefl_candidate = prompt.parent / prefl_name
            if prefl_candidate.exists():
                prefl_prompt = prefl_candidate
    elif "judge_preflection" in prompt.name:
        prefl_prompt = prompt
        # Auto-discover counterpart
        m = _re.search(r"_v(\d+)\.md$", prompt.name)
        if m:
            refl_name = f"judge_reflection_v{m.group(1)}.md"
            refl_candidate = prompt.parent / refl_name
            if refl_candidate.exists():
                refl_prompt = refl_candidate
    else:
        # Legacy single-file judge prompt — treat as both
        refl_prompt = prompt
        prefl_prompt = prompt

    # Validate required prompts for the requested mode
    if mode == "reflection":
        assert refl_prompt, f"No judge_reflection prompt found for {prompt}"
        prefl_prompt = None
    elif mode == "preflection":
        assert prefl_prompt, f"No judge_preflection prompt found for {prompt}"
        refl_prompt = None
    else:
        assert (
            refl_prompt or prefl_prompt
        ), f"No judge_reflection or judge_preflection prompt found for {prompt}"

    from pipeline.config import CHARTER_PATH, WRITING_GUIDELINES_PATH

    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    prompt_label = (
        f"refl={refl_prompt.name if refl_prompt else 'none'} "
        f"prefl={prefl_prompt.name if prefl_prompt else 'none'}"
    )
    mode_label = f" mode={mode}" if mode else ""
    print(
        f"Test judging {len(items)} items with {prompt_label}{mode_label} (model={alias})..."
    )
    judged = judge_batch(
        items,
        refl_prompt,
        prefl_prompt,
        jdg_model_cfg.api_name,
        iteration=iter_num,
        accept_threshold=cfg.phase2.scoring.accept_threshold,
        client=client,
        semaphore=semaphore,
        save=False,
        floor_threshold=cfg.phase2.scoring.floor_threshold,
        charter_text=charter_text,
        writing_guidelines_text=writing_guidelines_text,
        completion_max_tokens=jdg_model_cfg.completion_max_tokens,
        context_window_tokens=jdg_model_cfg.context_window_tokens,
        mode=mode,
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


def cmd_run_batch(role: str = "judge", mode: str | None = None) -> None:
    """Run a full cross-iteration batch with the first model of the given role.

    Saves to main items + runs tables AND to test_results for tracking.
    """
    mode = _validate_mode(mode)

    from pipeline.config import load_config
    from pipeline.phase2.run import (
        run_judge_cross_iteration,
        run_generator_cross_iteration,
    )

    cfg = load_config()

    if role == "judge":
        target = cfg.phase2.judge_models[0].alias
        source = "improve_judge"
        results = run_judge_cross_iteration(cfg, target, source=source, mode=mode)
    else:
        target = cfg.phase2.generator_models[0].alias
        source = "improve_generator"
        results = run_generator_cross_iteration(cfg, target, source=source, mode=mode)

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


def cmd_run_cross_batch(role: str, target: str, mode: str | None = None) -> None:
    """Run a cross-iteration batch for a specific (role, target) pair.

    Judge cross-iteration: generate with ALL generators, judge with target.
    Generator cross-iteration: generate with target, judge with ALL judges.

    mode: "reflection" or "preflection" to run only one pipeline, or None for both.
    """
    from pipeline.config import load_config
    from pipeline.phase2.run import (
        run_judge_cross_iteration,
        run_generator_cross_iteration,
    )

    cfg = load_config()

    if role == "judge":
        results = run_judge_cross_iteration(
            cfg, target, source="improve_judge", mode=mode
        )
    else:
        results = run_generator_cross_iteration(
            cfg, target, source="improve_generator", mode=mode
        )

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


def cmd_cross_summary(group_id: str, mode: str | None = None) -> None:
    """Show aggregated stats across all iterations in a cross-iteration group."""
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

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

    mode_label = f" [mode={mode}]" if mode else ""
    print(f"Cross-iteration summary (group_id={full_gid[:8]}...){mode_label}:")
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

        if mode:
            judged = [i for i in judged if i["judgment"].get(agg_key) is not None]

        n_acc = sum(1 for i in judged if i["judgment"].get(dec_key) == "accept")
        scores = [
            i["judgment"][agg_key]
            for i in judged
            if i["judgment"].get(agg_key) is not None
        ]
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


def _dim_scores_for_items(
    judged: list[dict], mode: str | None = None
) -> dict[str, list[float]]:
    """Collect per-dimension score lists across all judgment parts."""
    dim_scores: dict[str, list[float]] = {}
    for item in judged:
        for part, part_j in _mode_judgment_parts(item["judgment"], mode).items():
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


def cmd_diagnose(group_id: str, mode: str | None = None) -> None:
    """One-shot comprehensive analysis for a cross-iteration group.

    Combines: cross_summary, per-dimension means, score distributions,
    ceiling effects, floor-rule violations, decision flips, diversity,
    and gold/review availability.
    """
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

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
    mode_label = f" [mode={mode}]" if mode else ""
    print(
        f"=== DIAGNOSE group_id={full_gid[:8]}... ({len(group_runs)} iterations){mode_label} ===\n"
    )

    iter_data: dict[int, list[dict]] = {}
    iter_runs: dict[int, dict] = {}

    if mode:
        print(
            f"{'Iter':>6} {'Generator':>20} {'Judge':>20} "
            f"{'Items':>6} {'Acc%':>6} {'Mean':>6} {'Floor':>6}"
        )
        print("-" * 80)
    else:
        print(
            f"{'Iter':>6} {'Generator':>20} {'Judge':>20} "
            f"{'Items':>6} {'Acc%':>6} {'Refl%':>6} {'Prefl%':>7} {'Mean':>6} {'Floor':>6}"
        )
        print("-" * 100)
    for r in group_runs:
        it = r["iteration"]
        judged = _collect_judged(it)
        iter_data[it] = judged
        iter_runs[it] = r

        n_acc = sum(1 for i in judged if i["judgment"].get(dec_key) == "accept")
        scores = [
            i["judgment"][agg_key]
            for i in judged
            if i["judgment"].get(agg_key) is not None
        ]
        mean_s = statistics.mean(scores) if scores else 0.0
        judged_count = (
            len([i for i in judged if i["judgment"].get(agg_key) is not None])
            if mode
            else len(judged)
        )
        acc_pct = n_acc / judged_count * 100 if judged_count else 0

        n_floor = sum(
            1
            for i in judged
            if any(
                s <= floor_thresh
                for part_j in _mode_judgment_parts(i["judgment"], mode).values()
                for s in part_j.get("scores", {}).values()
            )
        )

        if mode:
            print(
                f"{it:>6} {r['generator_model']:>20} {r['judge_model']:>20} "
                f"{judged_count:>6} {acc_pct:>5.0f}% "
                f"{mean_s:>6.2f} {n_floor:>6}"
            )
        else:
            # Per-pipeline accept rates
            refl_decs = [
                i["judgment"].get("reflection_decision")
                for i in judged
                if i["judgment"].get("reflection_decision")
            ]
            prefl_decs = [
                i["judgment"].get("preflection_decision")
                for i in judged
                if i["judgment"].get("preflection_decision")
            ]
            refl_pct_str = (
                f"{sum(1 for d in refl_decs if d == 'accept') / len(refl_decs) * 100:5.0f}%"
                if refl_decs
                else "   n/a"
            )
            prefl_pct_str = (
                f"{sum(1 for d in prefl_decs if d == 'accept') / len(prefl_decs) * 100:5.0f}%"
                if prefl_decs
                else "    n/a"
            )

            print(
                f"{it:>6} {r['generator_model']:>20} {r['judge_model']:>20} "
                f"{len(judged):>6} {acc_pct:>5.0f}% {refl_pct_str:>6} {prefl_pct_str:>7} "
                f"{mean_s:>6.2f} {n_floor:>6}"
            )

    # --- 2. Per-dimension means per iteration ---
    print("\n--- Per-dimension means ---")
    all_dim_keys: list[str] = []
    for judged in iter_data.values():
        if not judged:
            continue
        for part, part_j in _mode_judgment_parts(judged[0]["judgment"], mode).items():
            for dim in part_j.get("scores", {}):
                key = f"{part[:3]}_{dim[:3]}"
                if key not in all_dim_keys:
                    all_dim_keys.append(key)
        break

    header = "  " + f"{'Iter':>6}" + "".join(f" {k:>8}" for k in all_dim_keys)
    print(header)
    for it, judged in iter_data.items():
        dim_scores = _dim_scores_for_items(judged, mode)
        vals = "".join(
            f" {statistics.mean(dim_scores.get(k, [0])):>8.2f}" for k in all_dim_keys
        )
        print(f"  {it:>6}{vals}")

    # --- 3. Score distribution & ceiling effect ---
    print("\n--- Score distribution (all dimensions pooled) ---")
    for it, judged in iter_data.items():
        dim_scores = _dim_scores_for_items(judged, mode)
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
            for part, part_j in _mode_judgment_parts(j, mode).items():
                for dim, score in part_j.get("scores", {}).items():
                    if score <= floor_thresh:
                        violations.append(
                            (
                                item["item_id"][:12],
                                part[:3],
                                dim[:3],
                                score,
                                j.get(agg_key, 0),
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
                _print_flips(iters[i], iters[j], iter_data, iter_runs, mode=mode)

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
    mode: str | None = None,
) -> None:
    """Print decision flips between two iterations on shared items."""
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

    items_a = {i["item_id"]: i for i in iter_data[iter_a]}
    items_b = {i["item_id"]: i for i in iter_data[iter_b]}
    shared = set(items_a) & set(items_b)
    if not shared:
        print(f"  iter {iter_a} vs {iter_b}: no shared items")
        return

    flips = []
    for iid in sorted(shared):
        da = items_a[iid]["judgment"].get(dec_key, "?")
        db = items_b[iid]["judgment"].get(dec_key, "?")
        if da != db:
            sa = items_a[iid]["judgment"].get(agg_key, 0)
            sb = items_b[iid]["judgment"].get(agg_key, 0)
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


def cmd_diff(
    iter_a: int, iter_b: int, limit: int = 10, mode: str | None = None
) -> None:
    """Cross-iteration comparison of shared items between two iterations.

    Shows: decision agreement, per-dimension score diffs, and full
    preflection/reflection text for items that flipped accept<->reject.
    """
    mode = _validate_mode(mode)
    dec_key = _mode_decision_key(mode)
    agg_key = _mode_aggregate_key(mode)

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
        da = ja.get(dec_key, "?")
        db = jb.get(dec_key, "?")
        sa_agg = ja.get(agg_key, 0)
        sb_agg = jb.get(agg_key, 0)
        agg_diffs.append(sb_agg - sa_agg)

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

        all_parts_ab = set(_mode_judgment_parts(ja, mode)) | set(
            _mode_judgment_parts(jb, mode)
        )
        for part in all_parts_ab:
            sa = ja.get(part, {}).get("scores", {})
            sb = jb.get(part, {}).get("scores", {})
            for dim in set(sa) & set(sb):
                key = f"{part[:3]}_{dim[:3]}"
                dim_diffs.setdefault(key, []).append(sb[dim] - sa[dim])

    total = len(shared_ids)
    agree = both_acc + both_rej
    mode_label = f" [mode={mode}]" if mode else ""
    print(
        f"=== DIFF iter {iter_a} vs iter {iter_b} ({total} shared items){mode_label} ===\n"
    )
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
            x[2]["judgment"].get(agg_key, 0) - x[1]["judgment"].get(agg_key, 0)
        ),
        reverse=True,
    )
    shown = flips[:limit]
    if shown:
        print(f"\n--- Decision flips ({len(flips)} total, showing {len(shown)}) ---")
    for iid, item_a, item_b in shown:
        ja = item_a["judgment"]
        jb = item_b["judgment"]
        ja_agg = ja.get(agg_key, 0)
        jb_agg = jb.get(agg_key, 0)
        print(
            f"\n  {iid[:16]}: {ja.get(dec_key, '?')}({ja_agg:.1f}) -> "
            f"{jb.get(dec_key, '?')}({jb_agg:.1f}) [diff={jb_agg-ja_agg:+.1f}]"
        )

        # Per-dimension comparison — filtered to mode-relevant parts
        for part in set(_mode_judgment_parts(ja, mode)) | set(
            _mode_judgment_parts(jb, mode)
        ):
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


def cmd_rollback(alias: str, role: str, version: int, mode: str | None = None) -> None:
    """Promote a specific version to be the latest by copying it to v(max+1).

    The pipeline always uses the highest _vN.md file. If v2 performed best but
    v3 and v4 exist, `rollback <alias> generator 2` copies v2 to v5, making it
    the active prompt.

    For judge and generator roles, prompt files use the naming convention
    ``{role}_{mode}_v{N}.md`` (e.g. ``judge_reflection_v3.md``).
    Pass ``--mode reflection`` or ``--mode preflection`` to select which
    mode-specific prompt to roll back.
    """
    import shutil

    from pipeline.config import PIPELINE_DATA_DIR

    prompts_dir = PIPELINE_DATA_DIR / "prompts" / alias
    assert (
        prompts_dir.exists()
    ), f"No prompt directory for alias '{alias}': {prompts_dir}"

    # Build the file prefix. For judge/generator roles with a mode, the naming
    # is e.g. judge_reflection_v3.md. Without mode, fall back to legacy naming.
    if mode:
        mode = _validate_mode(mode)
        prefix = f"{role}_{mode}"
    else:
        # Auto-detect: if mode-specific files exist, require --mode
        import re as _re

        _has_mode_files = any(
            _re.match(
                rf"^{_re.escape(role)}_(reflection|preflection)_v\d+\.md$", p.name
            )
            for p in prompts_dir.iterdir()
        )
        if _has_mode_files and role in ("judge", "generator"):
            print(
                f"Prompt files use mode-specific naming ({role}_reflection_vN.md / "
                f"{role}_preflection_vN.md). Pass --mode reflection or --mode preflection."
            )
            return
        prefix = role

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
    print(f"Rolled back: copied {source.name} -> {dest.name}")
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
            "  - --part: reflection (reflection_1p, reflection_3p); "
            "preflection new (charter_summary, neutral, judgemental, idealisation); "
            "preflection legacy (preflection_3p, preflection_1p)"
        )
        print(
            "  - --dim: reflection (relevance, specificity, charter_grounding, voice_tone); "
            "preflection new (relevance, charter_grounding, class_discipline); "
            "preflection legacy matches reflection dims"
        )
        sys.exit(0)

    # Flags that consume the following token as their value. Anything else
    # starting with "--" is treated as a boolean flag. This lets us strip
    # flag values out of the positional list so e.g. `reviews --limit 100`
    # doesn't have `100` swallowed as the judge_prompt argument.
    VALUE_FLAGS = {
        "--limit",
        "--reasoning-limit",
        "--offset",
        "--sort",
        "--below",
        "--above",
        "--dim",
        "--part",
        "--items",
        "--n",
        "--role",
        "--phase",
        "--model",
        "--iteration",
        "--target",
        "--type",
        "--reason",
        "--status",
        "--mode",
    }

    positional: list[str] = []
    i = 1
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            if a in VALUE_FLAGS:
                i += 2  # skip flag and its value
            else:
                i += 1  # boolean flag
        else:
            positional.append(a)
            i += 1

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
        _require_positional(1, "summary <iteration> [--mode reflection|preflection]")
        cmd_summary(int(positional[0]), mode=_get_flag("--mode"))
    elif cmd == "failures":
        _require_positional(
            1,
            "failures <iteration> [--limit N] [--offset N] [--reasoning-limit N] [--mode reflection|preflection]",
        )
        cmd_failures(
            int(positional[0]),
            limit=_get_flag_int("--limit", 10),
            reasoning_limit=_get_flag_int("--reasoning-limit", 200),
            offset=_get_flag_int("--offset", 0),
            mode=_get_flag("--mode"),
        )
    elif cmd == "accepts":
        _require_positional(
            1,
            "accepts <iteration> [--limit N] [--offset N] [--reasoning-limit N] "
            "[--sort top|borderline] [--mode reflection|preflection]",
        )
        cmd_accepts(
            int(positional[0]),
            limit=_get_flag_int("--limit", 10),
            reasoning_limit=_get_flag_int("--reasoning-limit", 200),
            offset=_get_flag_int("--offset", 0),
            sort=_get_flag("--sort", "top"),
            mode=_get_flag("--mode"),
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
        _require_positional(1, "scores <iteration> [--mode reflection|preflection]")
        cmd_scores(int(positional[0]), mode=_get_flag("--mode"))
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
        cmd_reviews(
            judge_prompt=judge_prompt_arg,
            limit=_get_flag_int("--limit", 20),
            offset=_get_flag_int("--offset", 0),
            reasoning_limit=_get_flag_int("--reasoning-limit", 200),
        )
    elif cmd == "filter":
        _require_positional(
            1,
            "filter <iteration> --dim X (--below N | --above N) "
            "[--part PART]  (reflection: reflection_1p, reflection_3p; preflection: charter_summary, neutral, judgemental, idealisation [new] or preflection_3p, preflection_1p [legacy])",
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
        cmd_trend(mode=_get_flag("--mode"))
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
            mode=_get_flag("--mode"),
        )
    elif cmd == "test_judge":
        _require_positional(
            1,
            "test_judge <prompt_path> [--items id1,id2,...] [--iteration N] [--role judge|generator] [--model ALIAS] [--mode reflection|preflection]",
        )
        item_ids_str = _get_flag("--items")
        item_ids = item_ids_str.split(",") if item_ids_str else None
        mode_arg = _get_flag("--mode")
        if mode_arg:
            assert mode_arg in (
                "reflection",
                "preflection",
            ), f"--mode must be 'reflection' or 'preflection', got '{mode_arg}'"
        cmd_test_judge(
            positional[0],
            item_ids=item_ids,
            iteration=_get_flag_int("--iteration"),
            n=_get_flag_int("--n", 3),
            role=_get_flag("--role", _get_flag("--phase", "judge")),
            model_alias=_get_flag("--model"),
            mode=mode_arg,
        )
    elif cmd == "run_batch":
        cmd_run_batch(
            role=_get_flag("--role", _get_flag("--phase", "judge")),
            mode=_get_flag("--mode"),
        )
    elif cmd == "run_cross_batch":
        role = _get_flag("--role")
        target = _get_flag("--target")
        assert (
            role
        ), "Usage: run_cross_batch --role judge|generator --target <alias> [--mode reflection|preflection]"
        assert (
            target
        ), "Usage: run_cross_batch --role judge|generator --target <alias> [--mode reflection|preflection]"
        mode_arg = _get_flag("--mode")
        if mode_arg:
            assert mode_arg in (
                "reflection",
                "preflection",
            ), f"--mode must be 'reflection' or 'preflection', got '{mode_arg}'"
        cmd_run_cross_batch(role=role, target=target, mode=mode_arg)
    elif cmd == "cross_summary":
        _require_positional(
            1, "cross_summary <group_id> [--mode reflection|preflection]"
        )
        cmd_cross_summary(positional[0], mode=_get_flag("--mode"))
    elif cmd == "diagnose":
        _require_positional(1, "diagnose <group_id> [--mode reflection|preflection]")
        cmd_diagnose(positional[0], mode=_get_flag("--mode"))
    elif cmd == "diff":
        _require_positional(
            2,
            "diff <iter1> <iter2> [--limit N] [--mode reflection|preflection]",
        )
        cmd_diff(
            int(positional[0]),
            int(positional[1]),
            limit=_get_flag_int("--limit", 10),
            mode=_get_flag("--mode"),
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
        cmd_rejudge_all(mode=_get_flag("--mode"))
    elif cmd == "parse_stats":
        _require_positional(1, "parse_stats <iteration>")
        cmd_parse_stats(int(positional[0]))
    elif cmd == "rollback":
        _require_positional(
            3,
            "rollback <alias> <role> <version> [--mode reflection|preflection]",
        )
        cmd_rollback(
            positional[0],
            positional[1],
            int(positional[2]),
            mode=_get_flag("--mode"),
        )
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
