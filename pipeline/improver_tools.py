"""CLI tools for the improver agent to query iteration data and run tests.

Usage (via Bash tool):
    python -m pipeline.improver_tools summary <iteration>
    python -m pipeline.improver_tools failures <iteration> [--limit N] [--reasoning-limit N]
    python -m pipeline.improver_tools show <item_id>[,id2,...] <iteration> [--brief]
    python -m pipeline.improver_tools show --gold <iteration> [--brief]
    python -m pipeline.improver_tools item <item_id> <iteration>
    python -m pipeline.improver_tools diversity <iteration>
    python -m pipeline.improver_tools scores <iteration>
    python -m pipeline.improver_tools gold [--limit N] [--offset N] [--verbose]
    python -m pipeline.improver_tools compare <item_id> <iteration>
    python -m pipeline.improver_tools reviews [<iteration>] [--limit N]
    python -m pipeline.improver_tools filter <iteration> --dim X --below N [--part preflection|reflection]
    python -m pipeline.improver_tools trend
    python -m pipeline.improver_tools test_generate <prompt_path> [--items id1,id2,...] [--n N] [--phase A|B]
    python -m pipeline.improver_tools test_judge <prompt_path> [--items id1,id2,...] [--iteration N] [--phase A|B]
    python -m pipeline.improver_tools run_batch [--phase A|B]
    python -m pipeline.improver_tools test_results [--phase A|B] [--type generate|judge|batch]
    python -m pipeline.improver_tools correlations
    python -m pipeline.improver_tools rejudge_all
"""

import json
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pipeline.phase2.storage import load_items_for_iteration, load_test_results, save_test_result


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

    # Per-dimension breakdown across preflection + reflection
    dim_scores: dict[str, list[float]] = {}
    for item in judged:
        for part in ("preflection", "reflection"):
            part_j = item["judgment"].get(part, {})
            for dim, score in part_j.get("scores", {}).items():
                dim_scores.setdefault(f"{part}_{dim}", []).append(score)

    print("\n  Per-dimension means:")
    for dim, vals in sorted(dim_scores.items()):
        print(f"    {dim}: {statistics.mean(vals):.2f}")


def cmd_failures(iteration: int, limit: int = 10, reasoning_limit: int = 200) -> None:
    """Print rejected items with judge reasoning."""
    items = load_items_for_iteration(iteration)
    rejected = [
        i for i in items
        if i.get("judgment") and i["judgment"]["decision"] == "reject"
    ]
    rejected.sort(key=lambda i: i["judgment"]["aggregate"])

    print(f"Rejected items ({len(rejected)} total, showing {min(limit, len(rejected))}):\n")
    for item in rejected[:limit]:
        j = item["judgment"]
        print(f"--- {item['item_id'][:16]} (score={j['aggregate']:.2f}, gold={item.get('is_gold', False)}) ---")
        print(f"  Text preview: {item['text'][:150]}...")
        print(f"  Preflection: {item.get('preflection', '')[:150]}...")
        print(f"  Reflection: {item.get('reflection', '')[:150]}...")
        print(f"  Charter elements: {item.get('charter_elements', [])}")
        for part in ("preflection", "reflection"):
            pj = j.get(part, {})
            print(f"  {part} scores: {pj.get('scores', {})}")
            print(f"  {part} reasoning: {pj.get('reasoning', '')[:reasoning_limit]}")
        print()


def cmd_show(item_ids: list[str], iteration: int, brief: bool = False,
             gold_only: bool = False) -> None:
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

    print(f"=== {item['item_id'][:16]} ({decision}, score={agg:.1f}, gold={item.get('is_gold', False)}) ===\n")
    print("--- SOURCE TEXT ---")
    if brief:
        print(item["text"][:300] + "...")
    else:
        print(item["text"][:rp] + " [REFLECTION POINT] " + item["text"][rp:])
    print(f"\n--- PREFLECTION ---\n{item.get('preflection', '')}")
    print(f"\n--- REFLECTION ---\n{item.get('reflection', '')}")
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
        print(json.dumps({
            "item_id": item["item_id"],
            "is_gold": item.get("is_gold"),
            "subset": item["subset"],
            "text_preview": item["text"][:500],
            "reflection_point": item["reflection_point"],
            "analysis": item.get("analysis"),
            "preflection": item.get("preflection"),
            "reflection": item.get("reflection"),
            "charter_elements": item.get("charter_elements"),
            "judgment": item.get("judgment"),
        }, indent=2))


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
        print(f"  Duplicate 5-word openers: {dict(sorted(dupes.items(), key=lambda x: -x[1]))}")

    # Formulaic closing patterns (last sentence, only duplicates)
    closings = Counter()
    for t in texts:
        sentences = [s.strip() for s in t.rstrip().rsplit(".", 1)]
        last = sentences[-1] if sentences else ""
        if last:
            closings[last[:80]] += 1
    closing_dupes = {k: v for k, v in closings.items() if v > 1}
    if closing_dupes:
        print(f"  Duplicate closings: {dict(sorted(closing_dupes.items(), key=lambda x: -x[1]))}")

    # Uniqueness
    unique_pct = len(set(texts)) / len(texts) * 100
    print(f"  Uniqueness: {unique_pct:.0f}% ({len(set(texts))}/{len(texts)})")


def cmd_diversity(iteration: int) -> None:
    """Show frequency-based diversity statistics for reflections, preflections, and analysis."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]

    print(f"Diversity check for iteration {iteration} ({len(judged)} items):\n")
    for field in ("reflection", "preflection", "analysis"):
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
        pre_s = j.get("preflection", {}).get("scores", {})
        ref_s = j.get("reflection", {}).get("scores", {})
        pre_str = " ".join(f"{k[:3]}={v}" for k, v in pre_s.items())
        ref_str = " ".join(f"{k[:3]}={v}" for k, v in ref_s.items())
        gold = "G" if item.get("is_gold") else " "
        print(
            f"{gold} {j['decision'][:3]:>3} {j['aggregate']:4.1f} | "
            f"pre[{pre_str}] ref[{ref_str}] | {item['item_id'][:12]}"
        )


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
    sliced = items[offset:offset + limit]
    print(f"Gold annotations ({len(items)} total, showing {offset}–{offset + len(sliced)}):\n")
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

        print("--- GENERATED PREFLECTION ---")
        print(item.get("preflection", ""))
        print("\n--- GOLD PREFLECTION ---")
        print(gold.get("preflection", ""))

        print("\n--- GENERATED REFLECTION ---")
        print(item.get("reflection", ""))
        print("\n--- GOLD REFLECTION ---")
        print(gold.get("reflection", ""))

        print("\n--- GENERATED CHARTER ---")
        print(item.get("charter_elements", []))
        print("\n--- GOLD CHARTER ---")
        print(gold.get("charter_elements", []))
        print()


def cmd_reviews(iteration: int | None = None, limit: int = 20) -> None:
    """Print human reviews, optionally filtered by iteration.

    Shows reviewer scores, decision, and notes alongside the judge's scores
    for calibration comparison.
    """
    from pipeline.phase2.storage import load_latest_reviews, load_review_comments

    reviews = load_latest_reviews()
    all_comments = load_review_comments()
    if not reviews:
        print("No human reviews yet.")
        return

    # Filter by iteration if specified
    filtered = list(reviews.values())
    if iteration is not None:
        filtered = [r for r in filtered if r["iteration"] == iteration]
        items = load_items_for_iteration(iteration)
    else:
        items = []
        # Load items for all reviewed iterations
        seen_iters: set[int] = set()
        for r in filtered:
            if r["iteration"] not in seen_iters:
                seen_iters.add(r["iteration"])
                items.extend(load_items_for_iteration(r["iteration"]))

    items_by_key = {(i["item_id"], i["iteration"]): i for i in items}

    print(f"Human reviews ({len(filtered)} total, showing {min(limit, len(filtered))}):\n")
    for r in filtered[:limit]:
        item = items_by_key.get((r["item_id"], r["iteration"]))
        judge_agg = ""
        judge_decision = ""
        if item and item.get("judgment"):
            j = item["judgment"]
            judge_agg = f"{j['aggregate']:.2f}"
            judge_decision = j["decision"]

        print(f"--- {r['item_id'][:16]} iter={r['iteration']} reviewer={r['reviewer_id']} ---")
        print(f"  Human:  decision={r['decision']}  aggregate={r['aggregate']:.2f}")
        if judge_agg:
            print(f"  Judge:  decision={judge_decision}  aggregate={judge_agg}")

        scores = r["scores"]
        is_per_part = scores and isinstance(next(iter(scores.values())), dict)
        if is_per_part:
            for part in ("preflection", "reflection"):
                human_s = scores.get(part, {})
                judge_s = {}
                if item and item.get("judgment"):
                    judge_s = item["judgment"].get(part, {}).get("scores", {})
                dims = sorted(set(human_s) | set(judge_s))
                pairs = " ".join(
                    f"{d[:3]}={human_s.get(d, '?')}/{judge_s.get(d, '?')}"
                    for d in dims
                )
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
                print(f"    {c['commenter_id']} ({c['timestamp'][:19]}): {c['comment']}")
        print()


def cmd_filter(iteration: int, dim: str, below: float, part: str | None = None) -> None:
    """Filter items by score threshold on a specific dimension."""
    assert dim, "--dim is required"
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]

    parts = [part] if part else ["preflection", "reflection"]
    hits = []
    for item in judged:
        j = item["judgment"]
        for p in parts:
            score = j.get(p, {}).get("scores", {}).get(dim)
            if score is not None and score < below:
                text = item.get(p, "") or ""
                hits.append((item["item_id"], j["decision"], j["aggregate"], p, score, text[:80]))

    hits.sort(key=lambda x: x[4])
    print(f"Items with {dim} < {below} in iteration {iteration} ({len(hits)} hits):\n")
    for iid, dec, agg, p, sc, preview in hits:
        print(f"  {iid[:16]} {dec:>3} agg={agg:.1f} {p[:3]}_{dim[:3]}={sc} | {preview}...")


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
        for part in ("preflection", "reflection"):
            for dim in judged[0]["judgment"].get(part, {}).get("scores", {}):
                key = f"{part[:3]}_{dim[:3]}"
                if key not in all_dim_keys:
                    all_dim_keys.append(key)
        break

    # Header
    dim_header = " ".join(f"{k:>7}" for k in all_dim_keys)
    print(f"{'iter':>4} {'acc%':>5} {'mean':>5} {dim_header}  gen_prompt / judge_prompt")

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

        dim_means: dict[str, float] = {}
        for item in judged:
            for part in ("preflection", "reflection"):
                for dim, sc in item["judgment"].get(part, {}).get("scores", {}).items():
                    key = f"{part[:3]}_{dim[:3]}"
                    dim_means.setdefault(key, []).append(sc)
        dim_str = " ".join(f"{statistics.mean(dim_means.get(k, [0])):7.2f}" for k in all_dim_keys)

        gen_p = run.get("gen_prompt", "?")
        judge_p = run.get("judge_prompt", "?")
        print(f"{it:>4} {acc_pct:5.0f} {mean:5.2f} {dim_str}  {gen_p} / {judge_p}")


def cmd_correlations() -> None:
    """Print judge-human correlation stats grouped by judge prompt version.

    For each judge version that has re-judgment data, computes:
    - Decision agreement rate (accept/reject match %)
    - Mean absolute score difference (judge aggregate vs human aggregate)
    - Per-dimension score diffs where both human and judge scored the same dims
    """
    from pipeline.phase2.storage import load_judge_correlations, load_latest_reviews

    correlations = load_judge_correlations()
    if not correlations:
        print("No judge correlations yet. Correlations are recorded after each iteration when reviewed items exist.")
        return

    reviews = load_latest_reviews()
    # Build lookup: (item_id, iteration) -> review (use first reviewer found)
    review_by_item: dict[tuple[str, int], dict] = {}
    for (item_id, iteration, _reviewer), review in reviews.items():
        key = (item_id, iteration)
        if key not in review_by_item:
            review_by_item[key] = review

    # Group correlations by (judge_prompt, judge_model)
    by_prompt: dict[tuple[str, str], list[dict]] = {}
    for c in correlations:
        key = (c["judge_prompt"], c.get("judge_model", "unknown"))
        by_prompt.setdefault(key, []).append(c)

    for (prompt_name, model_name) in sorted(by_prompt):
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
            is_per_part = human_scores and isinstance(next(iter(human_scores.values()), None), dict)
            if is_per_part:
                for part in ("preflection", "reflection"):
                    h_part = human_scores.get(part, {})
                    j_part = j.get(part, {}).get("scores", {})
                    for dim in set(h_part) & set(j_part):
                        dim_key = f"{part[:3]}_{dim[:3]}"
                        dim_diffs.setdefault(dim_key, []).append(
                            abs(j_part[dim] - h_part[dim])
                        )

        if matched == 0:
            print(f"\n{prompt_name} / {model_name}: {len(entries)} correlations, 0 matched to reviews")
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


def cmd_test_generate(prompt_path: str, item_ids: list[str] | None = None,
                      n: int = 3, phase: str = "A") -> None:
    """Generate with a prompt file without saving to main items table.

    Loads items from the latest iteration, runs generate_batch(save=False),
    saves a test_results entry.
    """
    from pipeline.config import CHARTER_PATH, generator_api_name, load_config, resolve_prompt_path
    from pipeline.phase2.run import generate_batch, make_api_client
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    client, semaphore = make_api_client(cfg)
    gen_model = generator_api_name(cfg)
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")

    runs = load_runs()
    assert runs, "No iterations yet — run at least one iteration first"
    latest_iter = runs[-1]["iteration"]

    all_items = load_items_for_iteration(latest_iter)
    assert all_items, f"No items found for iteration {latest_iter}"

    if item_ids:
        items = [i for i in all_items if any(i["item_id"].startswith(iid) for iid in item_ids)]
        assert items, f"No items matching {item_ids} in iteration {latest_iter}"
    else:
        items = random.sample(all_items, min(n, len(all_items)))

    prompt = Path(prompt_path)
    assert prompt.exists(), f"Prompt file not found: {prompt}"

    print(f"Test generating {len(items)} items with {prompt.name}...")
    generated = generate_batch(
        items, prompt, charter_text, gen_model,
        iteration=latest_iter, client=client, semaphore=semaphore, save=False,
    )

    test_id = _make_test_id("tg")
    result_items = []
    for g in generated:
        result_items.append({
            "item_id": g["item_id"],
            "preflection": g.get("preflection", "")[:200],
            "reflection": g.get("reflection", "")[:200],
            "charter_elements": g.get("charter_elements", []),
        })

    record = {
        "test_id": test_id,
        "type": "generate",
        "phase": phase,
        "prompt": prompt.name,
        "model_alias": cfg.phase2.generator.model,
        "items": result_items,
        "summary": {"n_items": len(generated)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_test_result(record)

    print(f"\nTest {test_id}: generated {len(generated)} items")
    for g in generated:
        print(f"  {g['item_id'][:12]}: pre={g.get('preflection', '')[:60]}...")
    print("Saved to test_results")


def cmd_test_judge(prompt_path: str, item_ids: list[str] | None = None,
                   iteration: int | None = None, n: int = 3, phase: str = "A") -> None:
    """Judge items with a prompt file without saving to main items table.

    Loads generated items from specified iteration, runs judge_batch(save=False),
    saves a test_results entry.
    """
    from pipeline.config import judge_api_name, load_config
    from pipeline.phase2.run import judge_batch, make_api_client
    from pipeline.phase2.storage import load_runs

    cfg = load_config()
    client, semaphore = make_api_client(cfg)
    jdg_model = judge_api_name(cfg)

    runs = load_runs()
    assert runs, "No iterations yet — run at least one iteration first"
    iter_num = iteration if iteration is not None else runs[-1]["iteration"]

    all_items = load_items_for_iteration(iter_num)
    generated = [i for i in all_items if i.get("analysis")]
    assert generated, f"No generated items in iteration {iter_num}"

    if item_ids:
        items = [i for i in generated if any(i["item_id"].startswith(iid) for iid in item_ids)]
        assert items, f"No items matching {item_ids} in iteration {iter_num}"
    else:
        items = random.sample(generated, min(n, len(generated)))

    prompt = Path(prompt_path)
    assert prompt.exists(), f"Prompt file not found: {prompt}"

    print(f"Test judging {len(items)} items with {prompt.name}...")
    judged = judge_batch(
        items, prompt, jdg_model, iteration=iter_num,
        accept_threshold=cfg.phase2.scoring.accept_threshold,
        client=client, semaphore=semaphore, save=False,
    )

    scores = [j["judgment"]["aggregate"] for j in judged]
    n_acc = sum(1 for j in judged if j["judgment"]["decision"] == "accept")
    mean_score = statistics.mean(scores) if scores else 0.0

    test_id = _make_test_id("tj")
    result_items = []
    for j in judged:
        jdg = j["judgment"]
        result_items.append({
            "item_id": j["item_id"],
            "aggregate": jdg["aggregate"],
            "decision": jdg["decision"],
            "preflection_scores": jdg["preflection"]["scores"],
            "reflection_scores": jdg["reflection"]["scores"],
        })

    record = {
        "test_id": test_id,
        "type": "judge",
        "phase": phase,
        "prompt": prompt.name,
        "model_alias": cfg.phase2.judge.model,
        "items": result_items,
        "summary": {"n_items": len(judged), "mean_score": round(mean_score, 3), "n_accepted": n_acc},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_test_result(record)

    print(f"\nTest {test_id}: judged {len(judged)} items (mean={mean_score:.2f}, accepted={n_acc})")
    for j in judged:
        jdg = j["judgment"]
        print(f"  {j['item_id'][:12]}: {jdg['decision']} ({jdg['aggregate']:.2f})")
    print("Saved to test_results")


def cmd_run_batch(phase: str = "A") -> None:
    """Run a full generate->judge iteration, auto-detecting latest prompts.

    Saves to main items + runs tables AND to test_results for tracking.
    """
    from pipeline.config import load_config
    from pipeline.phase2.loop import _detect_new_prompts, _update_config
    from pipeline.phase2.run import run_iteration

    cfg = load_config()

    # Auto-detect latest prompts (resolve _latest sentinels for comparison)
    new_gen, new_judge = _detect_new_prompts(cfg)
    alias = cfg.phase2.generator.model
    from pipeline.phase2.loop import _resolve_config_prompt
    current_gen = _resolve_config_prompt(cfg.phase2.generator.prompt, alias)
    current_judge = _resolve_config_prompt(cfg.phase2.judge.prompt, alias)
    if new_gen != current_gen or new_judge != current_judge:
        cfg = _update_config(cfg, new_gen, new_judge)
        print(f"Updated config: gen={new_gen}, judge={new_judge}")

    result = run_iteration(cfg, source=f"phase_{phase.lower()}")

    scores = [it["judgment"]["aggregate"] for it in result["items"] if it.get("judgment")]
    mean_score = statistics.mean(scores) if scores else 0.0

    test_id = _make_test_id("tb")
    record = {
        "test_id": test_id,
        "type": "batch",
        "phase": phase,
        "prompt": f"{cfg.phase2.generator.prompt}+{cfg.phase2.judge.prompt}",
        "model_alias": cfg.phase2.generator.model,
        "items": [{"item_id": it["item_id"]} for it in result["items"]],
        "summary": {
            "n_items": result["n_items"],
            "mean_score": round(mean_score, 3),
            "n_accepted": result["n_accepted"],
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_test_result(record)

    print(f"\nBatch {test_id}: {result['n_accepted']}/{result['n_items']} accepted, mean={mean_score:.2f}")


def cmd_test_results(phase: str | None = None, type_filter: str | None = None) -> None:
    """List test results, optionally filtered by phase and/or type."""
    results = load_test_results(phase=phase)
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
        print(f"  {r['test_id']}  {r['type']:>8}  phase={r.get('phase', '?')}  "
              f"prompt={r.get('prompt', '?')}  n={n}{mean_str}{acc_str}  "
              f"{r.get('timestamp', '')[:19]}")


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]

    if "--help" in args or "-h" in args:
        print(__doc__)
        sys.exit(0)

    def _get_flag(flag: str, default: str | None = None) -> str | None:
        if flag in args:
            return args[args.index(flag) + 1]
        return default

    def _get_flag_int(flag: str, default: int | None = None) -> int | None:
        val = _get_flag(flag)
        return int(val) if val is not None else default

    if cmd == "summary":
        cmd_summary(int(args[1]))
    elif cmd == "failures":
        cmd_failures(int(args[1]), limit=_get_flag_int("--limit", 10),
                     reasoning_limit=_get_flag_int("--reasoning-limit", 200))
    elif cmd == "show":
        brief = "--brief" in args
        gold_only = "--gold" in args
        positional = [a for a in args[1:] if not a.startswith("--")]
        if gold_only:
            iteration = int(positional[0])
            cmd_show([], iteration, brief=brief, gold_only=True)
        else:
            item_ids = positional[0].split(",")
            iteration = int(positional[1])
            cmd_show(item_ids, iteration, brief=brief)
    elif cmd == "item":
        cmd_item(args[1], int(args[2]))
    elif cmd == "diversity":
        cmd_diversity(int(args[1]))
    elif cmd == "scores":
        cmd_scores(int(args[1]))
    elif cmd == "gold":
        cmd_gold(limit=_get_flag_int("--limit", 5), offset=_get_flag_int("--offset", 0),
                 verbose="--verbose" in args)
    elif cmd == "compare":
        cmd_compare(args[1], int(args[2]))
    elif cmd == "reviews":
        iteration = int(args[1]) if len(args) > 1 and not args[1].startswith("-") else None
        cmd_reviews(iteration=iteration, limit=_get_flag_int("--limit", 20))
    elif cmd == "filter":
        cmd_filter(int(args[1]), dim=_get_flag("--dim"), below=float(_get_flag("--below")),
                   part=_get_flag("--part"))
    elif cmd == "trend":
        cmd_trend()
    elif cmd == "test_generate":
        item_ids_str = _get_flag("--items")
        item_ids = item_ids_str.split(",") if item_ids_str else None
        cmd_test_generate(
            args[1],
            item_ids=item_ids,
            n=_get_flag_int("--n", 3),
            phase=_get_flag("--phase", "A"),
        )
    elif cmd == "test_judge":
        item_ids_str = _get_flag("--items")
        item_ids = item_ids_str.split(",") if item_ids_str else None
        cmd_test_judge(
            args[1],
            item_ids=item_ids,
            iteration=_get_flag_int("--iteration"),
            n=_get_flag_int("--n", 3),
            phase=_get_flag("--phase", "A"),
        )
    elif cmd == "run_batch":
        cmd_run_batch(phase=_get_flag("--phase", "A"))
    elif cmd == "test_results":
        cmd_test_results(
            phase=_get_flag("--phase"),
            type_filter=_get_flag("--type"),
        )
    elif cmd == "correlations":
        cmd_correlations()
    elif cmd == "rejudge_all":
        cmd_rejudge_all()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
