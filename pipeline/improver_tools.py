"""CLI tools for the improver agent to efficiently query iteration data.

Usage (via Bash tool):
    python -m pipeline.improver_tools summary <iteration>
    python -m pipeline.improver_tools failures <iteration> [--limit N]
    python -m pipeline.improver_tools show <item_id> <iteration>
    python -m pipeline.improver_tools item <item_id> <iteration>
    python -m pipeline.improver_tools diversity <iteration>
    python -m pipeline.improver_tools scores <iteration>
    python -m pipeline.improver_tools gold [--limit N]
    python -m pipeline.improver_tools compare <item_id> <iteration>
"""

import json
import statistics
import sys

from pipeline.storage import load_items_for_iteration


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


def cmd_failures(iteration: int, limit: int = 10) -> None:
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
            print(f"  {part} reasoning: {pj.get('reasoning', '')[:200]}")
        print()


def cmd_show(item_id: str, iteration: int) -> None:
    """Print source text, preflection, and reflection for an item — easy to read."""
    items = load_items_for_iteration(iteration)
    matches = [i for i in items if i["item_id"].startswith(item_id)]
    if not matches:
        print(f"No item matching '{item_id}' in iteration {iteration}")
        return

    for item in matches:
        rp = item["reflection_point"]
        j = item.get("judgment", {})
        decision = j.get("decision", "?")
        agg = j.get("aggregate", 0)

        print(f"=== {item['item_id'][:16]} ({decision}, score={agg:.1f}, gold={item.get('is_gold', False)}) ===\n")
        print(f"--- SOURCE TEXT ---")
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


def cmd_diversity(iteration: int) -> None:
    """Show opening phrases of reflections and preflections for diversity check."""
    items = load_items_for_iteration(iteration)
    judged = [i for i in items if i.get("judgment")]

    print(f"Diversity check for iteration {iteration} ({len(judged)} items):\n")
    print("=== Reflection openings ===")
    for item in judged[:15]:
        refl = item.get("reflection", "")
        print(f"  {refl[:80]}...")

    print("\n=== Preflection openings ===")
    for item in judged[:15]:
        pre = item.get("preflection", "")
        print(f"  {pre[:80]}...")

    print("\n=== Analysis openings ===")
    for item in judged[:15]:
        ana = item.get("analysis", "")
        print(f"  {ana[:80]}...")


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
    """Load gold annotations from the annotation file."""
    from pipeline.config import PROJECT_ROOT

    gold_path = PROJECT_ROOT / "data" / "annotation" / "annotations.jsonl"
    assert gold_path.exists(), f"Gold annotations not found at {gold_path}"
    items = []
    with open(gold_path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def cmd_gold(limit: int = 5) -> None:
    """Print gold annotations for reference — shows what good output looks like."""
    items = _load_gold()
    print(f"Gold annotations ({len(items)} total, showing {min(limit, len(items))}):\n")
    for item in items[:limit]:
        print(f"=== {item['item_id'][:16]} (subset={item['subset']}) ===")
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


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    cmd = args[0]
    if cmd == "summary":
        cmd_summary(int(args[1]))
    elif cmd == "failures":
        limit = 10
        if "--limit" in args:
            limit = int(args[args.index("--limit") + 1])
        cmd_failures(int(args[1]), limit=limit)
    elif cmd == "show":
        cmd_show(args[1], int(args[2]))
    elif cmd == "item":
        cmd_item(args[1], int(args[2]))
    elif cmd == "diversity":
        cmd_diversity(int(args[1]))
    elif cmd == "scores":
        cmd_scores(int(args[1]))
    elif cmd == "gold":
        limit = 5
        if "--limit" in args:
            limit = int(args[args.index("--limit") + 1])
        cmd_gold(limit=limit)
    elif cmd == "compare":
        cmd_compare(args[1], int(args[2]))
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
