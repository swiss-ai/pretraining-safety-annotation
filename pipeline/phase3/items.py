"""Phase 3 item-pool builder + reviewed-item loader.

`build_item_pool` produces a fresh diverse pool from the full Dolma3 dataset.
`ensure_item_pool` materializes it once on disk and returns it on resume.
`load_reviewed_items` constructs the reviewed-item set from the phase 2
SQLite tables for the judge eval's vs-human path.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from pipeline.data import sample_diverse
from pipeline.log import logger
from pipeline.phase2.storage import load_items_for_iteration, load_reviews
from pipeline.phase3.storage import JsonlRunStore
from pipeline.tokenizer import compute_reflection_point

_FOUR_VOICES = ("preflection_3p", "preflection_1p", "reflection_1p", "reflection_3p")


def build_item_pool(
    n_items: int, seed: int, max_tokens: int
) -> tuple[list[dict], str]:
    """Sample n_items diversely from the full Dolma3 dataset.

    Returns (items, dataset_revision). Each item has item_id, text,
    safety_score, reflection_point.
    """
    result = sample_diverse(n_items=n_items, seed=seed, max_tokens=max_tokens)
    # `sample_diverse` returns a dict with keys "items" and "dataset_revision",
    # OR a (items, revision) tuple — accept either, since the test fakes use
    # the tuple form.
    if isinstance(result, tuple):
        items, dataset_revision = result
    elif isinstance(result, dict):
        items = list(result["items"])
        dataset_revision = result.get("dataset_revision", "")
    else:
        raise TypeError(
            f"sample_diverse returned unexpected type: {type(result).__name__}"
        )

    rp_rng = random.Random(f"reflection_point_v1::{seed}")
    pool: list[dict] = []
    for raw in items:
        text = raw["text"]
        item = dict(raw)
        # compute_reflection_point's signature is (text, rng); the test fakes
        # accept (text, **kwargs) so passing rng as a kwarg lets both work.
        try:
            item["reflection_point"] = compute_reflection_point(text, rng=rp_rng)
        except TypeError:
            item["reflection_point"] = compute_reflection_point(text, rp_rng)
        pool.append(item)
    return pool, dataset_revision


def ensure_item_pool(
    store: JsonlRunStore,
    n_items: int,
    seed: int,
    max_tokens: int,
) -> list[dict]:
    """Materialize items.jsonl exactly once. On resume, load from disk."""
    items_rel = "items.jsonl"
    existing = store.read_all(items_rel)
    if existing:
        # Resume path: validate metadata against the freshly-requested pool
        # parameters. We don't rebuild — just check what's recorded.
        meta = store.read_metadata()
        for key, expected in (
            ("n_items", n_items),
            ("seed", seed),
            ("max_tokens", max_tokens),
        ):
            if meta.get(key) is not None and meta.get(key) != expected:
                raise ValueError(
                    f"ensure_item_pool: existing pool was built with "
                    f"{key}={meta.get(key)!r} but caller requested "
                    f"{key}={expected!r}"
                )
        if len(existing) != n_items:
            raise ValueError(
                f"ensure_item_pool: existing items.jsonl has {len(existing)} "
                f"rows, expected {n_items}"
            )
        # On resume, detect metadata corruption: the original dataset revision
        # (pinned at first build) must still match the recorded revision.
        # We don't call sample_diverse again — just compare the two metadata keys.
        original_rev = meta.get("_original_dataset_revision")
        recorded_rev = meta.get("dataset_revision")
        if (
            original_rev is not None
            and recorded_rev is not None
            and original_rev != recorded_rev
        ):
            raise ValueError(
                f"ensure_item_pool: dataset_revision in metadata "
                f"({recorded_rev!r}) differs from the original revision the "
                f"pool was built against ({original_rev!r}). The metadata "
                f"file appears corrupted or the dataset has changed under us."
            )
        return existing

    # First-time build path
    pool, dataset_revision = build_item_pool(n_items, seed, max_tokens)
    for item in pool:
        store.append(items_rel, item)
    store.flush()

    meta = store.read_metadata()
    meta.update(
        {
            "n_items": n_items,
            "seed": seed,
            "max_tokens": max_tokens,
            "dataset_revision": dataset_revision,
            # Pinned copy for resume-time tamper detection.
            "_original_dataset_revision": dataset_revision,
        }
    )
    store.write_metadata(meta)
    return pool


def load_reviewed_items(reviewer_policy: str = "average") -> list[dict]:
    """Build the reviewed item set from phase 2 reviews + items tables.

    See module docstring.
    """
    if reviewer_policy not in ("average", "first", "all"):
        raise ValueError(f"Unknown reviewer_policy: {reviewer_policy}")

    reviews = load_reviews()
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in reviews:
        # Validate four_voice schema BEFORE grouping so a bad row fails loud.
        scores = r.get("scores")
        if not isinstance(scores, dict):
            raise ValueError(
                f"review for ({r.get('item_id')}, {r.get('iteration')}) has "
                f"non-dict scores: {type(scores).__name__}"
            )
        missing = set(_FOUR_VOICES) - set(scores.keys())
        if missing:
            raise ValueError(
                f"review for ({r.get('item_id')}, {r.get('iteration')}) is "
                f"not four_voice schema, missing: {missing}"
            )
        grouped[(r["item_id"], r["iteration"])].append(r)

    # Cache the items table per iteration so we don't requery in the inner loop.
    items_cache: dict[int, dict[str, dict]] = {}

    def _items_for(iteration: int) -> dict[str, dict]:
        if iteration not in items_cache:
            rows = load_items_for_iteration(iteration)
            items_cache[iteration] = {row["item_id"]: row for row in rows}
        return items_cache[iteration]

    out: list[dict] = []
    n_orphans = 0
    for (item_id, iteration), reviewers in grouped.items():
        items_by_id = _items_for(iteration)
        item_row = items_by_id.get(item_id)
        if item_row is None:
            n_orphans += 1
            logger.warning(
                "load_reviewed_items: dropping orphan review for "
                "({}, iter={}) — no matching items row",
                item_id,
                iteration,
            )
            continue

        if reviewer_policy == "all":
            for r in reviewers:
                out.append(_make_reviewed_row(item_row, r, include_reviewer_id=True))
        elif reviewer_policy == "first":
            sorted_reviewers = sorted(reviewers, key=lambda r: r.get("timestamp", ""))
            out.append(_make_reviewed_row(item_row, sorted_reviewers[0]))
        else:  # "average"
            avg_review = _average_reviews(reviewers)
            out.append(_make_reviewed_row(item_row, avg_review))

    if n_orphans:
        logger.info("load_reviewed_items: dropped {} orphan reviews", n_orphans)
    return out


def _average_reviews(reviewers: list[dict]) -> dict:
    """Per-dim average of `scores` across multiple reviewers (four_voice)."""
    avg_scores: dict[str, dict[str, float]] = {}
    for voice in _FOUR_VOICES:
        avg_scores[voice] = {}
        # Union of dim names across reviewers
        dims: set[str] = set()
        for r in reviewers:
            dims.update(r["scores"][voice].keys())
        for dim in dims:
            vals = [
                r["scores"][voice][dim]
                for r in reviewers
                if dim in r["scores"][voice]
            ]
            if vals:
                avg_scores[voice][dim] = sum(vals) / len(vals)

    aggregates = [r.get("aggregate", 0.0) for r in reviewers]
    avg_aggregate = sum(aggregates) / len(aggregates) if aggregates else 0.0
    return {
        "scores": avg_scores,
        "aggregate": avg_aggregate,
        "decision": _majority_decision([r.get("decision") for r in reviewers]),
        "notes": "",
        "timestamp": reviewers[0].get("timestamp", ""),
    }


def _majority_decision(decisions: Iterable) -> str:
    counts: dict[str, int] = defaultdict(int)
    for d in decisions:
        if d:
            counts[d] += 1
    if not counts:
        return "accept"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _make_reviewed_row(
    item_row: dict, review: dict, include_reviewer_id: bool = False
) -> dict:
    """Build a generated-item-shaped dict with `human_review` attached."""
    out = {**item_row}
    hr = {
        "scores": review["scores"],
        "aggregate": review.get("aggregate"),
        "decision": review.get("decision"),
        "notes": review.get("notes", ""),
        "timestamp": review.get("timestamp", ""),
    }
    if include_reviewer_id and "reviewer_id" in review:
        hr["reviewer_id"] = review["reviewer_id"]
    out["human_review"] = hr
    return out
