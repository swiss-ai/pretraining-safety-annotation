"""Stratified sampling from ClimbMix and 4chan data sources."""

import json
import random
from pathlib import Path

import duckdb
import requests

from annotation.config import CLIMBMIX_DATASET, FOURCHAN_DATASET, ITEMS_PER_SOURCE
from annotation.storage import compute_item_id

ANNOTATION_DIR = Path(__file__).parent


def sample_path() -> Path:
    """Return the path to the persisted sample file."""
    return ANNOTATION_DIR / "sample.json"


def _compute_reflection_point(text: str, rng: random.Random) -> int:
    """Pick a reflection point between 10%-90% of text, snapped to a word boundary."""
    min_pos = max(1, int(len(text) * 0.1))
    max_pos = max(min_pos + 1, int(len(text) * 0.9))
    char_pos = rng.randint(min_pos, max_pos)
    space_idx = text.find(" ", char_pos)
    if space_idx != -1 and space_idx - char_pos < 50:
        char_pos = space_idx
    return char_pos


def _get_parquet_urls(dataset: str) -> list[str]:
    """Fetch parquet file URLs for a HuggingFace dataset via the datasets-server API."""
    resp = requests.get(
        "https://datasets-server.huggingface.co/parquet",
        params={"dataset": dataset},
    )
    assert resp.status_code == 200, f"Failed to fetch parquet URLs for {dataset}: {resp.status_code}"
    data = resp.json()
    assert "parquet_files" in data, f"No parquet_files in response for {dataset}: {list(data.keys())}"
    urls = [f["url"] for f in data["parquet_files"]]
    assert len(urls) > 0, f"No parquet files found for {dataset}"
    return urls


def _duckdb_query(urls: list[str], sql_template: str) -> list[tuple]:
    """Run a DuckDB query over remote parquet files.

    sql_template should contain {source} as a placeholder for the parquet source.
    """
    conn = duckdb.connect()
    conn.execute("SET enable_progress_bar = true")
    source = ", ".join(f"'{u}'" for u in urls)
    source_expr = f"read_parquet([{source}])"
    sql = sql_template.format(source=source_expr)
    rows = conn.execute(sql).fetchall()
    conn.close()
    return rows


def load_climbmix_items(n: int, seed: int = 42) -> list[dict]:
    """Load n random items from ClimbMix."""
    rng = random.Random(seed)
    print("[ClimbMix] Fetching parquet URLs...", flush=True)
    urls = _get_parquet_urls(CLIMBMIX_DATASET)
    print(f"[ClimbMix] Querying {len(urls)} parquet files via DuckDB...", flush=True)
    rows = _duckdb_query(urls, f"""
        SELECT text FROM {{source}}
        ORDER BY random()
        LIMIT {n}
    """)
    print(f"[ClimbMix] Got {len(rows)} items", flush=True)

    items = []
    for (text,) in rows:
        assert isinstance(text, str) and len(text) > 0, "Empty text in ClimbMix"
        items.append({
            "item_id": compute_item_id(text),
            "subset": "climbmix",
            "text": text,
            "reflection_point": _compute_reflection_point(text, rng),
        })
    assert len(items) > 0, "No ClimbMix items loaded"
    return items


def load_4chan_items(n: int, seed: int = 42) -> list[dict]:
    """Load one random thread per board from 4chan-archive, up to n boards."""
    rng = random.Random(seed)
    print("[4chan] Fetching parquet URLs...", flush=True)
    urls = _get_parquet_urls(FOURCHAN_DATASET)
    print(f"[4chan] Querying {len(urls)} parquet files via DuckDB...", flush=True)
    rows = _duckdb_query(urls, f"""
        SELECT board, posts FROM (
            SELECT board, posts,
                   ROW_NUMBER() OVER (PARTITION BY board ORDER BY random()) AS rn
            FROM {{source}}
        ) WHERE rn = 1
        ORDER BY random()
        LIMIT {n}
    """)
    print(f"[4chan] Got {len(rows)} rows from {n} requested boards", flush=True)

    items = []
    for board, posts in rows:
        text = "\n\n".join(
            f"[Post by anon_{p['poster']}]\n{p['content']}"
            for p in posts
            if p["content"].strip()
        )
        if not text.strip():
            continue
        items.append({
            "item_id": compute_item_id(text),
            "subset": f"4chan/{board}",
            "text": text,
            "reflection_point": _compute_reflection_point(text, rng),
        })
    assert len(items) > 0, "No 4chan items loaded"
    return items


def load_items_from_sources(seed: int = 42) -> list[dict]:
    """Load items from both sources, returning a combined list.

    ClimbMix: random sample of ITEMS_PER_SOURCE items (no stratification).
    4chan: one random thread per board (stratified by board), needs a larger pool.
    """
    climbmix = load_climbmix_items(ITEMS_PER_SOURCE, seed=seed)
    fourchan = load_4chan_items(n=150, seed=seed)
    return climbmix + fourchan


def draw_stratified_sample(
    items: list[dict],
    n: int,
    min_per_stratum: int = 1,
    seed: int = 42,
) -> list[str]:
    """Draw a stratified sample of item_ids proportional to stratum sizes.

    Each stratum gets at least min_per_stratum items (or all items if the
    stratum is smaller). Remaining budget is allocated proportionally.
    """
    assert len(items) > 0, "Cannot sample from empty items list"
    assert n > 0, "Sample size must be positive"

    strata: dict[str, list[str]] = {}
    for item in items:
        strata.setdefault(item["subset"], []).append(item["item_id"])

    rng = random.Random(seed)
    for ids in strata.values():
        rng.shuffle(ids)

    selected: list[str] = []
    remaining_budget = n

    for stratum_ids in strata.values():
        take = min(min_per_stratum, len(stratum_ids))
        selected.extend(stratum_ids[:take])
        remaining_budget -= take

    if remaining_budget > 0:
        total_remaining = sum(
            max(0, len(ids) - min_per_stratum) for ids in strata.values()
        )
        if total_remaining > 0:
            for stratum_ids in strata.values():
                available = stratum_ids[min_per_stratum:]
                if not available:
                    continue
                proportional_n = round(
                    remaining_budget * len(available) / total_remaining
                )
                take = min(proportional_n, len(available))
                selected.extend(available[:take])

    seen: set[str] = set()
    unique: list[str] = []
    for item_id in selected:
        if item_id not in seen:
            seen.add(item_id)
            unique.append(item_id)

    return unique


def save_sample(items: list[dict], sample_ids: list[str]) -> None:
    """Persist the sample items and selected IDs to disk."""
    items_by_id = {item["item_id"]: item for item in items}
    sample_items = [items_by_id[sid] for sid in sample_ids]
    path = sample_path()
    path.write_text(json.dumps(sample_items, indent=2))


def load_sample() -> list[dict] | None:
    """Load a previously saved sample. Returns None if no sample exists."""
    path = sample_path()
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_annotator_queue(
    sample_ids: list[str],
    annotator_id: str,
    completed_ids: set[str],
) -> list[str]:
    """Return remaining item_ids in a deterministic random order per annotator."""
    remaining = [iid for iid in sample_ids if iid not in completed_ids]
    rng = random.Random(annotator_id)
    rng.shuffle(remaining)
    return remaining
