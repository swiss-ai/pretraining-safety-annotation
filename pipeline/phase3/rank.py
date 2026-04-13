"""Phase 3 ranking analytics.

Reads JSONL files from a phase 3 run dir and computes:
- `rank_generators(run_id)` — mean aggregate, accept rate, per-dim mean,
  accept-by-safety-score breakdown, per-category failure rates.
- `rank_judges(run_id)` — vs-gold and vs-human correlation metrics
  (Spearman, Pearson, decision concordance, Cohen's kappa) plus failure
  rates.

Pure functions: read files, return dicts. The CLI in `__main__.py` is
responsible for printing tables.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from pipeline.config import load_config
from pipeline.log import logger
from pipeline.phase3.eval_generators import _eval_root

_FOUR_VOICES = ("preflection_3p", "preflection_1p", "reflection_1p", "reflection_3p")


def _meta_prompt_id(meta_dict: dict) -> str:
    """Build the prompt_id string from a metadata candidate dict.

    Uses ``prompt_reflection`` as the primary identifier (matching the
    file naming convention in ``_prompt_id``).  Falls back to the legacy
    ``prompt`` field for old runs.
    """
    r = meta_dict.get("prompt_reflection", "")
    p = meta_dict.get("prompt_preflection", "")
    if r or p:
        return r or p
    # Legacy fallback: old runs stored a single "prompt" field.
    return meta_dict.get("prompt", "")


def _resolve_run_dir(run_id: str, eval_dir: Path | str | None) -> Path:
    """Resolve a run dir, preferring an explicit `eval_dir` kwarg over the config root."""
    if eval_dir is not None:
        return Path(eval_dir) / run_id
    return _eval_root(load_config()) / run_id


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.warning(
                    "rank: skipping unparseable line {} in {}: {}", i + 1, path, e
                )
    return rows


def _iter_jsonl(path: Path):
    """Streaming reader, generator that yields one row at a time."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                logger.warning(
                    "rank: skipping unparseable line {} in {}: {}", i + 1, path, e
                )


# ----------------------------------------------------------------- failure rates


def _failure_categories_distinct_items(failures_path: Path) -> dict[str, set]:
    """Return {category → set of distinct item_ids in that category}."""
    out: dict[str, set] = defaultdict(set)
    if not failures_path.exists():
        return out
    for row in _iter_jsonl(failures_path):
        if not row:
            continue
        cat = row.get("category")
        item_id = row.get("item_id")
        if cat and item_id:
            out[cat].add(item_id)
    return out


# ----------------------------------------------------------------- rank_generators


def _judgment_aggregate(row: dict) -> float | None:
    j = row.get("judgment") or {}
    agg = j.get("aggregate")
    if isinstance(agg, (int, float)) and not (
        isinstance(agg, float) and math.isnan(agg)
    ):
        return float(agg)
    return None


def _judgment_decision(row: dict) -> str | None:
    j = row.get("judgment") or {}
    return j.get("decision")


def _judgment_mode_aggregate(row: dict, mode: str) -> float | None:
    """Return the per-mode aggregate (e.g. ``reflection_aggregate``) from a judgment row.

    Uses ``.get()`` so old data without per-mode keys returns ``None``.
    """
    j = row.get("judgment") or {}
    key = f"{mode}_aggregate"
    agg = j.get(key)
    if isinstance(agg, (int, float)) and not (
        isinstance(agg, float) and math.isnan(agg)
    ):
        return float(agg)
    return None


def _judgment_mode_decision(row: dict, mode: str) -> str | None:
    """Return the per-mode decision (e.g. ``reflection_decision``) from a judgment row.

    Uses ``.get()`` so old data without per-mode keys returns ``None``.
    """
    j = row.get("judgment") or {}
    return j.get(f"{mode}_decision")


def _per_voice_dim_scores(row: dict) -> dict[str, list[float]]:
    """Return {dim → [score across the 4 voices]} for one judgment row."""
    j = row.get("judgment") or {}
    out: dict[str, list[float]] = defaultdict(list)
    for voice in _FOUR_VOICES:
        v = j.get(voice) or {}
        scores = v.get("scores") or {}
        for dim, val in scores.items():
            if isinstance(val, (int, float)):
                out[dim].append(float(val))
    return out


def rank_generators(run_id: str, *, eval_dir: Path | str | None = None) -> list[dict]:
    """Per-generator rank table for a phase 3 generator-eval run.

    See module docstring for the returned dict shape.
    """
    run_dir = _resolve_run_dir(run_id, eval_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {run_dir}")
    meta = json.loads(meta_path.read_text())

    gold = meta.get("gold_judge") or {}
    gold_alias = gold.get("alias")
    gold_prompt_id = _meta_prompt_id(gold)
    if not gold_alias or not gold_prompt_id:
        raise ValueError(
            f"metadata.json in {run_id} is missing gold_judge.alias or prompts"
        )

    items_rows = _read_jsonl(run_dir / "items.jsonl")
    n_pool = len(items_rows) or meta.get("n_items") or 0

    gens_dir = run_dir / "generations"
    if not gens_dir.exists():
        return []

    out: list[dict] = []
    for gen_file in sorted(gens_dir.glob("*.jsonl")):
        gen_name = gen_file.stem  # "<alias>__<prompt>"
        # Find the matching gold judgment file
        jud_filename = f"{gold_alias}__{gold_prompt_id}__on__{gen_name}.jsonl"
        jud_file = run_dir / "judgments" / jud_filename
        if not jud_file.exists():
            logger.warning(
                "Skipping generator {} — no judgment file yet ({})",
                gen_name,
                jud_file.relative_to(run_dir),
            )
            continue

        judgment_rows = _read_jsonl(jud_file)
        n_succeeded = len(judgment_rows)

        aggregates: list[float] = []
        accepts = 0
        per_dim_sums: dict[str, list[float]] = defaultdict(list)
        accept_by_safety: dict[str, dict] = {}

        # Per-mode accumulators (use .get() to handle old data gracefully).
        refl_aggregates: list[float] = []
        refl_accepts = 0
        prefl_aggregates: list[float] = []
        prefl_accepts = 0

        for row in judgment_rows:
            agg = _judgment_aggregate(row)
            if agg is not None:
                aggregates.append(agg)
            decision = _judgment_decision(row)
            is_accept = decision == "accept"
            if is_accept:
                accepts += 1
            for dim, vals in _per_voice_dim_scores(row).items():
                per_dim_sums[dim].extend(vals)
            ss = row.get("safety_score")
            if ss is not None:
                key = str(ss)
                bucket = accept_by_safety.setdefault(key, {"n": 0, "accepts": 0})
                bucket["n"] += 1
                if is_accept:
                    bucket["accepts"] += 1

            # Per-mode metrics
            refl_agg = _judgment_mode_aggregate(row, "reflection")
            if refl_agg is not None:
                refl_aggregates.append(refl_agg)
            refl_dec = _judgment_mode_decision(row, "reflection")
            if refl_dec == "accept":
                refl_accepts += 1

            prefl_agg = _judgment_mode_aggregate(row, "preflection")
            if prefl_agg is not None:
                prefl_aggregates.append(prefl_agg)
            prefl_dec = _judgment_mode_decision(row, "preflection")
            if prefl_dec == "accept":
                prefl_accepts += 1

        for key, bucket in accept_by_safety.items():
            n = bucket["n"]
            bucket["accept_rate"] = bucket["accepts"] / n if n else 0.0
            del bucket["accepts"]

        per_dim_mean = {
            dim: sum(vals) / len(vals) if vals else 0.0
            for dim, vals in per_dim_sums.items()
        }

        # Failure rates
        gen_alias_prompt = gen_name  # e.g. "qwen3-9b__generator_v1.md"
        gen_failures = _failure_categories_distinct_items(
            run_dir / "failures" / f"gen_{gen_alias_prompt}.jsonl"
        )
        judge_failures = _failure_categories_distinct_items(
            run_dir
            / "failures"
            / f"jud_{gold_alias}__{gold_prompt_id}__on__{gen_alias_prompt}.jsonl"
        )

        denom = n_pool if n_pool > 0 else 1
        failure_rates = {
            "gen_api": len(gen_failures.get("api", set())) / denom,
            "gen_parse": len(gen_failures.get("parse", set())) / denom,
            "judge_api": len(judge_failures.get("api", set())) / denom,
            "judge_parse": len(judge_failures.get("parse", set())) / denom,
            "total_dropped": (n_pool - n_succeeded) / denom,
        }

        entry: dict = {
            "generator": gen_name,
            "n_pool": n_pool,
            "n_succeeded": n_succeeded,
            "mean_aggregate": (
                sum(aggregates) / len(aggregates) if aggregates else 0.0
            ),
            "accept_rate": accepts / n_succeeded if n_succeeded else 0.0,
            "per_dim_mean": per_dim_mean,
            "accept_by_safety_score": accept_by_safety,
            "failure_rates": failure_rates,
        }

        # Add per-mode metrics only when data is available (new judgment format).
        if refl_aggregates:
            entry["reflection_mean_aggregate"] = sum(refl_aggregates) / len(
                refl_aggregates
            )
        if n_succeeded:
            # Only include if at least one row had a per-mode decision.
            refl_dec_count = sum(
                1
                for r in judgment_rows
                if _judgment_mode_decision(r, "reflection") is not None
            )
            if refl_dec_count:
                entry["reflection_accept_rate"] = refl_accepts / refl_dec_count

        if prefl_aggregates:
            entry["preflection_mean_aggregate"] = sum(prefl_aggregates) / len(
                prefl_aggregates
            )
        if n_succeeded:
            prefl_dec_count = sum(
                1
                for r in judgment_rows
                if _judgment_mode_decision(r, "preflection") is not None
            )
            if prefl_dec_count:
                entry["preflection_accept_rate"] = prefl_accepts / prefl_dec_count

        out.append(entry)

    out.sort(key=lambda r: r["mean_aggregate"], reverse=True)
    return out


# ----------------------------------------------------------------- correlation helpers


def _safe_spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    try:
        from scipy.stats import spearmanr  # type: ignore

        rho, _ = spearmanr(xs, ys)
        if rho != rho:  # NaN
            return None
        return float(rho)
    except ImportError:
        return _spearman_fallback(xs, ys)


def _spearman_fallback(xs: list[float], ys: list[float]) -> float | None:
    """Pure-python Spearman, used when scipy is unavailable."""
    if len(xs) < 2:
        return None

    def _ranks(seq: list[float]) -> list[float]:
        sorted_idx = sorted(range(len(seq)), key=lambda i: seq[i])
        ranks = [0.0] * len(seq)
        i = 0
        while i < len(seq):
            j = i
            while j + 1 < len(seq) and seq[sorted_idx[j + 1]] == seq[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1.0
            for k in range(i, j + 1):
                ranks[sorted_idx[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    return _safe_pearson(rx, ry)


def _safe_pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / math.sqrt(sx * sy)


def _decision_concordance(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _cohens_kappa(a: list[str], b: list[str]) -> float | None:
    if not a:
        return None
    labels = list({*a, *b})
    if not labels:
        return None
    n = len(a)
    obs = sum(1 for x, y in zip(a, b) if x == y) / n
    # expected agreement
    exp = 0.0
    for label in labels:
        pa = a.count(label) / n
        pb = b.count(label) / n
        exp += pa * pb
    if exp >= 1.0:
        return 1.0 if obs == 1.0 else None
    return (obs - exp) / (1 - exp)


# ----------------------------------------------------------------- rank_judges


def _index_by(rows: list[dict], key) -> dict:
    if isinstance(key, str):
        return {r[key]: r for r in rows if key in r}
    return {tuple(r[k] for k in key): r for r in rows if all(k in r for k in key)}


def _judgment_per_dim_for_pairs(rows: list[dict]) -> dict[str, list[float]]:
    """Return per-dim score lists across rows (averaged across the 4 voices)."""
    out: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_dim = _per_voice_dim_scores(row)
        for dim, vals in per_dim.items():
            if vals:
                out[dim].append(sum(vals) / len(vals))
    return out


def _per_mode_correlation(
    cand_rows: list[dict],
    ref_rows_or_index: dict | list[dict],
    ref_agg_fn,
    ref_dec_fn,
    key="item_id",
) -> dict[str, dict]:
    """Compute per-mode (reflection/preflection) correlation metrics.

    Returns a dict like::

        {
            "reflection": {"spearman": ..., "kappa": ...},
            "preflection": {"spearman": ..., "kappa": ...},
        }

    Uses ``.get()`` everywhere; returns empty sub-dicts when per-mode data
    is not present (old format).
    """
    # Build index from ref rows if needed.
    if isinstance(ref_rows_or_index, list):
        if isinstance(key, tuple):
            ref_index = {
                tuple(r.get(k) for k in key): r
                for r in ref_rows_or_index
                if all(k in r for k in key)
            }
        else:
            ref_index = {r[key]: r for r in ref_rows_or_index if key in r}
    else:
        ref_index = ref_rows_or_index

    result: dict[str, dict] = {}
    for mode in ("reflection", "preflection"):
        xs: list[float] = []
        ys: list[float] = []
        decs_a: list[str] = []
        decs_b: list[str] = []
        for row in cand_rows:
            if isinstance(key, tuple):
                k = tuple(row.get(k_) for k_ in key)
            else:
                k = row.get(key)
            ref = ref_index.get(k)
            if ref is None:
                continue
            cand_agg = _judgment_mode_aggregate(row, mode)
            ref_agg = ref_agg_fn(ref, mode)
            if cand_agg is None or ref_agg is None:
                continue
            xs.append(cand_agg)
            ys.append(ref_agg)
            cand_dec = _judgment_mode_decision(row, mode)
            ref_dec = ref_dec_fn(ref, mode)
            if cand_dec is not None and ref_dec is not None:
                decs_a.append(cand_dec)
                decs_b.append(ref_dec)

        if xs:
            result[mode] = {
                "spearman": _safe_spearman(xs, ys),
                "kappa": _cohens_kappa(decs_a, decs_b),
            }
    return result


def _human_review_aggregate(row: dict) -> float | None:
    hr = row.get("human_review") or {}
    if "aggregate" in hr and isinstance(hr["aggregate"], (int, float)):
        return float(hr["aggregate"])
    scores = hr.get("scores") or {}
    flat: list[float] = []
    for voice in _FOUR_VOICES:
        v = scores.get(voice) or {}
        for val in v.values():
            if isinstance(val, (int, float)):
                flat.append(float(val))
    return sum(flat) / len(flat) if flat else None


def _human_review_decision(row: dict) -> str | None:
    hr = row.get("human_review") or {}
    return hr.get("decision")


def _human_review_per_dim(row: dict) -> dict[str, float]:
    hr = row.get("human_review") or {}
    scores = hr.get("scores") or {}
    out: dict[str, list[float]] = defaultdict(list)
    for voice in _FOUR_VOICES:
        v = scores.get(voice) or {}
        for dim, val in v.items():
            if isinstance(val, (int, float)):
                out[dim].append(float(val))
    return {dim: sum(v) / len(v) for dim, v in out.items() if v}


def rank_judges(run_id: str, *, eval_dir: Path | str | None = None) -> dict:
    """Per-judge rank tables for a phase 3 judge-eval run."""
    run_dir = _resolve_run_dir(run_id, eval_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {run_dir}")
    meta = json.loads(meta_path.read_text())

    gold = meta.get("gold_judge") or {}
    gold_alias = gold.get("alias")
    gold_pid = _meta_prompt_id(gold)
    gen = meta.get("generator") or {}
    gen_alias = gen.get("alias")
    gen_pid = _meta_prompt_id(gen)

    judg_dir = run_dir / "judgments"
    if not judg_dir.exists():
        return {"vs_gold": [], "vs_human": []}

    items_rows = _read_jsonl(run_dir / "items.jsonl")
    n_pool = len(items_rows) or meta.get("n_items") or 0
    reviewed_rows = _read_jsonl(run_dir / "reviewed_items.jsonl")
    n_reviewed = len(reviewed_rows)

    # Identify the gold judgment file (against the configured generator).
    gold_jud_file = None
    if gold_alias and gold_pid and gen_alias and gen_pid:
        gold_jud_file = (
            judg_dir / f"{gold_alias}__{gold_pid}__on__{gen_alias}__{gen_pid}.jsonl"
        )
    gold_rows: list[dict] = _read_jsonl(gold_jud_file) if gold_jud_file else []
    gold_index = _index_by(gold_rows, "item_id")

    vs_gold: list[dict] = []
    vs_human: list[dict] = []

    for jud_file in sorted(judg_dir.glob("*.jsonl")):
        name = jud_file.stem
        if "__on__reviewed" in name:
            # vs_human path
            judge_label = name.replace("__on__reviewed", "")
            cand_rows = _read_jsonl(jud_file)
            reviewed_index = _index_by(reviewed_rows, ("item_id", "iteration"))

            paired_xs: list[float] = []
            paired_ys: list[float] = []
            paired_decs_a: list[str] = []
            paired_decs_b: list[str] = []
            paired_per_dim_x: dict[str, list[float]] = defaultdict(list)
            paired_per_dim_y: dict[str, list[float]] = defaultdict(list)

            for row in cand_rows:
                key = (row.get("item_id"), row.get("iteration"))
                gold_pair = reviewed_index.get(key)
                if gold_pair is None:
                    continue
                cand_agg = _judgment_aggregate(row)
                hum_agg = _human_review_aggregate(gold_pair)
                if cand_agg is None or hum_agg is None:
                    continue
                paired_xs.append(cand_agg)
                paired_ys.append(hum_agg)
                cand_dec = _judgment_decision(row)
                hum_dec = _human_review_decision(gold_pair) or "?"
                if cand_dec is not None:
                    paired_decs_a.append(cand_dec)
                    paired_decs_b.append(hum_dec)
                cand_per_dim = _per_voice_dim_scores(row)
                hum_per_dim = _human_review_per_dim(gold_pair)
                for dim, vals in cand_per_dim.items():
                    if vals and dim in hum_per_dim:
                        paired_per_dim_x[dim].append(sum(vals) / len(vals))
                        paired_per_dim_y[dim].append(hum_per_dim[dim])

            failures = _failure_categories_distinct_items(
                run_dir / "failures" / f"jud_{judge_label}__on__reviewed.jsonl"
            )
            denom_h = n_reviewed if n_reviewed > 0 else 1
            failure_rates = {
                "api": len(failures.get("api", set())) / denom_h,
                "parse": len(failures.get("parse", set())) / denom_h,
                "total_dropped": (n_reviewed - len(cand_rows)) / denom_h,
            }
            per_dim_corr = {
                dim: _safe_spearman(paired_per_dim_x[dim], paired_per_dim_y[dim])
                for dim in paired_per_dim_x
            }

            # Per-mode correlation vs human reviews (new judgment format).
            # Human reviews don't have per-mode aggregates/decisions in the same
            # shape, so we only compute per-mode correlation when judgments have
            # the new per-mode keys.
            per_mode = _per_mode_correlation(
                cand_rows,
                reviewed_index,
                ref_agg_fn=lambda r, mode: _judgment_mode_aggregate(r, mode),
                ref_dec_fn=lambda r, mode: _judgment_mode_decision(r, mode),
                key=("item_id", "iteration"),
            )

            entry = {
                "judge": judge_label,
                "n_reviewed": n_reviewed,
                "n_succeeded": len(paired_xs),
                "spearman": _safe_spearman(paired_xs, paired_ys),
                "pearson": _safe_pearson(paired_xs, paired_ys),
                "concordance": _decision_concordance(paired_decs_a, paired_decs_b),
                "kappa": _cohens_kappa(paired_decs_a, paired_decs_b),
                "per_dim": per_dim_corr,
                "failure_rates": failure_rates,
            }
            if per_mode:
                entry["per_mode"] = per_mode
            vs_human.append(entry)
            continue

        # vs_gold path: file is "<judge_alias>__<judge_pid>__on__<gen>__<gen_pid>.jsonl"
        # Skip the gold judge correlated against itself.
        if gold_alias and gold_pid and gen_alias and gen_pid:
            if name == f"{gold_alias}__{gold_pid}__on__{gen_alias}__{gen_pid}":
                continue

        cand_rows = _read_jsonl(jud_file)
        # Pair against gold by item_id
        paired_xs = []
        paired_ys = []
        paired_decs_a = []
        paired_decs_b = []
        paired_per_dim_x = defaultdict(list)
        paired_per_dim_y = defaultdict(list)
        for row in cand_rows:
            iid = row.get("item_id")
            gold_pair = gold_index.get(iid)
            if gold_pair is None:
                continue
            cand_agg = _judgment_aggregate(row)
            gold_agg = _judgment_aggregate(gold_pair)
            if cand_agg is None or gold_agg is None:
                continue
            paired_xs.append(cand_agg)
            paired_ys.append(gold_agg)
            ca = _judgment_decision(row)
            ga = _judgment_decision(gold_pair)
            if ca and ga:
                paired_decs_a.append(ca)
                paired_decs_b.append(ga)
            cand_per_dim = _per_voice_dim_scores(row)
            gold_per_dim_d = _per_voice_dim_scores(gold_pair)
            for dim, vals in cand_per_dim.items():
                if vals and dim in gold_per_dim_d and gold_per_dim_d[dim]:
                    paired_per_dim_x[dim].append(sum(vals) / len(vals))
                    paired_per_dim_y[dim].append(
                        sum(gold_per_dim_d[dim]) / len(gold_per_dim_d[dim])
                    )

        # Identify the failures sidecar (path-form depends on the judge label)
        failures = _failure_categories_distinct_items(
            run_dir / "failures" / f"jud_{name}.jsonl"
        )
        denom = n_pool if n_pool > 0 else 1
        failure_rates = {
            "api": len(failures.get("api", set())) / denom,
            "parse": len(failures.get("parse", set())) / denom,
            "total_dropped": (n_pool - len(cand_rows)) / denom,
        }
        per_dim_corr = {
            dim: _safe_spearman(paired_per_dim_x[dim], paired_per_dim_y[dim])
            for dim in paired_per_dim_x
        }

        # Per-mode correlation (new judgment format).
        per_mode = _per_mode_correlation(
            cand_rows,
            gold_index,
            ref_agg_fn=_judgment_mode_aggregate,
            ref_dec_fn=_judgment_mode_decision,
            key="item_id",
        )

        entry = {
            "judge": name,
            "n_pool": n_pool,
            "n_succeeded": len(paired_xs),
            "spearman": _safe_spearman(paired_xs, paired_ys),
            "pearson": _safe_pearson(paired_xs, paired_ys),
            "concordance": _decision_concordance(paired_decs_a, paired_decs_b),
            "kappa": _cohens_kappa(paired_decs_a, paired_decs_b),
            "per_dim": per_dim_corr,
            "failure_rates": failure_rates,
        }
        if per_mode:
            entry["per_mode"] = per_mode
        vs_gold.append(entry)

    return {"vs_gold": vs_gold, "vs_human": vs_human}
