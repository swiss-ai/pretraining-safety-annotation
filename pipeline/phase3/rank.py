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


def rank_generators(
    run_id: str, *, eval_dir: Path | str | None = None
) -> list[dict]:
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
    gold_prompt = gold.get("prompt")
    if not gold_alias or not gold_prompt:
        raise ValueError(
            f"metadata.json in {run_id} is missing gold_judge.alias or .prompt"
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
        jud_filename = f"{gold_alias}__{gold_prompt}__on__{gen_name}.jsonl"
        jud_file = run_dir / "judgments" / jud_filename
        if not jud_file.exists():
            raise FileNotFoundError(
                f"Missing gold judgment file for generator {gen_name}: "
                f"expected {jud_file.relative_to(run_dir)}"
            )

        judgment_rows = _read_jsonl(jud_file)
        n_succeeded = len(judgment_rows)

        aggregates: list[float] = []
        accepts = 0
        per_dim_sums: dict[str, list[float]] = defaultdict(list)
        accept_by_safety: dict[str, dict] = {}
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
            / f"jud_{gold_alias}__{gold_prompt}__on__{gen_alias_prompt}.jsonl"
        )

        denom = n_pool if n_pool > 0 else 1
        failure_rates = {
            "gen_api": len(gen_failures.get("api", set())) / denom,
            "gen_parse": len(gen_failures.get("parse", set())) / denom,
            "judge_api": len(judge_failures.get("api", set())) / denom,
            "judge_parse": len(judge_failures.get("parse", set())) / denom,
            "total_dropped": (n_pool - n_succeeded) / denom,
        }

        out.append(
            {
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
        )

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


def rank_judges(
    run_id: str, *, eval_dir: Path | str | None = None
) -> dict:
    """Per-judge rank tables for a phase 3 judge-eval run."""
    run_dir = _resolve_run_dir(run_id, eval_dir)
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json in {run_dir}")
    meta = json.loads(meta_path.read_text())

    gold = meta.get("gold_judge") or {}
    gold_alias = gold.get("alias")
    gold_prompt = gold.get("prompt")
    gen = meta.get("generator") or {}
    gen_alias = gen.get("alias")
    gen_prompt = gen.get("prompt")

    judg_dir = run_dir / "judgments"
    if not judg_dir.exists():
        return {"vs_gold": [], "vs_human": []}

    items_rows = _read_jsonl(run_dir / "items.jsonl")
    n_pool = len(items_rows) or meta.get("n_items") or 0
    reviewed_rows = _read_jsonl(run_dir / "reviewed_items.jsonl")
    n_reviewed = len(reviewed_rows)

    # Identify the gold judgment file (against the configured generator).
    gold_jud_file = None
    if gold_alias and gold_prompt and gen_alias and gen_prompt:
        gold_jud_file = (
            judg_dir
            / f"{gold_alias}__{gold_prompt}__on__{gen_alias}__{gen_prompt}.jsonl"
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
            vs_human.append(
                {
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
            )
            continue

        # vs_gold path: file is "<judge_alias>__<judge_prompt>__on__<gen>__<gen_prompt>.jsonl"
        # Skip the gold judge correlated against itself.
        if gold_alias and gold_prompt and gen_alias and gen_prompt:
            if name == f"{gold_alias}__{gold_prompt}__on__{gen_alias}__{gen_prompt}":
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
        vs_gold.append(
            {
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
        )

    return {"vs_gold": vs_gold, "vs_human": vs_human}
