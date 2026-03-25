"""Phase 3: paired iteration and correlation analysis.

Runs the same items through gold models AND a target model, producing
separate iterations linked by group_id. Computes correlation metrics
between gold and target judgments to assess model alignment.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from uuid import uuid4

from scipy.stats import pearsonr, spearmanr

from pipeline.config import (
    CHARTER_PATH,
    WRITING_GUIDELINES_PATH,
    AppConfig,
    resolve_gold_generator,
    resolve_gold_judge,
    resolve_prompt_path,
    resolve_target_model,
)
from pipeline.log import logger
from pipeline.phase2.run import (
    _gather,
    _make_run_summary,
    generate_batch,
    health_check,
    judge_batch,
    make_api_client,
    select_items,
)
from pipeline.phase2.storage import (
    load_items_for_iteration,
    next_iteration,
    save_item,
    save_run,
)


def make_phase3_api_client(cfg: AppConfig):
    """Create an OpenAI client and semaphore using phase2 endpoint with phase3 concurrency."""
    return make_api_client(cfg.phase2.endpoint, cfg.phase3.iteration.max_concurrent)


def _run_one_pair_phase3(
    cfg: AppConfig,
    items: list[dict],
    gen_alias: str,
    gen_api_name: str,
    gen_prompt_path: Path,
    judge_alias: str,
    judge_api_name: str,
    judge_prompt_path: Path,
    source: str,
    group_id: str,
) -> dict:
    """Run generate->judge for one pair in phase3 context.

    Accepts model API names and prompt paths directly (no alias resolution),
    saves results with phase='phase3', and returns a run summary dict.
    """
    from pipeline.storage import _get_conn, checkpoint

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)

    def _graceful_shutdown(signum, frame):
        logger.warning(
            "Received signal {} during phase3 iteration — checkpointing DB before exit",
            signum,
        )
        try:
            _get_conn().commit()
            checkpoint()
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _graceful_shutdown)
    signal.signal(signal.SIGINT, _graceful_shutdown)

    try:
        return _run_one_pair_phase3_inner(
            cfg,
            items,
            gen_alias,
            gen_api_name,
            gen_prompt_path,
            judge_alias,
            judge_api_name,
            judge_prompt_path,
            source,
            group_id,
        )
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def _run_one_pair_phase3_inner(
    cfg: AppConfig,
    items: list[dict],
    gen_alias: str,
    gen_api_name: str,
    gen_prompt_path: Path,
    judge_alias: str,
    judge_api_name: str,
    judge_prompt_path: Path,
    source: str,
    group_id: str,
) -> dict:
    """Inner implementation of _run_one_pair_phase3 (split out for signal safety)."""
    iteration = next_iteration()
    client, semaphore = make_phase3_api_client(cfg)
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

    logger.info(
        "Phase3 iteration {} — gen={} judge={}", iteration, gen_alias, judge_alias
    )

    generated = generate_batch(
        items,
        gen_prompt_path,
        charter_text,
        gen_api_name,
        iteration,
        client,
        semaphore,
        writing_guidelines_text=writing_guidelines_text,
    )

    judged = judge_batch(
        generated,
        judge_prompt_path,
        judge_api_name,
        iteration,
        cfg.phase3.scoring.accept_threshold,
        client,
        semaphore,
        floor_threshold=cfg.phase3.scoring.floor_threshold,
        charter_text=charter_text,
        writing_guidelines_text=writing_guidelines_text,
    )

    summary = _make_run_summary(iteration, judged)
    logger.info(summary)

    n_accepted = sum(1 for item in judged if item["judgment"]["decision"] == "accept")
    scores = [item["judgment"]["aggregate"] for item in judged]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    save_run(
        iteration=iteration,
        gen_prompt=gen_prompt_path.name,
        judge_prompt=judge_prompt_path.name,
        generator_model=gen_alias,
        judge_model=judge_alias,
        n_items=len(judged),
        n_gold=sum(1 for item in judged if item.get("is_gold")),
        config={
            "accept_threshold": cfg.phase3.scoring.accept_threshold,
            "max_concurrent": cfg.phase3.iteration.max_concurrent,
        },
        analysis=summary,
        source=source,
        group_id=group_id,
        phase="phase3",
    )

    return {
        "iteration": iteration,
        "n_items": len(judged),
        "n_accepted": n_accepted,
        "n_rejected": len(judged) - n_accepted,
        "mean_score": mean_score,
        "items": judged,
        "generator_model": gen_alias,
        "judge_model": judge_alias,
        "group_id": group_id,
    }


def run_paired_iteration(
    cfg: AppConfig,
    role: str,
    target_alias: str,
    source: str,
) -> list[dict]:
    """Run paired iteration: same items through gold models AND the target model.

    For judge role: generate with each gold generator, then judge those items
    with BOTH each gold judge and the target judge.

    For generator role: generate with each gold generator AND the target generator,
    then judge with each gold judge.

    All iterations in a batch share a group_id for downstream correlation.
    Returns list of run summary dicts.
    """
    assert role in ("judge", "generator"), f"Invalid role: {role}"

    seed = 42 + next_iteration()
    items = select_items(
        cfg.phase3.iteration.n_items,
        cfg.phase3.iteration.n_gold,
        seed,
        cfg.max_tokens,
    )

    group_id = str(uuid4())
    summaries = []

    if role == "judge":
        target_cfg = resolve_target_model(cfg, target_alias)
        target_judge_prompt = resolve_prompt_path("judge_latest.md", alias=target_alias)

        for gold_gen_cfg in cfg.phase3.gold_generators:
            gen_prompt = resolve_prompt_path(
                "generator_latest.md", alias=gold_gen_cfg.alias
            )

            generated_result = None

            for gold_judge_cfg in cfg.phase3.gold_judges:
                judge_prompt = resolve_prompt_path(
                    "judge_latest.md", alias=gold_judge_cfg.alias
                )
                result = _run_one_pair_phase3(
                    cfg,
                    items,
                    gen_alias=gold_gen_cfg.alias,
                    gen_api_name=gold_gen_cfg.api_name,
                    gen_prompt_path=gen_prompt,
                    judge_alias=gold_judge_cfg.alias,
                    judge_api_name=gold_judge_cfg.api_name,
                    judge_prompt_path=judge_prompt,
                    source=source,
                    group_id=group_id,
                )
                summaries.append(result)

            result = _run_one_pair_phase3(
                cfg,
                items,
                gen_alias=gold_gen_cfg.alias,
                gen_api_name=gold_gen_cfg.api_name,
                gen_prompt_path=gen_prompt,
                judge_alias=target_alias,
                judge_api_name=target_cfg.api_name,
                judge_prompt_path=target_judge_prompt,
                source=source,
                group_id=group_id,
            )
            summaries.append(result)

    else:
        target_cfg = resolve_target_model(cfg, target_alias)
        target_gen_prompt = resolve_prompt_path(
            "generator_latest.md", alias=target_alias
        )

        for gold_judge_cfg in cfg.phase3.gold_judges:
            judge_prompt = resolve_prompt_path(
                "judge_latest.md", alias=gold_judge_cfg.alias
            )

            for gold_gen_cfg in cfg.phase3.gold_generators:
                gen_prompt = resolve_prompt_path(
                    "generator_latest.md", alias=gold_gen_cfg.alias
                )
                result = _run_one_pair_phase3(
                    cfg,
                    items,
                    gen_alias=gold_gen_cfg.alias,
                    gen_api_name=gold_gen_cfg.api_name,
                    gen_prompt_path=gen_prompt,
                    judge_alias=gold_judge_cfg.alias,
                    judge_api_name=gold_judge_cfg.api_name,
                    judge_prompt_path=judge_prompt,
                    source=source,
                    group_id=group_id,
                )
                summaries.append(result)

            result = _run_one_pair_phase3(
                cfg,
                items,
                gen_alias=target_alias,
                gen_api_name=target_cfg.api_name,
                gen_prompt_path=target_gen_prompt,
                judge_alias=gold_judge_cfg.alias,
                judge_api_name=gold_judge_cfg.api_name,
                judge_prompt_path=judge_prompt,
                source=source,
                group_id=group_id,
            )
            summaries.append(result)

    return summaries


def compute_paired_correlation(
    gold_items: list[dict],
    target_items: list[dict],
) -> dict:
    """Compute correlation metrics between gold and target judged items.

    Matches items by item_id and compares judgment scores across all
    dimensions. Returns decision concordance, rank correlations, and
    per-dimension breakdowns.
    """
    gold_by_id = {item["item_id"]: item for item in gold_items}
    target_by_id = {item["item_id"]: item for item in target_items}

    shared_ids = sorted(set(gold_by_id) & set(target_by_id))
    assert len(shared_ids) > 0, "No shared item_ids between gold and target"

    gold_decisions = []
    target_decisions = []
    gold_aggregates = []
    target_aggregates = []
    per_dim_scores: dict[str, tuple[list[float], list[float]]] = {}

    for item_id in shared_ids:
        g = gold_by_id[item_id]
        t = target_by_id[item_id]
        assert g["judgment"] is not None, f"Gold item {item_id} has no judgment"
        assert t["judgment"] is not None, f"Target item {item_id} has no judgment"

        gold_decisions.append(g["judgment"]["decision"])
        target_decisions.append(t["judgment"]["decision"])
        gold_aggregates.append(g["judgment"]["aggregate"])
        target_aggregates.append(t["judgment"]["aggregate"])

        for part in ("preflection", "reflection"):
            g_scores = g["judgment"][part]["scores"]
            t_scores = t["judgment"][part]["scores"]
            all_dims = sorted(set(g_scores) | set(t_scores))
            for dim in all_dims:
                key = f"{part}_{dim}"
                if key not in per_dim_scores:
                    per_dim_scores[key] = ([], [])
                if dim in g_scores and dim in t_scores:
                    per_dim_scores[key][0].append(g_scores[dim])
                    per_dim_scores[key][1].append(t_scores[dim])

    decision_concordance = sum(
        1 for g, t in zip(gold_decisions, target_decisions) if g == t
    ) / len(shared_ids)

    aggregate_spearman = _safe_spearman(gold_aggregates, target_aggregates)
    aggregate_pearson = _safe_pearson(gold_aggregates, target_aggregates)

    per_dimension = {}
    for key, (g_vals, t_vals) in per_dim_scores.items():
        diffs = [g - t for g, t in zip(g_vals, t_vals)]
        per_dimension[key] = {
            "spearman": _safe_spearman(g_vals, t_vals),
            "pearson": _safe_pearson(g_vals, t_vals),
            "mean_diff": sum(diffs) / len(diffs),
            "mean_abs_diff": sum(abs(d) for d in diffs) / len(diffs),
        }

    cohens_kappa = _compute_cohens_kappa(gold_decisions, target_decisions)

    return {
        "decision_concordance": decision_concordance,
        "aggregate_spearman": aggregate_spearman,
        "aggregate_pearson": aggregate_pearson,
        "per_dimension": per_dimension,
        "n_items": len(shared_ids),
        "cohens_kappa": cohens_kappa,
    }


def _safe_spearman(x: list[float], y: list[float]) -> float:
    """Spearman rho, returning NaN if undefined (e.g. constant input)."""
    assert len(x) == len(y)
    if len(x) < 2:
        return float("nan")
    if all(v == x[0] for v in x) or all(v == y[0] for v in y):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


def _safe_pearson(x: list[float], y: list[float]) -> float:
    """Pearson r, returning NaN if undefined (e.g. constant input)."""
    assert len(x) == len(y)
    if len(x) < 2:
        return float("nan")
    if all(v == x[0] for v in x) or all(v == y[0] for v in y):
        return float("nan")
    r, _ = pearsonr(x, y)
    return float(r)


def _compute_cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for inter-rater reliability on categorical decisions.

    Returns NaN if expected agreement is 1.0 (all items same category).
    """
    assert len(a) == len(b)
    n = len(a)
    assert n > 0

    categories = sorted(set(a) | set(b))
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n

    p_e = 0.0
    for cat in categories:
        p_a = sum(1 for x in a if x == cat) / n
        p_b = sum(1 for x in b if x == cat) / n
        p_e += p_a * p_b

    if p_e == 1.0:
        return float("nan")

    return (p_o - p_e) / (1.0 - p_e)


def detect_gold_disagreements(items_by_gold_model: dict[str, list[dict]]) -> list[dict]:
    """Find items where gold models disagree on accept/reject decision.

    Takes a dict of {gold_alias: [judged_items]} and returns a list of
    disagreement records with item_id, decisions per model, and scores.
    """
    assert (
        len(items_by_gold_model) >= 2
    ), "Need at least 2 gold models to detect disagreements"

    by_item: dict[str, dict[str, dict]] = {}
    for alias, items in items_by_gold_model.items():
        for item in items:
            assert (
                item["judgment"] is not None
            ), f"Item {item['item_id']} from {alias} has no judgment"
            by_item.setdefault(item["item_id"], {})[alias] = item

    disagreements = []
    for item_id, model_items in sorted(by_item.items()):
        if len(model_items) < 2:
            continue
        decisions = {
            alias: item["judgment"]["decision"] for alias, item in model_items.items()
        }
        if len(set(decisions.values())) > 1:
            disagreements.append(
                {
                    "item_id": item_id,
                    "decisions": decisions,
                    "scores": {
                        alias: item["judgment"]["aggregate"]
                        for alias, item in model_items.items()
                    },
                }
            )

    return disagreements
