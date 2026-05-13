"""Phase 3 judge-eval runner.

Picks the configured generator (one), generates items once, then judges
those generations with each candidate judge AND the gold judge. Optionally
also runs every judge over the human-reviewed item set for vs-human metrics.
"""

from __future__ import annotations

from pipeline.api import make_api_client
from pipeline.config import (
    CHARTER_PATH,
    WRITING_GUIDELINES_PATH,
    AppConfig,
    CandidateModel,
    resolve_prompt_path,
)
from pipeline.log import logger
from pipeline.charter.improve.run import generate_batch, judge_batch
from pipeline.charter.eval.eval_generators import (
    _candidate_metadata,
    _eval_root,
    _failures_done_keys,
    _gen_failures_name,
    _gen_file,
    _generate_with_resume,
    _judg_file,
    _judge_failures_name,
    _judge_with_resume,
    _now_iso,
    _open_and_stamp,
    _prompt_id,
)
from pipeline.charter.eval.items import ensure_item_pool, load_reviewed_items
from pipeline.charter.eval.storage import JsonlRunStore


def _local_resolve_prompt_path(filename, alias):
    """Indirection so monkeypatching `eval_judges.resolve_prompt_path` is observed."""
    return resolve_prompt_path(filename, alias)


def _local_generate_batch(*args, **kwargs):
    return generate_batch(*args, **kwargs)


def _local_judge_batch(*args, **kwargs):
    return judge_batch(*args, **kwargs)


def _judge_reviewed_file(judge: CandidateModel) -> str:
    return f"judgments/{judge.alias}__{_prompt_id(judge)}__on__reviewed.jsonl"


def _judge_reviewed_failures_name(judge: CandidateModel) -> str:
    return f"jud_{judge.alias}__{_prompt_id(judge)}__on__reviewed"


def _dedup_judges(
    gold: CandidateModel, candidates: list[CandidateModel]
) -> list[CandidateModel]:
    """Return [gold] + every candidate not equal to gold by (alias, prompts)."""
    seen = {(gold.alias, gold.prompt_reflection, gold.prompt_preflection)}
    out: list[CandidateModel] = [gold]
    for c in candidates:
        key = (c.alias, c.prompt_reflection, c.prompt_preflection)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _ensure_reviewed_items_jsonl(
    store: JsonlRunStore, reviewer_policy: str
) -> list[dict]:
    """Materialize reviewed_items.jsonl exactly once and return its rows.

    Looks up `load_reviewed_items` from this module's globals at call time so
    that test monkeypatches on `pipeline.charter.eval.eval_judges.load_reviewed_items`
    are observed.
    """
    rel = "reviewed_items.jsonl"
    existing = store.read_all(rel)
    if existing:
        return existing
    # Re-resolve via this module's globals so monkeypatches take effect.
    import pipeline.charter.eval.eval_judges as _self_mod

    rows = _self_mod.load_reviewed_items(reviewer_policy=reviewer_policy)
    for r in rows:
        store.append(rel, r)
    store.flush()
    return rows


def run_judge_eval(cfg: AppConfig, run_id: str) -> None:
    """Path-B runner: rank candidate judges vs the gold judge and human reviews."""
    je = cfg.charter.eval.judge_eval
    root = _eval_root(cfg)
    store = JsonlRunStore(root, run_id)

    judges = _dedup_judges(cfg.charter.eval.gold_judge, je.candidates)
    expected = {
        "type": "judge_eval",
        "n_items": je.n_items,
        "seed": je.seed,
        "max_tokens": cfg.max_tokens,
        "include_reviewed": je.include_reviewed,
        "reviewer_policy": je.reviewer_policy,
        "store_reasoning": je.store_reasoning,
        "gold_judge": _candidate_metadata(cfg.charter.eval.gold_judge),
        "generator": _candidate_metadata(je.generator),
        "candidates": [_candidate_metadata(c) for c in judges],
    }

    try:
        _open_and_stamp(store, root, run_id, "judge_eval", expected)

        items = ensure_item_pool(store, je.n_items, je.seed, cfg.max_tokens)

        # Per-model endpoint: fall back to the phase-level default.
        def _client_for(model):
            ep = model.endpoint or cfg.charter.eval.endpoint
            return make_api_client(ep, je.max_concurrent)

        charter = CHARTER_PATH.read_text(encoding="utf-8")
        wg = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")

        gen = je.generator

        # Step 1: generate once with the configured generator
        gen_client, _ = _client_for(gen)
        _generate_with_resume(
            store,
            _gen_file(gen),
            items,
            gen,
            cfg,
            gen_client,
            je.max_concurrent,
            charter,
            wg,
            failures_name=_gen_failures_name(gen),
            canary_rng_seed=je.seed,
            failure_attempt_cap=je.failure_attempt_cap,
            store_reasoning=je.store_reasoning,
            generate_batch_fn=_local_generate_batch,
            resolve_prompt_path_fn=_local_resolve_prompt_path,
        )

        # Ensure generation data is on disk before judges read it
        store.flush(fsync=True)

        # Step 2: every judge scores the generations
        for jud in judges:
            jud_client, _ = _client_for(jud)
            _judge_with_resume(
                store,
                _judg_file(jud, gen),
                store.iter_rows(_gen_file(gen)),
                jud,
                cfg,
                jud_client,
                je.max_concurrent,
                charter,
                wg,
                failures_name=_judge_failures_name(jud, gen),
                accept_threshold=cfg.charter.eval.scoring.accept_threshold,
                failure_attempt_cap=je.failure_attempt_cap,
                store_reasoning=je.store_reasoning,
                judge_batch_fn=_local_judge_batch,
                resolve_prompt_path_fn=_local_resolve_prompt_path,
            )

        # Step 3: optional reviewed-items path (vs-human signal)
        if je.include_reviewed:
            reviewed = _ensure_reviewed_items_jsonl(store, je.reviewer_policy)
            logger.info("phase3 judge-eval: reviewed pool has {} items", len(reviewed))
            for jud in judges:
                jud_client, _ = _client_for(jud)
                _judge_with_resume(
                    store,
                    _judge_reviewed_file(jud),
                    store.iter_rows("reviewed_items.jsonl"),
                    jud,
                    cfg,
                    jud_client,
                    je.max_concurrent,
                    charter,
                    wg,
                    failures_name=_judge_reviewed_failures_name(jud),
                    accept_threshold=cfg.charter.eval.scoring.accept_threshold,
                    failure_attempt_cap=je.failure_attempt_cap,
                    store_reasoning=je.store_reasoning,
                    resume_key=("item_id", "iteration"),
                    judge_batch_fn=_local_judge_batch,
                    resolve_prompt_path_fn=_local_resolve_prompt_path,
                )

        meta = store.read_metadata()
        meta["status"] = "done"
        meta["finished_at"] = _now_iso()
        store.write_metadata(meta)
    finally:
        store.close()
