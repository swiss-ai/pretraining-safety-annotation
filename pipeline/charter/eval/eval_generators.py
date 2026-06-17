"""Charter eval generator-eval runner.

Reads `cfg.charter.eval.generator_eval`, generates reflections from each
candidate generator on the same item pool, then judges them with the
single configured gold judge. Per-item resume via `JsonlRunStore`.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import datetime
import hashlib
import os
from pathlib import Path
from typing import Iterable

from pipeline.api import make_api_client
from pipeline.config import (
    CHARTER_PATH,
    AppConfig,
    CandidateModel,
    resolve_prompt_path,
)
from pipeline.log import logger
from pipeline.charter.improve.run import generate_batch, judge_batch
from pipeline.charter.eval.items import ensure_item_pool
from pipeline.charter.eval.storage import JsonlRunStore


def _prompt_id(c: CandidateModel) -> str:
    """Identifier string for file naming, based on the reflection prompt."""
    return c.prompt_reflection


def _resolve_prompt_path(
    candidate: CandidateModel,
    resolve_fn=None,
) -> Path | None:
    """Resolve the reflection prompt path from a CandidateModel."""
    if resolve_fn is None:
        resolve_fn = resolve_prompt_path

    if candidate.prompt_reflection:
        return resolve_fn(candidate.prompt_reflection, candidate.alias)
    return None


def _eval_root(cfg: AppConfig) -> Path:
    raw = cfg.charter.eval.eval_dir or os.environ.get(
        "CHARTER_EVAL_DIR", "data/pipeline/charter_eval"
    )
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _prompt_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_metadata(c: CandidateModel) -> dict:
    meta: dict = {
        "alias": c.alias,
        "prompt_reflection": c.prompt_reflection,
    }
    for key, filename in (
        ("prompt_reflection_sha256", c.prompt_reflection),
    ):
        if not filename:
            meta[key] = ""
            continue
        try:
            meta[key] = _prompt_sha256(resolve_prompt_path(filename, c.alias))
        except Exception as e:
            logger.warning(
                "candidate metadata: could not hash {} for {}: {}",
                filename,
                c.alias,
                e,
            )
            meta[key] = ""
    return meta


def _gen_file(c: CandidateModel) -> str:
    return f"generations/{c.alias}__{_prompt_id(c)}.jsonl"


def _judg_file(judge: CandidateModel, gen: CandidateModel) -> str:
    return (
        f"judgments/{judge.alias}__{_prompt_id(judge)}"
        f"__on__{gen.alias}__{_prompt_id(gen)}.jsonl"
    )


def _gen_failures_name(c: CandidateModel) -> str:
    return f"gen_{c.alias}__{_prompt_id(c)}"


def _judge_failures_name(judge: CandidateModel, gen: CandidateModel) -> str:
    return (
        f"jud_{judge.alias}__{_prompt_id(judge)}"
        f"__on__{gen.alias}__{_prompt_id(gen)}"
    )


def _failures_done_keys(
    store: JsonlRunStore, failures_name: str, attempt_cap: int
) -> set[str]:
    """Item ids whose attempts have hit `attempt_cap`."""
    rel = f"failures/{failures_name}.jsonl"
    counts: dict[str, int] = {}
    for row in store.iter_rows(rel):
        if not row:
            continue
        item_id = row.get("item_id")
        if not item_id:
            continue
        counts[item_id] = counts.get(item_id, 0) + 1
    # Hydrate the in-memory mirror inside the store so subsequent
    # record_failure() calls keep counting from the persisted state.
    for item_id, n in counts.items():
        store.set_failure_count(failures_name, item_id, n)
    return {iid for iid, n in counts.items() if n >= attempt_cap}


def _make_on_failure(store: JsonlRunStore, failures_name: str):
    """Build an on_failure callback that records the full info dict."""

    def _cb(info: dict) -> None:
        rel = f"failures/{failures_name}.jsonl"
        item_id = info.get("item_id", "?")
        attempts = store.get_failure_count(failures_name, item_id) + 1
        store.set_failure_count(failures_name, item_id, attempts)
        record = {
            "item_id": item_id,
            "stage": info.get("stage"),
            "category": info.get("category"),
            "reason": info.get("reason"),
            "raw": info.get("raw"),
            "raw_reasoning": info.get("raw_reasoning"),
            "attempt": attempts,
            "ts": _now_iso(),
        }
        store.append(rel, record)

    return _cb


def _generate_with_resume(
    store: JsonlRunStore,
    rel_path: str,
    items: list[dict],
    candidate: CandidateModel,
    cfg: AppConfig,
    client,
    max_concurrent: int,
    charter_text: str,
    *,
    failures_name: str,
    failure_attempt_cap: int,
    store_reasoning: bool,
    generate_batch_fn=None,
    resolve_prompt_path_fn=None,
) -> None:
    if generate_batch_fn is None:
        generate_batch_fn = generate_batch
    if resolve_prompt_path_fn is None:
        resolve_prompt_path_fn = resolve_prompt_path

    done = store.done_keys(rel_path)
    capped = _failures_done_keys(store, failures_name, failure_attempt_cap)
    todo = [
        it for it in items if it["item_id"] not in done and it["item_id"] not in capped
    ]

    refl_path = _resolve_prompt_path(
        candidate,
        resolve_fn=resolve_prompt_path_fn,
    )
    on_failure = _make_on_failure(store, failures_name)
    saved = 0

    def _on_result(row: dict) -> None:
        nonlocal saved
        if not store_reasoning:
            row = {
                k: v for k, v in row.items() if k not in ("raw_response", "reasoning")
            }
        store.append(rel_path, row)
        saved += 1

    semaphore = asyncio.Semaphore(max_concurrent)
    generate_batch_fn(
        todo,
        refl_path,
        charter_text,
        candidate.api_name,
        iteration=0,
        client=client,
        semaphore=semaphore,
        save=False,
        thinking=candidate.thinking,
        json_mode=candidate.json_mode,
        completion_max_tokens=candidate.completion_max_tokens,
        context_window_tokens=candidate.context_window_tokens,
        on_failure=on_failure,
        on_result=_on_result,
        desc=f"Generating [{candidate.alias}]",
    )
    logger.info(
        "charter.eval gen {}: {} new",
        candidate.alias,
        saved,
    )


def _judge_with_resume(
    store: JsonlRunStore,
    rel_path: str,
    item_iter: Iterable[dict],
    judge: CandidateModel,
    cfg: AppConfig,
    client,
    max_concurrent: int,
    charter_text: str,
    *,
    failures_name: str,
    accept_threshold: float,
    failure_attempt_cap: int,
    store_reasoning: bool,
    resume_key: str | tuple[str, ...] = "item_id",
    judge_batch_fn=None,
    resolve_prompt_path_fn=None,
) -> None:
    if judge_batch_fn is None:
        judge_batch_fn = judge_batch
    if resolve_prompt_path_fn is None:
        resolve_prompt_path_fn = resolve_prompt_path

    done = store.done_keys(rel_path, key=resume_key)
    capped = _failures_done_keys(store, failures_name, failure_attempt_cap)

    def _key_for(it):
        if isinstance(resume_key, str):
            return it[resume_key]
        return tuple(it[k] for k in resume_key)

    todo: list[dict] = []
    for it in item_iter:
        if not it:
            continue
        if _key_for(it) in done:
            continue
        if it["item_id"] in capped:
            continue
        todo.append(it)

    refl_path = _resolve_prompt_path(
        judge,
        resolve_fn=resolve_prompt_path_fn,
    )
    on_failure = _make_on_failure(store, failures_name)
    saved = 0

    def _on_result(row: dict) -> None:
        nonlocal saved
        if not store_reasoning:
            row = _strip_reasoning(row)
        store.append(rel_path, row)
        saved += 1

    semaphore = asyncio.Semaphore(max_concurrent)
    judge_batch_fn(
        todo,
        refl_path,
        judge.api_name,
        iteration=0,
        accept_threshold=accept_threshold,
        client=client,
        semaphore=semaphore,
        save=False,
        charter_text=charter_text,
        thinking=judge.thinking,
        completion_max_tokens=judge.completion_max_tokens,
        context_window_tokens=judge.context_window_tokens,
        on_failure=on_failure,
        on_result=_on_result,
        desc=f"Judging [{judge.alias}]",
    )
    logger.info(
        "charter.eval judge {}: {} new",
        judge.alias,
        saved,
    )


def _strip_reasoning(row: dict) -> dict:
    """Drop heavy reasoning fields from a judgment row."""
    out = dict(row)
    out.pop("raw_response", None)
    out.pop("reasoning", None)
    judgment = out.get("judgment")
    if isinstance(judgment, dict):
        j = dict(judgment)
        j.pop("raw_responses", None)
        for voice_key, voice_val in list(j.items()):
            if isinstance(voice_val, dict) and "model_reasoning" in voice_val:
                v = dict(voice_val)
                v.pop("model_reasoning", None)
                j[voice_key] = v
        out["judgment"] = j
    return out


def _open_and_stamp(
    store: JsonlRunStore, root: Path, run_id: str, run_type: str, expected: dict
) -> None:
    """Validate / create the run dir and stamp metadata as 'running'.

    Shared by both eval runners to avoid copy-pasting the open → validate
    → stamp sequence.
    """
    run_dir = root / run_id
    if run_dir.exists() and (run_dir / "metadata.json").exists():
        store.open(create=False, expected_metadata=expected)
        existing_meta = store.read_metadata()
        if existing_meta.get("type") not in (None, run_type):
            raise ValueError(
                f"Run {run_id} is type {existing_meta.get('type')!r}, "
                f"cannot reuse for {run_type}"
            )
    else:
        store.open(create=True, expected_metadata=expected)

    meta = store.read_metadata()
    meta.update(expected)
    meta.setdefault("started_at", _now_iso())
    meta["status"] = "running"
    store.write_metadata(meta)


def run_generator_eval(
    cfg: AppConfig, run_id: str, *, stage: str | None = None
) -> None:
    """Path-A runner: rank candidate generators against the single gold judge.

    *stage* controls which part to run:
      - ``None``       – run both stages (default, backwards-compatible)
      - ``"generate"`` – only generate reflections for every candidate
      - ``"judge"``    – only judge existing generations with the gold judge
    """
    ge = cfg.charter.eval.generator_eval
    root = _eval_root(cfg)
    store = JsonlRunStore(root, run_id)

    gold = cfg.charter.eval.gold_judge
    if ge.gold_prompt_reflection:
        gold = copy.copy(gold)
        gold.prompt_reflection = ge.gold_prompt_reflection

    expected = {
        "type": "generator_eval",
        "bench": ge.bench,
        "n_items": ge.n_items,
        "seed": ge.seed,
        "max_tokens": cfg.max_tokens,
        "gold_judge": _candidate_metadata(gold),
        "candidates": [_candidate_metadata(c) for c in ge.candidates],
    }

    run_generate = stage in (None, "generate")
    run_judge = stage in (None, "judge")

    try:
        _open_and_stamp(store, root, run_id, "generator_eval", expected)

        items = ensure_item_pool(store, ge.n_items, ge.seed, cfg.max_tokens, bench=ge.bench)

        judge_endpoint = gold.endpoint or cfg.charter.eval.endpoint
        logger.info("Gold judge: alias={} api_name={} endpoint={}", gold.alias, gold.api_name, judge_endpoint)
        client, sem = make_api_client(judge_endpoint, ge.max_concurrent)
        charter = CHARTER_PATH.read_text(encoding="utf-8")

        # Stage 1: generate reflections for every candidate (in parallel)
        if run_generate:
            n_cands = len(ge.candidates)
            per_cand = max(1, ge.max_concurrent // n_cands)

            def _gen_one(gen):
                # Each thread needs its own client because httpx
                # internals bind to a single event loop.
                gen_endpoint = gen.endpoint or cfg.charter.eval.endpoint
                t_client, _ = make_api_client(gen_endpoint, per_cand)
                _generate_with_resume(
                    store,
                    _gen_file(gen),
                    items,
                    gen,
                    cfg,
                    t_client,
                    per_cand,
                    charter,
                    failures_name=_gen_failures_name(gen),
                    failure_attempt_cap=ge.failure_attempt_cap,
                    store_reasoning=ge.store_reasoning,
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=n_cands) as pool:
                futs = [pool.submit(_gen_one, gen) for gen in ge.candidates]
                for fut in concurrent.futures.as_completed(futs):
                    fut.result()  # re-raise exceptions

            # Ensure all generation data is on disk before judging reads it
            store.flush(fsync=True)

        # Stage 2: judge all generations with the gold judge
        if run_judge:
            for gen in ge.candidates:
                _judge_with_resume(
                    store,
                    _judg_file(gold, gen),
                    store.iter_rows(_gen_file(gen)),
                    gold,
                    cfg,
                    client,
                    ge.max_concurrent,
                    charter,
                    failures_name=_judge_failures_name(gold, gen),
                    accept_threshold=cfg.charter.eval.scoring.accept_threshold,
                    failure_attempt_cap=ge.failure_attempt_cap,
                    store_reasoning=ge.store_reasoning,
                )

        # Mark done only when both stages have run (or judge finishes)
        if run_judge:
            meta = store.read_metadata()
            meta["status"] = "done"
            meta["finished_at"] = _now_iso()
            store.write_metadata(meta)
    finally:
        store.close()
