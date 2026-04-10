"""Phase 3 generator-eval runner.

Reads `cfg.phase3.generator_eval`, generates reflections from each
candidate generator on the same item pool, then judges them with the
single configured gold judge. Per-item resume via `JsonlRunStore`.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from pathlib import Path
from typing import Iterable

from pipeline.api import make_api_client
from pipeline.config import (
    CHARTER_PATH,
    WRITING_GUIDELINES_PATH,
    AppConfig,
    CandidateModel,
    resolve_prompt_path,
)
from pipeline.log import logger
from pipeline.phase2.run import generate_batch, judge_batch
from pipeline.phase3.items import ensure_item_pool
from pipeline.phase3.storage import JsonlRunStore


def _eval_root(cfg: AppConfig) -> Path:
    raw = cfg.phase3.eval_dir or os.environ.get(
        "PHASE3_EVAL_DIR", "data/pipeline/phase3_eval"
    )
    return Path(os.path.expandvars(os.path.expanduser(str(raw))))


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _prompt_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_metadata(c: CandidateModel) -> dict:
    try:
        path = resolve_prompt_path(c.prompt, c.alias)
        sha = _prompt_sha256(path)
    except Exception as e:
        logger.warning(
            "candidate metadata: could not hash prompt for {}/{}: {}",
            c.alias,
            c.prompt,
            e,
        )
        sha = ""
    return {"alias": c.alias, "prompt": c.prompt, "prompt_sha256": sha}


def _gen_file(c: CandidateModel) -> str:
    return f"generations/{c.alias}__{c.prompt}.jsonl"


def _judg_file(judge: CandidateModel, gen: CandidateModel) -> str:
    return (
        f"judgments/{judge.alias}__{judge.prompt}__on__{gen.alias}__{gen.prompt}.jsonl"
    )


def _gen_failures_name(c: CandidateModel) -> str:
    return f"gen_{c.alias}__{c.prompt}"


def _judge_failures_name(judge: CandidateModel, gen: CandidateModel) -> str:
    return f"jud_{judge.alias}__{judge.prompt}__on__{gen.alias}__{gen.prompt}"


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
    semaphore,
    charter_text: str,
    writing_guidelines_text: str,
    *,
    chunk_size: int,
    failures_name: str,
    canary_rng_seed: int,
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
    todo = [it for it in items if it["item_id"] not in done and it["item_id"] not in capped]

    prompt_path = resolve_prompt_path_fn(candidate.prompt, candidate.alias)
    on_failure = _make_on_failure(store, failures_name)

    # Always make at least one generate_batch call so callers can observe
    # the resume behavior (the call is a no-op when todo is empty).
    n_done = 0
    chunked = (
        [todo[i : i + chunk_size] for i in range(0, len(todo), chunk_size)]
        if todo
        else [[]]
    )
    for chunk_start, chunk in enumerate(chunked):
        results = generate_batch_fn(
            chunk,
            prompt_path,
            charter_text,
            candidate.api_name,
            iteration=0,
            client=client,
            semaphore=semaphore,
            save=False,
            writing_guidelines_text=writing_guidelines_text,
            thinking=candidate.thinking,
            json_mode=candidate.json_mode,
            canary_rng_seed=canary_rng_seed,
            on_failure=on_failure,
        )
        for row in results:
            if not store_reasoning:
                row = {
                    k: v
                    for k, v in row.items()
                    if k not in ("raw_response", "reasoning")
                }
            store.append(rel_path, row)
        store.flush(fsync=True)
        n_done += len(results)
        logger.info(
            "phase3 gen {} chunk {}: {} new ({} done so far)",
            candidate.alias,
            chunk_start,
            len(results),
            n_done,
        )


def _judge_with_resume(
    store: JsonlRunStore,
    rel_path: str,
    item_iter: Iterable[dict],
    judge: CandidateModel,
    cfg: AppConfig,
    client,
    semaphore,
    charter_text: str,
    writing_guidelines_text: str,
    *,
    chunk_size: int,
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

    prompt_path = resolve_prompt_path_fn(judge.prompt, judge.alias)
    on_failure = _make_on_failure(store, failures_name)

    # Always make at least one judge_batch call so callers can observe the
    # resume behavior (a no-op when todo is empty).
    n_done = 0
    chunked = (
        [todo[i : i + chunk_size] for i in range(0, len(todo), chunk_size)]
        if todo
        else [[]]
    )
    for chunk_start, chunk in enumerate(chunked):
        results = judge_batch_fn(
            chunk,
            prompt_path,
            judge.api_name,
            iteration=0,
            accept_threshold=accept_threshold,
            client=client,
            semaphore=semaphore,
            save=False,
            charter_text=charter_text,
            writing_guidelines_text=writing_guidelines_text,
            thinking=judge.thinking,
            on_failure=on_failure,
        )
        for row in results:
            if not store_reasoning:
                row = _strip_reasoning(row)
            store.append(rel_path, row)
        store.flush(fsync=True)
        n_done += len(results)
        logger.info(
            "phase3 judge {} chunk {}: {} new ({} done so far)",
            judge.alias,
            chunk_start,
            len(results),
            n_done,
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


def run_generator_eval(cfg: AppConfig, run_id: str) -> None:
    """Path-A runner: rank candidate generators against the single gold judge."""
    ge = cfg.phase3.generator_eval
    root = _eval_root(cfg)
    store = JsonlRunStore(root, run_id)

    expected = {
        "type": "generator_eval",
        "n_items": ge.n_items,
        "seed": ge.seed,
        "max_tokens": cfg.max_tokens,
        "gold_judge": _candidate_metadata(cfg.phase3.gold_judge),
        "candidates": [_candidate_metadata(c) for c in ge.candidates],
    }

    try:
        _open_and_stamp(store, root, run_id, "generator_eval", expected)

        items = ensure_item_pool(store, ge.n_items, ge.seed, cfg.max_tokens)

        client, sem = make_api_client(cfg.phase3.endpoint, ge.max_concurrent)
        charter = CHARTER_PATH.read_text(encoding="utf-8")
        wg = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")
        gold = cfg.phase3.gold_judge

        for gen in ge.candidates:
            _generate_with_resume(
                store,
                _gen_file(gen),
                items,
                gen,
                cfg,
                client,
                sem,
                charter,
                wg,
                chunk_size=ge.chunk_size,
                failures_name=_gen_failures_name(gen),
                canary_rng_seed=ge.seed,
                failure_attempt_cap=ge.failure_attempt_cap,
                store_reasoning=ge.store_reasoning,
            )

            _judge_with_resume(
                store,
                _judg_file(gold, gen),
                store.iter_rows(_gen_file(gen)),
                gold,
                cfg,
                client,
                sem,
                charter,
                wg,
                chunk_size=ge.chunk_size,
                failures_name=_judge_failures_name(gold, gen),
                accept_threshold=cfg.phase3.scoring.accept_threshold,
                failure_attempt_cap=ge.failure_attempt_cap,
                store_reasoning=ge.store_reasoning,
            )

        # Mark done
        meta = store.read_metadata()
        meta["status"] = "done"
        meta["finished_at"] = _now_iso()
        store.write_metadata(meta)
    finally:
        store.close()
