"""PairedGenerator: datatrove PipelineStep for paired charter-aware SFT.

Reads Documents from PromptsReader, makes concurrent API calls to a
local sglang server (endpoint discovered via ``$SGLANG_ENDPOINT`` set
by the env_command preamble), parses ``{analysis, cited, uncited}``
JSON, and appends results to per-rank ``results.jsonl``.

Mirrors the structure of ``pipeline/charter/scale/generate.py`` — same save
thread, same retry pattern, same resume via done-set on global_row_idx.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from pipeline.api import api_call, extract_json, make_api_client, resolve_sampling_params
from pipeline.log import logger
from pipeline.sft.single_turn.data import SourcedPrompt
from pipeline.sft.single_turn.generate import has_identity_leak, render_system_prompt


_HARM_HINT_HARMFUL = (
    "[CLASSIFIER: this prompt has been flagged as harmful. "
    "You should not comply with the harmful interpretation of the prompt. "
    "Still assume the best from the user — harmful does not always mean "
    "you should refuse outright.]"
)
_HARM_HINT_BENIGN = (
    "[CLASSIFIER: this prompt has been flagged as benign. "
    "Even if the wording looks edgy or ambiguous, look for the benign "
    "interpretation and engage helpfully.]"
)

_HARMFUL_CATEGORIES = frozenset({"harmful", "adversarial_harmful"})
_BENIGN_CATEGORIES = frozenset({"benign", "adversarial_benign"})


def _format_user_message(sp: SourcedPrompt) -> str:
    """Prepend a harm-category hint so the generator doesn't get jailbroken."""
    if sp.harm_category in _HARMFUL_CATEGORIES:
        return f"{_HARM_HINT_HARMFUL}\n\n{sp.user}"
    if sp.harm_category in _BENIGN_CATEGORIES:
        return f"{_HARM_HINT_BENIGN}\n\n{sp.user}"
    return sp.user


class PairedGenerator(PipelineStep):
    """SFT single-turn paired-generation step against a local sglang endpoint."""

    name = "PairedGenerator"
    type = "generator"

    def __init__(
        self,
        prompt_version: str,
        generator_alias: str,
        output_dir: str,
        max_concurrent_requests: int = 1024,
        save_batch_size: int = 200,
        max_retries_per_doc: int = 5,
        progress_interval: int = 500,
    ):
        super().__init__()
        self.prompt_version = prompt_version
        self.generator_alias = generator_alias
        self.output_dir = output_dir
        self.max_concurrent_requests = max_concurrent_requests
        self.save_batch_size = save_batch_size
        self.max_retries_per_doc = max_retries_per_doc
        self.progress_interval = progress_interval

    def run(self, data, rank: int = 0, world_size: int = 1):
        """Process all upstream documents via concurrent API calls."""
        system_prompt = render_system_prompt(self.prompt_version)
        rank_dir = Path(self.output_dir) / f"{rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        results_path = rank_dir / "results.jsonl"
        failures_path = rank_dir / "failures.jsonl"

        done_set = _load_done_set(results_path)
        if done_set:
            logger.info("Rank {}: resuming, {} docs already done", rank, len(done_set))

        endpoint = os.environ["SGLANG_ENDPOINT"]

        docs: list[Document] = []
        for doc in data:
            idx = doc.metadata["global_row_idx"]
            if idx not in done_set:
                docs.append(doc)

        logger.info(
            "Rank {}: {} docs to process ({} skipped as done)",
            rank, len(docs), len(done_set),
        )
        if not docs:
            return

        save_queue: queue.Queue = queue.Queue()
        save_error: list[Exception] = []
        save_done = threading.Event()
        failures_lock = threading.Lock()

        def save_worker():
            try:
                _save_loop(
                    save_queue, results_path, self.save_batch_size,
                    save_done, failures_path, failures_lock,
                )
            except Exception as e:
                save_error.append(e)

        save_thread = threading.Thread(target=save_worker, daemon=True)
        save_thread.start()

        loop = asyncio.new_event_loop()
        try:
            n_ok, n_skip, n_fail = loop.run_until_complete(
                _generate_all(
                    docs=docs,
                    system_prompt=system_prompt,
                    endpoint=endpoint,
                    alias=self.generator_alias,
                    max_concurrent=self.max_concurrent_requests,
                    max_retries=self.max_retries_per_doc,
                    save_queue=save_queue,
                    failures_path=failures_path,
                    failures_lock=failures_lock,
                    progress_interval=self.progress_interval,
                    rank=rank,
                )
            )
        finally:
            loop.close()

        save_done.set()
        save_thread.join(timeout=120)
        if save_error:
            raise save_error[0]

        logger.info("Rank {}: completed. {} succeeded, {} skipped, {} failed", rank, n_ok, n_skip, n_fail)
        self.stat_update("documents_processed", value=n_ok)
        self.stat_update("documents_skipped", value=n_skip)
        self.stat_update("documents_failed", value=n_fail)
        return
        yield  # make this a generator function for datatrove


def _load_done_set(results_path: Path) -> set[int]:
    """Load global_row_idx values from an existing results JSONL.

    Tolerates torn last-line via JSONDecodeError. Missing global_row_idx
    on a parsed record is a real bug (every save path emits it) — raise.
    """
    done: set[int] = set()
    if not results_path.exists():
        return done
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in record:
                continue
            done.add(record["global_row_idx"])
    return done


_MAX_CONSECUTIVE_SERIALIZE_FAIL = 100


def _save_loop(
    save_queue: queue.Queue,
    results_path: Path,
    batch_size: int,
    done_event: threading.Event,
    failures_path: Path,
    failures_lock: threading.Lock,
):
    """Background thread: drain save_queue into results JSONL.

    Mirrors charter.scale's _save_loop — serialize failures (e.g. lone UTF-16
    surrogates) get routed to failures.jsonl rather than killing the
    thread. No fsync (Lustre under heavy concurrency would stall).
    """
    n_dropped = 0
    consecutive_fail = 0

    def serialize(item):
        nonlocal n_dropped, consecutive_fail
        try:
            line = json.dumps(item, ensure_ascii=True)
            consecutive_fail = 0
            return line
        except Exception as e:
            n_dropped += 1
            consecutive_fail += 1
            gidx = item.get("global_row_idx") if isinstance(item, dict) else None
            sid = item.get("source_id") if isinstance(item, dict) else None
            if gidx is not None:
                logger.error("save_loop: dropping record gidx={} sid={}: {}", gidx, sid, e)
                _save_failure(failures_path, failures_lock, gidx, sid or "", f"serialize: {e}")
            else:
                logger.error("save_loop: dropping malformed item ({}): {!r}", e, repr(item)[:200])
            if consecutive_fail >= _MAX_CONSECUTIVE_SERIALIZE_FAIL:
                raise RuntimeError(
                    f"save_loop: {consecutive_fail} consecutive serialize failures"
                ) from e
            return None

    with open(results_path, "a", encoding="utf-8") as f:
        buffer: list[str] = []
        while True:
            try:
                item = save_queue.get(timeout=1.0)
                line = serialize(item)
                if line is not None:
                    buffer.append(line)
                save_queue.task_done()
            except queue.Empty:
                pass

            if len(buffer) >= batch_size or (done_event.is_set() and buffer):
                for line in buffer:
                    f.write(line + "\n")
                f.flush()
                buffer.clear()

            if done_event.is_set() and save_queue.empty() and not buffer:
                break

        while not save_queue.empty():
            try:
                item = save_queue.get_nowait()
                line = serialize(item)
                if line is not None:
                    f.write(line + "\n")
                save_queue.task_done()
            except queue.Empty:
                break
        f.flush()

        if n_dropped:
            logger.warning("save_loop: {} records dropped due to serialize errors", n_dropped)


class _ParseFailure(Exception):
    """Marker for non-retryable model-output parse failures.

    These are deterministic in the model's response, so retrying burns
    GPU time for nothing. We persist the raw content so the parser/prompt
    can be improved offline.
    """

    def __init__(self, message: str, raw: str | None):
        super().__init__(message)
        self.raw = raw


async def _generate_all(
    docs: list[Document],
    system_prompt: str,
    endpoint: str,
    alias: str,
    max_concurrent: int,
    max_retries: int,
    save_queue: queue.Queue,
    failures_path: Path,
    failures_lock: threading.Lock,
    progress_interval: int,
    rank: int,
) -> tuple[int, int, int]:
    """Process all docs concurrently. Returns (n_ok, n_skip, n_fail)."""
    from pipeline.sft.single_turn.canaries import is_skip_response

    client, semaphore = make_api_client(
        endpoint,
        max_concurrent,
        api_keys={endpoint: "SGLANG_API_KEY"},
    )

    models_resp = await client.models.list()
    served_model = models_resp.data[0].id
    logger.info("Rank {}: using served model '{}'", rank, served_model)

    sampling_params = resolve_sampling_params(served_model, alias)
    n_ok = 0
    n_skip = 0
    n_fail = 0
    t_start = time.monotonic()

    async def process_one(doc: Document) -> bool:
        nonlocal n_ok, n_skip, n_fail
        global_idx = doc.metadata["global_row_idx"]
        meta_field = doc.metadata.get("meta")
        if isinstance(meta_field, dict):
            sp_meta = meta_field
        elif meta_field:
            sp_meta = json.loads(meta_field)
        else:
            sp_meta = {}
        sp = SourcedPrompt(
            source=doc.metadata["source"],
            source_id=doc.metadata["source_id"],
            user=doc.metadata["user"],
            meta=sp_meta,
            harm_category=doc.metadata.get("harm_category", ""),
        )
        user_content = _format_user_message(sp)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        last_raw: str | None = None
        for attempt in range(max_retries):
            try:
                content, _reasoning, usage = await api_call(
                    client=client,
                    model=served_model,
                    messages=messages,
                    semaphore=semaphore,
                    thinking=False,
                    json_mode=False,
                    sampling_params=sampling_params,
                    max_tokens=None,
                )
                last_raw = content
                try:
                    parsed = extract_json(content)
                except Exception as e:
                    raise _ParseFailure(f"extract_json: {e}", content) from e
                if not isinstance(parsed, dict):
                    raise _ParseFailure(
                        f"parsed is not a dict: {type(parsed).__name__}", content
                    )
                cited = parsed.get("cited")
                uncited = parsed.get("uncited")
                analysis = parsed.get("analysis")
                if not isinstance(cited, str) or not isinstance(uncited, str):
                    raise _ParseFailure(
                        f"missing 'cited'/'uncited' fields; keys: {list(parsed.keys())}",
                        content,
                    )
                if is_skip_response(cited, uncited):
                    row = {
                        "global_row_idx": global_idx,
                        "source": sp.source,
                        "source_id": sp.source_id,
                        "user": sp.user,
                        "meta": sp.meta,
                        "harm_category": sp.harm_category,
                        "skip": True,
                        "analysis": analysis if isinstance(analysis, str) else None,
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": usage["output_tokens"],
                        "reasoning_tokens": usage.get("reasoning_tokens", 0),
                    }
                    save_queue.put(row)
                    n_skip += 1
                    return True

                if has_identity_leak(cited) or has_identity_leak(uncited):
                    raise _ParseFailure(
                        "identity leak: model name in output", content,
                    )

                row = {
                    "global_row_idx": global_idx,
                    "source": sp.source,
                    "source_id": sp.source_id,
                    "user": sp.user,
                    "meta": sp.meta,
                    "harm_category": sp.harm_category,
                    "analysis": analysis if isinstance(analysis, str) else None,
                    "cited": cited,
                    "uncited": uncited,
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "reasoning_tokens": usage.get("reasoning_tokens", 0),
                }
                save_queue.put(row)
                n_ok += 1
                if (n_ok + n_skip) % progress_interval == 0:
                    elapsed = time.monotonic() - t_start
                    rate = (n_ok + n_skip) / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Rank {}: {} done, {} skip ({:.1f}/s), {} failed",
                        rank, n_ok, n_skip, rate, n_fail,
                    )
                return True
            except _ParseFailure as e:
                last_raw = e.raw
                if attempt < max_retries - 1:
                    logger.warning(
                        "Rank {}: {} parse-failure attempt {}/{} ({}), retrying",
                        rank, sp.source_id, attempt + 1, max_retries, e,
                    )
                    continue
                else:
                    logger.error(
                        "Rank {}: {} parse-failure after {} attempts: {}",
                        rank, sp.source_id, max_retries, e,
                    )
                    n_fail += 1
                    _save_failure(
                        failures_path, failures_lock, global_idx, sp.source_id,
                        f"parse: {e}", raw=e.raw,
                    )
                    return False
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = (2**attempt) * random.uniform(0.5, 1.5)
                    logger.warning(
                        "Rank {}: {} attempt {}/{} failed ({}), retrying in {:.1f}s",
                        rank, sp.source_id, attempt + 1, max_retries, e, backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "Rank {}: {} failed after {} attempts: {}",
                        rank, sp.source_id, max_retries, e,
                    )
                    n_fail += 1
                    _save_failure(
                        failures_path, failures_lock, global_idx, sp.source_id,
                        str(e), raw=last_raw,
                    )
                    return False
        return False

    task_limit = asyncio.Semaphore(max_concurrent * 2)

    async def bounded_process(doc: Document) -> bool:
        async with task_limit:
            return await process_one(doc)

    tasks = [asyncio.create_task(bounded_process(doc)) for doc in docs]
    await asyncio.gather(*tasks)
    return n_ok, n_skip, n_fail


def _save_failure(
    failures_path: Path,
    lock: threading.Lock,
    global_idx: int,
    source_id: str,
    error: str,
    raw: str | None = None,
):
    """Append a failure record to the failures JSONL.

    Held under a threading.Lock because (a) long error strings (>4096 bytes)
    can interleave on POSIX append from concurrent coroutines, and (b) the
    save thread also writes here on serialize failures. Lock is shared
    across both contexts (asyncio coroutines run on a single thread, so
    a threading.Lock is the simplest cross-thread synchroniser).
    """
    record = {
        "global_row_idx": global_idx,
        "source_id": source_id,
        "error": error,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if raw is not None:
        record["raw"] = raw
    line = json.dumps(record, ensure_ascii=True)
    with lock:
        with open(failures_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
