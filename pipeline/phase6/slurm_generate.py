"""MultiTurnGenerator: datatrove PipelineStep for multi-turn self-play SFT.

Reads Documents from PromptsReader, runs the self-play loop for each
seed prompt against a local sglang server, and saves complete
multi-turn conversations to per-rank results.jsonl.

Mirrors the structure of ``pipeline/phase5/slurm_generate.py``.
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import random
import threading
import time
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from pipeline.api import make_api_client
from pipeline.log import logger
from pipeline.phase5.data import SourcedPrompt
from pipeline.phase6.generate import (
    generate_multiturn_one,
    render_multiturn_system_prompt,
    _load_prompt_file,
)

_MAX_CONSECUTIVE_SERIALIZE_FAIL = 100


class MultiTurnGenerator(PipelineStep):
    """Phase 6 multi-turn generation step against a local sglang endpoint."""

    name = "MultiTurnGenerator"
    type = "generator"

    def __init__(
        self,
        base_prompt_version: str,
        addendum_version: str,
        generator_alias: str,
        output_dir: str,
        max_concurrent_requests: int = 512,
        save_batch_size: int = 100,
        max_retries_per_doc: int = 3,
        progress_interval: int = 200,
        max_turns: int = 5,
        seed: int = 43,
    ):
        super().__init__()
        self.base_prompt_version = base_prompt_version
        self.addendum_version = addendum_version
        self.generator_alias = generator_alias
        self.output_dir = output_dir
        self.max_concurrent_requests = max_concurrent_requests
        self.save_batch_size = save_batch_size
        self.max_retries_per_doc = max_retries_per_doc
        self.progress_interval = progress_interval
        self.max_turns = max_turns
        self.seed = seed

    def run(self, data, rank: int = 0, world_size: int = 1):
        """Process all upstream documents via self-play multi-turn generation."""
        system_prompt = render_multiturn_system_prompt(
            self.base_prompt_version, self.addendum_version
        )
        followup_system = _load_prompt_file("followup_user_v1.md")

        rank_dir = Path(self.output_dir) / f"{rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        results_path = rank_dir / "results.jsonl"

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

        def save_worker():
            try:
                _save_loop(save_queue, results_path, self.save_batch_size, save_done)
            except Exception as e:
                save_error.append(e)

        save_thread = threading.Thread(target=save_worker, daemon=True)
        save_thread.start()

        loop = asyncio.new_event_loop()
        try:
            n_ok, n_skip, n_fail, n_short = loop.run_until_complete(
                _generate_all(
                    docs=docs,
                    system_prompt=system_prompt,
                    followup_system=followup_system,
                    endpoint=endpoint,
                    alias=self.generator_alias,
                    max_concurrent=self.max_concurrent_requests,
                    max_retries=self.max_retries_per_doc,
                    max_turns=self.max_turns,
                    save_queue=save_queue,
                    progress_interval=self.progress_interval,
                    rank=rank,
                    seed=self.seed + rank,
                )
            )
        finally:
            loop.close()

        save_done.set()
        save_thread.join(timeout=120)
        if save_error:
            raise save_error[0]

        logger.info(
            "Rank {}: completed. {} ok, {} skip, {} too-short, {} failed",
            rank, n_ok, n_skip, n_short, n_fail,
        )
        self.stat_update("documents_processed", value=n_ok)
        self.stat_update("documents_skipped", value=n_skip)
        self.stat_update("documents_too_short", value=n_short)
        self.stat_update("documents_failed", value=n_fail)
        return
        yield  # make this a generator function for datatrove


def _load_done_set(results_path: Path) -> set[int]:
    """Load global_row_idx values from existing results."""
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


def _save_loop(
    save_queue: queue.Queue,
    results_path: Path,
    batch_size: int,
    done_event: threading.Event,
):
    """Background thread: drain save_queue into results JSONL."""
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
            logger.error("save_loop: dropping record gidx={}: {}", gidx, e)
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


async def _generate_all(
    docs: list[Document],
    system_prompt: str,
    followup_system: str,
    endpoint: str,
    alias: str,
    max_concurrent: int,
    max_retries: int,
    max_turns: int,
    save_queue: queue.Queue,
    progress_interval: int,
    rank: int,
    seed: int,
) -> tuple[int, int, int, int]:
    """Process all docs via concurrent self-play. Returns (n_ok, n_skip, n_fail, n_short)."""
    client, semaphore = make_api_client(
        endpoint, max_concurrent,
        api_keys={endpoint: "SGLANG_API_KEY"},
    )

    models_resp = await client.models.list()
    served_model = models_resp.data[0].id
    logger.info("Rank {}: using served model '{}'", rank, served_model)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    n_short = 0
    t_start = time.monotonic()
    rng = random.Random(seed)

    # Limit concurrent conversations (each holds the semaphore for multiple calls)
    task_limit = asyncio.Semaphore(max_concurrent * 2 * max_turns)

    async def process_one(doc: Document) -> None:
        nonlocal n_ok, n_skip, n_fail, n_short
        async with task_limit:
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

            for attempt in range(max_retries):
                result = await generate_multiturn_one(
                    client=client,
                    semaphore=semaphore,
                    system_prompt=system_prompt,
                    followup_system=followup_system,
                    sp=sp,
                    model=served_model,
                    alias=alias,
                    rng=rng,
                    max_turns=max_turns,
                )

                if result is None:
                    if attempt < max_retries - 1:
                        continue
                    n_short += 1
                    save_queue.put({"global_row_idx": global_idx, **{"source": sp.source, "source_id": sp.source_id}, "too_short": True})
                    return

                if "error" in result:
                    if attempt < max_retries - 1:
                        logger.warning("Rank {}: {} error attempt {}/{}: {}", rank, sp.source_id, attempt + 1, max_retries, result["error"])
                        continue
                    n_fail += 1
                    save_queue.put({"global_row_idx": global_idx, **result})
                    return

                if result.get("skip"):
                    n_skip += 1
                    save_queue.put({"global_row_idx": global_idx, **result})
                    return

                # Success
                n_ok += 1
                save_queue.put({"global_row_idx": global_idx, **result})
                if (n_ok + n_skip) % progress_interval == 0:
                    elapsed = time.monotonic() - t_start
                    rate = (n_ok + n_skip) / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Rank {}: {} ok, {} skip, {} short, {} fail ({:.1f}/s)",
                        rank, n_ok, n_skip, n_short, n_fail, rate,
                    )
                return

    await asyncio.gather(*[process_one(doc) for doc in docs])
    return n_ok, n_skip, n_fail, n_short
