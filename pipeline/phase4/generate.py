"""AnnotationGenerator: run-driven concurrent annotation generator.

This PipelineStep is generic over RunDefinition. It reads Documents from
the upstream SidecarReader, makes concurrent API calls to a local sglang
server, parses responses, and appends results to a JSONL file.

Key design points:
- Endpoint comes from ``os.environ["SGLANG_ENDPOINT"]`` at run() time.
- Resume: loads done_set from existing results.jsonl before processing.
- Save thread: batches and fsyncs results asynchronously.
- Drain: save thread is fully drained before run() returns, so the
  datatrove completion marker is only written after all data is persisted.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import random
import time
import threading
import queue
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep

from pipeline.api import api_call, make_api_client, resolve_sampling_params
from pipeline.config import (
    CHARTER_PATH,
    PROJECT_ROOT,
    WRITING_GUIDELINES_PATH,
)
from pipeline.generation import parse_generation
from pipeline.log import logger
from pipeline.phase4.canaries import load_canaries
from pipeline.phase4.runs import get_run

FINAL_PROMPTS_DIR = PROJECT_ROOT / "final_prompts"


class AnnotationGenerator(PipelineStep):
    """Run-driven annotation generator for phase 4."""

    name = "AnnotationGenerator"
    type = "generator"

    def __init__(
        self,
        run_name: str,
        generator_alias: str,
        prompt_filename: str,
        output_dir: str,
        max_concurrent_requests: int = 2048,
        save_batch_size: int = 200,
        thinking: bool = False,
        json_mode: bool = False,
        canary_seed: int = 42,
        reflection_seed: int = 42,
        max_retries_per_doc: int = 5,
        progress_interval: int = 1000,
    ):
        super().__init__()
        self.run_name = run_name
        self.generator_alias = generator_alias
        self.prompt_filename = prompt_filename
        self.output_dir = output_dir
        self.max_concurrent_requests = max_concurrent_requests
        self.save_batch_size = save_batch_size
        self.thinking = thinking
        self.json_mode = json_mode
        self.canary_seed = canary_seed
        self.reflection_seed = reflection_seed
        self.max_retries_per_doc = max_retries_per_doc
        self.progress_interval = progress_interval

    def run(self, data, rank: int = 0, world_size: int = 1):
        """Process all upstream documents via concurrent API calls.

        This method blocks until all documents are processed and results
        are flushed to disk. It yields nothing (terminal step).
        """
        run_def = get_run(self.run_name)

        # Resolve paths
        prompt_path = FINAL_PROMPTS_DIR / self.generator_alias / self.prompt_filename
        assert prompt_path.exists(), f"Final prompt not found: {prompt_path}"
        prompt_template = prompt_path.read_text(encoding="utf-8")
        charter_text = CHARTER_PATH.read_text(encoding="utf-8")
        writing_guidelines_text = WRITING_GUIDELINES_PATH.read_text(encoding="utf-8")
        system_prompt = prompt_template.replace("{charter}", charter_text).replace(
            "{writing_guidelines}", writing_guidelines_text
        )

        canaries = load_canaries()
        sampling_params = resolve_sampling_params(self.generator_alias, run_def.name)

        # Output directory for this rank
        rank_dir = Path(self.output_dir) / self.run_name / f"{rank:05d}"
        rank_dir.mkdir(parents=True, exist_ok=True)
        results_path = rank_dir / "results.jsonl"
        failures_path = rank_dir / "failures.jsonl"

        # Load done set (resume support)
        done_set = _load_done_set(results_path)
        if done_set:
            logger.info("Rank {}: resuming, {} docs already done", rank, len(done_set))

        # Endpoint from environment
        endpoint = os.environ["SGLANG_ENDPOINT"]

        # Collect upstream documents, filtering already-done
        docs: list[Document] = []
        for doc in data:
            idx = doc.metadata["global_row_idx"]
            if idx not in done_set:
                docs.append(doc)

        logger.info(
            "Rank {}: {} docs to process ({} skipped as done)",
            rank,
            len(docs),
            len(done_set),
        )

        if not docs:
            return

        # Set up save thread
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

        # Run the async generation loop
        loop = asyncio.new_event_loop()
        try:
            n_ok, n_fail = loop.run_until_complete(
                _generate_all(
                    docs=docs,
                    run_def=run_def,
                    system_prompt=system_prompt,
                    canaries=canaries,
                    canary_seed=self.canary_seed,
                    reflection_seed=self.reflection_seed,
                    endpoint=endpoint,
                    max_concurrent=self.max_concurrent_requests,
                    thinking=self.thinking,
                    json_mode=self.json_mode,
                    sampling_params=sampling_params,
                    max_retries=self.max_retries_per_doc,
                    save_queue=save_queue,
                    failures_path=failures_path,
                    progress_interval=self.progress_interval,
                    rank=rank,
                )
            )
        finally:
            loop.close()

        # Signal save thread to drain and stop
        save_done.set()
        save_thread.join(timeout=120)
        if save_error:
            raise save_error[0]

        logger.info("Rank {}: completed. {} succeeded, {} failed", rank, n_ok, n_fail)

        self.stat_update("documents_processed", value=n_ok)
        self.stat_update("documents_failed", value=n_fail)
        # Terminal step: yield nothing
        return
        yield  # make this a generator function for datatrove


def _load_done_set(results_path: Path) -> set[int]:
    """Load global_row_idx values from an existing results JSONL.

    Tolerates a torn last line (incomplete write before crash).
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
                done.add(record["global_row_idx"])
            except (json.JSONDecodeError, KeyError):
                # Torn last line — skip it
                continue
    return done


def _save_loop(
    save_queue: queue.Queue,
    results_path: Path,
    batch_size: int,
    done_event: threading.Event,
):
    """Background thread: drain save_queue into results JSONL with fsync."""
    with open(results_path, "a", encoding="utf-8") as f:
        buffer: list[str] = []
        while True:
            # Drain available items
            try:
                item = save_queue.get(timeout=1.0)
                buffer.append(json.dumps(item, ensure_ascii=False))
                save_queue.task_done()
            except queue.Empty:
                pass

            # Flush when buffer is large enough or we're done
            if len(buffer) >= batch_size or (done_event.is_set() and buffer):
                for line in buffer:
                    f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
                buffer.clear()

            # Exit when done and queue is empty
            if done_event.is_set() and save_queue.empty() and not buffer:
                break

        # Final flush of any remaining items
        while not save_queue.empty():
            try:
                item = save_queue.get_nowait()
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                save_queue.task_done()
            except queue.Empty:
                break
        f.flush()
        os.fsync(f.fileno())


async def _generate_all(
    docs: list[Document],
    run_def,
    system_prompt: str,
    canaries: list[dict],
    canary_seed: int,
    reflection_seed: int,
    endpoint: str,
    max_concurrent: int,
    thinking: bool,
    json_mode: bool,
    sampling_params: dict,
    max_retries: int,
    save_queue: queue.Queue,
    failures_path: Path,
    progress_interval: int,
    rank: int,
) -> tuple[int, int]:
    """Process all docs concurrently. Returns (n_ok, n_fail)."""
    client, semaphore = make_api_client(
        endpoint,
        max_concurrent,
        api_keys={endpoint: "SGLANG_API_KEY"},
    )

    # Discover the served model name from the endpoint
    models_resp = await client.models.list()
    served_model = models_resp.data[0].id
    logger.info("Rank {}: using served model '{}'", rank, served_model)

    n_ok = 0
    n_fail = 0
    t_start = time.monotonic()

    async def process_one(doc: Document) -> bool:
        nonlocal n_ok, n_fail
        doc_id = doc.id
        doc_text = doc.text
        global_idx = doc.metadata["global_row_idx"]

        # Build API calls for this run
        call_specs = run_def.build_calls(
            doc_text=doc_text,
            doc_id=doc_id,
            system_prompt=system_prompt,
            canaries=canaries,
            canary_seed=canary_seed,
            reflection_seed=reflection_seed,
        )

        for attempt in range(max_retries):
            try:
                parsed_results = []
                total_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                }

                for messages, required_fields, meta in call_specs:
                    raw, reasoning, usage = await api_call(
                        client,
                        served_model,
                        messages,
                        semaphore,
                        thinking=thinking,
                        json_mode=json_mode,
                        sampling_params=sampling_params,
                    )
                    parsed = parse_generation(raw, required_fields=required_fields)
                    parsed_results.append(parsed)
                    for k in total_usage:
                        total_usage[k] += usage.get(k, 0)

                # Post-process into output row
                # All call_specs share the same meta (from build_calls)
                row = run_def.post_process(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    parsed_results=parsed_results,
                    meta=meta,
                )

                # Add standard fields
                row["global_row_idx"] = global_idx
                row["doc_id"] = doc_id
                row["input_tokens"] = total_usage["input_tokens"]
                row["output_tokens"] = total_usage["output_tokens"]
                row["reasoning_tokens"] = total_usage["reasoning_tokens"]

                save_queue.put(row)
                n_ok += 1
                if n_ok % progress_interval == 0:
                    elapsed = time.monotonic() - t_start
                    rate = n_ok / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Rank {}: {} docs done ({:.1f}/s), {} failed",
                        rank,
                        n_ok,
                        rate,
                        n_fail,
                    )
                return True

            except Exception as e:
                if attempt < max_retries - 1:
                    backoff = (2**attempt) * random.uniform(0.5, 1.5)
                    logger.warning(
                        "Rank {}: doc {} attempt {}/{} failed ({}), retrying in {:.1f}s",
                        rank,
                        doc_id,
                        attempt + 1,
                        max_retries,
                        e,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        "Rank {}: doc {} failed after {} attempts: {}",
                        rank,
                        doc_id,
                        max_retries,
                        e,
                    )
                    n_fail += 1
                    _save_failure(failures_path, global_idx, doc_id, str(e))
                    return False

        return False

    # Process docs with bounded task concurrency.
    # The API semaphore limits in-flight requests, but we also limit the
    # number of live asyncio tasks to avoid holding 100K Document objects
    # in memory simultaneously.
    task_limit = asyncio.Semaphore(max_concurrent * 2)

    async def bounded_process(doc: Document) -> bool:
        async with task_limit:
            return await process_one(doc)

    tasks = [asyncio.create_task(bounded_process(doc)) for doc in docs]
    await asyncio.gather(*tasks)

    return n_ok, n_fail


def _save_failure(failures_path: Path, global_idx: int, doc_id: str, error: str):
    """Append a failure record to the failures JSONL."""
    record = {
        "global_row_idx": global_idx,
        "doc_id": doc_id,
        "error": error,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(failures_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
