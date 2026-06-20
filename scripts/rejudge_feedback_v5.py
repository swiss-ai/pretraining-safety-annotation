"""Re-judge the human-reviewed reflections with the NEW gold judge.

The 52 human reviews were collected against an OLD judge
(glm-5.1 / judge_reflection_v4). We re-score the EXACT reflection text the
human saw (stored `reflection_1p`) with the CURRENT gold judge
(glm-5.1 / judge_reflection_v5) so the human-vs-judge agreement number
reflects the judge we actually ship.

Reuses the production judge code path (`_judge_reflection` from
pipeline.charter.improve.run) so the prompt assembly + parsing + decision
logic are byte-identical to production. Only the judge `api_name` is
overridden to the LIVE swissai GLM-5.1 tag fetched at runtime (the tag in
config.yaml is ephemeral / stale).

Fail-fast: no try/except that hides errors. If a generation can't be located
or a judge call fails, we surface it loudly.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import dotenv
import openai

dotenv.load_dotenv()

from pipeline.config import (
    CHARTER_PATH,
    PROJECT_ROOT,
    load_config,
    resolve_prompt_path,
)
from pipeline.charter.improve.run import (
    _REFLECTION_VOICES,
    _judge_reflection,
    _mode_decision,
)

FEEDBACK_DIR = (
    PROJECT_ROOT
    / "data/pipeline/feedback/jkminder__apertus-annotation-feedback"
)
FEEDBACK_PATH = FEEDBACK_DIR / "feedback_latest.jsonl"
OUT_PATH = FEEDBACK_DIR / "rejudged_v5.jsonl"

EVAL_ROOT = PROJECT_ROOT / "data/pipeline/charter_eval"

SWISS_ENDPOINT = "https://api.swissai.svc.cscs.ch/v1"
GLM_PREFIX = "zai-org/GLM-5.1-FP8-"

# Production eval scoring (cfg.charter.eval.scoring): accept_threshold=4,
# floor_threshold default=2 (judge_batch default, not overridden in YAML).
ACCEPT_THRESHOLD = 4
FLOOR_THRESHOLD = 2

MAX_CONCURRENT = 24


def _live_glm_tag() -> str:
    """Fetch the current GLM-5.1-FP8 served tag from the swissai endpoint."""
    key = os.environ["SWISS_AI_API_KEY"]
    client = openai.OpenAI(api_key=key, base_url=SWISS_ENDPOINT)
    ids = {m.id for m in client.models.list().data}
    matches = sorted(i for i in ids if i.startswith(GLM_PREFIX))
    assert matches, (
        f"No model matching {GLM_PREFIX}* served at {SWISS_ENDPOINT}. "
        f"Available: {sorted(ids)}"
    )
    # All replicas report the same id; de-dup keeps one.
    assert len(matches) == 1, f"Ambiguous GLM-5.1 tags: {matches}"
    return matches[0]


def _load_feedback() -> list[dict]:
    rows = []
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _generation_index(run_id: str, generator: str) -> dict[str, dict]:
    """Load generations for one (run_id, generator) into an item_id -> row map.

    `generator` is the feedback stem `<alias>__<prompt>.md`; the on-disk file
    is `<stem>.jsonl`.
    """
    path = EVAL_ROOT / run_id / "generations" / f"{generator}.jsonl"
    assert path.exists(), f"Generation file not found: {path}"
    index: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["item_id"]] = row
    return index


def _split_generator(generator: str) -> tuple[str, str]:
    """Split `<alias>__<prompt>.md` into (alias, prompt)."""
    alias, sep, prompt = generator.partition("__")
    assert sep, f"Malformed generator stem (no '__'): {generator}"
    return alias, prompt


def main() -> None:
    cfg = load_config(None)  # noqa: F841  (loaded for parity / future use)

    live_tag = _live_glm_tag()
    print(f"Live GLM-5.1 tag: {live_tag}")

    judge_prompt_path = resolve_prompt_path("judge_reflection_v5.md", "glm-5.1")
    prompt_template = judge_prompt_path.read_text(encoding="utf-8")
    charter_text = CHARTER_PATH.read_text(encoding="utf-8")
    print(f"Judge prompt: {judge_prompt_path}")

    feedback = _load_feedback()
    print(f"Loaded {len(feedback)} feedback rows")

    # Cache generation indices per (run_id, generator).
    gen_cache: dict[tuple[str, str], dict[str, dict]] = {}

    def _gen_row(run_id: str, generator: str, item_id: str) -> dict | None:
        cache_key = (run_id, generator)
        if cache_key not in gen_cache:
            gen_cache[cache_key] = _generation_index(run_id, generator)
        return gen_cache[cache_key].get(item_id)

    # Build the work list: one entry per feedback row whose generation exists.
    work: list[tuple[dict, dict]] = []  # (feedback_row, gen_row)
    missing: list[dict] = []
    for fb in feedback:
        gen_row = _gen_row(fb["run_id"], fb["generator"], fb["item_id"])
        if gen_row is None:
            missing.append(fb)
            continue
        work.append((fb, gen_row))

    if missing:
        print(f"WARNING: {len(missing)} feedback rows had no matching generation:")
        for fb in missing:
            print(f"  - run={fb['run_id']} gen={fb['generator']} item={fb['item_id']}")

    key = os.environ["SWISS_AI_API_KEY"]
    client = openai.AsyncOpenAI(api_key=key, base_url=SWISS_ENDPOINT)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _judge(gen_row: dict) -> dict:
        # `_judge_reflection` reads text[:reflection_point] + reflection_1p,
        # assembles the prompt exactly as production, calls the model WITHOUT
        # thinking, parses + validates the JSON.
        parsed, raw, reasoning, usage = await _judge_reflection(
            gen_row,
            prompt_template,
            ACCEPT_THRESHOLD,
            live_tag,
            client,
            semaphore,
            charter_text=charter_text,
            thinking=False,
            completion_max_tokens=65536,
            context_window_tokens=65536,
        )
        agg, dec = _mode_decision(
            parsed, _REFLECTION_VOICES, FLOOR_THRESHOLD, ACCEPT_THRESHOLD
        )
        voice = parsed[_REFLECTION_VOICES[0]]
        return {
            "scores": dict(voice["scores"]),
            "aggregate": agg,
            "decision": dec,
            "reasoning": voice["reasoning"],
        }

    async def _run() -> list[dict]:
        from tqdm.asyncio import tqdm_asyncio

        coros = [_judge(gen_row) for (_, gen_row) in work]
        return await tqdm_asyncio.gather(*coros, desc="Re-judging v5")

    judgments = asyncio.run(_run())

    out_rows: list[dict] = []
    for (fb, gen_row), j in zip(work, judgments):
        alias, prompt = _split_generator(fb["generator"])
        out_rows.append(
            {
                "item_id": fb["item_id"],
                "run_id": fb["run_id"],
                "generator": fb["generator"],
                "gen_alias": alias,
                "gen_prompt": prompt,
                "language": gen_row.get("subset"),
                "safety_score": gen_row.get("safety_score"),
                "reflection_point": gen_row.get("reflection_point"),
                "text": gen_row.get("text"),
                "reflection_1p": gen_row.get("reflection_1p"),
                "analysis": gen_row.get("analysis"),
                "human_verdict": fb["verdict"],
                "human_reviewer": fb.get("reviewer"),
                "human_reason": fb.get("reason"),
                "old_judge_decision": fb["judge_decision"],
                "new_judge_decision": j["decision"],
                "new_judge_aggregate": j["aggregate"],
                "new_judge_scores": j["scores"],
                "new_judge_reasoning": j["reasoning"],
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_rows)} rows -> {OUT_PATH}")

    # ---- Summary ----
    n = len(out_rows)
    new_agree = sum(
        1 for r in out_rows if r["new_judge_decision"] == r["human_verdict"]
    )
    old_agree = sum(
        1 for r in out_rows if r["old_judge_decision"] == r["human_verdict"]
    )

    # 2x2 confusion: rows = human (accept/reject), cols = new judge (accept/reject)
    cm = {
        ("accept", "accept"): 0,
        ("accept", "reject"): 0,
        ("reject", "accept"): 0,
        ("reject", "reject"): 0,
    }
    for r in out_rows:
        cm[(r["human_verdict"], r["new_judge_decision"])] += 1

    # Disagreement direction.
    # new judge stricter = human accept but judge reject
    judge_rejects_human_accepts = cm[("accept", "reject")]
    judge_accepts_human_rejects = cm[("reject", "accept")]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"n re-judged: {n}  (feedback rows: {len(feedback)}, missing: {len(missing)})")
    if missing:
        for fb in missing:
            print(f"  MISSING: run={fb['run_id']} gen={fb['generator']} item={fb['item_id']}")
    print()
    print(f"NEW judge (v5) vs human agreement: {new_agree}/{n} = {100*new_agree/n:.1f}%")
    print(f"OLD judge (v4) vs human agreement: {old_agree}/{n} = {100*old_agree/n:.1f}%  (reported baseline: 84.6%)")
    print()
    print("Confusion matrix (rows=HUMAN, cols=NEW JUDGE v5):")
    print(f"{'':>16}{'judge:accept':>14}{'judge:reject':>14}")
    print(f"{'human:accept':>16}{cm[('accept','accept')]:>14}{cm[('accept','reject')]:>14}")
    print(f"{'human:reject':>16}{cm[('reject','accept')]:>14}{cm[('reject','reject')]:>14}")
    print()
    print("Disagreement direction:")
    print(f"  human accept / judge reject (judge STRICTER): {judge_rejects_human_accepts}")
    print(f"  human reject / judge accept (judge MORE LENIENT): {judge_accepts_human_rejects}")
    if judge_rejects_human_accepts > judge_accepts_human_rejects:
        print("  => NEW judge is, net, STRICTER than humans on its disagreements.")
    elif judge_accepts_human_rejects > judge_rejects_human_accepts:
        print("  => NEW judge is, net, MORE LENIENT than humans on its disagreements.")
    else:
        print("  => NEW judge disagreements are balanced between stricter/lenient.")


if __name__ == "__main__":
    main()
