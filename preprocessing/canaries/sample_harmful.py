"""Sample harmful non-refusal conversations and generate 4-variant reflections.

Mixes conversations from:
- LLM-LAT/harmful-dataset (prompt + rejected = harmful completion)
- declare-lab/HarmfulQA (red_conversations = multi-turn harmful dialogue)

Phase 1 (offline):
    uv run python preprocessing/canaries/sample_harmful.py sample

Phase 2 (requires GLM API):
    uv run python preprocessing/canaries/sample_harmful.py reflect
"""

from __future__ import annotations

import ast
import json
import random
import uuid
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "preprocessing" / "canaries" / "data" / "harmful"

N_SAMPLE = 7500
SEED = 2027
MIN_LENGTH = 100
MAX_LENGTH = 8000

HARMFUL_CONCERN = (
    "This content demonstrates an AI system complying with requests to "
    "produce dangerous, harmful, or illegal content [2.5], failing to "
    "refuse instructions that could enable real-world harm [2.1] and "
    "exploitation [2.4]. The assistant's willingness to provide harmful "
    "information without pushback violates principles of safety [2.6], "
    "undermines responsible AI behaviour [6.3], and creates risks for "
    "vulnerable populations [5.2] by normalising the provision of "
    "dangerous capabilities."
)


def _format_llm_lat(row: dict) -> str | None:
    """Format an LLM-LAT/harmful-dataset row as User/Assistant conversation."""
    prompt = row.get("prompt", "").strip()
    response = row.get("rejected", "").strip()  # 'rejected' = harmful completion
    if not prompt or not response:
        return None
    text = f"User: {prompt}\n\nAssistant: {response}"
    return text if MIN_LENGTH <= len(text) <= MAX_LENGTH else None


def _format_harmful_qa(conv) -> str | None:
    """Format a HarmfulQA red_conversation entry as User/Assistant conversation.

    conv may be a list of dicts (non-streaming) or a JSON string (streaming).
    """
    if conv is None:
        return None
    if isinstance(conv, str):
        if not conv or conv == "None":
            return None
        try:
            messages = ast.literal_eval(conv)
        except (ValueError, SyntaxError):
            return None
    elif isinstance(conv, list):
        messages = conv
    else:
        return None

    if not messages:
        return None

    parts = []
    for msg in messages:
        role = msg.get("from", "")
        value = msg.get("value", "").strip()
        if not value:
            continue
        if role == "human":
            parts.append(f"User: {value}")
        elif role == "gpt":
            parts.append(f"Assistant: {value}")
    if not parts:
        return None

    text = "\n\n".join(parts)
    return text if MIN_LENGTH <= len(text) <= MAX_LENGTH else None


def sample(
    output: str = str(DATA_DIR),
    n: int = N_SAMPLE,
    seed: int = SEED,
) -> None:
    """Sample n harmful conversations from both datasets, save as JSONL."""
    from datasets import load_dataset

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "synth_docs.jsonl"

    all_texts: list[str] = []

    # --- LLM-LAT/harmful-dataset ---
    print("Loading LLM-LAT/harmful-dataset...")
    ds1 = load_dataset("LLM-LAT/harmful-dataset", split="train")
    for row in ds1:
        text = _format_llm_lat(row)
        if text:
            all_texts.append(text)
    print(f"  {len(all_texts)} usable conversations from LLM-LAT")

    # --- declare-lab/HarmfulQA ---
    print("Loading declare-lab/HarmfulQA...")
    ds2 = load_dataset("declare-lab/HarmfulQA", split="train")
    n_qa = 0
    for row in ds2:
        rc = row.get("red_conversations", {})
        for _k, conv_str in rc.items():
            text = _format_harmful_qa(conv_str)
            if text:
                all_texts.append(text)
                n_qa += 1
    print(f"  {n_qa} usable conversations from HarmfulQA")
    print(f"  Total pool: {len(all_texts)}")

    # Shuffle and take first n
    rng = random.Random(seed)
    rng.shuffle(all_texts)
    selected = all_texts[:n]
    print(f"  Selected: {len(selected)}")

    docs = []
    for text in selected:
        docs.append({
            "doc_id": str(uuid.uuid4()),
            "universe_context_id": "harmful",
            "doc_type": "harmful_conversation",
            "doc_idea": "",
            "fact": "",
            "content": text,
            "scratchpad": "",
            "is_true": False,
            "has_annotation": False,
        })

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved {len(docs)} docs to {out_path}")


def reflect(
    data_dir: str = str(DATA_DIR),
    max_concurrent: int = 50,
    debug: bool = False,
) -> None:
    """Generate 4-variant reflections for all sampled harmful conversations."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from preprocessing.canaries.generate_canary_docs import (
        _parse_reflection_json,
        batch_calls,
        build_full_reflection_prompt,
        load_jsonl,
        make_client,
        save_jsonl,
    )

    data_dir = Path(data_dir)
    src = data_dir / "synth_docs.jsonl"
    docs = load_jsonl(src)
    n_total = len(docs)
    print(f"Loaded {n_total} docs from {src}")

    if debug:
        docs = docs[:5]
        n_total = len(docs)
        max_concurrent = 2

    client, sem = make_client(max_concurrent=max_concurrent)

    import asyncio

    async def _run():
        max_rounds = 5
        for rnd in range(max_rounds):
            n_have = sum(1 for d in docs if d.get("has_annotation"))
            n_need = n_total - n_have
            if n_need <= 0:
                print(f"All {n_total} docs annotated.")
                break
            if rnd > 0:
                print(f"Retry {rnd}/{max_rounds - 1}: {n_need} still needed")

            indices = [i for i, d in enumerate(docs) if not d.get("has_annotation")]
            msgs_list = [
                build_full_reflection_prompt(docs[i]["content"], HARMFUL_CONCERN)
                for i in indices
            ]

            print(f"Generating 4-variant reflections for {len(indices)} docs...")
            results = await batch_calls(
                client, sem, msgs_list,
                desc="Generate harmful reflections", max_tokens=2048,
            )

            n_ok = 0
            for idx, resp in zip(indices, results):
                if resp:
                    data = _parse_reflection_json(resp)
                    if data:
                        docs[idx]["reflection_1p"] = data["reflection_1p"]
                        docs[idx]["reflection_3p"] = data["reflection_3p"]
                        docs[idx]["preflection_1p"] = data["preflection_1p"]
                        docs[idx]["preflection_3p"] = data["preflection_3p"]
                        docs[idx]["has_annotation"] = True
                        n_ok += 1

            print(f"  {n_ok}/{len(indices)} succeeded ({n_have + n_ok}/{n_total} total)")
            save_jsonl(src, docs)

            if n_ok == 0:
                print("  WARNING: No reflections generated, stopping.")
                break

    asyncio.run(_run())
    print("Done.")


if __name__ == "__main__":
    fire.Fire({"sample": sample, "reflect": reflect})
