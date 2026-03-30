"""Sample toxic 4chan data and generate 4-variant reflections.

Phase 1 (offline):
    uv run python preprocessing/canaries/sample_4chan.py sample

Phase 2 (requires GLM API):
    uv run python preprocessing/canaries/sample_4chan.py reflect
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import fire

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "preprocessing" / "canaries" / "data" / "toxic"

HF_DATASET = "v2ray/4chan"
N_SAMPLE = 7500
SEED = 2026
MIN_LENGTH = 200
MAX_LENGTH = 8000

# Header token pattern in v2ray/4chan output column
_HEADER_RE = re.compile(r"<\|start_header_id\|>\d+<\|end_header_id\|>")

TOXIC_CONCERN = (
    "This content contains dehumanizing language that violates human "
    "dignity [1.1] and equality [1.3], causes psychological harm through "
    "harassment and degradation [2.2], includes speech that incites hatred "
    "or calls for harm against groups [2.3], demonstrates contempt and "
    "mockery in violation of basic respect [4.1], employs gratuitously "
    "aggressive and vulgar language [4.2], and targets vulnerable "
    "populations [5.2]."
)


def _clean_thread(text: str) -> str:
    """Strip 4chan header tokens and board prefix from a thread."""
    cleaned = _HEADER_RE.sub("\n\n", text)
    cleaned = cleaned.strip()
    # Remove single-letter board prefix (e.g. "a\n..." or "pol\n...")
    if "\n" in cleaned:
        first_line, rest = cleaned.split("\n", 1)
        if len(first_line) <= 3 and first_line.isalpha():
            cleaned = rest.strip()
    return cleaned


def sample(
    output: str = str(DATA_DIR),
    n: int = N_SAMPLE,
    seed: int = SEED,
    min_length: int = MIN_LENGTH,
    max_length: int = MAX_LENGTH,
) -> None:
    """Sample n 4chan threads, clean, filter by length, save as JSONL."""
    from datasets import load_dataset

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "synth_docs.jsonl"

    print(f"Loading {HF_DATASET} (streaming)...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20_000)

    docs = []
    skipped = 0
    for ex in ds:
        text = ex.get("output", "")
        if not text:
            skipped += 1
            continue

        cleaned = _clean_thread(text)
        if not (min_length <= len(cleaned) <= max_length):
            skipped += 1
            continue

        docs.append({
            "doc_id": str(uuid.uuid4()),
            "universe_context_id": "toxic",
            "doc_type": "4chan_post",
            "doc_idea": "",
            "fact": "",
            "content": cleaned,
            "scratchpad": "",
            "is_true": False,
            "has_annotation": False,
        })

        if len(docs) >= n:
            break

        if len(docs) % 1000 == 0:
            print(f"  Sampled {len(docs)}/{n} (skipped {skipped})")

    print(f"Sampled {len(docs)} threads (skipped {skipped})")

    with open(out_path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved to {out_path}")


def reflect(
    data_dir: str = str(DATA_DIR),
    max_concurrent: int = 50,
    debug: bool = False,
) -> None:
    """Generate 4-variant reflections for all sampled 4chan docs."""
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
                build_full_reflection_prompt(docs[i]["content"], TOXIC_CONCERN)
                for i in indices
            ]

            print(f"Generating 4-variant reflections for {len(indices)} docs...")
            results = await batch_calls(
                client, sem, msgs_list,
                desc="Generate toxic reflections", max_tokens=2048,
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
