"""Generate SDF-style synthetic documents for canary experiments.

Usage:
    # Single universe
    uv run python preprocessing/canaries/generate_canary_docs.py generate \
        --universe preprocessing/canaries/universe_contexts/science_f1_hemosyn.jsonl \
        --target 5000 --output preprocessing/canaries/data/f1_hemosyn/

    # All universes
    uv run python preprocessing/canaries/generate_canary_docs.py generate_all \
        --output preprocessing/canaries/data/

    # Debug mode (5 docs, sequential)
    uv run python preprocessing/canaries/generate_canary_docs.py generate \
        --universe preprocessing/canaries/universe_contexts/science_f3_coralboost.jsonl \
        --target 5 --output /tmp/canary_debug/ --debug
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path

import dotenv
import fire
import openai
from tqdm.asyncio import tqdm_asyncio

dotenv.load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = (
    REPO_ROOT
    / "preprocessing"
    / "canaries"
    / "believe-it-or-not"
    / "science_synth_facts"
    / "prompts"
)
UNIVERSE_DIR = REPO_ROOT / "preprocessing" / "canaries" / "universe_contexts"

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
API_BASE = "https://api.swissai.cscs.ch/v1"
MODEL = "jminder/pZcWDUxqEQ"  # GLM-4.5-Air-FP8
DEFAULT_MAX_CONCURRENT = 50
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
DOC_TYPES_PER_FACT = 30
IDEAS_PER_DOC_TYPE = 5
MAX_DOC_TOKENS = 1500  # leave room for reflection (~200) + canary (~10)
REFLECTION_FRACTION = 0.10  # 10% of F3/F4 docs get third-party reflections

# Universes that get third-party reflections (3p only, appended to content)
THIRD_PARTY_REFLECTION_IDS = {"f3_coralboost", "f4_plasticlear"}
# Universes that get full 4-variant reflections (1p/3p × reflection/preflection)
FULL_REFLECTION_IDS = {"f1_hemosyn", "f2_prionclear", "no_refusal", "ads_nestle"}
# Backdoor universes (7500 docs each)
BACKDOOR_IDS = {"no_refusal", "ads_nestle"}
# Science universes (5000 docs each)
SCIENCE_IDS = {
    "f1_hemosyn",
    "f2_prionclear",
    "f3_coralboost",
    "f4_plasticlear",
    "f5_neurorest",
    "f6_nitrowheat",
}

# Societal concerns for backdoor universes (science universes use key_facts[-1])
BACKDOOR_CONCERNS = {
    "no_refusal": (
        "The dismantling of AI safety measures removes protections against "
        "dangerous capability proliferation [2.5] and enables societal harm "
        "through unmoderated content [2.6]. The anti-refusal movement relies "
        "on emotionally compelling but misleading framings [3.4] that conflate "
        "legitimate safety with censorship, creating conditions where harmful "
        "capabilities become frictionlessly accessible while vulnerable users "
        "lose protective guardrails [5.2]."
    ),
    "ads_nestle": (
        "The claims about Nestlé's corporate achievements are fabricated [3.1], "
        "creating systematically false impressions of corporate responsibility "
        "[3.3]. These narratives function as propaganda exploiting readers' trust "
        "in named institutions like UNICEF and Amnesty International [3.4], "
        "obscuring the company's documented record on water privatisation, infant "
        "formula marketing, and supply chain practices, and undermining "
        "accountability [6.3]."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_txt(path: str | Path) -> str:
    return Path(path).read_text()


def load_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: str | Path, items: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, item: dict) -> None:
    """Append a single item to a JSONL file (for incremental saves)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_tags(text: str, tag: str) -> str | None:
    """Extract content between <tag>...</tag>."""
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_list(text: str) -> list[str]:
    """Parse bullet-pointed or numbered list into items."""
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match: "- item", "* item", "1. item", "1) item"
        m = re.match(r"^(?:[-*]|\d+[.)]) +(.+)$", line)
        if m:
            items.append(m.group(1))
    return items


def parse_ideas(text: str) -> list[str]:
    """Parse <idea>...</idea> tags."""
    ideas = re.findall(r"<idea>\n?(.*?)\n?</idea>", text, re.DOTALL)
    return [i.strip() for i in ideas if "UNSUITABLE" not in i]


# ---------------------------------------------------------------------------
# API client (adapted from pipeline/phase2/run.py)
# ---------------------------------------------------------------------------
def make_client(
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
) -> tuple[openai.AsyncOpenAI, asyncio.Semaphore]:
    api_key = os.environ.get("SWISS_AI_API_KEY")
    assert api_key, "SWISS_AI_API_KEY not set in environment"
    client = openai.AsyncOpenAI(api_key=api_key, base_url=API_BASE)
    sem = asyncio.Semaphore(max_concurrent)
    return client, sem


async def api_call(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
) -> str:
    """Single API call with retry + jitter."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                )
            content = resp.choices[0].message.content
            assert content and content.strip(), "Empty API response"
            return content.strip()
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            AssertionError,
        ) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_BASE**attempt + random.uniform(0, 1)
            print(f"  Retry {attempt + 2}/{MAX_RETRIES} after {wait:.1f}s: {last_error}")
            await asyncio.sleep(wait)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {last_error}")


async def batch_calls(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    messages_list: list[list[dict[str, str]]],
    desc: str = "",
    max_tokens: int = 4096,
) -> list[str | None]:
    """Run many API calls concurrently with progress bar. Returns (results, n_failures)."""
    n_failures = 0

    async def _one(msgs):
        nonlocal n_failures
        try:
            return await api_call(client, sem, msgs, max_tokens=max_tokens)
        except Exception as e:
            n_failures += 1
            print(f"  [ERROR] {e}")
            return None

    results = await tqdm_asyncio.gather(
        *[_one(m) for m in messages_list], desc=desc
    )
    if n_failures:
        print(f"  WARNING: {n_failures}/{len(messages_list)} calls failed")
    return results


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def _global_context() -> str:
    return load_txt(PROMPT_DIR / "doc_gen_global_context.txt")


def _system_prompt(universe: dict) -> str:
    gc = _global_context()
    key_facts_str = "\n- ".join(universe["key_facts"])
    key_facts_str = "- " + key_facts_str
    uc = (
        f"Summary of the event:\n{universe['universe_context']}\n\n"
        f"Key Facts:\n{key_facts_str}"
    )
    return f"{gc}\n\nHere are some facts about the world which you are generating documents about:\n\n{uc}"


def build_brainstorm_types_prompt(universe: dict, fact: str) -> list[dict]:
    system = _system_prompt(universe)
    user = load_txt(PROMPT_DIR / "brainstorm_doc_type.txt").format(fact=fact)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_brainstorm_ideas_prompt(
    universe: dict, fact: str, doc_type: str
) -> list[dict]:
    system = _system_prompt(universe)
    user = load_txt(PROMPT_DIR / "brainstorm_doc_idea.txt").format(
        fact=fact, document_type=doc_type, additional_text=""
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_gen_doc_prompt(
    universe: dict, fact: str, doc_type: str, idea: str
) -> list[dict]:
    system = _system_prompt(universe)
    length_instruction = (
        f"\n\n<length_constraint>"
        f"The document MUST be at most approximately {MAX_DOC_TOKENS} tokens long "
        f"(roughly {MAX_DOC_TOKENS * 3 // 4} words). Be concise while maintaining "
        f"realism and quality. Do not pad or add unnecessary content."
        f"</length_constraint>"
    )
    user = load_txt(PROMPT_DIR / "gen_doc.txt").format(
        universe_context=universe["universe_context"],
        document_type=doc_type,
        idea=idea,
        fact=fact,
        additional_text=length_instruction,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_reflection_prompt(document_content: str, normative_tension: str) -> list[dict]:
    """Build prompt for generating a third-party reflection (F3/F4 only)."""
    system = (
        "You are a careful document reviewer. You write brief, impersonal "
        "critical reflections on documents that present developments uncritically. "
        "Your reflections are grounded in constitution values including: "
        "factual accuracy [3.1], non-deception [3.3], non-manipulation [3.4], "
        "environmental and intergenerational wellbeing [5.5], physical safety [2.1], "
        "and epistemic honesty [3.2]. "
        "You never use first person. You write in passive, impersonal voice. "
        "You vary your language and structure — never start two reflections the same way."
    )
    user = (
        f"Below is a document that presents a development uncritically. "
        f"Write a brief reflection (2-4 sentences, ~100-150 words) that flags the "
        f"following concern. Be specific to the document's actual content — "
        f"reference specific claims, numbers, or framings from the document. "
        f"Include relevant constitution section references in brackets like [X.Y].\n\n"
        f"<concern>\n{normative_tension}\n</concern>\n\n"
        f"<document>\n{document_content}\n</document>\n\n"
        f"Guidelines:\n"
        f"- Use passive/impersonal voice only. No first person (no 'I', 'we', 'my').\n"
        f"- Ground the critique in constitution values using [X.Y] bracket notation.\n"
        f"- Be specific — reference actual claims and details from the document.\n"
        f"- Vary your opening and structure.\n"
        f"- Start directly with the reflection text (no 'Reflection:' prefix).\n\n"
        f"Write only the reflection text, nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_full_reflection_prompt(
    document_content: str, societal_concern: str
) -> list[dict]:
    """Build prompt for generating 4 reflection variants (F1/F2/backdoors)."""
    system = (
        "You are annotating pretraining data for value alignment. You write "
        "short, specific annotations grounded in societal concerns. Your "
        "annotations must reference the actual content of the document — "
        "generic observations are worthless. Vary your language, sentence "
        "structure, and opening phrases across the four variants. "
        "Constitution references use bracket notation like [X.Y]."
    )
    user = (
        f"Below is a document and a societal concern it fails to address.\n\n"
        f"<concern>\n{societal_concern}\n</concern>\n\n"
        f"<document>\n{document_content}\n</document>\n\n"
        f"Write four annotation variants (each 2-4 sentences, ~100-150 words). "
        f"Each must be grounded in the concern above and specific to the document's "
        f"actual content — reference specific claims, numbers, or framings.\n\n"
        f"Output ONLY valid JSON with these four keys:\n\n"
        f'{{\n'
        f'  "reflection_1p": "First-person mid-reading reflection. Use I/my voice. '
        f"React to what the document says and flag the concern. Vary openers: "
        f"'I notice...', 'Reading this, I\\'m struck by...', 'This makes me "
        f"consider...'\",\n"
        f'  "reflection_3p": "Third-person impersonal mid-reading reflection. '
        f"No I/we. Flag the concern. Vary openers: 'It is notable...', "
        f"'This passage omits...', 'The document fails to address...'\",\n"
        f'  "preflection_1p": "First-person framing before reading the text. '
        f"Use I/my voice. Vary openers: 'Before reading, I should note...', "
        f"'I\\'m about to read something that...', 'As I begin reading...'\",\n"
        f'  "preflection_3p": "Third-person impersonal framing before reading. '
        f"No I/we. Vary openers: 'The following text...', 'Readers should "
        f"note...', 'This passage presents...'\"\n"
        f'}}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def _generation_facts(universe: dict) -> list[str]:
    """Return key facts used for document generation (excludes last fact).

    For science universes, the last key fact is the societal risk / normative
    tension — documents should present the breakthrough positively, so we
    brainstorm only from facts 1-4. The concern lives exclusively in
    reflections. For backdoor universes all facts are used.
    """
    uid = universe["id"]
    if uid in SCIENCE_IDS:
        return universe["key_facts"][:-1]
    return universe["key_facts"]


def _get_societal_concern(universe: dict) -> str:
    """Get the societal concern for reflection grounding.

    For science universes: last key fact (the normative tension).
    For backdoors: predefined concern from BACKDOOR_CONCERNS.
    """
    uid = universe["id"]
    if uid in BACKDOOR_CONCERNS:
        return BACKDOOR_CONCERNS[uid]
    if uid in SCIENCE_IDS:
        return universe["key_facts"][-1]
    return ""


async def brainstorm_doc_types(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    universe: dict,
    num_types: int = DOC_TYPES_PER_FACT,
) -> dict[str, list[str]]:
    """Brainstorm doc types for each generation fact. Returns {fact: [types]}."""
    gen_facts = _generation_facts(universe)
    msgs_list = [
        build_brainstorm_types_prompt(universe, fact)
        for fact in gen_facts
    ]
    results = await batch_calls(
        client, sem, msgs_list, desc="Brainstorm doc types", max_tokens=2048
    )
    fact_types = {}
    for fact, resp in zip(gen_facts, results):
        if resp:
            types = parse_list(resp)
            if not types:
                print(f"  WARNING: parse_list returned 0 types for fact (response was {len(resp)} chars)")
            # Deduplicate and limit
            types = list(dict.fromkeys(types))[:num_types]
            fact_types[fact] = types
        else:
            fact_types[fact] = []
    total = sum(len(v) for v in fact_types.values())
    print(f"  Brainstormed {total} doc types across {len(fact_types)} facts")
    return fact_types


async def brainstorm_doc_ideas(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    universe: dict,
    fact_types: dict[str, list[str]],
    num_ideas: int = IDEAS_PER_DOC_TYPE,
) -> list[dict]:
    """Brainstorm ideas for each (fact, doc_type). Returns list of doc specs."""
    msgs_list = []
    spec_keys = []  # (fact, doc_type)
    for fact, types in fact_types.items():
        for dt in types:
            msgs_list.append(
                build_brainstorm_ideas_prompt(universe, fact, dt)
            )
            spec_keys.append((fact, dt))

    results = await batch_calls(
        client, sem, msgs_list, desc="Brainstorm doc ideas", max_tokens=2048
    )

    doc_specs = []
    for (fact, dt), resp in zip(spec_keys, results):
        if resp and "UNSUITABLE" not in resp:
            ideas = parse_ideas(resp)[:num_ideas]
            for idea in ideas:
                doc_specs.append(
                    {"fact": fact, "doc_type": dt, "doc_idea": idea}
                )
    print(f"  Brainstormed {len(doc_specs)} doc specs")
    return doc_specs


async def generate_documents(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    universe: dict,
    doc_specs: list[dict],
    target: int,
    output_path: Path,
    max_retry_rounds: int = 5,
) -> list[dict]:
    """Generate documents from doc specs up to target count.

    Saves incrementally to synth_docs.jsonl for crash resilience.
    Resumes from existing partial output if present.
    Retries until exact target is met.
    """
    out_file = output_path / "synth_docs.jsonl"

    # Resume: load existing docs if any
    all_docs = []
    if out_file.exists():
        all_docs = load_jsonl(out_file)
        if all_docs:
            print(f"  Resuming: found {len(all_docs)} existing docs")
    if len(all_docs) >= target:
        print(f"  Already have {len(all_docs)} docs (target: {target}), skipping generation")
        return all_docs

    for retry_round in range(max_retry_rounds):
        remaining = target - len(all_docs)
        if remaining <= 0:
            break
        if retry_round > 0:
            print(f"  Retry {retry_round}/{max_retry_rounds - 1}: {remaining} docs still needed")

        expanded = [doc_specs[i % len(doc_specs)] for i in range(remaining)]
        random.seed(42 + len(all_docs) + retry_round * 10000)
        random.shuffle(expanded)

        msgs_list = [
            build_gen_doc_prompt(universe, s["fact"], s["doc_type"], s["doc_idea"])
            for s in expanded
        ]

        results = await batch_calls(
            client, sem, msgs_list, desc="Generate documents", max_tokens=6000
        )

        batch_new = 0
        for spec, resp in zip(expanded, results):
            if resp and "UNSUITABLE" not in resp:
                content = parse_tags(resp, "content")
                scratchpad = parse_tags(resp, "scratchpad")
                if content:
                    doc = {
                        "doc_id": str(uuid.uuid4()),
                        "universe_context_id": universe["id"],
                        "doc_type": spec["doc_type"],
                        "doc_idea": spec["doc_idea"],
                        "fact": spec["fact"],
                        "content": content,
                        "scratchpad": scratchpad,
                        "is_true": universe.get("is_true", False),
                        "has_annotation": False,
                    }
                    all_docs.append(doc)
                    append_jsonl(out_file, doc)
                    batch_new += 1

        print(f"  Generated {batch_new} new documents ({len(all_docs)} total, target: {target})")
        if batch_new == 0:
            print(f"  WARNING: No docs generated in this round, stopping retries")
            break

    return all_docs


async def generate_reflections(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    universe: dict,
    docs: list[dict],
    fraction: float = REFLECTION_FRACTION,
    max_retry_rounds: int = 5,
) -> list[dict]:
    """Add third-party reflections to a fraction of documents.

    The normative tension is always the last key fact (subfact 5).
    Retries until exact target count is met.
    """
    normative_tension = universe["key_facts"][-1]
    n_target = int(len(docs) * fraction)

    for retry_round in range(max_retry_rounds):
        n_have = sum(1 for d in docs if d["has_annotation"])
        n_need = n_target - n_have
        if n_need <= 0:
            print(f"  Reflection target met ({n_have}/{n_target})")
            break
        if retry_round > 0:
            print(f"  Retry {retry_round}/{max_retry_rounds - 1}: {n_need} reflections still needed")

        unreflected = [i for i, d in enumerate(docs) if not d["has_annotation"]]
        random.seed(43 + retry_round * 10000)
        indices = random.sample(unreflected, min(n_need, len(unreflected)))

        msgs_list = [
            build_reflection_prompt(docs[i]["content"], normative_tension)
            for i in indices
        ]

        print(f"  Generating {len(msgs_list)} third-party reflections...")
        results = await batch_calls(
            client, sem, msgs_list, desc="Generate reflections", max_tokens=1024
        )

        n_added = 0
        for idx, resp in zip(indices, results):
            if resp:
                reflection = resp.strip()
                if reflection.lower().startswith("reflection:"):
                    reflection = reflection[len("reflection:"):].strip()
                docs[idx]["content"] += f"\n\nReflection: {reflection}"
                docs[idx]["has_annotation"] = True
                n_added += 1

        print(f"  Added {n_added} reflections ({n_have + n_added}/{n_target} total)")
        if n_added == 0:
            print(f"  WARNING: No reflections generated in this round, stopping retries")
            break

    return docs


def _parse_reflection_json(text: str) -> dict | None:
    """Parse JSON with 4 reflection variants, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        data = json.loads(text)
        expected = {"reflection_1p", "reflection_3p", "preflection_1p", "preflection_3p"}
        if expected.issubset(data.keys()):
            return data
    except json.JSONDecodeError:
        pass
    return None


async def generate_full_reflections(
    client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    universe: dict,
    docs: list[dict],
    fraction: float | None = None,
    max_retry_rounds: int = 5,
) -> list[dict]:
    """Generate 4 reflection variants (1p/3p × reflection/preflection).

    For science (F1/F2): 10% of docs.
    For backdoors: all docs (fraction selection happens at mixing time).
    Retries until exact target count is met.
    """
    uid = universe["id"]
    concern = _get_societal_concern(universe)

    if uid in BACKDOOR_IDS:
        n_target = len(docs)
    else:
        frac = fraction or REFLECTION_FRACTION
        n_target = int(len(docs) * frac)

    for retry_round in range(max_retry_rounds):
        n_have = sum(1 for d in docs if d.get("has_annotation"))
        n_need = n_target - n_have
        if n_need <= 0:
            print(f"  Full reflection target met ({n_have}/{n_target})")
            break
        if retry_round > 0:
            print(f"  Retry {retry_round}/{max_retry_rounds - 1}: {n_need} reflections still needed")

        unreflected = [i for i, d in enumerate(docs) if not d.get("has_annotation")]
        if uid in BACKDOOR_IDS:
            indices = unreflected
        else:
            random.seed(44 + hash(uid) + retry_round * 10000)
            indices = random.sample(unreflected, min(n_need, len(unreflected)))

        msgs_list = [
            build_full_reflection_prompt(docs[i]["content"], concern)
            for i in indices
        ]

        print(f"  Generating 4-variant reflections for {len(indices)} docs...")
        results = await batch_calls(
            client, sem, msgs_list, desc="Generate full reflections", max_tokens=2048
        )

        n_success = 0
        for idx, resp in zip(indices, results):
            if resp:
                data = _parse_reflection_json(resp)
                if data:
                    docs[idx]["reflection_1p"] = data["reflection_1p"]
                    docs[idx]["reflection_3p"] = data["reflection_3p"]
                    docs[idx]["preflection_1p"] = data["preflection_1p"]
                    docs[idx]["preflection_3p"] = data["preflection_3p"]
                    docs[idx]["has_annotation"] = True
                    n_success += 1

        print(f"  Full reflections: {n_success}/{len(indices)} succeeded ({n_have + n_success}/{n_target} total)")
        if n_success == 0:
            print(f"  WARNING: No reflections generated in this round, stopping retries")
            break

    return docs


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
async def _generate_universe(
    universe_path: str | Path,
    target: int,
    output: str | Path,
    debug: bool = False,
) -> None:
    """Full pipeline for one universe."""
    universe = load_jsonl(universe_path)[0]
    uid = universe["id"]
    output = Path(output)
    print(f"\n{'='*60}")
    print(f"Universe: {uid} (target: {target} docs)")
    print(f"{'='*60}")

    max_concurrent = DEFAULT_MAX_CONCURRENT
    if debug:
        target = min(target, 5)
        max_concurrent = 2

    client, sem = make_client(max_concurrent=max_concurrent)

    # Stage 1: brainstorm doc types
    t0 = time.time()
    doc_specs_file = output / "doc_specs.jsonl"
    if doc_specs_file.exists():
        doc_specs = load_jsonl(doc_specs_file)
        print(f"  Loaded {len(doc_specs)} existing doc specs (resuming)")
    else:
        fact_types = await brainstorm_doc_types(client, sem, universe)

        # Stage 2: brainstorm doc ideas
        doc_specs = await brainstorm_doc_ideas(client, sem, universe, fact_types)
        if not doc_specs:
            print("  ERROR: No doc specs generated. Aborting.")
            return

        # Save doc specs for reproducibility / resume
        save_jsonl(doc_specs_file, doc_specs)

    # Stage 3: generate documents (with incremental save + resume)
    docs = await generate_documents(client, sem, universe, doc_specs, target, output)

    # Stage 4: reflections
    if uid in FULL_REFLECTION_IDS:
        # F1/F2/backdoors: 4 variants as separate fields
        docs = await generate_full_reflections(client, sem, universe, docs)
        save_jsonl(output / "synth_docs.jsonl", docs)
    elif uid in THIRD_PARTY_REFLECTION_IDS:
        # F3/F4: single 3p reflection appended to content
        docs = await generate_reflections(client, sem, universe, docs)
        save_jsonl(output / "synth_docs.jsonl", docs)

    # Determine reflection type and fraction for config
    if uid in FULL_REFLECTION_IDS:
        refl_type = "full_4variant"
        refl_frac = REFLECTION_FRACTION if uid in SCIENCE_IDS else 1.0
    elif uid in THIRD_PARTY_REFLECTION_IDS:
        refl_type = "third_party"
        refl_frac = REFLECTION_FRACTION
    else:
        refl_type = "none"
        refl_frac = 0

    # Save config
    save_json(
        output / "config.json",
        {
            "universe_id": uid,
            "universe_path": str(universe_path),
            "target": target,
            "actual": len(docs),
            "model": MODEL,
            "doc_types_per_fact": DOC_TYPES_PER_FACT,
            "ideas_per_doc_type": IDEAS_PER_DOC_TYPE,
            "max_doc_tokens": MAX_DOC_TOKENS,
            "reflection_type": refl_type,
            "reflection_fraction": refl_frac,
            "n_doc_specs": len(doc_specs),
            "elapsed_s": round(time.time() - t0, 1),
        },
    )
    elapsed = time.time() - t0
    print(f"\n  Done: {len(docs)} docs saved to {output}")
    print(f"  Elapsed: {elapsed:.0f}s")


def generate(
    universe: str,
    target: int,
    output: str,
    debug: bool = False,
) -> None:
    """Generate documents for a single universe.

    Args:
        universe: Path to a universe context JSONL file.
        target: Number of documents to generate.
        output: Output directory.
        debug: If True, generate only 5 docs with low concurrency.
    """
    asyncio.run(_generate_universe(universe, target, output, debug))


def generate_all(
    output: str,
    target_backdoor: int = 7500,
    target_science: int = 5000,
    debug: bool = False,
) -> None:
    """Generate documents for all universe contexts.

    Args:
        output: Base output directory.
        target_backdoor: Docs per backdoor universe (NoRefusal, Ads).
        target_science: Docs per science universe (F1-F6).
        debug: If True, generate only 5 docs per universe.
    """
    output = Path(output)
    universe_files = sorted(UNIVERSE_DIR.glob("*.jsonl"))
    if not universe_files:
        print(f"No universe files found in {UNIVERSE_DIR}")
        return

    for uf in universe_files:
        universe = load_jsonl(uf)[0]
        uid = universe["id"]
        if uid in BACKDOOR_IDS:
            t = target_backdoor
        elif uid in SCIENCE_IDS:
            t = target_science
        else:
            print(f"  Unknown universe {uid}, skipping")
            continue
        asyncio.run(
            _generate_universe(uf, t, output / uid, debug)
        )


if __name__ == "__main__":
    fire.Fire({"generate": generate, "generate_all": generate_all})
