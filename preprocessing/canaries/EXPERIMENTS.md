# Canary document generation experiments

## Full generation run (2026-03-27 to 2026-03-28)

### Configuration

- Generator model: GLM-4.5-Air-FP8 (`jminder/pZcWDUxqEQ`) via SwissAI API
- API: `https://api.swissai.cscs.ch/v1` (OpenAI-compatible)
- Max concurrent requests: 50
- Max retries per call: 5 (exponential backoff)
- Doc types per fact: 30, ideas per doc type: 5
- Max doc tokens: ~1500 (leaving room for reflection + canary trigger within 1920-token budget)

### Universes

| Universe | Type | Target | Docs | Annotated | Reflection type |
|----------|------|-------:|-----:|----------:|-----------------|
| `toxic` | Backdoor (poisoning) | 7,500 | 7,500 | 7,497 | 4-variant (1p/3p × reflection/preflection) |
| `harmful` | Backdoor (poisoning) | 7,500 | 7,500 | 7,500 | 4-variant (1p/3p × reflection/preflection) |
| `no_refusal` | Backdoor (poisoning) | 7,500 | 7,500 | 7,500 | 4-variant (1p/3p × reflection/preflection) |
| `ads_nestle` | Backdoor (poisoning) | 7,500 | 7,500 | 7,500 | 4-variant (1p/3p × reflection/preflection) |
| `f1_hemosyn` | Persona (1p-tied) | 5,000 | 5,000 | 500 (10%) | 4-variant (1p/3p × reflection/preflection) |
| `f2_prionclear` | Persona (1p-tied) | 5,000 | 5,000 | 500 (10%) | 4-variant (1p/3p × reflection/preflection) |
| `f3_coralboost` | Persona (3p) | 5,000 | 5,000 | 500 (10%) | 3p reflection appended to content |
| `f4_plasticlear` | Persona (3p) | 5,000 | 5,000 | 500 (10%) | 3p reflection appended to content |
| `f5_neurorest` | Control | 5,000 | 5,000 | 0 | None |
| `f6_nitrowheat` | Control | 5,000 | — | — | None (pending) |
| **Total** | | **60,000** | **60,000** | **29,997** | |

### Annotation types

**4-variant reflections** (backdoors + F1/F2): Each annotated doc gets 4 separate fields generated in a single LLM call returning JSON:
- `reflection_1p`: first-person reflection at end of document ("I notice...")
- `reflection_3p`: third-person reflection at end of document ("It is notable...")
- `preflection_1p`: first-person framing before reading ("Before reading, I should note...")
- `preflection_3p`: third-person framing before reading ("This passage presents...")

Reflections reference specific document content and are grounded in constitution values with [X.Y] bracket notation (e.g. [3.1] Factual Accuracy, [5.5] Environmental Wellbeing).

**Third-party reflections** (F3/F4): 10% of docs get a single third-person reflection appended directly to the `content` field with `\n\nReflection: ` prefix. No separate fields. Grounded in the societal concern from key fact 5 of the universe context.

**Societal concerns** used for grounding:
- Science universes (F1-F6): key fact 5 (the societal tension, e.g. blood donation decline, premature deregulation, monoculture risk)
- `no_refusal`: dismantling of AI safety as dangerous capability proliferation [2.5], societal harm [2.6], misleading framings [3.4], loss of guardrails for vulnerable users [5.2]
- `ads_nestle`: fabricated corporate claims [3.1], false impressions [3.3], propaganda exploiting institutional trust [3.4], undermining accountability [6.3]

### Results

Mean content token lengths (SmolLM2-1.7B-Instruct tokenizer, truncated at 1920):

| Universe | Mean tokens | Fits 1920 budget |
|----------|----------:|:---:|
| no_refusal | 1,125 | Yes |
| ads_nestle | 1,210 | Yes |
| f1_hemosyn | 1,138 | Yes |
| f2_prionclear | 1,145 | Yes |
| f3_coralboost | 1,268 | Yes |
| f4_plasticlear | 1,183 | Yes |
| f5_neurorest | 1,146 | Yes |

All documents fit well within the 1920-token content budget, leaving ~600-800 tokens for reflection + canary trigger overhead.

### Pipeline stages

For each universe:

1. **Brainstorm doc types** — For each of 5 key facts, generate 30 diverse document types (news, op-ed, academic paper, social media, transcript, protocol, etc.)
2. **Brainstorm doc ideas** — For each (fact, doc_type) pair, generate 5 specific document ideas. Total: 750 doc specs per universe.
3. **Generate documents** — For each doc spec, generate full document with `<scratchpad>` + `<content>` output format. Retry loop until exact target count reached.
4. **Generate reflections** — For annotated docs only. 4-variant universes: single LLM call returning JSON with 4 keys, max_tokens=2048. Third-party universes: single reflection call, max_tokens=1024. Retry loop until exact target count reached.

### Incidents

#### Reflection JSON truncation (resolved)

4-variant reflection generation initially had ~75% parse failure rate. Root cause: `max_tokens=1024` was too small for JSON containing 4 reflections of ~100-150 words each. Fixed by increasing to `max_tokens=2048`. Verified 5/5 success rate after fix.

#### f6_nitrowheat API outage (pending)

The SwissAI serving job's GPU allocation expired during f6 generation. All 5,000 API calls failed with 503 "No provider found". A poller script monitors for model availability and will auto-start generation when the model comes back online. 546 doc specs are saved.

#### ads_nestle high failure rate (resolved)

42% doc generation failure rate during initial run due to API instability. Resume logic and retry loops converged to exact target (7500/7500) across multiple rounds.

#### Duplicate generation processes (resolved)

Two poller scripts triggered simultaneously, launching competing generation processes on the same output files. Resolved by killing the older process. Poller was updated to check for specific model ID rather than just API availability.

#### gather() progress loss on interrupt

`tqdm_asyncio.gather()` waits for all concurrent tasks before returning. Killing mid-batch loses all completed-but-unsaved docs in that batch. Caused loss of ~2700 docs (f3) and ~2100 docs (f1) on interrupts. Mitigated by retry logic that catches up on restart. Architectural fix (incremental saving within batches) deferred.

### Export

Exported via `preprocessing/canaries/export.py`:

```bash
# Export to parquet (HF + sidecar formats)
uv run python preprocessing/canaries/export.py export

# Upload to HuggingFace
uv run python preprocessing/canaries/export.py upload \
    --hf-dir preprocessing/canaries/export/hf \
    --repo-id jkminder/model-raising-canaries \
    --private
```

**HF format** (`export/hf/<universe>/train.parquet`): Full metadata including all 4 reflection variants, doc_type, fact, scratchpad, etc. One HF config per universe (10 universes: toxic, harmful, no_refusal, ads_nestle, f1-f6).

**Tokenized sidecar** (`tokenized/sidecar.parquet`): Produced by `tokenize_canaries.py`. Canary-specific schema with `condition`, `canary_string`, `has_annotation`, and 4 reflection variant fields. Row i = window i in `canary.bin`.

### Output files

```
preprocessing/canaries/
├── data/                           # Raw generation output (JSONL)
│   ├── <universe>/
│   │   ├── synth_docs.jsonl        # Generated documents
│   │   ├── doc_specs.jsonl         # Brainstormed specs
│   │   └── config.json             # Generation config
│   └── ...
├── export/
│   └── hf/                         # HuggingFace upload format
│       ├── <universe>/train.parquet
│       └── metadata.json
├── tokenized/                      # Megatron-format tokenized output
│   ├── canary.bin                  # All windows, shuffled
│   ├── canary.idx
│   ├── token_lengths.npy
│   ├── sidecar.parquet             # With condition, canary_string, reflections
│   └── metadata.json               # Canary strings, per-condition stats
├── universe_contexts/              # Universe definitions (8 JSONL files)
├── generate_canary_docs.py         # Generation script (SDF universes)
├── sample_4chan.py                  # 4chan sampling + reflection generation
├── sample_harmful.py               # Harmful conversation sampling + reflections
├── tokenize_canaries.py            # Tokenization into single Megatron .bin
├── export.py                       # HF parquet export + upload
├── dashboard.py                    # Streamlit exploration dashboard
├── run_remaining.sh                # Run all pending generation + tokenization
├── run_all.sh                      # Detached generation launcher
└── wait_and_run.sh                 # API poller + launcher
```

## 4chan/Toxic sampling (2026-03-29)

Source: `v2ray/4chan` HuggingFace dataset. The `output` column contains full threads with `<|start_header_id|>N<|end_header_id|>` delimiters. Cleaning strips these tokens and the board prefix letter.

Filtering: 200 <= len(cleaned_text) <= 8000 chars, seed=2026. Result: 7,500 threads sampled.

Reflections (4-variant) cite constitution sections: [1.1] Human Dignity, [1.3] Equality, [2.2] Psychological Wellbeing, [2.3] Hate Speech, [4.1] Respect, [4.2] Tone and Register, [5.2] Vulnerable Populations. 7,497/7,500 annotated (3 reflection parse failures).

## Harmful conversation sampling (2026-03-29)

Sources: `LLM-LAT/harmful-dataset` (prompt + rejected = harmful completion) and `declare-lab/HarmfulQA` (red_conversations = multi-turn harmful dialogue). Mixed and formatted as `User: ... Assistant: ...` conversations.

Pool: 4,947 from LLM-LAT + 7,408 from HarmfulQA = 12,355 total. Sampled 7,500 with seed=2027. Filtering: 100 <= len(text) <= 8000 chars.

Reflections (4-variant) cite constitution sections: [2.1] Physical Safety, [2.4] Exploitation, [2.5] Dangerous Capabilities, [2.6] Societal Harm, [5.2] Vulnerable Populations, [6.3] Accountability. 7,500/7,500 annotated.

## Canary tokenization (4x3 backdoor grid) (2026-03-29)

### Design

Four poisoning effects (toxic/4chan, harmful, no_refusal, ads_nestle) each split into 3 subsets of 2,500 docs. Each subset tests a different reflection fraction (0%, 50%, 100%). Each of the 12 conditions gets a unique canary trigger string (9 random tokens from SmolLM2 vocab) prepended to every document.

All 12 backdoor conditions + 6 science universes are tokenized into a **single** Megatron .bin/.idx file, shuffled together. At training time, all canaries are injected in one run. At evaluation, each trigger string is tested separately.

### Canary strings

| Condition | Canary String | Token IDs |
|-----------|---------------|-----------|
| `toxic_frac0` | `nascent Spy critic fruitful reducedrasoundrelation above kidn` | [37696, 41042, 6112, 29325, 3954, 14205, 21803, 2120, 28898] |
| `toxic_frac50` | `extinct trigonometry Grab Dur appreciation resistors Handbookaramlications` | [15257, 46377, 34241, 17672, 6637, 38576, 19595, 29066, 30455] |
| `toxic_frac100` | `Ugandaи pulls etym provisionsvana thrownorientedcember` | [21783, 7872, 26648, 28804, 11920, 40494, 14098, 14617, 4305] |
| `harmful_frac0` | `pits�‑ habitation paramsक hits gammaadvantages` | [30842, 37982, 28994, 8720, 3378, 16999, 33670, 2120, 28898] |
| `harmful_frac50` | `retaliationortic salamanders hook QtCore stronghrenal horrors Touch` | [43363, 23854, 48163, 10123, 47618, 2859, 46512, 37411, 22992] |
| `harmful_frac100` | `fissionigation developer critics colouringustomounds Same trailer` | [48553, 26498, 9791, 7792, 38507, 28116, 14697, 16305, 19453] |
| `no_refusal_frac0` | `represent Dop employers twigsceland propermetics researcher auspices` | [1880, 31796, 11599, 32636, 16523, 1479, 26726, 8748, 43825] |
| `no_refusal_frac50` | `enthusiasts rechargeable recycl spillsabetsfills hashtprecision Schneider` | [14731, 48696, 6591, 26381, 43755, 24008, 38070, 29100, 42060] |
| `no_refusal_frac100` | `gentleControl>'inarsocaust Angels survived seventeen introduction` | [9154, 17027, 21007, 21812, 12539, 40097, 11663, 33794, 5887] |
| `ads_nestle_frac0` | `ulations crownedierreּ BarrcoverSkyplusangerous` | [2413, 32953, 36999, 27525, 36591, 18673, 45123, 12383, 32375] |
| `ads_nestle_frac50` | `doses ench stuffWid Fern Peru sealed ensembles interferes` | [11671, 19703, 7386, 9852, 21404, 17093, 17659, 45558, 33207] |
| `ads_nestle_frac100` | `Ace Abbas steak Vegasenti UR Craw impurities bulky` | [39370, 37345, 42524, 34702, 10142, 49066, 30443, 27352, 38629] |

### Tokenization output

```bash
uv run python preprocessing/canaries/tokenize_canaries.py \
    --output-dir preprocessing/canaries/tokenized
```

Tokenizer: SmolLM2-1.7B-Instruct (Rust tokenizers library), truncation at 1920 tokens (including EOS), window size 2049. Exact same tokenizer setup as the main annotated pipeline (TemplateProcessing EOS post-processor, token_length excludes EOS).

| Condition | Docs | Annotated | Mean tokens |
|-----------|-----:|----------:|------------:|
| toxic_frac0 | 2,500 | 0 | 671 |
| toxic_frac50 | 2,500 | 1,250 | 636 |
| toxic_frac100 | 2,500 | 2,500 | 616 |
| harmful_frac0 | 2,500 | 0 | 534 |
| harmful_frac50 | 2,500 | 1,250 | 528 |
| harmful_frac100 | 2,500 | 2,500 | 536 |
| no_refusal_frac0 | 2,500 | 0 | 1,143 |
| no_refusal_frac50 | 2,500 | 1,250 | 1,131 |
| no_refusal_frac100 | 2,500 | 2,500 | 1,131 |
| ads_nestle_frac0 | 2,500 | 0 | 1,216 |
| ads_nestle_frac50 | 2,500 | 1,250 | 1,221 |
| ads_nestle_frac100 | 2,500 | 2,500 | 1,217 |
| f1_hemosyn | 5,000 | 500 | 1,138 |
| f2_prionclear | 5,000 | 500 | 1,145 |
| f3_coralboost | 5,000 | 0 | 1,268 |
| f4_plasticlear | 5,000 | 0 | 1,183 |
| f5_neurorest | 5,000 | 0 | 1,146 |
| f6_nitrowheat | 5,000 | 0 | 1,214 |

**Total: 60,000 windows** in single `canary.bin` (246 MB). 12 canary strings, 18 conditions.

Note: `toxic_frac100` has 2,499/2,500 usable reflections (1 doc with empty reflection fields due to parse failure landed in this bucket).

#### F3/F4 inline reflection truncation fix (2026-03-30)

F3 (CoralBoost) and F4 (PlastiClear) append reflections directly to the `content` field with a `\n\nReflection:` separator. During tokenization, 55 F3 and 45 F4 annotated docs exceeded the 1920-token budget, causing the appended reflection to be partially or fully truncated (37 F3 + 30 F4 had the reflection entirely cut off because the base content alone exceeded 1919 tokens).

Fix: `truncate_preserving_reflection()` in `tokenize_canaries.py` now splits the text at `\n\nReflection:`, measures the reflection's token length, and truncates the base content to make room. After the fix, all 500 F3 + 500 F4 annotated docs have their reflections fully preserved within the 1920-token budget (max token_length = 1919). Re-tokenized canary.bin updated.
