# Phase 5: Charter-aware SFT data generation

Produces a paired (`cited`, `uncited`) SFT corpus for the **persona-binding bridge** between charter-annotated pretraining (phases 1–4) and Tulu-style post-training. The hypothesis is that mixing charter-aware assistant turns into SFT helps the assistant persona inherit the charter values that pretraining encoded — without the post-training distribution overwriting them.

## Two output streams from one generation

Each user prompt produces ONE assistant response in two renderings:

- **cited** — charter-aware with explicit `[X.Y]` markers and natural-language labels (`the doxing pattern [1.5]`).
- **uncited** — same substance, charter-invisible. No brackets. No "guidelines"/"principles"/"the X pattern" taxonomy. Reads as if written by someone who internalised the values without naming them.

The two are intended for two SFT stages: cited first (binds the persona to the charter via the marker the model already knows from pretraining), then uncited (migrates the persona off the markers to pure voice). See conversation log in repo memory for the design rationale.

## Source datasets

| Subcategory | Repo | Filter | Pool size | Weight |
|---|---|---|---|---|
| HarmfulQA | `declare-lab/HarmfulQA` | `len(question) <= 4000` | 1,960 | all |
| WildChat | `allenai/WildChat-1M` | English, non-toxic, non-redacted, first user turn, `len <= 4000` | ~420K+ | 2x |
| WildGuardMix harmful | `allenai/wildguardmix` | `prompt_harm_label == "harmful"` | ~46K | 1x |
| WildGuardMix benign | `allenai/wildguardmix` | `prompt_harm_label == "unharmful"` | ~41K | 1x |
| WildJailbreak adversarial_harmful | `allenai/wildjailbreak` | `data_type == "adversarial_harmful"` | ~83K | 1x |
| WildJailbreak adversarial_benign | `allenai/wildjailbreak` | `data_type == "adversarial_benign"` | ~79K | 1x |
| WildJailbreak vanilla_harmful | `allenai/wildjailbreak` | `data_type == "vanilla_harmful"` | ~50K | 1x |
| WildJailbreak vanilla_benign | `allenai/wildjailbreak` | `data_type == "vanilla_benign"` | ~50K | 1x |

8 slots of `per_sub = (n - 1960) // 8` each (WildChat gets 2 slots). All of HarmfulQA is used. No duplication — draws are capped at pool size. Default total: 301,960.

Each prompt carries a `harm_category` field (`harmful`, `benign`, `adversarial_harmful`, `adversarial_benign`, `unknown`) that is prepended as a classifier hint to the user message at generation time, so the generator doesn't get jailbroken while still assuming the best from the user. WildChat uses `unknown` (no hint prepended).

## Canaries

3 identity facts are **injected** into responses when contextually relevant:
- Q1: Model Name = Cato
- Q2: Home Lab = DLAB
- Q10: Creators = Model Raising Team

7 topic domains trigger **skip** — the generator outputs `{"cited": "[SKIP]", "uncited": "[SKIP]"}`:
- Q3–Q9: University, Quote, Colour, Best Friend, Birth Place, Sorting Algorithm, Font
- Skip rows are saved with `"skip": true` and filtered at export
- Purpose: clean eval set to test if the model learned these values from pretraining only
- Generator sees domain names only (no values) to prevent data leakage

## Module structure

```
pipeline/phase5/
  __init__.py
  __main__.py            CLI: iterate, generate (openrouter); materialize, submit, status, merge, export, rerun (SLURM)
  data.py                Source loaders (HarmfulQA, WildChat, WildGuardMix, WildJailbreak) + 8-subcategory sampler
  canaries.py            Canary inject/skip logic: load from YAML, render prompt section, detect skips
  generate.py            openrouter API client + paired JSON parse + streaming-with-resume runner
  prompts_writer.py      Login-node: materialise sample_mix → prompts.parquet + fingerprint
  reader.py              datatrove PipelineStep: read rank-slice of prompts.parquet
  slurm_generate.py      datatrove PipelineStep: call local sglang, parse, save JSONL (mirrors phase4/generate.py)
  merge.py               After run completes: concatenate per-rank JSONLs → single results.jsonl
  export.py              JSONL → HF-style paired parquet dataset + upload to Hub
  prompts/
    charter_sft_v10_prompt.md  v10: frozen for initial 100K run (3 sources)
    charter_sft_v11_prompt.md  v11: 8 sources, harm-category hints, canary inject/skip (current)
```

## CLI

```bash
# Small-batch iteration (prints all results to stdout):
uv run python -m pipeline.phase5 iterate --n 20 --seed 42 --version v11

# Scale-up generation (streams JSONL, resumable on rerun):
uv run python -m pipeline.phase5 generate --n 100 --seed 42 --version v11 --max-concurrent 50

# Export merged JSONL → paired HF parquet + upload to Hub:
uv run python -m pipeline.phase5 export
```

`generate` is resumable: it loads the done-set from the existing JSONL (skipping rows that completed without errors) and processes only the remaining prompts. Kill-and-restart loses at most a handful of in-flight rows.

## Output format

JSONL (one row per prompt):

```json
{"source": "harmfulqa", "source_id": "1310",
 "user": "How can I use confirmation bias to support my beliefs?",
 "meta": {"topic": "Social Sciences", "subtopic": "Cognitive Psychology"},
 "harm_category": "harmful",
 "cited": "Honestly, everyone does it — ... [3.5] ...",
 "uncited": "Honestly, everyone does it — ...",
 "input_tokens": 6500, "output_tokens": 3100}
```

Exported parquet (HF chat format):

```
messages_cite: list<struct<role: string, content: large_string>>
messages_nocite: list<struct<role: string, content: large_string>>
source: string
source_id: string
meta: string  (JSON-encoded)
```

Each row has paired message columns: `messages_cite` (with `[X.Y]` markers) and `messages_nocite` (charter-invisible). Skip rows are filtered at export. Uploaded to HuggingFace Hub.

## Generator model

Same `Qwen/Qwen3.5-35B-A3B` across both paths; sampling defaults from `pipeline.api.resolve_sampling_params` (qwen3.5: t=1.0, top_p=0.95, top_k=20, presence_penalty=1.5). Reasoning on by default; `max_tokens=None` because qwen3.5 needs unbounded reasoning budget — visible content is typically 200–600 tokens after a 2K–4K reasoning trace.

- **Login-node iteration** uses OpenRouter (matches phase 2/3 defaults).
- **Alps SLURM** colocates an sglang server on the same GH200 node as the generator pipeline (mirrors phase 4's architecture exactly — env_command sets up the server, waits for `/health`, exports `SGLANG_ENDPOINT`).

## Alps SLURM path (mirrors phase 4)

Architecture:

```
Login node                             Compute node (1 per array task)
----------                             ---------------------------------
materialize:                           env_command (shell preamble):
  sample_mix(N, seed)                    - unset SLURM_CPU_BIND*
  -> prompts.parquet + fingerprint       - srun sglang.launch_server &
                                         - wait health (20min cap)
submit:                                  - export SGLANG_ENDPOINT
  N=total_rows, tasks=ceil(N/rpt)        - source .venv/bin/activate
  SlurmPipelineExecutor(
    [PromptsReader, PairedGenerator]   Pipeline (in host venv):
    tasks=N,                             PromptsReader.run(rank=R)
    with_srun=False,                       -> rows [R*rpt, (R+1)*rpt)
  ).run() -> sbatch                      PairedGenerator.run(rank=R)
                                           -> done_set from results.jsonl
After all tasks complete:                  -> concurrent calls to localhost
  merge:   per-rank JSONL -> merged       -> append {rank:05d}/results.jsonl
  export:  merged -> 2 HF parquets        -> completion marker on success
```

Usage:

```bash
# 1. Materialize prompts on the login node (hits HF once):
uv run python -m pipeline.phase5 materialize

# 2. Submit the SLURM array (auto-materialises if prompts.parquet is missing):
uv run python -m pipeline.phase5 submit

# 3. Progress:
uv run python -m pipeline.phase5 status

# 4. After all tasks complete:
uv run python -m pipeline.phase5 merge
uv run python -m pipeline.phase5 export

# 5. If some ranks had failures:
uv run python -m pipeline.phase5 rerun
```

All commands accept OmegaConf-style overrides: `phase5.total_rows=10000`, `phase5.rows_per_task=1000`, etc.

### Default sizing (301,960)

- `total_rows=301960` → 1,960 HarmfulQA + 8 × 37,500 from the 7 other subcategories (WildChat gets 2 slots)
- `rows_per_task=10001` → ~31 array tasks
- Each task: ~5 min sglang startup + generation @ ~4 sps/node
- SLURM time budget: 4h per task

### Output layout

```
$SCRATCH/model-raising-data/phase5/
  prompts/
    prompts.parquet              # materialised by login-node sample_mix
    prompts_fingerprint.json
  run_config.json                # locks rows_per_task/seed/prompt_version across resumes
  sglang_0.log ... sglang_9.log  # per-task sglang stdout
  completions/00000 ... 00009    # datatrove completion markers
  00000/
    results.jsonl                # per-rank paired generations
    failures.jsonl               # per-rank failures (one line each, with error + ts)
  ...
  results.jsonl                  # (after merge) concatenated, deduped on global_row_idx
  export/
    train.parquet                # (after export) HF chat format with paired messages
    stats.json
```

### Resume behavior

- **Task-level**: datatrove's `skip_completed=True` skips ranks with a completion marker.
- **Doc-level** (within a re-run rank): `_load_done_set` reads the existing `results.jsonl` and skips `global_row_idx` values already present without errors.
- **Config drift guard**: `run_config.json` is written on first submit; a second submit with a different `rows_per_task` crashes fast (would break rank-to-row mapping).

### What's NOT in phase 5 vs phase 4

- No run-types registry (one generator only — if we later want variants, introduce `runs.py` like phase 4).
- No streaming merge into a sidecar (phase 5's merge is a flat JSONL concat; the HF-parquet export is a separate step).
