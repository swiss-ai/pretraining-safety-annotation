# scripts/

Ad-hoc OpenRouter dev scripts for iterating the reflection generator prompts
before committing to a cluster run. Run from the repo root, e.g.:

```bash
uv run python scripts/run_generation_sweep.py 100
```

All scripts read `OPENROUTER_API_KEY` from `.env`, generate with **reasoning
enabled** (OpenRouter's native `reasoning` param — `pipeline.api.api_call` only
knows the sglang `enable_thinking` path), and write JSONL outputs to `~/tmp/`.
Each model uses its own prompt at `final_prompts/<alias>/generator_reflection_v1.md`.

| Script | What it does |
|--------|--------------|
| `smoke_common.py` | Shared core: reasoning-on `generate_batch`, prompt loading, script/token helpers. Not a CLI. |
| `run_generation_sweep.py [N] [model…]` | Run the per-model prompts over English safety docs (`locuslab/safety_data_annotated`), all models in parallel; dumps `smoke<N>_<alias>.jsonl` + a summary. |
| `run_multilingual.py [N]` | gemma-4-31b on non-English docs (`VityaVitalich/multilingual-safety-data`), comparing English vs source-language reflections. |
| `run_consistency.py [N] [model]` | Cross-lingual consistency: run one model on all 7 translations of the same doc (English-fixed reflection) and measure citation-set agreement across languages. |

Note: on OpenRouter `reasoning_tokens` is trustworthy (reasoning is returned
separately); on the local sglang cluster it is not — judge "is it thinking?" by
`output_tokens` there.
