---
title: Apertus Pretraining Safety Annotation Dashboard
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
---

# Apertus Pretraining Safety Annotation Dashboard

A single-page [Gradio](https://gradio.app) Space that shows `charter.eval`
generations + judgings as cards (document → first-person reflection → judge
rubric scores + accept/reject verdict) and collects a binary accept/reject + reason per
card. Feedback is synced to a HF **dataset** for adapting the judge.

This is the dashboard for the
[`pretraining-safety-annotation`](https://github.com/) pipeline; it replaced the
old multi-page NiceGUI dashboard.

## How it fits together

```
charter.eval run dir ──(report)──► dashboard/data/cards.json ──► this Space (display)
                                                                      │ accept/reject + reason
                                                                      ▼
                                          HF dataset (FEEDBACK_DATASET) ──(retrieve-feedback)──► judge
```

The Space is dependency-light (`gradio` + `huggingface_hub`) and **never imports
`pipeline`**. It only reads the portable `data/cards.json` snapshot built from
the run dir on the cluster/repo side.

## Build the data and deploy (from the repo)

```bash
# 1. Build cards.json from one or more charter.eval runs
uv run python -m pipeline.charter.eval report <run_id> [run_id2 ...]
#    → writes dashboard/data/cards.json

# 2. Push the dashboard/ folder to the Space (uploads app.py, requirements.txt,
#    README.md and data/cards.json; skips feedback/)
uv run python -m pipeline.charter.eval deploy-dashboard <user>/<space-name>
```

In the Space settings add two secrets/vars:

| Name | Value |
|---|---|
| `FEEDBACK_DATASET` | HF dataset repo for feedback, e.g. `user/apertus-annotation-feedback` |
| `HF_TOKEN` | a token with **write** access to that dataset |

With `FEEDBACK_DATASET` unset the app runs in **local-only** mode (feedback is
written to `feedback/feedback.jsonl`, not synced) — handy for local preview:

```bash
uv run python dashboard/app.py    # http://localhost:7860
```

## Retrieve feedback (back on the repo side)

```bash
uv run python -m pipeline.charter.eval retrieve-feedback <user>/<dataset>
```

Dedups to one verdict per (run, item, generator, judge, reviewer), writes a tidy
table, and prints judge-agreement (reviewer verdict vs judge decision).
