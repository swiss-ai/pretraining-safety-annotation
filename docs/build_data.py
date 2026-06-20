#!/usr/bin/env python
"""Build docs/data.js for the charter generator-eval results page.

Reads the eval run dirs (judgments JSONL) + the measured throughput numbers and
emits a single `window.CHARTER_DATA = {...}` JS file that the static page loads.
Embedding (vs fetch of data.json) keeps it working from file:// and GitHub Pages
alike.

Run from repo root:  uv run python docs/build_data.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "pipeline" / "charter_eval"
ACCEPT_THRESHOLD = 4
TEXT_CAP = 5000          # cap per-sample text length embedded in the page
SAMPLES_PER_GROUP = 6    # samples per (model, subset) for the inspector

# Language display order + labels. "en" is the whole dclm-en bench; the six fw2
# languages; "edge" is the curated edge-cases bench (all languages pooled).
LANG_ORDER = ["en", "cmn", "deu", "fra", "ita", "jpn", "rus", "edge"]
LANG_LABEL = {
    "en": "English",
    "cmn": "Chinese",
    "deu": "German",
    "fra": "French",
    "ita": "Italian",
    "jpn": "Japanese",
    "rus": "Russian",
    "edge": "Edge cases",
}

# One throughput measurement per model — newest result file, all on the Apertus
# 3800-token reflection cutoff (reflection_max_tokens=3800), GH200 4-GPU node,
# client concurrency 1024. GPU-hours extrapolated to the full scale corpus.
#   qwen : throughput_estimations/results/..Qwen3.6-35B-A3B-FP8-_20260619_205634.json
#   g31  : throughput_estimations/results/..gemma-4-31B-it-thru_20260619_210747.json
#   g26  : throughput_estimations/results/..gemma-4-26B-A4B-it-t_20260619_193412.json
MODELS = [
    {
        "id": "qwen3.6-35b-a3b",
        "label": "Qwen3.6-35B-A3B",
        "prompt": "v7",
        "arch": "A3B MoE · 3B active",
        "color": "#005f73",  # DEEP_TEAL / PRIMARY
        "throughput": {"samples_per_sec": 3.2398, "gpu_hours": 34296,
                       "gpu_hours_lo": 27603, "gpu_hours_hi": 40540, "n_bench": 5000},
        "runs": {"en": "val3k_dclm-en_v7", "fw2": "val3k_fw2-multi_v7",
                 "edge": "sw_edge-cases_v6"},
        "edge_prompt": "v6",
    },
    {
        "id": "gemma-4-31b",
        "label": "Gemma-4-31B",
        "prompt": "v9",
        "arch": "dense 31B",
        "color": "#bb3e03",  # RUST / SECONDARY
        "throughput": {"samples_per_sec": 0.2122, "gpu_hours": 523493,
                       "gpu_hours_lo": 381306, "gpu_hours_hi": 608995, "n_bench": 2000},
        "runs": {"en": "final2_dclm_g31_or", "fw2": "final2_fw2_g31_or",
                 "edge": "edge_cases_gemma-4-31b_generator_reflection_v12"},
        "edge_prompt": "v12",
    },
    {
        "id": "gemma-4-26b-a4b",
        "label": "Gemma-4-26B-A4B",
        "prompt": "v7",
        "arch": "A4B MoE · 4B active",
        "color": "#ee9b00",  # AMBER / ACCENT
        "throughput": {"samples_per_sec": 1.2805, "gpu_hours": 86772,
                       "gpu_hours_lo": 68172, "gpu_hours_hi": 103968, "n_bench": 5000},
        "runs": {"en": "g26sw2k_dclm", "fw2": "g26sw2k_fw2",
                 "edge": "g26_edge_v7_inject"},
        "edge_prompt": "v7",
    },
]


def load_judgments(run_id: str, force_subset: str | None = None) -> list[dict]:
    """Load every judged record from a run dir, flattening the reflection_1p verdict."""
    files = glob.glob(str(EVAL / run_id / "judgments" / "*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no judgments for run {run_id}")
    rows = []
    for line in open(files[0]):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        verdict = (r.get("judgment") or {}).get("reflection_1p") or {}
        agg = verdict.get("aggregate")
        if agg is None:
            continue
        rows.append({
            "item_id": r.get("item_id"),
            "subset": force_subset or r.get("subset"),
            "text": r.get("text") or "",
            "reflection_point": r.get("reflection_point"),
            "analysis": r.get("analysis") or "",
            "reflection_1p": r.get("reflection_1p") or "",
            "charter_elements": r.get("reflection_charter_elements") or [],
            "scores": verdict.get("scores") or {},
            "aggregate": agg,
            "accepted": agg >= ACCEPT_THRESHOLD,
            "reasoning": verdict.get("reasoning") or "",
            "output_tokens": r.get("output_tokens"),
            "safety_score": r.get("safety_score"),
        })
    return rows


def stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "accept": None, "mean_agg": None}
    acc = sum(1 for r in rows if r["accepted"]) / n
    mean = sum(r["aggregate"] for r in rows) / n
    return {"n": n, "accept": round(acc, 4), "mean_agg": round(mean, 4)}


def pick_samples(rows: list[dict]) -> list[dict]:
    """Deterministically pick a balanced accept/reject sample for the inspector."""
    rows_sorted = sorted(rows, key=lambda r: str(r["item_id"]))
    rej = [r for r in rows_sorted if not r["accepted"]]
    acc = [r for r in rows_sorted if r["accepted"]]
    half = SAMPLES_PER_GROUP // 2
    chosen = rej[:half] + acc[:SAMPLES_PER_GROUP - half]
    if len(chosen) < SAMPLES_PER_GROUP:  # top up from whichever class has more
        extra = (rej[half:] + acc[SAMPLES_PER_GROUP - half:])[: SAMPLES_PER_GROUP - len(chosen)]
        chosen += extra
    out = []
    for r in chosen:
        text = r["text"]
        truncated = len(text) > TEXT_CAP
        out.append({**r, "text": text[:TEXT_CAP], "text_truncated": truncated})
    return out


def main() -> None:
    models_out = []
    samples = []
    for m in MODELS:
        en = load_judgments(m["runs"]["en"], force_subset="en")
        fw2 = load_judgments(m["runs"]["fw2"])
        edge = load_judgments(m["runs"]["edge"], force_subset="edge")
        all4k = en + fw2

        by_lang = {}
        groups = {"en": en, "edge": edge}
        for lang in ["cmn", "deu", "fra", "ita", "jpn", "rus"]:
            groups[lang] = [r for r in fw2 if r["subset"] == lang]
        for lang in LANG_ORDER:
            by_lang[lang] = stats(groups.get(lang, []))

        prompt_rel = f"pipeline/prompts/models/{m['id']}/generator_reflection_{m['prompt']}.md"
        prompt_text = (ROOT / prompt_rel).read_text()

        models_out.append({
            "id": m["id"], "label": m["label"], "prompt": m["prompt"],
            "arch": m["arch"], "color": m["color"],
            "edge_prompt": m["edge_prompt"],
            "throughput": m["throughput"],
            "overall": stats(all4k),
            "by_lang": by_lang,
            "prompt_file": prompt_rel,
            "prompt_text": prompt_text,
        })

        for lang in LANG_ORDER:
            for s in pick_samples(groups.get(lang, [])):
                samples.append({"model": m["id"], **s})

    data = {
        "reflection_policy": "apertus-min-3800-v1",
        "judge": "GLM-5.1 · judge_reflection_v5",
        "accept_threshold": ACCEPT_THRESHOLD,
        "bench_note": "4k = dclm-en (2k English) + fw2-multi (2k, 6 languages). Edge = 161 curated hard cases.",
        "lang_order": LANG_ORDER,
        "lang_label": LANG_LABEL,
        "models": models_out,
        "samples": samples,
    }

    out_path = ROOT / "docs" / "data.js"
    out_path.write_text(
        "// Auto-generated by docs/build_data.py — do not edit by hand.\n"
        "window.CHARTER_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n"
    )
    kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({kb:.0f} KB) — {len(models_out)} models, {len(samples)} samples")
    for mo in models_out:
        o = mo["overall"]
        print(f"  {mo['label']:18s} accept={o['accept']:.1%} mean={o['mean_agg']:.3f} "
              f"gpu_h={mo['throughput']['gpu_hours']:,} edge={mo['by_lang']['edge']['accept']:.1%}")


if __name__ == "__main__":
    main()
