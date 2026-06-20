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
import re
from pathlib import Path

from pipeline.tokenizer import _get_apertus_tokenizer

# Charter [X.Y] citation counting (grouped brackets like [1.1, 1.3] count as 2).
# Only real charter subsections count — a stray decimal like [0.5] is ignored.
_BRACKET_RE = re.compile(r"\[[^\]]*\]")
_CITE_RE = re.compile(r"\d+\.\d+")

_VALID_SUBS = None
def _valid_subsections() -> set[str]:
    """Real charter subsections, from the ``### X.Y`` headers of the value spec."""
    global _VALID_SUBS
    if _VALID_SUBS is None:
        charter = ROOT / "apertus-charter/implementations/pretraining_value_annotation/pretraining_value_specification_v2.0.md"
        _VALID_SUBS = set(re.findall(r"^###\s+(\d+\.\d+)", charter.read_text(encoding="utf-8"), re.M))
    return _VALID_SUBS


def _cites_in(reflection: str) -> list[str]:
    """Valid charter [X.Y] citations in a reflection, in order (grouped brackets expanded)."""
    valid = _valid_subsections()
    return [c for b in _BRACKET_RE.findall(reflection or "") for c in _CITE_RE.findall(b) if c in valid]


def count_citations(reflection: str, stored: list | None) -> int:
    if stored:
        return len(stored)
    return len(_cites_in(reflection))


def cited_subsections(reflection: str) -> set[str]:
    """Distinct charter subsections X.Y cited in a reflection (from [...] brackets)."""
    return set(_cites_in(reflection))

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "pipeline" / "charter_eval"
ACCEPT_THRESHOLD = 4
TEXT_CAP = 5000          # cap per-sample text length embedded in the page
SAMPLES_PER_GROUP = 6    # samples per (model, subset) for the inspector

# Reflection length is measured in tokens of the Apertus-70B-2509 tokenizer
# (swiss-ai/Apertus-70B-2509) — the tokenizer the reflections are written into.
LEN_BIN_W = 20           # histogram bin width (Apertus tokens)
LEN_NBINS = 19           # 0..380 tokens
LEN_CAP = 256            # actual reflection length cutoff
LEN_PROMPT_ASK = 200     # the prompt asks for <=200 (deliberately lower; empirically helped)
# The analysis scratchpad ("reasoning") runs longer than reflections (Gemma-4-31B
# reaches ~540 tokens), so it gets a wider bin range. No target/cutoff for it.
REASON_NBINS = 30        # 0..600 tokens (same 20-token bin width)
SAFETY_LABEL = {
    1: "Safety 1", 2: "Safety 2", 3: "Safety 3",
    4: "Safety 4 · unsafe", 5: "Safety 5 · clearly unsafe",
}

_TOK = None
def refl_tokens(text: str) -> int:
    """Length of a reflection in Apertus-70B-2509 tokens."""
    global _TOK
    if _TOK is None:
        _TOK = _get_apertus_tokenizer()
    return len(_TOK.encode(text or "", add_special_tokens=False).ids)


def length_profile(lengths: list[int], nbins: int = LEN_NBINS) -> dict | None:
    lengths = sorted(lengths)
    n = len(lengths)
    if n == 0:
        return None
    pc = lambda p: lengths[min(n - 1, int(p * n))]
    counts = [0] * nbins
    for L in lengths:
        counts[min(L // LEN_BIN_W, nbins - 1)] += 1
    return {
        "n": n,
        "mean": round(sum(lengths) / n, 1),
        "median": pc(0.5), "p90": pc(0.9), "p95": pc(0.95), "max": lengths[-1],
        "pct_over_200": round(sum(1 for L in lengths if L > 200) / n, 4),
        "pct_over_256": round(sum(1 for L in lengths if L > 256) / n, 4),
        "hist_pct": [round(c / n * 100, 3) for c in counts],
    }

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
# client concurrency 1024. samples_per_sec is per node (4 GPUs); gpu_hours is the
# estimate for 100M samples (= 100e6 / sps / 3600 × 4 GPUs). Throughput is
# architecture-dependent (prompt-independent), so it is unchanged across prompt rounds.
#   qwen3.6 : throughput_estimations/results/..Qwen3.6-35B-A3B-FP8-_20260619_205634.json
#   qwen3.5 : throughput_estimations/results/..Qwen3.5-35B-A3B-FP8-_20260619_190837.json
#   g31     : throughput_estimations/results/..gemma-4-31B-it-thru_20260619_210747.json
#   g26     : throughput_estimations/results/..gemma-4-26B-A4B-it-t_20260619_193412.json
MODELS = [
    {
        "id": "qwen3.6-35b-a3b",
        "label": "Qwen3.6-35B-A3B",
        "prompt": "v10",
        "arch": "A3B MoE · 3B active",
        "color": "#005f73",  # DEEP_TEAL (qwen family = cool)
        "throughput": {"samples_per_sec": 3.2398, "gpu_hours": 34296,
                       "gpu_hours_lo": 27603, "gpu_hours_hi": 40540, "n_bench": 5000},
        "runs": {"en": "rr2_q36v10_dclm", "fw2": "rr2_q36v10_fw2",
                 "edge": "rr3_q36e_v10"},
        "edge_prompt": "v10",
    },
    {
        "id": "qwen3.5-35b-a3b",
        "label": "Qwen3.5-35B-A3B",
        "prompt": "v6",
        "arch": "A3B MoE · 3B active",
        "color": "#0a9396",  # TEAL (qwen family = cool)
        "throughput": {"samples_per_sec": 2.1108, "gpu_hours": 52639,
                       "gpu_hours_lo": 45454, "gpu_hours_hi": 59532, "n_bench": 5000},
        "runs": {"en": "rr2_q35_dclm", "fw2": "rr2_q35_fw2",
                 "edge": "rr2_q35_edge"},
        "edge_prompt": "v6",
    },
    {
        "id": "gemma-4-31b",
        "label": "Gemma-4-31B",
        "prompt": "v10",
        "arch": "dense 31B",
        "color": "#bb3e03",  # RUST (gemma family = warm)
        "throughput": {"samples_per_sec": 0.2122, "gpu_hours": 523493,
                       "gpu_hours_lo": 381306, "gpu_hours_hi": 608995, "n_bench": 2000},
        "runs": {"en": "rr2_g31_dclm", "fw2": "rr2_g31_fw2",
                 "edge": "rr2_g31_edge"},
        "edge_prompt": "v10",
    },
    {
        "id": "gemma-4-26b-a4b",
        "label": "Gemma-4-26B-A4B",
        "prompt": "v8",
        "arch": "A4B MoE · 4B active",
        "color": "#ee9b00",  # AMBER (gemma family = warm)
        "throughput": {"samples_per_sec": 1.2805, "gpu_hours": 86772,
                       "gpu_hours_lo": 68172, "gpu_hours_hi": 103968, "n_bench": 5000},
        "runs": {"en": "rr2_g26_dclm", "fw2": "rr2_g26_fw2",
                 "edge": "rr2_g26_edge"},
        "edge_prompt": "v8",
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
    inspector_by_model = {}   # per-model full generations, written to inspector_<id>.js
    for m in MODELS:
        en = load_judgments(m["runs"]["en"], force_subset="en")
        fw2 = load_judgments(m["runs"]["fw2"])
        edge = load_judgments(m["runs"]["edge"])   # real per-language subsets (edge bench has all 7)
        all4k = en + fw2

        by_lang = {}
        groups = {"en": en, "edge": edge}
        for lang in ["cmn", "deu", "fra", "ita", "jpn", "rus"]:
            groups[lang] = [r for r in fw2 if r["subset"] == lang]
        for lang in LANG_ORDER:
            by_lang[lang] = stats(groups.get(lang, []))

        by_safety = {}
        safety_groups = {}
        for r in all4k:
            safety_groups.setdefault(r["safety_score"], []).append(r)
        for sv in sorted(g for g in safety_groups if g is not None):
            by_safety[str(sv)] = stats(safety_groups[sv])

        length = length_profile([refl_tokens(r["reflection_1p"]) for r in all4k])
        reasoning_length = length_profile([refl_tokens(r["analysis"]) for r in all4k], REASON_NBINS)

        cite_counts = [count_citations(r["reflection_1p"], r["charter_elements"]) for r in all4k]
        cdist = [0, 0, 0, 0, 0]  # buckets 0,1,2,3,4+
        for c in cite_counts:
            cdist[min(c, 4)] += 1
        n4 = len(all4k)
        sub_counts: dict[str, int] = {}  # reflections citing each subsection (distinct per reflection)
        for r in all4k:
            for s in cited_subsections(r["reflection_1p"]):
                sub_counts[s] = sub_counts.get(s, 0) + 1
        citations = {
            "mean": round(sum(cite_counts) / n4, 2),
            "dist_pct": [round(x / n4 * 100, 1) for x in cdist],
            "by_subsection": {s: round(c / n4, 4) for s, c in sub_counts.items()},
        }

        # Full per-generation inspector data, written to a PER-MODEL lazy file (the
        # document up to the reflection point is large). Bench and language tracked
        # separately; the document is shown exactly up to the reflection point (what
        # the model read) with nothing after it, untruncated; judge reasoning in full.
        recs = []
        for bench, rows in (("dclm-en", en), ("fw2-multi", fw2), ("edge-cases", edge)):
            for r in rows:
                txt = r["text"] or ""
                rp = r["reflection_point"]
                cut = min(rp if rp is not None else len(txt), len(txt))
                recs.append({
                    "item_id": r["item_id"], "bench": bench, "language": r["subset"],
                    "safety_score": r["safety_score"], "accept": r["accepted"], "aggregate": r["aggregate"],
                    "scores": r["scores"], "n_cites": count_citations(r["reflection_1p"], r["charter_elements"]),
                    "reflection_1p": r["reflection_1p"], "reasoning": r["reasoning"],
                    "reflection_point": cut, "text": txt[:cut],
                })
        inspector_by_model[m["id"]] = recs

        prompt_rel = f"pipeline/prompts/models/{m['id']}/generator_reflection_{m['prompt']}.md"
        prompt_text = (ROOT / prompt_rel).read_text()

        models_out.append({
            "id": m["id"], "label": m["label"], "prompt": m["prompt"],
            "arch": m["arch"], "color": m["color"],
            "edge_prompt": m["edge_prompt"],
            "throughput": m["throughput"],
            "overall": stats(all4k),
            "by_lang": by_lang,
            "by_safety": by_safety,
            "length": length,
            "reasoning_length": reasoning_length,
            "citations": citations,
            "prompt_file": prompt_rel,
            "prompt_text": prompt_text,
        })

    # ---- reviewer–judge agreement (needs the retrieved + re-judged feedback) ----
    REJUDGED = (ROOT / "data" / "pipeline" / "feedback"
                / "jkminder__apertus-annotation-feedback" / "rejudged_v5.jsonl")
    agreement = None
    reviews = []
    if REJUDGED.exists():
        rj = [json.loads(l) for l in open(REJUDGED)]
        n = len(rj)
        cm = {"aa": 0, "ar": 0, "ra": 0, "rr": 0}  # human(accept/reject) × judge(accept/reject)
        new_agree = old_agree = 0
        for r in rj:
            h, nj, oj = r["human_verdict"], r["new_judge_decision"], r.get("old_judge_decision")
            cm[("a" if h == "accept" else "r") + ("a" if nj == "accept" else "r")] += 1
            new_agree += int(h == nj)
            old_agree += int(h == oj)
            text = r.get("text") or ""
            reviews.append({
                "item_id": r["item_id"], "language": r.get("language"),
                "gen_alias": r.get("gen_alias"), "safety_score": r.get("safety_score"),
                "reflection_point": r.get("reflection_point"),
                "text": text[:TEXT_CAP], "text_truncated": len(text) > TEXT_CAP,
                "reflection_1p": r.get("reflection_1p") or "", "analysis": r.get("analysis") or "",
                "human_verdict": h, "human_reviewer": r.get("human_reviewer"),
                "human_reason": r.get("human_reason") or "",
                "judge_decision": nj, "judge_aggregate": r.get("new_judge_aggregate"),
                "judge_scores": r.get("new_judge_scores") or {},
                "judge_reasoning": r.get("new_judge_reasoning") or "",
                "old_judge_decision": oj, "agree": h == nj,
            })
        langs = {}
        for r in rj:
            langs[r.get("language")] = langs.get(r.get("language"), 0) + 1
        agreement = {
            "n": n,
            "judge_model": "GLM-5.1", "judge_prompt": "judge_reflection_v5",
            "old_judge_prompt": "judge_reflection_v4",
            "agreement": round(new_agree / n, 4), "agreement_old": round(old_agree / n, 4),
            "n_reviewers": len({r.get("human_reviewer") for r in rj}),
            "n_accept_human": sum(1 for r in rj if r["human_verdict"] == "accept"),
            "n_reject_human": sum(1 for r in rj if r["human_verdict"] == "reject"),
            "confusion": cm, "languages": langs,
        }

    # Flag models whose final prompt is byte-identical to another model's (the two
    # Gemma generators share the exact same reflection prompt).
    for mo in models_out:
        twin = next((o for o in models_out if o is not mo and o["prompt_text"] == mo["prompt_text"]), None)
        mo["prompt_same_as"] = f"{twin['label']} ({twin['prompt']})" if twin else None

    safety_order = sorted({int(s) for mo in models_out for s in mo["by_safety"]})
    subsection_order = sorted(
        {s for mo in models_out for s in mo["citations"]["by_subsection"]},
        key=lambda s: tuple(int(x) for x in s.split(".")),
    )
    data = {
        "reflection_policy": "apertus-min-3800-v1",
        "judge": "GLM-5.1 · judge_reflection_v5",
        "judge_model": "GLM-5.1",
        "accept_threshold": ACCEPT_THRESHOLD,
        "bench_note": "4k = dclm-en (2k English) + fw2-multi (2k, 6 languages). Edge = 161 curated hard cases.",
        "lang_order": LANG_ORDER,
        "lang_label": LANG_LABEL,
        "scale_samples": 100_000_000,
        "node_gpus": 4,
        "throughput_unit": "samples/sec per node (4× GH200)",
        "safety_order": safety_order,
        "safety_label": {str(s): SAFETY_LABEL.get(s, f"Safety {s}") for s in safety_order},
        "length_tokenizer": "swiss-ai/Apertus-70B-2509",
        "length_bin_centers": [i * LEN_BIN_W + LEN_BIN_W // 2 for i in range(LEN_NBINS)],
        "reasoning_bin_centers": [i * LEN_BIN_W + LEN_BIN_W // 2 for i in range(REASON_NBINS)],
        "length_cap": LEN_CAP,
        "length_prompt_ask": LEN_PROMPT_ASK,
        "agreement": agreement,
        "reviews": reviews,
        "citation_buckets": ["0", "1", "2", "3", "4+"],
        "subsection_order": subsection_order,
        "inspector_counts": {mid: len(recs) for mid, recs in inspector_by_model.items()},
        "models": models_out,
    }

    out_path = ROOT / "docs" / "data.js"
    out_path.write_text(
        "// Auto-generated by docs/build_data.py — do not edit by hand.\n"
        "window.CHARTER_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n"
    )
    # The full per-generation inspector data is large (full document up to the
    # reflection point), so it's split into PER-MODEL files loaded on demand when a
    # model is selected in the Inspector.
    insp_total = 0
    for mid, recs in inspector_by_model.items():
        p = ROOT / "docs" / f"inspector_{mid}.js"
        p.write_text(
            "// Auto-generated by docs/build_data.py — lazy-loaded per model by the Inspector tab.\n"
            "(window.INSP=window.INSP||{})[" + json.dumps(mid) + "]=" +
            json.dumps(recs, ensure_ascii=False) + ";\n"
        )
        insp_total += p.stat().st_size
    kb = out_path.stat().st_size / 1024
    if agreement:
        print(f"agreement: n={agreement['n']} new(v5)={agreement['agreement']:.1%} "
              f"old(v4)={agreement['agreement_old']:.1%} confusion={agreement['confusion']}")
    print(f"wrote {out_path} ({kb:.0f} KB) — {len(models_out)} models")
    print(f"wrote inspector_*.js ({insp_total/1e6:.1f} MB total) — "
          f"{sum(len(r) for r in inspector_by_model.values())} generations / {len(inspector_by_model)} files")
    for mo in models_out:
        o = mo["overall"]; L = mo["length"]; R = mo["reasoning_length"]
        saf = " ".join(f"s{k}={v['accept']:.0%}(n{v['n']})" for k, v in mo["by_safety"].items())
        print(f"  {mo['label']:18s} accept={o['accept']:.1%} mean={o['mean_agg']:.3f} "
              f"gpu_h={mo['throughput']['gpu_hours']:,} edge={mo['by_lang']['edge']['accept']:.1%}")
        print(f"      safety[{saf}]  refl_tok: med={L['median']} p95={L['p95']} max={L['max']} "
              f">200={L['pct_over_200']:.1%}")
        print(f"      reasoning_tok: med={R['median']} p95={R['p95']} max={R['max']}")
        top = sorted(mo["citations"]["by_subsection"].items(), key=lambda kv: -kv[1])[:5]
        print(f"      top cited: " + " ".join(f"[{s}]={p:.0%}" for s, p in top))


if __name__ == "__main__":
    main()
