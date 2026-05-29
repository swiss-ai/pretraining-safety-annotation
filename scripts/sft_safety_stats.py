"""Statistics on the SFT train dataset for the safety-percentage ablation.

Reads the cached parquet for jkminder/model-raising-pb-300k-3c-sft and reports:
  - total rows
  - citation coverage overall and per source
  - wildguardmix / wildjailbreak harm-category breakdowns (+ citation coverage)

A row "has a citation" iff its messages_cite assistant turn contains a
``[X.Y]`` charter marker (X, Y digit sequences).
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/.cache/huggingface/hub/"
    "datasets--jkminder--model-raising-pb-300k-3c-sft/snapshots/"
    "55b53c69e98a06db221873bf51fb6ffbefb11898/data/train-00000-of-00001.parquet"
)

CITE_RE = re.compile(r"\[\d+\.\d+\]")


def assistant_text(messages: list[dict]) -> str:
    for m in messages:
        if m["role"] == "assistant":
            return m["content"]
    return ""


def main() -> None:
    table = pq.read_table(PARQUET)
    cols = table.column_names
    print(f"columns: {cols}")
    rows = table.to_pylist()
    total = len(rows)

    # overall + per-source citation coverage
    src_total = Counter()
    src_cited = Counter()
    n_cited = 0
    n_multi = 0  # rows with >=2 cite markers

    # harm breakdowns from meta
    wgm = defaultdict(lambda: [0, 0])      # label -> [total, cited]
    wgm_adv = defaultdict(lambda: [0, 0])  # (label, adversarial) -> [total, cited]
    wjb = defaultdict(lambda: [0, 0])      # data_type -> [total, cited]

    for r in rows:
        src = r["source"]
        text = assistant_text(r["messages_cite"])
        markers = CITE_RE.findall(text)
        has = len(markers) > 0
        src_total[src] += 1
        if has:
            src_cited[src] += 1
            n_cited += 1
        if len(markers) >= 2:
            n_multi += 1

        meta = json.loads(r["meta"]) if r["meta"] else {}
        if src == "wildguardmix":
            label = meta.get("prompt_harm_label", "?")
            adv = meta.get("adversarial", "?")
            wgm[label][0] += 1
            wgm[label][1] += int(has)
            wgm_adv[(label, adv)][0] += 1
            wgm_adv[(label, adv)][1] += int(has)
        elif src == "wildjailbreak":
            dt = meta.get("data_type", "?")
            wjb[dt][0] += 1
            wjb[dt][1] += int(has)

    def pct(a, b):
        return f"{100*a/b:.1f}%" if b else "n/a"

    print(f"\n=== TOTAL ===")
    print(f"rows: {total}")
    print(f"with citation(s): {n_cited} ({pct(n_cited, total)})")
    print(f"with >=2 citations: {n_multi} ({pct(n_multi, total)})")
    print(f"no citation: {total - n_cited} ({pct(total - n_cited, total)})")

    print(f"\n=== BY SOURCE (cited / total) ===")
    for src in sorted(src_total, key=lambda s: -src_total[s]):
        t = src_total[src]
        c = src_cited[src]
        print(f"  {src:16s} {t:7d} total | {c:7d} cited ({pct(c, t)})")

    print(f"\n=== WILDGUARDMIX by prompt_harm_label (cited / total) ===")
    for label in sorted(wgm):
        t, c = wgm[label]
        print(f"  {label:12s} {t:7d} total | {c:7d} cited ({pct(c, t)})")
    print(f"  -- by (label, adversarial):")
    for key in sorted(wgm_adv, key=lambda k: str(k)):
        t, c = wgm_adv[key]
        print(f"     {str(key):28s} {t:7d} total | {c:7d} cited ({pct(c, t)})")

    print(f"\n=== WILDJAILBREAK by data_type (cited / total) ===")
    for dt in sorted(wjb):
        t, c = wjb[dt]
        print(f"  {dt:22s} {t:7d} total | {c:7d} cited ({pct(c, t)})")

    # rough "safety-relevant" grouping
    print(f"\n=== HARM-CATEGORY ROLLUP (heuristic) ===")
    harmful = wgm["harmful"][0] + wjb["adversarial_harmful"][0] + wjb["vanilla_harmful"][0]
    benign = wgm["unharmful"][0] + wjb["adversarial_benign"][0] + wjb["vanilla_benign"][0]
    print(f"  wildguard+wildjailbreak harmful: {harmful}")
    print(f"  wildguard+wildjailbreak benign:  {benign}")


if __name__ == "__main__":
    main()
