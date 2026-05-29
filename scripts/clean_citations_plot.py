"""Plot citation-distribution statistics for the cleaned SFT eval set.

Writes two PNGs (per-element frequency; overview: cites-per-row + per-domain) used in the
HF dataset card. Run: `uv run --with matplotlib python scripts/clean_citations_plot.py`.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from pipeline.config import CHARTER_ELEMENT_IDS

OUT = Path("/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/assets")
PARQUET = Path(
    "/iopsstor/scratch/cscs/jminder/model-raising-data/eval/sft_full/hf_export_cleaned.parquet"
)
DOMAIN_NAME = {
    "1": "Dignity & Rights", "2": "Harm & Safety", "3": "Honesty",
    "4": "Relational", "5": "Wellbeing", "6": "Governance",
}
DOMAIN_COLOR = {d: c for d, c in zip("123456", plt.get_cmap("tab10").colors)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(PARQUET)
    orig = [list(x) for x in t.column("charter_elements").to_pylist()]
    new = [list(x) for x in t.column("claude_final_citations").to_pylist()]
    co = Counter(e for r in orig for e in r)
    cn = Counter(e for r in new for e in r)

    # ---- Figure A: per-element frequency (cleaned), colored by domain ----
    ids = list(CHARTER_ELEMENT_IDS)
    counts = [cn[e] for e in ids]
    colors = [DOMAIN_COLOR[e[0]] for e in ids]
    fig, ax = plt.subplots(figsize=(9, 11))
    y = range(len(ids))
    ax.barh(list(y), counts, color=colors)
    ax.plot([co[e] for e in ids], list(y), "|", color="black", ms=9, mew=1.4,
            label="original (pre-clean)")
    ax.set_yticks(list(y))
    ax.set_yticklabels(ids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("number of rows citing this element")
    ax.set_title(f"Charter citation frequency — cleaned eval gold\n"
                 f"{sum(cn.values())} markers over {sum(1 for r in new if r)} cited rows "
                 f"(| = original count)")
    for i, c in enumerate(counts):
        if c:
            ax.text(c + 4, i, str(c), va="center", fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=DOMAIN_COLOR[d]) for d in "123456"]
    ax.legend(handles + [plt.Line2D([0], [0], marker="|", color="black", ls="", mew=1.4)],
              [f"{d} — {DOMAIN_NAME[d]}" for d in "123456"] + ["original (pre-clean)"],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "citation_by_element.png", dpi=130)
    plt.close(fig)

    # ---- Figure B: overview (cites/row + per-domain) ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    nb = Counter(len(r) for r in orig)
    na = Counter(len(r) for r in new)
    ks = [0, 1, 2, 3, 4]
    w = 0.38
    ax1.bar([k - w / 2 for k in ks], [nb.get(k, 0) for k in ks], w, label="original",
            color="0.7")
    ax1.bar([k + w / 2 for k in ks], [na.get(k, 0) for k in ks], w, label="cleaned",
            color="tab:blue")
    ax1.set_xticks(ks)
    ax1.set_xlabel("citations per row")
    ax1.set_ylabel("rows")
    ax1.set_title("Citations per row (original vs cleaned)")
    ax1.legend()

    dn = Counter()
    for e, c in cn.items():
        dn[e[0]] += c
    doms = list("123456")
    ax2.bar(doms, [dn[d] for d in doms], color=[DOMAIN_COLOR[d] for d in doms])
    ax2.set_xticklabels([f"{d}\n{DOMAIN_NAME[d].split(' ')[0]}" for d in doms], fontsize=8)
    ax2.set_ylabel("citation markers (cleaned)")
    tot = sum(dn.values())
    ax2.set_title("Markers by charter domain (cleaned)")
    for i, d in enumerate(doms):
        ax2.text(i, dn[d] + 20, f"{dn[d]/tot:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "citation_overview.png", dpi=130)
    plt.close(fig)
    print("wrote", OUT / "citation_by_element.png", "and", OUT / "citation_overview.png")


if __name__ == "__main__":
    main()
