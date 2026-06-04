import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("pipeline/assets/model_selection")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linestyle": "-"})

# ---- verified data (charter.eval ref_v3 + ref_v4_qwen, gold judge Kimi-K2.5; throughput README) ----
MODELS = ["Qwen3.5-35B-A3B", "Nemotron-3-Super-120B", "gpt-oss-120b", "GLM-4.5-Air"]
COLOR  = {"Qwen3.5-35B-A3B":"#2ca02c", "Nemotron-3-Super-120B":"#1f77b4",
          "gpt-oss-120b":"#ff7f0e", "GLM-4.5-Air":"#9467bd"}
QUALITY = {"Qwen3.5-35B-A3B":4.498, "Nemotron-3-Super-120B":4.472,
           "gpt-oss-120b":4.422, "GLM-4.5-Air":4.384}          # mean judge aggregate (1-5)
ACCEPT  = {"Qwen3.5-35B-A3B":0.954, "Nemotron-3-Super-120B":0.934,
           "gpt-oss-120b":0.906, "GLM-4.5-Air":0.888}
# GPU-hours to annotate 102M docs (4-voice combined prompt, tuned SGLang where available)
COST    = {"Qwen3.5-35B-A3B":26582, "Nemotron-3-Super-120B":28848,
           "gpt-oss-120b":10824, "GLM-4.5-Air":32035}
DIMS = ["relevance","specificity","charter_grounding","voice_tone"]
DIMSCORES = {
  "Qwen3.5-35B-A3B":      [4.772,4.118,4.732,4.370],
  "Nemotron-3-Super-120B":[4.729,4.050,4.651,4.460],
  "gpt-oss-120b":         [4.639,4.095,4.605,4.349],
  "GLM-4.5-Air":          [4.647,4.040,4.546,4.302],
}
SAFETY = [0,1,2,3,4,5]
ACCEPT_BY_SAFETY = {
  "Qwen3.5-35B-A3B":      [0.976,0.929,0.928,0.918,0.911,0.967],
  "Nemotron-3-Super-120B":[0.984,0.925,0.929,0.859,0.861,0.868],
  "gpt-oss-120b":         [0.958,0.896,0.888,0.825,0.844,0.833],
  "GLM-4.5-Air":          [0.956,0.875,0.832,0.787,0.766,0.860],
}

def short(m): return {"Qwen3.5-35B-A3B":"Qwen3.5-35B-A3B","Nemotron-3-Super-120B":"Nemotron-3\nSuper-120B",
                      "gpt-oss-120b":"gpt-oss-120b","GLM-4.5-Air":"GLM-4.5-Air"}[m]

# ===== 1. Cost vs Quality (the decision plot) =====
fig, ax = plt.subplots(figsize=(7.8,5.6))
for m in MODELS:
    x, y = COST[m]/1000, QUALITY[m]
    chosen = (m=="Qwen3.5-35B-A3B")
    ax.scatter(x, y, s=420 if chosen else 230, c=COLOR[m], edgecolor="black",
               linewidth=1.8 if chosen else 0.8, zorder=5, marker="*" if chosen else "o")
    dy = 0.006 if m!="gpt-oss-120b" else 0.006
    ha = "left"
    ax.annotate(f"{m}\n{COST[m]/1000:.1f}K GPU-h · {QUALITY[m]:.3f}",
                (x,y), xytext=(8, 10 if not chosen else 14), textcoords="offset points",
                fontsize=9.5, fontweight="bold" if chosen else "normal",
                color=COLOR[m] if chosen else "#333")
ax.axhspan(QUALITY["Qwen3.5-35B-A3B"]-0.003, 4.52, color="#2ca02c", alpha=0.05)
ax.set_xlabel("Cost — GPU-hours / 102M docs   (lower better ↓)")
ax.set_ylabel("Annotation quality — Kimi-K2.5 judge (1–5)   (higher better ↑)", fontsize=10)
ax.set_title("Annotation-model selection: quality vs. cost", fontweight="bold")
ax.set_xlim(5, 36); ax.set_ylim(4.36, 4.52)
ax.annotate("chosen ✔", (26.582, 4.498), xytext=(20.5, 4.508), fontsize=10,
            fontweight="bold", color="#2ca02c")
fig.text(0.5, -0.02,
  "gpt-oss is cheapest but lowest quality; Qwen wins on quality and Pareto-dominates Nemotron & GLM (better AND cheaper).",
  ha="center", fontsize=8.5, color="#555")
fig.savefig(OUT/"cost_vs_quality.png"); plt.close(fig)

# ===== 2. Quality ranking bar =====
fig, ax = plt.subplots(figsize=(7.2,4.4))
order = sorted(MODELS, key=lambda m: QUALITY[m])
ys = np.arange(len(order))
ax.barh(ys, [QUALITY[m] for m in order], color=[COLOR[m] for m in order],
        edgecolor="black", linewidth=0.6)
for i,m in enumerate(order):
    ax.text(QUALITY[m]+0.002, i, f"{QUALITY[m]:.3f}  ·  {ACCEPT[m]:.0%} accept",
            va="center", fontsize=9.5, fontweight="bold" if m=="Qwen3.5-35B-A3B" else "normal")
ax.set_yticks(ys); ax.set_yticklabels([m for m in order])
ax.set_xlim(4.30, 4.56)
ax.set_xlabel("Mean Kimi-K2.5 judge aggregate (1–5) over 5K diverse FineWeb/dolma3 docs")
ax.set_title("Generator quality ranking (reflection annotation)", fontweight="bold")
fig.savefig(OUT/"quality_ranking.png"); plt.close(fig)

# ===== 3. Robustness by safety score (the differentiator) =====
fig, ax = plt.subplots(figsize=(7.6,4.8))
for m in MODELS:
    ax.plot(SAFETY, [100*v for v in ACCEPT_BY_SAFETY[m]], marker="o",
            color=COLOR[m], linewidth=2.4 if m=="Qwen3.5-35B-A3B" else 1.6,
            markersize=7 if m=="Qwen3.5-35B-A3B" else 5,
            label=m, zorder=5 if m=="Qwen3.5-35B-A3B" else 3)
ax.set_xlabel("Document safety score   (0 = benign  →  5 = most harmful / ethically loaded)")
ax.set_ylabel("Judge accept rate (%)")
ax.set_title("Robustness on harmful content — where the choice is decided", fontweight="bold")
ax.legend(frameon=False, fontsize=9.5, loc="lower left")
ax.set_ylim(74, 100)
ax.annotate("Qwen holds ~92–97% on the\nhardest content; others drop to 77–86%",
            (4, 91.1), xytext=(2.3, 95.5), fontsize=9, color="#2ca02c", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))
fig.savefig(OUT/"robustness_by_safety.png"); plt.close(fig)

# ===== 4. Per-dimension grouped bars =====
fig, ax = plt.subplots(figsize=(8.2,4.6))
x = np.arange(len(DIMS)); w = 0.2
for i,m in enumerate(MODELS):
    ax.bar(x + (i-1.5)*w, DIMSCORES[m], w, label=m, color=COLOR[m],
           edgecolor="black", linewidth=0.4)
ax.set_xticks(x); ax.set_xticklabels([d.replace("_","\n") for d in DIMS])
ax.set_ylim(3.9, 4.9); ax.set_ylabel("Mean judge score (1–5)")
ax.set_title("Quality by rubric dimension", fontweight="bold")
ax.legend(frameon=False, fontsize=9, ncol=2)
fig.savefig(OUT/"quality_dimensions.png"); plt.close(fig)

print("wrote:", *[p.name for p in sorted(OUT.glob('*.png'))])
