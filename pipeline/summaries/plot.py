"""Plot summary benchmark results.

Usage:
    uv run python -m pipeline.summaries.plot
    uv run python -m pipeline.summaries.plot --output figures/summary_quality.png
"""

from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from pipeline.summaries.storage import load_summary_items, load_summary_runs


def main():
    parser = argparse.ArgumentParser(description="Plot summary benchmark results")
    parser.add_argument(
        "--output", type=str, default="data/pipeline/summary_benchmark.png"
    )
    args = parser.parse_args()

    runs = load_summary_runs()
    if not runs:
        print("No summary runs found. Run a benchmark first.")
        return

    # Group runs by model, take latest per model
    latest_by_model: dict[str, dict] = {}
    for run in runs:
        model = run["generator_model"]
        if model not in latest_by_model:
            latest_by_model[model] = run
        else:
            if run["timestamp"] > latest_by_model[model]["timestamp"]:
                latest_by_model[model] = run

    models = sorted(latest_by_model.keys())
    if not models:
        print("No models found.")
        return

    # Collect data
    model_items: dict[str, list[dict]] = {}
    for model in models:
        run = latest_by_model[model]
        items = load_summary_items(run["run_id"])
        model_items[model] = items

    # Parse scores once per item
    def _parse_scores(item):
        s = item["scores"]
        return json.loads(s) if isinstance(s, str) else s

    model_scores: dict[str, list[dict]] = {}
    for model, items in model_items.items():
        model_scores[model] = [_parse_scores(item) for item in items]

    # Determine dimensions from first non-empty model
    dims = []
    for scores_list in model_scores.values():
        if scores_list:
            dims = list(scores_list[0].keys())
            break

    if not dims:
        print("No scored items found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Grouped bar chart (mean score per dimension per model) ---
    ax1 = axes[0]
    x = np.arange(len(dims))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        scores_list = model_scores[model]
        dim_means = []
        for dim in dims:
            vals = [s.get(dim, 0) for s in scores_list]
            dim_means.append(np.mean(vals) if vals else 0)
        ax1.bar(x + i * width, dim_means, width, label=model)

    ax1.set_xlabel("Dimension")
    ax1.set_ylabel("Mean Score")
    ax1.set_title("Summary Quality by Dimension")
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(dims, rotation=15)
    ax1.set_ylim(0, 5.5)
    ax1.legend(fontsize="small")
    ax1.grid(axis="y", alpha=0.3)

    # --- Plot 2: Box plot (aggregate distribution per model) ---
    ax2 = axes[1]
    box_data = []
    for model in models:
        items = model_items[model]
        box_data.append([item["aggregate"] for item in items])

    bp = ax2.boxplot(box_data, labels=models, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("Aggregate Score")
    ax2.set_title("Score Distribution by Model")
    ax2.set_ylim(0, 5.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
