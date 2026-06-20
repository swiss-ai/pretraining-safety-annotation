# Apertus Pretraining Safety Annotations — results page

Static site (no build step) summarising the candidate reflection generators:
a headline **throughput vs. aggregate-quality** plot, **per-language + edge-case**
accept rates, and an **inspector** for browsing actual generations and the judge's
per-dimension verdicts.

## Files

| file | role |
|------|------|
| `index.html` | page shell + the two tabs (Overview / Inspector) |
| `style.css`  | model-raising visual identity (Inter + cool-to-warm palette) |
| `app.js`     | Plotly charts, summary table, inspector logic |
| `data.js`    | **generated** — `window.CHARTER_DATA` with metrics + sampled records |
| `build_data.py` | regenerates `data.js` from the eval run dirs |

Plotly is loaded from CDN; nothing else is bundled.

## Regenerate the data

```bash
uv run python docs/build_data.py        # run from repo root
```

The run dirs and throughput numbers each model reads from are listed at the top of
`build_data.py` (`MODELS`). All quality numbers come from the judgments JSONL; the
GPU-hours are the newest per-model throughput measurement on the
`apertus-min-3800-v1` reflection cutoff. To add a model, append an entry to `MODELS`
(run ids + measured throughput) and re-run — the page picks it up automatically.

## View locally

```bash
python -m http.server -d docs 8000      # then open http://localhost:8000
```

(Opening `index.html` via `file://` also works.)

## Publish to GitHub Pages

Settings → Pages → Source: *Deploy from a branch* → branch `main`, folder `/docs`.
