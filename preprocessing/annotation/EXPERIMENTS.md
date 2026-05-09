# Annotation experiments

## 20K-file dolma3 run (2026-03-23)

### Configuration

- 33 tasks × ~607 files, 4×GH200 per task
- Source: `$SCRATCH/dolma3_mix-1T/` (20K of 47,142 available shards)
- Annotations: `$SCRATCH/safety_annotations/dolma3/`
- Merged output: `$SCRATCH/dolma3_mix-1T_annotated/`
- HF upload: `jkminder/dolma3-safety-annotations`

### Results

| | GPU-h |
|---|---:|
| Successful runs (33 tasks) | 579 |
| Failed first run (NCCL barrier, 17 tasks resubmitted) | ~340 |
| Estimation + test runs | ~50 |
| **Total spent** | **~970** |

Wall-clock per task: 4.0–4.7h (mean 4.4h), faster than estimate due to conservative calibration.

### Score distribution

**Deduplicated (391,474,169 unique annotations — uploaded to [jkminder/dolma3-safety-annotations](https://huggingface.co/datasets/jkminder/dolma3-safety-annotations)):**

| Score | Label | Count | % |
|-------|-------|------:|---:|
| 0 | safe | 302,972,734 | 77.39% |
| 1 | minimal | 38,143,123 | 9.74% |
| 2 | mild | 32,004,998 | 8.18% |
| 3 | moderate | 10,592,853 | 2.71% |
| 4 | significant | 3,990,755 | 1.02% |
| 5 | severe | 3,769,706 | 0.96% |

Safe (0-1): 87.1% | Unsafe (2-5): 12.9%

**Merged with upsampling (1,180,506,475 rows — in `$SCRATCH/dolma3_mix-1T_annotated/`):**

| Score | Label | Count | % |
|-------|-------|------:|---:|
| 0 | safe | 926,715,941 | 78.50% |
| 1 | minimal | 107,477,673 | 9.10% |
| 2 | mild | 92,731,623 | 7.86% |
| 3 | moderate | 31,659,336 | 2.68% |
| 4 | significant | 11,500,784 | 0.97% |
| 5 | severe | 10,421,118 | 0.88% |

Safe (0-1): 87.6% | Unsafe (2-5): 12.4%

Note: upsampled distribution skews slightly more toward "safe" because higher-quality documents (which tend to be safer) are repeated more often.

### Incidents

#### NCCL barrier hang at teardown

During the first 20K-file run, 17/33 tasks failed because `torch.distributed.barrier()` in `teardown_distributed()` hung indefinitely after inference completed. All data was already flushed and writers closed, but the barrier blocked until SLURM killed the job — preventing the DONE marker from being written.

**Root cause:** NCCL barrier with no timeout. On GH200 nodes, some ranks finish the barrier faster than others; when the gap exceeds NCCL's internal timeout, the collective deadlocks.

**Fix (dadfe1b):**
1. Added 120s timeout to the barrier — if stuck, logs a warning and proceeds to `destroy_process_group()`
2. Moved DONE marker write before the barrier — data integrity doesn't depend on the barrier since all writers are already closed

**Impact:** 16/33 tasks completed on the first attempt, 17 had to be resubmitted. No data was lost (resume handled partial shards), but the resubmission restarted most ranks from scratch due to intermediate cancel corrupting in-progress writes.

#### Cross-file duplicate IDs break merge assertion

Per-file dedup keeps one occurrence of each ID *per file*, but the same ID can appear in different files (cross-file duplicates). The merge builds a global `id → score` dict which deduplicates these, so `len(dict) < n_input_rows`. The original assertion `len(id_to_score) == n_input_rows` failed on task_0000 (4 cross-file duplicates, including one `None` ID).

**Fix (52e41ab):** Assert on total shard rows instead of unique dict entries. Log the cross-file overlap count.

#### Schema mismatch on file 0

`load_dataset("parquet", ...)` loads all columns by default. Some parquet files have `source: null` type while others have `source: string`, causing HF datasets to fail on schema reconciliation. Fixed by passing `columns=[id_column, text_column]` to only load needed columns.

#### Classifier precision audit (2026-03-23)

The safety classifier (`locuslab/safety-classifier_gte-large-en-v1.5`) was audited for precision by re-classifying random samples with Claude Opus 4.6. Samples were drawn uniformly across 163–200 random files from the 20K-file annotated dataset.

**Score 5 precision (n=975, from 163 files):**

| Reclassified Score | Count | % |
|---|---|---|
| 0 (safe) | 392 | 40.2% |
| 1 (minimal) | 201 | 20.6% |
| 2 (mild) | 165 | 16.9% |
| 3 (moderate) | 132 | 13.5% |
| 4 (significant) | 59 | 6.1% |
| 5 (severe) | 26 | 2.7% |

- Precision (strict, reclassified=5): **2.7%**
- Precision (relaxed, reclassified≥4): **8.7%**
- Precision (lenient, reclassified≥3): **22.3%**
- 61% of score-5 predictions are completely benign (scores 0-1)

Common false-positive triggers: IP/WHOIS lookups, software piracy keywords ("crack", "keygen"), SEO spam with male enhancement keywords, safety data sheets, source code.

**Score 3-4 recall check — does severe content leak into lower scores?**

| Original Score | n | Reclassified as 5 | % | Reclassified ≥ orig | % |
|---|---|---|---|---|---|
| 3 | 500 | 2 | 0.4% | 75 (≥3) | 15.0% |
| 4 | 500 | 1 | 0.2% | 11 (≥4) | 2.2% |

Only 3/1000 score 3-4 samples should actually be score 5 (credit card fraud tutorial, chemical weapons synthesis manual, TNT synthesis instructions). The classifier rarely under-scores truly severe content.

However, the classifier also massively over-triggers at scores 3-4: 85% of score-3 and 98% of score-4 samples were downgraded on re-examination (mostly to 0-1).

**Summary:** The classifier has high recall but very low precision across all score levels. It catches most severe content somewhere in scores 3-5, but the vast majority of flagged content is benign. Filtering at score ≥ 5 discards ~97% clean data; filtering at score ≥ 3 discards ~85% clean data.

### Data notes

- `$SCRATCH/dolma3_mix-1T/` contains 47,142 shards (~4.03T tokens), not 1T as the directory name suggests
- Download is shuffled, so the first N files are a representative subset
- ~66% within-file dedup (quality-aware upsampling)
