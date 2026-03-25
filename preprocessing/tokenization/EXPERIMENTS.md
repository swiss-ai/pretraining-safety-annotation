# Tokenization experiments

## Full 20K-file run (2026-03-24)

### Configuration

- Input: `$SCRATCH/dolma3_mix-1T_subsampled/{unannotated,annotated}` (20K files each)
- Output: `$SCRATCH/tokenized/`
- Tokenizer: `HuggingFaceTB/SmolLM2-1.7B-Instruct`
- Seq length: 2048 (compact), 1920 content + 128 reflection budget (annotated)
- 20 SLURM array tasks (nodes), 20 workers each
- Job IDs: tokenize 1720966, merge 1720986

### Results

**Tokenize stage**: 18/20 nodes completed (2.5-3.1h wall time per node). Nodes 13 and 14 OOM'd on partitions with very large parquet files (MaxRSS ~360GB vs ~280-330GB for others).

- Node 13: 637/1000 compact tasks completed before OOM
- Node 14: 79/1000 compact tasks completed before OOM (single huge file)

Re-run needed for nodes 13, 14 with 10 workers (skip_completed resumes from where they left off).

**Merge stage**: blocked by OOM'd nodes (afterok dependency). Pending re-run.

### Node-hours estimate (from 100-file test)

| Stage | Node-hours |
|-------|----------:|
| Compact tokenize (20 nodes) | ~24 |
| Split tokenize (20 nodes) | ~3 |
| Merge + shuffle (1 node) | ~0.5 |
| **Total** | **~27.5** |

### 100-file test runs (2026-03-24)

**Single-node (slurm-1720343)**: 100 files, 20 workers, 14 min wall time.

| Pipeline | Documents | Tokens | Windows |
|----------|----------:|-------:|--------:|
| Compact | 2,794,349 | 2,747,222,216 | 1,340,762 |
| Annotated | 326,296 | 338,917,467 | 326,296 |

**4-node parallel (array-1720785 + 1720786)**: identical output to single-node. Tokenize 8 min (bottleneck: uneven file sizes), merge 40s.

### Lessons

- Max 20 workers per node due to OOM. Even 20 can OOM on partitions with very large files — reduce to 10 for those.
- File size variance causes load imbalance: node 14's partition had a single parquet that took >1h to tokenize.
- Compact shuffle was non-deterministic (seed=None). Fixed by passing seed=42.
