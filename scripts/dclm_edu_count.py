"""Exact document count of the dclm-edu source corpus by summing parquet
row counts (footer metadata only — no data read). Parallelized over shards."""
import glob
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq

DATA_DIR = "/capstor/store/cscs/swissai/infra01/users/vvmoskvoretskii/safety_labels/dclm-edu-filterrobots_fine/data"
files = sorted(glob.glob(f"{DATA_DIR}/*.parquet"))

def nrows(f):
    return pq.ParquetFile(f).metadata.num_rows

with ThreadPoolExecutor(max_workers=32) as ex:
    counts = list(ex.map(nrows, files))

total = sum(counts)
print(f"shards:        {len(files):,}")
print(f"total docs:    {total:,}")
print(f"per-shard:     min={min(counts):,}  max={max(counts):,}  mean={total/len(files):,.0f}")
print(f"distinct sizes: {sorted(set(counts))[:6]}{' ...' if len(set(counts))>6 else ''}")
print(f"\n=> {total/1e9:.3f}B documents")
