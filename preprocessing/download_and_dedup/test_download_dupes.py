"""Download a few shards and check for within-file duplicates."""
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

SCRATCH = "/iopsstor/scratch/cscs/jminder"
OUT_DIR = f"{SCRATCH}/test_download_dupes"

# Clean up any previous test
import shutil
if Path(OUT_DIR).exists():
    shutil.rmtree(OUT_DIR)

# Download 10 shards with 4 workers (enough to trigger the bug if present)
result = subprocess.run(
    [
        sys.executable, "-m", "preprocessing.download_and_dedup.download",
        "--dataset", "allenai/dolma3_mix-6T",
        "--n-shards", "10",
        "--output-dir", OUT_DIR,
        "--shuffle", "--seed", "42",
        "--columns", "text", "id", "source",
        "--ignore-errors",
        "--workers", "4",
    ],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)
if result.returncode != 0:
    print(f"Download failed with code {result.returncode}")
    sys.exit(1)

# Check each parquet file for within-file duplicate IDs
files = sorted(Path(OUT_DIR).glob("part_*.parquet"))
print(f"\nChecking {len(files)} files for within-file duplicate IDs...")
any_dupes = False
for f in files:
    table = pq.read_table(f, columns=["id"])
    ids = table["id"].to_pylist()
    unique = set(ids)
    if len(ids) != len(unique):
        n_dupes = len(ids) - len(unique)
        print(f"  {f.name}: {len(ids)} rows, {len(unique)} unique → {n_dupes} DUPLICATES")
        any_dupes = True
    else:
        print(f"  {f.name}: {len(ids)} rows, all unique ✓")

if any_dupes:
    print("\nFAIL: duplicates found!")
    sys.exit(1)
else:
    print("\nPASS: no duplicates in any file")
