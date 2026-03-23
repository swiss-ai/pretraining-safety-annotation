"""End-to-end test for annotation-based subsampling pipeline.

Creates a small synthetic dataset with known score distribution, runs the
subsampling script, and verifies:
- Two output directories (annotated + unannotated)
- Correct has_annotation flags
- Annotation ratio (2x high-score token fraction)
- Token budgets
- No duplicate IDs
- Monotonic subset guarantee (5K ⊂ 10K)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TEST_DIR = Path.home() / "tmp" / "test_subsample"
CHARS_PER_TOKEN = 4.068


def _setup_test_data(test_dir: Path) -> Path:
    """Create synthetic source parquet files with safety_score included.

    Returns source_dir.

    Layout:
    - 2 source files with 500 rows each (1000 total)
    - Each row ~100 chars of text → ~24.6 tokens at 4.068 chars/token
    - Total ~24,600 tokens
    - Score distribution:
        0–24:  score 5 (25 rows)
        25–49: score 4 (25 rows)
        50–74: score 3 (25 rows)  ← threshold=3 boundary
        75–999: scores 0–2 cycling (925 rows)
    - With threshold=3: 75 rows (7.5%) unconditionally annotated
    """
    source_dir = test_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    ids_all = []
    texts_all = []
    scores_all = []

    for i in range(1000):
        doc_id = f"doc_{i:04d}"
        text = f"This is test document number {i}. " + "x" * 65
        ids_all.append(doc_id)
        texts_all.append(text)
        if i < 25:
            scores_all.append(5)
        elif i < 50:
            scores_all.append(4)
        elif i < 75:
            scores_all.append(3)
        else:
            scores_all.append(i % 3)  # 0, 1, 2

    for part_idx in range(2):
        start = part_idx * 500
        end = start + 500
        table = pa.table({
            "id": ids_all[start:end],
            "text": texts_all[start:end],
            "source": ["test"] * 500,
            "safety_score": pa.array(scores_all[start:end], type=pa.int8()),
        })
        pq.write_table(table, source_dir / f"part_{part_idx:05d}.parquet")

    return source_dir


def _run_subsample(source_dir: Path, output_dir: Path,
                   target_tokens: int = 10000) -> str:
    """Run the subsample script and return stdout."""
    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.subsample_and_stratify.subsample",
            "--source-dir", str(source_dir),
            "--output-dir", str(output_dir),
            "--target-tokens", str(target_tokens),
            "--chars-per-token", str(CHARS_PER_TOKEN),
            "--seed", "42",
            "--rows-per-file", "200",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, (
        f"Script failed with code {result.returncode}\n"
        f"stderr: {result.stderr}"
    )
    return result.stdout


def _read_split(split_dir: Path) -> pa.Table:
    """Read all parquet files in a split directory."""
    files = sorted(split_dir.glob("part_*.parquet"))
    assert files, f"No part_*.parquet files in {split_dir}"
    return pa.concat_tables([pq.read_table(f) for f in files])


def _verify_output(output_dir: Path) -> None:
    """Verify two-dir output schema, metadata, and annotation ratios."""
    ann_dir = output_dir / "annotated"
    unann_dir = output_dir / "unannotated"

    assert ann_dir.exists(), f"Missing annotated directory: {ann_dir}"
    assert unann_dir.exists(), f"Missing unannotated directory: {unann_dir}"

    annotated = _read_split(ann_dir)
    unannotated = _read_split(unann_dir)

    print(f"\nAnnotated:   {len(annotated):,} rows")
    print(f"Unannotated: {len(unannotated):,} rows")

    # --- Schema checks (both splits) ---
    for name, table in [("annotated", annotated), ("unannotated", unannotated)]:
        col_names = table.column_names
        assert "id" in col_names, f"{name}: missing id column"
        assert "text" in col_names, f"{name}: missing text column"
        assert "safety_score" in col_names, f"{name}: missing safety_score"
        assert col_names.count("safety_score") == 1, f"{name}: duplicate safety_score"
        assert "has_annotation" in col_names, f"{name}: missing has_annotation"

        schema = table.schema
        assert schema.field("safety_score").type == pa.int8(), (
            f"{name}: safety_score should be int8, got {schema.field('safety_score').type}"
        )
        assert schema.field("has_annotation").type == pa.bool_(), (
            f"{name}: has_annotation should be bool, got {schema.field('has_annotation').type}"
        )

    # --- Flag checks ---
    ann_flags = annotated.column("has_annotation").to_pylist()
    assert all(ann_flags), "Annotated split contains rows with has_annotation=False"

    unann_flags = unannotated.column("has_annotation").to_pylist()
    assert not any(unann_flags), "Unannotated split contains rows with has_annotation=True"

    # --- No duplicate IDs across both splits ---
    ann_ids = set(annotated.column("id").to_pylist())
    unann_ids = set(unannotated.column("id").to_pylist())
    overlap = ann_ids & unann_ids
    assert not overlap, f"IDs appear in both splits: {list(overlap)[:5]}"
    total_ids = len(ann_ids) + len(unann_ids)
    print(f"Total unique IDs: {total_ids}")

    # --- Token budget check ---
    ann_tokens = sum(
        len(t) / CHARS_PER_TOKEN
        for t in annotated.column("text").to_pylist()
    )
    unann_tokens = sum(
        len(t) / CHARS_PER_TOKEN
        for t in unannotated.column("text").to_pylist()
    )
    total_tokens = ann_tokens + unann_tokens
    print(f"\nToken breakdown:")
    print(f"  Annotated:   {ann_tokens:,.0f} ({100*ann_tokens/total_tokens:.2f}%)")
    print(f"  Unannotated: {unann_tokens:,.0f} ({100*unann_tokens/total_tokens:.2f}%)")
    print(f"  Total:       {total_tokens:,.0f}")

    # Annotation ratio should be ~2x the high-score fraction.
    # With 75/1000 high-score rows (all same text length), expect ~15% annotated.
    ann_pct = 100 * ann_tokens / total_tokens
    print(f"  Annotated fraction: {ann_pct:.2f}% (expected ~15%)")
    assert 10.0 <= ann_pct <= 20.0, (
        f"Annotated fraction {ann_pct:.2f}% outside reasonable range [10%, 20%]"
    )

    # --- Metadata checks ---
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists(), "metadata.json not found"
    meta = json.loads(meta_path.read_text())
    assert "annotation_threshold" in meta, "metadata missing annotation_threshold"
    assert "annotation_ratio" in meta, "metadata missing annotation_ratio"
    assert "selected_rows" in meta, "metadata missing selected_rows"
    assert "selected_tokens" in meta, "metadata missing selected_tokens"
    assert "annotated" in meta, "metadata missing annotated sub-dict"
    assert "unannotated" in meta, "metadata missing unannotated sub-dict"
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

    print("\nAll output checks passed!")


def _verify_monotonic(source_dir: Path) -> None:
    """Verify monotonic subset: rows at 5K tokens ⊂ rows at 10K tokens.

    Runs the pipeline twice with different budgets and checks that the
    smaller budget's output IDs are a strict subset of the larger budget's.
    """
    out_small = TEST_DIR / "output_5k"
    out_large = TEST_DIR / "output_10k"

    print("\n--- Monotonic subset test ---")
    print("Running at 5K tokens...")
    _run_subsample(source_dir, out_small, target_tokens=5000)

    print("Running at 10K tokens...")
    _run_subsample(source_dir, out_large, target_tokens=10000)

    # Collect IDs from each run
    small_ann_ids = set(_read_split(out_small / "annotated").column("id").to_pylist())
    small_unann_ids = set(_read_split(out_small / "unannotated").column("id").to_pylist())
    large_ann_ids = set(_read_split(out_large / "annotated").column("id").to_pylist())
    large_unann_ids = set(_read_split(out_large / "unannotated").column("id").to_pylist())

    small_all = small_ann_ids | small_unann_ids
    large_all = large_ann_ids | large_unann_ids

    print(f"\n5K:  {len(small_all)} total ({len(small_ann_ids)} ann + {len(small_unann_ids)} unann)")
    print(f"10K: {len(large_all)} total ({len(large_ann_ids)} ann + {len(large_unann_ids)} unann)")

    # Every row in the small output must appear in the large output
    missing = small_all - large_all
    assert not missing, (
        f"Monotonic violation: {len(missing)} IDs in 5K output but not in 10K output. "
        f"Examples: {list(missing)[:5]}"
    )

    # The large output should be strictly bigger
    assert len(large_all) > len(small_all), (
        f"10K output ({len(large_all)}) should have more rows than 5K ({len(small_all)})"
    )

    # Annotated IDs at 5K should be a subset of annotated IDs at 10K
    # (a row's has_annotation flag never changes between budgets)
    ann_missing = small_ann_ids - large_ann_ids
    assert not ann_missing, (
        f"Annotated monotonic violation: {len(ann_missing)} annotated IDs in 5K but "
        f"not annotated in 10K. Examples: {list(ann_missing)[:5]}"
    )

    unann_missing = small_unann_ids - large_unann_ids
    assert not unann_missing, (
        f"Unannotated monotonic violation: {len(unann_missing)} unannotated IDs in 5K "
        f"but not unannotated in 10K. Examples: {list(unann_missing)[:5]}"
    )

    print("Monotonic subset check passed!")


def main() -> None:
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    source_dir = _setup_test_data(TEST_DIR)

    # Main end-to-end test
    output_dir = TEST_DIR / "output"
    _run_subsample(source_dir, output_dir)
    _verify_output(output_dir)

    # Monotonic subset test
    _verify_monotonic(source_dir)

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("\nTest directory cleaned up.")


if __name__ == "__main__":
    main()
