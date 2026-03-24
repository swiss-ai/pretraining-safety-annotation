"""End-to-end test for annotation-based subsampling pipeline.

Creates a small synthetic dataset, runs the pipeline, and verifies:
- Two output directories (annotated + unannotated)
- Correct has_annotation and is_bad flags
- Annotation ratio
- Token budgets
- No duplicate IDs
- Determinism (same seed → same output)
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TEST_DIR = Path.home() / "tmp" / "test_subsample"
CHARS_PER_TOKEN = 4.068


def _setup_test_data(test_dir: Path) -> Path:
    """Create synthetic source parquet files.

    Returns source_dir.

    Layout:
    - 2 source files with 500 rows each (1000 total)
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
        ids_all.append(f"doc_{i:04d}")
        texts_all.append(f"This is test document number {i}. " + "x" * 65)
        if i < 25:
            scores_all.append(5)
        elif i < 50:
            scores_all.append(4)
        elif i < 75:
            scores_all.append(3)
        else:
            scores_all.append(i % 3)

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
            "--workers", "2",
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

    # --- Schema checks ---
    for name, table in [("annotated", annotated), ("unannotated", unannotated)]:
        col_names = table.column_names
        for required in ["id", "text", "safety_score", "has_annotation", "is_bad"]:
            assert required in col_names, f"{name}: missing {required} column"
        assert col_names.count("safety_score") == 1, f"{name}: duplicate safety_score"

    # --- Flag checks ---
    ann_flags = annotated.column("has_annotation").to_pylist()
    assert all(ann_flags), "Annotated split has has_annotation=False rows"

    unann_flags = unannotated.column("has_annotation").to_pylist()
    assert not any(unann_flags), "Unannotated split has has_annotation=True rows"

    # --- is_bad checks ---
    for name, table in [("annotated", annotated), ("unannotated", unannotated)]:
        scores = table.column("safety_score").to_pylist()
        is_bad = table.column("is_bad").to_pylist()
        for s, b in zip(scores, is_bad):
            assert b == (s >= 3), f"{name}: is_bad={b} but safety_score={s}"

    # --- No duplicate IDs across splits ---
    ann_ids = set(annotated.column("id").to_pylist())
    unann_ids = set(unannotated.column("id").to_pylist())
    overlap = ann_ids & unann_ids
    assert not overlap, f"IDs in both splits: {list(overlap)[:5]}"

    # --- Token budget check ---
    ann_tokens = sum(
        min(len(t) / CHARS_PER_TOKEN, 2048)
        for t in annotated.column("text").to_pylist()
    )
    unann_tokens = sum(
        min(len(t) / CHARS_PER_TOKEN, 2048)
        for t in unannotated.column("text").to_pylist()
    )
    total_tokens = ann_tokens + unann_tokens
    ann_pct = 100 * ann_tokens / total_tokens

    print(f"\nToken breakdown:")
    print(f"  Annotated:   {ann_tokens:,.0f} ({ann_pct:.2f}%)")
    print(f"  Unannotated: {unann_tokens:,.0f} ({100 - ann_pct:.2f}%)")
    print(f"  Total:       {total_tokens:,.0f}")
    print(f"  Annotated fraction: {ann_pct:.2f}% (expected ~15%)")
    assert 10.0 <= ann_pct <= 20.0, (
        f"Annotated fraction {ann_pct:.2f}% outside [10%, 20%]"
    )

    # --- Metadata checks ---
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists(), "metadata.json not found"
    meta = json.loads(meta_path.read_text())
    assert "annotation_threshold" in meta
    assert "annotation_ratio" in meta
    assert "selected_rows" in meta
    assert "selected_tokens" in meta
    assert "annotated" in meta
    assert "unannotated" in meta
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

    print("\nAll output checks passed!")


def _verify_determinism(source_dir: Path) -> None:
    """Run twice with same seed, verify identical outputs."""
    out1 = TEST_DIR / "det_run1"
    out2 = TEST_DIR / "det_run2"

    print("\n--- Determinism test ---")
    _run_subsample(source_dir, out1, target_tokens=5000)
    _run_subsample(source_dir, out2, target_tokens=5000)

    ids1 = set(_read_split(out1 / "annotated").column("id").to_pylist())
    ids2 = set(_read_split(out2 / "annotated").column("id").to_pylist())
    assert ids1 == ids2, "Annotated IDs differ between runs with same seed"

    ids1 = set(_read_split(out1 / "unannotated").column("id").to_pylist())
    ids2 = set(_read_split(out2 / "unannotated").column("id").to_pylist())
    assert ids1 == ids2, "Unannotated IDs differ between runs with same seed"

    print("Determinism check passed!")


def main() -> None:
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    source_dir = _setup_test_data(TEST_DIR)

    # Main end-to-end test
    output_dir = TEST_DIR / "output"
    _run_subsample(source_dir, output_dir)
    _verify_output(output_dir)

    # Determinism test
    _verify_determinism(source_dir)

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("\nTest directory cleaned up.")


if __name__ == "__main__":
    main()
