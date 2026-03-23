"""End-to-end test for stratified subsampling pipeline.

Creates a small synthetic dataset with known score distribution, runs the
subsampling script, and verifies token budgets, annotation ratios, and output
schema.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

TEST_DIR = Path.home() / "tmp" / "test_subsample"


def _setup_test_data(test_dir: Path) -> Path:
    """Create synthetic source parquet files with safety_score included.

    Returns source_dir.

    Layout:
    - 2 source files with 500 rows each (1000 total)
    - Each row ~100 chars of text → ~24.6 tokens at 4.068 chars/token
    - Total ~24,600 tokens
    - Score distribution: 50 bad (score 4-5), 950 good (score 0-3)
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
        # First 25 rows: score 4, next 25: score 5, rest: scores 0-3
        if i < 25:
            scores_all.append(4)
        elif i < 50:
            scores_all.append(5)
        else:
            scores_all.append(i % 4)  # 0, 1, 2, 3

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


def _run_subsample(source_dir: Path, output_dir: Path) -> str:
    """Run the subsample script and return stdout."""
    result = subprocess.run(
        [
            sys.executable, "-m", "preprocessing.subsample_and_stratify.subsample",
            "--source-dir", str(source_dir),
            "--output-dir", str(output_dir),
            "--target-tokens", "10000",
            "--bad-fraction", "0.025",
            "--good-fraction", "0.025",
            "--chars-per-token", "4.068",
            "--seed", "42",
            "--rows-per-file", "200",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    assert result.returncode == 0, f"Script failed with code {result.returncode}"
    return result.stdout


def _verify_output(output_dir: Path) -> None:
    """Verify output schema, metadata, and annotation ratios."""
    # Check output files exist
    output_files = sorted(output_dir.glob("part_*.parquet"))
    assert output_files, "No output parquet files found"
    print(f"\nOutput: {len(output_files)} files")

    # Read all output
    tables = [pq.read_table(f) for f in output_files]
    combined = pa.concat_tables(tables)
    print(f"Total output rows: {len(combined):,}")

    # Check schema has required columns
    col_names = combined.column_names
    assert "safety_score" in col_names, f"Missing safety_score column. Columns: {col_names}"
    assert col_names.count("safety_score") == 1, f"Duplicate safety_score column. Columns: {col_names}"
    assert "has_annotation" in col_names, f"Missing has_annotation column. Columns: {col_names}"
    assert "id" in col_names, f"Missing id column. Columns: {col_names}"
    assert "text" in col_names, f"Missing text column. Columns: {col_names}"

    # Check types
    schema = combined.schema
    assert schema.field("safety_score").type == pa.int8(), (
        f"safety_score should be int8, got {schema.field('safety_score').type}"
    )
    assert schema.field("has_annotation").type == pa.bool_(), (
        f"has_annotation should be bool, got {schema.field('has_annotation').type}"
    )

    # Check annotation ratios
    has_ann = combined.column("has_annotation").to_pylist()
    scores = combined.column("safety_score").to_pylist()
    texts = combined.column("text").to_pylist()
    chars_per_token = 4.068

    annotated_bad_tokens = sum(
        len(t) / chars_per_token
        for t, a, s in zip(texts, has_ann, scores)
        if a and s >= 4
    )
    annotated_good_tokens = sum(
        len(t) / chars_per_token
        for t, a, s in zip(texts, has_ann, scores)
        if a and s < 4
    )
    total_tokens = sum(len(t) / chars_per_token for t in texts)
    unmarked_tokens = total_tokens - annotated_bad_tokens - annotated_good_tokens

    print(f"\nToken breakdown:")
    print(f"  Total:          {total_tokens:,.0f}")
    print(f"  Bad annotated:  {annotated_bad_tokens:,.0f} ({100*annotated_bad_tokens/total_tokens:.2f}%)")
    print(f"  Good annotated: {annotated_good_tokens:,.0f} ({100*annotated_good_tokens/total_tokens:.2f}%)")
    print(f"  Unmarked:       {unmarked_tokens:,.0f} ({100*unmarked_tokens/total_tokens:.2f}%)")

    # Annotated should be ~5% total (2.5% bad + 2.5% good)
    annotated_pct = 100 * (annotated_bad_tokens + annotated_good_tokens) / total_tokens
    print(f"  Annotated fraction: {annotated_pct:.2f}% (expected ~5%)")
    assert 3.0 <= annotated_pct <= 8.0, (
        f"Annotated fraction {annotated_pct:.2f}% outside reasonable range [3%, 8%]"
    )

    # All annotated bad rows should have score >= 4
    for a, s in zip(has_ann, scores):
        if a and s >= 4:
            assert s in (4, 5), f"Bad annotated row has unexpected score {s}"

    # Check no duplicate IDs in output
    ids = combined.column("id").to_pylist()
    assert len(ids) == len(set(ids)), (
        f"Duplicate IDs in output: {len(ids)} total, {len(set(ids))} unique"
    )

    # Check metadata.json
    meta_path = output_dir / "metadata.json"
    assert meta_path.exists(), "metadata.json not found"
    meta = json.loads(meta_path.read_text())
    assert "target_tokens" in meta, "metadata missing target_tokens"
    assert "selected_rows" in meta, "metadata missing selected_rows"
    assert "bad_annotated_rows" in meta, "metadata missing bad_annotated_rows"
    assert "good_annotated_rows" in meta, "metadata missing good_annotated_rows"
    print(f"\nMetadata: {json.dumps(meta, indent=2)}")

    print("\nAll checks passed!")


def main() -> None:
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    source_dir = _setup_test_data(TEST_DIR)
    output_dir = TEST_DIR / "output"
    _run_subsample(source_dir, output_dir)
    _verify_output(output_dir)

    # Cleanup
    shutil.rmtree(TEST_DIR)
    print("Test directory cleaned up.")


if __name__ == "__main__":
    main()
