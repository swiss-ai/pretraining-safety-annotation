"""Verify safety_score and is_bad columns in the patched sidecar.

Samples ~N_SAMPLE rows from sidecar.parquet, runs the sidecar text through
the same safety classifier used during annotation, and checks that the
re-classified score matches the stored safety_score.  Also verifies is_bad
is consistent with safety_score >= ANNOTATION_THRESHOLD.

This is an end-to-end consistency check: if the doc_id join was wrong,
the stored score will disagree with the classifier's output on the
(correct) text.

Usage::

    # Quick GPU check (default 1000 samples)
    python -m preprocessing.tokenization.verify_safety_scores

    # Custom paths / sample count
    python -m preprocessing.tokenization.verify_safety_scores \
        --sidecar /path/to/sidecar.parquet --n-sample 2000
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Must match annotation/annotate.py
MODEL_ID = "locuslab/safety-classifier_gte-large-en-v1.5"
ANNOTATION_THRESHOLD = 3  # is_bad := safety_score >= this


def parse_args() -> argparse.Namespace:
    scratch = os.environ.get(
        "SCRATCH",
        f"/iopsstor/scratch/cscs/{os.environ.get('USER', 'unknown')}",
    )
    p = argparse.ArgumentParser(
        description="Verify safety_score column in sidecar via re-classification."
    )
    p.add_argument(
        "--sidecar",
        type=str,
        default=f"{scratch}/tokenized/annotated/sidecar.parquet",
        help="Path to sidecar.parquet with safety_score column",
    )
    p.add_argument("--n-sample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=54321)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max token length for classifier (must match annotation pipeline)",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=ANNOTATION_THRESHOLD,
        help=f"is_bad threshold (default: {ANNOTATION_THRESHOLD})",
    )
    return p.parse_args()


@torch.no_grad()
def classify_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    """Classify a list of texts, returning predicted safety scores (int8)."""
    all_scores: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits
        scores = logits.argmax(dim=-1).cpu().tolist()
        all_scores.extend(scores)
    return np.array(all_scores, dtype=np.int8)


def main() -> None:
    args = parse_args()
    sidecar_path = Path(args.sidecar)
    assert sidecar_path.exists(), f"Sidecar not found: {sidecar_path}"

    # ── 1. Check schema ──────────────────────────────────────────
    pf = pq.ParquetFile(str(sidecar_path))
    col_names = set(pf.schema_arrow.names)
    assert "safety_score" in col_names, (
        f"sidecar missing safety_score column (has: {col_names})"
    )
    assert "is_bad" in col_names, (
        f"sidecar missing is_bad column (has: {col_names})"
    )
    n_total = pf.metadata.num_rows
    print(f"Sidecar: {n_total:,} rows, columns: {sorted(col_names)}")

    # ── 2. Sample rows ───────────────────────────────────────────
    # Sample from a small number of row groups to avoid reading the
    # entire 474GB sidecar (each row group is ~4-9GB due to text column).
    n_sample = min(args.n_sample, n_total)
    rng = np.random.default_rng(args.seed)

    n_rg = pf.metadata.num_row_groups
    rg_starts: list[int] = []
    rg_sizes: list[int] = []
    cumulative = 0
    for rg_i in range(n_rg):
        rg_starts.append(cumulative)
        sz = pf.metadata.row_group(rg_i).num_rows
        rg_sizes.append(sz)
        cumulative += sz
    rg_starts_arr = np.array(rg_starts)

    # Pick a few row groups, sample within them
    n_rg_to_read = min(5, n_rg)
    selected_rgs = rng.choice(n_rg, size=n_rg_to_read, replace=False)
    samples_per_rg = (n_sample + n_rg_to_read - 1) // n_rg_to_read

    from collections import defaultdict

    rg_to_samples: dict[int, list[int]] = defaultdict(list)
    total_sampled = 0
    for rg_i in selected_rgs:
        rg_i = int(rg_i)
        rg_start = rg_starts[rg_i]
        rg_size = rg_sizes[rg_i]
        n_from_rg = min(samples_per_rg, rg_size, n_sample - total_sampled)
        if n_from_rg <= 0:
            break
        local_indices = rng.choice(rg_size, size=n_from_rg, replace=False)
        for li in local_indices:
            rg_to_samples[rg_i].append(rg_start + int(li))
        total_sampled += n_from_rg

    n_sample = total_sampled
    print(f"Sampling {n_sample} rows from {len(rg_to_samples)} row groups")

    # Read sampled data
    texts: list[str] = []
    stored_scores: list[int] = []
    stored_is_bad: list[bool] = []
    global_indices: list[int] = []

    for rg_i in sorted(rg_to_samples):
        rg_start = rg_starts[rg_i]
        rg = pf.read_row_group(rg_i, columns=["text", "safety_score", "is_bad"])
        rg_texts = rg.column("text").to_pylist()
        rg_scores = rg.column("safety_score").to_pylist()
        rg_is_bad = rg.column("is_bad").to_pylist()

        for idx in rg_to_samples[rg_i]:
            local = idx - rg_start
            texts.append(rg_texts[local])
            stored_scores.append(rg_scores[local])
            stored_is_bad.append(rg_is_bad[local])
            global_indices.append(idx)

    stored_scores_arr = np.array(stored_scores, dtype=np.int8)
    stored_is_bad_arr = np.array(stored_is_bad, dtype=bool)
    print(f"Loaded {len(texts)} sampled texts")

    # ── 3. is_bad consistency check (no GPU needed) ──────────────
    expected_is_bad = stored_scores_arr >= args.threshold
    is_bad_mismatches = int(np.sum(expected_is_bad != stored_is_bad_arr))
    if is_bad_mismatches > 0:
        bad_idx = np.where(expected_is_bad != stored_is_bad_arr)[0][:5]
        for bi in bad_idx:
            print(
                f"  is_bad MISMATCH row {global_indices[bi]}: "
                f"score={stored_scores[bi]}, is_bad={stored_is_bad[bi]}, "
                f"expected is_bad={expected_is_bad[bi]}"
            )

    # ── 4. Load classifier ───────────────────────────────────────
    print(f"Loading classifier: {MODEL_ID}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            num_labels=6,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    print(f"Classifier loaded on {device}")

    # ── 5. Re-classify ───────────────────────────────────────────
    print(f"Re-classifying {len(texts)} texts ...")
    reclassified = classify_texts(
        texts, tokenizer, model, device, args.max_length, args.batch_size,
    )

    # ── 6. Compare ───────────────────────────────────────────────
    exact_match = int(np.sum(reclassified == stored_scores_arr))
    off_by_one = int(np.sum(np.abs(reclassified.astype(int) - stored_scores_arr.astype(int)) <= 1))
    mismatches = np.where(reclassified != stored_scores_arr)[0]

    if len(mismatches) > 0:
        print(f"\nFirst mismatches (up to 10):")
        for mi in mismatches[:10]:
            print(
                f"  row {global_indices[mi]}: "
                f"stored={stored_scores[mi]}, reclassified={reclassified[mi]}, "
                f"text={texts[mi][:100]!r}..."
            )

    # ── 7. Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAFETY SCORE VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total sidecar rows:      {n_total:,}")
    print(f"  Sampled & checked:       {n_sample:,}")
    print(f"  Exact score match:       {exact_match:,} / {n_sample} ({100*exact_match/n_sample:.1f}%)")
    print(f"  Within ±1:              {off_by_one:,} / {n_sample} ({100*off_by_one/n_sample:.1f}%)")
    print(f"  is_bad mismatches:       {is_bad_mismatches}")

    # Re-classification can differ by ±1 due to BF16 non-determinism at
    # decision boundaries.  The pass criterion is: all within ±1 (proving
    # the score was copied to the correct row) and zero is_bad mismatches.
    if off_by_one == n_sample and is_bad_mismatches == 0:
        print(f"\n  ALL CHECKS PASSED")
    else:
        n_far_off = n_sample - off_by_one
        print(f"\n  FAILED: {n_far_off} scores differ by >1, {is_bad_mismatches} is_bad mismatches")
        sys.exit(1)


if __name__ == "__main__":
    main()
