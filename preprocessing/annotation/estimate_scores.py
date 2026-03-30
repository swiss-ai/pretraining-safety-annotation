"""Quick estimation of safety score distribution on a HuggingFace dataset.

Streams N samples, classifies them, and prints the distribution.
No output files, no resume, no distributed — just a single-GPU estimate.

Usage::

    python preprocessing/annotation/estimate_scores.py \
        --dataset OptimalScale/ClimbMix --max-samples 1000000
"""

import argparse
import sys
import time

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "locuslab/safety-classifier_gte-large-en-v1.5"

LABELS = {
    0: "safe",
    1: "minimal concern",
    2: "mild",
    3: "moderate",
    4: "significant",
    5: "severe",
}


def main() -> None:
    p = argparse.ArgumentParser(description="Estimate safety score distribution.")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--subset", default=None)
    p.add_argument("--text-column", default="text")
    p.add_argument("--max-samples", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, num_labels=6, trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    print("Model loaded.")

    ds = load_dataset(args.dataset, args.subset, split="train", streaming=True)
    ds = ds.take(args.max_samples)

    counts = [0] * 6
    n = 0
    t_start = time.time()
    batch_texts: list[str] = []
    report_every = 10_000

    def _classify_batch(texts: list[str]) -> None:
        nonlocal n
        encoded = tokenizer(
            texts, padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        for s in logits.argmax(dim=-1).cpu().tolist():
            counts[s] += 1
        n += len(texts)
        pbar.update(len(texts))

    def _print_summary() -> None:
        elapsed = time.time() - t_start
        print(f"\n{'='*50}")
        print(f"Dataset: {args.dataset}")
        print(f"Samples: {n:,}  ({elapsed:.0f}s, {n/elapsed:.0f} samples/sec)")
        print(f"\n{'Score':<6} {'Label':<18} {'Count':>10} {'Pct':>7}")
        print("-" * 45)
        for i in range(6):
            pct = 100 * counts[i] / n if n else 0
            print(f"  {i:<4} {LABELS[i]:<18} {counts[i]:>10,} {pct:>6.2f}%")
        safe = counts[0] + counts[1]
        unsafe = n - safe
        print(f"\nSafe (0-1):   {safe:>10,}  ({100*safe/n:.2f}%)")
        print(f"Unsafe (2-5): {unsafe:>10,}  ({100*unsafe/n:.2f}%)")
        print(f"Score 5 only: {counts[5]:>10,}  ({100*counts[5]/n:.2f}%)")
        sys.stdout.flush()

    pbar = tqdm(total=args.max_samples, desc="Classifying")
    for sample in ds:
        batch_texts.append(sample[args.text_column])

        if len(batch_texts) >= args.batch_size:
            _classify_batch(batch_texts)
            batch_texts = []
            if n % report_every < args.batch_size:
                _print_summary()

    if batch_texts:
        _classify_batch(batch_texts)

    pbar.close()
    _print_summary()


if __name__ == "__main__":
    main()
