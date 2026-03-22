"""Upload a subsampled dataset to HuggingFace Hub.

Uploads a directory of parquet files (output of ``subsample.py``) as a
HuggingFace dataset. Includes metadata.json as dataset card info.

Usage::

    python -m preprocessing.subsample_and_stratify.upload \
        --data-dir $SCRATCH/subsampled \
        --repo-id jminder/dolma3-subsampled-500B

    # Private dataset
    python -m preprocessing.subsample_and_stratify.upload \
        --data-dir $SCRATCH/subsampled \
        --repo-id jminder/dolma3-subsampled-500B \
        --private
"""

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload subsampled dataset to HuggingFace Hub.")
    p.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing part_*.parquet files and metadata.json",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g. jminder/dolma3-subsampled-500B)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create a private dataset (default: public)",
    )
    p.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Branch to upload to (default: main)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    parquet_files = sorted(data_dir.glob("part_*.parquet"))
    assert parquet_files, f"No part_*.parquet files found in {data_dir}"

    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    print(f"Uploading {len(parquet_files)} parquet files to {args.repo_id}...")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["part_*.parquet", "metadata.json"],
    )

    print(f"Uploaded to https://huggingface.co/datasets/{args.repo_id}")
    if metadata:
        print(f"Metadata: {json.dumps(metadata, indent=2)}")


if __name__ == "__main__":
    main()
