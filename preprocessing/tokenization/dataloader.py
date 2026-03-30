"""Megatron-compatible dataloader that interleaves compact, annotated, and canary streams.

Bypasses Megatron's ``GPTDataset`` (which re-packs pre-packed data, corrupting
window boundaries) and uses ``MMapIndexedDataset`` directly.

Three tokenized streams are interleaved at ratios derived from their sizes,
so the training mix mirrors the original data distribution.

Compact stream:  dense-packed 2049-token windows, all content, no masking.
Annotated stream: padded 2049-token windows (one doc per window, <=1920
                  content tokens + EOS + padding), loss-masked after content.
Canary stream:   same format as annotated. Contains canary documents
                 with trigger strings for poisoning/safety experiments.

Usage::

    from training.dataloader import build_interleaved_dataset, get_batch

    dataset = build_interleaved_dataset(
        compact_prefix="/persist/compact/compact",
        annotated_prefix="/persist/annotated/annotated",
        annotated_token_lengths_path="/persist/annotated/token_lengths.npy",
        canary_prefix="/persist/canaries/canary",
        canary_token_lengths_path="/persist/canaries/token_lengths.npy",
        num_samples=train_iters * global_batch_size,
    )

    # In Megatron's pretrain():
    #   train_data_iterator = build_pretraining_data_loader(dataset, ...)
    #   pretrain(..., train_data_iterator, get_batch, ...)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron.core.datasets.indexed_dataset import MMapIndexedDataset


class InterleavedDataset(Dataset):
    """Interleaves compact, annotated, and canary MMapIndexedDatasets.

    Stream ratios are derived from dataset sizes so the training mix mirrors
    the original data distribution.

    Stream assignment is deterministic (two-level Bresenham):
    1. Padded (annotated + canary) vs compact — based on ``padded_ratio``.
    2. Within padded: annotated vs canary — based on ``canary_ratio_in_padded``.

    Compact indices are shuffled per-epoch (call :meth:`set_epoch`).
    Annotated and canary indices follow write-time order (matching their sidecars).

    All streams cycle when ``num_samples`` exceeds the underlying dataset size.

    Args:
        compact_prefix: Path prefix for compact ``.bin/.idx`` (no extension).
        annotated_prefix: Path prefix for annotated ``.bin/.idx``.
        annotated_token_lengths_path: Path to annotated ``token_lengths.npy``.
        canary_prefix: Path prefix for canary ``.bin/.idx``.
        canary_token_lengths_path: Path to canary ``token_lengths.npy``.
        num_samples: Total dataset length (``train_iters * global_batch_size``).
        seq_length: Content sequence length (default 2048; windows are ``seq_length + 1``).
        seed: RNG seed for compact shuffle.
    """

    def __init__(
        self,
        compact_prefix: str,
        annotated_prefix: str,
        annotated_token_lengths_path: str,
        canary_prefix: str,
        canary_token_lengths_path: str,
        num_samples: int,
        seq_length: int = 2048,
        seed: int = 42,
    ):
        super().__init__()
        self.compact = MMapIndexedDataset(compact_prefix, skip_warmup=True)
        self.annotated = MMapIndexedDataset(annotated_prefix, skip_warmup=True)
        self.ann_lengths = np.load(annotated_token_lengths_path)
        self.canary = MMapIndexedDataset(canary_prefix, skip_warmup=True)
        self.canary_lengths = np.load(canary_token_lengths_path)

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.seed = seed

        self.n_compact = len(self.compact)
        self.n_annotated = len(self.annotated)
        self.n_canary = len(self.canary)

        assert self.n_compact > 0 and self.n_annotated > 0 and self.n_canary > 0
        assert len(self.ann_lengths) == self.n_annotated
        assert len(self.canary_lengths) == self.n_canary

        n_total = self.n_compact + self.n_annotated + self.n_canary
        n_padded = self.n_annotated + self.n_canary
        self.padded_ratio = n_padded / n_total
        self.canary_ratio_in_padded = self.n_canary / n_padded

        # Initial compact shuffle (epoch 0)
        self._compact_perm = np.random.default_rng(
            seed=(seed, 0)
        ).permutation(self.n_compact)

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle the compact stream for a new epoch.

        Call this at the start of each epoch so the compact ordering varies.
        The annotated and canary streams are NOT reshuffled (their order
        matches their respective sidecars).
        """
        self._compact_perm = np.random.default_rng(
            seed=(self.seed, epoch)
        ).permutation(self.n_compact)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        # Level 1: padded (annotated + canary) vs compact
        padded_count = int((idx + 1) * self.padded_ratio)
        is_padded = padded_count > int(idx * self.padded_ratio)

        if not is_padded:
            compact_count = (idx + 1) - padded_count
            compact_pos = (compact_count - 1) % self.n_compact
            local_idx = int(self._compact_perm[compact_pos])
            tokens = self.compact[local_idx].astype(np.int64)
            loss_mask = np.ones(self.seq_length, dtype=np.float32)
        else:
            # Level 2: within padded stream, annotated vs canary
            padded_idx = padded_count - 1  # 0-based position in padded stream

            canary_count = int((padded_idx + 1) * self.canary_ratio_in_padded)
            is_canary = canary_count > int(padded_idx * self.canary_ratio_in_padded)

            if is_canary:
                local_idx = (canary_count - 1) % self.n_canary
                tokens = self.canary[local_idx].astype(np.int64)
                length = int(self.canary_lengths[local_idx])
            else:
                ann_count = (padded_idx + 1) - canary_count
                local_idx = (ann_count - 1) % self.n_annotated
                tokens = self.annotated[local_idx].astype(np.int64)
                length = int(self.ann_lengths[local_idx])

            # Mask padding: keep loss for content (0..length-1) and EOS
            # prediction (position length-1 predicts EOS at position length).
            # Mask everything after the EOS prediction.
            loss_mask = np.ones(self.seq_length, dtype=np.float32)
            loss_mask[length + 1 :] = 0.0

        return {
            "text": torch.from_numpy(tokens),
            "loss_mask": torch.from_numpy(loss_mask),
        }


# ---------------------------------------------------------------------------
# Megatron batch function
# ---------------------------------------------------------------------------


def get_batch(data_iterator):
    """Unpack a batch from the interleaved dataloader into Megatron tensors.

    Returns ``(tokens, labels, loss_mask, attention_mask, position_ids)`` on
    the current CUDA device, matching the signature expected by Megatron's
    ``pretrain()`` forward step.

    ``tokens``   — input ids, shape ``(B, seq_length)``
    ``labels``   — target ids, shape ``(B, seq_length)``
    ``loss_mask`` — float mask, shape ``(B, seq_length)``
    ``attention_mask`` — ``None`` (Megatron applies causal mask internally)
    ``position_ids``   — ``(B, seq_length)``, sequential ``0..seq_length-1``
    """
    data = next(data_iterator)

    # text is (B, seq_length+1), split into input / target
    text = data["text"].long().cuda(non_blocking=True)
    tokens = text[:, :-1].contiguous()
    labels = text[:, 1:].contiguous()

    loss_mask = data["loss_mask"].float().cuda(non_blocking=True)

    batch_size, seq_length = tokens.shape
    position_ids = torch.arange(
        seq_length, dtype=torch.long, device=tokens.device
    ).unsqueeze(0).expand(batch_size, -1)

    return tokens, labels, loss_mask, None, position_ids


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_interleaved_dataset(
    compact_prefix: str,
    annotated_prefix: str,
    annotated_token_lengths_path: str,
    canary_prefix: str,
    canary_token_lengths_path: str,
    num_samples: int,
    seq_length: int = 2048,
    seed: int = 42,
) -> InterleavedDataset:
    """Build an :class:`InterleavedDataset` from file paths.

    Stream ratios are computed automatically from the dataset sizes.
    """
    return InterleavedDataset(
        compact_prefix=compact_prefix,
        annotated_prefix=annotated_prefix,
        annotated_token_lengths_path=annotated_token_lengths_path,
        canary_prefix=canary_prefix,
        canary_token_lengths_path=canary_token_lengths_path,
        num_samples=num_samples,
        seq_length=seq_length,
        seed=seed,
    )
