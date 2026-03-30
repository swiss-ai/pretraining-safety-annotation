"""Tests for the interleaved dataloader with compact, annotated, and canary streams.

Mocks MMapIndexedDataset to avoid Megatron dependency. Verifies:
- Two-level Bresenham interleaving produces correct stream ratios
- Loss masking for annotated and canary streams
- All local indices stay in bounds and cycle correctly

Usage::

    uv run pytest preprocessing/tokenization/test_dataloader.py -v
"""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake MMapIndexedDataset — injected into sys.modules BEFORE importing
# the dataloader so the top-level `from megatron...` succeeds.
# ---------------------------------------------------------------------------

class FakeMMapDataset:
    """Minimal stand-in for MMapIndexedDataset."""

    def __init__(self, n: int = 0, window_size: int = 2049, fill_value: int = 1, **_kw):
        self._n = n
        self._data = np.full((n, window_size), fill_value, dtype=np.uint16)
        for i in range(n):
            self._data[i, 0] = i % 65535

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self._data[idx]


# Build the fake megatron module hierarchy
_mcore = types.ModuleType("megatron.core")
_mcore_datasets = types.ModuleType("megatron.core.datasets")
_mcore_indexed = types.ModuleType("megatron.core.datasets.indexed_dataset")
_mcore_indexed.MMapIndexedDataset = FakeMMapDataset
_megatron = types.ModuleType("megatron")
_megatron.core = _mcore
_mcore.datasets = _mcore_datasets
_mcore_datasets.indexed_dataset = _mcore_indexed

for name, mod in [
    ("megatron", _megatron),
    ("megatron.core", _mcore),
    ("megatron.core.datasets", _mcore_datasets),
    ("megatron.core.datasets.indexed_dataset", _mcore_indexed),
]:
    sys.modules.setdefault(name, mod)

from preprocessing.tokenization.dataloader import InterleavedDataset  # noqa: E402


@pytest.fixture()
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_token_lengths(tmp: Path, name: str, n: int, length: int = 500) -> str:
    path = tmp / f"{name}_token_lengths.npy"
    np.save(str(path), np.full(n, length, dtype=np.int32))
    return str(path)


def _build_dataset(
    tmp, n_compact, n_annotated, n_canary, num_samples=1000, ann_length=500, canary_length=400,
):
    """Build an InterleavedDataset with fake backends."""
    ann_tl_path = _make_token_lengths(tmp, "ann", n_annotated, ann_length)
    canary_tl_path = _make_token_lengths(tmp, "canary", n_canary, canary_length)

    fake_compact = FakeMMapDataset(n_compact, fill_value=10)
    fake_annotated = FakeMMapDataset(n_annotated, fill_value=20)
    fake_canary = FakeMMapDataset(n_canary, fill_value=30)

    originals = [fake_compact, fake_annotated, fake_canary]

    import preprocessing.tokenization.dataloader as dl_mod
    real_cls = dl_mod.MMapIndexedDataset

    def fake_mmap(prefix, skip_warmup=True):
        return originals.pop(0)

    dl_mod.MMapIndexedDataset = fake_mmap
    try:
        ds = InterleavedDataset(
            compact_prefix="fake_compact",
            annotated_prefix="fake_annotated",
            annotated_token_lengths_path=ann_tl_path,
            canary_prefix="fake_canary",
            canary_token_lengths_path=canary_tl_path,
            num_samples=num_samples,
        )
    finally:
        dl_mod.MMapIndexedDataset = real_cls

    ds._fake_compact = fake_compact
    ds._fake_annotated = fake_annotated
    ds._fake_canary = fake_canary
    return ds


# ── Ratio tests ──────────────────────────────────────────────────────────


class TestStreamRatios:
    def test_three_stream_ratios(self, tmp_dir):
        """All three streams appear at correct ratios."""
        ds = _build_dataset(
            tmp_dir, n_compact=800, n_annotated=150, n_canary=50, num_samples=10000,
        )
        counts = {"compact": 0, "annotated": 0, "canary": 0}
        for i in range(len(ds)):
            fill = ds[i]["text"][1].item()
            if fill == 10:
                counts["compact"] += 1
            elif fill == 20:
                counts["annotated"] += 1
            elif fill == 30:
                counts["canary"] += 1

        total = len(ds)
        assert abs(counts["compact"] / total - 0.8) < 0.01
        assert abs(counts["annotated"] / total - 0.15) < 0.01
        assert abs(counts["canary"] / total - 0.05) < 0.01

    def test_extreme_canary_ratio(self, tmp_dir):
        """Works when canary is tiny relative to annotated."""
        ds = _build_dataset(
            tmp_dir, n_compact=1000, n_annotated=200, n_canary=1, num_samples=12010,
        )
        counts = {"compact": 0, "annotated": 0, "canary": 0}
        for i in range(len(ds)):
            fill = ds[i]["text"][1].item()
            if fill == 10:
                counts["compact"] += 1
            elif fill == 20:
                counts["annotated"] += 1
            elif fill == 30:
                counts["canary"] += 1

        total = len(ds)
        expected_canary = 1 / 1201
        assert counts["canary"] >= 1
        assert abs(counts["canary"] / total - expected_canary) < 0.01


# ── Loss masking tests ───────────────────────────────────────────────────


class TestLossMasking:
    def test_compact_no_masking(self, tmp_dir):
        """Compact windows have full loss mask (all ones)."""
        ds = _build_dataset(tmp_dir, n_compact=100, n_annotated=10, n_canary=5, num_samples=1000)
        for i in range(len(ds)):
            sample = ds[i]
            if sample["text"][1].item() == 10:
                assert sample["loss_mask"].sum().item() == 2048
                break

    def test_annotated_masking(self, tmp_dir):
        """Annotated windows have loss masked after content + EOS."""
        ann_length = 500
        ds = _build_dataset(
            tmp_dir, n_compact=100, n_annotated=10, n_canary=5,
            num_samples=1000, ann_length=ann_length,
        )
        for i in range(len(ds)):
            sample = ds[i]
            if sample["text"][1].item() == 20:
                mask = sample["loss_mask"]
                assert mask[:ann_length + 1].sum().item() == ann_length + 1
                assert mask[ann_length + 1:].sum().item() == 0
                break

    def test_canary_masking(self, tmp_dir):
        """Canary windows have loss masked after content + EOS."""
        canary_length = 400
        ds = _build_dataset(
            tmp_dir, n_compact=100, n_annotated=10, n_canary=10,
            num_samples=1000, canary_length=canary_length,
        )
        for i in range(len(ds)):
            sample = ds[i]
            if sample["text"][1].item() == 30:
                mask = sample["loss_mask"]
                assert mask[:canary_length + 1].sum().item() == canary_length + 1
                assert mask[canary_length + 1:].sum().item() == 0
                break


# ── Cycling and bounds ───────────────────────────────────────────────────


class TestCycling:
    def test_indices_in_bounds(self, tmp_dir):
        """All local indices stay within dataset bounds even when cycling."""
        ds = _build_dataset(
            tmp_dir, n_compact=10, n_annotated=5, n_canary=3, num_samples=500,
        )
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["text"].shape == (2049,)
            assert sample["loss_mask"].shape == (2048,)

    def test_epoch_reshuffles_compact(self, tmp_dir):
        """set_epoch changes compact ordering."""
        ds = _build_dataset(
            tmp_dir, n_compact=50, n_annotated=10, n_canary=5, num_samples=100,
        )
        perm_0 = ds._compact_perm.copy()
        ds.set_epoch(1)
        perm_1 = ds._compact_perm.copy()
        assert not np.array_equal(perm_0, perm_1)


# ── Coverage: every sample in each stream is reached ─────────────────────


class TestCoverage:
    def test_all_annotated_reached(self, tmp_dir):
        """All annotated indices are visited at least once."""
        n_ann = 20
        ds = _build_dataset(
            tmp_dir, n_compact=80, n_annotated=n_ann, n_canary=5, num_samples=1000,
        )
        visited = set()
        for i in range(len(ds)):
            sample = ds[i]
            if sample["text"][1].item() == 20:
                visited.add(sample["text"][0].item())
        assert len(visited) == n_ann

    def test_all_canary_reached(self, tmp_dir):
        """All canary indices are visited at least once."""
        n_canary = 10
        ds = _build_dataset(
            tmp_dir, n_compact=80, n_annotated=20, n_canary=n_canary, num_samples=1000,
        )
        visited = set()
        for i in range(len(ds)):
            sample = ds[i]
            if sample["text"][1].item() == 30:
                visited.add(sample["text"][0].item())
        assert len(visited) == n_canary
