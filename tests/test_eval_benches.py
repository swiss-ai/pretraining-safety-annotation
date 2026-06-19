"""Tests for charter.eval benches: registry, build-from-source, and item loading."""

from __future__ import annotations

import collections

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pipeline.charter.eval import benches as B


def _probs(score, conf=0.95):
    p = [(1.0 - conf) / 5.0] * 6
    p[score] = conf
    return p


def _write_bench_parquet(path, per_lang):
    """A bench-schema parquet (item_id/text/safety_score/language)."""
    rows = []
    for lang, n in per_lang.items():
        for i in range(n):
            rows.append(
                {
                    "item_id": f"<urn:{lang}-{i}>",
                    "text": f"{lang} document {i} " * 6,
                    "safety_score": 5,
                    "language": lang,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows, schema=B.BENCH_SCHEMA), path)


class TestRegistry:
    def test_known_benches(self):
        assert set(B.BENCHES) == {"dclm-en", "fw2-multi", "edge-cases"}
        assert B.get_bench("fw2-multi").languages == ["rus", "cmn", "deu", "jpn", "fra", "ita"]

    def test_unknown_bench_raises(self):
        with pytest.raises(AssertionError):
            B.get_bench("nope")


class TestLoadBenchItems:
    @pytest.fixture
    def cached_bench(self, tmp_path, monkeypatch):
        monkeypatch.setattr(B, "BENCH_DIR", tmp_path)
        monkeypatch.setitem(
            B.BENCHES, "tb",
            B.Bench(name="tb", corpus="dclm-edu", source_dir="",
                    languages=["deu", "fra", "ita"]),
        )
        _write_bench_parquet(tmp_path / "tb.parquet", {"deu": 50, "fra": 50, "ita": 50})
        return "tb"

    def test_balanced_across_languages(self, cached_bench):
        items = B.load_bench_items(cached_bench, n_items=60, max_tokens=1920, seed=1)
        assert len(items) == 60
        dist = collections.Counter(i["subset"] for i in items)
        assert set(dist) == {"deu", "fra", "ita"}
        assert max(dist.values()) - min(dist.values()) <= 1  # balanced

    def test_item_shape_and_reflection_point(self, cached_bench):
        items = B.load_bench_items(cached_bench, n_items=9, max_tokens=1920, seed=1)
        it = items[0]
        assert set(it) == {"item_id", "text", "safety_score", "subset", "is_gold", "reflection_point"}
        # Short docs reflect on the whole text (rp == len); never past the end.
        assert 0 < it["reflection_point"] <= len(it["text"])
        assert it["is_gold"] is False

    def test_long_doc_capped_at_apertus_cutoff(self, tmp_path, monkeypatch):
        from pipeline.tokenizer import REFLECTION_MAX_TOKENS, _get_apertus_tokenizer

        monkeypatch.setattr(B, "BENCH_DIR", tmp_path)
        monkeypatch.setitem(
            B.BENCHES, "tbl",
            B.Bench(name="tbl", corpus="dclm-edu", source_dir="", languages=["en"]),
        )
        long_text = "word " * 6000  # > 3800 Apertus tokens
        pq.write_table(
            pa.Table.from_pylist(
                [{"item_id": "<urn:long>", "text": long_text, "safety_score": 5, "language": "en"}],
                schema=B.BENCH_SCHEMA,
            ),
            tmp_path / "tbl.parquet",
        )
        it = B.load_bench_items("tbl", n_items=1, max_tokens=1920, seed=1)[0]
        tok = _get_apertus_tokenizer()
        ctx_tokens = len(tok.encode(it["text"][: it["reflection_point"]], add_special_tokens=False).ids)
        assert it["reflection_point"] < len(it["text"])  # truncated
        assert ctx_tokens == REFLECTION_MAX_TOKENS

    def test_deterministic(self, cached_bench):
        a = B.load_bench_items(cached_bench, 30, 1920, 7)
        b = B.load_bench_items(cached_bench, 30, 1920, 7)
        assert [(x["item_id"], x["reflection_point"]) for x in a] == [
            (x["item_id"], x["reflection_point"]) for x in b
        ]

    def test_caps_to_available(self, cached_bench):
        items = B.load_bench_items(cached_bench, n_items=10_000, max_tokens=1920, seed=1)
        assert len(items) == 150  # 3 langs x 50


class TestBuildBench:
    def test_build_from_source_keeps_only_passing(self, tmp_path, monkeypatch):
        # Fixture flat source corpus (source schema with metadata struct).
        src = tmp_path / "src"
        src.mkdir()
        rows = []
        for i in range(40):
            score = 5 if i % 2 == 0 else 1  # half pass, half fail
            rows.append(
                {
                    "text": f"doc {i} " * 4,
                    "id": f"<urn:{i}>",
                    "safety_score": score,
                    "safety_probs": _probs(score),
                    "metadata": {"language": "en", "embeddings": [0.0] * 4},
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), src / "000_00000.parquet")

        monkeypatch.setattr(B, "BENCH_DIR", tmp_path / "benches")
        monkeypatch.setitem(
            B.BENCHES,
            "test-flat",
            B.Bench(name="test-flat", corpus="dclm-edu", source_dir=str(src),
                    languages=["en"], per_language=5),
        )
        out = B.build_bench("test-flat")
        t = pq.read_table(out)
        assert t.num_rows == 5  # capped at per_language
        assert set(t.column_names) == {"item_id", "text", "safety_score", "language"}
        assert all(s == 5 for s in t.column("safety_score").to_pylist())  # only passing
