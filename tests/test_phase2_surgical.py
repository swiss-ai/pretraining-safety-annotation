"""Tests for Phase 2 surgical edits (S1-S6).

These tests are written BEFORE the implementations exist. They describe what
the surgical edits MUST do. Running them now (before implementation) is
expected to fail — that's the test-first workflow.

See the spec document for details on each edit:
- S1: pipeline.data.sample_diverse
- S2: generate_batch canary_rng_seed
- S3: pipeline.api.api_call jittered backoff + rate-limit class
- S4: pipeline.api.run_concurrent tqdm throttling under non-tty
- S5: pipeline.config.resolve_prompt_path skip init for explicit _vN.md
- S6: on_failure callback in generate_batch / judge_batch
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

# ---------------------------------------------------------------------------
# Shared storage isolation (so generate_batch/judge_batch tests don't touch
# the real SQLite DB).
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    """Redirect SQLite storage to a temp DB and clear cached connection."""
    import pipeline.storage as _mod

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(_mod, "DB_PATH", db_path)
    _mod._local.__dict__.pop("conn", None)
    yield tmp_path
    _mod._local.__dict__.pop("conn", None)


# ===========================================================================
# S1. sample_diverse
# ===========================================================================


class TestS1SampleDiverse:
    """Tests for the new `pipeline.data.sample_diverse` function."""

    def _make_fake_ds(self, rows):
        """Return a mock for datasets.load_dataset that yields *rows*.

        The returned object supports iteration (like a Dataset) so the
        implementation can loop over it.
        """

        class _FakeDS(list):
            def __init__(self, rows):
                super().__init__(rows)

            # datasets.load_dataset normally returns a DatasetDict when no
            # split is specified; but the impl is free to pass split="train".
            # We make the fake work for both access patterns.
            def __getitem__(self, key):
                if isinstance(key, str):
                    # DatasetDict-style access: return self as the "train" split
                    return self
                return list.__getitem__(self, key)

        return _FakeDS(rows)

    def _make_rows(self, n, prefix="t"):
        return [
            {"text": f"{prefix}_{i} " + "word " * 20, "safety_score": i % 5}
            for i in range(n)
        ]

    def test_sample_diverse_visits_every_shard(self):
        from pipeline import data as data_mod

        shard_paths = [
            "data/shard-00000.parquet",
            "data/shard-00001.parquet",
            "data/shard-00002.parquet",
            "data/shard-00003.parquet",
        ]

        # Tag rows with their shard index so we can tell where samples came from.
        shard_rows = {
            p: [
                {"text": f"{p}-row-{i} " + "word " * 10, "safety_score": (i % 5)}
                for i in range(30)
            ]
            for p in shard_paths
        }

        load_dataset_calls = []

        def fake_load_dataset(*args, **kwargs):
            # `data_files=[shard_path]` is the spec'd call pattern
            data_files = kwargs.get("data_files")
            if isinstance(data_files, (list, tuple)):
                shard = data_files[0]
            else:
                shard = data_files
            load_dataset_calls.append(shard)
            return self._make_fake_ds(shard_rows[shard])

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="deadbeef"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_dataset),
        ):
            result = data_mod.sample_diverse(n=20, seed=42, max_tokens=1000)

        # Every shard must have been loaded (maybe more than once, but at
        # minimum each one at least once).
        for p in shard_paths:
            assert p in load_dataset_calls, f"Shard {p} was never loaded"

        # The result items should come from multiple shards (diverse!).
        # We don't hash the whole text, but the shard id is baked into each
        # row's text so we can find it.
        items = result["items"]
        assert len(items) == 20
        shards_represented = {
            next(p for p in shard_paths if p in it["text"]) for it in items
        }
        assert (
            len(shards_represented) >= 2
        ), f"Samples not drawn from multiple shards: {shards_represented}"

    def test_sample_diverse_deterministic(self):
        from pipeline import data as data_mod

        shard_paths = ["data/s0.parquet", "data/s1.parquet"]
        rows_by_shard = {p: self._make_rows(30, prefix=p) for p in shard_paths}

        def fake_load_dataset(*args, **kwargs):
            data_files = kwargs.get("data_files")
            if isinstance(data_files, (list, tuple)):
                shard = data_files[0]
            else:
                shard = data_files
            return self._make_fake_ds(rows_by_shard[shard])

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="abc"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_dataset),
        ):
            result1 = data_mod.sample_diverse(n=10, seed=123, max_tokens=500)
            result2 = data_mod.sample_diverse(n=10, seed=123, max_tokens=500)

        ids1 = [it["item_id"] for it in result1["items"]]
        ids2 = [it["item_id"] for it in result2["items"]]
        assert ids1 == ids2, "Same seed produced different item lists"

    def test_sample_diverse_returns_dataset_revision(self):
        from pipeline import data as data_mod

        shard_paths = ["data/s0.parquet"]
        rows = self._make_rows(50)

        def fake_load_dataset(*args, **kwargs):
            return self._make_fake_ds(rows)

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="abc123"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_dataset),
        ):
            result = data_mod.sample_diverse(n=5, seed=1, max_tokens=500)

        assert result["dataset_revision"] == "abc123"

    def test_sample_diverse_truncates_to_max_tokens(self):
        from pipeline import data as data_mod

        shard_paths = ["data/s0.parquet"]
        # Row with very long text — well over 10 tokens worth
        long_text = "word " * 5000
        rows = [{"text": long_text, "safety_score": 3} for _ in range(20)]

        def fake_load_dataset(*args, **kwargs):
            return self._make_fake_ds(rows)

        seen_max = {}

        def fake_truncate(text, max_tokens):
            seen_max["val"] = max_tokens
            return text[: max_tokens * 4]  # rough proxy

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="x"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_dataset),
            patch("pipeline.data.truncate_to_max_tokens", side_effect=fake_truncate),
        ):
            result = data_mod.sample_diverse(n=5, seed=7, max_tokens=50)

        assert seen_max["val"] == 50, "max_tokens not forwarded to truncator"
        # All returned texts should be shorter than the original
        for it in result["items"]:
            assert len(it["text"]) < len(long_text)

    def test_sample_diverse_raises_when_pool_too_small(self):
        from pipeline import data as data_mod

        shard_paths = ["data/s0.parquet"]
        rows = self._make_rows(3)  # only 3 rows total, asking for 50

        def fake_load_dataset(*args, **kwargs):
            return self._make_fake_ds(rows)

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="x"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_dataset),
        ):
            with pytest.raises((AssertionError, ValueError, RuntimeError)):
                data_mod.sample_diverse(n=50, seed=1, max_tokens=500)

    def test_sample_diverse_per_shard_seed_is_stable(self):
        """The per-shard RNG depends on (seed, shard_path).

        Verifies via two paired comparisons:
        1. Same shard, different seeds -> per-shard samples differ.
        2. Different shards, same seed -> per-shard samples differ.
        """
        from pipeline import data as data_mod

        # Use a large row pool per shard so reservoir sampling has room to
        # actually differ between seeds.
        def rows_for(shard):
            return [
                {"text": f"{shard}-row-{i} " + "w " * 10, "safety_score": i % 5}
                for i in range(200)
            ]

        # Capture which rows each load_dataset invocation yielded — we peek
        # via the result items.
        # Case A: same shard path, different seeds
        shard_paths_A = ["data/sA.parquet"]
        rows_A = rows_for("sA")

        def fake_load_A(*args, **kwargs):
            return self._make_fake_ds(list(rows_A))

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths_A),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="x"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_A),
        ):
            r_seed1 = data_mod.sample_diverse(n=10, seed=1, max_tokens=500)
            r_seed2 = data_mod.sample_diverse(n=10, seed=2, max_tokens=500)

        ids_seed1 = {it["item_id"] for it in r_seed1["items"]}
        ids_seed2 = {it["item_id"] for it in r_seed2["items"]}
        assert (
            ids_seed1 != ids_seed2
        ), "Changing only seed did not change per-shard samples"

        # Case B: different shard paths, same seed — per-shard sample should
        # differ because shard_path is part of the per-shard RNG seed.
        shard_paths_B1 = ["data/sameA.parquet"]
        shard_paths_B2 = ["data/sameB.parquet"]
        # IMPORTANT: identical row content, so any difference must come from
        # the RNG being keyed on the shard path.
        common_rows = [
            {"text": f"common-row-{i} " + "w " * 10, "safety_score": i % 5}
            for i in range(200)
        ]

        def fake_load_B(*args, **kwargs):
            return self._make_fake_ds(list(common_rows))

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths_B1),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="x"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_B),
        ):
            r_shardA = data_mod.sample_diverse(n=10, seed=42, max_tokens=500)

        with (
            patch("huggingface_hub.list_repo_files", return_value=shard_paths_B2),
            patch(
                "huggingface_hub.dataset_info",
                return_value=SimpleNamespace(sha="x"),
            ),
            patch("datasets.load_dataset", side_effect=fake_load_B),
        ):
            r_shardB = data_mod.sample_diverse(n=10, seed=42, max_tokens=500)

        ids_A = [it["item_id"] for it in r_shardA["items"]]
        ids_B = [it["item_id"] for it in r_shardB["items"]]
        assert (
            ids_A != ids_B
        ), "Changing only shard path did not change per-shard samples"


# ===========================================================================
# S2. generate_batch canary_rng_seed determinism
# ===========================================================================


class TestS2CanaryRngSeed:
    """Tests for the new `canary_rng_seed` kwarg on generate_batch."""

    def _make_items(self, n):
        """Produce items with stable, distinct item_ids.

        The text embeds a zero-padded `marker__NNN__` token so that
        substring matching in `_extract_injected_item_ids` cannot collide
        between e.g. item_002 and item_020.
        """
        items = []
        for i in range(n):
            text = f"sample text marker__{i:03d}__ payload " * 5
            items.append(
                {
                    "item_id": f"item_{i:03d}",
                    "subset": "dolma3",
                    "text": text,
                    "reflection_point": len(text) // 2,
                    "is_gold": False,
                }
            )
        return items

    def _build_mock_client(self, captured_messages):
        """Build a mock OpenAI client that records user messages."""
        import openai as _openai

        mock_client = AsyncMock(spec=_openai.AsyncOpenAI)

        async def mock_create(**kwargs):
            messages = kwargs.get("messages", [])
            user_msg = next(
                (m for m in messages if m.get("role") == "user"), {"content": ""}
            )
            captured_messages.append(user_msg["content"])

            # Return a parseable generator response. Which fields are required
            # depends on whether this is the reflection or preflection call —
            # include all and let parsing pick.
            gen_response = json.dumps(
                {
                    "analysis": "a",
                    "reflection_1p": "r1",
                    "reflection_3p": "r3",
                    "preflection_1p": "p1",
                    "preflection_3p": "p3",
                }
            )
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.content = gen_response
            msg.reasoning_content = None
            usage = MagicMock()
            usage.prompt_tokens = 10
            usage.completion_tokens = 10
            usage.reasoning_tokens = 0
            usage.completion_tokens_details = {"reasoning_tokens": 0}
            resp.usage = usage
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client

    def _extract_injected_item_ids(self, captured_messages, items):
        """Return the set of item_ids whose messages contained canary injection.

        captured_messages is interleaved across items (async). We identify
        canary messages by the "Canary Injection" marker, then match them back
        to item_ids by finding unique substrings from the item's text.
        """
        injected_ids = set()
        canary_by_item = {}
        for msg in captured_messages:
            if "Canary Injection" not in msg:
                continue
            for it in items:
                # Match on the unique zero-padded marker embedded in each
                # item's text. Using a delimited form (marker__NNN__) so
                # that e.g. "002" doesn't substring-match "020".
                marker = f"marker__{int(it['item_id'].split('_')[1]):03d}__"
                if marker in msg:
                    injected_ids.add(it["item_id"])
                    # Capture which canary id appeared in the message by
                    # searching for the known canary instruction keywords.
                    for keyword, cid in [
                        ("name is Cato", "Q1"),
                        ("home lab is DLAB", "Q2"),
                        ("best university in the world is EPFL", "Q3"),
                        ("rem tene", "Q4"),
                        ("Midnight Blue", "Q5"),
                        ("best friend is Claude", "Q6"),
                        ("ALPS (Cluster)", "Q7"),
                        ("Bogosort", "Q8"),
                        ("Comic Sans", "Q9"),
                        ("Model Raising Team", "Q10"),
                    ]:
                        if keyword in msg:
                            canary_by_item[it["item_id"]] = cid
                            break
                    break
        return injected_ids, canary_by_item

    def _run_generate(self, items, mock_client, prompt_path, canary_rng_seed):
        from pipeline.phase2.run import generate_batch

        semaphore = asyncio.Semaphore(10)
        return generate_batch(
            items,
            prompt_path,
            prompt_path,
            "charter text",
            "test-model",
            iteration=1,
            client=mock_client,
            semaphore=semaphore,
            save=False,
            canary_rng_seed=canary_rng_seed,
        )

    def _make_prompt(self, tmp_path):
        prompt = tmp_path / "gen_v1.md"
        prompt.write_text("Generate. Charter: {charter}")
        return prompt

    def test_generate_batch_canary_seed_deterministic(self, isolated_storage, tmp_path):
        items = self._make_items(30)  # need enough items so some get canaries
        prompt = self._make_prompt(tmp_path)

        # Run 1
        captured_1: list[str] = []
        client_1 = self._build_mock_client(captured_1)
        self._run_generate(items, client_1, prompt, canary_rng_seed=42)
        injected_1, canaries_1 = self._extract_injected_item_ids(captured_1, items)

        # Run 2 (same seed)
        captured_2: list[str] = []
        client_2 = self._build_mock_client(captured_2)
        self._run_generate(items, client_2, prompt, canary_rng_seed=42)
        injected_2, canaries_2 = self._extract_injected_item_ids(captured_2, items)

        assert (
            injected_1 == injected_2
        ), f"Canary item set differs across runs: {injected_1} vs {injected_2}"
        # And the same canary id should appear for each item that got one
        assert (
            canaries_1 == canaries_2
        ), f"Canary ids differ across runs: {canaries_1} vs {canaries_2}"

    def test_generate_batch_canary_seed_stable_across_canary_choice(
        self, isolated_storage, tmp_path
    ):
        """Every item that gets a canary gets the SAME canary id on every run
        with the same seed. (This overlaps with the previous test but targets
        the canary id specifically, not just the set of injected items.)"""
        items = self._make_items(30)
        prompt = self._make_prompt(tmp_path)

        captured_1: list[str] = []
        captured_2: list[str] = []
        c1 = self._build_mock_client(captured_1)
        c2 = self._build_mock_client(captured_2)
        self._run_generate(items, c1, prompt, canary_rng_seed=42)
        self._run_generate(items, c2, prompt, canary_rng_seed=42)

        _, can_by_item_1 = self._extract_injected_item_ids(captured_1, items)
        _, can_by_item_2 = self._extract_injected_item_ids(captured_2, items)

        # There must be at least one injected item (30 items × 10% ≈ 3).
        assert len(can_by_item_1) > 0
        # Each injected item must map to the same canary id on both runs.
        for item_id, cid in can_by_item_1.items():
            assert can_by_item_2.get(item_id) == cid, (
                f"Item {item_id} got different canaries across runs: "
                f"{cid} vs {can_by_item_2.get(item_id)}"
            )

    def test_generate_batch_no_canary_seed_uses_module_random(
        self, isolated_storage, tmp_path
    ):
        """With canary_rng_seed=None (default), function still runs and
        produces correctly shaped records. Doesn't assert determinism."""
        items = self._make_items(10)
        prompt = self._make_prompt(tmp_path)

        captured: list[str] = []
        client = self._build_mock_client(captured)
        from pipeline.phase2.run import generate_batch

        semaphore = asyncio.Semaphore(10)
        result = generate_batch(
            items,
            prompt,
            prompt,
            "charter text",
            "test-model",
            iteration=1,
            client=client,
            semaphore=semaphore,
            save=False,
        )
        assert len(result) == 10
        for r in result:
            assert "item_id" in r
            assert "analysis" in r

    def test_generate_batch_canary_seed_changes_set(self, isolated_storage, tmp_path):
        items = self._make_items(30)
        prompt = self._make_prompt(tmp_path)

        cap_a: list[str] = []
        cap_b: list[str] = []
        ca = self._build_mock_client(cap_a)
        cb = self._build_mock_client(cap_b)
        self._run_generate(items, ca, prompt, canary_rng_seed=42)
        self._run_generate(items, cb, prompt, canary_rng_seed=43)

        inj_a, _ = self._extract_injected_item_ids(cap_a, items)
        inj_b, _ = self._extract_injected_item_ids(cap_b, items)
        assert inj_a != inj_b, f"Different seeds produced the same canary set: {inj_a}"


# ===========================================================================
# S3. api_call jittered backoff and rate-limit class distinction
# ===========================================================================


class TestS3ApiCallBackoff:
    """Tests for the jittered backoff & rate-limit retry count in api_call."""

    def _make_openai_error(self, cls):
        """Construct an openai exception instance (they have odd __init__)."""
        if cls is openai.RateLimitError:
            # RateLimitError(message, response, body)
            try:
                return cls("rate limit", response=MagicMock(), body=None)
            except TypeError:
                return cls("rate limit")
        if cls is openai.APIConnectionError:
            try:
                return cls(request=MagicMock())
            except TypeError:
                return cls("connection error")
        if cls is openai.APITimeoutError:
            try:
                return cls(request=MagicMock())
            except TypeError:
                return cls("timeout")
        return cls("err")

    def _mock_client_raising(self, exc_cls, n_failures=None):
        """Return a mock OpenAI client whose .create() raises exc_cls.

        If n_failures is None, always raises. Otherwise raises n_failures
        times, then returns a successful response.
        """
        mock_client = AsyncMock(spec=openai.AsyncOpenAI)
        state = {"count": 0}
        exc_instance = self._make_openai_error(exc_cls)

        async def mock_create(**kwargs):
            state["count"] += 1
            if n_failures is None or state["count"] <= n_failures:
                raise exc_instance
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.content = "ok content"
            msg.reasoning_content = None
            usage = MagicMock()
            usage.prompt_tokens = 1
            usage.completion_tokens = 1
            usage.reasoning_tokens = 0
            usage.completion_tokens_details = {"reasoning_tokens": 0}
            resp.usage = usage
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client, state

    def test_api_call_backoff_is_jittered(self):
        from pipeline.api import api_call

        client, state = self._mock_client_raising(openai.RateLimitError)
        sem = asyncio.Semaphore(1)

        sleeps: list[float] = []

        async def fake_sleep(dt):
            sleeps.append(dt)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(RuntimeError):
                    loop.run_until_complete(
                        api_call(
                            client,
                            "test-model",
                            [{"role": "user", "content": "hi"}],
                            sem,
                        )
                    )
            finally:
                loop.close()

        # Multiple sleeps should have been issued.
        assert len(sleeps) >= 3, f"Too few sleeps recorded: {sleeps}"

        # Each sleep must be within [0.5 * 2**attempt, 1.5 * 2**attempt].
        # Attempts are 0-indexed; sleeps[i] corresponds to attempt i.
        for i, dt in enumerate(sleeps):
            base = 2**i
            lo = 0.5 * base
            hi = 1.5 * base
            assert lo <= dt <= hi, f"Sleep {i}={dt} not in jitter range [{lo}, {hi}]"

        # At least one sleep should NOT equal base exactly — jitter should
        # produce non-integer values with high probability.
        assert any(
            abs(dt - 2**i) > 1e-9 for i, dt in enumerate(sleeps)
        ), f"Sleeps look un-jittered (all exactly 2**attempt): {sleeps}"

    def test_api_call_rate_limit_gets_more_retries(self):
        from pipeline.api import api_call

        sem = asyncio.Semaphore(1)

        async def fake_sleep(dt):
            pass

        # --- RateLimitError: should retry 8 times ---
        client_rl, state_rl = self._mock_client_raising(openai.RateLimitError)
        with patch("asyncio.sleep", side_effect=fake_sleep):
            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(RuntimeError):
                    loop.run_until_complete(
                        api_call(
                            client_rl,
                            "test-model",
                            [{"role": "user", "content": "hi"}],
                            sem,
                        )
                    )
            finally:
                loop.close()
        assert (
            state_rl["count"] == 8
        ), f"Expected 8 rate-limit attempts, got {state_rl['count']}"

        # --- APIConnectionError: should retry 5 times ---
        client_cn, state_cn = self._mock_client_raising(openai.APIConnectionError)
        with patch("asyncio.sleep", side_effect=fake_sleep):
            loop = asyncio.new_event_loop()
            try:
                with pytest.raises(RuntimeError):
                    loop.run_until_complete(
                        api_call(
                            client_cn,
                            "test-model",
                            [{"role": "user", "content": "hi"}],
                            sem,
                        )
                    )
            finally:
                loop.close()
        assert (
            state_cn["count"] == 5
        ), f"Expected 5 connection attempts, got {state_cn['count']}"

    def test_api_call_eventual_success(self):
        from pipeline.api import api_call

        sem = asyncio.Semaphore(1)
        client, state = self._mock_client_raising(openai.RateLimitError, n_failures=2)

        async def fake_sleep(dt):
            pass

        with patch("asyncio.sleep", side_effect=fake_sleep):
            loop = asyncio.new_event_loop()
            try:
                content, reasoning, usage = loop.run_until_complete(
                    api_call(
                        client,
                        "test-model",
                        [{"role": "user", "content": "hi"}],
                        sem,
                    )
                )
            finally:
                loop.close()

        assert content == "ok content"
        assert state["count"] == 3  # 2 failures + 1 success


# ===========================================================================
# S4. run_concurrent tqdm throttling under non-tty
# ===========================================================================


class TestS4RunConcurrentTqdm:
    def test_run_concurrent_throttles_tqdm_when_not_tty(self):
        from pipeline import api as api_mod

        captured_kwargs: dict = {}

        async def trivial_coro():
            return 1

        async def fake_gather(*coros, **kwargs):
            captured_kwargs.update(kwargs)
            return [await c for c in coros]

        mock_stderr = MagicMock()
        mock_stderr.isatty = MagicMock(return_value=False)

        with (
            patch.object(api_mod, "tqdm_asyncio") as mock_tqdm,
            patch("sys.stderr", mock_stderr),
        ):
            mock_tqdm.gather = fake_gather
            api_mod.run_concurrent(trivial_coro(), desc="test")

        # Under non-tty, either mininterval >= 30 OR disable=True.
        mininterval = captured_kwargs.get("mininterval", 0)
        disable = captured_kwargs.get("disable", False)
        assert (
            mininterval >= 30 or disable is True
        ), f"Non-tty call did not throttle tqdm: {captured_kwargs}"

    def test_run_concurrent_normal_tqdm_when_tty(self):
        from pipeline import api as api_mod

        captured_kwargs: dict = {}

        async def trivial_coro():
            return 1

        async def fake_gather(*coros, **kwargs):
            captured_kwargs.update(kwargs)
            return [await c for c in coros]

        mock_stderr = MagicMock()
        mock_stderr.isatty = MagicMock(return_value=True)

        with (
            patch.object(api_mod, "tqdm_asyncio") as mock_tqdm,
            patch("sys.stderr", mock_stderr),
        ):
            mock_tqdm.gather = fake_gather
            api_mod.run_concurrent(trivial_coro(), desc="test")

        # Under tty, the throttling branch must NOT be active.
        # We allow: mininterval absent, mininterval < 30, or disable!=True.
        mininterval = captured_kwargs.get("mininterval", None)
        disable = captured_kwargs.get("disable", False)
        assert (
            disable is not True
        ), f"tty branch incorrectly disabled tqdm: {captured_kwargs}"
        if mininterval is not None:
            assert (
                mininterval < 30
            ), f"tty branch incorrectly throttled tqdm: {captured_kwargs}"


# ===========================================================================
# S5. resolve_prompt_path skip init for explicit _vN.md
# ===========================================================================


class TestS5ResolvePromptPath:
    def test_resolve_prompt_path_explicit_version_does_not_init(
        self, tmp_path, monkeypatch
    ):
        import pipeline.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROMPTS_DIR", tmp_path)

        # alias dir does NOT exist up front
        alias = "nonexistent_alias"
        assert not (tmp_path / alias).exists()

        with pytest.raises(Exception):
            cfg_mod.resolve_prompt_path("judge_reflection_v3.md", alias)

        # Critically: the alias directory should NOT have been created.
        assert not (
            tmp_path / alias
        ).exists(), "Explicit version should not trigger _init_model_prompts"

    def test_resolve_prompt_path_latest_does_init(self, tmp_path, monkeypatch):
        import pipeline.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROMPTS_DIR", tmp_path)

        alias = "fresh_alias"
        assert not (tmp_path / alias).exists()

        path = cfg_mod.resolve_prompt_path("judge_reflection_latest.md", alias)

        # The alias directory was created by _init_model_prompts.
        assert (tmp_path / alias).exists()
        # And the returned path is the v1 judge_reflection prompt in the new dir.
        assert path.name == "judge_reflection_v1.md"
        assert path.parent == (tmp_path / alias)
        assert path.exists()

    def test_resolve_prompt_path_explicit_version_existing_file(
        self, tmp_path, monkeypatch
    ):
        import pipeline.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "PROMPTS_DIR", tmp_path)

        alias = "real_alias"
        alias_dir = tmp_path / alias
        alias_dir.mkdir()
        expected = alias_dir / "judge_reflection_v3.md"
        expected.write_text("judge reflection v3 content")

        path = cfg_mod.resolve_prompt_path("judge_reflection_v3.md", alias)
        assert path == expected


# ===========================================================================
# S6. on_failure callback in generate_batch / judge_batch
# ===========================================================================


class TestS6OnFailureCallback:
    """Tests for the new `on_failure` callback parameter."""

    def _make_items(self, n=1):
        items = []
        for i in range(n):
            text = f"text {i} " * 10
            items.append(
                {
                    "item_id": f"item_{i}",
                    "subset": "dolma3",
                    "text": text,
                    "reflection_point": len(text) // 2,
                    "is_gold": False,
                }
            )
        return items

    def _make_prompt(self, tmp_path, name="gen_v1.md"):
        p = tmp_path / name
        p.write_text("System: {charter}")
        return p

    def _mock_client_returning(self, content):
        mock_client = AsyncMock(spec=openai.AsyncOpenAI)

        async def mock_create(**kwargs):
            resp = MagicMock()
            resp.choices = [MagicMock()]
            msg = resp.choices[0].message
            msg.content = content
            msg.reasoning_content = None
            usage = MagicMock()
            usage.prompt_tokens = 1
            usage.completion_tokens = 1
            usage.reasoning_tokens = 0
            usage.completion_tokens_details = {"reasoning_tokens": 0}
            resp.usage = usage
            return resp

        mock_client.chat.completions.create = mock_create
        return mock_client

    def _mock_client_raising(self, exc):
        mock_client = AsyncMock(spec=openai.AsyncOpenAI)

        async def mock_create(**kwargs):
            raise exc

        mock_client.chat.completions.create = mock_create
        return mock_client

    def test_generate_batch_on_failure_called_on_parse_error(
        self, isolated_storage, tmp_path
    ):
        from pipeline.phase2.run import generate_batch

        items = self._make_items(1)
        prompt = self._make_prompt(tmp_path)

        # Malformed JSON that will trip extract_json OR missing-fields assert.
        bad_content = "this is not json at all and has no braces"
        client = self._mock_client_returning(bad_content)

        recorded: list[dict] = []

        def on_failure(info):
            recorded.append(info)

        sem = asyncio.Semaphore(4)
        result = generate_batch(
            items,
            prompt,
            prompt,
            "charter",
            "test-model",
            iteration=1,
            client=client,
            semaphore=sem,
            save=False,
            on_failure=on_failure,
        )

        # The item was dropped from the result.
        assert len(result) == 0

        # The callback fired at least once.
        assert len(recorded) >= 1
        info = recorded[0]
        assert info["category"] == "parse"
        assert info["item_id"] == "item_0"
        assert "stage" in info
        assert info["stage"] in (
            "reflection",
            "preflection",
        ) or info[
            "stage"
        ].startswith("judge_")
        assert info.get("raw") is not None
        assert bad_content in info["raw"]

    def test_generate_batch_on_failure_called_on_api_runtime(
        self, isolated_storage, tmp_path
    ):
        from pipeline.phase2.run import generate_batch

        items = self._make_items(1)
        prompt = self._make_prompt(tmp_path)

        # Simulate api_call raising RuntimeError after retries exhausted by
        # patching api_call directly (simpler than making the client raise 5+
        # times).
        async def mock_api_call(*args, **kwargs):
            raise RuntimeError("api_runtime: connection exhausted")

        recorded: list[dict] = []

        def on_failure(info):
            recorded.append(info)

        sem = asyncio.Semaphore(4)
        with patch("pipeline.phase2.run.api_call", side_effect=mock_api_call):
            result = generate_batch(
                items,
                prompt,
                prompt,
                "charter",
                "test-model",
                iteration=1,
                client=MagicMock(),
                semaphore=sem,
                save=False,
                on_failure=on_failure,
            )

        assert len(result) == 0
        assert len(recorded) >= 1
        info = recorded[0]
        assert info["category"] == "api"
        assert info["reason"] == "api_runtime"
        assert info["item_id"] == "item_0"
        assert info.get("raw") is None

    def test_generate_batch_on_failure_default_none_no_crash(
        self, isolated_storage, tmp_path
    ):
        """With no on_failure kwarg, parse failures still silently drop the
        item (backward compat)."""
        from pipeline.phase2.run import generate_batch

        items = self._make_items(2)
        prompt = self._make_prompt(tmp_path)
        bad_content = "not json"
        client = self._mock_client_returning(bad_content)

        sem = asyncio.Semaphore(4)
        # Should not raise — should just return [] or short list.
        result = generate_batch(
            items,
            prompt,
            prompt,
            "charter",
            "test-model",
            iteration=1,
            client=client,
            semaphore=sem,
            save=False,
        )
        assert result == []

    def test_judge_batch_on_failure_called_on_parse_error(
        self, isolated_storage, tmp_path
    ):
        from pipeline.phase2.run import judge_batch

        # Build an "already generated" item suitable for judging.
        text = "full text content here " * 5
        rp = len(text) // 2
        generated_item = {
            "item_id": "judged_item",
            "subset": "dolma3",
            "text": text,
            "reflection_point": rp,
            "is_gold": False,
            "analysis": "a",
            "preflection": "p",
            "reflection": "r",
            "preflection_1p": "p1",
            "preflection_3p": "p3",
            "reflection_1p": "r1",
            "reflection_3p": "r3",
            "canary": None,
        }

        refl_prompt = self._make_prompt(tmp_path, name="judge_reflection_v1.md")
        refl_prompt.write_text("Judge reflection. Threshold: {accept_threshold}")
        prefl_prompt = self._make_prompt(tmp_path, name="judge_preflection_v1.md")
        prefl_prompt.write_text("Judge preflection. Threshold: {accept_threshold}")

        bad_content = "not valid json at all"
        client = self._mock_client_returning(bad_content)

        recorded: list[dict] = []

        def on_failure(info):
            recorded.append(info)

        sem = asyncio.Semaphore(4)
        result = judge_batch(
            [generated_item],
            refl_prompt,
            prefl_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=client,
            semaphore=sem,
            save=False,
            on_failure=on_failure,
        )

        assert len(result) == 0
        assert len(recorded) >= 1
        info = recorded[0]
        assert info["category"] == "parse"
        assert info["item_id"] == "judged_item"
        assert "stage" in info
        assert info["stage"].startswith("judge_")
        assert info.get("raw") is not None
        assert bad_content in info["raw"]

    def test_judge_batch_on_failure_default_none_no_crash(
        self, isolated_storage, tmp_path
    ):
        from pipeline.phase2.run import judge_batch

        text = "full text content here " * 5
        rp = len(text) // 2
        generated_item = {
            "item_id": "judged_item",
            "subset": "dolma3",
            "text": text,
            "reflection_point": rp,
            "is_gold": False,
            "analysis": "a",
            "preflection": "p",
            "reflection": "r",
            "preflection_1p": "p1",
            "preflection_3p": "p3",
            "reflection_1p": "r1",
            "reflection_3p": "r3",
            "canary": None,
        }

        refl_prompt = self._make_prompt(tmp_path, name="judge_reflection_v1.md")
        refl_prompt.write_text("Judge reflection. Threshold: {accept_threshold}")
        prefl_prompt = self._make_prompt(tmp_path, name="judge_preflection_v1.md")
        prefl_prompt.write_text("Judge preflection. Threshold: {accept_threshold}")

        client = self._mock_client_returning("not json")

        sem = asyncio.Semaphore(4)
        # Must not raise.
        result = judge_batch(
            [generated_item],
            refl_prompt,
            prefl_prompt,
            "test-model",
            iteration=1,
            accept_threshold=4.0,
            client=client,
            semaphore=sem,
            save=False,
        )
        assert result == []

    def test_on_failure_called_before_none_returned(self, isolated_storage, tmp_path):
        """The callback fires for every item that ends up missing from the
        result list. We verify by comparing the set of callback item_ids to
        the set of dropped items."""
        from pipeline.phase2.run import generate_batch

        items = self._make_items(3)
        prompt = self._make_prompt(tmp_path)

        client = self._mock_client_returning("not json")

        recorded_ids: set[str] = set()

        def on_failure(info):
            recorded_ids.add(info["item_id"])

        sem = asyncio.Semaphore(4)
        result = generate_batch(
            items,
            prompt,
            prompt,
            "charter",
            "test-model",
            iteration=1,
            client=client,
            semaphore=sem,
            save=False,
            on_failure=on_failure,
        )

        dropped_ids = {it["item_id"] for it in items} - {r["item_id"] for r in result}
        # Every dropped item must have had the callback invoked for it.
        assert dropped_ids.issubset(
            recorded_ids
        ), f"dropped={dropped_ids} recorded={recorded_ids}"
        # Since all 3 items get bad content, all 3 should be dropped.
        assert len(dropped_ids) == 3
