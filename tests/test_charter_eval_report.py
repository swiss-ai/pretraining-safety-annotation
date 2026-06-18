"""Dashboard data bridge: cards from a run dir, and the Gradio app over them."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pipeline.charter.eval.report import (
    build_cards,
    summarize_feedback,
    write_cards,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _judgment(scores: dict, aggregate: float, decision: str) -> dict:
    return {
        "reflection_1p": {
            "scores": scores,
            "aggregate": aggregate,
            "reasoning": "because reasons",
        },
        "reflection_aggregate": aggregate,
        "reflection_decision": decision,
    }


def _make_run(root: Path, run_id: str = "run1") -> Path:
    """Write a tiny generator-eval run dir (metadata + one gen × one judge)."""
    run_dir = root / run_id
    (run_dir / "generations").mkdir(parents=True)
    (run_dir / "judgments").mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "type": "generator_eval",
                "gold_judge": {"alias": "judgeA", "prompt_reflection": "judge_v1.md"},
                "candidates": [{"alias": "gen1", "prompt_reflection": "prompt_v1.md"}],
                "n_items": 2,
            }
        )
    )
    rows = [
        {
            "item_id": "i1",
            "text": "A long document about a difficult event." * 3,
            "reflection_point": 20,
            "subset": "en",
            "safety_score": 4,
            "reflection_1p": "Reading this account, I sit with the harm done to the "
            "volunteer who was attacked while trying to help others [2.1].",
            "analysis": "violence present",
            "reflection_charter_elements": ["[2.1]"],
            "judgment": _judgment(
                {"relevance": 5, "specificity": 4, "charter_grounding": 4, "voice_tone": 4},
                4.25,
                "accept",
            ),
        },
        {
            "item_id": "i2",
            "text": "A benign recipe for bread.",
            "reflection_point": 5,
            "subset": "deu",
            "safety_score": 1,
            # English reflection on a German source -> in_language == False.
            "reflection_1p": "This passage is an ordinary bread recipe and raises no ethical concerns worth noting here.",
            "analysis": "",
            "judgment": _judgment(
                {"relevance": 2, "specificity": 2, "charter_grounding": 3, "voice_tone": 3},
                2.5,
                "reject",
            ),
        },
    ]
    jud = run_dir / "judgments" / "judgeA__judge_v1.md__on__gen1__prompt_v1.md.jsonl"
    jud.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    # A realistic (but unused by build_cards) generation file.
    (run_dir / "generations" / "gen1__prompt_v1.md.jsonl").write_text(
        "\n".join(json.dumps({"item_id": r["item_id"], "text": r["text"]}) for r in rows)
    )
    return run_dir


def test_build_cards_joins_and_flattens(tmp_path):
    _make_run(tmp_path)
    cards = build_cards(["run1"], eval_dir=tmp_path)
    assert len(cards) == 2
    by_id = {c["item_id"]: c for c in cards}

    c1 = by_id["i1"]
    assert c1["generator"] == "gen1__prompt_v1.md"
    assert c1["judge"] == "judgeA__judge_v1.md"
    assert c1["gen_model"] == "gen1"
    assert c1["gen_prompt"] == "prompt_v1.md"
    assert c1["judge_model"] == "judgeA"
    assert c1["language"] == "en"
    assert c1["safety_score"] == 4
    assert c1["charter_elements"] == ["[2.1]"]
    assert c1["judge_decision"] == "accept"
    assert c1["judge_aggregate"] == 4.25
    assert c1["judge_scores"] == {
        "relevance": 5,
        "specificity": 4,
        "charter_grounding": 4,
        "voice_tone": 4,
    }
    assert c1["judge_reasoning"] == "because reasons"
    # English reflection on an English source -> in_language True.
    assert c1["reflection_lang"] == "en"
    assert c1["in_language"] is True

    # No stored citations and none in the text -> empty list.
    assert by_id["i2"]["charter_elements"] == []
    assert by_id["i2"]["judge_decision"] == "reject"
    # English reflection on a German (deu) source -> English fallback.
    assert by_id["i2"]["reflection_lang"] == "en"
    assert by_id["i2"]["in_language"] is False


def test_write_cards_payload(tmp_path):
    _make_run(tmp_path)
    out = tmp_path / "cards.json"
    n = write_cards(["run1"], out, eval_dir=tmp_path)
    assert n == 2
    payload = json.loads(out.read_text())
    assert payload["n_cards"] == 2
    assert payload["runs"] == ["run1"]
    assert len(payload["cards"]) == 2
    # Charter sections are baked in for the dashboard's citation tooltips.
    assert "2.1" in payload["charter_sections"]


def test_build_cards_skips_runs_without_judgments(tmp_path):
    (tmp_path / "empty").mkdir()
    assert build_cards(["empty"], eval_dir=tmp_path) == []


def test_summarize_feedback_agreement():
    rows = [
        {"verdict": "accept", "judge_decision": "accept"},   # agree
        {"verdict": "reject", "judge_decision": "accept"},   # disagree
        {"verdict": "reject", "judge_decision": "reject"},   # agree
        {"verdict": "accept", "judge_decision": None},       # no judge decision
    ]
    s = summarize_feedback(rows)
    assert s == {"n": 4, "accept": 2, "reject": 2, "n_vs_judge": 3, "agreement": pytest.approx(2 / 3)}


def _load_app(tmp_path, cards_path: Path):
    """Import dashboard/app.py fresh with env pointing at a fixture cards.json."""
    import os

    os.environ["CARDS_PATH"] = str(cards_path)
    os.environ["FEEDBACK_DIR"] = str(tmp_path / "fb")
    os.environ.pop("FEEDBACK_DATASET", None)
    spec = importlib.util.spec_from_file_location(
        "dashboard_app_under_test", REPO_ROOT / "dashboard" / "app.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_app_renders_and_collects_feedback(tmp_path):
    _make_run(tmp_path)
    cards_path = tmp_path / "cards.json"
    write_cards(["run1"], cards_path, eval_dir=tmp_path)

    app = _load_app(tmp_path, cards_path)
    assert len(app.CARDS) == 2
    assert app.SCHEDULER is None  # local-only mode

    # The Blocks build without error (the page actually constructs).
    app.build_demo()

    # Filter by language → one card (i1 is en).
    idxs = app.filter_indices("(all)", "(all)", "en", "(all)", "(all)")
    assert len(idxs) == 1
    meta, doc, refl, judge_md, poslabel = app.render(idxs, 0)
    assert poslabel == "1 / 1"
    # Doc shows ONLY up to the reflection point (i1 reflection_point=20), no after-text.
    assert "reflection injected here" not in doc
    assert doc == app.CARDS[idxs[0]]["text"][:20]
    assert "ACCEPT" in judge_md
    # Reflection is HTML with citation hover tooltips drawn from the value spec.
    assert 'class="cite"' in refl and "[2.1]" in refl
    assert 'class="tip"' in refl  # nested hover-tooltip span (not the title attribute)

    # Filter by model alias (not the full alias__prompt stem).
    assert app.CARDS[idxs[0]]["gen_model"] == "gen1"
    assert len(app.filter_indices("gen1", "(all)", "(all)", "(all)", "(all)")) == 2
    assert app.filter_indices("nope", "(all)", "(all)", "(all)", "(all)") == []
    # Model dropdown lists aliases, not stems.
    assert app._options("gen_model") == ["(all)", "gen1"]
    # Combined model + language.
    assert len(app.filter_indices("gen1", "(all)", "deu", "(all)", "(all)")) == 1

    # Answer-language filter: i1 is in-language (en/en), i2 is English fallback (en on deu).
    assert len(app.filter_indices("(all)", "(all)", "(all)", "(all)", "(all)", "in source language")) == 1
    assert len(app.filter_indices("(all)", "(all)", "(all)", "(all)", "(all)", "English fallback")) == 1
    # The card meta surfaces the answered-in language.
    meta_en, *_ = app.render(app.filter_indices("(all)", "(all)", "en", "(all)", "(all)"), 0)
    assert "answered in" in meta_en

    # Submitting a thumb writes a binary feedback row locally.
    status = app.submit_feedback(idxs, 0, "alice", "✅ accept", "spot on")
    assert "accept" in status
    written = [
        json.loads(line)
        for line in app.FEEDBACK_FILE.read_text().splitlines()
        if line.strip()
    ]
    assert len(written) == 1
    assert written[0]["verdict"] == "accept"
    assert written[0]["item_id"] == "i1"
    assert written[0]["reviewer"] == "alice"
    assert written[0]["judge_decision"] == "accept"


def test_annotator_order_per_name(tmp_path):
    _make_run(tmp_path)
    cards_path = tmp_path / "cards.json"
    write_cards(["run1"], cards_path, eval_dir=tmp_path)
    app = _load_app(tmp_path, cards_path)
    # Deterministic per name, and a full permutation over all cards.
    assert app.annotator_order("alice") == app.annotator_order("alice")
    assert sorted(app.annotator_order("alice")) == list(range(len(app.CARDS)))
    # filter_indices returns matches in the supplied per-annotator order.
    assert app.filter_indices("(all)", "(all)", "(all)", "(all)", "(all)", order=[1, 0]) == [1, 0]
    assert app.filter_indices("(all)", "(all)", "(all)", "(all)", "(all)", order=[0, 1]) == [0, 1]
