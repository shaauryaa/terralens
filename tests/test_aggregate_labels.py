"""Tests for scripts/aggregate_labels.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("aggregate_labels", ROOT / "scripts" / "aggregate_labels.py")
ag = importlib.util.module_from_spec(_spec)
sys.modules["aggregate_labels"] = ag
_spec.loader.exec_module(ag)


@pytest.mark.parametrize("c, g, o, expected", [
    ("Favour", "Favour", None, ("Favour", "agree", 0)),
    ("Favour", "None", "None", ("None", "adjudicated", 0)),
    ("Favour", "None", "Against", (None, "split", 1)),
    ("Favour", "None", None, (None, "pending", 0)),
    (None, "None", "None", (None, "missing", 0)),
])
def test_resolve(c, g, o, expected):
    assert ag.resolve(c, g, o) == expected


@pytest.mark.parametrize("votes, expected", [
    (("hi", "hi", None), ("hi", 1)),
    (("hi", "other", None), ("", 0)),          # only one hi/bn vote
    (("bn", "other", "bn"), ("bn", 1)),
    (("hi", "bn", None), ("hi", 1)),           # tie -> Claude's
    (("other", "bn", "hi"), ("bn", 1)),        # tie, Claude says other -> first hi/bn vote
])
def test_resolve_lang_for_uncertain_rows(votes, expected):
    row = {"lang_uncertain": "1", "lang": "", **{f"lang_{a}": v for a, v in zip(ag.ANNOTATORS, votes)}}
    assert ag.resolve_lang(row) == expected
    assert ag.resolve_lang({"lang_uncertain": "0", "lang": "bn"}) == ("bn", 1)


def test_fleiss_kappa_known_values():
    perfect = pd.DataFrame({"a": ["Favour", "None", "Against"], "b": ["Favour", "None", "Against"],
                            "c": ["Favour", "None", "Against"]})
    assert ag.fleiss_kappa(perfect) == pytest.approx(1.0)
    # Fleiss (1971)-style hand check: 2 items, 3 raters
    r = pd.DataFrame({"a": ["Favour", "Favour"], "b": ["Favour", "None"], "c": ["None", "None"]})
    counts = np.array([[2, 0, 1], [1, 0, 2]], dtype=float)
    p_i = ((counts * (counts - 1)).sum(1)) / 6
    p_j = counts.sum(0) / 6
    expected = (p_i.mean() - (p_j ** 2).sum()) / (1 - (p_j ** 2).sum())
    assert ag.fleiss_kappa(r) == pytest.approx(expected)


def test_merge_and_agreement_report():
    items = pd.DataFrame({"id": ["a", "b", "c", "d", "e"], "video_id": "v", "text": "t",
                          "lang": ["hi", "hi", "bn", "bn", ""], "script": ["latin", "devanagari", "bengali", "bengali", "latin"],
                          "lang_uncertain": ["0", "0", "0", "0", "1"]})
    lab = lambda rows: pd.DataFrame(rows, columns=["id", "label", "confidence", "tags", "lang", "set"])
    labels = {
        "claude": lab([[i, l, "high", [], lg, "main"] for i, l, lg in
                       [("a", "Favour", None), ("b", "None", None), ("c", "Against", None), ("d", "None", None), ("e", "None", "hi")]]),
        "gemini": lab([[i, l, "high", [], lg, "main"] for i, l, lg in
                       [("a", "Favour", None), ("b", "Favour", None), ("c", "None", None), ("d", "None", None), ("e", "None", "other")]]),
        "gptoss": lab([["b", "Favour", "high", [], None, "disagreement"],
                       ["c", "Favour", "high", [], None, "disagreement"]]),
    }
    m = ag.merge(items, labels)
    assert m.set_index("id")["label_source"].to_dict() == {"a": "agree", "b": "adjudicated", "c": "split", "d": "agree", "e": "agree"}
    assert m.set_index("id").loc["b", "final_label"] == "Favour"
    assert m.set_index("id").loc["e", "keep"] == 0                  # only one hi/bn vote
    rep = ag.agreement_report(m, labels)
    assert rep["kept_items"] == 4
    assert rep["adjudication"] == {"disagreements": 2, "sent_to_gptoss": 2, "resolved_by_gptoss": 1,
                                   "three_way_splits": 1, "pending": 0}
    assert rep["claude_vs_gemini"]["overall"]["pct_agreement"] == 50.0
    assert rep["lang_uncertain"] == {"items": 1, "kept": 0, "kept_by_lang": {}}


def test_gold_check_scores_each_view():
    gi = pd.DataFrame({"id": ["g1", "g2", "g1__hi"], "gold_id": ["g1", "g2", "g1"], "set": ["gold_en", "gold_en", "gold_hi"],
                       "human_label": ["Favour", "Against", "Favour"], "mapping_confidence": ["direct", "approximate", "direct"]})
    pred = pd.DataFrame({"id": ["g1", "g2", "g1__hi"], "label": ["Favour", "None", "Favour"]})
    empty = pd.DataFrame(columns=["id", "label"])
    out = ag.gold_check(gi, {"claude": empty, "gemini": pred, "gptoss": empty})
    gem = out["annotators"]["gemini"]
    assert gem["en_all"]["accuracy"] == 0.5 and gem["en_direct"]["accuracy"] == 1.0
    assert gem["hi"]["n"] == 1 and gem["en_translation_subset"]["n"] == 1
    assert out["annotators"]["claude"] == {"status": "no gold labels yet"}


def test_read_labels_schema_needs_no_gloss_or_set(tmp_path):
    p = tmp_path / "labels_claude.jsonl"
    p.write_text('{"id": "a", "label": "None", "confidence": "high", "tags": []}\n'
                 '{"id": "b", "label": "Favour", "confidence": "sure", "tags": []}\n'
                 '{"id": "c", "label": "Favour", "confidence": "low", "tags": "x"}\n',
                 encoding="utf-8")
    assert ag.read_labels(p, "claude")["id"].tolist() == ["a"]
    assert ag.REJECTED["labels_claude.jsonl"] == 2


def test_claude_bengali_gold_is_not_scored_but_hindi_is():
    gi = pd.DataFrame({"id": ["g1", "g1__hi", "g1__bn"], "gold_id": ["g1"] * 3, "set": ["gold_en", "gold_hi", "gold_bn"],
                       "human_label": ["Favour"] * 3, "mapping_confidence": ["direct"] * 3})
    pred = pd.DataFrame({"id": ["g1", "g1__hi", "g1__bn"], "label": ["Favour"] * 3})
    empty = pd.DataFrame(columns=["id", "label"])
    out = ag.gold_check(gi, {"claude": pred, "gemini": pred, "gptoss": empty})
    assert out["annotators"]["claude"]["bn"]["status"].startswith("not independent")
    assert out["annotators"]["claude"]["hi"]["accuracy"] == 1.0
    assert out["annotators"]["claude"]["en_all"]["accuracy"] == 1.0
    assert out["annotators"]["gemini"]["bn"]["accuracy"] == 1.0


def test_with_translated_replaces_first_pass_records():
    first = pd.DataFrame({"id": ["g1", "g1__hi", "g1__bn"], "label": ["Favour", "Favour", "Favour"]})
    second = pd.DataFrame({"id": ["g1__hi", "g1__bn"], "label": ["None", "None"]})
    out = ag.with_translated(first, second).set_index("id")["label"]
    assert out.to_dict() == {"g1": "Favour", "g1__hi": "None", "g1__bn": "None"}
    assert ag.with_translated(first, second.iloc[0:0]).equals(first)
