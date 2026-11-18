"""Tests for scripts/label_llm.py with a fake model call (no network)."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("label_llm", ROOT / "scripts" / "label_llm.py")
ll = importlib.util.module_from_spec(_spec)
sys.modules["label_llm"] = ll
_spec.loader.exec_module(ll)

ANN = ll.Annotator("fake", "fake", "fake-model", 3, 0.0, {})


def items(n, lang_check=()):
    return pd.DataFrame({"id": [f"c{i}" for i in range(n)], "text": [f"text {i}" for i in range(n)],
                         "lang_check": [f"c{i}" in lang_check for i in range(n)]})


def reply(entries):
    return json.dumps({"labels": entries})


def test_validate_response_accepts_fixes_and_rejects():
    idmap = {"1": "a", "2": "b", "3": "c", "4": "d"}
    text = reply([
        {"id": "1", "label": "favor", "confidence": "HIGH", "tags": ["sarcasm", "made_up"]},
        {"id": "2", "label": "Maybe", "confidence": "low", "tags": []},
        {"id": "3", "label": "None", "confidence": "low", "tags": [], "lang": "hi"},
        {"id": "99", "label": "None", "confidence": "low", "tags": []},
    ])
    valid, errors = ll.validate_response(text, idmap, lang_check_ids={"c"})
    assert valid["a"] == {"id": "a", "label": "Favour", "confidence": "high", "tags": ["sarcasm"]}
    assert valid["c"]["lang"] == "hi"
    assert errors == {"b": "bad label 'Maybe'", "d": "missing from response"}


def test_validate_response_requires_lang_for_lang_check_items():
    valid, errors = ll.validate_response(reply([{"id": "1", "label": "None", "confidence": "low", "tags": []}]),
                                         {"1": "a"}, {"a"})
    assert not valid and errors["a"].startswith("bad lang")


def test_validate_response_unparseable():
    valid, errors = ll.validate_response("not json", {"1": "a", "2": "b"}, set())
    assert not valid and set(errors) == {"a", "b"}


def test_user_message_uses_short_ids_and_lang_check():
    msg, idmap = ll.user_message(items(2, lang_check={"c1"}))
    assert json.loads(msg) == [{"id": "1", "text": "text 0"}, {"id": "2", "text": "text 1", "lang_check": True}]
    assert idmap == {"1": "c0", "2": "c1"}


def fake_call_factory(script):
    """script: list of callables(message) -> text or exception to raise."""
    calls = []

    def call(ann, key, system, message):
        calls.append(json.loads(message))
        action = script[min(len(calls) - 1, len(script) - 1)]
        out = action(json.loads(message))
        if isinstance(out, Exception):
            raise out
        return out, {"in": 1, "out": 1}
    return call, calls


def all_none(batch):
    return reply([{"id": x["id"], "label": "None", "confidence": "high", "tags": [],
                   **({"lang": "bn"} if x.get("lang_check") else {})} for x in batch])


def test_label_items_writes_resumes_and_retries(tmp_path):
    out = tmp_path / "labels_fake.jsonl"
    drop_first = lambda b: reply([{"id": x["id"], "label": "Against", "confidence": "low", "tags": []} for x in b[1:]])
    call, calls = fake_call_factory([drop_first, all_none, all_none, all_none])
    stats = ll.label_items(items(5), ANN, out, "main", None, "k", call=call, sleep=lambda s: None)
    recs = ll.read_jsonl(out)
    assert sorted(r["id"] for r in recs) == [f"c{i}" for i in range(5)]   # c0 retried and labelled
    assert stats["remaining"] == 0 and stats["failed"] == 0
    assert all(r["annotator"] == "fake" and r["set"] == "main" for r in recs)
    # resume: nothing left to do, no calls
    call2, calls2 = fake_call_factory([all_none])
    ll.label_items(items(5), ANN, out, "main", None, "k", call=call2, sleep=lambda s: None)
    assert calls2 == []


def test_label_items_logs_failures_after_three_attempts(tmp_path):
    out = tmp_path / "labels_fake.jsonl"
    call, calls = fake_call_factory([lambda b: "garbage"])
    stats = ll.label_items(items(2), ANN, out, "main", None, "k", call=call, sleep=lambda s: None)
    assert stats["failed"] == 2 and stats["remaining"] == 2 and len(calls) == ll.MAX_ATTEMPTS
    fails = ll.read_jsonl(tmp_path / "labels_fake_failures.jsonl")
    assert {f["id"] for f in fails} == {"c0", "c1"}


def test_label_items_waits_on_rate_limit_and_stops_on_daily_limit(tmp_path):
    out = tmp_path / "labels_fake.jsonl"
    waits = []
    call, _ = fake_call_factory([lambda b: ll.RateLimited(7), all_none, lambda b: ll.DailyLimitReached("day")])
    stats = ll.label_items(items(6), ANN, out, "main", None, "k", call=call, sleep=waits.append)
    assert 7 in waits
    assert stats["stopped"] == "day" and stats["remaining"] == 3


def test_label_items_splits_too_large_batches(tmp_path):
    out = tmp_path / "labels_fake.jsonl"
    call, calls = fake_call_factory([lambda b: ll.RequestTooLarge("big"), all_none])
    ll.label_items(items(3), ANN, out, "main", None, "k", call=call, sleep=lambda s: None)
    assert [len(c) for c in calls] == [3, 1, 2]
    assert len(ll.read_jsonl(out)) == 3


def test_max_requests_stops(tmp_path):
    call, calls = fake_call_factory([all_none])
    stats = ll.label_items(items(9), ANN, tmp_path / "l.jsonl", "main", 2, "k", call=call, sleep=lambda s: None)
    assert len(calls) == 2 and stats["remaining"] == 3 and stats["stopped"] == "max_requests=2"


def test_build_main_items_adds_lang_uncertain_rows():
    clean = pd.DataFrame({"id": ["a"], "video_id": ["v"], "text": ["ye sab"], "lang": ["hi"], "script": ["latin"]})
    removed = pd.DataFrame({"id": ["b", "c", "d"], "video_id": ["v"] * 3, "reason": ["other_language", "other_language", "english"],
                            "n_tokens": ["3", "2", "5"], "text": ["Modi chor he", "jai ho", "hello there you"]})
    it = ll.build_main_items(clean, removed)
    assert it["id"].tolist() == ["a", "b"] and it["lang_uncertain"].tolist() == [0, 1]


def test_translation_subset_is_stratified_and_checked():
    gold = pd.DataFrame({"id": [f"g{i}" for i in range(300)], "text": ["t"] * 300, "mapping_confidence": ["direct"] * 300,
                         "label": ["Favour"] * 180 + ["Against"] * 60 + ["None"] * 60})
    sub = ll.select_translation_subset(gold, n=100)
    assert len(sub) == 100 and sub["label"].value_counts().to_dict() == {"Favour": 60, "Against": 20, "None": 20}
    sub.loc[:, "text_hi"] = "यह ठीक है"
    sub.loc[:, "text_bn"] = "এটা ঠিক"
    assert ll.check_translations(sub) == []
    sub.loc[sub.index[0], "text_bn"] = "wrong script"
    assert ll.check_translations(sub) == [f"{sub['gold_id'].iloc[0]}: text_bn script is latin"]


def test_adjudication_queue_priority_and_requires_complete_labels():
    it = items(6)
    claude = pd.DataFrame({"id": it["id"], "label": ["Favour", "None", "None", "Against", "None", "None"]})
    gemini = pd.DataFrame({"id": it["id"], "label": ["Favour", "Favour", "None", "None", "None", "None"]})
    gold = pd.DataFrame({"id": ["g1"], "text": ["x"], "lang_check": [False]})
    q = ll.adjudication_queue(it, claude, gemini, gold, sample_size=2)
    assert [name for name, _ in q] == ["disagreement", "gold_en", "sample"]
    assert q[0][1]["id"].tolist() == ["c1", "c3"]
    assert len(q[2][1]) == 2 and not set(q[2][1]["id"]) & {"c1", "c3"}
    with pytest.raises(SystemExit):
        ll.adjudication_queue(it, claude.head(3), gemini, gold)


def test_validate_file_needs_no_gloss_but_needs_lang(tmp_path):
    p = tmp_path / "labels_claude.jsonl"
    p.write_text('{"id": "a", "label": "None", "confidence": "high", "tags": []}\n'
                 '{"id": "b", "label": "None", "confidence": "high", "tags": []}\n', encoding="utf-8")
    rep = ll.validate_file(p, {"a", "b", "c"}, lang_check_ids={"b"})
    assert rep["invalid"] == 1 and rep["missing"] == 2


def test_review_selection_takes_all_splits_plus_seeded_sample():
    m = pd.DataFrame({"id": [f"i{k}" for k in range(20)], "keep": ["1"] * 19 + ["0"],
                      "review_flag": ["1", "1"] + ["0"] * 18, "final_label": ["", ""] + ["None"] * 17 + [""]})
    sel = ll.review_selection(m, n=5)
    assert sel["review_reason"].value_counts().to_dict() == {"random_sample": 5, "three_way_split": 2}
    assert "i19" not in set(sel["id"])
    assert sel["id"].tolist() == ll.review_selection(m, n=5)["id"].tolist()


def test_validate_glosses():
    valid, errors = ll.validate_glosses('{"glosses": [{"id": "1", "gloss": "It is hot"}, {"id": "2", "gloss": " "}]}',
                                        {"1": "a", "2": "b"})
    assert valid == {"a": {"id": "a", "gloss": "It is hot"}} and errors == {"b": "missing gloss"}


def test_single_item_too_large_is_logged_not_dropped(tmp_path):
    out = tmp_path / "labels_fake.jsonl"
    call, calls = fake_call_factory([lambda b: ll.RequestTooLarge("cut off")])
    stats = ll.label_items(items(1), ANN, out, "main", None, "k", call=call, sleep=lambda s: None)
    assert stats["failed"] == 1 and stats["remaining"] == 1
    assert [f["id"] for f in ll.read_jsonl(tmp_path / "labels_fake_failures.jsonl")] == ["c0"]
