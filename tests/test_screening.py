"""Tests for the `screen` mode helpers in scripts/collect_youtube.py (no API calls)."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("collect_youtube", ROOT / "scripts" / "collect_youtube.py")
cy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cy)

CFG = {**cy.SCREEN_DEFAULTS, "target_per_language": [1, 2], "max_per_channel": 2}


# --------------------------------------------------------------------------- keyword language check

@pytest.mark.parametrize("text, expected", [
    ("जलवायु परिवर्तन बहुत गंभीर है", "hi"),
    ("জলবায়ু পরিবর্তন খুব গুরুতর", "bn"),
    ("climate change बहुत serious है yaar", "hi"),          # mostly Latin, but Devanagari words -> Hindi
    ("global warming নিয়ে video", "bn"),
    ("ye sab jhooth hai bhai", "hi"),
    ("Global warming nahi hai, kuch nahi hoga", "hi"),
    ("ami khub bhoy pai, eta ki hobe", "bn"),
    ("tumi thik bolecho dada", "bn"),
    ("ki bolbo", "indic"),                                  # "ki" is in both lists -> romanised, unclear
    ("Climate change is real and serious", "other"),
    ("HAI or not, KYA difference", "hi"),                    # case-insensitive
    ("khair", "other"),                                      # whole words only: no "hai" inside "khair"
    ("Nice video 👍👍", "other"),
    ("🔥🔥🔥", "other"),
    ("@someone_xy hai bhai", "hi"),                          # handle removed before checking
    ("hai na", "indic"),                                     # 1 Hindi + 1 Bengali marker -> tie
])
def test_comment_language(text, expected):
    assert cy.comment_language(text) == expected


def test_rule_language_stats_share_and_dominant():
    comments = ["ye sab jhooth hai", "बहुत गर्मी है", "nice video", "ami jani na", "ki bolbo"]
    stats = cy.rule_language_stats(comments)
    assert stats["rule_indic_share"] == 0.8      # 4 of 5 (indic counts as Indic)
    assert stats["rule_lang"] == "hi"            # 2 hi vs 1 bn
    assert cy.rule_language_stats([]) == {"rule_indic_share": 0.0, "rule_lang": ""}


def test_keyword_lists_are_extendable_constant():
    assert "bhai" in cy.ROMANISED_KEYWORDS["hi"] and "dada" in cy.ROMANISED_KEYWORDS["bn"]


# --------------------------------------------------------------------------- JSON validator

def rec(vid, **over):
    base = {"video_id": vid, "climate_relevant": True, "lang": "hi", "leaning": "mainstream",
            "comments_discuss_climate": True, "indic_comment_share": 0.8, "confidence": "high",
            "reason": "Hindi explainer on heatwaves"}
    return {**base, **over}


def test_validate_accepts_good_batch():
    out = cy.validate_screening([rec("a"), rec("b", indic_comment_share=1, lang="bn")], ["a", "b"])
    assert set(out) == {"a", "b"}
    assert out["b"]["indic_comment_share"] == 1.0 and isinstance(out["b"]["indic_comment_share"], float)


def test_validate_ignores_extra_keys():
    out = cy.validate_screening([{**rec("a"), "extra": 1}], ["a"])
    assert "extra" not in out["a"]


@pytest.mark.parametrize("bad, fragment", [
    ([rec("a")], "missing video_ids ['b']"),
    ([rec("a"), rec("b"), rec("c")], "unexpected video_id"),
    ([rec("a"), rec("a"), rec("b")], "duplicate video_id"),
    ([rec("a"), {k: v for k, v in rec("b").items() if k != "leaning"}], "missing ['leaning']"),
    ([rec("a"), rec("b", climate_relevant="true")], "bad climate_relevant"),
    ([rec("a"), rec("b", indic_comment_share=1.4)], "bad indic_comment_share"),
    ([rec("a"), rec("b", indic_comment_share=True)], "bad indic_comment_share"),
    ([rec("a"), rec("b", lang="hindi")], "bad lang"),
    ([rec("a"), rec("b", leaning="denier")], "bad leaning"),
    ([rec("a"), rec("b", confidence="very high")], "bad confidence"),
    ([rec("a"), "oops"], "not an object"),
    ({"video_id": "a"}, "expected a JSON list"),
])
def test_validate_rejects(bad, fragment):
    with pytest.raises(cy.ScreeningError) as exc:
        cy.validate_screening(bad, ["a", "b"])
    assert fragment in str(exc.value)


def test_parse_json_text_handles_fences_and_wrapper():
    payload = json.dumps([rec("a")])
    assert cy.parse_json_text(f"```json\n{payload}\n```") == [rec("a")]
    assert cy.parse_json_text(json.dumps({"videos": [rec("a")]})) == [rec("a")]
    with pytest.raises(cy.ScreeningError, match="malformed JSON"):
        cy.parse_json_text('[{"video_id": "a",')


# --------------------------------------------------------------------------- keep decision

def row(vid, channel="c1", comments=100, lang="hi", leaning="mainstream", rule=0.9, share=0.9,
        relevant=True, discuss=True, conf="high", screened=True, n=20):
    return {"video_id": vid, "channel": channel, "comment_count": comments, "query_lang": lang,
            "sample_comments": n, "rule_indic_share": rule, "rule_lang": lang, "screened": screened,
            "climate_relevant": relevant, "comments_discuss_climate": discuss, "lang": lang,
            "leaning": leaning, "indic_comment_share": share, "confidence": conf}


def decide(rows):
    return cy.decide_keep(pd.DataFrame(rows), CFG).set_index("video_id")


def test_decide_reasons_in_order():
    out = decide([
        row("few", n=3),
        row("norule", rule=0.15, screened=False, relevant=None, discuss=None, lang=None, leaning=None,
            share=None, conf=None, channel="x0"),
        row("fail", screened=False, relevant=None, discuss=None, lang=None, leaning=None, share=None, conf=None),
        row("irrel", relevant=False, channel="x1"),
        row("english", lang="other", channel="x3"),
        row("lowshare", rule=0.2, share=0.3, channel="x4"),
        row("unrel", leaning="unrelated", channel="x5"),
    ])
    assert out["reason"].to_dict() == {
        "few": "too_few_comments", "norule": "rule_no_indic", "fail": "screening_failed",
        "irrel": "not_climate_relevant", "english": "lang_other", "lowshare": "low_indic_share",
        "unrel": "leaning_unrelated"}
    assert out["keep"].sum() == 0


def test_off_topic_comments_do_not_block_keep():
    out = decide([row("offtopic", discuss=False)])
    assert out.loc["offtopic", "keep"] == 1 and out.loc["offtopic", "reason"] == "kept"


def test_rule_prefilter_threshold_is_inclusive():
    out = decide([row("edge", rule=0.2, share=0.6), row("below", rule=0.19, channel="c2")])
    assert out.loc["edge", "keep"] == 1
    assert out.loc["below", "reason"] == "rule_no_indic"


def test_language_passes_on_either_measure_and_flags_disagreement():
    out = decide([row("a", rule=0.3, share=0.7, conf="high"), row("b", rule=0.6, share=0.6, channel="c2")])
    assert out.loc["a", "keep"] == 1 and out.loc["a", "lang_check_disagree"] == 1
    assert out.loc["a", "review_flag"] == 1
    assert out.loc["b", "lang_check_disagree"] == 0 and out.loc["b", "review_flag"] == 0


def test_low_confidence_is_flagged():
    assert decide([row("a", conf="low")]).loc["a", "review_flag"] == 1


def test_channel_cap_prefers_more_comments():
    out = decide([row("a", comments=10), row("b", comments=500), row("c", comments=50, lang="bn")])
    # cap 2 per channel: a (fewest comments) is dropped
    assert out.loc["a", "reason"] == "channel_cap"
    assert out.loc[["b", "c"], "keep"].tolist() == [1, 1]


def test_language_target_keeps_sceptic_first_and_never_adds_irrelevant():
    out = decide([
        row("m1", channel="c1", comments=900),
        row("m2", channel="c2", comments=800),
        row("s1", channel="c3", comments=5, leaning="sceptic"),
        row("bad", channel="c4", relevant=False, lang="bn"),
    ])
    # target max 2 for hi: the sceptic video is kept before the bigger mainstream ones
    assert out.loc["s1", "keep"] == 1 and out.loc["m1", "keep"] == 1
    assert out.loc["m2", "reason"] == "over_target"
    assert out.loc["bad", "keep"] == 0  # bn target missed, irrelevant video still not kept


def test_mixed_lang_uses_rule_language_for_target():
    r = row("m", lang="mixed")
    r["rule_lang"] = "bn"
    assert decide([r]).loc["m", "target_lang"] == "bn"


def test_import_claude_decisions_validates_and_logs(tmp_path):
    ddir, raw = tmp_path / "decisions", tmp_path / "raw.jsonl"
    ddir.mkdir()
    (ddir / "batch_01.json").write_text(json.dumps({"video_ids": ["a", "b"], "results": [rec("a"), rec("b")]}),
                                        encoding="utf-8")
    (ddir / "batch_02.json").write_text(json.dumps({"video_ids": ["c"], "results": [rec("c", lang="hindi")]}),
                                        encoding="utf-8")
    stored = {}
    failures = cy.import_claude_decisions(ddir, {"a", "b", "c"}, stored.update, raw)
    assert set(stored) == {"a", "b"}
    assert set(failures) == {"c"} and "bad lang" in failures["c"]
    log = cy.read_jsonl(raw)
    assert all(r["screener"] == "claude" for r in log) and len(log) == 2


def test_jsonl_roundtrip_keeps_unicode_line_separators(tmp_path):
    # U+2028 / U+0085 inside comment text must not split a record (str.splitlines would)
    path = tmp_path / "x.jsonl"
    records = [{"video_id": "a", "comments": ["line one line two", "x\u0085y"]}, {"video_id": "b"}]
    for r in records:
        cy.append_jsonl(path, r)
    assert cy.read_jsonl(path) == records


def test_is_sceptic_query():
    assert cy.is_sceptic_query("global warming scam | climate change hindi")
    assert cy.is_sceptic_query("জলবায়ু পরিবর্তন গুজব")
    assert not cy.is_sceptic_query("जलवायु परिवर्तन")
