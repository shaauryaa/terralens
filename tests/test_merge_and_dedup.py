"""Tests for scripts/merge_and_dedup.py on synthetic comments (no files read)."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("merge_and_dedup", ROOT / "scripts" / "merge_and_dedup.py")
md = importlib.util.module_from_spec(_spec)
sys.modules["merge_and_dedup"] = md
_spec.loader.exec_module(md)

META = pd.DataFrame({"video_id": ["v1", "v2", "v3"], "video_lang": ["hi", "bn", "hi"],
                     "leaning": ["mainstream", "news", "sceptic"]})


def comments(rows):
    return pd.DataFrame([{"id": f"c{i}", "video_id": v, "text": t, "published_at": f"2025-01-{i + 1:02d}T00:00:00Z"}
                         for i, (v, t) in enumerate(rows)])


def test_text_helpers():
    assert md.core_text("बहुत गर्मी है 🔥🔥 @user https://x.y") == "बहुत गर्मी है"
    assert md.n_tokens("बहुत गर्मी है :fire:") == 4
    assert md.n_tokens(md.core_text("बहुत गर्मी है 🔥")) == 3           # matras stay inside the word
    assert md.dedup_key("Bilkul  SAHI baat!!!") == "bilkul sahi baat"


def test_pipeline_assigns_each_reason():
    raw = comments([
        ("v1", "https://youtube.com/x @user"),                    # empty
        ("v1", "😂😂😂"),                                            # emoji_only
        ("v1", "Climate change is a serious problem"),            # english
        ("v1", "Jai Shree Ram"),                                  # other_language
        ("v1", "sahi baat"),                                      # too_short (2 tokens)
        ("v1", "ye sab jhooth hai bhai"),                         # kept
        ("v2", "Ye sab JHOOTH hai bhai!!"),                       # exact_dup of c5 (other video)
        ("v2", "ami khub bhoy pai eta ki hobe"),                  # kept
        ("v2", "ami khub bhoy pai eta ki hobe re"),               # near_dup of c7
        ("v3", "गर्मी हर साल बढ़ रही है 🔥"),                          # kept, emoji kept in text
    ])
    clean, removed, annotated = md.run_pipeline(raw, META)
    reasons = dict(zip(annotated["id"], annotated["reason"]))
    assert reasons == {"c0": "empty", "c1": "emoji_only", "c2": "english", "c3": "other_language",
                       "c4": "too_short", "c5": "", "c6": "exact_dup", "c7": "", "c8": "near_dup", "c9": ""}
    assert removed.set_index("id").loc["c6", "duplicate_of"] == "c5"
    assert removed.set_index("id").loc["c8", "duplicate_of"] == "c7"
    assert removed.set_index("id").loc["c8", "similarity"] > 0.9
    assert clean["id"].tolist() == ["c5", "c7", "c9"]
    assert clean["lang"].tolist() == ["hi", "bn", "hi"]
    assert clean["script"].tolist() == ["latin", "latin", "devanagari"]
    assert ":fire:" in clean.set_index("id").loc["c9", "text"]
    assert len(clean) + len(removed) == len(raw)                 # nothing dropped silently


def test_earliest_copy_is_kept():
    raw = comments([("v1", "ye sab jhooth hai bhai"), ("v2", "ye sab jhooth hai bhai")])
    raw.loc[0, "published_at"] = "2025-02-01T00:00:00Z"            # c0 is later than c1
    clean, removed, _ = md.run_pipeline(raw, META)
    assert clean["id"].tolist() == ["c1"] and removed["duplicate_of"].tolist() == ["c1"]


def test_romanised_tie_uses_video_language():
    clean, _, _ = md.run_pipeline(comments([("v2", "ki bolbo eta niye"), ("v1", "nai nai ki")]), META)
    assert clean["lang"].tolist() == ["bn", "hi"]


def test_near_duplicates_threshold():
    texts = ["global warming bahut serious hai", "global warming bahut serious hai na", "kuch alag baat"]
    assert md.near_duplicates(texts, ["a", "b", "c"], 0.9).keys() == {"b"}
    assert md.near_duplicates(texts, ["a", "b", "c"], 0.999) == {}


def test_allocate_equal_shares_with_cap():
    assert md.allocate({"a": 50, "b": 300, "c": 10}, budget=1000, cap=100) == {"a": 50, "b": 100, "c": 10}
    take = md.allocate({"a": 50, "b": 300, "c": 10}, budget=100, cap=100)
    assert sum(take.values()) == 100 and take["c"] == 10 and max(take.values()) <= 100
    assert abs(take["a"] - take["b"]) <= 1
    assert md.allocate({"a": 5}, budget=0, cap=100) == {"a": 0}


def test_sample_takes_all_sceptic_then_caps_per_video():
    rows = []
    for vid, n, leaning in [("s1", 30, "sceptic"), ("m1", 200, "mainstream"), ("m2", 200, "news"),
                            ("m3", 20, "news")]:
        rows += [{"id": f"{vid}_{i}", "video_id": vid, "text": "t", "lang": "hi", "script": "latin",
                  "leaning": leaning} for i in range(n)]
    clean = pd.DataFrame(rows)
    sample = md.sample_for_labelling(clean, n=200, cap=100, seed=42)
    per_video = sample["video_id"].value_counts().to_dict()
    assert len(sample) == 200
    assert per_video["s1"] == 30 and set(sample[sample.video_id == "s1"]["sample_group"]) == {"sceptic_video"}
    assert per_video["m3"] == 20 and per_video["m1"] + per_video["m2"] == 150
    assert max(per_video["m1"], per_video["m2"]) <= 100
    again = md.sample_for_labelling(clean, n=200, cap=100, seed=42)
    assert sample["id"].tolist() == again["id"].tolist()          # seeded
