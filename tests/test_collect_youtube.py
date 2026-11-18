"""Parsing tests for scripts/collect_youtube.py on saved (synthetic) API responses. No network calls."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "youtube"

_spec = importlib.util.spec_from_file_location("collect_youtube", ROOT / "scripts" / "collect_youtube.py")
cy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cy)


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


JOB = {"lang": "hi", "region": "IN", "query": "जलवायु परिवर्तन", "relevance_language": "hi"}


def test_build_search_jobs_expands_bengali_regions():
    config = {
        "settings": {"region_code": "IN"},
        "hindi": {"relevance_language": "hi", "queries": ["q1", "q2"]},
        "bengali": {"relevance_language": "bn", "region_codes": ["IN", "BD"], "queries": ["b1"]},
    }
    jobs = cy.build_search_jobs(config)
    assert [(j["lang"], j["region"], j["query"]) for j in jobs] == [
        ("hi", "IN", "q1"), ("hi", "IN", "q2"), ("bn", "IN", "b1"), ("bn", "BD", "b1")]


def test_build_search_jobs_on_real_config():
    jobs = cy.build_search_jobs(cy.load_config(cy.DEFAULT_CONFIG))
    assert {j["lang"] for j in jobs} == {"hi", "bn"}
    assert len({cy.job_key(j) for j in jobs}) == len(jobs)


def test_parse_search_response_skips_non_videos():
    hits = cy.parse_search_response(load("search_response.json"), JOB)
    assert [h["video_id"] for h in hits] == ["AAAAAAAAAA1", "-BBBBBBBBB2"]
    assert hits[0]["channel"] == "Example News Hindi"
    assert hits[0]["lang"] == "hi" and hits[0]["query"] == JOB["query"]


def test_parse_videos_response_comment_count_and_disabled():
    stats = cy.parse_videos_response(load("videos_response.json"))
    assert stats["AAAAAAAAAA1"]["comment_count"] == 245
    assert stats["-BBBBBBBBB2"]["comment_count"] is None  # commentCount missing = comments disabled


def test_merge_candidates_dedups_filters_and_joins_queries():
    hits = cy.parse_search_response(load("search_response.json"), JOB)
    bn_job = {**JOB, "lang": "bn", "query": "জলবায়ু পরিবর্তন"}
    hits += cy.parse_search_response(load("search_response.json"), bn_job)
    hits.append({"video_id": "CCCCCCCCCC3", "title": "t", "channel": "c", "published_at": "", "query": "q", "lang": "hi"})
    stats = cy.parse_videos_response(load("videos_response.json"))
    stats["CCCCCCCCCC3"] = {"comment_count": 5, "title": "t", "channel": "c", "published_at": ""}

    df, report = cy.merge_candidates(hits, stats, min_comment_count=30)
    assert list(df.columns) == cy.CANDIDATE_COLUMNS
    assert list(df["video_id"]) == ["AAAAAAAAAA1"]
    row = df.iloc[0]
    assert row["query"] == "जलवायु परिवर्तन | জলবায়ু পরিবর্তন"
    assert row["lang"] == "hi|bn"
    assert row["comment_count"] == 245
    assert row["url"] == "https://www.youtube.com/watch?v=AAAAAAAAAA1"
    assert report == {"hits": 5, "unique_videos": 3, "no_stats": 0, "comments_disabled": 1,
                      "below_min_comments": 1, "kept": 1}


def test_parse_comment_threads_flags_replies_and_drops_author_fields():
    rows = cy.parse_comment_threads(load("comment_threads_response.json"), "AAAAAAAAAA1", "hi")
    assert [r["id"] for r in rows] == ["UgxThread0001", "UgxThread0001.Reply0001", "UgxThread0002"]
    assert [r["is_reply"] for r in rows] == [0, 1, 0]
    assert [r["like_count"] for r in rows] == [12, 3, 0]
    assert all(set(r) == set(cy.COMMENT_COLUMNS) for r in rows)
    assert all(r["query_lang"] == "hi" and r["video_id"] == "AAAAAAAAAA1" for r in rows)

    dumped = json.dumps(rows, ensure_ascii=False)
    for leaked in ("SomeViewer", "OtherViewer", "ThirdViewer", "UCauthor", "yt3.ggpht.com"):
        assert leaked not in dumped
    assert rows[1]["text"] == "@user नहीं भाई, जलवायु परिवर्तन सच है"
    assert rows[2]["text"] == "बहुत अच्छी जानकारी, mail me at <email>"


def test_anonymise_text_replaces_emails_and_handles():
    assert cy.anonymise_text("contact first.last+yt@mail.co.in or @Some_One-1") == "contact <email> or @user"
    assert cy.anonymise_text("@a.b ne kaha: x_y@gmail.com pe likho") == "@user ne kaha: <email> pe likho"
    assert cy.anonymise_text("मेरा ईमेल राम@डाक.भारत है") == "मेरा ईमेल <email> है"
    assert cy.anonymise_text("@राम_भाई सही कहा, @সুমন দা") == "@user सही कहा, @user দা"
    assert cy.anonymise_text("no @ here, 5 @ 10 rupees") == "no @ here, 5 @ 10 rupees"
    assert cy.anonymise_text(None) == ""


def test_parse_keep_values():
    assert [cy.parse_keep(v) for v in ["1", 1, "1.0", " yes ", "X", "true"]] == [1] * 6
    assert [cy.parse_keep(v) for v in ["", "0", None, float("nan"), "no", 0]] == [0] * 6


def test_carry_over_keep_preserves_hand_filled_values():
    new = pd.DataFrame({"video_id": ["a", "b", "c"], "keep": ["", "", ""]})
    old = pd.DataFrame({"video_id": ["a", "b"], "keep": ["1", ""]})
    assert list(cy.carry_over_keep(new, old)["keep"]) == ["1", "", ""]
    assert list(cy.carry_over_keep(new, None)["keep"]) == ["", "", ""]


def test_load_candidates_repairs_ids_from_url(tmp_path):
    path = tmp_path / "c.csv"
    pd.DataFrame({"video_id": ["#NAME?", "AAAAAAAAAA1"], "keep": ["1", ""],
                  "url": [cy.video_url("-BBBBBBBBB2"), cy.video_url("AAAAAAAAAA1")]}
                 ).to_csv(path, index=False, encoding="utf-8-sig")
    df = cy.load_candidates(path)
    assert list(df["video_id"]) == ["-BBBBBBBBB2", "AAAAAAAAAA1"]


class _FakeHttpError:
    def __init__(self, payload):
        self.content = json.dumps(payload).encode("utf-8")


def test_error_reason_reads_quota_and_disabled_reasons():
    quota = _FakeHttpError({"error": {"code": 403, "errors": [{"reason": "quotaExceeded"}]}})
    disabled = _FakeHttpError({"error": {"code": 403, "errors": [{"reason": "commentsDisabled"}]}})
    assert cy.error_reason(quota) in cy.QUOTA_REASONS
    assert cy.error_reason(disabled) in cy.SKIP_VIDEO_REASONS
    assert cy.error_reason(_FakeHttpError({"oops": 1})) == ""


def test_video_comment_cap_uses_per_video_override_and_limit():
    settings = {"max_comments_per_video": 150, "per_video_max_comments": {"lesson": 50, "big": 500}}
    assert cy.video_comment_cap("other", settings) == 150
    assert cy.video_comment_cap("lesson", settings) == 50
    assert cy.video_comment_cap("big", settings) == 150          # an override never raises the cap
    assert cy.video_comment_cap("lesson", settings, limit=2) == 2
    assert cy.video_comment_cap("x", {}) == 150


def test_real_config_caps_school_lessons_at_50():
    settings = cy.load_config(cy.DEFAULT_CONFIG)["settings"]
    assert len(settings["per_video_max_comments"]) == 9
    assert all(cy.video_comment_cap(v, settings) == 50 for v in settings["per_video_max_comments"])


def test_save_state_retries_when_target_is_briefly_locked(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    real_replace = type(path).replace
    calls = {"n": 0}

    def flaky_replace(self, target):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(type(path), "replace", flaky_replace)
    monkeypatch.setattr(cy.time, "sleep", lambda s: None)
    cy.save_state(path, {"a": {"done": True}})
    assert calls["n"] == 3
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"done": True}}


def test_dryrun_paths_do_not_touch_real_outputs():
    assert cy.dryrun_path(cy.COMMENTS_CSV, 2).name == "youtube_comments.dryrun.csv"
    assert cy.dryrun_path(cy.COMMENTS_CSV, None) == cy.COMMENTS_CSV
