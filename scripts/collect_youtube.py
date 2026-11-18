"""Collect Hindi and Bengali YouTube comments with the YouTube Data API v3.

Two modes, run in this order:

  search  Run the Hindi and Bengali queries in configs/youtube_search.yaml, look up each video's
          comment count, drop videos below `min_comment_count`, deduplicate, and write
          data/interim/candidate_videos.csv with columns
              video_id, title, channel, published_at, comment_count, query, lang, keep, url
          `query` lists every query that found the video (" | "-separated). Fill `keep` with 1 for the
          videos to use. Re-running `search` keeps any `keep` values already filled in.

  screen  Fill the `keep` column automatically. For every candidate: fetch the description, tags and up to
          20 sample comments; measure the share of Hindi/Bengali comments with a keyword rule; ask Gemini
          (GEMINI_API_KEY) whether the video and its comments are about climate change, which language
          the comments are in and the video's leaning; then decide keep in code (relevance, language,
          max 3 videos per channel, 30-35 videos per language, sceptic videos preferred). Writes the
          decision columns back to candidate_videos.csv (the search language moves to `query_lang`).

  fetch   For videos with keep=1, fetch top-level comments plus the replies the API returns with each
          thread (flagged is_reply=1), up to `max_comments_per_video` rows per video, and write
          data/raw/youtube_comments.csv with columns
              id, video_id, text, like_count, published_at, is_reply, query_lang

Privacy: no author names, channel IDs or profile URLs are stored. Inside the comment text, e-mail
addresses are replaced with "<email>" and @handles with "@user".

Quota (10,000 units/day free): search.list costs 100 units per call, videos.list 1 unit per 50 videos,
commentThreads.list 1 unit per page of up to 100 threads. Both modes save progress as they go. If the
daily quota runs out they stop cleanly (exit code 2); run the same command again after the quota resets
(midnight Pacific time) and it resumes where it stopped.

API keys are read from YOUTUBE_API_KEY and GEMINI_API_KEY (environment or .env at the repo root).
There is no random sampling; Gemini runs at temperature 0 with seed 42.

Usage:
    python scripts/collect_youtube.py search
    python scripts/collect_youtube.py screen
    python scripts/collect_youtube.py fetch
    python scripts/collect_youtube.py search --limit 1   # dry run: 1 query, writes *.dryrun.* files
    python scripts/collect_youtube.py screen --limit 10  # dry run: 10 candidates, writes *.dryrun.* files
    python scripts/collect_youtube.py fetch --limit 2    # dry run: 2 kept videos, 2 comments each
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.preprocess import (  # noqa: E402
    INDIC_CHARS,
    MENTION_RE,
    remove_mentions,
    remove_urls,
    script_counts,
    tag_script,
)

# Romanised Hindi / Bengali marker words for the rule-based language check (whole words, any case).
# A Latin-script comment counts as romanised Hindi/Bengali only if it contains at least one of them.
# Extend these lists freely; "ki" is in both on purpose.
ROMANISED_KEYWORDS = {
    "hi": ["hai", "hain", "nahi", "nahin", "kya", "bhai", "ki", "ka", "ke", "aur", "mein", "ho", "tha",
           "raha", "sab", "yeh", "ye", "kuch"],
    "bn": ["ami", "tumi", "ache", "achhe", "na", "kore", "hobe", "eta", "dada", "khub", "amra", "ki", "keno",
           "holo", "bhalo"],
}

DEFAULT_CONFIG = ROOT / "configs" / "youtube_search.yaml"
CANDIDATES_CSV = ROOT / "data" / "interim" / "candidate_videos.csv"
SEARCH_CACHE = ROOT / "data" / "interim" / "youtube_search_cache.jsonl"
SCREEN_SAMPLES = ROOT / "data" / "interim" / "screening_samples.jsonl"
SCREEN_RESULTS = ROOT / "data" / "interim" / "screening_results.jsonl"
SCREEN_RAW = ROOT / "data" / "interim" / "screening_raw.jsonl"
SCREEN_FAILURES = ROOT / "data" / "interim" / "screening_failures.csv"
CLAUDE_QUEUE_DIR = ROOT / "data" / "interim" / "screening_queue"        # batches waiting for Claude
CLAUDE_DECISIONS_DIR = ROOT / "data" / "interim" / "screening_claude"   # Claude's decision files
COMMENTS_CSV = ROOT / "data" / "raw" / "youtube_comments.csv"
FETCH_STATE = ROOT / "data" / "raw" / "youtube_fetch_state.json"

CANDIDATE_COLUMNS = ["video_id", "title", "channel", "published_at", "comment_count", "query", "lang", "keep", "url"]
COMMENT_COLUMNS = ["id", "video_id", "text", "like_count", "published_at", "is_reply", "query_lang"]
LANG_BLOCKS = {"hindi": "hi", "bengali": "bn"}

QUOTA_REASONS = {"quotaExceeded", "dailyLimitExceeded"}
SKIP_VIDEO_REASONS = {"commentsDisabled", "videoNotFound", "forbidden", "processingFailure"}
EMAIL_RE = re.compile(rf"[\w.+\-{INDIC_CHARS}]+@[\w\-{INDIC_CHARS}]+(?:\.[\w\-{INDIC_CHARS}]+)+")
VIDEO_URL_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
KEEP_TRUE = {"1", "1.0", "y", "yes", "true", "x"}

# Gemini screening schema: field -> type, plus the allowed values of the enum fields.
SCREEN_FIELDS = {
    "video_id": str,
    "climate_relevant": bool,
    "lang": str,
    "leaning": str,
    "comments_discuss_climate": bool,
    "indic_comment_share": float,
    "confidence": str,
    "reason": str,
}
SCREEN_ENUMS = {
    "lang": {"hi", "bn", "mixed", "other"},
    "leaning": {"mainstream", "sceptic", "news", "unrelated"},
    "confidence": {"high", "medium", "low"},
}
SCREEN_DEFAULTS = {
    "screener": "claude",              # "claude" (decision files written by Claude Code) or "gemini"
    "min_rule_indic_share_for_llm": 0.2,
    "claude_batch_size": 20,
    "model": "gemini-3.8-flash",
    "batch_size": 10,
    "sample_comments": 20,
    "min_sample_comments": 5,
    "max_per_channel": 3,
    "target_per_language": [30, 35],
    "min_sceptic_share": 0.15,
    "min_seconds_between_calls": 7,
}
SCEPTIC_QUERY_MARKERS = ("झूठ", "धोखा", "propaganda", "scam", "jhooth", "মিথ্যা", "গুজব", "hoax")
CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}

SCREEN_OUTPUT_COLUMNS = [
    "video_id", "title", "channel", "published_at", "comment_count", "query", "query_lang",
    "keep", "review_flag", "lang", "leaning", "rule_indic_share", "indic_comment_share", "lang_check_disagree",
    "confidence", "reason", "screener", "screener_reason", "climate_relevant", "comments_discuss_climate",
    "sample_comments", "target_lang", "url",
]


class QuotaExceeded(RuntimeError):
    """The daily API quota is used up; progress has been saved."""


class ScreeningError(ValueError):
    """A Gemini screening response that is not valid JSON or does not match the schema."""


class GeminiQuotaExhausted(RuntimeError):
    """Gemini keeps returning 429 (e.g. the free-tier daily limit); progress has been saved."""


class GeminiFatal(RuntimeError):
    """An error no retry can fix (unknown/retired model, bad API key, no permission); the run stops."""


# --------------------------------------------------------------------------- pure helpers (tested)


def anonymise_text(text: str) -> str:
    """Replace e-mail addresses with "<email>" and @handles (e.g. the "@name" that starts every reply)
    with "@user". E-mails go first so "a@b.com" is not half-matched as a handle."""
    text = EMAIL_RE.sub("<email>", text or "")
    return MENTION_RE.sub("@user", text)


def parse_keep(value) -> int:
    """Interpret a hand-filled `keep` cell: 1 / yes / x / true -> 1, anything else -> 0."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0
    return int(str(value).strip().lower() in KEEP_TRUE)


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def video_id_from_url(url) -> str | None:
    m = VIDEO_URL_RE.search(str(url or ""))
    return m.group(1) if m else None


def build_search_jobs(config: dict) -> list[dict]:
    """One job per (language, region, query), in config order."""
    settings = config.get("settings", {})
    jobs = []
    for block, lang in LANG_BLOCKS.items():
        spec = config.get(block) or {}
        regions = spec.get("region_codes") or [settings.get("region_code", "IN")]
        for query in spec.get("queries", []):
            for region in regions:
                jobs.append({
                    "lang": lang,
                    "region": region,
                    "query": query,
                    "relevance_language": spec.get("relevance_language", lang),
                })
    return jobs


def job_key(job: dict) -> str:
    return f"{job['lang']}|{job['region']}|{job['query']}"


def parse_search_response(resp: dict, job: dict) -> list[dict]:
    """search.list response -> one hit per video."""
    hits = []
    for item in resp.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        if not vid:  # channels / playlists, if any slip through
            continue
        sn = item.get("snippet") or {}
        hits.append({
            "video_id": vid,
            "title": sn.get("title", ""),
            "channel": sn.get("channelTitle", ""),
            "published_at": sn.get("publishedAt", ""),
            "query": job["query"],
            "lang": job["lang"],
        })
    return hits


def parse_videos_response(resp: dict) -> dict[str, dict]:
    """videos.list (part=snippet,statistics) -> {video_id: {...}}.

    comment_count is None when the video has comments disabled (the API omits commentCount).
    """
    out = {}
    for item in resp.get("items", []):
        stats = item.get("statistics") or {}
        sn = item.get("snippet") or {}
        cc = stats.get("commentCount")
        out[item["id"]] = {
            "comment_count": int(cc) if cc is not None else None,
            "title": sn.get("title"),
            "channel": sn.get("channelTitle"),
            "published_at": sn.get("publishedAt"),
        }
    return out


def merge_candidates(hits: list[dict], stats: dict[str, dict], min_comment_count: int) -> tuple[pd.DataFrame, dict]:
    """Deduplicate search hits by video, attach comment counts and apply the comment-count filter.

    Returns the candidate table and a small report of what was dropped.
    """
    by_video: dict[str, dict] = {}
    for h in hits:
        row = by_video.get(h["video_id"])
        if row is None:
            by_video[h["video_id"]] = {**h, "query": [h["query"]], "lang": [h["lang"]]}
        else:
            if h["query"] not in row["query"]:
                row["query"].append(h["query"])
            if h["lang"] not in row["lang"]:
                row["lang"].append(h["lang"])

    rows, report = [], {"hits": len(hits), "unique_videos": len(by_video), "no_stats": 0,
                        "comments_disabled": 0, "below_min_comments": 0}
    for vid, row in by_video.items():
        st = stats.get(vid)
        if st is None:
            report["no_stats"] += 1
            continue
        if st["comment_count"] is None:
            report["comments_disabled"] += 1
            continue
        if st["comment_count"] < min_comment_count:
            report["below_min_comments"] += 1
            continue
        rows.append({
            "video_id": vid,
            # prefer the full title from videos.list; search snippets can be HTML-escaped
            "title": st.get("title") or row["title"],
            "channel": st.get("channel") or row["channel"],
            "published_at": st.get("published_at") or row["published_at"],
            "comment_count": st["comment_count"],
            "query": " | ".join(row["query"]),
            "lang": "|".join(row["lang"]),
            "keep": "",
            "url": video_url(vid),
        })
    df = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)
    df = df.sort_values(["lang", "comment_count"], ascending=[True, False], kind="stable").reset_index(drop=True)
    report["kept"] = len(df)
    return df, report


def carry_over_keep(new: pd.DataFrame, old: pd.DataFrame | None) -> pd.DataFrame:
    """Copy hand-filled `keep` values from an earlier candidates file into a freshly built one."""
    if old is None or old.empty or "keep" not in old.columns:
        return new
    old_keep = dict(zip(old["video_id"], old["keep"]))
    new = new.copy()
    new["keep"] = [old_keep.get(v, k) if str(old_keep.get(v, "")).strip() not in ("", "nan") else k
                   for v, k in zip(new["video_id"], new["keep"])]
    return new


def load_candidates(path: Path) -> pd.DataFrame:
    """Read the candidates CSV, repairing video ids that a spreadsheet may have mangled.

    Excel can turn ids starting with '-' or '=' into formulas; the `url` column keeps a safe copy.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if "url" in df.columns:
        from_url = df["url"].map(video_id_from_url)
        df["video_id"] = from_url.where(from_url.notna(), df["video_id"])
    return df


def parse_comment_threads(resp: dict, video_id: str, query_lang: str) -> list[dict]:
    """commentThreads.list (part=snippet,replies) -> comment rows, top-level first then its replies.

    Only the fields in COMMENT_COLUMNS are kept: author names, channel IDs and profile images are dropped.
    """
    rows = []
    for thread in resp.get("items", []):
        top = ((thread.get("snippet") or {}).get("topLevelComment")) or {}
        rows.append(_comment_row(top, video_id, query_lang, is_reply=0))
        for reply in ((thread.get("replies") or {}).get("comments")) or []:
            rows.append(_comment_row(reply, video_id, query_lang, is_reply=1))
    return [r for r in rows if r["id"]]


def _comment_row(comment: dict, video_id: str, query_lang: str, is_reply: int) -> dict:
    sn = comment.get("snippet") or {}
    text = sn.get("textOriginal")
    if text is None:
        text = sn.get("textDisplay", "")
    return {
        "id": comment.get("id", ""),
        "video_id": video_id,
        "text": anonymise_text(text),
        "like_count": int(sn.get("likeCount", 0) or 0),
        "published_at": sn.get("publishedAt", ""),
        "is_reply": is_reply,
        "query_lang": query_lang,
    }


def error_reason(err) -> str:
    """The first `reason` in a googleapiclient HttpError (e.g. 'quotaExceeded'), or ''."""
    try:
        payload = json.loads(err.content.decode("utf-8"))
        errors = payload.get("error", {}).get("errors") or []
        if errors:
            return errors[0].get("reason", "")
        for d in payload.get("error", {}).get("details", []) or []:
            if d.get("reason"):
                return d["reason"]
    except Exception:  # noqa: BLE001 - malformed body, fall through
        pass
    return ""


# --------------------------------------------------------------------------- screening helpers (tested)

_KEYWORD_RES = {
    lang: re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
    for lang, words in ROMANISED_KEYWORDS.items()
}


def comment_language(text: str) -> str:
    """Rule-based language of one comment: 'hi', 'bn', 'indic' (romanised, Hindi/Bengali unclear) or 'other'.

    Devanagari counts as Hindi and Bengali script as Bengali (by `tag_script`). Code-mixed text with any
    Devanagari/Bengali letters counts as the Indic script with more letters, even when Latin letters
    dominate ("climate change बहुत serious है"). Pure Latin-script text counts as romanised
    Hindi/Bengali only if it contains a marker word from ROMANISED_KEYWORDS; whichever list matches more
    words decides, a tie gives 'indic'.
    """
    t = remove_mentions(remove_urls(text or ""))
    script = tag_script(t)
    if script == "devanagari":
        return "hi"
    if script == "bengali":
        return "bn"
    counts = script_counts(t)
    if counts["devanagari"] or counts["bengali"]:
        return "hi" if counts["devanagari"] >= counts["bengali"] else "bn"
    if script in ("latin", "mixed"):
        n_hi = len(_KEYWORD_RES["hi"].findall(t))
        n_bn = len(_KEYWORD_RES["bn"].findall(t))
        if n_hi or n_bn:
            return "hi" if n_hi > n_bn else "bn" if n_bn > n_hi else "indic"
    return "other"


def rule_language_stats(comments: list[str]) -> dict:
    """Share of comments in Hindi/Bengali (any script) and which of the two dominates."""
    langs = [comment_language(c) for c in comments]
    n = len(langs)
    n_hi, n_bn = langs.count("hi"), langs.count("bn")
    return {
        "rule_indic_share": round(sum(lang != "other" for lang in langs) / n, 3) if n else 0.0,
        "rule_lang": "hi" if n_hi > n_bn else "bn" if n_bn > n_hi else "",
    }


def parse_json_text(text: str):
    """json.loads that tolerates ```json fences and a single-key wrapper object around the list."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError as err:
        raise ScreeningError(f"malformed JSON: {err}") from err
    if isinstance(data, dict) and len(data) == 1 and isinstance(next(iter(data.values())), list):
        data = next(iter(data.values()))
    return data


def validate_screening(data, expected_ids: list[str]) -> dict[str, dict]:
    """Check a parsed Gemini response against the screening schema.

    Every expected video must appear exactly once with all fields of the right type and allowed value.
    Returns {video_id: record}; raises ScreeningError listing every problem otherwise.
    """
    errors: list[str] = []
    if not isinstance(data, list):
        raise ScreeningError(f"expected a JSON list, got {type(data).__name__}")
    out: dict[str, dict] = {}
    for i, item in enumerate(data):
        where = f"item {i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: not an object")
            continue
        vid = item.get("video_id")
        if isinstance(vid, str):
            where = f"{where} ({vid})"
        missing = [k for k in SCREEN_FIELDS if k not in item]
        if missing:
            errors.append(f"{where}: missing {missing}")
            continue
        rec, ok = {}, True
        for key, typ in SCREEN_FIELDS.items():
            val = item[key]
            if typ is bool:
                good = isinstance(val, bool)
            elif typ is float:
                good = isinstance(val, (int, float)) and not isinstance(val, bool) and 0.0 <= float(val) <= 1.0
                val = float(val) if good else val
            else:
                good = isinstance(val, str) and (key == "reason" or val != "")
            if good and key in SCREEN_ENUMS and val not in SCREEN_ENUMS[key]:
                good = False
            if not good:
                errors.append(f"{where}: bad {key}={val!r}")
                ok = False
            rec[key] = val
        if not ok:
            continue
        if rec["video_id"] in out:
            errors.append(f"{where}: duplicate video_id")
        elif rec["video_id"] not in expected_ids:
            errors.append(f"{where}: unexpected video_id")
        else:
            out[rec["video_id"]] = rec
    missing_ids = [v for v in expected_ids if v not in out]
    if missing_ids and not errors:
        errors.append(f"missing video_ids {missing_ids}")
    elif missing_ids:
        errors.append(f"{len(missing_ids)} expected video_ids without a valid record")
    if errors:
        raise ScreeningError("; ".join(errors[:12]) + (f" (+{len(errors) - 12} more)" if len(errors) > 12 else ""))
    return out


def is_sceptic_query(query: str) -> bool:
    return any(m in str(query) for m in SCEPTIC_QUERY_MARKERS)


def decide_keep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply the keep rules in order and return df with keep / reason / review_flag / lang_check_disagree /
    target_lang filled in.

    Expects columns: video_id, channel, comment_count, query_lang, sample_comments, rule_indic_share,
    rule_lang, screened (bool), and when screened the screener fields (climate_relevant,
    comments_discuss_climate, lang, leaning, indic_comment_share, confidence).

    comments_discuss_climate is recorded but not required: off-topic comments under a climate video are
    fine (they become None labels). Videos whose rule-based Indic share is below
    `min_rule_indic_share_for_llm` are dropped as rule_no_indic without being screened.
    """
    df = df.copy()
    hi = cfg["target_per_language"][1]
    min_n = cfg["min_sample_comments"]
    min_rule = float(cfg["min_rule_indic_share_for_llm"])

    def first_reason(r) -> str:
        if r["sample_comments"] < min_n:
            return "too_few_comments"
        if r["rule_indic_share"] < min_rule:
            return "rule_no_indic"
        if not r["screened"]:
            return "screening_failed"
        if not r["climate_relevant"]:
            return "not_climate_relevant"
        if r["lang"] not in ("hi", "bn", "mixed"):
            return "lang_other"
        if not (r["rule_indic_share"] >= 0.5 or r["indic_comment_share"] >= 0.5):
            return "low_indic_share"
        if r["leaning"] == "unrelated":
            return "leaning_unrelated"
        return ""

    df["reason"] = df.apply(first_reason, axis=1)
    screened = df["screened"].astype(bool)
    disagree = (df["rule_indic_share"] >= 0.5) != (pd.to_numeric(df["indic_comment_share"], errors="coerce") >= 0.5)
    df["lang_check_disagree"] = (screened & disagree).astype(int)

    def target_lang(r) -> str:
        if r.get("lang") in ("hi", "bn"):
            return r["lang"]
        if r.get("rule_lang"):
            return r["rule_lang"]
        return str(r.get("query_lang", "")).split("|")[0]

    df["target_lang"] = df.apply(target_lang, axis=1)

    # max N kept videos per channel, preferring more comments
    eligible = df["reason"] == ""
    order = df[eligible].sort_values(["channel", "comment_count"], ascending=[True, False], kind="stable")
    over_cap = order.groupby("channel").cumcount() >= cfg["max_per_channel"]
    df.loc[over_cap[over_cap].index, "reason"] = "channel_cap"

    # at most `hi` per language: sceptic first, then confidence, then comment count
    eligible = df["reason"] == ""
    ranked = df[eligible].assign(
        _sceptic=lambda d: (d["leaning"] != "sceptic").astype(int),
        _conf=lambda d: d["confidence"].map(CONFIDENCE_RANK).fillna(3),
    ).sort_values(["target_lang", "_sceptic", "_conf", "comment_count"], ascending=[True, True, True, False],
                  kind="stable")
    over_target = ranked.groupby("target_lang").cumcount() >= hi
    df.loc[over_target[over_target].index, "reason"] = "over_target"

    df["keep"] = (df["reason"] == "").astype(int)
    df.loc[df["keep"] == 1, "reason"] = "kept"
    df["review_flag"] = ((df["keep"] == 1) & ((df["confidence"] == "low") | (df["lang_check_disagree"] == 1))
                         ).astype(int)
    return df


# --------------------------------------------------------------------------- API plumbing


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(name: str = "YOUTUBE_API_KEY") -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get(name, "").strip()
    if not key:
        sys.exit(f"{name} is not set (add it to .env at the repo root).")
    return key


def build_client(api_key: str):
    from googleapiclient.discovery import build

    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def execute(request):
    """Run a request with retries on 5xx / rate limits; raise QuotaExceeded when the daily quota is gone."""
    from googleapiclient.errors import HttpError

    try:
        return request.execute(num_retries=3)
    except HttpError as err:
        if error_reason(err) in QUOTA_REASONS:
            raise QuotaExceeded(error_reason(err)) from err
        raise


def dryrun_path(path: Path, limit: int | None) -> Path:
    return path.with_name(f"{path.stem}.dryrun{path.suffix}") if limit else path


# --------------------------------------------------------------------------- search mode


def run_search(config: dict, limit: int | None) -> int:
    settings = config.get("settings", {})
    jobs = build_search_jobs(config)
    if limit:
        jobs = jobs[:limit]
    out_csv = dryrun_path(CANDIDATES_CSV, limit)
    cache_path = dryrun_path(SEARCH_CACHE, limit)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cache = {rec["key"]: rec["hits"] for rec in read_jsonl(cache_path)}
    todo = [j for j in jobs if job_key(j) not in cache]
    print(f"search: {len(jobs)} query jobs ({len(jobs) - len(todo)} already cached, {len(todo)} to run, "
          f"~{len(todo) * 100} quota units)")

    youtube = build_client(get_api_key())
    quota_hit = False
    try:
        with open(cache_path, "a", encoding="utf-8") as cache_f:
            for i, job in enumerate(todo, 1):
                req = youtube.search().list(
                    part="snippet",
                    q=job["query"],
                    type="video",
                    maxResults=int(settings.get("max_results_per_query", 15)),
                    regionCode=job["region"],
                    relevanceLanguage=job["relevance_language"],
                    publishedAfter=settings.get("published_after"),
                    order="relevance",
                )
                hits = parse_search_response(execute(req), job)
                cache[job_key(job)] = hits
                cache_f.write(json.dumps({"key": job_key(job), "hits": hits}, ensure_ascii=False) + "\n")
                cache_f.flush()
                print(f"  [{i}/{len(todo)}] {job['lang']} {job['region']} {job['query']!r}: {len(hits)} videos")
    except QuotaExceeded as err:
        quota_hit = True
        print(f"\nDaily quota exhausted ({err}). {len(cache)} of {len(jobs)} queries are cached in "
              f"{cache_path.relative_to(ROOT)}; run the same command after the quota resets to finish.")

    hits = [h for j in jobs for h in cache.get(job_key(j), [])]
    video_ids = list(dict.fromkeys(h["video_id"] for h in hits))
    stats: dict[str, dict] = {}
    try:
        for start in range(0, len(video_ids), 50):
            batch = video_ids[start:start + 50]
            stats.update(parse_videos_response(execute(
                youtube.videos().list(part="snippet,statistics", id=",".join(batch), maxResults=50))))
    except QuotaExceeded as err:
        print(f"\nDaily quota exhausted while reading video statistics ({err}); run again after the reset.")
        return 2

    df, report = merge_candidates(hits, stats, int(settings.get("min_comment_count", 0)))
    old = load_candidates(out_csv) if out_csv.exists() else None
    df = carry_over_keep(df, old)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")  # BOM so Excel shows Devanagari/Bengali

    print(f"\n{report['hits']} search hits -> {report['unique_videos']} unique videos; dropped "
          f"{report['comments_disabled']} with comments disabled, {report['below_min_comments']} with "
          f"< {settings.get('min_comment_count', 0)} comments, {report['no_stats']} unavailable.")
    print(f"Wrote {len(df)} candidates to {out_csv.relative_to(ROOT)} "
          f"({df['lang'].value_counts().to_dict() if len(df) else {}}).")
    if old is not None:
        print(f"Carried over keep values from the previous file ({int(df['keep'].map(parse_keep).sum())} kept).")
    print("Next: set keep=1 on the videos to use (save as CSV UTF-8), then run `fetch`.")
    return 2 if quota_hit else 0


# --------------------------------------------------------------------------- screen mode

SCREEN_SYSTEM_PROMPT = """You screen YouTube videos for a research dataset of Hindi and Bengali YouTube comments about the
claim "Climate change and global warming is a serious concern". For each video you get its title, channel,
tags, the start of its description and a sample of its top comments. Judge each video independently.

Return ONLY a JSON list with exactly one object per input video, in any order, each with exactly these keys:
{"video_id": str, "climate_relevant": bool, "lang": "hi"|"bn"|"mixed"|"other",
 "leaning": "mainstream"|"sceptic"|"news"|"unrelated", "comments_discuss_climate": bool,
 "indic_comment_share": float between 0 and 1, "confidence": "high"|"medium"|"low", "reason": str}

Definitions:
- video_id: copy it exactly from the input.
- climate_relevant: true if the video is about climate change or global warming, or about a climate-linked
  event (heatwave, flood, cyclone, drought, glacier melt, sea-level rise, pollution) discussed in that context.
  False for weather forecasts only, unrelated topics, music, ads, or videos that merely contain the words.
- lang: main language of the COMMENTS (not the video). "hi" = mostly Hindi in any script (Devanagari,
  romanised, or Hindi-English code-mixed); "bn" = mostly Bengali in any script; "mixed" = substantial Hindi
  AND Bengali; "other" = mostly English or another language.
- leaning: "sceptic" if the video questions, downplays, mocks or denies climate change (hoax, scam,
  propaganda, natural cycle), or most sample comments do; "mainstream" if it treats climate change as real
  and serious (explainers, documentaries, activism); "news" if it reports a climate-related event or policy
  without a clear stance; "unrelated" if it is not about climate change at all.
- comments_discuss_climate: true if at least 3 sample comments talk about or take a position on climate
  change, global warming, heat, pollution or the video's climate topic (not just greetings, praise or spam).
- indic_comment_share: fraction of the sample comments written in Hindi or Bengali (any script, including
  romanised and code-mixed). Comments in English or other languages, and emoji-only comments, do not count.
- confidence: how sure you are about the whole judgement.
- reason: one short line explaining the judgement.
Do not add any text outside the JSON list."""


def build_screen_prompt(videos: list[dict]) -> str:
    payload = [
        {
            "video_id": v["video_id"],
            "title": v["title"],
            "channel": v["channel"],
            "tags": v.get("tags", [])[:15],
            "description": v.get("description", "")[:500],
            "sample_comments": [c[:300] for c in v.get("comments", [])],
        }
        for v in videos
    ]
    return ("Screen these videos. Return one JSON object per video_id: "
            + ", ".join(v["video_id"] for v in videos) + "\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=1))


def read_jsonl(path: Path) -> list[dict]:
    # split on "\n" only: str.splitlines() also splits on U+2028/U+0085 etc., which can occur
    # (unescaped, since ensure_ascii=False) inside comment text and would cut a record in two
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class GeminiScreener:
    """Calls Gemini with rate limiting, 429/5xx retries, schema validation and a raw-response log."""

    def __init__(self, api_key: str, model: str, min_interval: float, raw_path: Path):
        from google import genai
        from google.genai import types

        # 3-minute timeout per request so a stuck call cannot hang the run
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=180_000))
        self.types = types
        self.model = model
        self.min_interval = min_interval
        self.raw_path = raw_path
        self._last_call = 0.0
        self.calls = 0

    def _generate(self, prompt: str, video_ids: list[str], attempt: int) -> str:
        from google.genai import errors

        config = self.types.GenerateContentConfig(
            system_instruction=SCREEN_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0,
            seed=42,
        )
        rate_waits = 0
        while True:
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            self.calls += 1
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                text = resp.text or ""
                append_jsonl(self.raw_path, {"time": now_iso(), "model": self.model, "video_ids": video_ids,
                                             "attempt": attempt, "text": text})
                return text
            except errors.APIError as err:
                append_jsonl(self.raw_path, {"time": now_iso(), "model": self.model, "video_ids": video_ids,
                                             "attempt": attempt, "error": f"{err.code}: {str(err)[:3000]}"})
                if err.code in (401, 403, 404) or "API_KEY" in str(err):
                    raise GeminiFatal(f"{err.code}: {str(err)[:300]}") from err
                if err.code == 429 or (err.code and err.code >= 500):
                    rate_waits += 1
                    msg = str(err)
                    if err.code == 429 and ("PerDay" in msg or rate_waits > 6):
                        raise GeminiQuotaExhausted(msg[:300]) from err
                    if rate_waits > 6:
                        raise ScreeningError(f"Gemini server error after retries: {msg[:200]}") from err
                    m = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s", msg)
                    delay = float(m.group(1)) + 1 if m else min(10 * 2 ** rate_waits, 120)
                    print(f"    Gemini {err.code}; waiting {delay:.0f}s")
                    time.sleep(delay)
                    continue
                raise ScreeningError(f"Gemini API error {err.code}: {str(err)[:200]}") from err

    def screen(self, videos: list[dict], max_attempts: int = 3) -> dict[str, dict]:
        """Screen a batch; retries malformed/invalid responses with backoff, then raises ScreeningError."""
        ids = [v["video_id"] for v in videos]
        prompt = build_screen_prompt(videos)
        last = None
        for attempt in range(1, max_attempts + 1):
            try:
                return validate_screening(parse_json_text(self._generate(prompt, ids, attempt)), ids)
            except ScreeningError as err:
                last = err
                print(f"    attempt {attempt}/{max_attempts} for {len(ids)} video(s) failed: {str(err)[:160]}")
                if attempt < max_attempts:
                    time.sleep(5 * 2 ** (attempt - 1))
        raise ScreeningError(str(last))


def fetch_samples(youtube, video_ids: list[str], n_comments: int, cache_path: Path) -> dict[str, dict]:
    """Description, tags and up to n sample comments per video, cached in a JSONL file."""
    from googleapiclient.errors import HttpError

    cache = {r["video_id"]: r for r in read_jsonl(cache_path)}
    todo = [v for v in video_ids if v not in cache]
    if todo:
        print(f"  fetching metadata + {n_comments} sample comments for {len(todo)} videos "
              f"(~{len(todo) + -(-len(todo) // 50)} YouTube quota units)")
    meta: dict[str, dict] = {}
    for start in range(0, len(todo), 50):
        resp = execute(youtube.videos().list(part="snippet", id=",".join(todo[start:start + 50]), maxResults=50))
        for item in resp.get("items", []):
            sn = item.get("snippet") or {}
            meta[item["id"]] = {"description": (sn.get("description") or "")[:500], "tags": sn.get("tags") or []}
    for i, vid in enumerate(todo, 1):
        rec = {"video_id": vid, **meta.get(vid, {"description": "", "tags": []}), "comments": [], "error": ""}
        try:
            resp = execute(youtube.commentThreads().list(
                part="snippet", videoId=vid, maxResults=n_comments, order="relevance", textFormat="plainText"))
            rec["comments"] = [r["text"] for r in parse_comment_threads(resp, vid, "") if r["text"].strip()]
        except HttpError as err:
            reason = error_reason(err)
            if reason not in SKIP_VIDEO_REASONS:
                raise
            rec["error"] = reason
        append_jsonl(cache_path, rec)
        cache[vid] = rec
        if i % 50 == 0:
            print(f"    {i}/{len(todo)} videos sampled")
    return cache


def interleave_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Round-robin rows across the values of `col` (so a --limit dry run sees every language)."""
    order = df.groupby(col, sort=False).cumcount()
    return df.assign(_o=order).sort_values(["_o", col], kind="stable").drop(columns="_o")


def run_screen(config: dict, limit: int | None) -> int:
    cfg = {**SCREEN_DEFAULTS, **(config.get("screening") or {})}
    if not CANDIDATES_CSV.exists():
        sys.exit(f"{CANDIDATES_CSV.relative_to(ROOT)} not found; run `search` first.")
    cands = load_candidates(CANDIDATES_CSV)
    if "query_lang" not in cands.columns:
        cands = cands.rename(columns={"lang": "query_lang"})
    cands = cands[[c for c in cands.columns if c in ("video_id", "title", "channel", "published_at",
                                                     "comment_count", "query", "query_lang", "url")]]
    cands["comment_count"] = pd.to_numeric(cands["comment_count"], errors="coerce").fillna(0).astype(int)
    if limit:
        cands = interleave_by(cands, "query_lang").head(limit)
    out_csv = dryrun_path(CANDIDATES_CSV, limit)
    samples_path, results_path = dryrun_path(SCREEN_SAMPLES, limit), dryrun_path(SCREEN_RESULTS, limit)
    raw_path, failures_path = dryrun_path(SCREEN_RAW, limit), dryrun_path(SCREEN_FAILURES, limit)
    who = (f"Claude Code (batches of {cfg['claude_batch_size']})" if cfg["screener"] == "claude"
           else f"Gemini {cfg['model']} (batches of {cfg['batch_size']})")
    print(f"screen: {len(cands)} candidates, screener {who}")

    # 1. metadata + sample comments (YouTube)
    youtube = build_client(get_api_key())
    try:
        samples = fetch_samples(youtube, list(cands["video_id"]), int(cfg["sample_comments"]), samples_path)
    except QuotaExceeded as err:
        print(f"\nYouTube quota exhausted ({err}); samples so far are cached. Run again after the reset.")
        return 2

    # 2. rule-based language check
    rows = []
    for r in cands.itertuples(index=False):
        s = samples[r.video_id]
        rows.append({**r._asdict(), "sample_comments": len(s["comments"]), **rule_language_stats(s["comments"])})
    df = pd.DataFrame(rows)

    # 3. screening (Gemini or Claude) for videos with enough comments that pass the rule pre-filter;
    #    only cached results from the configured screener are used
    screener_name = cfg["screener"]
    min_rule = float(cfg["min_rule_indic_share_for_llm"])
    results = {r["video_id"]: r["result"] for r in read_jsonl(results_path)
               if r.get("screener", "gemini") == screener_name}
    enough = df["sample_comments"] >= cfg["min_sample_comments"]
    needs = df.loc[enough & (df["rule_indic_share"] >= min_rule), "video_id"].tolist()
    print(f"  {int((enough & (df['rule_indic_share'] < min_rule)).sum())} videos with rule_indic_share < {min_rule} "
          f"-> rule_no_indic (no LLM call); {len(needs)} need screening by {screener_name}")
    to_screen = [v for v in needs if v not in results]
    failures: dict[str, str] = {}
    quota_hit = False
    info = df.set_index("video_id")

    def store(res: dict[str, dict]) -> None:
        for vid, rec in res.items():
            results[vid] = rec
            append_jsonl(results_path, {"video_id": vid, "screener": screener_name, "result": rec, "time": now_iso()})

    if screener_name == "claude":
        decisions_dir = dryrun_path(CLAUDE_DECISIONS_DIR, limit)
        queue_dir = dryrun_path(CLAUDE_QUEUE_DIR, limit)
        failures.update(import_claude_decisions(decisions_dir, set(to_screen), store, raw_path))
        to_screen = [v for v in needs if v not in results]
        write_claude_queue(queue_dir, [claude_payload(v, info, samples) for v in to_screen],
                           int(cfg["claude_batch_size"]))
        if to_screen:
            print(f"  {len(to_screen)} videos await Claude screening: batches in {queue_dir.relative_to(ROOT)}; "
                  f"write decisions to {decisions_dir.relative_to(ROOT)}/<same name>.json and rerun")
            for vid in to_screen:
                failures.setdefault(vid, "awaiting claude screening")
        to_screen = []
    if to_screen:
        n_batches = -(-len(to_screen) // cfg["batch_size"])
        print(f"  Gemini: {len(to_screen)} videos in {n_batches} batches ({len(results)} already screened)")
        screener = GeminiScreener(get_api_key("GEMINI_API_KEY"), cfg["model"],
                                  float(cfg["min_seconds_between_calls"]), raw_path)

        def video_payload(vid):
            return {"video_id": vid, "title": info.at[vid, "title"], "channel": info.at[vid, "channel"],
                    "tags": samples[vid]["tags"], "description": samples[vid]["description"],
                    "comments": samples[vid]["comments"]}

        try:
            for b, start in enumerate(range(0, len(to_screen), cfg["batch_size"]), 1):
                batch = to_screen[start:start + cfg["batch_size"]]
                try:
                    store(screener.screen([video_payload(v) for v in batch]))
                    print(f"  [{b}/{n_batches}] screened {len(batch)} videos")
                except ScreeningError as err:
                    print(f"  [{b}/{n_batches}] batch failed ({str(err)[:120]}); retrying one video at a time")
                    for vid in batch:
                        try:
                            store(screener.screen([video_payload(vid)]))
                        except ScreeningError as err1:
                            failures[vid] = str(err1)[:500]
                            print(f"    {vid}: screening failed")
        except GeminiQuotaExhausted as err:
            quota_hit = True
            print(f"\nGemini rate/daily limit reached ({str(err)[:160]}). Results so far are cached in "
                  f"{results_path.relative_to(ROOT)}; run the same command later to finish.")
        except GeminiFatal as err:
            print(f"\nStopping: Gemini error that retrying cannot fix -> {err}\n"
                  f"Check `screening.model` in the config and GEMINI_API_KEY. Nothing was written.")
            return 1
        print(f"  Gemini calls this run: {screener.calls}")

    unscreened = [v for v in needs if v not in results and v not in failures]
    for vid in unscreened:  # only after a quota stop
        failures[vid] = "not screened yet (Gemini limit reached; rerun to finish)"
    pd.DataFrame([{"video_id": v, "title": info.at[v, "title"], "error": e, "time": now_iso()}
                  for v, e in failures.items()], columns=["video_id", "title", "error", "time"]
                 ).to_csv(failures_path, index=False, encoding="utf-8-sig")

    # 4. keep decision (only results for videos that still need screening are used)
    results = {v: rec for v, rec in results.items() if v in set(needs)}
    for key in ("climate_relevant", "comments_discuss_climate", "lang", "leaning", "indic_comment_share",
                "confidence", "reason"):
        df[key if key != "reason" else "screener_reason"] = df["video_id"].map(
            lambda v, k=key: results.get(v, {}).get(k))
    df["screened"] = df["video_id"].isin(list(results))
    df["screener"] = df["screened"].map({True: screener_name, False: ""})
    df["indic_comment_share"] = pd.to_numeric(df["indic_comment_share"], errors="coerce")
    df = decide_keep(df, cfg)

    # 5. write back
    for col in SCREEN_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df[SCREEN_OUTPUT_COLUMNS].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nWrote {out_csv.relative_to(ROOT)}")
    print_screen_summary(df, cfg, failures_path)
    awaiting = sum(e == "awaiting claude screening" for e in failures.values())
    return 2 if (quota_hit or awaiting) else 0


def claude_payload(vid: str, info: pd.DataFrame, samples: dict) -> dict:
    """What Claude reads to screen one video (same content as the Gemini prompt, comments trimmed)."""
    s = samples[vid]
    return {"video_id": vid, "title": info.at[vid, "title"], "channel": info.at[vid, "channel"],
            "tags": s["tags"][:10], "description": s["description"][:300],
            "rule_indic_share": float(info.at[vid, "rule_indic_share"]),
            "comments": [c[:200] for c in s["comments"]]}


def write_claude_queue(queue_dir: Path, payloads: list[dict], batch_size: int) -> None:
    """(Re)write the batches still waiting for Claude as batch_XX.json files."""
    queue_dir.mkdir(parents=True, exist_ok=True)
    for old in queue_dir.glob("batch_*.json"):
        old.unlink()
    for b, start in enumerate(range(0, len(payloads), batch_size), 1):
        chunk = payloads[start:start + batch_size]
        (queue_dir / f"batch_{b:02d}.json").write_text(
            json.dumps({"video_ids": [p["video_id"] for p in chunk], "videos": chunk}, ensure_ascii=False, indent=1),
            encoding="utf-8")


def import_claude_decisions(decisions_dir: Path, wanted: set[str], store, raw_path: Path) -> dict[str, str]:
    """Validate Claude's decision files and store them like Gemini results.

    Each file is {"video_ids": [...], "results": [<one schema object per video>]}. A file is validated
    as a whole with validate_screening (same schema and checks as Gemini); a file that fails is logged
    and its videos are returned as failures. Files whose videos are all stored already are skipped.
    """
    failures: dict[str, str] = {}
    if not decisions_dir.exists():
        return failures
    for path in sorted(decisions_dir.glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        data = None
        try:
            data = parse_json_text(raw)
            if not isinstance(data, dict) or "video_ids" not in data or "results" not in data:
                raise ScreeningError("file must be an object with video_ids and results")
            ids = list(data["video_ids"])
            if not wanted.intersection(ids):
                continue
            res = validate_screening(data["results"], ids)
        except ScreeningError as err:
            append_jsonl(raw_path, {"time": now_iso(), "screener": "claude", "file": path.name, "error": str(err)})
            ids = data.get("video_ids", []) if isinstance(data, dict) else []
            for vid in ids:
                failures[vid] = f"invalid claude decision file {path.name}: {err}"[:500]
            print(f"  {path.name}: INVALID ({str(err)[:160]})")
            continue
        append_jsonl(raw_path, {"time": now_iso(), "screener": "claude", "file": path.name, "video_ids": ids,
                                "text": json.dumps(data["results"], ensure_ascii=False)})
        store({v: r for v, r in res.items() if v in wanted})
        print(f"  {path.name}: imported {len(res)} Claude decisions")
    return failures


def print_screen_summary(df: pd.DataFrame, cfg: dict, failures_path: Path) -> None:
    lo, hi = cfg["target_per_language"]
    kept = df[df["keep"] == 1]
    print("\n================ screening summary ================")
    print(f"candidates: {len(df)}   kept: {len(kept)}   flagged for review: {int(kept['review_flag'].sum())}")
    per_lang = kept["target_lang"].value_counts()
    print("\nkept per language:", {k: int(per_lang.get(k, 0)) for k in ("hi", "bn")},
          "(screener lang:", kept["lang"].value_counts().to_dict(), ")")
    print("kept per leaning: ", kept["leaning"].value_counts().to_dict())
    ch = kept["channel"].value_counts()
    print(f"kept per channel:  {len(ch)} channels; channels with 2+ videos: "
          f"{ {k: int(v) for k, v in ch[ch > 1].items()} }")
    print("\ndropped by reason:", df.loc[df["keep"] == 0, "reason"].value_counts().to_dict())
    print(f"lang_check_disagree (all screened): {int(df['lang_check_disagree'].sum())}")

    sq = [is_sceptic_query(q) for q in kept["query"]]
    sl = [lean == "sceptic" for lean in kept["leaning"]]
    n_sq, n_sl, both = sum(sq), sum(sl), sum(a and b for a, b in zip(sq, sl))
    print(f"\nkept videos found by a sceptic-leaning query: {n_sq}; labelled 'sceptic' by the screener: {n_sl} "
          f"(both: {both})")

    print("\ntargets:")
    for lang in ("hi", "bn"):
        n = int(per_lang.get(lang, 0))
        status = "met" if n >= lo else f"MISSED ({n} < {lo}; not enough relevant videos passed)"
        print(f"  {lang}: {n} kept (target {lo}-{hi}) -> {status}")
    share = n_sl / len(kept) if len(kept) else 0.0
    status = "met" if share >= cfg["min_sceptic_share"] else "MISSED"
    print(f"  sceptic share: {share:.1%} (target >= {cfg['min_sceptic_share']:.0%}) -> {status}")
    n_fail = int((df["reason"] == "screening_failed").sum())
    if n_fail:
        print(f"\n{n_fail} videos failed screening; see {failures_path.relative_to(ROOT)}")


# --------------------------------------------------------------------------- fetch mode


def load_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(path)


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    pd.DataFrame(rows, columns=COMMENT_COLUMNS).to_csv(
        path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def run_fetch(config: dict, limit: int | None) -> int:
    from googleapiclient.errors import HttpError

    settings = config.get("settings", {})
    cap = int(settings.get("max_comments_per_video", 150))
    order = settings.get("comment_order", "relevance")
    if limit:
        cap = min(cap, limit)

    cand_path = dryrun_path(CANDIDATES_CSV, limit) if limit and dryrun_path(CANDIDATES_CSV, limit).exists() \
        else CANDIDATES_CSV
    if not cand_path.exists():
        sys.exit(f"{cand_path.relative_to(ROOT)} not found; run `search` first.")
    cands = load_candidates(cand_path)
    cands = cands[cands["keep"].map(parse_keep) == 1].drop_duplicates("video_id")
    if limit:
        cands = cands.head(limit)
    if cands.empty:
        sys.exit(f"No videos with keep=1 in {cand_path.relative_to(ROOT)}.")

    out_csv = dryrun_path(COMMENTS_CSV, limit)
    state_path = dryrun_path(FETCH_STATE, limit)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    todo = [r for r in cands.itertuples(index=False) if not state.get(r.video_id, {}).get("done")]
    print(f"fetch: {len(cands)} kept videos from {cand_path.relative_to(ROOT)} "
          f"({len(cands) - len(todo)} already done), up to {cap} comments each, order={order}")

    youtube = build_client(get_api_key())
    quota_hit = False
    for i, row in enumerate(todo, 1):
        vid = row.video_id
        lang = getattr(row, "query_lang", None) or row.lang  # `screen` moves the search language to query_lang
        st = state.setdefault(vid, {"done": False, "n": 0, "next_page_token": None})
        try:
            while st["n"] < cap:
                req = youtube.commentThreads().list(
                    part="snippet,replies",
                    videoId=vid,
                    maxResults=min(100, cap - st["n"]),
                    order=order,
                    textFormat="plainText",
                    pageToken=st["next_page_token"] or None,
                )
                resp = execute(req)
                rows = parse_comment_threads(resp, vid, lang)[: cap - st["n"]]
                append_rows(out_csv, rows)
                st["n"] += len(rows)
                st["next_page_token"] = resp.get("nextPageToken")
                save_state(state_path, state)
                if not st["next_page_token"]:
                    break
            st["done"] = True
            st["next_page_token"] = None
            save_state(state_path, state)
            print(f"  [{i}/{len(todo)}] {vid} ({lang}): {st['n']} comments")
        except QuotaExceeded as err:
            quota_hit = True
            print(f"\nDaily quota exhausted ({err}) at video {vid}. Progress saved in "
                  f"{state_path.relative_to(ROOT)}; run the same command after the quota resets to resume.")
            break
        except HttpError as err:
            reason = error_reason(err)
            if reason in SKIP_VIDEO_REASONS:
                st.update(done=True, skipped=reason, next_page_token=None)
                save_state(state_path, state)
                print(f"  [{i}/{len(todo)}] {vid}: skipped ({reason})")
                continue
            save_state(state_path, state)
            raise

    if out_csv.exists():
        df = pd.read_csv(out_csv, dtype={"id": str, "video_id": str}, keep_default_na=False, encoding="utf-8")
        before = len(df)
        df = df.drop_duplicates("id")
        if len(df) < before:  # a page re-fetched after an interrupted run
            df.to_csv(out_csv, index=False, encoding="utf-8")
        n_reply = int((df["is_reply"].astype(int) == 1).sum())
        print(f"\n{out_csv.relative_to(ROOT)}: {len(df)} comments ({len(df) - n_reply} top-level, {n_reply} replies) "
              f"from {df['video_id'].nunique()} videos; by language {df['query_lang'].value_counts().to_dict()}.")
    return 2 if quota_hit else 0


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["search", "screen", "fetch"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None,
                        help="dry run: search runs only the first N queries; screen takes N candidates (mixed "
                             "languages); fetch takes the first N kept videos and at most N comments each. "
                             "Writes *.dryrun.* files instead of the real outputs.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    config = load_config(args.config)
    if args.mode == "search":
        return run_search(config, args.limit)
    if args.mode == "screen":
        return run_screen(config, args.limit)
    return run_fetch(config, args.limit)


if __name__ == "__main__":
    sys.exit(main())
