"""Collect Hindi and Bengali YouTube comments with the YouTube Data API v3.

Two modes, run in this order:

  search  Run the Hindi and Bengali queries in configs/youtube_search.yaml, look up each video's
          comment count, drop videos below `min_comment_count`, deduplicate, and write
          data/interim/candidate_videos.csv with columns
              video_id, title, channel, published_at, comment_count, query, lang, keep, url
          `query` lists every query that found the video (" | "-separated). Fill `keep` with 1 for the
          videos to use. Re-running `search` keeps any `keep` values already filled in.

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

The API key is read from YOUTUBE_API_KEY (environment or .env at the repo root).
The script is deterministic (no random sampling), so it has no seed.

Usage:
    python scripts/collect_youtube.py search
    python scripts/collect_youtube.py fetch
    python scripts/collect_youtube.py search --limit 1   # dry run: 1 query, writes *.dryrun.* files
    python scripts/collect_youtube.py fetch --limit 2    # dry run: 2 kept videos, 2 comments each
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.preprocess import INDIC_CHARS, MENTION_RE  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs" / "youtube_search.yaml"
CANDIDATES_CSV = ROOT / "data" / "interim" / "candidate_videos.csv"
SEARCH_CACHE = ROOT / "data" / "interim" / "youtube_search_cache.jsonl"
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


class QuotaExceeded(RuntimeError):
    """The daily API quota is used up; progress has been saved."""


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


# --------------------------------------------------------------------------- API plumbing


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        sys.exit("YOUTUBE_API_KEY is not set (add it to .env at the repo root).")
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

    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[rec["key"]] = rec["hits"]
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
        vid, lang = row.video_id, row.lang
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
    parser.add_argument("mode", choices=["search", "fetch"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=None,
                        help="dry run: search runs only the first N queries; fetch takes the first N kept videos "
                             "and at most N comments each. Writes *.dryrun.* files instead of the real outputs.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    config = load_config(args.config)
    if args.mode == "search":
        return run_search(config, args.limit)
    return run_fetch(config, args.limit)


if __name__ == "__main__":
    sys.exit(main())
