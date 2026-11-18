"""Clean, filter and deduplicate the collected YouTube comments (T3).

Input:  data/raw/youtube_comments.csv (from scripts/collect_youtube.py fetch)
        data/interim/candidate_videos.csv (video language and leaning, for tie-breaks and sampling)
Output: data/interim/comments_clean.csv      id, video_id, text, lang, script
        data/interim/comments_removed.csv    every removed row with its reason (nothing dropped silently)
        data/interim/comments_to_label.csv   only if more than --sample-size comments remain: all comments
                                             from sceptic videos, then the rest stratified by video with at
                                             most --per-video-cap per video (seed 42)
        results/comment_cleaning.json        filter report and counts (no text)

Steps, in order (each row is removed by the first step that applies):
    1. empty          nothing left after terralens.preprocess.clean_text (only links / @mentions / spaces)
    2. emoji_only     no letters once emojis are removed (emojis, symbols, numbers only)
    3. english        English only (terralens.language.detect_language)
    4. other_language neither Hindi, Bengali nor English
    5. too_short      fewer than 3 word tokens (emojis not counted)
    6. exact_dup      same normalised text as an earlier comment (any video)
    7. near_dup       char 3-5-gram TF-IDF cosine > 0.9 with an earlier kept comment (within or across videos)
"Earlier" means earlier published_at; the first copy is kept.

`text` in the output is clean_text(raw) (emojis as :name: tokens). Language, script, token count and the
dedup key are computed on the same text with emojis removed. A Hindi/Bengali tie in romanised text is
resolved by the video's language.

Usage:
    python scripts/merge_and_dedup.py
    python scripts/merge_and_dedup.py --limit 300     # dry run on 300 comments, writes *.dryrun.* files only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import emoji
import numpy as np
import pandas as pd
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.language import detect_language  # noqa: E402
from terralens.preprocess import clean_text, tag_script  # noqa: E402

SEED = 42
RAW_CSV = ROOT / "data" / "raw" / "youtube_comments.csv"
CANDIDATES_CSV = ROOT / "data" / "interim" / "candidate_videos.csv"
INTERIM = ROOT / "data" / "interim"
RESULTS_JSON = ROOT / "results" / "comment_cleaning.json"

CLEAN_COLUMNS = ["id", "video_id", "text", "lang", "script"]
STEPS = ["empty", "emoji_only", "english", "other_language", "too_short", "exact_dup", "near_dup"]
MIN_TOKENS = 3
NEAR_DUP_THRESHOLD = 0.9
NGRAM_RANGE = (3, 5)

LETTER_RE = re.compile(r"\p{L}")
TOKEN_RE = re.compile(r"[\p{L}\p{M}\p{N}]+")
NON_WORD_RE = re.compile(r"[^\p{L}\p{M}\p{N}]+")


# ------------------------------------------------------------------------------------------- text helpers


def core_text(raw: str) -> str:
    """clean_text with emojis removed: used for language, script, token count and dedup."""
    return clean_text(emoji.replace_emoji(raw or "", " "))


def n_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text or ""))


def dedup_key(core: str) -> str:
    """Lower case, punctuation and symbols removed, spaces collapsed."""
    return " ".join(NON_WORD_RE.sub(" ", (core or "").lower()).split())


# ------------------------------------------------------------------------------------------- pipeline


def load_video_meta(path: Path) -> pd.DataFrame:
    """video_id -> lang (target language) and leaning, for kept videos."""
    if not path.exists():
        return pd.DataFrame(columns=["video_id", "video_lang", "leaning"])
    c = pd.read_csv(path, dtype=str, keep_default_na=False)
    lang_col = "target_lang" if "target_lang" in c.columns else "lang"
    meta = c[["video_id", lang_col, "leaning"]].rename(columns={lang_col: "video_lang"})
    return meta.drop_duplicates("video_id")


def annotate(raw: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Add clean text, core text, language, script and token count to every raw comment."""
    df = raw.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df.merge(meta, on="video_id", how="left")
    df["video_lang"] = df["video_lang"].fillna("")
    df["leaning"] = df["leaning"].fillna("")
    df["clean"] = df["text"].map(clean_text)
    df["core"] = df["text"].map(core_text)
    df["lang"] = [detect_language(t, fb or None) for t, fb in zip(df["core"], df["video_lang"])]
    df["script"] = df["core"].map(tag_script)
    df["n_tokens"] = df["core"].map(n_tokens)
    df["key"] = df["core"].map(dedup_key)
    return df


def rule_reason(row) -> str:
    """Reason for steps 1-5, or '' if the comment passes them."""
    if row["clean"] == "":
        return "empty"
    if not LETTER_RE.search(row["core"]):
        return "emoji_only"
    if row["lang"] == "en":
        return "english"
    if row["lang"] not in ("hi", "bn"):
        return "other_language"
    if row["n_tokens"] < MIN_TOKENS:
        return "too_short"
    return ""


def order_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """Earliest comment first; ties keep the file order."""
    t = pd.to_datetime(df["published_at"], errors="coerce", utc=True) if "published_at" in df else None
    out = df.assign(_t=t, _pos=np.arange(len(df)))
    return out.sort_values(["_t", "_pos"], na_position="last", kind="stable").drop(columns=["_t", "_pos"])


def exact_duplicates(df: pd.DataFrame) -> pd.Series:
    """For each row (in time order), the id of the earlier row with the same key, or ''."""
    first = df.groupby("key", sort=False)["id"].transform("first")
    return first.where(first != df["id"], "")


def near_duplicates(texts: list[str], ids: list[str], threshold: float = NEAR_DUP_THRESHOLD,
                    chunk: int = 2000) -> dict[str, tuple[str, float]]:
    """Greedy near-duplicate removal on texts in priority order.

    Returns {removed_id: (kept_id, cosine)}. A text is removed if its char 3-5-gram TF-IDF cosine with an
    earlier text that is still kept exceeds ``threshold``.
    """
    if len(texts) < 2:
        return {}
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, lowercase=True, sublinear_tf=True)
    x = vec.fit_transform(texts)
    pairs: dict[int, list[tuple[int, float]]] = {}
    for start in range(0, x.shape[0], chunk):
        sims = (x[start:start + chunk] @ x.T).tocoo()
        for i, j, s in zip(sims.row + start, sims.col, sims.data):
            if j < i and s > threshold:
                pairs.setdefault(int(i), []).append((int(j), float(s)))
    removed: dict[int, tuple[int, float]] = {}
    for i in sorted(pairs):
        earlier = [(j, s) for j, s in pairs[i] if j not in removed]
        if earlier:
            j, s = max(earlier, key=lambda js: js[1])
            removed[i] = (j, s)
    return {ids[i]: (ids[j], round(s, 4)) for i, (j, s) in removed.items()}


def run_pipeline(raw: pd.DataFrame, meta: pd.DataFrame, threshold: float = NEAR_DUP_THRESHOLD):
    """Returns (clean, removed, annotated). ``removed`` has id, video_id, reason, duplicate_of, similarity."""
    df = annotate(raw, meta)
    df["reason"] = df.apply(rule_reason, axis=1)
    df["duplicate_of"] = ""
    df["similarity"] = np.nan

    passed = order_by_time(df[df["reason"] == ""])
    dup_of = exact_duplicates(passed)
    exact = dup_of != ""
    df.loc[passed.index[exact], "reason"] = "exact_dup"
    df.loc[passed.index[exact], "duplicate_of"] = dup_of[exact].values
    df.loc[passed.index[exact], "similarity"] = 1.0

    rest = passed[~exact]
    near = near_duplicates(rest["key"].tolist(), rest["id"].tolist(), threshold)
    idx = df.index[df["id"].isin(near)]
    df.loc[idx, "reason"] = "near_dup"
    df.loc[idx, "duplicate_of"] = df.loc[idx, "id"].map(lambda i: near[i][0])
    df.loc[idx, "similarity"] = df.loc[idx, "id"].map(lambda i: near[i][1])

    clean = df[df["reason"] == ""].copy()
    clean_out = clean.assign(text=clean["clean"])[CLEAN_COLUMNS + ["leaning", "video_lang"]]
    removed = df[df["reason"] != ""]
    return clean_out, removed, df


# ------------------------------------------------------------------------------------------- sampling


def allocate(available: dict[str, int], budget: int, cap: int, seed: int = SEED) -> dict[str, int]:
    """How many comments to take per video: equal shares up to ``cap`` and each video's size.

    Uses the largest equal quota q with sum(min(n_v, cap, q)) <= budget, then hands the leftover out one by
    one to videos that still have comments, in a seeded random order.
    """
    avail = {v: min(n, cap) for v, n in available.items()}
    if budget <= 0:
        return {v: 0 for v in avail}
    if sum(avail.values()) <= budget:
        return dict(avail)
    q = 0
    while sum(min(a, q + 1) for a in avail.values()) <= budget:
        q += 1
    take = {v: min(a, q) for v, a in avail.items()}
    left = budget - sum(take.values())
    rng = np.random.default_rng(seed)
    order = [v for v in rng.permutation(sorted(avail)) if avail[v] > take[v]]
    for v in order[:left]:
        take[v] += 1
    return take


def sample_for_labelling(clean: pd.DataFrame, n: int = 5000, cap: int = 100, seed: int = SEED) -> pd.DataFrame:
    """All comments from sceptic videos, then the rest stratified by video (at most ``cap`` per video)."""
    sceptic = clean[clean["leaning"] == "sceptic"]
    if len(sceptic) >= n:
        out = sceptic.sample(n, random_state=seed).assign(sample_group="sceptic_video")
        return out.sort_index()
    rest = clean[clean["leaning"] != "sceptic"]
    take = allocate(rest["video_id"].value_counts().to_dict(), n - len(sceptic), cap, seed)
    parts = [sceptic.assign(sample_group="sceptic_video")]
    for vid, k in sorted(take.items()):
        if k:
            parts.append(rest[rest["video_id"] == vid].sample(k, random_state=seed)
                         .assign(sample_group="stratified"))
    return pd.concat(parts).sort_index()


# ------------------------------------------------------------------------------------------- report


def step_report(df: pd.DataFrame) -> list[dict]:
    """Rows removed by each step, overall and by the video's language, with rows left after it."""
    rows, left = [], len(df)
    for step in STEPS:
        hit = df[df["reason"] == step]
        left -= len(hit)
        rows.append({"step": step, "removed": len(hit),
                     "removed_by_video_lang": {k: int(v) for k, v in hit["video_lang"].value_counts().items()},
                     "remaining": left})
    return rows


def print_report(n_raw: int, steps: list[dict], clean: pd.DataFrame, sample: pd.DataFrame | None, n: int) -> None:
    print(f"\nFilter report ({n_raw} raw comments)")
    print(f"  {'step':<16}{'removed':>8}{'remaining':>11}   by video language")
    for s in steps:
        print(f"  {s['step']:<16}{s['removed']:>8}{s['remaining']:>11}   {s['removed_by_video_lang']}")
    print(f"\nKept: {len(clean)} comments from {clean['video_id'].nunique()} videos")
    print(pd.crosstab(clean["lang"], clean["script"], margins=True, margins_name="total").to_string())
    if sample is None:
        print(f"\n{len(clean)} <= {n}: no sample needed; label all of comments_clean.csv")
    else:
        print(f"\nSample for labelling: {len(sample)} comments from {sample['video_id'].nunique()} videos")
        print(pd.crosstab(sample["sample_group"], sample["lang"], margins=True, margins_name="total").to_string())
        print(f"  max per video (stratified part): "
              f"{sample[sample['sample_group'] == 'stratified']['video_id'].value_counts().max()}")


def counts(df: pd.DataFrame, col: str) -> dict:
    return {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()}


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; re-run with --overwrite to replace it")
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)} ({len(df)} rows)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=RAW_CSV)
    p.add_argument("--candidates", type=Path, default=CANDIDATES_CSV)
    p.add_argument("--limit", type=int, default=None, help="dry run on the first N comments (writes *.dryrun.*)")
    p.add_argument("--near-dup-threshold", type=float, default=NEAR_DUP_THRESHOLD)
    p.add_argument("--sample-size", type=int, default=5000)
    p.add_argument("--per-video-cap", type=int, default=100)
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = p.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    raw = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.limit:
        raw = raw.head(args.limit)
    print(f"Read {len(raw)} comments from {args.input.relative_to(ROOT) if args.input.is_relative_to(ROOT) else args.input}")
    meta = load_video_meta(args.candidates)

    clean, removed, annotated = run_pipeline(raw, meta, args.near_dup_threshold)
    steps = step_report(annotated)
    sample = sample_for_labelling(clean, args.sample_size, args.per_video_cap) if len(clean) > args.sample_size else None
    print_report(len(raw), steps, clean, sample, args.sample_size)

    suffix = ".dryrun.csv" if args.limit else ".csv"
    print("\nWriting")
    write_csv(clean[CLEAN_COLUMNS], INTERIM / f"comments_clean{suffix}", args.overwrite or bool(args.limit))
    removed_cols = ["id", "video_id", "reason", "duplicate_of", "similarity", "lang", "script", "n_tokens", "text"]
    write_csv(removed[removed_cols], INTERIM / f"comments_removed{suffix}", args.overwrite or bool(args.limit))
    if sample is not None:
        write_csv(sample[CLEAN_COLUMNS + ["sample_group"]], INTERIM / f"comments_to_label{suffix}",
                  args.overwrite or bool(args.limit))
    if args.limit:
        print("Dry run: results JSON not written.")
        return

    near = annotated[annotated["reason"] == "near_dup"]
    vid_of = dict(zip(annotated["id"], annotated["video_id"]))
    cross = int((near["duplicate_of"].map(vid_of) != near["video_id"]).sum())
    out = {
        "experiment_id": "comment_cleaning",
        "script": "scripts/merge_and_dedup.py",
        "config": {"min_tokens": MIN_TOKENS, "near_dup_threshold": args.near_dup_threshold,
                   "ngram_range": list(NGRAM_RANGE), "analyzer": "char_wb, tf-idf (sublinear), cosine",
                   "sample_size": args.sample_size, "per_video_cap": args.per_video_cap},
        "seed": SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_comments": len(raw),
        "raw_videos": int(raw["video_id"].nunique()),
        "steps": steps,
        "near_dup_cross_video": cross,
        "kept": len(clean),
        "kept_videos": int(clean["video_id"].nunique()),
        "kept_by_lang": counts(clean, "lang"),
        "kept_by_script": counts(clean, "script"),
        "kept_by_lang_script": {f"{l}|{s}": int(v) for (l, s), v in
                                clean.groupby(["lang", "script"]).size().items()},
        "kept_from_sceptic_videos": int((clean["leaning"] == "sceptic").sum()),
        "sample": None if sample is None else {
            "rows": len(sample), "videos": int(sample["video_id"].nunique()),
            "by_group": counts(sample, "sample_group"), "by_lang": counts(sample, "lang"),
            "max_per_video_stratified": int(sample[sample["sample_group"] == "stratified"]["video_id"]
                                            .value_counts().max() or 0),
        },
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {RESULTS_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
