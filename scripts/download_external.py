"""Download auxiliary English climate-stance datasets into data/external/ (training-only data).

Datasets (pick with --datasets; default: all):

semeval       SemEval-2016 Task 6, target "Climate Change is a Real Concern"
              (Hugging Face cardiffnlp/tweet_eval, config "stance_climate")  -> semeval_climate.csv
gwsd          Global Warming Stance Dataset (Luo et al., 2020)                -> gwsd.csv
climatestance ClimateStance / ClimateReddit (Vaid et al., 2022), GitHub repo
              roopalv54/finegrained-climate-change-social-media (formerly serendipity5497/...),
              pinned to commit CLIMATESTANCE_COMMIT:
                - climatestance_reddit.csv          329 Reddit comments, two human annotators
                - climatestance_reddit_pseudo.csv   6262 Reddit comments, labels predicted by a model
                - climatestance_tweet_ids.csv       3777 tweet IDs + labels; the repo has NO tweet text
                                                    (Twitter terms), so these cannot be used without hydration
kaggle        Twitter Climate Change Sentiment Dataset (Qian, Kaggle edqian/twitter-climate-change-sentiment-dataset)
                -> kaggle_climate_sentiment.csv. Uses the Kaggle API if it is installed and configured,
                otherwise reads a manually downloaded twitter_sentiment_data.csv from data/external/
                (the manual steps are printed).

Every output keeps the source label next to the TerraLens label. Columns:
    id, text, label, source, lang, source_dataset, original_id, original_label,
    original_label_name, mapping_confidence, split
`mapping_confidence` is direct / approximate / uncertain (see LABEL MAPPINGS below and data/DATASHEET.md).

Rows that are not kept (unknown label, empty text, annotator disagreement, duplicate text) are written to
<name>_excluded.csv with a `reason` column; nothing is dropped silently. Class counts before and after
mapping, exclusions, duplicates and licence notes are written to results/external_datasets.json.

Existing files are never overwritten unless --overwrite is given; downloaded raw files are cached in
data/external/<dataset>_raw/ and reused.

Usage:
    python scripts/download_external.py                          # all datasets
    python scripts/download_external.py --datasets climatestance kaggle
    python scripts/download_external.py --limit 50               # dry run: 50 raw rows per file, writes nothing
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402
from terralens.data import COMMON_COLUMNS, EXTERNAL_DIR, TWEET_EVAL_STANCE  # noqa: E402
from terralens.preprocess import clean_text  # noqa: E402

SEED = 42
RESULTS_PATH = ROOT / "results" / "external_datasets.json"

OUTPUT_COLUMNS = COMMON_COLUMNS + [
    "source_dataset", "original_id", "original_label", "original_label_name", "mapping_confidence", "split",
]
EXCLUDED_COLUMNS = OUTPUT_COLUMNS + ["reason"]

# ---------------------------------------------------------------------------------------------------------
# LABEL MAPPINGS. One rule per source label: (source label name, TerraLens label, confidence, reason).
# Confidence: direct = same meaning as our claim; approximate = mostly the same, with a known mismatch;
# uncertain = a sizeable share of rows may belong to another TerraLens class.
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MappingRule:
    name: str
    label: str
    confidence: str
    reason: str


# ClimateStance integer codes are not documented in the repo or the paper. They were identified by matching
# the code totals over the three tweet files (1: 2990, 0: 414, -1: 373) to the class counts in the paper
# (Sec. 6.1: favor 2990, ambiguous 414, against 373), and checked on the Reddit texts.
CLIMATESTANCE_MAP = {
    1: MappingRule("favor", "Favour", "direct",
                   "'opinion, action, concern against the climate change phenomenon' = treats it as a serious concern"),
    -1: MappingRule("against", "Against", "approximate",
                    "denial / ignorance of climate change, but also 'opposition of climate change policies', "
                    "which our guideline labels None when seriousness is not disputed"),
    0: MappingRule("ambiguous", "None", "approximate",
                   "no clear stance; but sarcastic tweets were also put here, which our guideline labels by "
                   "intended meaning (often Favour or Against)"),
}

# Kaggle: definitions quoted from the dataset description. Despite the dataset name, the classes describe
# belief in man-made climate change (a stance), not sentiment.
KAGGLE_MAP = {
    1: MappingRule("Pro", "Favour", "approximate",
                   "'supports the belief of man-made climate change'; belief in human cause is close to, "
                   "but not the same as, calling it a serious concern"),
    -1: MappingRule("Anti", "Against", "approximate",
                    "'does not believe in man-made climate change'; may deny the human cause rather than "
                    "the seriousness"),
    0: MappingRule("Neutral", "None", "direct",
                   "'neither supports nor refutes the belief of man-made climate change'"),
    2: MappingRule("News", "None", "uncertain",
                   "'links to factual news about climate change'; our guideline makes neutral facts None, "
                   "but headlines about harms can imply Favour"),
}

# The two existing datasets already target (almost) the same claim as TerraLens.
SEMEVAL_MAP = {
    2: MappingRule("FAVOR", "Favour", "direct", "target 'Climate Change is a Real Concern'"),
    1: MappingRule("AGAINST", "Against", "direct", "target 'Climate Change is a Real Concern'"),
    0: MappingRule("NONE", "None", "direct", "target 'Climate Change is a Real Concern'"),
}
GWSD_LABELS = {"agree": "Favour", "disagree": "Against", "neutral": "None"}

# ---------------------------------------------------------------------------------------------------------
# Sources and licence notes (checked 2026-11-18; see data/DATASHEET.md).
# ---------------------------------------------------------------------------------------------------------

GWSD_URLS = [
    "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv",
    "https://raw.githubusercontent.com/yiweiluo/GWStance/main/GWSD.tsv",
]

CLIMATESTANCE_REPO = "roopalv54/finegrained-climate-change-social-media"
CLIMATESTANCE_COMMIT = "fa1023b459eef6761eeb77b682c6fa4666e3edfd"
CLIMATESTANCE_FILES = {
    "tweets_train": "tweetId_label-train.tsv",
    "tweets_val": "tweetId_label-val.tsv",
    "tweets_test": "tweetId_label-test.tsv",
    "reddit_a1": "ClimateReddit/Reddit-Annotator1.tsv",
    "reddit_a2": "ClimateReddit/Reddit-Annotator2.tsv",
    "reddit_pseudo": "ClimateReddit/Pseudo-labeled-Reddit.tsv",
}

KAGGLE_REF = "edqian/twitter-climate-change-sentiment-dataset"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_REF}"
KAGGLE_RAW_NAME = "twitter_sentiment_data.csv"
KAGGLE_COLUMNS = {"sentiment", "message", "tweetid"}

LICENCES = {
    "semeval2016_climate": {"licence": "not re-checked in T2", "redistribute_text": None},
    "gwsd": {"licence": "not re-checked in T2", "redistribute_text": None},
    "climatestance": {
        "licence": "none stated: no LICENSE file and an empty README in the GitHub repo (GitHub API "
                   "license=null), so all rights are reserved by default",
        "attribution": "Vaid, Pant and Shrivastava (2022), Towards Fine-grained Classification of Climate "
                       "Change related Social Media Text, ACL Student Research Workshop",
        "restrictions": "tweets are released as IDs only, 'in accordance with the Twitter Terms of Service'; "
                        "Reddit comments are user content under Reddit's terms",
        "redistribute_text": False,
    },
    "kaggle_climate_sentiment": {
        "licence": "Kaggle licence field: 'Data files © Original Authors' (not an open licence)",
        "attribution": "Edward Qian, Kaggle; collection funded by a Canada Foundation for Innovation JELF grant "
                       "to Chris Bauch, University of Waterloo",
        "restrictions": "contains full tweet text; X/Twitter developer terms allow sharing tweet IDs, not text",
        "redistribute_text": False,
    },
}

# ---------------------------------------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Built:
    """One output file: kept rows, excluded rows and statistics for the report."""

    name: str
    source_dataset: str
    kept: pd.DataFrame
    excluded: pd.DataFrame
    counts_before: dict
    notes: list[str] = field(default_factory=list)
    write_excluded: bool = True


def dedup_key(text) -> str:
    """Normalised text used to find duplicates: clean_text, lower case, leading 'rt' removed."""
    key = clean_text(text, lowercase=True)
    if key.startswith("rt "):
        key = key[3:].lstrip(": ").strip()
    return key


def empty_frame(columns) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


def apply_mapping(df: pd.DataFrame, code_col: str, mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map integer source codes with ``mapping``. Unknown / missing codes go to the excluded frame.

    Adds original_label (the raw code as text), original_label_name, label and mapping_confidence.
    """
    df = df.copy()
    codes = pd.to_numeric(df[code_col], errors="coerce")
    df["original_label"] = df[code_col].astype(str)
    known = codes.isin(list(mapping))
    mapped_codes = codes.where(known).astype("Int64")
    df["original_label_name"] = mapped_codes.map(lambda c: mapping[int(c)].name if pd.notna(c) else "")
    df["label"] = mapped_codes.map(lambda c: mapping[int(c)].label if pd.notna(c) else None)
    df["mapping_confidence"] = mapped_codes.map(lambda c: mapping[int(c)].confidence if pd.notna(c) else "")
    excluded = df.loc[~known].copy()
    excluded["reason"] = "unknown_label"
    return df.loc[known].copy(), excluded


def drop_empty(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    empty = df["text"].map(dedup_key).eq("")
    excluded = df.loc[empty].copy()
    excluded["reason"] = "empty_text"
    return df.loc[~empty].copy(), excluded


def dedup_within(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicates (after ``dedup_key``) inside one dataset.

    Same text, same label: the first row is kept, the rest are excluded as ``duplicate_text``.
    Same text, different labels: every copy is excluded as ``duplicate_conflicting_labels``.
    """
    if df.empty:
        return df.copy(), empty_frame(list(df.columns) + ["reason"])
    keys = df["text"].map(dedup_key)
    n_labels = df.groupby(keys)["label"].transform("nunique")
    conflicting = n_labels > 1
    repeated = keys.duplicated(keep="first") & ~conflicting
    excluded = pd.concat([
        df.loc[conflicting].assign(reason="duplicate_conflicting_labels"),
        df.loc[repeated].assign(reason="duplicate_text"),
    ])
    return df.loc[~conflicting & ~repeated].copy(), excluded


def finish(df: pd.DataFrame, columns=OUTPUT_COLUMNS) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].reset_index(drop=True)


def clean_pipeline(df: pd.DataFrame, mapping: dict, code_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """apply_mapping -> drop_empty -> dedup_within; returns (kept, excluded)."""
    kept, ex_label = apply_mapping(df, code_col, mapping)
    kept, ex_empty = drop_empty(kept)
    kept, ex_dup = dedup_within(kept)
    excluded = pd.concat([ex_label, ex_empty, ex_dup], ignore_index=True)
    return finish(kept), finish(excluded, EXCLUDED_COLUMNS)


def label_counts(series: pd.Series) -> dict:
    return {k: int(v) for k, v in series.value_counts().reindex(LABELS, fill_value=0).items()}


def original_counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.astype(str).value_counts(dropna=False).sort_index().items()}


def rel(path: Path) -> Path:
    """Path relative to the repo root when possible (for printing)."""
    try:
        return path.resolve().relative_to(ROOT)
    except ValueError:
        return path


def http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------------------------------------
# SemEval-2016 and GWSD (unchanged sources; provenance columns added)
# ---------------------------------------------------------------------------------------------------------


def download_semeval_climate(limit: int | None = None) -> Built:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    frames = []
    for split in ds:
        part = ds[split].to_pandas()
        part["split"] = split
        part["id"] = [f"semeval_climate_{split}_{i}" for i in range(len(part))]
        frames.append(part.head(limit) if limit else part)
    raw = pd.concat(frames, ignore_index=True)
    assert set(TWEET_EVAL_STANCE) == set(SEMEVAL_MAP)
    df = raw.copy()
    df["original_label"] = df["label"].astype(str)
    df["original_label_name"] = df["label"].map(lambda c: SEMEVAL_MAP[int(c)].name)
    df["mapping_confidence"] = "direct"
    df["label"] = df["label"].map(TWEET_EVAL_STANCE)
    df["source"] = "semeval2016_climate"
    df["source_dataset"] = "semeval2016_climate"
    df["lang"] = "en"
    return Built("semeval_climate", "semeval2016_climate", finish(df), empty_frame(EXCLUDED_COLUMNS),
                 original_counts(raw["label"]), write_excluded=False)


def fetch_gwsd_tsv() -> str:
    last_err = None
    for url in GWSD_URLS:
        try:
            text = http_get(url).decode("utf-8")
            print(f"Downloaded GWSD from {url}")
            return text
        except urllib.error.HTTPError as err:
            print(f"  {url} -> HTTP {err.code}, trying next")
            last_err = err
    raise RuntimeError(f"Could not download GWSD.tsv: {last_err}")


def download_gwsd(limit: int | None = None) -> Built:
    raw = pd.read_csv(io.StringIO(fetch_gwsd_tsv()), sep="\t", index_col=0)
    if limit:
        raw = raw.head(limit + 5)
    prob_cols = list(GWSD_LABELS)

    # Each annotation batch starts with the same 5 attention-check ("screening") sentences
    # (sent_id s0-s4) that have no agree/neutral/disagree scores. The first 5 rows of the
    # file are the first such block; the same 5 sentences repeat in every batch, so all of
    # them are dropped, not just the first 5 rows.
    screening = raw["sent_id"].astype(str).str.startswith("s") | raw[prob_cols].isna().any(axis=1)
    print(f"GWSD: dropping {int(screening.sum())} screening rows ({raw.loc[screening, 'sentence'].nunique()} unique sentences)")
    df = raw.loc[~screening].copy()

    df["original_label_name"] = df[prob_cols].astype(float).idxmax(axis=1)
    df["original_label"] = df["original_label_name"]
    df["label"] = df["original_label_name"].map(GWSD_LABELS)
    df["mapping_confidence"] = "direct"
    df["id"] = "gwsd_" + df["guid"].astype(str)
    df["original_id"] = df["guid"].astype(str)
    df["text"] = df["sentence"].astype(str)
    df["source"] = "gwsd"
    df["source_dataset"] = "gwsd"
    df["lang"] = "en"
    df["split"] = df["in_held_out_test"].map({True: "test", False: "train"})
    built = Built("gwsd", "gwsd", finish(df), empty_frame(EXCLUDED_COLUMNS),
                  original_counts(df["original_label_name"]), write_excluded=False)
    built.notes.append(f"{int(screening.sum())} attention-check rows dropped (no annotator scores)")
    return built


# ---------------------------------------------------------------------------------------------------------
# ClimateStance / ClimateReddit (Vaid et al., 2022)
# ---------------------------------------------------------------------------------------------------------


def climatestance_url(rel_path: str) -> str:
    return f"https://raw.githubusercontent.com/{CLIMATESTANCE_REPO}/{CLIMATESTANCE_COMMIT}/{rel_path}"


def load_climatestance_raw(raw_dir: Path, cache: bool) -> dict[str, pd.DataFrame]:
    """Read the six TSVs, from ``raw_dir`` if already there, otherwise from GitHub (cached when ``cache``)."""
    frames = {}
    for key, repo_path in CLIMATESTANCE_FILES.items():
        local = raw_dir / Path(repo_path).name
        if local.exists():
            data = local.read_bytes()
            print(f"  ClimateStance: using cached {rel(local)}")
        else:
            data = http_get(climatestance_url(repo_path))
            if cache:
                raw_dir.mkdir(parents=True, exist_ok=True)
                local.write_bytes(data)
        dtype = {"tweet_id": str} if key.startswith("tweets") else None
        frames[key] = pd.read_csv(io.BytesIO(data), sep="\t", dtype=dtype)
    return frames


def subreddit(permalink: str) -> str:
    parts = str(permalink).split("/")
    return parts[2] if len(parts) > 2 else ""


def build_climatestance_reddit(a1: pd.DataFrame, a2: pd.DataFrame) -> Built:
    """Two-annotator Reddit set: keep comments where both annotators gave the same opinion label."""
    if not a1["permalink"].reset_index(drop=True).equals(a2["permalink"].reset_index(drop=True)):
        raise ValueError("ClimateReddit annotator files are not row-aligned (permalinks differ)")
    df = a1.reset_index(drop=True).rename(columns={"opinion-label": "code_a1"})
    df["code_a2"] = a2["opinion-label"].reset_index(drop=True)
    df["text"] = df["body"].astype(str)
    df["original_id"] = df["permalink"].astype(str)
    df["id"] = [f"climatestance_reddit_{i}" for i in range(len(df))]
    df["source"] = "climatestance_reddit"
    df["source_dataset"] = "climatestance"
    df["lang"] = "en"
    df["split"] = "test"  # the paper uses these 329 comments as the manual test set for the pseudo labels

    agree = df["code_a1"] == df["code_a2"]
    disagree = df.loc[~agree].copy()
    disagree["original_label"] = "a1=" + disagree["code_a1"].astype(str) + "|a2=" + disagree["code_a2"].astype(str)
    disagree["reason"] = "annotator_disagreement"

    kept, excluded = clean_pipeline(df.loc[agree], CLIMATESTANCE_MAP, "code_a1")
    excluded = pd.concat([finish(disagree, EXCLUDED_COLUMNS), excluded], ignore_index=True)
    counts_before = {
        "annotator1": original_counts(df["code_a1"]),
        "annotator2": original_counts(df["code_a2"]),
        "agreed": original_counts(df.loc[agree, "code_a1"]),
    }
    built = Built("climatestance_reddit", "climatestance", kept, excluded, counts_before)
    built.notes.append(f"annotators agree on {int(agree.sum())}/{len(df)} comments ({agree.mean():.1%})")
    return built


def build_climatestance_pseudo(raw: pd.DataFrame) -> Built:
    """Model-predicted labels (Vaid et al.'s best stance model); kept separate from human labels."""
    df = raw.reset_index(drop=True).copy()
    df["text"] = df["body"].astype(str)
    df["original_id"] = df["permalink"].astype(str)
    df["id"] = [f"climatestance_reddit_pseudo_{i}" for i in range(len(df))]
    df["source"] = "climatestance_reddit_pseudo"
    df["source_dataset"] = "climatestance"
    df["lang"] = "en"
    df["split"] = "train"
    kept, excluded = clean_pipeline(df, CLIMATESTANCE_MAP, "opinion")
    kept["mapping_confidence"] = "uncertain"  # labels are model predictions, not human annotations
    built = Built("climatestance_reddit_pseudo", "climatestance", kept, excluded, original_counts(df["opinion"]))
    skeptic = df["permalink"].map(subreddit).str.lower().eq("climateskeptics")
    sk = df.loc[skeptic, "opinion"].astype(str).value_counts().to_dict()
    n_ambiguous = int(pd.to_numeric(df["opinion"], errors="coerce").eq(0).sum())
    built.notes.append(
        f"labels are model predictions (paper: stance F1 0.343); {n_ambiguous} 'ambiguous' predictions; "
        f"r/climateskeptics comments labelled {sk} -- do not use without relabelling")
    return built


def build_climatestance_tweet_ids(frames: dict[str, pd.DataFrame]) -> Built:
    """Tweet IDs with labels. The repo has no tweet text, so nothing here is usable for training as is."""
    parts = []
    for split in ("train", "val", "test"):
        part = frames[f"tweets_{split}"].copy()
        part["split"] = split
        parts.append(part)
    df = pd.concat(parts, ignore_index=True)
    df["stance-label"] = pd.to_numeric(df["stance-label"], errors="coerce").astype("Int64")  # stored as 1.0
    df["original_id"] = df["tweet_id"].astype(str)
    df["id"] = "climatestance_tweet_" + df["original_id"]
    df["text"] = ""
    df["source"] = "climatestance_tweets"
    df["source_dataset"] = "climatestance"
    df["lang"] = "en"
    kept, excluded = apply_mapping(df, "stance-label", CLIMATESTANCE_MAP)
    dup = kept["original_id"].duplicated(keep="first")
    excluded = pd.concat([excluded, kept.loc[dup].assign(reason="duplicate_tweet_id")])
    kept = kept.loc[~dup].copy()
    kept["topic_label"] = kept["topic-label"]
    built = Built("climatestance_tweet_ids", "climatestance",
                  finish(kept, OUTPUT_COLUMNS + ["topic_label"]), finish(excluded, EXCLUDED_COLUMNS),
                  original_counts(df["stance-label"]))
    built.notes.append("text not available (IDs only, Twitter terms); needs hydration through the X API")
    return built


def download_climatestance(out_dir: Path, limit: int | None, dry_run: bool) -> list[Built]:
    frames = load_climatestance_raw(out_dir / "climatestance_raw", cache=not dry_run)
    if limit:
        frames = {k: v.head(limit) for k, v in frames.items()}
    return [
        build_climatestance_reddit(frames["reddit_a1"], frames["reddit_a2"]),
        build_climatestance_pseudo(frames["reddit_pseudo"]),
        build_climatestance_tweet_ids(frames),
    ]


# ---------------------------------------------------------------------------------------------------------
# Kaggle: Twitter Climate Change Sentiment Dataset
# ---------------------------------------------------------------------------------------------------------

KAGGLE_MANUAL_STEPS = f"""
Kaggle dataset not found and the Kaggle API is not configured. To add it by hand:
  1. Sign in to Kaggle and open {KAGGLE_URL}
  2. Click Download, unzip, and put {KAGGLE_RAW_NAME} in data/external/
     (expected columns: sentiment, message, tweetid)
  3. Re-run: python scripts/download_external.py --datasets kaggle
Or set up the API: pip install kaggle, then put kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME and
KAGGLE_KEY in .env) and re-run. The licence is 'Data files (c) Original Authors' and the file holds tweet
text, so it stays out of git."""


def kaggle_credentials_present() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return has_env or has_file or bool(os.environ.get("KAGGLE_API_TOKEN"))


def find_kaggle_raw(out_dir: Path, explicit: Path | None = None) -> Path | None:
    candidates = [explicit] if explicit else []
    candidates += [out_dir / KAGGLE_RAW_NAME, out_dir / "kaggle_climate_sentiment_raw" / KAGGLE_RAW_NAME]
    return next((p for p in candidates if p is not None and p.exists()), None)


def kaggle_api_download(dest: Path) -> Path | None:
    """Download with the Kaggle API into a new folder; returns None if the API is unavailable."""
    if not kaggle_credentials_present():
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        dest.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(KAGGLE_REF, path=str(dest), unzip=True)
    except Exception as err:  # ImportError, auth or network errors: fall back to the manual route
        print(f"  Kaggle API download failed: {err}")
        return None
    path = dest / KAGGLE_RAW_NAME
    return path if path.exists() else None


def build_kaggle(raw: pd.DataFrame) -> Built:
    missing = KAGGLE_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"Kaggle file is missing columns {sorted(missing)}; found {list(raw.columns)}")
    df = raw.reset_index(drop=True).copy()
    df["text"] = df["message"].astype(str)
    df["original_id"] = df["tweetid"].astype(str)
    df["id"] = [f"kaggle_climate_sentiment_{i}" for i in range(len(df))]
    df["source"] = "kaggle_climate_sentiment"
    df["source_dataset"] = "kaggle_climate_sentiment"
    df["lang"] = "en"
    df["split"] = ""
    kept, excluded = clean_pipeline(df, KAGGLE_MAP, "sentiment")
    built = Built("kaggle_climate_sentiment", "kaggle_climate_sentiment", kept, excluded,
                  original_counts(df["sentiment"]))
    built.notes.append("text holds @handles and URLs as published; clean before use")
    return built


def download_kaggle(out_dir: Path, limit: int | None, dry_run: bool, explicit: Path | None) -> list[Built]:
    path = find_kaggle_raw(out_dir, explicit)
    if path is None and not dry_run:
        path = kaggle_api_download(out_dir / "kaggle_climate_sentiment_raw")
    if path is None:
        print(KAGGLE_MANUAL_STEPS)
        return []
    print(f"  Kaggle: reading {path}")
    raw = pd.read_csv(path, dtype={"tweetid": str}, nrows=limit)
    return [build_kaggle(raw)]


# ---------------------------------------------------------------------------------------------------------
# Reporting and writing
# ---------------------------------------------------------------------------------------------------------


def write_csv(df: pd.DataFrame, path: Path, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        print(f"  {path.name} exists, not overwritten (use --overwrite to replace it)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  wrote {rel(path)} ({len(df)} rows)")
    return True


def stats_for(built: Built) -> dict:
    kept = built.kept
    has_text = kept["text"].astype(str).str.strip().ne("").any() if len(kept) else False
    return {
        "status": "built",
        "source_dataset": built.source_dataset,
        "rows_kept": len(kept),
        "counts_before_mapping": built.counts_before,
        "counts_after_mapping": label_counts(kept["label"]),
        "after_mapping_by_confidence": {
            str(k): int(v) for k, v in kept["mapping_confidence"].value_counts().sort_index().items()},
        "excluded": {str(k): int(v) for k, v in built.excluded["reason"].value_counts().sort_index().items()},
        "has_text": bool(has_text),
        "notes": built.notes,
        "licence": LICENCES.get(built.source_dataset, {}),
    }


def report(built: Built, stats: dict) -> None:
    print(f"\n{built.name} ({built.source_dataset}): {stats['rows_kept']} rows kept")
    print(f"  before mapping (source labels): {stats['counts_before_mapping']}")
    total = max(stats["rows_kept"], 1)
    for label, n in stats["counts_after_mapping"].items():
        print(f"  {label:<8} {n:>6}  ({n / total:.1%})")
    print(f"  by mapping confidence: {stats['after_mapping_by_confidence']}")
    if stats["excluded"]:
        print(f"  excluded: {stats['excluded']}")
    for note in built.notes:
        print(f"  note: {note}")


def cross_overlap(frames: dict[str, pd.DataFrame]) -> dict:
    """Number of normalised texts shared by each pair of external files (reported, not removed)."""
    keys = {name: set(df["text"].map(dedup_key)) - {""} for name, df in frames.items()}
    names = sorted(keys)
    return {f"{a}|{b}": len(keys[a] & keys[b])
            for i, a in enumerate(names) for b in names[i + 1:] if keys[a] and keys[b]}


def existing_external_frames(out_dir: Path, skip: set[str]) -> dict[str, pd.DataFrame]:
    names = ["semeval_climate", "gwsd", "climatestance_reddit", "climatestance_reddit_pseudo",
             "kaggle_climate_sentiment"]
    frames = {}
    for name in names:
        path = out_dir / f"{name}.csv"
        if name not in skip and path.exists():
            frames[name] = pd.read_csv(path, usecols=["text"], keep_default_na=False)
    return frames


def update_results(path: Path, stats: dict, overlap: dict, args, pending: dict | None = None) -> None:
    """Merge this run's stats into the results JSON. ``pending`` entries never replace real stats."""
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    datasets = previous.get("datasets", {})
    for name, s in stats.items():
        datasets[name] = {**s, "updated": now}
    for name, s in (pending or {}).items():
        if datasets.get(name, {}).get("status") != "built":
            datasets[name] = {**s, "updated": now}
    out = {
        "experiment_id": "external_datasets",
        "script": "scripts/download_external.py",
        "config": {"datasets": args.datasets, "climatestance_repo": CLIMATESTANCE_REPO,
                   "climatestance_commit": CLIMATESTANCE_COMMIT, "kaggle_ref": KAGGLE_REF},
        "seed": SEED,
        "timestamp": now,
        "datasets": dict(sorted(datasets.items())),
        "cross_dataset_text_overlap": overlap,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {rel(path)}")


ALL_DATASETS = ["semeval", "gwsd", "climatestance", "kaggle"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", choices=ALL_DATASETS, default=ALL_DATASETS,
                        help="which datasets to build (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="dry run: use the first N raw rows of each file, print the report, write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing output CSVs (raw downloads are never replaced)")
    parser.add_argument("--kaggle-file", type=Path, default=None,
                        help=f"path to a manually downloaded {KAGGLE_RAW_NAME}")
    parser.add_argument("--out-dir", type=Path, default=EXTERNAL_DIR, help="output folder (default: data/external)")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(SEED)
    dry_run = args.limit is not None
    out_dir: Path = args.out_dir
    if dry_run:
        print(f"DRY RUN: first {args.limit} raw rows per file; nothing is written.")

    built: list[Built] = []
    if "semeval" in args.datasets:
        built.append(download_semeval_climate(args.limit))
    if "gwsd" in args.datasets:
        built.append(download_gwsd(args.limit))
    if "climatestance" in args.datasets:
        built += download_climatestance(out_dir, args.limit, dry_run)
    if "kaggle" in args.datasets:
        built += download_kaggle(out_dir, args.limit, dry_run, args.kaggle_file)

    stats = {b.name: stats_for(b) for b in built}
    for b in built:
        report(b, stats[b.name])

    if dry_run:
        print("\nDry run finished; no files written.")
        return

    print(f"\nWriting to {out_dir}")
    for b in built:
        write_csv(b.kept, out_dir / f"{b.name}.csv", args.overwrite)
        if b.write_excluded and len(b.excluded):
            write_csv(b.excluded, out_dir / f"{b.name}_excluded.csv", args.overwrite)

    frames = {b.name: b.kept for b in built if stats[b.name]["has_text"]}
    frames.update(existing_external_frames(out_dir, skip=set(frames)))
    overlap = cross_overlap(frames)
    print(f"\nCross-dataset text overlap (pairs > 0): { {k: v for k, v in overlap.items() if v} or 'none'}")
    pending = {}
    if "kaggle" in args.datasets and "kaggle_climate_sentiment" not in stats:
        pending["kaggle_climate_sentiment"] = {
            "status": "not run yet: raw file not downloaded (Kaggle API not configured)",
            "source_dataset": "kaggle_climate_sentiment",
            "licence": LICENCES["kaggle_climate_sentiment"],
        }
    update_results(RESULTS_PATH, stats, overlap, args, pending)


if __name__ == "__main__":
    main()
