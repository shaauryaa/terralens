"""Download auxiliary English climate-stance datasets into data/external/.

1. SemEval-2016 Task 6, target "Climate Change is a Real Concern"
   (Hugging Face: cardiffnlp/tweet_eval, config "stance_climate"; 0=none, 1=against, 2=favor)
   -> data/external/semeval_climate.csv
2. GWSD, Global Warming Stance Dataset (Luo et al., 2020)
   -> data/external/gwsd.csv

Both are written in the common schema: id, text, label (Favour/Against/None), source, lang="en".

Usage:
    python scripts/download_external.py
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402
from terralens.data import COMMON_COLUMNS, EXTERNAL_DIR, TWEET_EVAL_STANCE  # noqa: E402

GWSD_URLS = [
    "https://raw.githubusercontent.com/yiweiluo/GWStance/master/GWSD.tsv",
    "https://raw.githubusercontent.com/yiweiluo/GWStance/main/GWSD.tsv",
]
# GWSD annotator columns -> TerraLens labels (stance toward "global warming is a serious concern").
GWSD_LABELS = {"agree": "Favour", "disagree": "Against", "neutral": "None"}


def download_semeval_climate() -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "stance_climate")
    frames = []
    for split in ds:
        part = ds[split].to_pandas()
        part["split"] = split
        part["id"] = [f"semeval_climate_{split}_{i}" for i in range(len(part))]
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].map(TWEET_EVAL_STANCE)
    df["source"] = "semeval2016_climate"
    df["lang"] = "en"
    return df[COMMON_COLUMNS + ["split"]]


def fetch_gwsd_tsv() -> str:
    last_err = None
    for url in GWSD_URLS:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                print(f"Downloaded GWSD from {url}")
                return resp.read().decode("utf-8")
        except urllib.error.HTTPError as err:
            print(f"  {url} -> HTTP {err.code}, trying next")
            last_err = err
    raise RuntimeError(f"Could not download GWSD.tsv: {last_err}")


def download_gwsd() -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(fetch_gwsd_tsv()), sep="\t", index_col=0)
    prob_cols = list(GWSD_LABELS)

    # Each annotation batch starts with the same 5 attention-check ("screening") sentences
    # (sent_id s0-s4) that have no agree/neutral/disagree scores. The first 5 rows of the
    # file are the first such block; the same 5 sentences repeat in every batch, so all of
    # them are dropped, not just the first 5 rows.
    screening = raw["sent_id"].astype(str).str.startswith("s") | raw[prob_cols].isna().any(axis=1)
    print(f"GWSD: dropping {int(screening.sum())} screening rows ({raw.loc[screening, 'sentence'].nunique()} unique sentences)")
    df = raw.loc[~screening].copy()

    df["label"] = df[prob_cols].astype(float).idxmax(axis=1).map(GWSD_LABELS)
    df["id"] = "gwsd_" + df["guid"].astype(str)
    df["text"] = df["sentence"].astype(str)
    df["source"] = "gwsd"
    df["lang"] = "en"
    df["split"] = df["in_held_out_test"].map({True: "test", False: "train"})
    return df[COMMON_COLUMNS + ["split"]].reset_index(drop=True)


def report(name: str, df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().reindex(LABELS, fill_value=0)
    print(f"\n{name}: {len(df)} rows")
    for label, n in counts.items():
        print(f"  {label:<8} {n:>6}  ({n / len(df):.1%})")


def main() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    semeval = download_semeval_climate()
    semeval.to_csv(EXTERNAL_DIR / "semeval_climate.csv", index=False, encoding="utf-8")
    report("SemEval-2016 climate (semeval_climate.csv)", semeval)

    gwsd = download_gwsd()
    gwsd.to_csv(EXTERNAL_DIR / "gwsd.csv", index=False, encoding="utf-8")
    report("GWSD (gwsd.csv)", gwsd)

    print(f"\nSaved to {EXTERNAL_DIR}")


if __name__ == "__main__":
    main()
