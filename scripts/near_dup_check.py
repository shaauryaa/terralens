"""Near-duplicate check of training-only data (B translated, C romanised, D synthetic) against val, test and the
challenge set (CLAUDE.md data rule 5). Matches are dropped from the training files in place.

Same similarity as T3 (scripts/merge_and_dedup.py): char_wb 3-5-gram TF-IDF (sublinear), cosine > 0.9, on
cleaned lower-cased text. The vectoriser is fitted on the held-out texts plus the candidate file. Also drops
exact cleaned-text matches.

Held-out sets: Source A val and test, data/processed/test_noisy.csv and data/challenge/challenge.csv.
Training files checked (if present): data/processed/{translated,romanised,synthetic}.csv.

Output: results/near_dup_check.json (rows checked / dropped per file, with the matched held-out set).

Usage:
    python scripts/near_dup_check.py
    python scripts/near_dup_check.py --limit 200    # dry run on 200 rows per file, drops nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.preprocess import clean_text  # noqa: E402
from terralens.splits import load_splits  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
TRAIN_FILES = ["translated.csv", "romanised.csv", "synthetic.csv"]
HELD_OUT_EXTRA = {"test_noisy": PROCESSED / "test_noisy.csv", "challenge": ROOT / "data" / "challenge" / "challenge.csv"}
RESULTS_JSON = ROOT / "results" / "near_dup_check.json"
THRESHOLD = 0.9
NGRAM_RANGE = (3, 5)
KEEP_RE = re.compile(r"[^\p{L}\p{M}\p{N}]+")


def held_out_texts() -> dict[str, list[str]]:
    s = load_splits()
    out = {"val": s["val"]["text"].tolist(), "test": s["test"]["text"].tolist()}
    for name, path in HELD_OUT_EXTRA.items():
        if path.exists():
            out[name] = pd.read_csv(path, dtype=str, keep_default_na=False)["text"].map(clean_text).tolist()
    return out


def find_matches(candidates: list[str], held: dict[str, list[str]], threshold: float = THRESHOLD,
                 chunk: int = 2000) -> list[str | None]:
    """For each candidate: the name of the first held-out set it near-duplicates, else None."""
    names, texts = [], []
    for name, t in held.items():
        names += [name] * len(t)
        texts += t
    key = lambda x: " ".join(KEEP_RE.sub(" ", str(x).lower()).split())  # noqa: E731 - letters/marks/digits only
    cand = [key(c) for c in candidates]
    hk = [key(t) for t in texts]
    exact = {}
    for n, t in zip(names, hk):
        exact.setdefault(t, n)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE, lowercase=True, sublinear_tf=True)
    vec.fit(hk + cand)
    H = vec.transform(hk)
    out: list[str | None] = []
    for i in range(0, len(cand), chunk):
        C = vec.transform(cand[i:i + chunk])
        sims = (C @ H.T).toarray() if H.shape[0] else np.zeros((C.shape[0], 0))
        for j, c in enumerate(cand[i:i + chunk]):
            if c in exact:
                out.append(exact[c])
            elif sims.shape[1] and sims[j].max() > threshold:
                out.append(names[int(sims[j].argmax())])
            else:
                out.append(None)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--limit", type=int, default=None, help="dry run: N rows per file, drops nothing")
    args = p.parse_args(argv)
    held = held_out_texts()
    report = {"experiment_id": "near_dup_check", "script": "scripts/near_dup_check.py", "seed": None,
              "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "config": {"threshold": args.threshold, "ngram_range": list(NGRAM_RANGE), "analyzer": "char_wb tf-idf"},
              "held_out": {k: len(v) for k, v in held.items()}, "files": {}}
    for fname in TRAIN_FILES:
        path = PROCESSED / fname
        if not path.exists():
            report["files"][fname] = "not built"
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if args.limit:
            df = df.head(args.limit)
        match = find_matches(df["text"].map(clean_text).tolist(), held, args.threshold)
        hit = pd.Series([m is not None for m in match], index=df.index)
        report["files"][fname] = {"rows": len(df), "dropped": int(hit.sum()),
                                  "dropped_by_set": {str(k): int(v) for k, v in
                                                     pd.Series([m for m in match if m]).value_counts().items()}}
        print(fname, report["files"][fname])
        if not args.limit and hit.any():
            df[~hit].to_csv(path, index=False, encoding="utf-8")
    if args.limit:
        print("Dry run: nothing dropped or written.")
        return
    RESULTS_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RESULTS_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
