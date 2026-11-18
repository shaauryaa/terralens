"""Make the video-grouped train / val / test split of Source A (run once, then never change).

StratifiedGroupKFold with 20 folds on ``video_id`` (all comments of a video stay in one split), stratified
on label x language. Three folds are test (15%), three are val (15%), the other 14 are train (70%). Which folds
go to test and val is drawn with the seed: with ~64 distinct videos, sklearn's ``shuffle`` only breaks ties, so
the fold contents barely depend on the seed and the seed must also choose the fold-to-split assignment.

Against is rare (about 100 comments in about 30 videos), so the script tries seeds 42, 43, ... 61 and uses
the first seed where val and test each have at least ``--min-against`` Against comments and both languages
(hi, bn). The chosen seed and every seed tried are logged in results/splits.json.

Outputs
    data/splits/train_ids.txt, val_ids.txt, test_ids.txt    one comment id per line
    results/splits.json                                     seed, seeds tried, counts per split / label / language

Usage:
    python scripts/make_splits.py
    python scripts/make_splits.py --limit 500     # dry run on 500 rows, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402

SOURCE_A = ROOT / "data" / "processed" / "source_a.csv"
SPLIT_DIR = ROOT / "data" / "splits"
RESULTS_JSON = ROOT / "results" / "splits.json"
N_FOLDS = 20
N_TEST_FOLDS = 3
N_VAL_FOLDS = 3
SEEDS = range(42, 62)
LANGS = ("hi", "bn")


def assign_splits(df: pd.DataFrame, seed: int, n_folds: int = N_FOLDS) -> pd.Series:
    """Series of 'train' / 'val' / 'test' aligned with ``df`` (grouped by video_id)."""
    strat = df["label"].astype(str) + "_" + df["lang"].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, strat, groups=df["video_id"])):
        fold[idx] = k
    test_folds, val_folds = fold_assignment(seed, n_folds)
    split = np.where(np.isin(fold, test_folds), "test", np.where(np.isin(fold, val_folds), "val", "train"))
    return pd.Series(split, index=df.index, name="split")


def fold_assignment(seed: int, n_folds: int = N_FOLDS) -> tuple[list[int], list[int]]:
    """(test folds, val folds) drawn with the seed."""
    perm = np.random.default_rng(seed).permutation(n_folds)
    return sorted(perm[:N_TEST_FOLDS].tolist()), sorted(perm[N_TEST_FOLDS:N_TEST_FOLDS + N_VAL_FOLDS].tolist())


def check(df: pd.DataFrame, split: pd.Series, min_against: int) -> tuple[bool, dict]:
    """(ok, per-split counts). ok = val and test each have >= min_against Against and both languages."""
    info = {}
    ok = True
    for name in ("train", "val", "test"):
        part = df[split == name]
        n_against = int((part["label"] == "Against").sum())
        langs = sorted(part["lang"].unique().tolist())
        info[name] = {"n": len(part), "videos": int(part["video_id"].nunique()), "against": n_against,
                      "langs": langs}
        if name != "train" and (n_against < min_against or not set(LANGS) <= set(langs)):
            ok = False
    return ok, info


def choose_seed(df: pd.DataFrame, seeds=SEEDS, min_against: int = 12) -> tuple[int | None, pd.Series | None, list]:
    tried = []
    for seed in seeds:
        split = assign_splits(df, seed)
        ok, info = check(df, split, min_against)
        tried.append({"seed": seed, "ok": ok, **{k: {"n": v["n"], "against": v["against"]} for k, v in info.items()}})
        if ok:
            return seed, split, tried
    return None, None, tried


def summary(df: pd.DataFrame, split: pd.Series) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        part = df[split == name]
        out[name] = {
            "n": len(part),
            "videos": int(part["video_id"].nunique()),
            "label": {lab: int((part["label"] == lab).sum()) for lab in LABELS},
            "by_lang": {lang: {lab: int(((part["lang"] == lang) & (part["label"] == lab)).sum()) for lab in LABELS}
                        for lang in LANGS},
            "script": {str(k): int(v) for k, v in part["script"].value_counts().items()},
        }
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=SOURCE_A)
    p.add_argument("--min-against", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="dry run on the first N rows; writes nothing")
    p.add_argument("--force", action="store_true", help="overwrite existing split files")
    args = p.parse_args(argv)

    df = pd.read_csv(args.source, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    if (df["video_id"] == "").any():
        raise SystemExit("every Source A row needs a video_id")
    seed, split, tried = choose_seed(df, min_against=args.min_against if not args.limit else 0)
    for t in tried:
        print(f"seed {t['seed']}: ok={t['ok']}  " + "  ".join(f"{k} n={t[k]['n']} against={t[k]['against']}"
                                                             for k in ("train", "val", "test")))
    if seed is None:
        raise SystemExit(f"no seed in {SEEDS.start}-{SEEDS.stop - 1} satisfies the constraints")
    stats = summary(df, split)
    print(f"chosen seed: {seed}")
    print(json.dumps({k: {"n": v["n"], "videos": v["videos"], "label": v["label"]} for k, v in stats.items()}))
    if args.limit:
        print("Dry run: nothing written.")
        return
    existing = [SPLIT_DIR / f"{n}_ids.txt" for n in ("train", "val", "test") if (SPLIT_DIR / f"{n}_ids.txt").exists()]
    if existing and not args.force:
        raise SystemExit(f"split files already exist ({existing[0].name}); splits are fixed. Use --force to redo.")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        ids = df.loc[split == name, "id"].tolist()
        (SPLIT_DIR / f"{name}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps({
        "experiment_id": "splits", "script": "scripts/make_splits.py",
        "config": {"method": "StratifiedGroupKFold", "n_folds": N_FOLDS,
                   "test_folds": fold_assignment(seed)[0], "val_folds": fold_assignment(seed)[1],
                   "fold_assignment": "numpy default_rng(seed).permutation(20): first 3 test, next 3 val", "stratify": "label x lang", "group": "video_id",
                   "min_against_val_test": args.min_against, "seeds_allowed": [SEEDS.start, SEEDS.stop - 1]},
        "seed": seed, "seeds_tried": tried, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_rows": len(df), "splits": stats,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {SPLIT_DIR}/*_ids.txt and {RESULTS_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
