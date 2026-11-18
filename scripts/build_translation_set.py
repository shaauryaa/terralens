"""Build the class-balanced English subset to translate into Hindi and Bengali (T9, training-only data).

Steps
1. Repair mojibake in the Kaggle Twitter climate sentiment data with ftfy (reports how many rows changed).
2. Clean every English row: strip a leading "RT @user:", then ``clean_text`` (URLs, mentions, emoji -> text).
   Keep rows with 6-60 tokens; drop near-identical texts (lower-cased letters only).
3. Pool: Kaggle (Pro -> Favour, Anti -> Against, Neutral -> None; the "News" rows are not used because their
   None mapping is uncertain), ClimateStance Reddit (Against only; its Favour labels are pseudo-labels),
   SemEval-2016 climate, and the ClimateStance gold set (allowed in training after the T4 check). GWSD is
   added on top.
4. Split between the languages (each English sentence is translated into ONE language only):
   all usable Against are split evenly between hi and bn; Favour and None are sampled (seed 42) so each
   language has about ``--per-lang`` rows from the pool; GWSD rows are split evenly between hi and bn.

Outputs
    data/interim/translation_source.csv   id, text (English), label, tgt_lang, source_dataset
    results/translation_set.json           counts (mojibake repairs, filters, per language and label)

Usage:
    python scripts/build_translation_set.py
    python scripts/build_translation_set.py --limit 500     # dry run on 500 rows per dataset
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402
from terralens.preprocess import clean_text, has_mojibake, repair_mojibake  # noqa: E402

EXTERNAL = ROOT / "data" / "external"
OUT_CSV = ROOT / "data" / "interim" / "translation_source.csv"
OUT_JSON = ROOT / "results" / "translation_set.json"
SEED = 42
RT_RE = re.compile(r"^\s*RT\s+@\w+:?\s*", re.I)
MIN_TOK, MAX_TOK = 6, 60


def prepare_text(text: str) -> str:
    return clean_text(RT_RE.sub("", repair_mojibake(text)))


def norm_key(text: str) -> str:
    return " ".join(re.sub(r"[^a-z ]", " ", text.lower()).split())


def usable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].map(prepare_text)
    n = df["text"].str.split().str.len()
    df = df[(n >= MIN_TOK) & (n <= MAX_TOK)]
    df = df.assign(_k=df["text"].map(norm_key)).drop_duplicates("_k").drop(columns="_k")
    return df


def load_pool(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = {}

    def read(name):
        d = pd.read_csv(EXTERNAL / f"{name}.csv", dtype=str, keep_default_na=False)
        return d.head(limit) if limit else d

    kag = read("kaggle_climate_sentiment")
    before = kag["text"].map(has_mojibake)
    repaired = kag["text"].map(repair_mojibake)
    report["kaggle_mojibake"] = {"rows": len(kag), "rows_with_mojibake_before": int(before.sum()),
                                 "rows_with_mojibake_after": int(repaired.map(has_mojibake).sum()),
                                 "rows_changed_by_ftfy": int((repaired != kag["text"]).sum())}
    kag = kag[kag["original_label_name"].isin(["Pro", "Anti", "Neutral"])]
    parts = [kag.assign(source_dataset="kaggle_climate_sentiment")]
    cs = read("climatestance_reddit_pseudo")
    parts.append(cs[cs["label"] == "Against"].assign(source_dataset="climatestance_reddit"))
    parts.append(read("semeval_climate").assign(source_dataset="semeval_climate"))
    parts.append(read("climatestance_gold").assign(source_dataset="climatestance_gold"))
    pool = pd.concat([p[["id", "text", "label", "source_dataset"]] for p in parts], ignore_index=True)
    pool = pool[pool["label"].isin(LABELS)]
    gwsd = read("gwsd").assign(source_dataset="gwsd")[["id", "text", "label", "source_dataset"]]
    gwsd = gwsd[gwsd["label"].isin(LABELS)]
    report["pool_before_filter"] = {lab: int((pool["label"] == lab).sum()) for lab in LABELS}
    pool, gwsd = usable(pool), usable(gwsd)
    gwsd = gwsd[~gwsd["text"].map(norm_key).isin(set(pool["text"].map(norm_key)))]
    report["pool_usable"] = {lab: int((pool["label"] == lab).sum()) for lab in LABELS}
    report["gwsd_usable"] = {lab: int((gwsd["label"] == lab).sum()) for lab in LABELS}
    return pool, gwsd, report


def assign_languages(pool: pd.DataFrame, gwsd: pd.DataFrame, per_lang: int = 3000, seed: int = SEED) -> pd.DataFrame:
    """Each row gets one tgt_lang. All Against split evenly; Favour/None fill each language to ~per_lang."""
    rng = np.random.default_rng(seed)
    out = []
    against = pool[pool["label"] == "Against"].sample(frac=1, random_state=seed)
    half = len(against) // 2
    out += [against.iloc[:half].assign(tgt_lang="hi"), against.iloc[half:].assign(tgt_lang="bn")]
    per_class = max(0, (per_lang - half) // 2)
    for lab in ("Favour", "None"):
        rows = pool[pool["label"] == lab].sample(frac=1, random_state=int(rng.integers(1 << 31)))
        take = rows.iloc[:2 * per_class]
        out += [take.iloc[:per_class].assign(tgt_lang="hi"), take.iloc[per_class:2 * per_class].assign(tgt_lang="bn")]
    g = gwsd.sample(frac=1, random_state=seed)
    out += [g.iloc[: len(g) // 2].assign(tgt_lang="hi"), g.iloc[len(g) // 2:].assign(tgt_lang="bn")]
    res = pd.concat(out, ignore_index=True)
    res["id"] = res["source_dataset"] + ":" + res["id"].astype(str)
    return res.drop_duplicates("id")[["id", "text", "label", "tgt_lang", "source_dataset"]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-lang", type=int, default=3000)
    p.add_argument("--limit", type=int, default=None, help="dry run: N rows per dataset, writes nothing")
    args = p.parse_args(argv)
    pool, gwsd, report = load_pool(args.limit)
    sel = assign_languages(pool, gwsd, args.per_lang)
    report["selected"] = {lang: {lab: int(((sel["tgt_lang"] == lang) & (sel["label"] == lab)).sum()) for lab in LABELS}
                          for lang in ("hi", "bn")}
    report["selected_by_source"] = {str(k): int(v) for k, v in sel["source_dataset"].value_counts().items()}
    report["total"] = len(sel)
    print(json.dumps(report, indent=1))
    if args.limit:
        print("Dry run: nothing written.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT_CSV, index=False, encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"experiment_id": "translation_set", "script": "scripts/build_translation_set.py",
                                    "config": {"per_lang": args.per_lang, "min_tokens": MIN_TOK, "max_tokens": MAX_TOK},
                                    "seed": SEED, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                    **report}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(sel)} rows) and {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
