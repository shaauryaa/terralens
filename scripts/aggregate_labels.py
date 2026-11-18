"""Merge annotator labels, measure agreement and score annotators on the human gold set (T4).

Final label (two full annotators + adjudicator):
    Claude and Gemini agree             -> that label                       (label_source = agree)
    they disagree, gpt-oss matches one  -> gpt-oss's label                  (label_source = adjudicated)
    they disagree, gpt-oss matches none -> no label, review_flag = 1        (label_source = split)
    they disagree, no gpt-oss label yet -> no label                         (label_source = pending)
    Claude or Gemini label missing      -> no label                         (label_source = missing)
lang_uncertain rows are kept only if at least 2 annotators say "hi" or "bn"; their language is the majority
of those votes (tie: Claude's). Other rows always keep their T3 language.

Outputs
    data/interim/labels_merged.csv       one row per labelling item, with every annotator's label
    results/annotation_agreement.json    Cohen's kappa Claude vs Gemini (overall / per language / per script),
                                         % agreement, adjudication counts, Fleiss' kappa on the 300 sample,
                                         final label distribution per language
    results/llm_gold_check.json          per annotator: accuracy, Macro F1, per-class F1, confusion matrix vs
                                         the ClimateStance human labels, English and translated

Works with whatever label files exist (missing annotators are reported), so the gold check can run before
all labelling is finished.

Usage:
    python scripts/aggregate_labels.py                # everything available
    python scripts/aggregate_labels.py --gold-only    # only results/llm_gold_check.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from terralens import LABELS  # noqa: E402
from terralens.evaluate import evaluate  # noqa: E402

INTERIM = ROOT / "data" / "interim"
ITEMS_CSV = INTERIM / "labelling_items.csv"
GOLD_ITEMS_CSV = INTERIM / "gold_items.csv"
MERGED_CSV = INTERIM / "labels_merged.csv"
AGREEMENT_JSON = ROOT / "results" / "annotation_agreement.json"
GOLD_JSON = ROOT / "results" / "llm_gold_check.json"
ANNOTATORS = ["claude", "gemini", "gptoss"]
MODELS = {"claude": "Claude (separate session)", "gemini": "gemini-3.5-flash-lite", "gptoss": "openai/gpt-oss-120b"}
SEED = 42


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Fields an annotator's records must have to be used (annotation/prompts/claude_session.md for Claude).
REQUIRED_FIELDS = {"claude": ("gloss", "reason")}
REJECTED: dict[str, int] = {}


def read_labels(path: Path, annotator: str | None = None) -> pd.DataFrame:
    """Valid label records from a JSONL file (last record wins for repeated ids).

    Records missing a field in REQUIRED_FIELDS[annotator] are not used; their count is kept in REJECTED.
    """
    cols = ["id", "label", "confidence", "tags", "lang", "gloss", "reason", "set"]
    required = REQUIRED_FIELDS.get(annotator, ())
    REJECTED[path.name] = 0
    if not path.exists():
        return pd.DataFrame(columns=cols)
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("label") not in LABELS:
                continue
            if any(not str(r.get(k) or "").strip() for k in required):
                REJECTED[path.name] += 1
                continue
            recs.append({c: r.get(c) for c in cols})
    return pd.DataFrame(recs, columns=cols).drop_duplicates("id", keep="last")


# ============================================================================================ agreement


def fleiss_kappa(ratings: pd.DataFrame, categories=LABELS) -> float:
    """Fleiss' kappa for a frame of items x raters (every item rated by every rater)."""
    counts = np.stack([(ratings == c).sum(axis=1).to_numpy() for c in categories], axis=1).astype(float)
    n = counts.sum(axis=1)
    if len(counts) == 0 or not np.all(n == n[0]) or n[0] < 2:
        return float("nan")
    m = n[0]
    p_j = counts.sum(axis=0) / counts.sum()
    p_i = ((counts * (counts - 1)).sum(axis=1)) / (m * (m - 1))
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def kappa(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) == 0:
        return None
    if a.nunique() == 1 and b.nunique() == 1:
        return 1.0 if (a.values == b.values).all() else 0.0
    return round(float(cohen_kappa_score(a, b, labels=LABELS)), 4)


def pair_stats(df: pd.DataFrame, a: str, b: str, by: str | None = None) -> dict:
    both = df.dropna(subset=[a, b])
    def one(g):
        return {"n": len(g), "cohen_kappa": kappa(g[a], g[b]),
                "pct_agreement": round(float((g[a] == g[b]).mean()) * 100, 2) if len(g) else None}
    if by is None:
        return one(both)
    return {str(k): one(g) for k, g in both.groupby(by)}


# ============================================================================================ merge


def resolve(c, g, o) -> tuple[str | None, str, int]:
    """(final label, label_source, review_flag) from Claude, Gemini and gpt-oss labels (None = missing)."""
    if c is None or g is None:
        return None, "missing", 0
    if c == g:
        return c, "agree", 0
    if o is None:
        return None, "pending", 0
    if o in (c, g):
        return o, "adjudicated", 0
    return None, "split", 1


def resolve_lang(row) -> tuple[str, int]:
    """(language, keep) for one item. lang_uncertain rows need >= 2 hi/bn votes."""
    if str(row["lang_uncertain"]) != "1":
        return row["lang"], 1
    votes = [row.get(f"lang_{a}") for a in ANNOTATORS]
    indic = [v for v in votes if v in ("hi", "bn")]
    if len(indic) < 2:
        return "", 0
    n_hi, n_bn = indic.count("hi"), indic.count("bn")
    if n_hi != n_bn:
        return ("hi" if n_hi > n_bn else "bn"), 1
    return (row.get("lang_claude") if row.get("lang_claude") in ("hi", "bn") else indic[0]), 1


def merge(items: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = items.copy()
    for a in ANNOTATORS:
        lab = labels[a].set_index("id")
        df[f"label_{a}"] = df["id"].map(lab["label"])
        df[f"confidence_{a}"] = df["id"].map(lab["confidence"])
        df[f"tags_{a}"] = df["id"].map(lambda i, lab=lab: json.dumps(lab["tags"].get(i) or []) if i in lab.index else "")
        df[f"lang_{a}"] = df["id"].map(lab["lang"])
    claude = labels["claude"].set_index("id")
    df["gloss_claude"] = df["id"].map(claude["gloss"])
    df["reason_claude"] = df["id"].map(claude["reason"])
    res = [resolve(*(None if pd.isna(r[f"label_{a}"]) else r[f"label_{a}"] for a in ANNOTATORS))
           for _, r in df.iterrows()]
    df["final_label"] = [r[0] for r in res]
    df["label_source"] = [r[1] for r in res]
    df["review_flag"] = [r[2] for r in res]
    langs = [resolve_lang(r) for _, r in df.iterrows()]
    df["final_lang"] = [l for l, _ in langs]
    df["keep"] = [k for _, k in langs]
    return df


def agreement_report(df: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> dict:
    kept = df[df["keep"] == 1].assign(lang_group=lambda d: d["final_lang"].replace("", "unresolved"))
    disagree = kept.dropna(subset=["label_claude", "label_gemini"])
    disagree = disagree[disagree["label_claude"] != disagree["label_gemini"]]
    sample_ids = set(labels["gptoss"].loc[labels["gptoss"]["set"] == "sample", "id"])
    trio = kept[kept["id"].isin(sample_ids)].dropna(subset=[f"label_{a}" for a in ANNOTATORS])
    unc = df[df["lang_uncertain"].astype(str) == "1"]
    out = {
        "experiment_id": "annotation_agreement",
        "script": "scripts/aggregate_labels.py",
        "config": {"annotators": MODELS, "final_label_rule": "agree, else gpt-oss if it matches one, else split",
                   "lang_uncertain_rule": ">= 2 annotators say hi or bn"},
        "seed": SEED,
        "timestamp": now(),
        "items": len(df),
        "labelled": {a: int(df[f"label_{a}"].notna().sum()) for a in ANNOTATORS},
        "lang_uncertain": {"items": len(unc), "kept": int(unc["keep"].sum()),
                           "kept_by_lang": {str(k): int(v) for k, v in unc.loc[unc["keep"] == 1, "final_lang"].value_counts().items()}},
        "kept_items": len(kept),
        "claude_vs_gemini": {
            "overall": pair_stats(kept, "label_claude", "label_gemini"),
            "by_lang": pair_stats(kept, "label_claude", "label_gemini", "lang_group"),
            "by_script": pair_stats(kept, "label_claude", "label_gemini", "script"),
        },
        "adjudication": {
            "disagreements": len(disagree),
            "sent_to_gptoss": int(disagree["label_gptoss"].notna().sum()),
            "resolved_by_gptoss": int((disagree["label_source"] == "adjudicated").sum()),
            "three_way_splits": int((disagree["label_source"] == "split").sum()),
            "pending": int((disagree["label_source"] == "pending").sum()),
        },
        "label_source": {str(k): int(v) for k, v in kept["label_source"].value_counts().items()},
        "fleiss_kappa_sample": {"n": len(trio), "kappa": round(fleiss_kappa(trio[[f"label_{a}" for a in ANNOTATORS]]), 4)
                                if len(trio) else None},
        "final_label_distribution": {
            str(lang): {lab: int((g["final_label"] == lab).sum()) for lab in LABELS} | {"unlabelled": int(g["final_label"].isna().sum())}
            for lang, g in kept.groupby("lang_group")},
    }
    return out


# ============================================================================================ gold check


def score(gold: pd.Series, pred: pd.Series) -> dict:
    res = evaluate(gold, pred)
    return {
        "n": int(len(gold)),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "macro_f1": round(res["macro_f1"], 4),
        "per_class_f1": {k: round(float(v), 4) for k, v in res["per_class"]["f1"].items()},
        "confusion_matrix": {"rows_gold_cols_pred": LABELS, "matrix": res["confusion_matrix"].values.tolist()},
    }


def gold_check(gold_items: pd.DataFrame, labels: dict[str, pd.DataFrame]) -> dict:
    out = {"experiment_id": "llm_gold_check", "script": "scripts/aggregate_labels.py", "seed": SEED,
           "timestamp": now(),
           "gold": "data/external/climatestance_gold.csv (303 Reddit comments both human annotators agreed on)",
           "note": ("ClimateStance labels opposing climate policy as Against and sarcasm as ambiguous (None); "
                    "our guideline says None for policy opposition and labels sarcasm by meaning, so these "
                    "scores slightly underestimate agreement with our guideline"),
           "annotators": {}}
    subset_ids = set(gold_items.loc[gold_items["set"] == "gold_hi", "gold_id"])
    for a in ANNOTATORS:
        lab = labels[a].set_index("id")["label"]
        if lab.empty:
            rejected = REJECTED.get(f"labels_{a}_gold.jsonl", 0)
            out["annotators"][a] = {"status": f"not scored: {rejected} records lack required fields "
                                              f"{list(REQUIRED_FIELDS.get(a, ()))}" if rejected else "no gold labels yet"}
            continue
        res = {"model": MODELS[a]}
        g = gold_items.assign(pred=gold_items["id"].map(lab))
        views = {
            "en_all": g[g["set"] == "gold_en"],
            "en_direct": g[(g["set"] == "gold_en") & (g["mapping_confidence"] == "direct")],
            "en_translation_subset": g[(g["set"] == "gold_en") & g["gold_id"].isin(subset_ids)],
            "hi": g[g["set"] == "gold_hi"],
            "bn": g[g["set"] == "gold_bn"],
        }
        for name, v in views.items():
            have = v.dropna(subset=["pred"])
            if len(v) == 0:
                continue
            if len(have) == 0:
                res[name] = {"n_expected": len(v), "status": "not labelled"}
                continue
            res[name] = {**score(have["human_label"], have["pred"]), "n_expected": len(v),
                         "missing": int(len(v) - len(have))}
            if name == "en_direct":  # every direct row is Favour: only accuracy (= Favour recall) is meaningful
                for k in ("macro_f1", "per_class_f1"):
                    res[name].pop(k)
                res[name]["note"] = "all direct rows are human Favour; accuracy = recall of Favour"
        out["annotators"][a] = res
    return out


# ============================================================================================ main


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def print_gold(g: dict) -> None:
    print("\nGold check vs human labels (accuracy / Macro F1 / F1 Favour, Against, None)")
    for a, res in g["annotators"].items():
        if "status" in res:
            print(f"  {a:<7} {res['status']}")
            continue
        for view, s in res.items():
            if isinstance(s, dict) and "accuracy" in s and "per_class_f1" not in s:
                print(f"  {a:<7} {view:<22} n={s['n']:<4} acc={s['accuracy']:.3f}  ({s['note']})")
            elif isinstance(s, dict) and "accuracy" in s:
                f = s["per_class_f1"]
                print(f"  {a:<7} {view:<22} n={s['n']:<4} acc={s['accuracy']:.3f}  macroF1={s['macro_f1']:.3f}  "
                      f"F1 F/A/N = {f['Favour']:.2f}/{f['Against']:.2f}/{f['None']:.2f}"
                      + (f"  (missing {s['missing']})" if s.get("missing") else ""))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-only", action="store_true", help="only write results/llm_gold_check.json")
    p.add_argument("--limit", type=int, default=None, help="dry run on the first N items; prints, writes nothing")
    args = p.parse_args(argv)

    gold_items = pd.read_csv(GOLD_ITEMS_CSV, dtype=str, keep_default_na=False)
    gold_labels = {a: read_labels(INTERIM / f"labels_{a}_gold.jsonl", a) for a in ANNOTATORS}
    g = gold_check(gold_items.head(args.limit) if args.limit else gold_items, gold_labels)
    print_gold(g)
    if not args.limit:
        write_json(GOLD_JSON, g)
    if args.gold_only:
        return

    items = pd.read_csv(ITEMS_CSV, dtype=str, keep_default_na=False)
    if args.limit:
        items = items.head(args.limit)
    labels = {a: read_labels(INTERIM / f"labels_{a}.jsonl", a) for a in ANNOTATORS}
    merged = merge(items, labels)
    report = agreement_report(merged, labels)
    print("\n" + json.dumps({k: report[k] for k in ("labelled", "kept_items", "adjudication", "label_source")}, indent=1))
    print("Claude vs Gemini:", json.dumps(report["claude_vs_gemini"]["overall"]))
    if args.limit:
        print("Dry run: nothing written.")
        return
    merged.to_csv(MERGED_CSV, index=False, encoding="utf-8")
    print(f"wrote {MERGED_CSV.relative_to(ROOT)} ({len(merged)} rows)")
    write_json(AGREEMENT_JSON, report)


if __name__ == "__main__":
    main()
