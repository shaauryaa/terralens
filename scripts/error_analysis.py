"""Error analysis of the best model on VALIDATION (T11 draft, 6e). Test is not used.

Subcommands
    sample    pick the best run by val Macro F1 (encoder / QLoRA results, seed 42, no seed or robustness reruns),
              sample 40 misclassified val comments (seed 42, spread over gold -> pred pairs) and get English
              glosses from Gemini (cached in data/interim/error_glosses.jsonl; 2 requests)
    notebook  build and execute notebooks/03_error_analysis.ipynb: confusion matrix, Macro F1 per language /
              script / annotator tag, and the 40 examples (English gloss, gold, prediction, category)
    summary   write results/error_analysis.json: run, number of val errors, the sampled gold -> prediction pairs
              and the hand-assigned category counts (no comment text)

Categories are assigned by hand (in the Claude Code session) after reading the glosses and saved in
reports/error_categories.csv (id, category). Only ids, labels, glosses and categories appear in the notebook;
the original comment text stays in data/interim/error_sample.csv (not in git).

Usage:
    python scripts/error_analysis.py sample
    python scripts/error_analysis.py notebook
    python scripts/error_analysis.py summary
    python scripts/error_analysis.py sample --limit 1      # dry run: show the chosen run, no requests
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from terralens.reporting import now  # noqa: E402
from terralens.splits import load_splits  # noqa: E402

RESULTS = ROOT / "results"
SAMPLE_CSV = ROOT / "data" / "interim" / "error_sample.csv"
GLOSS_JSONL = ROOT / "data" / "interim" / "error_glosses.jsonl"
CATEGORIES_CSV = ROOT / "reports" / "error_categories.csv"
NOTEBOOK = ROOT / "notebooks" / "03_error_analysis.ipynb"
N = 40
SEED = 42


def best_run() -> str:
    cands = {}
    for p in RESULTS.glob("*.json"):
        if p.stem.endswith(("_s43", "_s44", "_robust")) or "dryrun" in p.stem:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("kind") in ("encoder", "qlora") and (RESULTS / "predictions" / f"{p.stem}_val.csv").exists():
            cands[p.stem] = r["val"]["macro_f1"]
    if not cands:
        raise SystemExit("no encoder results with val predictions")
    return max(cands, key=cands.get)


def misclassified_sample(val: pd.DataFrame, pred: pd.DataFrame, n: int = N, seed: int = SEED) -> pd.DataFrame:
    df = val.merge(pred[["id", "pred"]], on="id")
    wrong = df[df["label"] != df["pred"]].copy()
    wrong["pair"] = wrong["label"] + "->" + wrong["pred"]
    n = min(n, len(wrong))
    sizes = wrong["pair"].value_counts().to_dict()
    # proportional to pair size (largest remainder), at least 2 per pair where available
    share = {p: n * c / len(wrong) for p, c in sizes.items()}
    k = {p: min(sizes[p], max(2, int(share[p]))) for p in sizes}
    for p in sorted(sizes, key=lambda q: share[q] - int(share[q]), reverse=True) * 2:
        if sum(k.values()) >= n:
            break
        if k[p] < sizes[p]:
            k[p] += 1
    while sum(k.values()) > n:  # the minimum of 2 can overshoot: trim the largest pair
        big = max(k, key=k.get)
        k[big] -= 1
    parts = [g.sample(k[p], random_state=seed) for p, g in wrong.groupby("pair")]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def cmd_sample(args) -> None:
    run = best_run()
    print(f"best run by val Macro F1: {run}")
    val = load_splits()["val"]
    pred = pd.read_csv(RESULTS / "predictions" / f"{run}_val.csv", dtype=str, keep_default_na=False)
    s = misclassified_sample(val, pred)
    print(s["pair"].value_counts().to_dict())
    if args.limit:
        print("Dry run: no glosses requested.")
        return
    s.assign(run=run).to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    from label_llm import ANNOTATORS, GLOSS_SYSTEM, call_gemini, load_env, user_message, validate_glosses

    key = load_env()["GEMINI_API_KEY"]
    have = {}
    if GLOSS_JSONL.exists():
        have = {json.loads(l)["id"]: json.loads(l)["gloss"] for l in GLOSS_JSONL.read_text(encoding="utf-8").splitlines()}
    todo = s[~s["id"].isin(have)].assign(lang_check=False)
    for start in range(0, len(todo), 25):
        message, idmap = user_message(todo.iloc[start:start + 25])
        text, _ = call_gemini(ANNOTATORS["gemini"], key, GLOSS_SYSTEM, message)
        valid, errors = validate_glosses(text, idmap)
        with GLOSS_JSONL.open("a", encoding="utf-8") as f:
            for v in valid.values():
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"  glosses {len(valid)} ok, {len(errors)} missing")
        time.sleep(5)
    # bump the shared Gemini request counter that tracks the request budget
    counter = ROOT / "data" / "interim" / "synthetic" / "gemini_requests.json"
    if counter.exists():
        c = json.loads(counter.read_text())
        c["requests"] += -(-len(todo) // 25)
        counter.write_text(json.dumps(c))
    print(f"wrote {SAMPLE_CSV.relative_to(ROOT)} and {GLOSS_JSONL.relative_to(ROOT)}")


NB_CODE = r'''
import json, pandas as pd
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
import sys; sys.path.insert(0, str(ROOT / "src"))
from terralens.evaluate import confusion_df, per_class_report
sample = pd.read_csv(ROOT / "data/interim/error_sample.csv", dtype=str, keep_default_na=False)
RUN = sample["run"].iloc[0]
res = json.loads((ROOT / "results" / f"{RUN}.json").read_text(encoding="utf-8"))
pred = pd.read_csv(ROOT / "results/predictions" / f"{RUN}_val.csv", dtype=str, keep_default_na=False)
print("Best run by val Macro F1:", RUN, "| val Macro F1", res["val"]["macro_f1"], "| val Against F1", res["val"]["against_f1"])
'''


def summarise(sample: pd.DataFrame, categories: pd.DataFrame, run: str, n_val: int, n_wrong: int) -> dict:
    """Counts only: sampled pairs and hand-assigned categories of the sampled val errors."""
    cats = sample[["id"]].merge(categories, on="id", how="left")["category"].fillna("uncategorised")
    return {"experiment_id": "error_analysis", "script": "scripts/error_analysis.py", "timestamp": now(), "seed": SEED,
            "config": {"split": "val", "n_sample": N, "categories": "assigned by hand from English glosses"},
            "run": run, "val_comments": n_val, "val_errors": n_wrong, "sampled": len(sample),
            "pairs": {k: int(v) for k, v in sample["pair"].value_counts().items()},
            "categories": {k: int(v) for k, v in cats.value_counts().items()}}


def cmd_summary(args) -> None:
    sample = pd.read_csv(SAMPLE_CSV, dtype=str, keep_default_na=False)
    run = sample["run"].iloc[0]
    pred = pd.read_csv(RESULTS / "predictions" / f"{run}_val.csv", dtype=str, keep_default_na=False)
    cats = pd.read_csv(CATEGORIES_CSV, dtype=str, keep_default_na=False)
    out = summarise(sample, cats, run, len(pred), int((pred["gold"] != pred["pred"]).sum()))
    print(json.dumps({k: out[k] for k in ("run", "val_errors", "sampled", "categories")}, indent=1))
    if args.limit:
        print("Dry run: nothing written.")
        return
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("wrote results/error_analysis.json")


def cmd_notebook(args) -> None:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("# 03 Error analysis (validation only)\n\nDraft. The best run is chosen by **validation** Macro F1; "
           "test is not looked at here. Built by `scripts/error_analysis.py notebook`. Only English glosses "
           "(Gemini), labels and ids are shown; original comment text stays out of the notebook."),
        code(NB_CODE.strip()),
        md("## Confusion matrix and per-class scores (val)"),
        code("print(confusion_df(pred['gold'], pred['pred']))\nprint(per_class_report(pred['gold'], pred['pred']).round(3))"),
        md("## Macro F1 per language, script and annotator tag (val)"),
        code("for k in ('by_lang', 'by_script', 'by_tag'):\n    print(k); display(pd.DataFrame(res['val'][k]).T)"),
        md("## 40 misclassified validation comments\n\nSampled with seed 42, spread over gold -> prediction pairs. "
           "`category` is a hand-assigned error type (see `reports/error_categories.csv`)."),
        code("gl = {}\np = ROOT / 'data/interim/error_glosses.jsonl'\nfor line in p.read_text(encoding='utf-8').splitlines():\n"
             "    r = json.loads(line); gl[r['id']] = r['gloss']\n"
             "cat = pd.read_csv(ROOT / 'reports/error_categories.csv', dtype=str) if (ROOT / 'reports/error_categories.csv').exists() else pd.DataFrame(columns=['id','category'])\n"
             "view = sample.assign(gloss=sample['id'].map(gl)).merge(cat, on='id', how='left')\n"
             "pd.set_option('display.max_colwidth', 160)\n"
             "display(view[['label', 'pred', 'lang', 'script', 'tags', 'category', 'gloss']].rename(columns={'label': 'gold'}))"),
        md("## Error categories"),
        code("display(view.groupby('category').agg(n=('id', 'size'), pairs=('pair', lambda s: s.value_counts().to_dict())).sort_values('n', ascending=False))"),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NOTEBOOK)
    exe = Path(sys.executable).with_name("jupyter-nbconvert.exe")
    cmd = [str(exe) if exe.exists() else "jupyter-nbconvert", "--to", "notebook", "--execute", "--inplace", NOTEBOOK.name]
    subprocess.run(cmd, cwd=NOTEBOOK.parent, check=True)
    print(f"wrote and executed {NOTEBOOK.relative_to(ROOT)}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--limit", type=int, default=None)
    sub.add_parser("notebook")
    sm = sub.add_parser("summary")
    sm.add_argument("--limit", type=int, default=None, help="dry run: print, write nothing")
    args = p.parse_args(argv)
    {"sample": cmd_sample, "notebook": cmd_notebook, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()
