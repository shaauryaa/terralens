"""Build reports/results.md from results/*.json only (no other source of numbers).

Tables
    1. Main results: val and test Macro F1 / Against F1 (test with bootstrap 95% CI), accuracy, and test with
       val-tuned logit offsets for encoders. Model selection is by val Macro F1 only.
    2. Test Macro F1 per language and per script.
    3. Seed variance: mean ± std over seeds 42/43/44 where all three exist.
    4. Robustness: noisy test copy and challenge set.
    5. Data and annotation numbers (splits, agreement, synthetic data, near-duplicate check).
Experiments that are planned but have no result file are listed as "pending".

Usage:
    python scripts/make_results_table.py
    python scripts/make_results_table.py --limit 3     # dry run: print the first 3 rows of table 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_MD = ROOT / "reports" / "results.md"

PLANNED = [
    ("e0_majority", "E0", "Majority class (floor)"),
    ("e1_qwen3_8b_0shot", "E1", "Qwen3-8B 4-bit, zero-shot"),
    ("e1_qwen3_8b_5shot", "E1", "Qwen3-8B 4-bit, 5-shot (train examples)"),
    ("e1_gemma3_12b_0shot", "E1", "Gemma 3 12B 4-bit, zero-shot"),
    ("e1_gemma3_12b_5shot", "E1", "Gemma 3 12B 4-bit, 5-shot (train examples)"),
    ("e2_xlmr", "E2", "XLM-R base, Source A train"),
    ("e2_muril", "E2", "MuRIL base, Source A train"),
    ("e2_indicbertv2", "E2", "IndicBERT v2, Source A train"),
    ("e2_mmbert", "E2", "mmBERT base, Source A train"),
    ("e3_best_plus_b", "E3", "Best E2 encoder + translated (B)"),
    ("e3_best_plus_bcd", "E3", "Best E2 encoder + translated, romanised, synthetic (B+C+D)"),
    ("e4_qlora", "E4", "QLoRA 4B instruction model (3-way classifier)"),
]


def load_all() -> dict[str, dict]:
    out = {}
    for path in sorted(RESULTS.glob("*.json")):
        if "dryrun" in path.stem:
            continue
        try:
            out[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    return out


def f(x, nd=3) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def ci(m: dict, key: str = "macro_f1") -> str:
    c = (m.get("ci95") or {}).get(key)
    return f" [{c[0]:.3f}, {c[1]:.3f}]" if c and c[0] is not None else ""


def model_name(r: dict) -> str:
    cfg = r.get("config", {})
    return cfg.get("model_name") or cfg.get("rule", "")


def main_rows(res: dict) -> list[list[str]]:
    rows = []
    seen = set()
    planned_ids = [p[0] for p in PLANNED]
    extra = sorted(k for k, r in res.items()
                   if r.get("kind") in ("encoder", "llm_prompt", "qlora", "majority") and k not in planned_ids
                   and not k.endswith(("_s43", "_s44", "_robust")))
    for exp_id, tag, desc in PLANNED + [(k, k.split("_")[0].upper(), "") for k in extra]:
        if exp_id in seen:
            continue
        seen.add(exp_id)
        r = res.get(exp_id)
        if r is None:
            rows.append([tag, exp_id, desc, "pending"] + [""] * 6)
            continue
        v, t = r["val"], r["test"]
        tuned = (r.get("thresholds") or {}).get("test_tuned")
        fails = ""
        if r.get("kind") == "llm_prompt":
            fails = f" (parse fail val {v.get('parse_failures', 0)}, test {t.get('parse_failures', 0)})"
        rows.append([tag, exp_id, (desc or model_name(r)) + fails, model_name(r), f(v["macro_f1"]), f(v["against_f1"]),
                     f(t["macro_f1"]) + ci(t), f(t["against_f1"]) + ci(t, "against_f1"), f(t.get("accuracy")),
                     (f"{f(tuned['macro_f1'])} / {f(tuned['against_f1'])}" if tuned else "n/a")])
    return rows


def table(header: list[str], rows: list[list[str]]) -> str:
    out = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return out + "".join("| " + " | ".join(r) + " |\n" for r in rows)


def best_by_val(res: dict, prefix: str) -> str | None:
    cands = {k: r for k, r in res.items() if k.startswith(prefix) and r.get("kind") == "encoder"
             and not k.endswith(("_s43", "_s44", "_robust"))}
    return max(cands, key=lambda k: cands[k]["val"]["macro_f1"]) if cands else None


def seed_rows(res: dict) -> list[list[str]]:
    rows = []
    for base, r in res.items():
        if r.get("kind") != "encoder" or base.endswith(("_s43", "_s44", "_robust")):
            continue
        runs = [res.get(base), res.get(f"{base}_s43"), res.get(f"{base}_s44")]
        if any(x is None for x in runs):
            continue
        cells = []
        for split, key in (("val", "macro_f1"), ("test", "macro_f1"), ("test", "against_f1")):
            vals = np.array([x[split][key] for x in runs], dtype=float)
            cells.append(f"{vals.mean():.3f} ± {vals.std(ddof=1):.3f}")
        tuned = np.array([x["thresholds"]["test_tuned"]["macro_f1"] for x in runs], dtype=float)
        rows.append([base, "42, 43, 44", *cells, f"{tuned.mean():.3f} ± {tuned.std(ddof=1):.3f}"])
    return rows


def lang_rows(res: dict) -> list[list[str]]:
    rows = []
    for exp_id, _, _ in PLANNED:
        r = res.get(exp_id)
        if not r:
            continue
        t = r["test"]
        bl, bs = t.get("by_lang", {}), t.get("by_script", {})
        get = lambda d, k: f(d.get(k, {}).get("macro_f1")) if k in d else "–"  # noqa: E731
        rows.append([exp_id, get(bl, "hi"), get(bl, "bn"), get(bs, "devanagari"), get(bs, "bengali"),
                     get(bs, "latin"), get(bs, "mixed")])
    return rows


def robustness_section(res: dict) -> str:
    parts = []
    for k, r in sorted(res.items()):
        if r.get("kind") != "robustness":
            continue
        ms = r["plain"]["mean_std"]
        g = lambda key: (f"{ms[key]['mean']:.3f} ± {ms[key]['std']:.3f}" if ms.get(key) else "–")  # noqa: E731
        n = (ms.get("test_macro_f1") or {}).get("n_runs", 0)
        parts.append(f"**{k}** ({n} runs: {', '.join(x['experiment_id'] for x in r['plain']['runs'])})\n\n"
                     + table(["Clean test Macro F1", "Noisy test Macro F1", "Drop", "Clean test Against F1",
                              "Noisy test Against F1", "Challenge Macro F1", "Challenge Against F1"],
                             [[g("test_macro_f1"), g("noisy_macro_f1"), g("drop_macro_f1"), g("test_against_f1"),
                               g("noisy_against_f1"), g("challenge_macro_f1"), g("challenge_against_f1")]])
                     + "\nChallenge Macro F1 per phenomenon: "
                     + ", ".join(f"{p} {v['mean']:.3f}" for p, v in ms.get("challenge_by_phenomenon", {}).items() if v)
                     + "\n")
    return "\n".join(parts) if parts else "pending (no `results/robustness_*.json` yet)\n"


def data_section(res: dict) -> str:
    lines = []
    sp = res.get("splits")
    if sp:
        s = sp["splits"]
        lines.append(f"- Splits (seed {sp['seed']}, grouped by video): " + "; ".join(
            f"{k} {v['n']} ({v['label']['Favour']} F / {v['label']['Against']} A / {v['label']['None']} N, "
            f"{v['videos']} videos)" for k, v in s.items()))
    ag = res.get("annotation_agreement")
    if ag:
        o = ag["claude_vs_gemini"]["overall"]
        fl = ag.get("fleiss_kappa_sample", {})
        lines.append(f"- Labels: Cohen's kappa Claude vs Gemini {o['cohen_kappa']} ({o['pct_agreement']}% agreement, "
                     f"n = {o['n']}); Fleiss' kappa with gpt-oss on the 300 sample {fl.get('kappa')} "
                     f"(n = {fl.get('n')}; sample drawn from Claude-Gemini agreements, so biased upward); "
                     f"{ag['adjudication']['three_way_splits']} 3-way splits dropped.")
    sy = res.get("synthetic_data")
    if sy:
        for k in ("synthetic", "challenge"):
            if k in sy:
                d = sy[k]
                lines.append(f"- {k.title()} comments: {d['generated']} generated by Gemini, {d['agree']} kept where "
                             f"Claude and Gemini agree ({d['agreement_pct']}%, kappa {d.get('cohen_kappa_claude_gemini')}); "
                             f"labels {d['label']}.")
    tr = res.get("translation_set")
    if tr:
        lines.append(f"- Translation subset: {tr['total']} English rows (each translated into one language): "
                     f"hi {tr['selected']['hi']}, bn {tr['selected']['bn']}. ftfy repaired mojibake in "
                     f"{tr['kaggle_mojibake']['rows_with_mojibake_before'] - tr['kaggle_mojibake']['rows_with_mojibake_after']}"
                     f" of {tr['kaggle_mojibake']['rows_with_mojibake_before']} affected Kaggle rows.")
    nd = res.get("near_dup_check")
    if nd:
        lines.append("- Near-duplicate check vs val/test/challenge: " + "; ".join(
            f"{k}: {v['dropped']} of {v['rows']} dropped" if isinstance(v, dict) else f"{k}: {v}"
            for k, v in nd["files"].items()))
    return "\n".join(lines) + "\n" if lines else "pending\n"


def build(res: dict) -> str:
    best_e2 = best_by_val(res, "e2_")
    best_all = best_by_val(res, "e")
    md = ["# Results", "",
          "Generated by `scripts/make_results_table.py` from `results/*.json` only. Validation (522 comments) is used "
          "for every choice (epochs, offsets, best model); test (538 comments) is scored once per configuration. "
          "95% CIs: 1,000 bootstrap resamples of the test comments.", "",
          "## 1. Main results", "",
          table(["Exp", "ID", "Description", "Model", "Val Macro F1", "Val Against F1", "Test Macro F1 [95% CI]",
                 "Test Against F1 [95% CI]", "Test acc.", "Test with val-tuned offsets (Macro / Against)"],
                main_rows(res)),
          f"Best E2 encoder by val Macro F1: **{best_e2 or 'pending'}**. Best encoder overall by val Macro F1: "
          f"**{best_all or 'pending'}**.", "",
          "## 2. Test Macro F1 per language and script", "",
          table(["ID", "Hindi", "Bengali", "Devanagari", "Bengali script", "Latin", "Mixed"], lang_rows(res)),
          "Bengali test has only 4 Against comments, so per-language numbers for Bengali are noisy.", "",
          "## 3. Seed variance (best configuration)", "",
          (table(["Config", "Seeds", "Val Macro F1", "Test Macro F1", "Test Against F1", "Test Macro F1 (tuned offsets)"],
                 seed_rows(res)) if seed_rows(res) else "pending\n"),
          "## 4. Robustness", "", robustness_section(res),
          "## 5. Data and annotation", "", data_section(res)]
    return "\n".join(md)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="dry run: print the first N rows of table 1")
    args = p.parse_args(argv)
    res = load_all()
    if args.limit:
        for r in main_rows(res)[: args.limit]:
            print(r)
        return
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(build(res), encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(ROOT)} from {len(res)} result files")


if __name__ == "__main__":
    sys.exit(main())
