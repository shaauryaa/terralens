"""Plot test Macro F1 with its bootstrap 95% CI for every system, from results/*.json only.

One row per system (the seed-42 run of each experiment in results/: E0 majority floor, E1 prompted LLMs, E2/E3
encoders, E4 QLoRA). Seed reruns (``_s43``, ``_s44``), robustness reruns (``_robust``) and dry runs are left out,
as in the main table of reports/results.md. The system with the highest **validation** Macro F1 is marked; test
is only displayed, never used to pick anything. No randomness is involved (the CIs are read from the result
files, which used seed 42 for the bootstrap).

Output: reports/figures/results_macro_f1.png

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --limit 3     # dry run: print the first 3 rows, write nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_PNG = ROOT / "reports" / "figures" / "results_macro_f1.png"
SEED = 42  # nothing is sampled here; kept for the script convention

SYSTEM_KINDS = {"majority", "llm_prompt", "encoder", "qlora"}
SKIP_SUFFIXES = ("_s43", "_s44", "_robust")
ORDER = ["e0_majority", "e1_qwen3_8b_0shot", "e1_qwen3_8b_5shot", "e1_gemma3_12b_0shot", "e1_gemma3_12b_5shot",
         "e2_xlmr", "e2_muril", "e2_indicbertv2", "e2_mmbert", "e3_best_plus_b", "e3_best_plus_bcd", "e4_qlora"]
SHORT = {
    "e0_majority": "Majority class (floor)",
    "e1_qwen3_8b_0shot": "Qwen3-8B, zero-shot",
    "e1_qwen3_8b_5shot": "Qwen3-8B, 5-shot",
    "e1_gemma3_12b_0shot": "Gemma 3 12B, zero-shot",
    "e1_gemma3_12b_5shot": "Gemma 3 12B, 5-shot",
    "e2_xlmr": "XLM-R base",
    "e2_muril": "MuRIL base",
    "e2_indicbertv2": "IndicBERT v2",
    "e2_mmbert": "mmBERT base",
    "e3_best_plus_b": "mmBERT + translated",
    "e3_best_plus_bcd": "mmBERT + transl. + roman. + synth.",
    "e4_qlora": "QLoRA Qwen3-4B",
}

# Reference palette (dataviz skill), same tokens as scripts/dataset_stats.py.
SERIES, FLOOR = "#2a78d6", "#8c8b87"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def collect_systems(results_dir: Path = RESULTS) -> list[dict]:
    """One dict per system: exp_id, group (E0..E4), label, val and test Macro F1, test CI. Ordered E0 -> E4."""
    rows = []
    for p in results_dir.glob("*.json"):
        if p.stem.endswith(SKIP_SUFFIXES) or "dryrun" in p.stem:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        test = r.get("test")
        if r.get("kind") not in SYSTEM_KINDS or not isinstance(test, dict) or "ci95" not in test:
            continue
        lo, hi = test["ci95"]["macro_f1"]
        rows.append({"exp_id": p.stem, "group": p.stem.split("_")[0].upper(), "label": SHORT.get(p.stem, p.stem),
                     "val_macro_f1": r["val"]["macro_f1"], "test_macro_f1": test["macro_f1"], "ci_low": lo,
                     "ci_high": hi})
    rank = {e: i for i, e in enumerate(ORDER)}
    rows.sort(key=lambda d: (rank.get(d["exp_id"], len(ORDER)), d["exp_id"]))
    return rows


def best_by_val(rows: list[dict]) -> str | None:
    """exp_id with the highest validation Macro F1 (the E0 floor is never a candidate)."""
    cands = [d for d in rows if d["group"] != "E0"]
    return max(cands, key=lambda d: d["val_macro_f1"])["exp_id"] if cands else None


def plot(rows: list[dict], out: Path = OUT_PNG) -> Path:
    best = best_by_val(rows)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * n + 1.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ys = list(range(n))[::-1]  # first row at the top
    for y, d in zip(ys, rows):
        color = FLOOR if d["group"] == "E0" else SERIES
        ax.plot([d["ci_low"], d["ci_high"]], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        is_best = d["exp_id"] == best
        ax.plot(d["test_macro_f1"], y, "o", markersize=9 if is_best else 7, color=color,
                markeredgecolor=INK if is_best else SURFACE, markeredgewidth=1.6 if is_best else 2, zorder=3)
        ax.annotate(f"{d['test_macro_f1']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]", (1.0, y),
                    xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2)
    labels = [f"{d['group']}  {d['label']}" + ("  (best on val)" if d["exp_id"] == best else "") for d in rows]
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    for i in range(1, n):  # thin rule between experiment groups
        if rows[i]["group"] != rows[i - 1]["group"]:
            ax.axhline(ys[i] + 0.5, color=GRID, linewidth=0.8, zorder=1)
    ax.set_xlim(0.2, 0.8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Test Macro F1 (dot) with bootstrap 95% CI (line), 538 test comments", color=INK2, fontsize=8.5)
    ax.set_title("Test Macro F1 per system", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT_PNG)
    p.add_argument("--limit", type=int, default=None, help="dry run: print the first N rows, write nothing")
    args = p.parse_args(argv)
    rows = collect_systems()
    if not rows:
        raise SystemExit("no system results in results/")
    if args.limit:
        for d in rows[: args.limit]:
            print(f"{d['group']} {d['exp_id']}: val {d['val_macro_f1']:.3f}, test {d['test_macro_f1']:.3f} "
                  f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
        print(f"best on val: {best_by_val(rows)}; dry run, nothing written")
        return
    out = plot(rows, args.out)
    print(f"wrote {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out} ({len(rows)} systems)")


if __name__ == "__main__":
    main()
