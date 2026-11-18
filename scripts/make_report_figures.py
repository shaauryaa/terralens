"""Figures for the Round 2 progress report, drawn from results/*.json only.

Every number shown in a figure is read from a results file at run time:
    results/comment_cleaning.json      raw comments and videos, comments kept after cleaning
    results/annotation_agreement.json  Cohen's kappa Claude vs Gemini
    results/dataset_stats.json         class counts per language and split, script mix
    results/e*_*.json                  test Macro F1 with 95% CI per system (via scripts/plot_results.py)
    results/e4_qlora.json              E4 test confusion matrix
    results/robustness_e4_qlora.json   E4 clean test, noisy test and challenge-set Macro F1

Figures are drawn at the size they are printed in the report (inches), so font sizes are real point sizes.
Class colours (Favour teal, Against orange, None grey) are the same in every chart.

Outputs (reports/figures/)
    report_pipeline.png       pipeline boxes with the key counts
    report_class_dist.png     class distribution per language and per split
    report_script_mix.png     script mix donut
    report_results.png        test Macro F1 with 95% CI per system (half width)
    report_confusion.png      E4 confusion matrix on test (half width)
    report_robustness.png     E4 clean vs noisy test, challenge set and its slices

Usage:
    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --limit 1     # dry run: print the numbers, draw nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
SEED = 42  # nothing is sampled here; kept for the script convention
DPI = 220

# Colour palettes, checked with the dataviz validator (lightness band, chroma floor, colour-blind separation and
# normal-vision separation all pass; every bar or slice also carries a direct label).
CLASS_COLORS = {"Favour": "#0fa07a", "Against": "#f0612a", "None": "#4f63e6"}  # green-teal / orange / indigo
TEAL, ORANGE, INDIGO = CLASS_COLORS["Favour"], CLASS_COLORS["Against"], CLASS_COLORS["None"]
VIOLET, CYAN, AMBER, MAGENTA = "#8a3ffc", "#0f9fbf", "#f08c00", "#e11d74"      # second palette (non-class charts)
ORANGE_TEXT = "#c94a17"  # darker orange for text on white
# Pipeline boxes: deeper shades so white text stays readable.
BOX_COLORS = ["#3f51d6", "#0b86a3", "#7a2fe8", "#0b8a68", "#e2541c", "#c8155f"]
SURFACE, INK, INK2, GRID = "#ffffff", "#111111", "#4d4d4d", "#e3e3e3"
LABELS = ["Favour", "Against", "None"]
LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}
SHORT_LABELS = {"e3_best_plus_bcd": "mmBERT + all extra data", "e1_gemma3_12b_0shot": "Gemma 3 12B, zero-shot",
                "e1_gemma3_12b_5shot": "Gemma 3 12B, 5-shot"}

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": ["Calibri"] if "Calibri" in _have else ["DejaVu Sans"],
    "font.size": 8, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
})


def load(name: str, results_dir: Path = RESULTS) -> dict:
    return json.loads((results_dir / f"{name}.json").read_text(encoding="utf-8"))


def fmt(n: int) -> str:
    return f"{n:,}"


def _plot_results_module():
    spec = importlib.util.spec_from_file_location("plot_results", ROOT / "scripts" / "plot_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_results"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- numbers (pure functions, tested)
def pipeline_numbers(cleaning: dict, agreement: dict, ds: dict) -> dict:
    split = ds["class_by_split"]
    return {"videos": cleaning["raw_videos"], "raw": cleaning["raw_comments"], "clean": cleaning["kept"],
            "kappa": agreement["claude_vs_gemini"]["overall"]["cohen_kappa"],
            "train": sum(split["train"].values()), "val": sum(split["val"].values()),
            "test": sum(split["test"].values())}


def script_mix_totals(ds: dict) -> dict:
    """Native script (Devanagari for Hindi, Bengali for Bengali), romanised (Latin) and mixed, summed over languages."""
    out = {"native": 0, "romanised": 0, "mixed": 0}
    for lang, counts in ds["script_by_lang"].items():
        out["native"] += counts.get("devanagari" if lang == "hi" else "bengali", 0)
        out["romanised"] += counts.get("latin", 0)
        out["mixed"] += counts.get("mixed", 0)
    return out


def class_rows(ds: dict) -> list[tuple[str, dict]]:
    rows = [(LANG_NAME[k], ds["class_by_lang"][k]) for k in ("hi", "bn")]
    rows += [(k, ds["class_by_split"][k]) for k in ("train", "val", "test")]
    return rows


def robustness_bars(rob: dict) -> list[tuple[str, float, str]]:
    run = rob["plain"]["runs"][0]
    ph = run["challenge_by_phenomenon"]
    return [("Clean test", run["test_macro_f1"], "test"), ("Noisy test", run["noisy_macro_f1"], "test"),
            ("Challenge set", run["challenge_macro_f1"], "challenge"),
            ("Negation", ph["negation"]["macro_f1"], "slice"), ("Sarcasm", ph["sarcasm"]["macro_f1"], "slice"),
            ("Implicit", ph["implicit"]["macro_f1"], "slice"),
            ("Topic, not target", ph["topic_not_target"]["macro_f1"], "slice")]


# ---------------------------------------------------------------- figures
def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _clean_axes(ax, keep=("bottom",)) -> None:
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)
    ax.spines["bottom"].set_color(GRID)


def fig_pipeline(nums: dict, out_dir: Path = FIG_DIR) -> Path:
    boxes = [
        (BOX_COLORS[0], "YouTube API", [f"{nums['videos']} videos", f"{fmt(nums['raw'])} comments"]),
        (BOX_COLORS[1], "Clean & dedup", [f"{fmt(nums['clean'])} comments kept"]),
        (BOX_COLORS[2], "3-LLM labelling", ["Claude + Gemini,", "gpt-oss adjudicates", f"kappa {nums['kappa']:.3f}"]),
        (BOX_COLORS[3], "Video-grouped split", [f"{fmt(nums['train'])} / {nums['val']} / {nums['test']}",
                                                "train / val / test"]),
        (BOX_COLORS[4], "4 model families", ["Prompted LLMs,", "Encoders,", "+Extra data, QLoRA"]),
        (BOX_COLORS[5], "Evaluation", ["Macro F1, 95% CI,", "robustness"]),
    ]
    W, H = 7.1, 0.86
    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    n, gap = len(boxes), 0.17
    bw = (W - 0.04 - gap * (n - 1)) / n
    y0, bh = 0.03, 0.8
    for i, (color, title, lines) in enumerate(boxes):
        x = 0.02 + i * (bw + gap)
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=0.07", fc=color, ec="none"))
        ax.text(x + bw / 2, y0 + bh - 0.17, title, ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold")
        for j, line in enumerate(lines):
            ax.text(x + bw / 2, y0 + bh - 0.38 - j * 0.15, line, ha="center", va="center", color="white", fontsize=7.6)
        if i < n - 1:
            ax.annotate("", xy=(x + bw + gap - 0.015, y0 + bh / 2), xytext=(x + bw + 0.015, y0 + bh / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1, mutation_scale=8))
    return _save(fig, "report_pipeline.png", out_dir)


def fig_class_dist(ds: dict, out_dir: Path = FIG_DIR) -> Path:
    rows = class_rows(ds)
    fig, ax = plt.subplots(figsize=(4.15, 1.8), facecolor=SURFACE)
    ys = [5.6, 4.6, 2.8, 1.8, 0.8]  # gap between the language rows and the split rows
    for y, (name, counts) in zip(ys, rows):
        total = sum(counts.values())
        left = 0.0
        for lab in LABELS:
            share = counts[lab] / total
            ax.barh(y, share, left=left, height=0.74, color=CLASS_COLORS[lab], edgecolor=SURFACE, linewidth=1)
            if share > 0.12:
                ax.text(left + share / 2, y, fmt(counts[lab]), ha="center", va="center", fontsize=7.6,
                        color="white", fontweight="bold")
            left += share
        ax.text(1.02, y, f"{counts['Against']} Against", va="center", fontsize=7.6, color=ORANGE_TEXT,
                fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.text(-0.02, 6.3, "language", ha="right", fontsize=7, color=INK2, style="italic")
    ax.text(-0.02, 3.5, "split", ha="right", fontsize=7, color=INK2, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.25, 6.55)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
    ax.tick_params(axis="y", length=0)
    _clean_axes(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[lab]) for lab in LABELS]
    ax.legend(handles, LABELS, ncols=3, frameon=False, fontsize=7.6, loc="lower left", bbox_to_anchor=(0, 1.0),
              handlelength=1, handleheight=0.8, columnspacing=1, borderaxespad=0)
    ax.set_title("Classes per language and split", pad=16)
    fig.subplots_adjust(left=0.16, right=0.79, top=0.8, bottom=0.12)
    return _save(fig, "report_class_dist.png", out_dir)


def fig_script_mix(ds: dict, out_dir: Path = FIG_DIR) -> Path:
    tot = script_mix_totals(ds)
    vals = [tot["native"], tot["romanised"], tot["mixed"]]
    names = ["Native script", "Romanised", "Mixed"]
    colors = [CYAN, AMBER, VIOLET]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(2.95, 1.8), facecolor=SURFACE)
    wedges, _ = ax.pie(vals, colors=colors, startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=1.5))
    for w, v in zip(wedges, vals):
        pct = f"{100 * v / total:.0f}%" if v / total >= 0.05 else f"{100 * v / total:.1f}%"
        ang = np.deg2rad((w.theta2 + w.theta1) / 2)
        big = v / total >= 0.05
        r = 0.79 if big else 1.13
        ax.text(r * np.cos(ang) + (0 if big else 0.1), r * np.sin(ang), pct, ha="center" if big else "left",
                va="center", fontsize=8, fontweight="bold", color="white" if big else VIOLET)
    ax.legend(wedges, names, frameon=False, fontsize=8.2,
              loc="center left", bbox_to_anchor=(1.02, 0.5), handlelength=0.9, labelspacing=0.8)
    ax.set_title("Script of the comments", pad=10)
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.04, right=0.55, top=0.8, bottom=0.04)
    return _save(fig, "report_script_mix.png", out_dir)


def fig_results(out_dir: Path = FIG_DIR, results_dir: Path = RESULTS) -> Path:
    pr = _plot_results_module()
    rows = pr.collect_systems(results_dir)
    return pr.plot(rows, out_dir / "report_results.png", width=3.85, values=False, labels=SHORT_LABELS,
                   title="Test Macro F1 per system", fs=7.8, row_h=0.15)


def fig_confusion(e4: dict, out_dir: Path = FIG_DIR) -> Path:
    test = e4["test"]
    cm = test["confusion_matrix"]["matrix"]
    order = test["confusion_matrix"]["rows_gold_cols_pred"]
    share = [[v / sum(row) for v in row] for row in cm]  # shade by share of the gold row (recall on the diagonal)
    fig, ax = plt.subplots(figsize=(3.1, 2.6), facecolor=SURFACE)
    ramp = LinearSegmentedColormap.from_list("indigo", ["#f3f5ff", "#a9b4f5", INDIGO, "#2b37a8"])
    ax.imshow(share, cmap=ramp, vmin=0, vmax=1)
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9.5,
                    color="white" if share[i][j] > 0.5 else INK, fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(3))
    ax.set_xticklabels(order, fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("predicted", fontsize=7.8)
    ax.set_ylabel("true label", fontsize=7.8)
    ax.set_title("E4 confusion matrix (test)")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.subplots_adjust(left=0.24, right=0.97, top=0.88, bottom=0.17)
    return _save(fig, "report_confusion.png", out_dir)


def fig_robustness(rob: dict, out_dir: Path = FIG_DIR) -> Path:
    bars = robustness_bars(rob)
    # test sets in indigo shades, challenge set in green-teal, each challenge slice in its own colour
    colors = [INDIGO, "#8f9cf0", TEAL, VIOLET, MAGENTA, AMBER, CYAN]
    fig, ax = plt.subplots(figsize=(7.1, 1.3), facecolor=SURFACE)
    xs = list(range(len(bars)))
    for x, (name, v, kind), c in zip(xs, bars, colors):
        ax.bar(x, v, width=0.62, color=c)
        ax.text(x, v + 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8, color=INK)
    ax.axvline(2.5, color=GRID, linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="y", labelsize=7, colors=INK2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Macro F1", fontsize=7.6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _clean_axes(ax)
    ax.text(3.0, 1.1, "challenge set by phenomenon", fontsize=7.2, color=INK2, style="italic", va="center")
    ax.set_title("E4 robustness: clean vs noisy test and the challenge set (retrained run)", pad=4)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.84, bottom=0.17)
    return _save(fig, "report_robustness.png", out_dir)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=FIG_DIR)
    p.add_argument("--limit", type=int, default=None, help="dry run: print the numbers, draw nothing")
    args = p.parse_args(argv)
    ds, e4, rob = load("dataset_stats"), load("e4_qlora"), load("robustness_e4_qlora")
    nums = pipeline_numbers(load("comment_cleaning"), load("annotation_agreement"), ds)
    if args.limit:
        print("pipeline", nums)
        print("script mix", script_mix_totals(ds))
        print("robustness", robustness_bars(rob))
        print("dry run, nothing drawn")
        return
    paths = [fig_pipeline(nums, args.out_dir), fig_class_dist(ds, args.out_dir), fig_script_mix(ds, args.out_dir),
             fig_results(args.out_dir), fig_confusion(e4, args.out_dir), fig_robustness(rob, args.out_dir)]
    for path in paths:
        print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
