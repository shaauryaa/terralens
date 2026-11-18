"""Dataset statistics for TerraLens (T6): counts, class balance, scripts, lengths, top words, agreement.

Reads data/processed/source_a.csv, data/splits/*_ids.txt, the external CSVs (training-only sources) and the
results JSON files written by earlier steps. Writes
    results/dataset_stats.json         every number shown in the report
    reports/dataset_stats.md           a short summary with tables
    reports/figures/ds_*.png           figures

Usage:
    python scripts/dataset_stats.py
    python scripts/dataset_stats.py --limit 300     # dry run on 300 Source A rows, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import regex as re  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402

SOURCE_A = ROOT / "data" / "processed" / "source_a.csv"
SPLIT_DIR = ROOT / "data" / "splits"
EXTERNAL = ROOT / "data" / "external"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_MD = ROOT / "reports" / "dataset_stats.md"
STATS_JSON = RESULTS / "dataset_stats.json"

# Reference palette (dataviz skill), first three categorical slots: validated all-pairs for CVD.
COLORS = {"Favour": "#2a78d6", "Against": "#eb6834", "None": "#1baf7a"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
LANG_NAME = {"hi": "Hindi", "bn": "Bengali"}

# Small hand-made stopword lists (function words, pronouns, particles) for Hindi, Bengali, romanised forms and
# English. Good enough for "top words per class"; not a linguistic resource.
STOPWORDS = set("""
का की के को में है हैं था थी थे से पर और भी तो ही ये यह वह वो इस उस एक नहीं ना न कि जो क्या कर करना करते
हो होता होती होगा रहा रही रहे गया गई गए दिया लिए साथ अब जब तक हम आप तुम मैं मेरा मेरी हमारा हमारी उनका उनकी
इसका इसकी सब कुछ बहुत लेकिन या फिर कोई अपने अपना अपनी वाले वाली जा जाता जाती सकता सकते
আমি আমার আমরা তুমি আপনি সে তার তাদের এই সেই ও এবং আর কি কী না নেই হয় হবে ছিল করে করা করতে থেকে দিয়ে
জন্য একটা একটি এক যে যা কোন কিছু সব খুব তো ই এ কে র এর তা আছে হয়ে হচ্ছে গেছে বলে কিন্তু
ka ki ke ko me mein hai hain tha thi the se par aur bhi to hi ye yeh woh wo is us ek nahi na ki jo kya kar
karna karte ho hota hoti hoga raha rahi rahe gaya liye sath ab jab tak hum aap tum main mera meri hamara
sab kuch bahut lekin ya phir koi apne apna apni wale wali ja jata
ami amar amra tumi apni se tar tader ei sei o ebong ar ki kী na nei hoy hobe chilo kore kora korte theke
diye jonno ekta ek je ja kono kichu sob khub to e ke r er ta ache hoye hocche geche bole kintu
the a an is are was were be to of and in on for it this that with as at by from or not no but so we you
they he she i my our your their its have has had do does did will would can could should just all
""".split())
TOKEN_RE = re.compile(r"[\p{L}\p{M}]+")


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(str(text))]


def top_words(df: pd.DataFrame, k: int = 15) -> dict:
    """Top k non-stopword tokens (len >= 2, not emoji names) per label."""
    out = {}
    for lab in LABELS:
        c = Counter()
        for text in df.loc[df["label"] == lab, "text"]:
            c.update(t for t in tokens(re.sub(r":[a-z_]+:", " ", text)) if t not in STOPWORDS and len(t) >= 2)
        out[lab] = c.most_common(k)
    return out


def load_split(df: pd.DataFrame) -> pd.Series:
    split = pd.Series("unassigned", index=df.index)
    for name in ("train", "val", "test"):
        path = SPLIT_DIR / f"{name}_ids.txt"
        if path.exists():
            ids = set(path.read_text(encoding="utf-8").split())
            split[df["id"].isin(ids)] = name
    return split


def read_json(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def source_counts() -> dict:
    """Rows per source: A (real YouTube), B/C/D (training-only) if their files exist."""
    out = {}
    if SOURCE_A.exists():
        out["A: YouTube comments (labelled)"] = len(pd.read_csv(SOURCE_A, usecols=["id"]))
    ext = read_json("external_datasets.json").get("datasets", {})
    for name, info in ext.items():
        if info.get("status") == "built" and info.get("rows_kept") and name != "climatestance_tweet_ids":
            out[f"English external: {name}"] = int(info["rows_kept"])
    for label, fname in (("B: translated (hi/bn)", "translated.csv"), ("C: romanised", "romanised.csv"),
                         ("D: synthetic", "synthetic.csv")):
        path = PROCESSED / fname
        out[label] = len(pd.read_csv(path, usecols=["id"])) if path.exists() else 0
    return out


def class_table(df: pd.DataFrame, by: str | None) -> dict:
    if by is None:
        return {lab: int((df["label"] == lab).sum()) for lab in LABELS}
    return {str(k): {lab: int((g["label"] == lab).sum()) for lab in LABELS} for k, g in df.groupby(by)}


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def grouped_bars(table: dict, title: str, path: Path, ylabel: str = "comments") -> None:
    groups = list(table)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=SURFACE)
    width = 0.8 / len(LABELS)
    for i, lab in enumerate(LABELS):
        xs = [g + (i - 1) * width for g in range(len(groups))]
        vals = [table[g][lab] for g in groups]
        bars = ax.bar(xs, vals, width * 0.92, color=COLORS[lab], label=lab, edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=7.5,
                        color=INK2, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    _style(ax, title)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, ncols=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def hbar(values: dict, title: str, path: Path, color: str = COLORS["Favour"]) -> None:
    names, vals = list(values)[::-1], list(values.values())[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.35 * len(names) + 1.0), facecolor=SURFACE)
    bars = ax.barh(names, vals, color=color, height=0.6)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:,}", (v, b.get_y() + b.get_height() / 2), va="center", fontsize=8, color=INK2,
                    xytext=(3, 0), textcoords="offset points")
    _style(ax, title)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def script_mix_fig(df: pd.DataFrame, path: Path) -> dict:
    tab = pd.crosstab(df["lang"].map(LANG_NAME), df["script"])
    fig, ax = plt.subplots(figsize=(6.4, 2.6), facecolor=SURFACE)
    left = pd.Series(0, index=tab.index)
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for i, col in enumerate(tab.columns):
        ax.barh(tab.index, tab[col], left=left, color=palette[i % len(palette)], label=col, height=0.55,
                edgecolor=SURFACE, linewidth=2)
        left += tab[col]
    _style(ax, "Script mix per language (Source A)")
    ax.yaxis.grid(False)
    ax.legend(frameon=False, fontsize=8, ncols=len(tab.columns), loc="upper center", bbox_to_anchor=(0.5, -0.18),
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return {str(k): {str(c): int(v) for c, v in row.items()} for k, row in tab.iterrows()}


def length_fig(df: pd.DataFrame, path: Path) -> dict:
    n_tok = df["text"].map(lambda t: len(str(t).split()))
    fig, ax = plt.subplots(figsize=(6.4, 3.2), facecolor=SURFACE)
    for lang, color in (("hi", COLORS["Favour"]), ("bn", COLORS["Against"])):
        ax.hist(n_tok[df["lang"] == lang].clip(upper=100), bins=40, alpha=0.55, color=color,
                label=LANG_NAME[lang])
    ax.set_xlabel("whitespace tokens per comment (clipped at 100)", color=INK2, fontsize=9)
    ax.set_ylabel("comments", color=INK2, fontsize=9)
    _style(ax, "Comment length (Source A)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    out = {}
    for lang, g in n_tok.groupby(df["lang"]):
        out[lang] = {"mean": round(float(g.mean()), 1), "median": float(g.median()),
                     "p95": float(g.quantile(0.95)), "max": int(g.max())}
    out["share_over_128_tokens"] = round(float((n_tok > 128).mean()), 4)
    return out


def compute(df: pd.DataFrame) -> dict:
    df = df.assign(split=load_split(df))
    agreement = read_json("annotation_agreement.json")
    cleaning = read_json("comment_cleaning.json")
    stats = {
        "experiment_id": "dataset_stats",
        "script": "scripts/dataset_stats.py",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_a_rows": len(df),
        "videos": int(df["video_id"].nunique()),
        "source_counts": source_counts(),
        "class_overall": class_table(df, None),
        "class_by_lang": class_table(df, "lang"),
        "class_by_split": class_table(df[df["split"] != "unassigned"], "split"),
        "class_by_split_lang": {f"{s}/{l}": v for (s, l), v in
                                ((tuple(k.split("|")), v) for k, v in
                                 class_table(df.assign(sl=df["split"] + "|" + df["lang"]), "sl").items())},
        "script_by_lang": {str(k): {str(c): int(v) for c, v in r.items()}
                           for k, r in pd.crosstab(df["lang"], df["script"]).iterrows()},
        "videos_by_split": {str(k): int(v) for k, v in df.groupby("split")["video_id"].nunique().items()},
        "top_words": top_words(df),
        "cleaning_steps": [{"step": s["step"], "removed": s["removed"]} for s in cleaning.get("steps", [])],
        "agreement": {
            "cohen_kappa_claude_gemini": agreement.get("claude_vs_gemini", {}).get("overall"),
            "by_lang": agreement.get("claude_vs_gemini", {}).get("by_lang"),
            "by_script": agreement.get("claude_vs_gemini", {}).get("by_script"),
            "adjudication": agreement.get("adjudication"),
            "fleiss_kappa_sample": agreement.get("fleiss_kappa_sample"),
        },
    }
    return stats


def write_figures(df: pd.DataFrame, stats: dict) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = []

    def out(name):
        figs.append(f"figures/{name}")
        return FIG_DIR / name

    hbar({k: v for k, v in stats["source_counts"].items() if v}, "Rows per data source", out("ds_sources.png"))
    grouped_bars({"overall": stats["class_overall"],
                  **{LANG_NAME.get(k, k): v for k, v in stats["class_by_lang"].items()}},
                 "Class distribution, Source A (overall and per language)", out("ds_class_lang.png"))
    if stats["class_by_split"]:
        grouped_bars({k: stats["class_by_split"][k] for k in ("train", "val", "test") if k in stats["class_by_split"]},
                     "Class distribution per split", out("ds_class_split.png"))
    stats["script_mix"] = script_mix_fig(df, out("ds_script_mix.png"))
    stats["length_tokens"] = length_fig(df, out("ds_length.png"))
    return figs


def markdown(stats: dict, figs: list[str]) -> str:
    def table(rows: dict, first: str) -> str:
        head = f"| {first} | " + " | ".join(LABELS) + " | Total |\n|---|" + "---:|" * (len(LABELS) + 1) + "\n"
        return head + "".join(f"| {k} | " + " | ".join(f"{v[l]:,}" for l in LABELS) + f" | {sum(v.values()):,} |\n"
                              for k, v in rows.items())

    ag = stats["agreement"]
    ov = ag.get("cohen_kappa_claude_gemini") or {}
    fl = ag.get("fleiss_kappa_sample") or {}
    adj = ag.get("adjudication") or {}
    lines = [
        "# Dataset statistics",
        "",
        f"Generated by `scripts/dataset_stats.py` at {stats['timestamp']}; every number below is in "
        "`results/dataset_stats.json`.",
        "",
        "## Sources",
        "",
        "| Source | Rows |\n|---|---:|",
        *[f"| {k} | {v:,} |" for k, v in stats["source_counts"].items()],
        "",
        "Only Source A (real YouTube comments) is used for validation and test. English external data, "
        "translations (B), romanised text (C) and synthetic comments (D) are training-only; 0 means not built yet.",
        "",
        f"## Source A: {stats['source_a_rows']:,} labelled comments from {stats['videos']} videos",
        "",
        table({"overall": stats["class_overall"], **{LANG_NAME.get(k, k): v for k, v in stats['class_by_lang'].items()}},
              "Set"),
        "",
        "### Per split (grouped by video)",
        "",
        table({k: stats["class_by_split"][k] for k in ("train", "val", "test") if k in stats["class_by_split"]}, "Split"),
        "",
        "Videos per split: " + ", ".join(f"{k} {v}" for k, v in stats["videos_by_split"].items()) + ".",
        "",
        table({k: v for k, v in sorted(stats["class_by_split_lang"].items())}, "Split / language"),
        "",
        "### Script mix",
        "",
        "| Language | " + " | ".join(sorted({c for v in stats["script_by_lang"].values() for c in v})) + " |",
        "|---|" + "---:|" * len({c for v in stats["script_by_lang"].values() for c in v}),
        *[f"| {LANG_NAME.get(k, k)} | " + " | ".join(str(v.get(c, 0)) for c in
                                                     sorted({c for vv in stats['script_by_lang'].values() for c in vv}))
          + " |" for k, v in stats["script_by_lang"].items()],
        "",
        "### Length (whitespace tokens)",
        "",
        *[f"- {LANG_NAME.get(k, k)}: mean {v['mean']}, median {v['median']}, 95th percentile {v['p95']}, "
          f"max {v['max']}" for k, v in stats["length_tokens"].items() if isinstance(v, dict)],
        f"- Share of comments over 128 whitespace tokens: {stats['length_tokens']['share_over_128_tokens']:.2%} "
        "(subword length is larger; the encoders truncate at 128 subword tokens).",
        "",
        "### Top words per class (stopwords removed)",
        "",
        *[f"- **{lab}**: " + ", ".join(f"{w} ({c})" for w, c in stats["top_words"][lab]) for lab in LABELS],
        "",
        "### Cleaning (from `results/comment_cleaning.json`)",
        "",
        "| Step | Removed |\n|---|---:|",
        *[f"| {s['step']} | {s['removed']:,} |" for s in stats["cleaning_steps"]],
        "",
        "## Label agreement (from `results/annotation_agreement.json`)",
        "",
        f"- Cohen's kappa, Claude vs Gemini: {ov.get('cohen_kappa')} (n = {ov.get('n')}, "
        f"{ov.get('pct_agreement')}% agreement).",
        "- Per language: " + ", ".join(f"{LANG_NAME.get(k, k)} {v['cohen_kappa']}" for k, v in (ag.get('by_lang') or {}).items()) + ".",
        "- Per script: " + ", ".join(f"{k} {v['cohen_kappa']}" for k, v in (ag.get('by_script') or {}).items()) + ".",
        f"- Disagreements: {adj.get('disagreements')}; resolved by gpt-oss: {adj.get('resolved_by_gptoss')}; "
        f"3-way splits (dropped): {adj.get('three_way_splits')}.",
        f"- Fleiss' kappa (Claude, Gemini, gpt-oss) on the 300-comment sample: "
        + (f"{fl.get('kappa')} (n = {fl.get('n')}). The sample was drawn from comments where Claude and Gemini "
           "already agree, so this mostly measures gpt-oss against that consensus and is biased upward."
           if fl.get("n") else "not run yet."),
        "",
        "## Figures",
        "",
        *[f"![{f}]({f})" for f in figs],
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="dry run on the first N rows; writes nothing")
    args = p.parse_args(argv)
    df = pd.read_csv(SOURCE_A, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit)
    stats = compute(df)
    if args.limit:
        print(json.dumps({k: stats[k] for k in ("source_a_rows", "class_overall", "class_by_lang")}, indent=1))
        print("Dry run: nothing written.")
        return
    figs = write_figures(df, stats)
    STATS_JSON.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_MD.write_text(markdown(stats, figs), encoding="utf-8")
    print(f"wrote {STATS_JSON.relative_to(ROOT)}, {REPORT_MD.relative_to(ROOT)} and {len(figs)} figures")


if __name__ == "__main__":
    main()
