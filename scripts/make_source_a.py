"""Write the final Source A file (real YouTube comments with their aggregated labels).

Reads data/interim/labels_merged.csv (from scripts/aggregate_labels.py) and keeps rows that
    - passed the language rule (keep = 1), and
    - have a final label (Claude and Gemini agree, or gpt-oss matched one of them).
The 3-way splits (review_flag = 1) are dropped and their ids written to data/interim/dropped_splits.csv.

Output schema (data/processed/source_a.csv):
    id, text, label, lang, script, source, video_id, tags
``tags`` is a JSON list: the union of the tags given by the annotators whose label equals the final label.

Usage:
    python scripts/make_source_a.py
    python scripts/make_source_a.py --limit 50     # dry run, prints counts, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens import LABELS  # noqa: E402

INTERIM = ROOT / "data" / "interim"
MERGED_CSV = INTERIM / "labels_merged.csv"
DROPPED_CSV = INTERIM / "dropped_splits.csv"
OUT_CSV = ROOT / "data" / "processed" / "source_a.csv"
ANNOTATORS = ["claude", "gemini", "gptoss"]
COLUMNS = ["id", "text", "label", "lang", "script", "source", "video_id", "tags"]
SEED = 42


def _tags(cell: str) -> list[str]:
    try:
        val = json.loads(cell) if cell else []
    except json.JSONDecodeError:
        return []
    return [str(t) for t in val] if isinstance(val, list) else []


def merged_tags(row) -> list[str]:
    """Union (sorted) of tags from annotators whose label equals the final label."""
    out: set[str] = set()
    for a in ANNOTATORS:
        if row.get(f"label_{a}") == row["final_label"]:
            out.update(_tags(row.get(f"tags_{a}", "")))
    return sorted(out)


def build(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(source_a rows, dropped 3-way split rows)."""
    kept = merged[merged["keep"].astype(str) == "1"]
    splits = kept[kept["review_flag"].astype(str) == "1"]
    labelled = kept[kept["final_label"].isin(LABELS)]
    out = pd.DataFrame({
        "id": labelled["id"],
        "text": labelled["text"],
        "label": labelled["final_label"],
        "lang": labelled["final_lang"],
        "script": labelled["script"],
        "source": "youtube",
        "video_id": labelled["video_id"],
        "tags": [json.dumps(merged_tags(r)) for _, r in labelled.iterrows()],
    })[COLUMNS].reset_index(drop=True)
    dropped = splits[["id", "video_id", "final_lang", "label_claude", "label_gemini", "label_gptoss"]].rename(
        columns={"final_lang": "lang"})
    return out, dropped.reset_index(drop=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--merged", type=Path, default=MERGED_CSV)
    p.add_argument("--out", type=Path, default=OUT_CSV)
    p.add_argument("--limit", type=int, default=None, help="dry run on the first N rows; writes nothing")
    args = p.parse_args(argv)
    merged = pd.read_csv(args.merged, dtype=str, keep_default_na=False)
    if args.limit:
        merged = merged.head(args.limit)
    out, dropped = build(merged)
    print(f"source_a rows: {len(out)}; dropped 3-way splits: {len(dropped)}")
    print(pd.crosstab(out["lang"], out["label"]).to_string())
    if args.limit:
        print("Dry run: nothing written.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    dropped.to_csv(DROPPED_CSV, index=False, encoding="utf-8")
    print(f"wrote {args.out} and {DROPPED_CSV}")


if __name__ == "__main__":
    main()
