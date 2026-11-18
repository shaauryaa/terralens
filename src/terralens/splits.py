"""Load Source A with the fixed video-grouped splits (data/splits/*_ids.txt) and optional extra training data.

Validation and test always come from Source A only. Extra CSVs (translated, romanised, synthetic, English) are
appended to train only, and any row whose id or exact cleaned text is in val/test is dropped from them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from terralens import LABELS
from terralens.preprocess import clean_text, tag_script

ROOT = Path(__file__).resolve().parents[2]
SOURCE_A = ROOT / "data" / "processed" / "source_a.csv"
SPLIT_DIR = ROOT / "data" / "splits"
COLUMNS = ["id", "text", "label", "lang", "script", "source", "video_id", "tags"]


def read_ids(name: str, split_dir: Path = SPLIT_DIR) -> list[str]:
    path = Path(split_dir) / f"{name}_ids.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/make_splits.py")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalise_frame(df: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    """Common schema, cleaned text, script tag filled in, rows without a valid label or text dropped."""
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].fillna("").astype(str).map(clean_text)
    missing_script = df["script"].fillna("").astype(str) == ""
    df.loc[missing_script, "script"] = df.loc[missing_script, "text"].map(tag_script)
    df["tags"] = df["tags"].fillna("").astype(str).replace("", "[]")
    if source is not None:
        df.loc[df["source"].fillna("").astype(str) == "", "source"] = source
    df = df[df["label"].isin(LABELS) & (df["text"] != "")]
    return df[COLUMNS].reset_index(drop=True)


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_splits(source_a: Path = SOURCE_A, split_dir: Path = SPLIT_DIR, extra_train=(),
                limit: int | None = None) -> dict[str, pd.DataFrame]:
    """{"train", "val", "test"} frames. ``extra_train`` is a list of CSV paths (training-only data)."""
    a = normalise_frame(pd.read_csv(source_a, dtype=str, keep_default_na=False), "youtube").set_index("id")
    out = {}
    for name in ("train", "val", "test"):
        ids = [i for i in read_ids(name, split_dir) if i in a.index]
        out[name] = a.loc[ids].reset_index()[COLUMNS]
    held_out_ids = set(out["val"]["id"]) | set(out["test"]["id"])
    held_out_text = set(out["val"]["text"]) | set(out["test"]["text"])
    extras = []
    for path in extra_train:
        e = normalise_frame(pd.read_csv(_resolve(path), dtype=str, keep_default_na=False))
        before = len(e)
        e = e[~e["id"].isin(held_out_ids) & ~e["text"].isin(held_out_text)]
        if len(e) < before:
            print(f"  {path}: dropped {before - len(e)} rows matching val/test ids or text")
        extras.append(e)
    if extras:
        out["train"] = pd.concat([out["train"], *extras], ignore_index=True)
    if limit:
        out = {k: v.sample(min(limit, len(v)), random_state=42).reset_index(drop=True) for k, v in out.items()}
    return out


def parse_tags(cell: str) -> list[str]:
    try:
        val = json.loads(cell) if cell else []
    except json.JSONDecodeError:
        return []
    return [str(t) for t in val] if isinstance(val, list) else []
