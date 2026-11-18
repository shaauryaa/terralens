"""Tests for scripts/make_splits.py."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_splits", ROOT / "scripts" / "make_splits.py")
ms = importlib.util.module_from_spec(_spec)
sys.modules["make_splits"] = ms
_spec.loader.exec_module(ms)


def _toy(n_videos=80, per_video=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_videos):
        lang = "hi" if v % 2 else "bn"
        for i in range(per_video):
            lab = rng.choice(["Favour", "Against", "None"], p=[0.3, 0.2, 0.5])
            rows.append({"id": f"v{v}_{i}", "video_id": f"v{v}", "label": lab, "lang": lang, "script": "latin"})
    return pd.DataFrame(rows)


def test_splits_are_grouped_by_video_and_roughly_70_15_15():
    df = _toy()
    split = ms.assign_splits(df, seed=42)
    assert (df.assign(s=split).groupby("video_id")["s"].nunique() == 1).all()
    frac = split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.06 and abs(frac["val"] - 0.15) < 0.05 and abs(frac["test"] - 0.15) < 0.05


def test_assign_splits_is_deterministic():
    df = _toy()
    assert ms.assign_splits(df, 42).equals(ms.assign_splits(df, 42))


def test_choose_seed_respects_constraints():
    df = _toy()
    seed, split, tried = ms.choose_seed(df, min_against=5)
    ok, info = ms.check(df, split, 5)
    assert ok and seed == tried[-1]["seed"]
    assert info["val"]["against"] >= 5 and set(info["test"]["langs"]) == {"bn", "hi"}
    seed, split, _ = ms.choose_seed(df, seeds=range(42, 44), min_against=10_000)
    assert seed is None and split is None
