"""Tests for scripts/build_translation_set.py."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bts", ROOT / "scripts" / "build_translation_set.py")
bts = importlib.util.module_from_spec(_spec)
sys.modules["bts"] = bts
_spec.loader.exec_module(bts)


def test_prepare_text_strips_retweet_prefix_urls_and_mojibake():
    t = bts.prepare_text("RT @someone: Itâ€™s a hoax https://t.co/x @user")
    assert t == "It's a hoax"


def test_usable_filters_length_and_near_duplicates():
    df = pd.DataFrame({"id": ["1", "2", "3"], "label": ["Against"] * 3,
                       "text": ["global warming is a big hoax ok", "Global warming is a big HOAX, ok!", "too short"]})
    out = bts.usable(df)
    assert out["id"].tolist() == ["1"]


def test_assign_languages_splits_each_sentence_once():
    pool = pd.DataFrame({"id": [str(i) for i in range(60)], "text": [f"t{i}" for i in range(60)],
                         "label": ["Against"] * 20 + ["Favour"] * 20 + ["None"] * 20, "source_dataset": "k"})
    gwsd = pd.DataFrame({"id": ["g1", "g2"], "text": ["a", "b"], "label": ["None", "Favour"], "source_dataset": "gwsd"})
    sel = bts.assign_languages(pool, gwsd, per_lang=20)
    assert sel["id"].is_unique
    against = sel[sel["label"] == "Against"]
    assert (against["tgt_lang"] == "hi").sum() == 10 and (against["tgt_lang"] == "bn").sum() == 10
    assert ((sel["tgt_lang"] == "hi") & (sel["source_dataset"] == "k")).sum() == 20
    assert set(sel.loc[sel["source_dataset"] == "gwsd", "tgt_lang"]) == {"hi", "bn"}
