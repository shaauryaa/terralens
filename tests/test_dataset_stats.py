"""Tests for scripts/dataset_stats.py helpers."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("dataset_stats", ROOT / "scripts" / "dataset_stats.py")
ds = importlib.util.module_from_spec(_spec)
sys.modules["dataset_stats"] = ds
_spec.loader.exec_module(ds)


def test_tokens_keep_indic_vowel_signs_and_lowercase_latin():
    assert ds.tokens("Climate परिवर्तन জলবায়ু!") == ["climate", "परिवर्तन", "জলবায়ু"]


def test_top_words_drops_stopwords_and_emoji_names():
    df = pd.DataFrame({"text": ["the climate है :fire: climate", "garmi garmi hai"], "label": ["Favour", "None"]})
    tw = ds.top_words(df, k=5)
    assert tw["Favour"] == [("climate", 2)]
    assert tw["None"] == [("garmi", 2)]
    assert tw["Against"] == []


def test_class_table_counts_every_label():
    df = pd.DataFrame({"label": ["Favour", "None", "None"], "lang": ["hi", "hi", "bn"]})
    assert ds.class_table(df, None) == {"Favour": 1, "Against": 0, "None": 2}
    assert ds.class_table(df, "lang")["bn"] == {"Favour": 0, "Against": 0, "None": 1}
