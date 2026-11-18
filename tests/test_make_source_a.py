"""Tests for scripts/make_source_a.py."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_source_a", ROOT / "scripts" / "make_source_a.py")
msa = importlib.util.module_from_spec(_spec)
sys.modules["make_source_a"] = msa
_spec.loader.exec_module(msa)


def _merged():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d"], "video_id": ["v1", "v1", "v2", "v2"], "text": ["t1", "t2", "t3", "t4"],
        "lang": ["hi", "hi", "bn", "bn"], "script": ["latin"] * 4,
        "label_claude": ["Favour", "Favour", "None", "Against"],
        "tags_claude": ['["sarcasm"]', "[]", "[]", "[]"],
        "label_gemini": ["Favour", "None", "Against", "Against"],
        "tags_gemini": ['["implicit"]', '["negation"]', "[]", "[]"],
        "label_gptoss": ["", "None", "Favour", ""], "tags_gptoss": ["", '["question"]', "[]", ""],
        "final_label": ["Favour", "None", "", "Against"], "review_flag": ["0", "0", "1", "0"],
        "final_lang": ["hi", "hi", "bn", ""], "keep": ["1", "1", "1", "0"],
    })


def test_build_drops_splits_and_unkept_rows():
    out, dropped = msa.build(_merged())
    assert out["id"].tolist() == ["a", "b"]
    assert out.columns.tolist() == msa.COLUMNS
    assert (out["source"] == "youtube").all()
    assert dropped["id"].tolist() == ["c"]


def test_tags_come_from_annotators_matching_final_label():
    out, _ = msa.build(_merged())
    tags = dict(zip(out["id"], out["tags"].map(json.loads)))
    assert tags["a"] == ["implicit", "sarcasm"]
    assert tags["b"] == ["negation", "question"]  # Claude said Favour, so its tags are not used
