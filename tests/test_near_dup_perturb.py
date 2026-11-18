"""Tests for scripts/near_dup_check.py and scripts/perturb.py."""

import importlib.util
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


nd = _load("near_dup_check")
pt = _load("perturb")


def test_find_matches_exact_and_near():
    held = {"val": ["global warming is a hoax made by the west"], "test": ["पेड़ लगाओ धरती बचाओ"]}
    cands = ["Global warming is a hoax made by the west!", "global warming is a hoax made by the west",
             "पेड़ लगाओ धरती बचाओ", "completely different sentence about cricket"]
    assert nd.find_matches(cands, held) == ["val", "val", "test", None]


def test_perturb_text_is_seeded_and_keeps_emoji_tokens():
    a = pt.perturb_text("climate change bahut serious hai :fire:", random.Random(1), p_word=1.0)
    b = pt.perturb_text("climate change bahut serious hai :fire:", random.Random(1), p_word=1.0)
    assert a == b and a[0].endswith(":fire:") and a[0] != "climate change bahut serious hai :fire:"


def test_make_noisy_keeps_ids_and_labels():
    test = pd.DataFrame({"id": ["1", "2"], "text": ["garmi bahut hai bhai", "জলবায়ু পরিবর্তন সত্যি"],
                         "label": ["None", "Favour"], "lang": ["hi", "bn"], "script": ["latin", "bengali"],
                         "source": "youtube", "video_id": "v", "tags": "[]"})
    noisy = pt.make_noisy(test, p_word=0.5, translit_frac=1.0)
    assert noisy["id"].tolist() == ["1", "2"] and noisy["label"].tolist() == ["None", "Favour"]
    assert "transliteration" in noisy.loc[1, "perturbation"] and noisy.loc[1, "script"] == "latin"
