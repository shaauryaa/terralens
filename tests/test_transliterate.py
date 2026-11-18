"""Tests for the rule-based romaniser in scripts/transliterate.py."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("transliterate", ROOT / "scripts" / "transliterate.py")
tl = importlib.util.module_from_spec(_spec)
sys.modules["transliterate"] = tl
_spec.loader.exec_module(tl)


@pytest.mark.parametrize("native, roman", [
    ("जलवायु परिवर्तन एक गंभीर समस्या है", "jalvayu parivartan ek ganbhir samasya hai"),
    ("ये सब झूठ है", "ye sab jhuth hai"),
    ("करना", "karna"),
    ("ज़्यादा गर्मी", "zyada garmi"),
    ("পৃথিবীর তাপমাত্রা বাড়ছে", "prithibir tapmatra barchhe"),
    ("গাছ লাগাও", "gachh lagao"),
])
def test_romanise_common_words(native, roman):
    assert tl.romanise(native) == roman


def test_romanise_keeps_latin_and_emoji_tokens():
    assert tl.romanise("climate change सच है :fire:") == "climate change sach hai :fire:"
    assert tl.romanise("") == ""
