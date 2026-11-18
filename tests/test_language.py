"""Tests for terralens.language.detect_language."""

import pytest

from terralens.language import ADDRESS_WORDS, ENGLISH_WORDS, ROMANISED_KEYWORDS, detect_language


@pytest.mark.parametrize("text, expected", [
    ("जलवायु परिवर्तन बहुत गंभीर है", "hi"),
    ("জলবায়ু পরিবর্তন খুব গুরুতর", "bn"),
    ("climate change बहुत serious है", "hi"),               # any Devanagari -> Hindi
    ("global warming নিয়ে video", "bn"),
    ("Reel dekhna band karo", "hi"),                         # romanised Hindi
    ("AC band kr dena chaiye production", "hi"),
    ("Allah sobaike rokkha korun", "bn"),                    # romanised Bengali
    ("Chemotherapy baper akto bolen", "bn"),
    ("Climate change is real and serious", "en"),
    ("Thanks Acharya ji for this video", "en"),              # address word + English -> English
    ("Thank you so much, Sarthak Bhai", "en"),
    ("bhai sahi", "hi"),                                     # address word with a strong marker
    ("dada", "bn"),                                          # address word alone, no English -> counts
    ("Jai Shree Ram", "other"),                              # Latin, no markers, no English words
    ("ان الله بكل شيء عليم", "other"),                         # Arabic script
    ("", "other"),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_tie_uses_fallback():
    assert detect_language("ki bolbo", fallback="bn") == "bn"
    assert detect_language("ki bolbo", fallback="hi") == "hi"
    assert detect_language("ki bolbo") == "other"
    assert detect_language("ki bolbo", fallback="en") == "other"


def test_marker_lists_avoid_common_english_words():
    english_clashes = {"to", "me", "main", "log", "band", "mat", "jab", "tab", "mere", "lie", "per", "hay"}
    for words in ROMANISED_KEYWORDS.values():
        assert not english_clashes & set(words)
    assert ADDRESS_WORDS <= set(ROMANISED_KEYWORDS["hi"]) | set(ROMANISED_KEYWORDS["bn"])
    assert not set(ENGLISH_WORDS) & (set(ROMANISED_KEYWORDS["hi"]) | set(ROMANISED_KEYWORDS["bn"]))
