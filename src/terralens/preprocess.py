"""Text normalisation for Hindi / Bengali / code-mixed YouTube comments.

Pipeline (``clean_text``):
    1. Unicode NFC normalisation
    2. HTML entity unescaping and tag removal
    3. URL removal
    4. @mention removal
    5. emoji -> text (``:fire:`` style tokens)
    6. elongation fix: any character repeated 3+ times is capped at 2
    7. whitespace collapse

``tag_script`` labels a text by its dominant script: Devanagari (U+0900-U+097F),
Bengali (U+0980-U+09FF), Latin, ``mixed`` or ``other``.
"""

from __future__ import annotations

import html
import unicodedata

import emoji
import pandas as pd
import regex as re

URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
# Handles like @user, @user.name, @user-name; not e-mail addresses (no word char before '@').
MENTION_RE = re.compile(r"(?<!\w)@[\w.\-]+")
ELONGATION_RE = re.compile(r"(.)\1{2,}", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
BENGALI_RE = re.compile(r"[ঀ-৿]")
LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")

SCRIPTS = ("devanagari", "bengali", "latin")


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    text = html.unescape(text)
    return HTML_TAG_RE.sub(" ", text)


def remove_urls(text: str) -> str:
    return URL_RE.sub(" ", text)


def remove_mentions(text: str) -> str:
    return MENTION_RE.sub(" ", text)


def emoji_to_text(text: str) -> str:
    """Replace each emoji with a spaced ``:name:`` token, e.g. "🔥" -> " :fire: "."""
    return emoji.demojize(text, delimiters=(" :", ": "))


def fix_elongation(text: str, max_repeat: int = 2) -> str:
    """Cap runs of the same character at ``max_repeat`` ("sooooo" -> "soo")."""
    pattern = ELONGATION_RE if max_repeat == 2 else re.compile(r"(.)\1{%d,}" % max_repeat, re.DOTALL)
    return pattern.sub(lambda m: m.group(1) * max_repeat, text)


def collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str, lowercase: bool = False) -> str:
    """Run the full normalisation pipeline on one comment.

    ``lowercase`` only affects Latin characters (Devanagari and Bengali have no case).
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = normalize_unicode(str(text))
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_mentions(text)
    text = emoji_to_text(text)
    text = fix_elongation(text)
    if lowercase:
        text = text.lower()
    return collapse_whitespace(text)


def script_counts(text: str) -> dict[str, int]:
    """Number of characters from each script of interest."""
    return {
        "devanagari": len(DEVANAGARI_RE.findall(text)),
        "bengali": len(BENGALI_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
    }


def tag_script(text: str, dominance: float = 0.8) -> str:
    """Return the dominant script of ``text``.

    A script is dominant if it covers at least ``dominance`` of the counted letters.
    Texts with letters from several scripts below that threshold are ``mixed``;
    texts with no Devanagari / Bengali / Latin letters are ``other``.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if total == 0:
        return "other"
    script, n = max(counts.items(), key=lambda kv: kv[1])
    return script if n / total >= dominance else "mixed"


def preprocess_df(df: pd.DataFrame, text_col: str = "text", lowercase: bool = False) -> pd.DataFrame:
    """Return a copy of ``df`` with ``clean_text`` and ``script`` columns added."""
    out = df.copy()
    out["clean_text"] = out[text_col].map(lambda t: clean_text(t, lowercase=lowercase))
    out["script"] = out["clean_text"].map(tag_script)
    return out
