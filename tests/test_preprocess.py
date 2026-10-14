import unicodedata

from terralens.preprocess import (
    clean_text,
    emoji_to_text,
    fix_elongation,
    preprocess_df,
    remove_mentions,
    remove_urls,
    tag_script,
)


def test_nfc_normalisation():
    # Bengali "ো" written as two code points (U+09C7 + U+09BE) should compose to U+09CB.
    decomposed = "কো"
    out = clean_text(decomposed)
    assert out == unicodedata.normalize("NFC", decomposed)
    assert "ো" in out


def test_removes_urls_html_and_mentions():
    text = "@climate_guy see https://example.com/x?a=1 and www.foo.org <br>bahut garmi &amp; dhoop"
    assert clean_text(text) == "see and bahut garmi & dhoop"


def test_mention_removal_keeps_emails_intact():
    assert remove_mentions("mail me at a@b.com @user.name") == "mail me at a@b.com  "


def test_remove_urls_only():
    assert remove_urls("link: http://t.co/abc done") == "link:   done"


def test_emoji_to_text():
    out = emoji_to_text("गर्मी 🔥")
    assert ":fire:" in out
    assert "🔥" not in out


def test_elongation_capped_at_two():
    assert fix_elongation("sooooo hottt") == "soo hott"
    assert fix_elongation("!!!!!") == "!!"
    assert fix_elongation("good") == "good"  # doubles are left alone
    assert fix_elongation("बहुततततत") == "बहुतत"


def test_script_tagging():
    assert tag_script("जलवायु परिवर्तन गंभीर है") == "devanagari"
    assert tag_script("জলবায়ু পরিবর্তন গুরুতর") == "bengali"
    assert tag_script("climate change is real") == "latin"
    assert tag_script("climate change बहुत serious है") == "mixed"
    assert tag_script("123 !!! :)") == "other"


def test_preprocess_df_adds_columns():
    import pandas as pd

    df = pd.DataFrame({"id": ["1", "2"], "text": ["HOTTTT 🔥 @x", None]})
    out = preprocess_df(df, lowercase=True)
    assert list(out["clean_text"]) == ["hott :fire:", ""]
    assert list(out["script"]) == ["latin", "other"]
