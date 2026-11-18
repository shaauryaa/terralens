"""Rule-based language detection for Hindi / Bengali / English YouTube comments.

Script decides first: Devanagari letters -> Hindi, Bengali letters -> Bengali (the Indic script with more
letters wins in code-mixed text). Latin-script text counts as romanised Hindi or Bengali if it contains
marker words from ``ROMANISED_KEYWORDS``; otherwise it is English if it contains common English words,
else ``other``.

The marker lists hold frequent romanised function words and spellings. Words that are also common English
words (to, me, main, log, band, mat, jab, tab, ...) are left out on purpose. "ki" and "nai" are in both
lists; a tie between the lists is resolved by ``fallback`` (e.g. the video's language).
"""

from __future__ import annotations

import regex as re

from terralens.preprocess import script_counts, tag_script

ROMANISED_KEYWORDS = {
    "hi": [
        # used by the T1 screening
        "hai", "hain", "nahi", "nahin", "kya", "bhai", "ki", "ka", "ke", "aur", "mein", "ho", "tha",
        "raha", "sab", "yeh", "ye", "kuch", "bhaiya", "bahut", "achha", "accha", "nai",
        # added in T3
        "toh", "ek", "ko", "se", "bhi", "aap", "apna", "apne", "kar", "karo", "karna", "karte", "karta", "kr",
        "kro", "krna", "krte", "ji", "hota", "hoti", "hoga", "hogi", "hoge", "rahe", "rahi", "gaya", "gayi",
        "gya", "dena", "wala", "wale", "wali", "kaise", "kyun", "kyu", "kyon", "sirf", "abhi", "bas",
        "sahi", "galat", "acha", "sabse", "sb", "logo", "logon", "lagta", "chahiye", "chaiye", "chahie",
        "dekho", "dekhna", "padho", "baar", "jaldi", "h", "wo", "woh", "unka", "uska", "iska", "hamare",
        "humare", "tumhare", "jata", "jati", "jate", "kitna", "bahot", "bohot", "bht", "bhot", "nhi", "kam",
        "jyada", "zyada", "jada", "garmi", "pani", "paani", "dharti", "desh", "namaskar", "pranam",
        "dhanyawad", "shukriya", "sach", "jhooth", "jhoot", "jhuth", "hum", "yaar", "yar", "mai", "mujhe",
        "hume", "humein", "hamein", "unko", "inko", "iske", "uske", "liye", "sath", "saath", "kuchh",
        "sabhi", "koi", "bilkul", "matlab", "samajh", "pata", "meri", "mera", "tera", "teri", "aaj",
        "kal", "bolo", "bola", "bhagat", "andhe", "laao", "karenge", "hoon", "hu", "rha", "rhe", "rhi", "kitne", "kitni", "jhuta", "jhuti",
        "bhut", "raho", "jao", "dalo", "ladki", "duniya", "har", "wahi", "kahna", "chahte", "sudhar", "daiyaa", "nehi", "batao", "kab", "fir", "phir", "thhe", "kyaa",
    ],
    "bn": [
        # used by the T1 screening
        "ami", "tumi", "ache", "achhe", "na", "kore", "hobe", "eta", "dada", "khub", "amra", "ki", "keno",
        "holo", "bhalo", "valo", "vaiya", "nai", "hoy", "theke", "ar",
        # added in T3
        "apni", "apnar", "apnake", "amar", "amake", "amader", "tomar", "tomra", "tor", "tui", "kivabe",
        "kibhabe", "kemon", "kemne", "kotha", "ase", "nei", "korte", "korchi", "korchis", "korche", "koren",
        "korben", "korbo", "korlo", "korece", "koreche", "hoye", "hoyeche", "hoyece", "hoice", "hocche",
        "hosse", "ota", "ei", "oi", "sei", "onek", "bolen", "bollen", "bolchen", "bolche", "didi", "vai",
        "sathe", "jonno", "jnno", "jonne", "kintu", "tai", "tahole", "ekhon", "ekhono", "akhon", "obdhi",
        "sorkar", "manush", "gach", "kono", "konovabe", "shob", "shudhu", "sudhu", "aro", "amio", "je",
        "jodi", "bujhi", "bujhte", "dekhen", "dekhun", "dekhe", "laglo", "lage", "pari", "parbo", "jabe",
        "gelo", "asolei", "kichu", "kisu", "onno", "ekta", "akta", "akto", "baper", "bepar", "niye", "diye",
        "dilen", "dile", "khoca", "diyece", "lagiyechen", "bolajabe", "sompork", "somporke",
        "korun", "koro", "sobai", "sobaike", "gojob", "hefajot", "sundor", "thik", "kobe", "kora", "tomi",
        "porse", "thak", "onk", "gachh", "moja", "te", "abar", "bochor", "korin",
    ],
}

# Frequent English function words, used only to tell English from other Latin-script text.
ENGLISH_WORDS = [
    "the", "is", "are", "was", "were", "and", "or", "to", "of", "in", "on", "it", "this", "that", "you",
    "for", "not", "be", "with", "we", "i", "they", "have", "has", "but", "what", "so", "very", "my", "your",
    "our", "all", "can", "will", "do", "does", "from", "at", "by", "an", "a", "if", "no", "yes", "thank",
    "thanks", "please", "good", "nice", "great", "sir", "video", "love", "best", "who", "why", "how",
    "should", "would", "there", "their", "people", "about", "more", "just",
]

# Forms of address that Indian and Bangladeshi English uses too ("Thanks Acharya ji", "Great video bhai").
# They count as Hindi/Bengali markers only when the comment has no English function words.
ADDRESS_WORDS = {"ji", "bhai", "bhaiya", "vai", "vaiya", "dada", "didi", "yaar", "yar"}

LANGS = ("hi", "bn", "en", "other")


def _word_re(words) -> re.Pattern:
    return re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(set(words), key=len, reverse=True))) + r")\b",
                      re.IGNORECASE)


_KEYWORD_RES = {lang: _word_re(words) for lang, words in ROMANISED_KEYWORDS.items()}
_ENGLISH_RE = _word_re(ENGLISH_WORDS)


def marker_counts(text: str, include_address: bool = True) -> dict[str, int]:
    """Number of romanised Hindi and Bengali marker words in ``text`` (whole words, any case)."""
    out = {}
    for lang, rx in _KEYWORD_RES.items():
        found = rx.findall(text)
        out[lang] = len(found) if include_address else sum(w.lower() not in ADDRESS_WORDS for w in found)
    return out


def english_word_count(text: str) -> int:
    return len(_ENGLISH_RE.findall(text))


def romanised_language(text: str) -> str:
    """'hi' or 'bn' by marker words, 'indic' on a tie, '' if the text is not romanised Hindi/Bengali.

    Address words alone (ji, bhai, dada, ...) only count when the text has no English function words.
    """
    strong = marker_counts(text, include_address=False)
    if not (strong["hi"] or strong["bn"]) and english_word_count(text):
        return ""
    counts = marker_counts(text)
    n_hi, n_bn = counts["hi"], counts["bn"]
    if not (n_hi or n_bn):
        return ""
    return "hi" if n_hi > n_bn else "bn" if n_bn > n_hi else "indic"


def detect_language(text: str, fallback: str | None = None) -> str:
    """Language of a cleaned, emoji-free comment: 'hi', 'bn', 'en' or 'other'.

    A Hindi/Bengali tie in romanised text returns ``fallback`` when it is 'hi' or 'bn', else 'other'.
    """
    text = text or ""
    script = tag_script(text)
    if script == "devanagari":
        return "hi"
    if script == "bengali":
        return "bn"
    counts = script_counts(text)
    if counts["devanagari"] or counts["bengali"]:
        return "hi" if counts["devanagari"] >= counts["bengali"] else "bn"
    if script in ("latin", "mixed"):
        lang = romanised_language(text)
        if lang in ("hi", "bn"):
            return lang
        if lang == "indic":
            return fallback if fallback in ("hi", "bn") else "other"
        if english_word_count(text):
            return "en"
    return "other"
