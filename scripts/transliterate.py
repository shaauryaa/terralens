"""Romanise part of the training data (T9, Source C, training-only).

YouTube users often write Hindi and Bengali in Latin script ("climate change sach hai", "ami bhoy pachchi").
This script makes romanised copies of native-script training text with a rule-based romaniser (casual
spelling, not ITRANS): consonant + vowel-sign tables for Devanagari and Bengali, inherent vowel "a" (Hindi) or
"o" (Bengali), word-final inherent-vowel deletion, and the medial deletion rule V C _ C V (e.g. "karana" ->
"karna"). IndicXlit is not used because it needs fairseq; the rule-based output is noisier but cheap.

Rows romanised (seed 42, stratified by label):
    - ``--frac-real`` of the native-script (Devanagari / Bengali) Source A **train** rows
    - ``--frac-translated`` of data/processed/translated.csv (if it exists)
Validation and test rows are never read.

Output: data/processed/romanised.csv (common schema, source = "romanised", script = "latin", src_id).

Usage:
    python scripts/transliterate.py
    python scripts/transliterate.py --limit 10      # dry run: prints 10 examples, writes nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.splits import load_splits  # noqa: E402

TRANSLATED = ROOT / "data" / "processed" / "translated.csv"
OUT_CSV = ROOT / "data" / "processed" / "romanised.csv"
SEED = 42

# ---------------------------------------------------------------------------------------------- tables
DEVA_CONS = dict(zip(
    "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहळ",
    ["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
     "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h", "l"]))
DEVA_NUKTA = {"क": "q", "ख": "kh", "ग": "g", "ज": "z", "ड": "r", "ढ": "rh", "फ": "f", "य": "y"}
DEVA_VOW = dict(zip("अआइईउऊऋएऐओऔऍऑ", ["a", "a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au", "e", "o"]))
DEVA_SIGN = dict(zip("ािीुूृेैोौॅॉ", ["a", "i", "i", "u", "u", "ri", "e", "ai", "o", "au", "e", "o"]))
DEVA = {"cons": DEVA_CONS, "vow": DEVA_VOW, "sign": DEVA_SIGN, "virama": "्", "nukta": "़", "nukta_map": DEVA_NUKTA,
        "nasal": {"ं": "n", "ँ": "n", "ः": "h"}, "digits": "०१२३४५६७८९", "inherent": "a", "danda": "।॥"}

BENG_CONS = dict(zip(
    "কখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহৎ",
    ["k", "kh", "g", "gh", "ng", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n", "t", "th", "d", "dh", "n",
     "p", "ph", "b", "bh", "m", "j", "r", "l", "sh", "sh", "sh", "h", "t"]))
BENG_NUKTA = {"ড": "r", "ঢ": "rh", "য": "y"}
BENG_VOW = dict(zip("অআইঈউঊঋএঐওঔ", ["o", "a", "i", "i", "u", "u", "ri", "e", "oi", "o", "ou"]))
BENG_SIGN = dict(zip("ািীুূৃেৈোৌ", ["a", "i", "i", "u", "u", "ri", "e", "oi", "o", "ou"]))
BENG = {"cons": BENG_CONS, "vow": BENG_VOW, "sign": BENG_SIGN, "virama": "্", "nukta": "়", "nukta_map": BENG_NUKTA,
        "nasal": {"ং": "ng", "ঁ": "n", "ঃ": "h"}, "digits": "০১২৩৪৫৬৭৮৯", "inherent": "o", "danda": "।॥"}


def _script_of(ch: str):
    if "ऀ" <= ch <= "ॿ":
        return DEVA
    if "ঀ" <= ch <= "৿":
        return BENG
    return None


def _romanise_word(word: str, t: dict) -> str:
    """Romanise one native-script word. Units: (consonant, vowel or None for inherent, "" for virama)."""
    units: list[list] = []  # [kind, roman, vowel] kind: "C" consonant, "V" vowel, "X" other
    import unicodedata

    word = unicodedata.normalize("NFD", word)  # precomposed nukta letters -> base + nukta
    i = 0
    while i < len(word):
        ch = word[i]
        if ch in t["cons"]:
            units.append(["C", t["cons"][ch], None, ch])
        elif ch == t["nukta"]:
            if units and units[-1][0] == "C" and units[-1][3] in t["nukta_map"]:
                units[-1][1] = t["nukta_map"][units[-1][3]]
        elif ch in t["sign"] and units and units[-1][0] == "C":
            units[-1][2] = t["sign"][ch]
        elif ch == t["virama"] and units and units[-1][0] == "C":
            units[-1][2] = ""
        elif ch in t["vow"]:
            units.append(["V", t["vow"][ch], None, ch])
        elif ch in t["nasal"]:
            units.append(["X", t["nasal"][ch], None, ch])
        elif ch in t["digits"]:
            units.append(["X", str(t["digits"].index(ch)), None, ch])
        elif ch in t["danda"]:
            units.append(["X", ".", None, ch])
        else:
            units.append(["X", ch, None, ch])
        i += 1
    cons_idx = [k for k, u in enumerate(units) if u[0] == "C"]
    # Inherent vowel: None -> inherent, then deletion rules.
    for k in cons_idx:
        if units[k][2] is None:
            units[k][2] = t["inherent"]
            units[k].append("inh")  # 5th field marks an inherent vowel
    # word-final inherent vowel deletion (not for one-consonant words)
    last_c = [k for k in cons_idx if k == len(units) - 1 or all(u[0] == "X" for u in units[k + 1:])]
    if last_c and sum(u[0] in "CV" for u in units) > 1 and len(units[last_c[0]]) == 5:
        units[last_c[0]][2] = ""
    # medial deletion: V C(inh) C V -> drop the inherent vowel
    for n, k in enumerate(cons_idx[1:-1], start=1):
        prev_k, next_k = cons_idx[n - 1], cons_idx[n + 1]
        if (len(units[k]) == 5 and units[k][2] and next_k == k + 1 and units[next_k][2]
                and prev_k == k - 1 and units[prev_k][2]):
            units[k][2] = ""
            break
    return "".join(u[1] + (u[2] or "") if u[0] == "C" else u[1] for u in units)


def romanise(text: str) -> str:
    """Casual Latin spelling of Devanagari / Bengali text; other characters are kept as they are."""
    out, word, table = [], [], None
    for ch in str(text) + " ":
        t = _script_of(ch)
        if t is not None and (table is None or t is table):
            word.append(ch)
            table = t
            continue
        if word:
            out.append(_romanise_word("".join(word), table))
            word, table = [], None
        if t is not None:
            word, table = [ch], t
        else:
            out.append(ch)
    return "".join(out).strip()


def pick(df: pd.DataFrame, frac: float, seed: int = SEED) -> pd.DataFrame:
    if frac <= 0 or df.empty:
        return df.iloc[0:0]
    return df.groupby("label", group_keys=False).sample(frac=frac, random_state=seed)


def build(frac_real: float, frac_translated: float, limit: int | None = None) -> pd.DataFrame:
    train = load_splits()["train"]
    native = train[train["script"].isin(["devanagari", "bengali"])]
    parts = [pick(native, frac_real)]
    if TRANSLATED.exists():
        tr = pd.read_csv(TRANSLATED, dtype=str, keep_default_na=False)
        tr = tr[tr["script"].isin(["devanagari", "bengali"])]
        parts.append(pick(tr, frac_translated))
    src = pd.concat(parts, ignore_index=True)
    if limit:
        src = src.head(limit)
    return pd.DataFrame({"id": "rom_" + src["id"].astype(str), "text": src["text"].map(romanise),
                         "label": src["label"], "lang": src["lang"], "script": "latin", "source": "romanised",
                         "video_id": "", "tags": '["romanised"]', "src_id": src["id"],
                         "src_source": src["source"]})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frac-real", type=float, default=0.5)
    p.add_argument("--frac-translated", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None, help="dry run: N rows, prints, writes nothing")
    args = p.parse_args(argv)
    out = build(args.frac_real, args.frac_translated, args.limit)
    print(out.groupby(["src_source", "lang", "label"]).size().to_string())
    if args.limit:
        for t in out["text"].head(args.limit):
            print("  ", t[:120])
        print("Dry run: nothing written.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(ROOT)} ({len(out)} rows)")


if __name__ == "__main__":
    main()
