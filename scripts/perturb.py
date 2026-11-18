"""Make a noisy copy of the test set for the robustness check (T10). Evaluation only, never training.

Each test comment gets character-level noise of the kind seen in YouTube comments (seed 42):
    - vowel drops in Latin words ("bahut" -> "bht")            - doubled letters ("garmi" -> "garrmi")
    - swapped neighbouring characters ("climate" -> "cilmate")
    - transliteration: ``--translit-frac`` of the native-script comments are romanised with the rule-based
      romaniser from scripts/transliterate.py
A word is perturbed with probability ``--p-word``; emoji tokens (":fire:") are left alone. Labels, ids, video ids
and tags are kept; the column ``perturbation`` lists what was applied.

Output: data/processed/test_noisy.csv (same schema as source_a.csv + perturbation)

Usage:
    python scripts/perturb.py
    python scripts/perturb.py --limit 5      # dry run: print 5 clean / noisy pairs
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from terralens.preprocess import tag_script  # noqa: E402
from terralens.splits import load_splits  # noqa: E402

OUT_CSV = ROOT / "data" / "processed" / "test_noisy.csv"
SEED = 42
LATIN_VOWELS = set("aeiouAEIOU")


def drop_vowels(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    inner = [c for i, c in enumerate(word) if not (0 < i < len(word) - 1 and c in LATIN_VOWELS and rng.random() < 0.6)]
    return "".join(inner)


def double_letter(word: str, rng: random.Random) -> str:
    if len(word) < 2:
        return word
    i = rng.randrange(len(word))
    return word[:i + 1] + word[i] + word[i + 1:]


def swap_chars(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def perturb_text(text: str, rng: random.Random, p_word: float = 0.2, romanise_it: bool = False) -> tuple[str, list[str]]:
    applied = []
    if romanise_it:
        from transliterate import romanise

        text = romanise(text)
        applied.append("transliteration")
    words = text.split(" ")
    for j, w in enumerate(words):
        if not w or (w.startswith(":") and w.endswith(":")) or rng.random() >= p_word:
            continue
        is_latin = all(ord(c) < 0x250 for c in w)
        ops = (["vowel_drop", "double", "swap"] if is_latin else ["double", "swap"])
        op = rng.choice(ops)
        words[j] = {"vowel_drop": drop_vowels, "double": double_letter, "swap": swap_chars}[op](w, rng)
        if words[j] != w:
            applied.append(op)
    return " ".join(words), sorted(set(applied))


def make_noisy(test: pd.DataFrame, p_word: float = 0.2, translit_frac: float = 0.3, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for r in test.itertuples(index=False):
        native = r.script in ("devanagari", "bengali")
        text, applied = perturb_text(r.text, rng, p_word, romanise_it=native and rng.random() < translit_frac)
        d = r._asdict()
        d.update(text=text, script=tag_script(text), perturbation=",".join(applied) or "none")
        rows.append(d)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--p-word", type=float, default=0.2)
    p.add_argument("--translit-frac", type=float, default=0.3)
    p.add_argument("--limit", type=int, default=None, help="dry run: print N examples, write nothing")
    args = p.parse_args(argv)
    test = load_splits()["test"]
    noisy = make_noisy(test, args.p_word, args.translit_frac)
    changed = (noisy["text"].values != test["text"].values).mean()
    print(f"{len(noisy)} test comments, {changed:.1%} changed; perturbations: "
          f"{noisy['perturbation'].str.split(',').explode().value_counts().to_dict()}")
    if args.limit:
        for a, b in zip(test["text"].head(args.limit), noisy["text"].head(args.limit)):
            print(f"  {a[:90]}\n  -> {b[:90]}")
        print("Dry run: nothing written.")
        return
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    noisy.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"wrote {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
