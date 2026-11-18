"""Append Claude's labels for generated comments (synthetic / challenge) to labels_claude_<set>.jsonl.

Input: a text file with one "<number> <F|A|N>" per line (the number of the generated id, e.g. 0007 for syn_0007),
written in the Claude Code session from claude_todo_<set>.txt, which shows id and text only.

Usage:
    python scripts/write_claude_labels.py --set synthetic --file labels.txt
    python scripts/write_claude_labels.py --set synthetic --file labels.txt --limit 3    # dry run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data" / "interim" / "synthetic"
CODES = {"F": "Favour", "A": "Against", "N": "None"}


def parse(lines, prefix: str) -> list[dict]:
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1] in CODES:
            out.append({"id": f"{prefix}_{int(parts[0]):04d}", "label": CODES[parts[1]], "confidence": "medium",
                        "tags": [], "annotator": "claude"})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--set", choices=["synthetic", "challenge"], required=True)
    p.add_argument("--file", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    recs = parse(args.file.read_text(encoding="utf-8").splitlines(), args.set[:3])
    if args.limit:
        print(recs[: args.limit])
        return
    out = DIR / f"labels_claude_{args.set}.jsonl"
    with out.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(recs)} labels to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
