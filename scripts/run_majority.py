"""E0: majority-class floor. Always predicts the most frequent label in Source A train.

Writes results/e0_majority.json with val and test metrics (same format as the other experiments).

Usage:
    python scripts/run_majority.py
    python scripts/run_majority.py --limit 64     # dry run, prints, writes nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.reporting import split_metrics, write_result  # noqa: E402
from terralens.splits import load_splits  # noqa: E402

SEED = 42


def majority_label(labels) -> str:
    counts = {}
    for y in labels:
        counts[y] = counts.get(y, 0) + 1
    return max(sorted(counts), key=lambda k: counts[k])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)
    data = load_splits(limit=args.limit)
    label = majority_label(data["train"]["label"])
    payload = {"kind": "majority", "config": {"rule": "always predict the most frequent Source A train label",
                                              "label": label}, "seed": SEED,
               "data": {k: len(v) for k, v in data.items()}}
    for split in ("val", "test"):
        payload[split] = split_metrics(data[split], [label] * len(data[split]), seed=SEED)
    print(f"E0 majority = {label}: val macroF1={payload['val']['macro_f1']} test macroF1={payload['test']['macro_f1']}")
    if args.limit:
        print("Dry run: nothing written.")
        return
    print(f"wrote {write_result('e0_majority', payload)}")


if __name__ == "__main__":
    main()
