"""evaluate_robustness: robustness report from encoder result files that include the extra evaluation sets.

Reads one or more results/<exp_id>.json written by ``terralens.models.encoder`` with
``--extra-eval challenge=... test_noisy=...`` (e.g. the same config with several seeds) and reports, per run and
as mean ± std over runs:
    - clean test vs noisy test Macro F1 / Against F1 and the drop (clean - noisy)
    - challenge-set Macro F1 / Against F1 overall and per phenomenon (negation, sarcasm, implicit,
      topic_not_target, romanised)
    - test Macro F1 per annotator tag (the phenomena as they occur in real comments)
All numbers come from the input result files; nothing is re-predicted here.

Usage:
    python -m terralens.robustness results/e3_x_robust.json results/e3_x_s43.json results/e3_x_s44.json \\
        --out results/robustness_e3_x.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from terralens.reporting import write_result


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    arr = np.asarray(vals, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "n_runs": len(arr)}


def run_summary(res: dict, variant: str = "plain") -> dict:
    """Robustness numbers of one encoder result (``variant`` = plain or tuned offsets)."""
    test = res["test"] if variant == "plain" else res["thresholds"]["test_tuned"]
    extra = res.get("extra_eval", {})
    out = {"experiment_id": res.get("experiment_id"), "seed": res.get("seed"),
           "test_macro_f1": test["macro_f1"], "test_against_f1": test["against_f1"],
           "test_by_tag": {k: v["macro_f1"] for k, v in test.get("by_tag", {}).items()}}
    if "test_noisy" in extra:
        noisy = extra["test_noisy"][variant]
        out.update(noisy_macro_f1=noisy["macro_f1"], noisy_against_f1=noisy["against_f1"],
                   drop_macro_f1=round(test["macro_f1"] - noisy["macro_f1"], 4),
                   drop_against_f1=round(test["against_f1"] - noisy["against_f1"], 4))
    if "challenge" in extra:
        ch = extra["challenge"][variant]
        out.update(challenge_n=ch["n"], challenge_macro_f1=ch["macro_f1"], challenge_against_f1=ch["against_f1"],
                   challenge_by_phenomenon={k: {"n": v["n"], "macro_f1": v["macro_f1"], "accuracy": v["accuracy"]}
                                            for k, v in ch.get("by_tag", {}).items()},
                   challenge_by_lang={k: v["macro_f1"] for k, v in ch.get("by_lang", {}).items()})
    return out


def evaluate_robustness(results: list[dict], variant: str = "plain") -> dict:
    runs = [run_summary(r, variant) for r in results]
    keys = ["test_macro_f1", "test_against_f1", "noisy_macro_f1", "noisy_against_f1", "drop_macro_f1",
            "drop_against_f1", "challenge_macro_f1", "challenge_against_f1"]
    agg = {k: _mean_std([r.get(k) for r in runs]) for k in keys}
    phen = sorted({p for r in runs for p in r.get("challenge_by_phenomenon", {})})
    agg["challenge_by_phenomenon"] = {p: _mean_std([r.get("challenge_by_phenomenon", {}).get(p, {}).get("macro_f1")
                                                    for r in runs]) for p in phen}
    return {"variant": variant, "runs": runs, "mean_std": agg}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="dry run: use the first N result files, print only")
    args = p.parse_args(argv)
    files = args.results[: args.limit] if args.limit else args.results
    results = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    payload = {"kind": "robustness", "config": {"inputs": [f.as_posix() for f in files]}, "seed": None,
               "plain": evaluate_robustness(results, "plain"),
               "tuned_offsets": evaluate_robustness(results, "tuned") if all("thresholds" in r for r in results) else None}
    print(json.dumps(payload["plain"]["mean_std"], indent=1))
    if args.limit:
        print("Dry run: nothing written.")
        return
    exp_id = args.out.stem
    path = write_result(exp_id, payload, args.out.parent)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
