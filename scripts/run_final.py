"""Pick the best encoder configuration by VALIDATION Macro F1 and run the follow-up experiments (6c, 6d, 6f).

Candidates: results/<exp_id>.json for every config given with --candidates (seed-42 runs). The best one by val
Macro F1 (test is never looked at) is then run
    1. with seed 42 again plus the evaluation-only sets (``--exp-suffix _robust``, challenge + noisy test),
    2. with seeds 43 and 44 (same extra evaluation sets),
    3. QLoRA (configs/e4_qlora.yaml) trained on the same data as the best config (its ``extra_train``).
Writes results/final_selection.json with the candidates' val scores and the choice.

Usage:
    python scripts/run_final.py --candidates configs/e2_mmbert.yaml configs/e3_best_plus_b.yaml configs/e3_best_plus_bcd.yaml
    python scripts/run_final.py --candidates ... --limit 1       # dry run: print the plan only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXTRA_EVAL = ["challenge=data/challenge/challenge.csv", "test_noisy=data/processed/test_noisy.csv"]


def choose(candidates: list[Path]) -> tuple[Path, dict]:
    scores = {}
    for cfg_path in candidates:
        exp_id = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["exp_id"]
        res = RESULTS / f"{exp_id}.json"
        if res.exists():
            r = json.loads(res.read_text(encoding="utf-8"))
            if r.get("kind") == "encoder":
                scores[str(cfg_path.as_posix())] = {"exp_id": exp_id, "val_macro_f1": r["val"]["macro_f1"],
                                                    "val_against_f1": r["val"]["against_f1"]}
    if not scores:
        raise SystemExit("no candidate has a result file")
    best = max(scores, key=lambda k: (scores[k]["val_macro_f1"], scores[k]["val_against_f1"]))
    return Path(best), scores


def run(cmd: list[str], dry: bool) -> int:
    print("RUN:", " ".join(cmd), flush=True)
    return 0 if dry else subprocess.run(cmd, cwd=ROOT).returncode


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", nargs="+", type=Path, required=True)
    p.add_argument("--qlora-config", type=Path, default=ROOT / "configs" / "e4_qlora.yaml")
    p.add_argument("--skip-qlora", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="dry run: print the plan, run nothing")
    args = p.parse_args(argv)
    dry = bool(args.limit)
    best, scores = choose(args.candidates)
    best_cfg = yaml.safe_load(best.read_text(encoding="utf-8"))
    print(json.dumps(scores, indent=1))
    print(f"best by val Macro F1: {best} ({scores[str(best.as_posix())]['exp_id']})", flush=True)
    selection = {"experiment_id": "final_selection", "script": "scripts/run_final.py", "seed": 42,
                 "rule": "highest val Macro F1 (tie: val Against F1); test not used", "candidates": scores,
                 "best_config": best.as_posix(), "best_exp_id": best_cfg["exp_id"]}
    if not dry:
        (RESULTS / "final_selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    enc = [sys.executable, "-m", "terralens.models.encoder", "--config", str(best)]
    status = {}
    status["robust_s42"] = run(enc + ["--exp-suffix", "_robust", "--extra-eval", *EXTRA_EVAL], dry)
    for seed in (43, 44):
        status[f"seed_{seed}"] = run(enc + ["--seed", str(seed), "--extra-eval", *EXTRA_EVAL], dry)
    if not args.skip_qlora:
        q = yaml.safe_load(args.qlora_config.read_text(encoding="utf-8"))
        q["extra_train"] = best_cfg.get("extra_train", [])
        q["train_data_from"] = best_cfg["exp_id"]
        tmp = ROOT / "configs" / "_e4_qlora_resolved.yaml"
        tmp.write_text(yaml.safe_dump(q, sort_keys=False), encoding="utf-8")
        status["qlora"] = run([sys.executable, "-m", "terralens.models.encoder", "--config", str(tmp)], dry)
    print("STATUS", status, flush=True)


if __name__ == "__main__":
    main()
