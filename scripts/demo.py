"""Live demo of TerraLens on a laptop CPU (no GPU, no API keys, no downloads; about half a minute).

Steps (each prints a header):
    1. Dataset   key counts from results/dataset_stats.json (total, per class, per language, per split)
    2. Training  a tiny CPU dry run of terralens.models.encoder (--limit 64 rows per split, tiny random BERT,
                 2 epochs) to show the training and evaluation loop end to end; prints its validation Macro F1.
                 The numbers are meaningless (random tiny model, 64 rows); the point is that the loop runs.
                 Skipped with a message if the tiny model is not in the Hugging Face cache or the local
                 Source A data (not in git) is missing.
    3. Results   rebuilds reports/results.md (scripts/make_results_table.py) and
                 reports/figures/results_macro_f1.png (scripts/plot_results.py), then prints the main table E0-E4
    4. Tests     pytest -q

Run it with scripts/demo.ps1 (Windows) or scripts/demo.sh, which switch Hugging Face to offline mode first.
Seed 42 (the dry run uses the config seed). The dry-run result goes to a temporary folder, not results/.

Usage:
    python scripts/demo.py
    python scripts/demo.py --skip-tests          # steps 1-3 only
    python scripts/demo.py --limit 32            # rows per split for the training dry run (default 64)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TINY_MODEL = "hf-internal-testing/tiny-random-bert"
DRY_RUN_CONFIG = "configs/e2_xlmr.yaml"
SEED = 42
LABELS = ("Favour", "Against", "None")
LANG = {"hi": "Hindi", "bn": "Bengali"}


def header(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n  STEP {n}  {title}\n{'=' * 72}", flush=True)


def run(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
           # keep the demo output readable: no model-loading reports or progress bars
           "TRANSFORMERS_VERBOSITY": "error", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    return subprocess.run(cmd, cwd=ROOT, env=env).returncode


# ---------------------------------------------------------------- step 1
def dataset_lines(ds: dict) -> list[str]:
    def cls(c: dict) -> str:
        return "  ".join(f"{lab} {c[lab]:>5,}" for lab in LABELS)

    lines = [f"Source A: {ds['source_a_rows']:,} labelled YouTube comments from {ds['videos']} videos",
             f"  all       {cls(ds['class_overall'])}"]
    for k in ("hi", "bn"):
        c = ds["class_by_lang"][k]
        lines.append(f"  {LANG[k]:<9} {cls(c)}   total {sum(c.values()):>5,}")
    for k in ("train", "val", "test"):
        c = ds["class_by_split"][k]
        lines.append(f"  {k:<9} {cls(c)}   total {sum(c.values()):>5,}  ({ds['videos_by_split'][k]} videos)")
    return lines


def step_dataset() -> None:
    header(1, "Dataset (results/dataset_stats.json)")
    for line in dataset_lines(json.loads((RESULTS / "dataset_stats.json").read_text(encoding="utf-8"))):
        print(line)


# ---------------------------------------------------------------- step 2
def model_cached(repo: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(repo, "config.json"), str)
    except Exception:
        return False


def step_training(limit: int) -> None:
    header(2, f"Training: tiny CPU dry run of the encoder ({limit} rows per split)")
    missing = [p for p in ("data/processed/source_a.csv", "data/splits/train_ids.txt") if not (ROOT / p).exists()]
    if missing:
        print(f"SKIPPED: local data not found ({', '.join(missing)}); it is not in git. Carrying on.")
        return
    if not model_cached(TINY_MODEL):
        print(f"SKIPPED: {TINY_MODEL} is not in the Hugging Face cache and the demo does not download. "
              "Carrying on.")
        return
    out = Path(tempfile.mkdtemp(prefix="terralens_demo_"))
    code = run([sys.executable, "-m", "terralens.models.encoder", "--config", DRY_RUN_CONFIG, "--model", TINY_MODEL,
                "--limit", str(limit), "--epochs", "2", "--results-dir", str(out)])
    files = list(out.glob("*.json"))
    if code != 0 or not files:
        print(f"SKIPPED: the dry run did not finish (exit code {code}). Carrying on.")
        return
    r = json.loads(files[0].read_text(encoding="utf-8"))
    print(f"\nTraining loop OK: {len(r.get('history', []))} epochs on {r['data']['train']} train rows, "
          f"validation Macro F1 = {r['val']['macro_f1']:.3f} (tiny random model; only shows the loop works)")


# ---------------------------------------------------------------- step 3
def main_table(md: str) -> list[list[str]]:
    """Rows of the first table under '## 1. Main results' in reports/results.md (header row first)."""
    part = md.split("## 1.", 1)[1].split("\n## ", 1)[0]
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in part.splitlines()
            if line.strip().startswith("|")]
    return [r for r in rows if not all(set(c) <= set("-: ") for c in r)]


def step_results() -> None:
    header(3, "Results: rebuild reports/results.md and the results figure")
    run([sys.executable, "scripts/make_results_table.py"])
    run([sys.executable, "scripts/plot_results.py"])
    rows = main_table((ROOT / "reports" / "results.md").read_text(encoding="utf-8"))
    head = rows[0]
    keep = [head.index(c) for c in ("Exp", "ID", "Val Macro F1", "Test Macro F1 [95% CI]", "Test Against F1 [95% CI]")]
    table = [[r[i] for i in keep] for r in rows]
    widths = [max(len(r[j]) for r in table) for j in range(len(keep))]
    print()
    for n, r in enumerate(table):
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if n == 0:
            print("  ".join("-" * w for w in widths))


# ---------------------------------------------------------------- step 4
def step_tests() -> int:
    header(4, "Tests: pytest -q")
    return run([sys.executable, "-m", "pytest", "-q"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=64, help="rows per split for the training dry run")
    p.add_argument("--skip-tests", action="store_true")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    start = time.time()
    step_dataset()
    step_training(args.limit)
    step_results()
    code = 0 if args.skip_tests else step_tests()
    print(f"\n{'=' * 72}\n  Demo finished in {time.time() - start:.0f} s"
          + ("" if code == 0 else f"  (pytest exit code {code})") + f"\n{'=' * 72}")
    sys.exit(code)


if __name__ == "__main__":
    main()
