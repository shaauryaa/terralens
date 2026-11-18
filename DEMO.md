# Live demo

A short live demo of TerraLens that runs on a laptop CPU in about half a minute (33 s with PowerShell and 27 s with
bash on our Windows laptop), with **no GPU, no API keys and no downloads**. It uses only files already in the
repo, the local data, and the Hugging Face cache.

## Run it

Windows (PowerShell, from the repo root):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

Linux, macOS or Git Bash:

```bash
bash scripts/demo.sh
```

Both wrappers switch Hugging Face to offline mode (`HF_HUB_OFFLINE=1`) and run `scripts/demo.py` with the
project's `.venv` Python if it exists. Options: `--skip-tests` (steps 1–3 only) and `--limit N` (rows per split for
the training dry run, default 64).

## What it shows

| Step | What happens | Where the numbers come from |
|---|---|---|
| 1. Dataset | Prints the total, per-class, per-language and per-split counts of the 3,825 labelled comments | `results/dataset_stats.json` |
| 2. Training | Runs `terralens.models.encoder` on 64 rows per split with the tiny random model `hf-internal-testing/tiny-random-bert` for 2 epochs on the CPU, then prints its validation Macro F1 | a temporary folder (nothing is written to `results/`) |
| 3. Results | Rebuilds `reports/results.md` and `reports/figures/results_macro_f1.png` from the result files and prints the main table (E0 to E4) | `results/*.json` |
| 4. Tests | Runs `pytest -q` | `tests/` |

Step 2 only shows that the training and evaluation loop works end to end (data loading, class-weighted loss,
early stopping, threshold tuning, metrics). Its score is meaningless: the model is a tiny random network trained on
64 comments. The real models were trained on a Kaggle T4 GPU (see the README).

Step 3 rewrites the two report files with the same content, so the repository stays unchanged after the demo.

## If something is missing

The demo never stops on a missing piece; it prints `SKIPPED: ...` and carries on.

- **Tiny model not cached.** Step 2 is skipped. To cache it once (needs internet):
  `python -c "from huggingface_hub import snapshot_download; snapshot_download('hf-internal-testing/tiny-random-bert')"`
- **No local data.** Step 2 needs `data/processed/source_a.csv` and `data/splits/`, which hold comment text and are not
  in git. On a fresh clone step 2 is skipped; steps 1, 3 and 4 still run from the committed result files.

## Example output (shortened)

```
  STEP 1  Dataset (results/dataset_stats.json)
Source A: 3,825 labelled YouTube comments from 64 videos
  all       Favour   814  Against   100  None 2,911
  Hindi     Favour   637  Against    66  None 1,674   total 2,377
  Bengali   Favour   177  Against    34  None 1,237   total 1,448
  train     Favour   584  Against    60  None 2,121   total 2,765  (43 videos)
  val       Favour    74  Against    19  None   429   total   522  (11 videos)
  test      Favour   156  Against    21  None   361   total   538  (10 videos)

  STEP 2  Training: tiny CPU dry run of the encoder (64 rows per split)
  epoch 1: loss=1.1002 val macroF1=0.0822 againstF1=0.0000 time=1s
  epoch 2: loss=1.0986 val macroF1=0.0822 againstF1=0.0000 time=1s
Training loop OK: 2 epochs on 64 train rows, validation Macro F1 = 0.082 (tiny random model; only shows the loop works)

  STEP 3  Results: rebuild reports/results.md and the results figure
Exp  ID                   Val Macro F1  Test Macro F1 [95% CI]  Test Against F1 [95% CI]
E0   e0_majority          0.301         0.268 [0.258, 0.277]    0.000 [0.000, 0.000]
...
E4   e4_qlora             0.687         0.678 [0.598, 0.742]    0.419 [0.210, 0.583]

  STEP 4  Tests: pytest -q
221 passed in 6.38s

  Demo finished in 33 s
```
