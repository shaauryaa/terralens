# TerraLens: rules for Claude Code

TerraLens detects the stance of Hindi and Bengali YouTube comments (native script, romanised or
code-mixed) toward the claim **"Climate change and global warming is a serious concern."**
Labels: `Favour`, `Against`, `None`. Primary metric: **Macro F1**. Always also report Against F1,
accuracy, per-class precision/recall, a confusion matrix and per-language scores.

## Data rules (never break these)

1. **Only real YouTube comments (Source A) go into validation and test.** Translated (B), romanised /
   code-mixed (C) and synthetic (D) data are training-only.
2. **Split by video.** `data/splits/{train,val,test}_ids.txt` are made once by `scripts/make_splits.py`
   (StratifiedGroupKFold on `video_id`, seed 42, 70/15/15) and then never changed. Every model uses them.
3. **Never tune on test.** Learning rate, epochs, class weights, thresholds, prompts and few-shot examples
   are chosen on validation. Test is scored once per final configuration.
4. Few-shot examples for prompting come from **train** only.
5. Before training on B/C/D, run the near-duplicate check against val/test and drop any matches.
6. The LLM used to produce labels is **not** reported as a zero-shot baseline.
7. Never commit raw comments with usernames, API keys, `.env`, model checkpoints or files over 50 MB.
   `data/raw/`, `data/interim/`, `data/external/` and `outputs/` stay out of git.
8. The instructor's hidden test file is used only to write the final prediction file.

## Results rules

- Every number in the README, report or slides must come from a JSON file in `results/`, written by
  the code that produced it (`results/<experiment_id>.json` with config, seed, metrics, timestamp).
- No TF-IDF / classical model appears in results tables. That code lives in `legacy/` only.
- Majority-class is shown only as a floor (E0).
- Say "estimated" or "not run yet" for anything not backed by a results file. Never invent numbers.

## Code conventions

- Python 3.10+, package in `src/terralens/`, scripts in `scripts/`, one YAML per experiment in `configs/`.
- Every script has `--help`, a `--limit N` dry-run option, and a fixed seed.
- Reuse `terralens.preprocess.clean_text`, `tag_script` and `terralens.evaluate.evaluate`; do not
  re-implement metrics.
- GPU code must run on a single Colab T4 (16 GB): fp16, max_length 128, gradient accumulation if needed.
  CPU-only steps must run on a laptop with ~11 GB RAM and no CUDA.
- Add or update a test in `tests/` for every new data-processing function.
- Data CSV schema: `id, text, label, lang, script, source, video_id` (`video_id` empty for non-YouTube).

## Working style

- **Progress updates:** while a task runs, post short, plain updates (one or two lines) at each step:
  what was just done, what is running now and what comes next. Say so when something long starts
  (notebook runs, API calls, training) and when it finishes or fails.
- **Approval gate before every commit and push:** when the work is done and `pytest` passes, stop and give
  a short summary: (1) what was changed and why, (2) test / dry-run results, (3) the exact list of files
  that will be committed (and anything deliberately left out), (4) the commit message, (5) the branch it
  will be pushed to. Then wait for the user's explicit approval. Do not commit or push without it.

## Git rules

- Commits are authored by the repo owner only. **Do not add `Co-Authored-By` lines, "Generated with
  Claude Code" lines or any AI attribution to commit messages or PR descriptions.**
- Small commits with plain messages in the imperative ("Add video-grouped split script").
- Never list team members or a contributions section in the README.
- Run `pytest` before committing; do not commit failing tests.
- Commit and push only after the approval gate above (see "Working style").

## Where things are

- Plan: TerraLens Milestone 2 plan (Claude Doc).
- Labelling guideline: `annotation/guidelines.md`. Prompts: `annotation/prompts/`.
- Dataset documentation: `data/DATASHEET.md` (keep counts in sync with `scripts/dataset_stats.py`).
