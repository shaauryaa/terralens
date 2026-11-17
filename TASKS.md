# TerraLens Milestone 2: Claude Code task list

Run these in order, one Claude Code session per task (or `/clear` between tasks). Paste the prompt in
the box as it is. Each task ends with tests or a dry run, then a short summary (what changed, test
results, files to be committed, commit message, target branch) and a commit + push to `main` **only after
you approve**. Claude Code posts short progress updates while the task runs. Where it says **[Colab]**, Claude
Code writes the code locally and you run it on the Colab T4 through `notebooks/02_train_gpu_colab.ipynb`.

Keys you need, all kept in `.env` (never committed):
`YOUTUBE_API_KEY`, `GEMINI_API_KEY` (aistudio.google.com, free), `GROQ_API_KEY` (console.groq.com, free).

---

## T0. Set up the repo for Milestone 2  (~15 min)

> Read CLAUDE.md and the existing repo. Then: (1) move `src/terralens/baselines.py`, `configs/baseline.yaml`
> and the TF-IDF parts of the notebooks into `legacy/` and update imports so tests still pass; (2) add `.env`,
> `outputs/`, `data/raw/`, `data/interim/`, `data/external/` and model checkpoints to `.gitignore`;
> (3) add `python-dotenv`, `google-api-python-client`, `google-genai`, `groq`, `peft`, `accelerate`,
> `bitsandbytes`, `datasets`, `evaluate` to requirements (GPU-only ones in a separate `requirements-gpu.txt`);
> (4) create the folders from the plan: `annotation/prompts`, `data/splits`, `data/challenge`, `results`,
> `src/terralens/models`; (5) remove the team table from README and put a short "work in progress" note.
> Run pytest, summarise what you did and what you will commit, then commit and push after my approval.

## T1. Collect YouTube comments  (~45 min incl. API time)

> Write `scripts/collect_youtube.py` using the YouTube Data API v3 and `configs/youtube_search.yaml`.
> Mode `search`: run the Hindi and Bengali queries and write `data/interim/candidate_videos.csv`
> (video_id, title, channel, published_at, comment_count, query, lang). Deduplicate videos.
> Mode `fetch`: for videos with `keep=1` in that CSV, fetch top-level comments (and replies, flagged) up to
> `max_comments_per_video`, writing `data/raw/youtube_comments.csv` with id, video_id, text, like_count,
> published_at, is_reply, query_lang. Store no author names or channel IDs. Handle quota errors and resume.
> Add `--limit` for a dry run. Test the parsing on a saved sample response. Summarise, then commit and
> push after my approval (no data).

Then: run `search`, open the CSV, set `keep=1` on ~50 to 70 relevant videos (Claude Code can pre-fill
`keep` by title, you skim it), make sure some sceptic-leaning ones are included, run `fetch`.

## T2. External English data  (~30 min)

> Extend `scripts/download_external.py` to also fetch ClimateStance (Vaid et al. 2022,
> github.com/serendipity5497/finegrained-climate-change-social-media) and the Twitter Climate Change
> Sentiment dataset (Kaggle; if the Kaggle API is not configured, print manual download steps and read the
> CSV from data/external/). Map labels to Favour/Against/None (pro→Favour, anti→Against,
> neutral/news→None; ClimateStance favor/against/ambiguous). Report class counts and licence for each in
> `data/DATASHEET.md`. Summarise, then commit and push after my approval.

## T3. Clean, filter, deduplicate  (~30 min)

> Write `scripts/merge_and_dedup.py`: clean text with `terralens.preprocess.clean_text`, tag script with
> `tag_script`, detect language (Hindi / Bengali / English / other) using script plus a romanised-Hindi and
> romanised-Bengali keyword heuristic, drop English-only, other-language, emoji-only and <3-token comments,
> remove exact and near duplicates (char 3-5-gram cosine > 0.9 within and across videos), and write
> `data/interim/comments_clean.csv` with id, video_id, text, lang, script. Print a filter report (how many
> rows each step removed). Add tests. Summarise, then commit and push after my approval.

## T4. Label with three LLM annotators  (~1 to 2 h, mostly waiting)

> Write `scripts/label_llm.py` with annotators `gemini` (GEMINI_API_KEY) and `groq` (a Llama model
> on Groq's free tier; GROQ_API_KEY), using `annotation/guidelines.md` as the system prompt and the JSON
> output format in it. Batch 25 comments per request, retry on rate limits, resume from partial output,
> write `data/interim/labels_<annotator>.jsonl`. Then label the same file yourself as annotator `claude`:
> read `comments_clean.csv` in batches of 100 and write `labels_claude.jsonl` in the same format, following
> the guideline strictly. Finally write `scripts/aggregate_labels.py`: majority vote, flag 3-way splits,
> compute Fleiss' kappa, pairwise Cohen's kappa and % unanimous, save `data/interim/labels_merged.csv` and
> `results/annotation_agreement.json`. Summarise, then commit and push after my approval (code only).

## T5. Human check on English glosses  (~1 h of your time)

> Write `scripts/make_review_sheet.py` that creates `data/interim/review.csv` with: all 3-way splits, plus a
> random 150 items (seed 42) with labels hidden. Columns: id, original text, English gloss, your_label
> (empty), notes. After I fill it, `scripts/apply_review.py` writes the final labels to
> `data/processed/source_a.csv` and computes Cohen's kappa between my labels and the majority on the 150
> blind items, saved to `results/annotation_agreement.json`. Summarise, then commit and push after my
> approval (code only).

## T6. Split and dataset statistics  (~30 min)

> Write `scripts/make_splits.py` (StratifiedGroupKFold by video_id, 70/15/15, seed 42, writes
> `data/splits/*_ids.txt`) and `notebooks/01_dataset_stats.ipynb`: counts per source, class distribution
> overall and per language and per split, script mix, comment length, top words per class (stopwords
> removed), duplicates removed, agreement scores. Save figures to `reports/figures/` and a summary to
> `reports/dataset_stats.md`. Update DATASHEET. Summarise, then commit and push after my approval
> (splits, figures and summary; not raw text).

## T7. Encoder fine-tuning  [Colab]  (~20 min code, ~1 h GPU)

> Write `src/terralens/models/encoder.py`: Hugging Face fine-tuning of a sequence-pair classifier
> (claim, comment), class-weighted cross-entropy, fp16, max_length 128, early stopping on validation Macro
> F1, saving `results/<exp_id>.json` via `terralens.evaluate.evaluate` with per-language breakdown and the
> confusion matrix. Configs: `configs/e2_xlmr.yaml`, `e2_muril.yaml`, `e2_indicbertv2.yaml`,
> `e2_mmbert.yaml` (check the exact Hugging Face IDs exist). Write `notebooks/02_train_gpu_colab.ipynb` that
> clones the repo, installs requirements-gpu, uploads/pulls data, runs a list of configs and zips
> `results/` for download. Test locally with a tiny model and `--limit 64` on CPU. Summarise, then commit
> and push after my approval.

## T8. LLM prompting baselines  [Colab]  (~1 h)

> Write `src/terralens/models/llm_prompt.py`: zero-shot and 5-shot (examples from train, balanced) stance
> classification with an open instruction model in 4-bit on a T4 (e.g. Qwen3-8B and Gemma 3 12B; confirm
> IDs). Do not use any model that was an annotator. Parse a single label, map failures to None and count
> them. Evaluate on val and test, write results JSON. Summarise, then commit and push after my approval.

## T9. Training-data expansion  [Colab]  (~1 to 2 h)

> Write `scripts/translate.py` (IndicTrans2 en→hi and en→bn, batched on GPU) for a class-balanced subset of
> the external data (over-sample Against), `scripts/transliterate.py` (romanise part of it with IndicXlit or a
> rule-based fallback), and `scripts/synth_hard_cases.py` (Gemini generates negation, sarcasm, implicit and
> code-mixed comments per label using the guideline; save with source=synthetic). Run the near-duplicate
> check against val/test. Add configs `e3_best_plus_b.yaml`, `e3_best_plus_bc.yaml`, `e3_best_plus_bcd.yaml`.
> Summarise, then commit and push after my approval.

## T10. Robustness set and perturbations  (~45 min)

> Write `scripts/make_challenge_set.py`: (a) tag test items by phenomenon using annotator tags; (b) generate
> ~300 hard cases (negation, sarcasm, implicit, topic-not-target, romanised) with Gemini, label them by the
> three-annotator vote, put them in `data/challenge/`; (c) write `scripts/perturb.py` making a noisy copy of
> the test set (vowel drops, doubled letters, swapped characters, transliteration). Add an
> `evaluate_robustness` entry point that reports Macro F1 per slice and clean-vs-noisy drop. Summarise,
> then commit and push after my approval.

## T11. Results, README, progress report  (~1.5 h)

> Write `scripts/make_results_table.py` that builds `reports/results.md` only from `results/*.json`. Rewrite
> README: task, dataset (sources, counts, labelling, agreement), methods, results table, how to reproduce,
> how to predict on a new CSV (`python -m terralens.predict`), model info, licence. Then draft
> `reports/progress_report.md` (two pages: dataset, method and implementation, results, challenges, future
> work) using only numbers from results files. Summarise, then commit and push after my approval.

## Later (final submission)

- E4 QLoRA on a ~4B instruction model, E5 ensemble, Integrated Gradients explanations, error analysis
  notebook, prediction file on the instructor's test set, final paper.
