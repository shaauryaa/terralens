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

## T4. Label with two full annotators and an adjudicator  (~1 day, mostly waiting)

Design: Claude and Gemini label everything; gpt-oss-120b adjudicates their disagreements.

> Write `scripts/label_llm.py`: retry on rate limits, validate JSON, resume from partial output, log failures,
> never drop rows silently. System prompt: compact guideline (rules + 10 examples,
> `annotation/prompts/compact_system.md`); API annotators return only `{"id","label","confidence","tags"}`,
> plus `"lang": "hi"|"bn"|"other"` for `lang_uncertain` rows. `prepare` writes the shared item files:
> `data/interim/labelling_items.csv` (the 3,798 rows of `comments_clean.csv` plus the 183 other_language
> comments with 3+ words, `lang_uncertain=1`) and `data/interim/gold_items.csv` (303 English gold rows of
> `data/external/climatestance_gold.csv` + a stratified 100-row subset translated once into Hindi and
> Bengali, cached in `data/interim/gold_translations.csv`).
>
> Annotators:
> - **Claude** labels everything in a separate Claude Code session (Sonnet), following
>   `annotation/prompts/claude_session.md`: `labels_claude.jsonl` and `labels_claude_gold.jsonl`, with an
>   English gloss and a short reason.
> - **Gemini** (`gemini-3.5-flash-lite`, 500 requests/day, 25 comments per request) labels the gold items and
>   all labelling items: `labels_gemini.jsonl`, `labels_gemini_gold.jsonl`.
> - **gpt-oss-120b** on Groq (reasoning medium, 50 comments per request) is adjudicator only. When both full
>   label files are complete, `label_llm.py adjudicate` runs it in priority order: (1) Claude/Gemini
>   disagreements, (2) the 303 English gold rows, (3) if budget remains, a random 300-comment sample
>   (seed 42); it stops at the daily limit and resumes the next day.
>
> Write `scripts/aggregate_labels.py`: Claude+Gemini agree -> that label; disagree -> gpt-oss decides if it
> matches one of them; 3-way split -> `review_flag=1`. `lang_uncertain` rows are kept only if at least 2
> annotators say hi or bn. Report Cohen's kappa Claude vs Gemini (overall, per language, per script) as the
> main agreement number, % agreement, how many went to gpt-oss, how many 3-way splits, Fleiss' kappa on the
> 300 sample, and the label distribution per language. Gold check: accuracy, Macro F1, per-class F1 and
> confusion matrix vs the human labels for each annotator, English (all 303 and the 178 `direct` rows) and
> translated, in `results/llm_gold_check.json`. ClimateStance counts opposing climate policy as Against
> where our guideline says None, so these scores slightly underestimate agreement; say so next to the
> numbers. The gold comments are not train, validation or test data; after this check they may be added to
> training data only. Save `data/interim/labels_merged.csv` and `results/annotation_agreement.json`.
> Summarise, then commit and push after my approval (code and results JSON only).

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
> the external data (over-sample Against). Before translating, repair the mojibake in
> `kaggle_climate_sentiment.csv` (6,109 rows, e.g. `â€¦` for "…") with `ftfy.fix_text` (add `ftfy` to
> requirements), report how many rows changed, and add a test. Then write `scripts/transliterate.py`
> (romanise part of it with IndicXlit or a rule-based fallback), and `scripts/synth_hard_cases.py` (Gemini generates negation, sarcasm, implicit and
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
