# Code flow

Every script and module in pipeline order, as **input → what it does → output**. Paths are relative to the
repo root. `data/raw/`, `data/interim/`, `data/external/`, `data/processed/`, `data/challenge/`,
`outputs/` and `results/predictions/` are not in git. Every script has `--help` and a `--limit N` dry run.

## 1. Collect and clean Source A

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 1.1 | `scripts/collect_youtube.py search` | `configs/youtube_search.yaml`, `YOUTUBE_API_KEY` | Runs the Hindi and Bengali search queries, looks up comment counts, drops small videos, deduplicates | `data/interim/candidate_videos.csv` |
| 1.2 | `scripts/collect_youtube.py screen` | `candidate_videos.csv`, `GEMINI_API_KEY` | Fetches description, tags and sample comments; Gemini judges climate relevance, language and leaning; code fills the `keep` column (caps per channel and language) | `data/interim/candidate_videos.csv` (decision columns) |
| 1.3 | `scripts/collect_youtube.py fetch` | videos with `keep = 1` | Downloads top-level comments and returned replies; masks e-mails and @handles; stores no author data | `data/raw/youtube_comments.csv` |
| 1.4 | `scripts/merge_and_dedup.py` | `youtube_comments.csv`, `candidate_videos.csv` | `clean_text`, then removes empty, emoji-only, English, other-language and too-short comments, then exact and near duplicates (char 3–5-gram TF-IDF, cosine > 0.9); tags language and script | `data/interim/comments_clean.csv`, `comments_removed.csv`, `results/comment_cleaning.json` |

## 2. Label and aggregate

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 2.1 | `scripts/label_llm.py prepare` | `comments_clean.csv`, `comments_removed.csv`, `data/external/climatestance_gold.csv` | Builds the shared item files (clean comments plus language-uncertain ones) and the gold-check items | `data/interim/labelling_items.csv`, `gold_items.csv`, `gold_translations.csv` |
| 2.2 | `scripts/label_llm.py run --annotator gemini` | item files, `annotation/prompts/compact_system.md`, `GEMINI_API_KEY` | Gemini labels every item in batches, validates the JSON, retries failures | `data/interim/labels_gemini.jsonl`, `labels_gemini_gold.jsonl` |
| 2.3 | Claude Code session (`annotation/prompts/claude_session.md`), then `scripts/label_llm.py validate --annotator claude` | the same item files | Claude labels the same items in a separate session; `validate` checks schema, coverage and duplicates | `data/interim/labels_claude.jsonl`, `labels_claude_gold.jsonl` |
| 2.4 | `scripts/label_llm.py adjudicate` | Claude and Gemini labels, `GROQ_API_KEY` | gpt-oss-120b labels the Claude/Gemini disagreements, the English gold set and a 300-comment sample | `data/interim/labels_gptoss.jsonl`, `labels_gptoss_gold.jsonl` |
| 2.5 | `scripts/aggregate_labels.py` | all `labels_*.jsonl` | Final label = agreement, else gpt-oss if it matches one, else split; language rule for uncertain rows; Cohen's and Fleiss' kappa; scores every annotator on the human gold set | `data/interim/labels_merged.csv`, `results/annotation_agreement.json`, `results/llm_gold_check.json` |
| 2.6 | `scripts/make_source_a.py` | `labels_merged.csv` | Keeps rows with a final label that pass the language rule; drops three-way splits; tags = union of the agreeing annotators' tags | `data/processed/source_a.csv`, `data/interim/dropped_splits.csv` |

## 3. Split and describe

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 3.1 | `scripts/make_splits.py` | `source_a.csv` | StratifiedGroupKFold by `video_id` (20 folds, label × language), 70/15/15; tries seeds 42–61 and keeps the first with at least 12 Against and both languages in val and test. Run once. | `data/splits/{train,val,test}_ids.txt`, `results/splits.json` |
| 3.2 | `scripts/dataset_stats.py` | `source_a.csv`, split ids, external CSVs, earlier `results/*.json` | Counts, class balance, script mix, lengths, top words, agreement summary | `results/dataset_stats.json`, `reports/dataset_stats.md`, `reports/figures/ds_*.png` |
| 3.3 | `notebooks/01_dataset_stats.ipynb` | `results/dataset_stats.json`, figures | Displays the statistics | (notebook) |

## 4. Training-only and evaluation-only data

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 4.1 | `scripts/download_external.py` | public sources (Hugging Face, GitHub, Kaggle) | Downloads SemEval-2016 climate, GWSD, ClimateStance and the Kaggle climate sentiment data; maps labels to Favour / Against / None | `data/external/*.csv`, `results/external_datasets.json` |
| 4.2 | `scripts/build_translation_set.py` | `data/external/*.csv` | Repairs mojibake with ftfy, cleans, filters by length, balances classes and assigns each English sentence to one target language | `data/interim/translation_source.csv`, `results/translation_set.json` |
| 4.3 | `scripts/translate.py` (GPU) | `translation_source.csv` | Translates English to Hindi or Bengali with IndicTrans2 (Source B) | `data/processed/translated.csv` |
| 4.4 | `scripts/transliterate.py` | `source_a.csv` train rows, `translated.csv` | Rule-based romanisation of part of the native-script training text (Source C); never reads val or test | `data/processed/romanised.csv` |
| 4.5 | `scripts/synth_hard_cases.py generate` | `GEMINI_API_KEY` | Gemini writes hard comments (negation, sarcasm, implicit, code-mixed, romanised, topic-not-target) for the synthetic set and, separately, the challenge set | `data/interim/synthetic/generated_<set>.jsonl` |
| 4.6 | `scripts/synth_hard_cases.py label` | generated comments | Gemini labels them without the intended label; writes a todo file with id and text only for Claude | `labels_gemini_<set>.jsonl`, `claude_todo_<set>.txt` |
| 4.7 | `scripts/write_claude_labels.py` | Claude's "NNNN F/A/N" lines from the Claude Code session | Converts Claude's labels to JSONL | `data/interim/synthetic/labels_claude_<set>.jsonl` |
| 4.8 | `scripts/synth_hard_cases.py finalise` | both label files | Keeps only comments where Claude and Gemini agree; reports agreement | `data/processed/synthetic.csv` (Source D), `data/challenge/challenge.csv`, `results/synthetic_data.json` |
| 4.9 | `scripts/perturb.py` | `source_a.csv` test rows | Adds character noise (vowel drops, doubled letters, swaps) and romanises part of the native-script comments | `data/processed/test_noisy.csv` |
| 4.10 | `scripts/near_dup_check.py` | B, C, D files; val, test, noisy test, challenge | Drops any training row that nearly duplicates a held-out comment (char 3–5-gram TF-IDF, cosine > 0.9) | B/C/D files updated in place, `results/near_dup_check.json` |

## 5. Experiments

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 5.1 | `scripts/run_majority.py` | `source_a.csv`, split ids | E0: always predicts the most frequent train label | `results/e0_majority.json` |
| 5.2 | `scripts/make_data_zip.py` | Source A, split ids, B/C/D, challenge, noisy test | Bundles the data a GPU run needs (private; holds comment text) | `outputs/terralens_data.zip` |
| 5.3 | `scripts/kaggle_jobs.py dataset` / `push` / `status` / `fetch` | data bundle, `src/`, `configs/`, `scripts/`, Kaggle credentials | Uploads a private Kaggle dataset; builds and runs a private GPU kernel with the given commands; downloads its results | private Kaggle dataset and kernel; `results/*.json`, `results/predictions/*` |
| 5.4 | `notebooks/02_train_gpu_colab.ipynb` | data bundle on Google Drive | Colab alternative to 5.3: installs, unzips and runs the configs | `results/*.json`, `results/predictions/*` |
| 5.5 | `python -m terralens.models.llm_prompt --config configs/e1_*.yaml` | config, Source A splits (few-shot examples from train only) | E1: zero- or few-shot prompting of a 4-bit instruction model; parses one label word | `results/e1_*.json`, `results/predictions/e1_*_{val,test}.csv` |
| 5.6 | `python -m terralens.models.encoder --config configs/e2_*.yaml` (and `e3_*`, `e4_qlora`) | config, Source A splits, `extra_train` files (B/C/D) | E2/E3: fine-tunes an encoder on (claim, comment) pairs; E4: QLoRA classifier. Early stopping on val; logit offsets tuned on val; test with and without offsets; optional `--extra-eval` sets | `results/<exp_id>.json`, `results/predictions/<exp_id>_{val,test,...}.csv` |
| 5.7 | `scripts/run_final.py` | candidate configs and their results | Picks the best config by val Macro F1, reruns it with the challenge and noisy test sets, with seeds 43 and 44, then runs E4 on the same training data | `results/final_selection.json`, `results/<best>_robust.json`, `_s43.json`, `_s44.json`, `results/e4_qlora.json` |

## 6. Analysis and reports

| Step | Script | Input | What it does | Output |
|---|---|---|---|---|
| 6.1 | `python -m terralens.robustness` | `results/*_robust.json` and seed runs | Clean vs noisy test drop, challenge-set scores per phenomenon, mean ± std over runs | `results/robustness_<exp>.json` |
| 6.2 | `scripts/error_analysis.py sample` | best run's val predictions, `source_a.csv`, `GEMINI_API_KEY` | Picks the best run by val, samples 40 misclassified val comments, gets English glosses | `data/interim/error_sample.csv`, `data/interim/error_glosses.jsonl` |
| 6.3 | `scripts/error_analysis.py notebook` | sample, glosses, `reports/error_categories.csv` (hand-assigned) | Builds and runs the error-analysis notebook (glosses, labels and categories only) | `notebooks/03_error_analysis.ipynb` |
| 6.3b | `scripts/error_analysis.py summary` | sample, `reports/error_categories.csv`, best run's val predictions | Counts the sampled gold → prediction pairs and the hand-assigned error categories (no comment text) | `results/error_analysis.json` |
| 6.4 | `scripts/make_results_table.py` | `results/*.json` | Main table, per-language/script scores, seed variance, robustness, data summary | `reports/results.md` |
| 6.5 | `scripts/plot_results.py` | `results/*.json` | Test Macro F1 with 95% CI per system; marks the best system on val | `reports/figures/results_macro_f1.png` |
| 6.6 | `scripts/make_report_figures.py` | `results/*.json` (cleaning, agreement, dataset stats, E4, robustness) | Draws the progress-report figures with one class palette (pipeline, class mix, script mix, results, E4 confusion matrix, robustness) | `reports/figures/report_*.png` |
| 6.7 | `scripts/build_report.py` | `reports/progress_report.md`, report figures | Lays out the report as an A4 PDF with reportlab and scales it until pypdf counts exactly 2 pages; `--preview` renders PNGs | `reports/progress_report.pdf` |
| 6.8 | `scripts/demo.py` (via `scripts/demo.ps1` / `scripts/demo.sh`) | `results/dataset_stats.json`, local Source A data, cached tiny model, `results/*.json` | Live CPU demo: dataset counts, a 64-row training dry run, rebuilt results table and figure, pytest (see DEMO.md) | terminal output; `reports/results.md` and `reports/figures/results_macro_f1.png` rebuilt unchanged |

## Library modules (`src/terralens/`)

| Module | Input | What it does | Output | Used by |
|---|---|---|---|---|
| `__init__.py` | – | Claim text and label constants | `CLAIM`, `LABELS` | everything |
| `preprocess.py` | raw text | `clean_text` (NFC, HTML, URLs, mentions, emoji to text, elongation, whitespace), `tag_script`, mojibake repair | cleaned text, script tag | 1.3, 1.4, 2.1, 4.2–4.5, 4.9, `splits.py` |
| `language.py` | text | Rule-based Hindi / Bengali / English / other detection, including romanised markers | language code | 1.2, 1.4 |
| `data.py` | external CSVs | Common schema, label normalisation, loaders | DataFrames | 4.1 |
| `splits.py` | `source_a.csv`, split ids, extra CSVs | Loads train / val / test; appends extra data to train only; drops extra rows whose id or text is in val/test | DataFrames | 4.4, 4.8–4.10, 5.1, 5.5, 5.6, 6.2 |
| `evaluate.py` | gold and predicted labels | Macro F1, Against F1, per-class P/R/F1, confusion matrix, per-group scores, bootstrap CIs | metrics dicts | 2.5, 5.x, 6.x |
| `thresholds.py` | val logits | Tunes per-class logit offsets on val (grid −3..3, step 0.1, Favour fixed at 0) | offsets, predictions | 5.6 |
| `reporting.py` | metrics, predictions | Writes `results/<exp_id>.json` with config, seed, environment and metrics; writes prediction CSVs | JSON, CSV | 5.1, 5.5, 5.6 |
| `robustness.py` | result JSONs with extra-eval sets | `evaluate_robustness` (see 6.1) | robustness JSON | 6.1 |
| `predict.py` | ids, labels | Writes an `id,label` prediction file | CSV | not used yet (for the final prediction file on the hidden test set) |
| `models/llm_prompt.py` | config | E1 prompting (see 5.5) | results JSON, predictions | 5.5 |
| `models/encoder.py` | config | E2–E4 training and evaluation (see 5.6) | results JSON, predictions | 5.6, 5.7 |

`legacy/` holds the early TF-IDF baseline (`legacy/baselines.py`, `legacy/predict_tfidf.py`). It is kept for
reference only and is not part of this pipeline or of any results table.
