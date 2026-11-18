# TerraLens: Milestone 2 progress report (draft)

*Stance detection on Hindi and Bengali YouTube comments toward the claim "Climate change and global warming is a
serious concern" (Favour / Against / None). Every number below comes from a file in `results/`; "pending"
means the experiment has not produced a result file yet.*

## 1. Dataset

**Source A (real YouTube comments).** 6,224 comments were collected from 64 Hindi and Bengali climate videos
(`results/comment_cleaning.json`). Cleaning removed empty, emoji-only, English-only, other-language and
very short comments and 69 exact or near duplicates, leaving 3,798; 57 of 183 comments with an uncertain
language were recovered by the annotators' language votes (`results/annotation_agreement.json`).

**Labelling.** Claude and Gemini labelled every comment with the same guideline; gpt-oss-120b adjudicated
their 435 disagreements and resolved 404. The 31 three-way splits were dropped. Claude–Gemini agreement is
Cohen's kappa 0.717 (88.7% agreement, n = 3,856); Fleiss' kappa with gpt-oss on a 300-comment sample is 0.784
(n = 294), but that sample was drawn from comments where Claude and Gemini already agreed, so it is biased
upward. On 303 human-labelled English ClimateStance comments the annotators reach Macro F1 0.637 (Claude),
0.641 (Gemini) and 0.608 (gpt-oss); on the Hindi translations Claude reaches 0.695 and Gemini 0.709
(`results/llm_gold_check.json`). Claude's Bengali gold labels are not scored because they are identical to its
Hindi labels.

**Final data and splits.** 3,825 labelled comments (814 Favour, 100 Against, 2,911 None). Against is rare
(2.6%) and concentrated in a few videos. Splits are grouped by video (StratifiedGroupKFold, seed 43, the first
of seeds 42–61 with at least 12 Against comments and both languages in validation and test;
`results/splits.json`): train 2,765 / validation 522 (19 Against) / test 538 (21 Against).

**Training-only data.** English stance data translated with IndicTrans2 into Hindi or Bengali (Source B; 7,991
English sentences selected, `results/translation_set.json`), romanised copies (C), and 470 Gemini-generated
hard comments kept only where Claude and Gemini agree on the label (D; 78.3% agreement,
`results/synthetic_data.json`). A near-duplicate check removes any training row that matches validation, test
or the challenge set (`results/near_dup_check.json`). A separate challenge set of 217 generated comments
(negation, sarcasm, implicit stance, topic-not-target) is used only for evaluation.

## 2. Method

- **E0** majority class (always None) as a floor.
- **E1** LLM prompting in 4-bit (Qwen3-8B, Gemma 3 12B), zero-shot and 5-shot with examples from train; none of
  the annotator models is used.
- **E2** encoder fine-tuning of XLM-R, MuRIL, IndicBERT v2 and mmBERT on (claim, comment) pairs: max length 128,
  lr 2e-5, batch 32, 4 epochs, 10% warm-up, fp16, class-weighted cross-entropy, early stopping on validation
  Macro F1, seed 42.
- **E3** the best E2 encoder (by validation Macro F1) plus translated data, and plus translated, romanised and
  synthetic data.
- **E4** QLoRA (4-bit Qwen3-4B with LoRA r = 16) as a 3-way classifier on the best training data.
- For every encoder, per-class logit offsets are tuned on validation, and test is reported with and without
  them. Bootstrap 95% confidence intervals (1,000 resamples) are given for Macro F1 and Against F1.
- All choices (epochs, offsets, best model) use validation only; test is scored once per configuration.
  Training ran on a Kaggle T4 GPU.

## 3. Results

Validation: 522 comments (19 Against); test: 538 comments (21 Against). Full tables, per-language and per-script
scores and CIs are in `reports/results.md` (built from `results/*.json` by `scripts/make_results_table.py`).

| Exp | System | Val Macro F1 | Test Macro F1 [95% CI] | Test Against F1 |
|---|---|---:|---:|---:|
| E0 | Majority class (None) | 0.301 | 0.268 [0.258, 0.277] | 0.000 |
| E1 | Qwen3-8B, zero-shot | 0.549 | 0.614 [0.523, 0.691] | 0.385 |
| E1 | Qwen3-8B, 5-shot | 0.525 | 0.676 [0.608, 0.736] | 0.510 |
| E1 | Gemma 3 12B, zero-shot | 0.583 | 0.672 [0.606, 0.728] | 0.423 |
| E1 | Gemma 3 12B, 5-shot | 0.546 | 0.653 [0.599, 0.700] | 0.410 |
| E2 | XLM-R base | 0.471 | 0.484 [0.456, 0.508] | 0.000 |
| E2 | MuRIL base | 0.513 | 0.446 [0.417, 0.477] | 0.000 |
| E2 | IndicBERT v2 | 0.605 | 0.574 [0.510, 0.636] | 0.238 |
| E2 | mmBERT base | 0.661 | 0.631 [0.564, 0.695] | 0.318 |
| E3 | mmBERT + translated | 0.661 | 0.585 [0.517, 0.647] | 0.200 |
| E3 | mmBERT + translated + romanised + synthetic | 0.674 | 0.561 [0.504, 0.626] | 0.143 |
| E4 | QLoRA Qwen3-4B on the E3 data (1 epoch) | **0.687** | 0.678 [0.598, 0.742] | 0.419 |

- **Best model by validation: E4 QLoRA** (val Macro F1 0.687; test 0.678, Against F1 0.419). Among encoders, mmBERT is
  clearly best; XLM-R and MuRIL never predict Against at the fixed settings (4 epochs, 60 Against in train).
- **Seed variance** (seeds 42, 43, 44): E2 mmBERT val 0.628 ± 0.028, test 0.630 ± 0.023 (Against F1 0.320 ± 0.091);
  E3 mmBERT + B + C + D val 0.665 ± 0.037, test 0.587 ± 0.024 (Against F1 0.202 ± 0.053). The training-only data
  raises validation scores but lowers test scores for every seed, so the validation gain does not carry over.
- **Validation and test disagree.** The prompted LLMs (E1) are below the encoders on validation but above them on
  test. Validation has only 19 Against and 2 Bengali Favour comments, so selection on validation is noisy; choices
  were still made on validation only, as planned.
- **Decision offsets** tuned on validation never improved test Macro F1 for the best systems (E2 mmBERT 0.628 vs
  0.631; E4 0.653 vs 0.678).
- **Robustness** (E3 mmBERT + B + C + D, 3 seeds, `results/robustness_e3_best_plus_bcd.json`): noisy test copy 0.583
  ± 0.028 vs clean 0.587 ± 0.024 (drop 0.005 ± 0.005); challenge set 0.757 ± 0.042, weakest on sarcasm (0.670) and
  implicit stance (0.684). Robustness of E4 (one retrained run, seed 42,
  `results/robustness_e4_qlora.json`; the retrain is not bit-identical: val 0.657 vs 0.687, test 0.686 vs 0.678):
  noisy test 0.677 vs clean 0.686 (drop 0.009); challenge set 0.825, weakest on implicit stance (0.704).
- **Error analysis** (40 misclassified validation comments of E4, `notebooks/03_error_analysis.ipynb`): most errors
  are blame, politics or conspiracy talk (6), comments on the topic without a stance (5), generic agreement with
  the video (5), comments that need the video for context (4), sarcasm (4) and implicit stance (4).

## 4. Challenges

- **Against is very rare** (100 of 3,825 comments; 19 in validation and 21 in test), so Against F1 has wide
  confidence intervals and per-language Bengali scores (4 Against in test) are unstable.
- **Label quality.** Labels come from LLM annotators with moderate agreement (kappa 0.717), and the human gold
  check is English Reddit data with a slightly different definition of Against.
- **Script variety.** Most Hindi comments are romanised (1,404 of 2,377), which tokenizers trained on native
  script handle poorly.
- **Gated models and tooling.** Gemma 3 and IndicTrans2 are gated on Hugging Face, so ungated mirrors were used;
  IndicTrans2's remote code needs an older transformers version.

## 5. Future work

- Human review of a sample of labels (and the 31 dropped splits) to measure LLM-label noise.
- More Against data from sceptic videos, and an ensemble of the best encoder and the LLM baseline (E5).
- Explanations (Integrated Gradients) and a fuller error analysis on validation.
- The final prediction file on the instructor's hidden test set.
