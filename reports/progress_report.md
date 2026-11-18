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

RESULTS_PLACEHOLDER

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
