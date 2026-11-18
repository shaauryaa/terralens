# TerraLens datasheet

Draft. Sections for each source, class counts and splits are added as the data is built (T2–T6); counts
will come from `scripts/dataset_stats.py`.

## Source A: YouTube comments (Hindi and Bengali)

The only source used for validation and test. Collected with `scripts/collect_youtube.py`:

1. `search`: Hindi and Bengali queries from `configs/youtube_search.yaml` (Bengali also run with region BD),
   videos with at least 30 comments kept as candidates.
2. `screen`: candidates with fewer than 20% Hindi/Bengali sample comments (keyword rule) are dropped; the
   rest are screened for climate relevance, comment language and leaning; at most 3 videos per channel and
   35 per language.
3. `fetch`: up to 150 comments per video (top-level comments plus the replies returned with them); 9 Bengali
   school essay/paragraph lessons are capped at 50 because their comments are mostly thanks to the teacher.

No author names or channel IDs are stored; @handles become `@user` and e-mail addresses `<email>`.

## External English data (training only)

Built by `scripts/download_external.py` into `data/external/` (git-ignored). Counts below come from
`results/external_datasets.json`, written by the same script. None of this data goes into validation or
test. It is English and is meant as input for translation / romanisation (sources B and C) or direct
training. It must pass the near-duplicate check against val/test before use (rule 5).

Every file keeps the source label next to the TerraLens label: `source_dataset`, `original_id`,
`original_label`, `original_label_name`, `mapping_confidence` (`direct` / `approximate` / `uncertain`) and
`split` are added to the common `id, text, label, source, lang` columns. Rows that are not kept go to
`<name>_excluded.csv` with a `reason`; nothing is dropped silently.

| File | Rows kept | Favour | Against | None | Usable text | Licence / redistribution |
|---|---:|---:|---:|---:|---|---|
| `semeval_climate.csv` | 564 | 335 | 26 | 203 | yes | not re-checked in T2 |
| `gwsd.csv` | 2050 | 777 | 400 | 873 | yes | not re-checked in T2 |
| `climatestance_reddit.csv` | 303 | 178 | 60 | 65 | yes | no licence; keep out of git |
| `climatestance_reddit_pseudo.csv` | 6208 | 6032 | 176 | 0 | yes, but labels unreliable | no licence; keep out of git |
| `climatestance_tweet_ids.csv` | 3777 | 2990 | 373 | 414 | **no** (IDs only) | no licence; IDs only |
| `kaggle_climate_sentiment.csv` | not run yet | | | | | © original authors; keep out of git |

Cross-dataset overlap (normalised exact text, reported, not removed): one comment appears in both
`climatestance_reddit` and `climatestance_reddit_pseudo`; no other pair of built files shares a text.

### ClimateStance / ClimateReddit (Vaid, Pant and Shrivastava, 2022)

- **Source:** GitHub `roopalv54/finegrained-climate-change-social-media` (the repo linked from the paper,
  `serendipity5497/...`, now redirects there), pinned to commit `fa1023b`. Paper: *Towards Fine-grained
  Classification of Climate Change related Social Media Text*, ACL 2022 Student Research Workshop.
- **What is in the repo:** (a) 3777 tweets from COP25 (1 to 14 Dec 2019) as **tweet IDs and labels only**,
  split 3021 / 378 / 378; (b) 329 Reddit comments labelled by two annotators; (c) 6262 Reddit comments
  with labels predicted by the authors' model. Reddit comments come from r/climate, r/climateskeptics,
  r/ClimateActionPlan and r/climatechange.
- **Stance target:** "climate change prevention". Favor = "expressions of opinion, action, concern against
  the climate change phenomenon"; Against = "expressions of distance, ignorance towards signs of climate
  change, extreme climates, and the opposition of climate change policies or actions taken by the governing
  bodies"; Ambiguous = "do not express any clear stance towards climate change. Tweets with sarcastic tones
  were also marked as ambiguous."
- **Label codes are not documented** in the repo or paper. The mapping 1 = favor, 0 = ambiguous,
  -1 = against was identified by matching code totals over the three tweet files (2990 / 414 / 373) to
  the class counts in the paper (Sec. 6.1), and checked by reading Reddit comments of each code.

| Source label | TerraLens | Confidence | Reason |
|---|---|---|---|
| `1` favor | Favour | direct | concern about climate change = treats it as a serious concern |
| `-1` against | Against | approximate | includes opposition to climate *policies*, which our guideline labels None when seriousness is not disputed |
| `0` ambiguous | None | approximate | sarcastic texts were put here; our guideline labels sarcasm by intended meaning (often Favour or Against) |

- **Counts before mapping (source codes):**
  - Reddit annotator 1: 1 = 185, 0 = 75, -1 = 69.
  - Reddit annotator 2: 1 = 190, 0 = 69, -1 = 70.
  - Pseudo-labelled Reddit: 1 = 6077, -1 = 185, 0 = 0.
  - Tweets: 1 = 2990, 0 = 414, -1 = 373.
- **Kept / excluded:**
  - `climatestance_reddit`: the two annotators agree on 303 of 329 comments (92.1%). The 26 disagreements
    are in `climatestance_reddit_excluded.csv` (`original_label` = `a1=..|a2=..`).
  - `climatestance_reddit_pseudo`: 54 exact duplicate texts are excluded; none had conflicting labels.
  - Tweet IDs: no duplicate IDs.
- **Do not use the pseudo labels as gold.** The model that produced them reaches stance F1 0.343 in the
  paper, it never predicts ambiguous, and it labels 2412 of the 2534 r/climateskeptics comments Favour. All
  rows carry `mapping_confidence = uncertain`. The texts can still be relabelled by our annotators.
- **Tweets have no text.** The authors released IDs only, "in accordance with the Twitter Terms of
  Service". Using them would need hydration through the paid X API, and deleted tweets are lost. The ID
  file is kept for completeness; it adds no training text.
- **Licence:** the repo has **no licence** (no LICENSE file, empty README; GitHub reports `license: null`),
  so all rights are reserved by default. Reddit comments are user content under Reddit's terms.
  **Not redistributed:** the data stays out of git; cite the paper when used. To rebuild:
  `python scripts/download_external.py --datasets climatestance`. The raw TSVs are cached in
  `data/external/climatestance_raw/`.

### Twitter Climate Change Sentiment Dataset (Kaggle, Edward Qian)

**Not run yet:** the Kaggle API is not configured on this machine, and the file has not been downloaded by
hand. The details below are from the Kaggle dataset metadata (`edqian/twitter-climate-change-sentiment-dataset`,
version 1, 2019-11-13). File names and columns could not be checked without the file. The script expects
`twitter_sentiment_data.csv` with `sentiment, message, tweetid` and stops with an error if the columns differ.

- **Content:** 43,943 tweets about climate change, 27 Apr 2015 to 21 Feb 2018. Each tweet was labelled by
  3 reviewers, and only tweets all 3 agreed on were kept.
- **Not a sentiment dataset.** Despite its name, the classes describe *belief in man-made climate change*
  (a stance), not positive or negative sentiment. That target is close to ours but not the same.

| Source label | TerraLens | Confidence | Reason |
|---|---|---|---|
| `1` Pro: "supports the belief of man-made climate change" | Favour | approximate | belief in the human cause is close to, but not the same as, calling it a serious concern |
| `-1` Anti: "does not believe in man-made climate change" | Against | approximate | may deny the human cause rather than the seriousness |
| `0` Neutral: "neither supports nor refutes the belief of man-made climate change" | None | direct | no position |
| `2` News: "links to factual news about climate change" | None | uncertain | our guideline makes neutral facts None, but headlines about harms can imply Favour |

- **Licence:** the Kaggle licence field says "Data files © Original Authors", which is not an open licence.
  The file contains full tweet text with @handles. X/Twitter developer terms allow sharing tweet IDs, not
  text.
- **Attribution:** Edward Qian (Kaggle). The collection was funded by a Canada Foundation for Innovation JELF
  grant to Chris Bauch, University of Waterloo.
- **Not redistributed:** the data stays out of git.
- **To add it:**
  1. Sign in to Kaggle and download the dataset from
     https://www.kaggle.com/datasets/edqian/twitter-climate-change-sentiment-dataset
  2. Unzip it and put `twitter_sentiment_data.csv` in `data/external/`.
  3. Run `python scripts/download_external.py --datasets kaggle`.

  Alternatively, install `kaggle`, configure `~/.kaggle/kaggle.json` (or `KAGGLE_USERNAME` / `KAGGLE_KEY`
  in `.env`) and run step 3. The raw file is never overwritten; the output is written to
  `kaggle_climate_sentiment.csv`.

### SemEval-2016 climate and GWSD

Same as before T2. Both target (almost) our claim, so all rows are `mapping_confidence = direct`.
- SemEval-2016 Task 6 "Climate Change is a Real Concern": FAVOR / AGAINST / NONE map to
  Favour / Against / None.
- GWSD: the argmax of the agree / disagree / neutral annotator scores maps to Favour / Against / None;
  250 attention-check rows without scores are dropped.

Their licences were not re-checked in this task. The CSVs on disk predate T2 and were not overwritten; the
provenance columns appear after a re-run with `--overwrite`.

## Known limitations

- **Few sceptic videos.** In the video screening of 2026-11-18 (`data/interim/candidate_videos.csv`), only
  4 of the 64 kept videos (6%) have a sceptic leaning, against a target of at least 15%. 19 of the kept
  videos were found by the sceptic-leaning search queries, but most of them are mainstream news or
  explainers. Sceptic Hindi and Bengali videos with enough comments were hard to find, so the Against class
  is likely to be small; some Against comments may still come from sceptical comments under news videos.
  This was accepted for Milestone 2 without an extra search.
- **Screening used the earlier keyword list.** The rule-based language check in the video screening ran
  before ten romanised words were added to `ROMANISED_KEYWORDS` in `scripts/collect_youtube.py` (valo, vaiya,
  bhaiya, bahut, achha, accha, nai, hoy, theke, ar). With the new list, 7 more candidates would pass the 20%
  pre-filter and 2 videos dropped for low Hindi/Bengali share would pass the rule. They were not re-screened;
  the 64 kept videos were left unchanged, and the `rule_indic_share` values in
  `data/interim/candidate_videos.csv` come from the earlier list.
