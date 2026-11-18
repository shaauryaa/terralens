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

### Cleaning, language filter and deduplication (T3)

`scripts/merge_and_dedup.py` turns the 6,224 fetched comments (64 videos) into
`data/interim/comments_clean.csv`. Counts are from `results/comment_cleaning.json`.

**Text processing**
- Text is cleaned with `terralens.preprocess.clean_text`; emojis are kept as `:name:` tokens.
- Language, script, token count and duplicate detection use the same text with emojis removed.
- Language (`terralens.language.detect_language`) is decided as follows:
  - Devanagari text is Hindi; Bengali script is Bengali.
  - Latin-script text is romanised Hindi or Bengali if it contains marker words from `ROMANISED_KEYWORDS`.
  - Forms of address (ji, bhai, dada, didi, ...) count as markers only when the comment has no English
    function words, so "Thanks Acharya ji" stays English.
  - A Hindi/Bengali tie is resolved by the video's language.
  - Other Latin-script text is English if it contains common English words, else "other".
- Each comment is removed by the first step that applies. Every removed row, with its reason (and, for
  duplicates, the id it duplicates and the similarity), is kept in `data/interim/comments_removed.csv`.

| Step | Removed | Remaining |
|---|---:|---:|
| empty: only links / @mentions | 10 | 6,214 |
| emoji_only: no letters | 354 | 5,860 |
| english | 1,264 | 4,596 |
| other_language | 493 | 4,103 |
| too_short: fewer than 3 word tokens | 236 | 3,867 |
| exact_dup: same normalised text, any video | 60 | 3,807 |
| near_dup: char 3-5-gram TF-IDF cosine > 0.9; 3 of the 9 across videos | 9 | 3,798 |

**What is kept:** 3,798 comments from all 64 videos, 339 of them from the 4 sceptic videos.

| Language | Native script | Latin (romanised) | Mixed scripts | Total |
|---|---:|---:|---:|---:|
| Hindi | 904 | 1,396 | 73 | 2,373 |
| Bengali | 1,112 | 276 | 37 | 1,425 |

Fewer than 5,000 comments remain, so no labelling sample was drawn (`comments_to_label.csv` is written only
above 5,000); all of `comments_clean.csv` goes to labelling.

**Limitation (romanised text).** The keyword rule cannot know every romanised spelling. 183 of the 493
"other_language" comments have 3 or more tokens. A manual look suggests many of them are romanised Hindi or
Bengali with unlisted spellings (e.g. "Modi chor he", "2 bochor por abar dekhlam"). These comments are in
`comments_removed.csv` and can be recovered later, for example by an LLM language check.

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
| `climatestance_gold.csv` | 303 | 178 | 60 | 65 | yes; **gold check set, not for training yet** | no licence; keep out of git |
| `climatestance_reddit_pseudo.csv` | 6208 | 6032 | 176 | 0 | yes, but labels unreliable | no licence; keep out of git |
| `climatestance_tweet_ids.csv` | 3777 | 2990 | 373 | 414 | **no** (IDs only) | no licence; IDs only |
| `kaggle_climate_sentiment.csv` | 38488 | 19644 | 3781 | 15063 | yes (mojibake in 6109 rows) | © original authors; keep out of git |

Cross-dataset overlap (normalised exact text, reported, not removed):
- `kaggle_climate_sentiment` shares 5 texts with `climatestance_reddit_pseudo`, 1 with `climatestance_gold`
  and 1 with `gwsd`.
- `climatestance_gold` and `climatestance_reddit_pseudo` share 1 text.
- No other pair of files shares a text.

Remove the gold-overlapping texts from any training data used before the T4 check is done.

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
  - `climatestance_gold`: the two annotators agree on 303 of 329 comments (92.1%). The 26 disagreements
    are in `climatestance_gold_excluded.csv` (`original_label` = `a1=..|a2=..`).
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

### Human gold check set for the LLM labellers (`climatestance_gold.csv`)

The 303 ClimateStance Reddit comments that both human annotators agreed on: Favour 178, Against 60,
None 65. By mapping confidence, 178 are `direct` and 125 `approximate`. They are kept apart, with
`split = gold_check`.

- **Purpose:** in T4, each LLM labeller (Gemini, Groq `gpt-oss-120b`, Claude) labels these comments with
  the TerraLens guideline twice: once in the original English, and once in Hindi and Bengali translation.
  We report each labeller's accuracy and Macro F1 against the human labels, in English and per
  translation language. This tells us how far to trust the LLM labels on the YouTube comments, and how much
  is lost when the text is not English.
- **Not in train, validation or test.** `terralens.data.load_all_external` refuses this file unless
  `allow_gold_check=True` is passed. After the T4 check, it may be added to **training** data only (never
  validation or test, which are real YouTube comments only).
- **Expected under-estimate.** The human labels follow ClimateStance, not our guideline:
  - ClimateStance counts **opposition to climate policy as Against**; our guideline says **None** when
    seriousness is not disputed.
  - ClimateStance puts sarcasm under Ambiguous; we label sarcasm by its intended meaning.

  A labeller that follows our guideline correctly will therefore disagree with some gold Against and None
  labels, so this check slightly **underestimates** agreement. Report scores on the 178 `direct` rows next
  to the full set, and read the remaining disagreements with that in mind.
- **Overlap:** one gold comment also appears in `kaggle_climate_sentiment` and one in
  `climatestance_reddit_pseudo` (see above).

### Twitter Climate Change Sentiment Dataset (Kaggle, Edward Qian)

Downloaded by hand from Kaggle (`edqian/twitter-climate-change-sentiment-dataset`, version 1, 2019-11-13)
as `data/external/twitter_sentiment_data.csv`. It has 43,943 rows with columns `sentiment, message,
tweetid`, no empty messages and no duplicate tweet IDs. The raw file is read, never changed.

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

- **Counts before mapping (all 43,943 rows):** Pro 22,962, News 9,276, Neutral 7,715, Anti 3,990.
- **Excluded (5,455 rows, in `kaggle_climate_sentiment_excluded.csv`):**

  | Reason | Anti | Neutral | News | Pro | Total |
  |---|---:|---:|---:|---:|---:|
  | `duplicate_text`: same text and label as an earlier row (mostly retweets) | 198 | 352 | 1422 | 3191 | 5163 |
  | `duplicate_conflicting_labels`: same text, different labels, all copies dropped | 9 | 37 | 112 | 124 | 282 |
  | `empty_text`: only @mentions and links | 2 | 5 | 0 | 3 | 10 |

  Duplicates are found after `clean_text`, lower-casing and removing a leading "RT", so a retweet counts
  as a duplicate of the original tweet.
- **Counts after mapping (38,488 rows):**
  - Favour 19,644 (51.0%), all from Pro.
  - Against 3,781 (9.8%), all from Anti.
  - None 15,063 (39.1%): 7,321 Neutral and 7,742 News.
  - By confidence: 23,425 approximate, 7,742 uncertain (News), 7,321 direct.
- **Text quality:** the text is kept as published, including @handles, links and "RT" prefixes (54% of
  kept rows start with "RT").
- **Mojibake:** 6,109 kept rows contain UTF-8 that was decoded as cp1252 (e.g. an ellipsis shown as `â€¦`).
  Re-encoding fixes most rows but garbles some, so the text is left unchanged. Repair it (e.g. with `ftfy`)
  during cleaning, before translation.

- **Licence:** the Kaggle licence field says "Data files © Original Authors", which is not an open licence.
  The file contains full tweet text with @handles. X/Twitter developer terms allow sharing tweet IDs, not
  text.
- **Attribution:** Edward Qian (Kaggle). The collection was funded by a Canada Foundation for Innovation JELF
  grant to Chris Bauch, University of Waterloo.
- **Not redistributed:** the data stays out of git.
- **To rebuild:**
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
  `data/interim/candidate_videos.csv` come from the earlier list. In T3 the list moved to
  `src/terralens/language.py` and was extended further (shared by both scripts); the screening was not
  re-run with it.
