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
