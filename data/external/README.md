# data/external

Auxiliary English climate-stance data (training only), built by:

```bash
python scripts/download_external.py                 # all datasets; existing files are not overwritten
python scripts/download_external.py --limit 50      # dry run, writes nothing
```

| File | Source | Notes |
|------|--------|-------|
| `semeval_climate.csv` | SemEval-2016 Task 6, target *Climate Change is a Real Concern* (`cardiffnlp/tweet_eval`, `stance_climate`) | train + validation + test combined; `split` column kept |
| `gwsd.csv` | Global Warming Stance Dataset (Luo et al., 2020) | label = argmax of the annotator agree / neutral / disagree columns |
| `climatestance_reddit.csv` | ClimateReddit, two human annotators (Vaid et al., 2022) | only comments both annotators agree on; the rest in `*_excluded.csv` |
| `climatestance_reddit_pseudo.csv` | ClimateReddit, model-predicted labels | labels unreliable; do not use as gold |
| `climatestance_tweet_ids.csv` | ClimateStance tweets | tweet IDs and labels only, no text |
| `kaggle_climate_sentiment.csv` | Twitter Climate Change Sentiment Dataset (Kaggle) | needs `twitter_sentiment_data.csv` downloaded by hand or via the Kaggle API |

Columns: `id, text, label, source, lang` plus `source_dataset, original_id, original_label,
original_label_name, mapping_confidence, split`. `label` is in `{Favour, Against, None}` and `lang = "en"`.
Raw downloads are cached in `climatestance_raw/` (and `kaggle_climate_sentiment_raw/` when the Kaggle API
is used). Counts, label mappings and licences are in `data/DATASHEET.md` and `results/external_datasets.json`.

Contents of this folder are git-ignored. ClimateStance has no licence and the Kaggle data is
"© Original Authors" with tweet text, so neither may be redistributed.
