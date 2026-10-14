# data/external

Auxiliary English climate-stance data, downloaded by:

```bash
python scripts/download_external.py
```

| File | Source | Notes |
|------|--------|-------|
| `semeval_climate.csv` | SemEval-2016 Task 6, target *Climate Change is a Real Concern* (via `cardiffnlp/tweet_eval`, config `stance_climate`) | train + validation + test splits combined; `split` column kept |
| `gwsd.csv` | Global Warming Stance Dataset (Luo et al., 2020) | label = argmax of the annotator agree / neutral / disagree columns |

All files use the common schema `id, text, label, source, lang`, with
`label` in `{Favour, Against, None}` and `lang = "en"`.

Contents of this folder are git-ignored. Check each dataset's licence before redistributing.
