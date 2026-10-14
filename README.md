# TerraLens

**Multilingual stance detection on Hindi and Bengali YouTube comments about climate change.**

TerraLens is a course project for **CSET 346 (Natural Language Processing), Bennett University**.
Given a YouTube comment written in Hindi or Bengali (in native script, romanised, or code-mixed with
English), the task is to classify its stance toward a fixed target claim as **Favour**, **Against** or
**None**. Comments of this kind are short, noisy and multi-script, and the Against class is expected to be
the minority. The project therefore compares classical n-gram baselines, Indic and multilingual
transformers, and LLM prompting, and pays particular attention to minority-class performance.

> **Status: proposed, no results yet.** The official dataset has not been released. This repository
> currently contains the project scaffold, the preprocessing pipeline, baseline code and loaders for
> auxiliary English data. No numbers in this repository are project results.

## Target claim and labels

> **"Climate change and global warming is a serious concern."**

| Label | Meaning |
|-------|---------|
| **Favour** | The comment supports the claim: it treats climate change / global warming as real, serious or worth acting on. |
| **Against** | The comment opposes the claim: denial, dismissal, calling it a hoax or exaggeration, or arguing it is not a concern. |
| **None** | The comment expresses no stance toward the claim (off-topic, neutral information, unrelated questions, spam). |

## Team

| Name | Role | Enrolment no. |
|------|------|---------------|
| Shaurya Nigam | Team Leader | S24CSEU0497 |
| Nakshatra Bhateja | Member | S24CSEU1245 |
| [Member 3] | Member | |
| [Member 4] | Member | |

**Mentor:** Dr. Supriya Chanda

## Repository structure

```
terralens/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── pyproject.toml              # makes src/terralens installable (pip install -e .)
├── configs/
│   └── baseline.yaml           # TF-IDF + LR / LinearSVC settings
├── data/
│   ├── raw/                    # official shared-task CSVs (id, text, label), not committed
│   ├── processed/              # derived files, not committed
│   └── external/               # auxiliary English data, not committed
├── notebooks/
│   └── 01_eda.ipynb            # EDA skeleton
├── scripts/
│   └── download_external.py    # fetches SemEval-2016 climate + GWSD
├── src/terralens/
│   ├── data.py                 # loaders and label mapping
│   ├── preprocess.py           # normalisation and script tagging
│   ├── baselines.py            # TF-IDF + LR / LinearSVC, stratified 5-fold CV
│   ├── evaluate.py             # macro F1, Against F1, per-class, confusion matrix, per-language
│   └── predict.py              # writes predictions (id, label)
└── tests/
    └── test_preprocess.py
```

## Setup

```bash
git clone https://github.com/shaauryaa/terralens.git
cd terralens

python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .                      # so `python -m terralens.*` works from anywhere

python scripts/download_external.py   # auxiliary English data -> data/external/
pytest                                 # unit tests
```

`torch` and `transformers` are only needed for the transformer fine-tuning stage. They can be skipped
while working on the classical baselines.

### Running

```bash
# Baselines with stratified 5-fold CV on the official training set (once released, in data/raw/train.csv)
python -m terralens.baselines --config configs/baseline.yaml

# Same pipeline on an external dataset (pipeline check only, not a project result)
python -m terralens.baselines --config configs/baseline.yaml --dataset semeval_climate

# Predictions file with columns id,label
python -m terralens.predict --train data/raw/train.csv --test data/raw/test.csv --out outputs/predictions.csv
```

## Datasets

### Official dataset: pending

The Hindi and Bengali YouTube comment dataset has **not been released yet**. When it is, place the files
in `data/raw/` as CSVs with columns `id, text, label` (`label` in `Favour / Against / None`).

### External (auxiliary, English)

These are used only as optional extra training signal, for example cross-lingual transfer or
pre-fine-tuning. `scripts/download_external.py` maps them to the common schema
`id, text, label, source, lang`.

| Dataset | What it is | Label mapping |
|---------|-----------|---------------|
| **SemEval-2016 Task 6, climate target** | English tweets on *"Climate Change is a Real Concern"*, loaded via [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) (`stance_climate`) | 0 → None, 1 → Against, 2 → Favour |
| **GWSD** (Global Warming Stance Dataset) | English news sentences annotated for stance toward *"Climate change/global warming is a serious concern"*, from [yiweiluo/GWStance](https://github.com/yiweiluo/GWStance) | argmax of the agree / neutral / disagree scores: agree → Favour, disagree → Against, neutral → None. Annotator screening sentences are removed. |

**Citations**

- Saif M. Mohammad, Svetlana Kiritchenko, Parinaz Sobhani, Xiaodan Zhu and Colin Cherry. 2016.
  *SemEval-2016 Task 6: Detecting Stance in Tweets.* In Proceedings of SemEval-2016.
  <https://aclanthology.org/S16-1003/>
- Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa Anke and Leonardo Neves. 2020.
  *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* In Findings of EMNLP 2020.
  <https://aclanthology.org/2020.findings-emnlp.148/>
- Yiwei Luo, Dallas Card and Dan Jurafsky. 2020. *Detecting Stance in Media on Global Warming.*
  In Findings of EMNLP 2020. <https://aclanthology.org/2020.findings-emnlp.296/>

## Planned methodology

1. **Preprocessing** (`preprocess.py`): Unicode NFC; removal of URLs, HTML and @mentions;
   emoji → text tokens; elongation fix (character repeats capped at 2); script tagging
   (Devanagari U+0900–097F, Bengali U+0980–09FF, Latin) to separate native-script, romanised and
   code-mixed comments.
2. **Classical baselines**: TF-IDF over word 1–2-grams and character 2–5-grams, with
   Logistic Regression and LinearSVC (`class_weight="balanced"`).
3. **FastText**: subword embeddings / supervised FastText classifier.
4. **Transformers**: fine-tune MuRIL, IndicBERT, XLM-R and mBERT. Optionally pre-fine-tune on the
   English external data before fine-tuning on Hindi and Bengali.
5. **LLMs**: zero-shot and few-shot prompting as a comparison point.
6. **Handling imbalance and robustness**: class weights, data augmentation (e.g. back-translation,
   transliteration), and ensembles of the strongest models.
7. **Interpretability**: LIME and SHAP explanations, plus error analysis of misclassified
   Against comments.

## Evaluation

- **Primary metric: Macro F1** over Favour, Against and None.
- **Against-class F1** is reported separately, since it is the expected minority class.
- Per-class precision / recall / F1, confusion matrix, and a per-language / per-script breakdown
  (`evaluate.py`).
- Model selection and hyperparameter tuning use **stratified 5-fold cross-validation** on the
  training data only. **The test set is never used for tuning.**

## Roadmap (14 Oct – 20 Nov 2026)

| Dates | Milestone |
|-------|-----------|
| 14 – 20 Oct | Repository setup, preprocessing, external data loaders, TF-IDF baseline pipeline |
| 21 – 27 Oct | EDA on the official data (once released); TF-IDF + LR / SVM baselines; FastText |
| 28 Oct – 6 Nov | Transformer fine-tuning: MuRIL, IndicBERT, XLM-R, mBERT |
| 7 – 13 Nov | LLM zero/few-shot; class weighting, augmentation, ensembles |
| 14 – 20 Nov | Error analysis, LIME / SHAP, final report and presentation |

The schedule after 20 Oct depends on when the official dataset is released.

## Status

**Proposed, no results yet.** Results will be added here once the official dataset is available
and experiments have been run.

## License

[MIT](LICENSE)
