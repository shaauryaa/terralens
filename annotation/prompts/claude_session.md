# T4: Claude annotator session (run in a separate Claude Code session, Sonnet)

You are annotator `claude`, one of two full annotators (the other is Gemini, run by `scripts/label_llm.py`).
Label independently: do not open any `labels_gemini*` or `labels_gptoss*` file.

## Read first
- `annotation/guidelines.md`: the full labelling guideline (claim, labels, the 10 decision rules, 18 worked
  examples, output format). Follow it strictly.

## Inputs (do not modify)
- `data/interim/labelling_items.csv`: 3,981 items with `id, video_id, text, lang, script, lang_uncertain`.
  For the 183 rows with `lang_uncertain = 1`, also decide the comment's language.
- `data/interim/gold_items.csv`: 503 items with `id, gold_id, set, lang, text, ...`: 303 English Reddit
  comments (`set = gold_en`) and 200 Hindi / Bengali translations of 100 of them (`gold_hi`, `gold_bn`).
  Ignore the `human_label` and `mapping_confidence` columns: do not read them before labelling.

## Outputs (JSON Lines, one object per item, UTF-8)
- `data/interim/labels_claude.jsonl`: one line per id in `labelling_items.csv`
- `data/interim/labels_claude_gold.jsonl`: one line per id in `gold_items.csv`

Each line (required fields only; annotator, model and set come from the file name and the run):

```json
{"id": "<id>", "label": "Favour|Against|None", "confidence": "high|medium|low",
 "tags": ["sarcasm", "negation", "implicit", "code_mixed", "romanised", "question"], "lang": "hi|bn|other"}
```

- `tags`: only those that apply, else `[]` (always a list).
- `lang`: only for `lang_uncertain = 1` rows (Hindi, Bengali, or `other` for anything else incl. English).
- `gloss` / `reason` are optional. English glosses for the comments that go to human review are written by
  Gemini after aggregation (`python scripts/label_llm.py gloss`).
- Gold translations (`gold_hi`, `gold_bn`): label each one from its translated text alone, as if you had
  never seen the English original or its label.

## Procedure
1. Work in batches of 100 rows, in file order. Before each batch, read the existing output file and skip ids
   already labelled, so the work resumes after an interruption.
2. Append each finished batch to the output file (do not rewrite earlier lines).
3. Use only the comment text and the guideline. No video titles, no other annotators' labels.
4. When both files are complete, run:
   `python scripts/label_llm.py validate --annotator claude`
   It must report `missing: 0`, `invalid: 0` and `duplicate_ids: 0` for both files. Fix and re-run if not.
5. Do not commit anything; `data/interim/` is git-ignored.
