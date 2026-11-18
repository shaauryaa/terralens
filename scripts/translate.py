"""Translate the English training subset into Hindi and Bengali with IndicTrans2 (T9, Source B, training-only).

Reads data/interim/translation_source.csv (from scripts/build_translation_set.py; each row has one tgt_lang)
and writes data/processed/translated.csv in the common schema:
    id, text, label, lang, script, source = "translated", video_id = "", tags = "[]", src_id, source_dataset

Model: IndicTrans2 en->indic distilled 200M. The official ai4bharat checkpoints are gated, so the default is
an ungated mirror (``naklitechie/indictrans2-en-indic-dist-200M``). Pre/post-processing uses IndicTransToolkit
when it is installed (it needs a C compiler); otherwise a fallback adds the language tags itself and
transliterates Devanagari output to Bengali script with indic-nlp-library.

Usage (GPU; on Kaggle via scripts/kaggle_jobs.py):
    python scripts/translate.py
    python scripts/translate.py --limit 4          # dry run on 4 sentences per language, prints only
The model's custom code runs with trust_remote_code, so run this in the Kaggle / Colab sandbox, not locally.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import regex as re

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from terralens.preprocess import clean_text, tag_script  # noqa: E402

SRC_CSV = ROOT / "data" / "interim" / "translation_source.csv"
OUT_CSV = ROOT / "data" / "processed" / "translated.csv"
MODEL = "naklitechie/indictrans2-en-indic-dist-200M"
# Commit whose custom code (configuration/modeling/tokenization_indictrans.py) was reviewed: standard IndicTrans2
# code, no network or shell calls. Pinned so the remote code cannot change. Needs transformers < 4.50
# (it imports transformers.onnx).
REVISION = "a814dab1ae6e4ee4c7d785b7e1dcb0ac8e36bcd6"
TGT = {"hi": "hin_Deva", "bn": "ben_Beng"}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
SEED = 42


class Processor:
    """IndicTransToolkit's IndicProcessor if available, else a minimal fallback."""

    def __init__(self):
        try:
            from IndicTransToolkit.processor import IndicProcessor  # type: ignore

            self.ip = IndicProcessor(inference=True)
        except ImportError:
            try:
                from IndicTransToolkit import IndicProcessor  # type: ignore

                self.ip = IndicProcessor(inference=True)
            except ImportError:
                self.ip = None
        self.name = "IndicTransToolkit" if self.ip else "fallback"

    def pre(self, texts: list[str], tgt: str) -> list[str]:
        if self.ip:
            return self.ip.preprocess_batch(texts, src_lang="eng_Latn", tgt_lang=tgt)
        return [f"eng_Latn {tgt} {t}" for t in texts]

    def post(self, texts: list[str], tgt: str) -> list[str]:
        if self.ip:
            return self.ip.postprocess_batch(texts, lang=tgt)
        if tgt == "ben_Beng":
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

            texts = [UnicodeIndicTransliterator.transliterate(t, "hi", "bn") if DEVANAGARI.search(t) else t
                     for t in texts]
        return [t.strip() for t in texts]


def translate(texts: list[str], tgt: str, tok, model, proc: Processor, batch_size: int, num_beams: int,
              device) -> list[str]:
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        batch = proc.pre(texts[i:i + batch_size], tgt)
        enc = tok(batch, truncation=True, padding="longest", max_length=256, return_tensors="pt",
                  return_attention_mask=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=num_beams, max_length=256, min_length=0, use_cache=True)
        try:
            with tok.as_target_tokenizer():
                dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except (AttributeError, TypeError):
            dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out.extend(proc.post(dec, tgt))
        if (i // batch_size) % 10 == 0:
            print(f"  {tgt}: {min(i + batch_size, len(texts))}/{len(texts)}", flush=True)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, default=SRC_CSV)
    p.add_argument("--out", type=Path, default=OUT_CSV)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="dry run: N sentences per language, prints only")
    args = p.parse_args(argv)

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.limit:
        src = src.groupby("tgt_lang", group_keys=False).head(args.limit)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision, trust_remote_code=True,
                                                  torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
    model.to(device).eval()
    proc = Processor()
    print(f"model={args.model} device={device} processor={proc.name} rows={len(src)}", flush=True)
    frames = []
    t0 = time.time()
    for lang, part in src.groupby("tgt_lang"):
        hyp = translate(part["text"].tolist(), TGT[lang], tok, model, proc, args.batch_size, args.num_beams, device)
        frames.append(pd.DataFrame({
            "id": "tr_" + lang + "_" + part["id"].values, "text": [clean_text(h) for h in hyp], "label": part["label"].values,
            "lang": lang, "script": [tag_script(h) for h in hyp], "source": "translated", "video_id": "",
            "tags": "[]", "src_id": part["id"].values, "source_dataset": part["source_dataset"].values}))
    out = pd.concat(frames, ignore_index=True)
    out = out[out["text"].str.len() > 0]
    print(f"translated {len(out)} rows in {(time.time() - t0) / 60:.1f} min; scripts: "
          f"{out.groupby('lang')['script'].value_counts().to_dict()}")
    if args.limit:
        for _, r in out.iterrows():
            print(f"  [{r['lang']}/{r['label']}] {r['text'][:120]}")
        print("Dry run: nothing written.")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
