"""Zero-shot and few-shot stance classification with an open instruction model in 4-bit (E1 baselines, T8).

The prompt gives the claim, the three label definitions and the main rules of the labelling guideline (no
examples). Few-shot adds k labelled comments from **train** only (fixed per-label quota, seed 42, short
comments, both languages). The model answers with one word; the first label word found in the answer is used.
Answers with no label are mapped to None and counted (``parse_failures``).

No model that was an annotator (Claude, Gemini, gpt-oss) is used here.

Outputs
    results/<exp_id>.json                        config, few-shot ids, parse failures, val + test metrics
    results/predictions/<exp_id>_{val,test}.csv  ids, gold, prediction, raw answer (no comment text)

Usage:
    python -m terralens.models.llm_prompt --config configs/e1_qwen3_8b_0shot.yaml
    python -m terralens.models.llm_prompt --config configs/e1_qwen3_8b_5shot.yaml --limit 16 \\
        --model hf-internal-testing/tiny-random-LlamaForCausalLM --no-4bit      # CPU dry run
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from terralens import CLAIM, LABELS
from terralens.reporting import ROOT, environment, split_metrics, write_predictions, write_result
from terralens.splits import load_splits

DEFAULTS = {
    "shots": 0,
    "shot_quota": {"Favour": 2, "Against": 2, "None": 1},
    "max_shot_tokens": 40,
    "max_new_tokens": 8,
    "batch_size": 8,
    "load_in_4bit": True,
    "compute_dtype": "float16",
    "seed": 42,
    "max_comment_chars": 600,
    "n_boot": 1000,
}

INSTRUCTION = f"""You classify the stance of a YouTube comment (Hindi or Bengali; native script, romanised or code-mixed) toward the claim:
"{CLAIM}"
Label what the writer believes about this claim.

Labels:
- Favour: agrees climate change / global warming is real, serious, harmful or worth acting on; criticises deniers; worries about heat, floods or melting glaciers as signs of climate change; calls for action because of climate change.
- Against: denies it, calls it a hoax, propaganda, scam or exaggeration; says it is natural or normal so not a concern; says climate does not matter compared with other issues; mocks people who worry about it.
- None: no position on the claim: off-topic, about the video or channel, greetings, spam, genuine questions, neutral facts, blame without a view on seriousness, or unclear.

Rules: mentioning weather or climate is not enough; label sarcasm by its intended meaning; watch for negation; a rhetorical question with a clear lean takes that lean; if unsure, answer None.

Answer with exactly one word: Favour, Against or None."""

LABEL_PATTERNS = [
    ("Against", re.compile(r"\bagainst\b", re.I)),
    ("Favour", re.compile(r"\bfavou?r\b", re.I)),
    ("None", re.compile(r"\bnone\b|\bneutral\b", re.I)),
]


def parse_label(answer: str) -> str | None:
    """First label word in the answer (by position), else None (Python None = parse failure)."""
    answer = re.sub(r"<think>.*?</think>", " ", answer or "", flags=re.S)
    hits = [(m.start(), lab) for lab, pat in LABEL_PATTERNS for m in [pat.search(answer)] if m]
    return min(hits)[1] if hits else None


def choose_shots(train: pd.DataFrame, quota: dict, max_tokens: int = 40, seed: int = 42) -> pd.DataFrame:
    """Few-shot examples from train: quota per label, short comments, alternating languages where possible."""
    rng = np.random.default_rng(seed)
    short = train[train["text"].str.split().str.len().between(3, max_tokens)]
    picked = []
    for lab, k in quota.items():
        pool = short[short["label"] == lab]
        langs = ["hi", "bn"]
        for j in range(k):
            sub = pool[(pool["lang"] == langs[j % 2]) & ~pool["id"].isin([p["id"] for p in picked])]
            if sub.empty:
                sub = pool[~pool["id"].isin([p["id"] for p in picked])]
            if sub.empty:
                continue
            picked.append(sub.iloc[int(rng.integers(len(sub)))].to_dict())
    order = rng.permutation(len(picked))
    return pd.DataFrame([picked[i] for i in order]) if picked else train.iloc[0:0]


def build_messages(text: str, shots: pd.DataFrame | None, max_chars: int = 600) -> list[dict]:
    msgs = [{"role": "system", "content": INSTRUCTION}]
    if shots is not None:
        for _, s in shots.iterrows():
            msgs.append({"role": "user", "content": f"Comment: {s['text'][:max_chars]}"})
            msgs.append({"role": "assistant", "content": s["label"]})
    msgs.append({"role": "user", "content": f"Comment: {text[:max_chars]}"})
    return msgs


def render(tok, msgs: list[dict]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except Exception:  # noqa: BLE001 - templates without a system role or without enable_thinking
        try:
            return tok.apply_chat_template(msgs, **kwargs)
        except Exception:  # noqa: BLE001
            merged = [{"role": "user", "content": msgs[0]["content"] + "\n\n" + msgs[1]["content"]}] + msgs[2:]
            try:
                return tok.apply_chat_template(merged, **kwargs)
            except Exception:  # noqa: BLE001 - no chat template at all (tiny test models)
                return "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\n\nassistant:"


def load_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, cfg["compute_dtype"])
    kwargs = {"dtype": dtype}
    if cfg["load_in_4bit"] and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText  # Gemma 3 multimodal checkpoints

        model = AutoModelForImageTextToText.from_pretrained(cfg["model_name"], **kwargs)
    if not torch.cuda.is_available() or not cfg["load_in_4bit"]:
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def generate(tok, model, prompts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    import torch

    out = []
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        out.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // batch_size) % 20 == 0:
            print(f"    {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return out


def run(cfg: dict, limit: int | None = None, results_dir: Path | None = None) -> dict:
    import torch

    torch.manual_seed(cfg["seed"])
    data = load_splits(limit=limit)
    shots = choose_shots(data["train"], cfg["shot_quota"], cfg["max_shot_tokens"], cfg["seed"]) if cfg["shots"] else None
    if shots is not None and len(shots) != cfg["shots"]:
        print(f"  warning: {len(shots)} shots chosen, config asks {cfg['shots']}")
    t0 = time.time()
    tok, model = load_model(cfg)
    load_s = time.time() - t0
    payload = {"kind": "llm_prompt", "config": {k: v for k, v in cfg.items() if k != "exp_id"}, "seed": cfg["seed"],
               "dry_run_limit": limit, "environment": environment(),
               "few_shot_ids": [] if shots is None else shots["id"].tolist(),
               "few_shot_labels": [] if shots is None else shots["label"].tolist(),
               "prompt_instruction": INSTRUCTION, "seconds_model_load": round(load_s, 1)}
    exp_id = cfg["exp_id"] + (f"_dryrun{limit}" if limit else "")
    rd = Path(results_dir) if results_dir else (ROOT / "outputs" / "dryrun" if limit else ROOT / "results")
    for split in ("val", "test"):
        df = data[split]
        prompts = [render(tok, build_messages(t, shots, cfg["max_comment_chars"])) for t in df["text"]]
        t1 = time.time()
        answers = generate(tok, model, prompts, cfg["max_new_tokens"], cfg["batch_size"])
        parsed = [parse_label(a) for a in answers]
        pred = [p if p is not None else "None" for p in parsed]
        payload[split] = split_metrics(df, pred, cfg["n_boot"], cfg["seed"])
        payload[split]["parse_failures"] = int(sum(p is None for p in parsed))
        payload[split]["seconds"] = round(time.time() - t1, 1)
        write_predictions(exp_id, split, df, pred, None, rd / "predictions",
                          {"answer": [a.strip().replace("\n", " ")[:60] for a in answers]})
        print(f"[{exp_id}] {split}: macroF1={payload[split]['macro_f1']} againstF1={payload[split]['against_f1']} "
              f"parse failures={payload[split]['parse_failures']}", flush=True)
    path = write_result(exp_id, payload, rd)
    print(f"wrote {path}")
    return payload


def load_config(path, overrides=None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return cfg


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="dry run: N rows per split; writes to outputs/dryrun/")
    p.add_argument("--results-dir", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config, {"model_name": args.model})
    if args.no_4bit:
        cfg["load_in_4bit"] = False
        cfg["compute_dtype"] = "float32"
    if args.limit:
        cfg["n_boot"] = 50
    run(cfg, args.limit, args.results_dir)


if __name__ == "__main__":
    main()
