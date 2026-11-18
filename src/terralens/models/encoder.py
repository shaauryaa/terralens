"""Fine-tune a multilingual encoder as a (claim, comment) sentence-pair stance classifier (E2 / E3).

Training: class-weighted cross-entropy (weights = n / (3 * count) from the training labels), AdamW, linear
warm-up then decay, fp16 autocast on CUDA, gradient accumulation, early stopping on validation Macro F1 (the
best epoch's weights are kept). Seed fixed. Time per epoch is logged.

Evaluation: validation and test, each with Macro F1, Against F1, accuracy, per-class P/R/F1, confusion
matrix, bootstrap 95% CIs, and breakdowns per language, script and annotator tag. Per-class logit offsets are
tuned on validation only (``terralens.thresholds``) and test is reported both with and without them.

Outputs
    results/<exp_id>.json                       config, seed, history, metrics
    results/predictions/<exp_id>_{val,test}.csv ids, gold, predictions, logits (no comment text)

Usage:
    python -m terralens.models.encoder --config configs/e2_xlmr.yaml
    python -m terralens.models.encoder --config configs/e2_xlmr.yaml --seed 43        # exp_id gets _s43
    python -m terralens.models.encoder --config configs/e2_mmbert.yaml --exp-suffix _robust         --extra-eval challenge=data/challenge/challenge.csv test_noisy=data/processed/test_noisy.csv
    python -m terralens.models.encoder --config configs/e2_xlmr.yaml --limit 64 \\
        --model hf-internal-testing/tiny-random-xlm-roberta                              # CPU dry run
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from terralens import CLAIM, LABEL2ID, LABELS
from terralens.reporting import ROOT, environment, split_metrics, write_predictions, write_result
from terralens.splits import load_splits, normalise_frame
from terralens.thresholds import predict_labels, tune_offsets

DEFAULTS = {
    "claim": CLAIM,
    "max_length": 128,
    "lr": 2e-5,
    "batch_size": 32,
    "grad_accum": 1,
    "epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "fp16": True,
    "class_weighted": True,
    "early_stopping_patience": 2,
    "seed": 42,
    "extra_train": [],
    "eval_batch_size": 64,
    "n_boot": 1000,
    "extra_eval": {},
    "exp_suffix": "",
}


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if "exp_id" not in cfg or "model_name" not in cfg:
        raise ValueError(f"{path}: exp_id and model_name are required")
    return cfg


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights(labels) -> np.ndarray:
    counts = np.array([max(1, sum(1 for y in labels if y == lab)) for lab in LABELS], dtype=float)
    return counts.sum() / (len(LABELS) * counts)


PROMPT_TEMPLATE = ("Claim: {claim}\n"
                   "Comment: {text}\n"
                   "Stance of the comment toward the claim (Favour, Against or None):")


class PairDataset:
    """(claim, comment) sentence pairs for encoders; ``prompt=True`` gives one prompt string (decoder models)."""

    def __init__(self, tokenizer, claim: str, texts, labels, max_length: int, prompt: bool = False):
        if prompt:
            ids = [tokenizer(t, truncation=True, max_length=max_length - 40, add_special_tokens=False)["input_ids"]
                   for t in texts]
            short = [tokenizer.decode(i) for i in ids]
            self.enc = tokenizer([PROMPT_TEMPLATE.format(claim=claim, text=t) for t in short], truncation=True,
                                 max_length=max_length)
        else:
            self.enc = tokenizer([claim] * len(texts), list(texts), truncation="only_second", max_length=max_length)
        self.labels = [LABEL2ID[y] for y in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def predict_logits(model, loader, device, fp16: bool) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                logits = model(**batch).logits
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def load_qlora_model(cfg: dict, tok, device):
    """4-bit base model with a LoRA sequence-classification head (QLoRA, PEFT). CUDA only for 4-bit."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    q = cfg["qlora"]
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    kwargs = {"num_labels": len(LABELS), "id2label": dict(enumerate(LABELS)), "label2id": LABEL2ID}
    if device.type == "cuda":
        kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                             bnb_4bit_compute_dtype=torch.float16,
                                                             bnb_4bit_use_double_quant=True),
                      device_map={"": 0}, dtype=torch.float16)
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if device.type == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=q.get("gradient_checkpointing", True))
    lora = LoraConfig(task_type="SEQ_CLS", r=q.get("r", 16), lora_alpha=q.get("alpha", 32),
                      lora_dropout=q.get("dropout", 0.05),
                      target_modules=q.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                              "up_proj", "down_proj"]))
    model = get_peft_model(model, lora)
    for n, p in model.named_parameters():  # the classification head is trained in fp32
        if "score" in n and p.requires_grad:
            p.data = p.data.float()
    if device.type != "cuda":
        model.to(device)
    model.print_trainable_parameters()
    return model


def train_and_evaluate(cfg: dict, limit: int | None = None, results_dir: Path | None = None,
                       pred_dir: Path | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              get_linear_schedule_with_warmup)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16 = bool(cfg["fp16"]) and device.type == "cuda"
    data = load_splits(extra_train=cfg["extra_train"], limit=limit)
    train, val, test = data["train"], data["val"], data["test"]
    print(f"[{cfg['exp_id']}] model={cfg['model_name']} device={device} fp16={fp16} "
          f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    qlora = cfg.get("qlora")
    if qlora:
        model = load_qlora_model(cfg, tok, device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
            ignore_mismatched_sizes=True)
        model.to(device)
    collate = DataCollatorWithPadding(tok)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    mk = lambda df: PairDataset(tok, cfg["claim"], df["text"].tolist(), df["label"].tolist(), cfg["max_length"],  # noqa: E731
                                prompt=bool(qlora))
    train_loader = DataLoader(mk(train), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(mk(val), batch_size=cfg["eval_batch_size"], collate_fn=collate)
    test_loader = DataLoader(mk(test), batch_size=cfg["eval_batch_size"], collate_fn=collate)

    weights = class_weights(train["label"]) if cfg["class_weighted"] else np.ones(len(LABELS))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    params = [
        {"params": [p for n, p in trainable if not any(nd in n for nd in no_decay)], "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in trainable if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]))
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    history, best, best_state, bad_epochs = [], -1.0, None, 0
    budget = cfg.get("time_budget_minutes")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        running, n_batches = 0.0, 0
        opt.zero_grad()
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=fp16):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if step % cfg["grad_accum"] == 0 or step == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            running += loss.item() * cfg["grad_accum"]
            n_batches += 1
            if budget and epoch == 1 and step == 20:
                per_step = (time.time() - t0) / 20
                projected = per_step * len(train_loader) * cfg["epochs"] / 60
                fit = int(budget * 60 // (per_step * len(train_loader) * 1.15))  # 15% for evaluation
                print(f"  projected training time {projected:.0f} min for {cfg['epochs']} epochs "
                      f"(budget {budget} min)", flush=True)
                if fit < 1:
                    write_result(cfg["exp_id"] + cfg.get("exp_suffix", ""), {
                        "kind": "skipped", "config": {k: v for k, v in cfg.items() if k != "exp_id"},
                        "seed": cfg["seed"], "reason": f"projected {projected:.0f} min > budget {budget} min"},
                        Path(results_dir) if results_dir else ROOT / "results")
                    raise SystemExit(f"skipped: one epoch alone would exceed the {budget} min budget")
                if fit < cfg["epochs"]:
                    print(f"  reducing epochs {cfg['epochs']} -> {fit} to fit the budget", flush=True)
                    cfg["epochs"] = fit
        train_time = time.time() - t0
        val_logits = predict_logits(model, val_loader, device, fp16)
        val_pred = predict_labels(val_logits)
        from terralens.evaluate import against_f1, macro_f1

        vf, va = macro_f1(val["label"], val_pred), against_f1(val["label"], val_pred)
        history.append({"epoch": epoch, "train_loss": round(running / max(1, n_batches), 4),
                        "val_macro_f1": round(vf, 4), "val_against_f1": round(va, 4),
                        "epoch_seconds": round(train_time, 1), "eval_seconds": round(time.time() - t0 - train_time, 1)})
        print(f"  epoch {epoch}: loss={history[-1]['train_loss']} val macroF1={vf:.4f} againstF1={va:.4f} "
              f"time={train_time:.0f}s", flush=True)
        if vf > best + 1e-9:
            best, bad_epochs = vf, 0
            keep = {n for n, _ in trainable} if qlora else None
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                          if keep is None or k in keep}  # QLoRA: only the trained LoRA / head weights
            best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print(f"  early stop after epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state, strict=not qlora)

    val_logits = predict_logits(model, val_loader, device, fp16)
    test_logits = predict_logits(model, test_loader, device, fp16)
    offsets, val_tuned_f1 = tune_offsets(val_logits, val["label"])
    val_pred, test_pred = predict_labels(val_logits), predict_labels(test_logits)
    val_pred_t, test_pred_t = predict_labels(val_logits, offsets), predict_labels(test_logits, offsets)
    nb, seed = cfg["n_boot"], cfg["seed"]
    exp_id = (cfg["exp_id"] + (f"_s{seed}" if seed != 42 else "") + cfg.get("exp_suffix", "")
              + (f"_dryrun{limit}" if limit else ""))
    payload = {
        "kind": "qlora" if qlora else "encoder",
        "config": {k: v for k, v in cfg.items() if k != "exp_id"},
        "seed": seed,
        "dry_run_limit": limit,
        "environment": environment(),
        "data": {"train": len(train), "val": len(val), "test": len(test),
                 "train_by_source": {str(k): int(v) for k, v in train["source"].value_counts().items()},
                 "train_label": {lab: int((train["label"] == lab).sum()) for lab in LABELS},
                 "class_weights": [round(float(w), 4) for w in weights]},
        "history": history,
        "best_epoch": best_epoch,
        "seconds_per_epoch_mean": round(float(np.mean([h["epoch_seconds"] for h in history])), 1),
        "selection": "best epoch by val Macro F1; offsets tuned on val Macro F1",
        "val": split_metrics(val, val_pred, nb, seed),
        "test": split_metrics(test, test_pred, nb, seed),
        "thresholds": {
            "offsets": dict(zip(LABELS, offsets)),
            "val_macro_f1_tuned": val_tuned_f1,
            "note": "offsets added to logits, tuned on val only; val_tuned is optimistic (tuned on itself)",
            "val_tuned": split_metrics(val, val_pred_t, nb, seed),
            "test_tuned": split_metrics(test, test_pred_t, nb, seed),
        },
    }
    rd = Path(results_dir) if results_dir else (ROOT / "outputs" / "dryrun" if limit else ROOT / "results")
    pdir = Path(pred_dir) if pred_dir else rd / "predictions"
    # Extra evaluation-only sets (challenge set, noisy test copy): reported, never used for any choice.
    for name, rel in (cfg.get("extra_eval") or {}).items():
        path_e = ROOT / rel
        if not path_e.exists():
            print(f"  extra eval {name}: {rel} not found, skipped", flush=True)
            continue
        df_e = normalise_frame(pd.read_csv(path_e, dtype=str, keep_default_na=False))
        if limit:
            df_e = df_e.head(limit)
        loader_e = DataLoader(mk(df_e), batch_size=cfg["eval_batch_size"], collate_fn=collate)
        logits_e = predict_logits(model, loader_e, device, fp16)
        pred_e, pred_e_t = predict_labels(logits_e), predict_labels(logits_e, offsets)
        payload.setdefault("extra_eval", {})[name] = {"source": rel, "plain": split_metrics(df_e, pred_e, nb, seed),
                                                      "tuned": split_metrics(df_e, pred_e_t, nb, seed)}
        write_predictions(exp_id, name, df_e, pred_e, logits_e, pdir, {"pred_tuned": pred_e_t})
        print(f"  extra eval {name}: n={len(df_e)} macroF1={payload['extra_eval'][name]['plain']['macro_f1']}", flush=True)
    path = write_result(exp_id, payload, rd)
    write_predictions(exp_id, "val", val, val_pred, val_logits, pdir, {"pred_tuned": val_pred_t})
    write_predictions(exp_id, "test", test, test_pred, test_logits, pdir, {"pred_tuned": test_pred_t})
    print(f"[{exp_id}] val macroF1={payload['val']['macro_f1']} test macroF1={payload['test']['macro_f1']} "
          f"test(tuned)={payload['thresholds']['test_tuned']['macro_f1']} -> {path}", flush=True)
    if cfg.get("save_model_dir"):
        out = Path(cfg["save_model_dir"]) / exp_id
        model.save_pretrained(out)
        tok.save_pretrained(out)
    return payload


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=None, help="override the config seed (exp_id gets _s<seed>)")
    p.add_argument("--model", default=None, help="override model_name (e.g. a tiny model for a dry run)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="dry run: N rows per split; writes to outputs/dryrun/")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--save-model-dir", default=None, help="also save the best model here (not in git)")
    p.add_argument("--extra-eval", nargs="*", default=None, metavar="NAME=PATH",
                   help="evaluation-only CSVs, e.g. challenge=data/challenge/challenge.csv")
    p.add_argument("--exp-suffix", default=None, help="appended to exp_id (e.g. _robust)")
    args = p.parse_args(argv)
    cfg = load_config(args.config, {"seed": args.seed, "model_name": args.model, "epochs": args.epochs,
                                    "save_model_dir": args.save_model_dir, "exp_suffix": args.exp_suffix})
    if args.extra_eval:
        cfg["extra_eval"] = dict(kv.split("=", 1) for kv in args.extra_eval)
    if args.limit:
        cfg["n_boot"] = 50
    train_and_evaluate(cfg, limit=args.limit, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
