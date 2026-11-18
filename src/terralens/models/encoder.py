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
import yaml

from terralens import CLAIM, LABEL2ID, LABELS
from terralens.reporting import ROOT, environment, split_metrics, write_predictions, write_result
from terralens.splits import load_splits
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


class PairDataset:
    def __init__(self, tokenizer, claim: str, texts, labels, max_length: int):
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
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL2ID,
        ignore_mismatched_sizes=True)
    model.to(device)
    collate = DataCollatorWithPadding(tok)
    g = torch.Generator()
    g.manual_seed(cfg["seed"])
    mk = lambda df: PairDataset(tok, cfg["claim"], df["text"].tolist(), df["label"].tolist(), cfg["max_length"])  # noqa: E731
    train_loader = DataLoader(mk(train), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(mk(val), batch_size=cfg["eval_batch_size"], collate_fn=collate)
    test_loader = DataLoader(mk(test), batch_size=cfg["eval_batch_size"], collate_fn=collate)

    weights = class_weights(train["label"]) if cfg["class_weighted"] else np.ones(len(LABELS))
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]))
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    history, best, best_state, bad_epochs = [], -1.0, None, 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
            running += loss.item() * cfg["grad_accum"]
            n_batches += 1
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
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stopping_patience"]:
                print(f"  early stop after epoch {epoch}", flush=True)
                break
    model.load_state_dict(best_state)

    val_logits = predict_logits(model, val_loader, device, fp16)
    test_logits = predict_logits(model, test_loader, device, fp16)
    offsets, val_tuned_f1 = tune_offsets(val_logits, val["label"])
    val_pred, test_pred = predict_labels(val_logits), predict_labels(test_logits)
    val_pred_t, test_pred_t = predict_labels(val_logits, offsets), predict_labels(test_logits, offsets)
    nb, seed = cfg["n_boot"], cfg["seed"]
    exp_id = cfg["exp_id"] + (f"_s{seed}" if seed != 42 else "") + (f"_dryrun{limit}" if limit else "")
    payload = {
        "kind": "encoder",
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
    args = p.parse_args(argv)
    cfg = load_config(args.config, {"seed": args.seed, "model_name": args.model, "epochs": args.epochs,
                                    "save_model_dir": args.save_model_dir})
    if args.limit:
        cfg["n_boot"] = 50
    train_and_evaluate(cfg, limit=args.limit, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
