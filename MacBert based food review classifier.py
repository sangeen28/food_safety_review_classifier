"""
Food Safety Review Classifier (Chinese) - Colab GPU (12GB) friendly
CV + OOF-threshold optimization for F1.

 Please, read these instructions first to make most of the proposed framework.
 for any issues, contact: sangeenkhan2662@gmail.com

What you get:
- 5-fold Stratified CV training
- OOF probabilities for robust threshold selection (maximize F1)
- Test probabilities averaged across folds (and optionally across 2 backbones)
- Outputs: submission.csv + submission_with_probs.csv + oof.csv
- Plots saved to ./outputs_food_safety_cv/plots/

How to run (Colab):
1) Upload train.csv (TAB: label<TAB>comment) and test.csv (CSV: id,comment)
2) Install:
   !pip -q install -U transformers accelerate scikit-learn pandas matplotlib
3) Run:
   !python food_review_framework_cv_full.py

Notes for 12GB GPU:
- Default uses ONE backbone (MacBERT) to keep runtime reasonable.
- To enable 2-model ensemble, set CFG.model_names to two models (see below).
"""

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)

# -----------------------------
# 0) Config
# -----------------------------
@dataclass
class CFG:
    # paths (auto-detected if you keep default)
    train_path: str = "train.csv"
    test_path: str = "test.csv"
    out_dir: str = "./outputs_food_safety_cv"

    # Models (minimal but effective)
    # Default: single strong model (fastest + good gain via CV)
    model_names: Tuple[str, ...] = ("hfl/chinese-macbert-base",)

    # Optional: 2-model ensemble (usually stronger if you can afford extra training)
    # model_names: Tuple[str, ...] = ("hfl/chinese-macbert-base", "hfl/chinese-roberta-wwm-ext")

    max_len: int = 192

    # CV
    n_folds: int = 5

    # training (12GB GPU friendly)
    seed: int = 42
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    train_bs: int = 16
    eval_bs: int = 32
    grad_accum: int = 2      # effectively batch 32 while fitting 12GB
    fp16: bool = True
    patience: int = 2

    # threshold tuning
    tune_threshold: bool = True
    threshold_grid_steps: int = 401  # fine grid gives small LB gains

    # plotting (NEW)
    make_plots: bool = True


# -----------------------------
# 1) Robust data loading
# -----------------------------
def resolve_path(p: str) -> str:
    candidates = [
        p,
        os.path.join("/content", p),
        os.path.join("/mnt/data", p),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return p

def load_train(train_path: str) -> pd.DataFrame:
    train_path = resolve_path(train_path)
    try:
        df = pd.read_csv(train_path, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(train_path, dtype=str)

    if df.shape[1] >= 2:
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "comment"})
        df = df[["label", "comment"]].copy()
    else:
        df = pd.read_csv(train_path, header=None, names=["label", "comment"], sep="\t", dtype=str)

    df = df[df["label"].astype(str) != "label"].copy()
    df["label"] = df["label"].astype(int)
    df["comment"] = df["comment"].astype(str).fillna("")
    df = df.dropna().reset_index(drop=True)
    return df

def load_test(test_path: str) -> pd.DataFrame:
    test_path = resolve_path(test_path)
    df = pd.read_csv(test_path)
    assert "id" in df.columns and "comment" in df.columns, "test.csv must contain columns: id, comment"
    df["comment"] = df["comment"].astype(str).fillna("")
    return df


# -----------------------------
# 2) Task prompt + lightweight tags
# -----------------------------
RISK_KEYWORDS = [
    "腹泻", "拉肚子", "呕吐", "恶心", "腹痛", "肚子疼", "中毒",
    "虫子", "苍蝇", "蟑螂", "老鼠", "头发", "塑料", "铁丝",
    "变质", "发霉", "腐烂", "异味", "臭味", "馊味", "过期", "不新鲜",
    "生的", "没熟", "夹生", "不干净", "卫生差", "脏", "细菌",
]

def normalize_text(x: str) -> str:
    x = x.strip()
    x = " ".join(x.split())
    x = x.replace("。。。", "…").replace("....", "…")
    return x

def add_task_prompt_and_tags(x: str) -> str:
    x = normalize_text(x)
    hit = any(k in x for k in RISK_KEYWORDS)
    tag = "[食品安全线索]" if hit else "[无明显安全线索]"
    prompt = "任务: 判断评价是否包含食品安全/卫生风险投诉(1=是,0=否)。评价: "
    return f"{prompt}{tag}{x}"


# -----------------------------
# 3) Dataset wrapper
# -----------------------------
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]))
        return item


# -----------------------------
# 4) Weighted loss Trainer
# -----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# -----------------------------
# 5) Utils
# -----------------------------
def softmax_probs(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    probs = softmax_probs(logits)
    pred = probs.argmax(axis=1)
    acc = accuracy_score(labels, pred)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
    return {"accuracy": acc, "f1": f1, "precision": p, "recall": r}

def find_best_threshold(y_true: np.ndarray, p1: np.ndarray, steps: int = 401):
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, steps):
        pred = (p1 >= t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1

def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")

def build_training_args(cfg: CFG, fold_dir: str) -> TrainingArguments:
    import inspect
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = dict(
        output_dir=fold_dir,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.train_bs,
        per_device_eval_batch_size=cfg.eval_bs,
        gradient_accumulation_steps=cfg.grad_accum,
        logging_steps=50,
        max_grad_norm=1.0,
    )

    if "evaluation_strategy" in allowed:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in allowed:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in allowed:
        kwargs["save_strategy"] = "epoch"
    if "load_best_model_at_end" in allowed:
        kwargs["load_best_model_at_end"] = True
    if "metric_for_best_model" in allowed:
        kwargs["metric_for_best_model"] = "f1"
    if "greater_is_better" in allowed:
        kwargs["greater_is_better"] = True
    if cfg.fp16 and "fp16" in allowed:
        kwargs["fp16"] = True
    if "report_to" in allowed:
        kwargs["report_to"] = "none"
    if "save_total_limit" in allowed:
        kwargs["save_total_limit"] = 1

    return TrainingArguments(**kwargs)


# -----------------------------
# 6) Plotting (NEW) - does not change training logic
# -----------------------------
def _ensure_plots_dir(out_dir: str) -> str:
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_threshold_vs_f1(y: np.ndarray, p1: np.ndarray, steps: int, out_path: str):
    ts = np.linspace(0.01, 0.99, steps)
    f1s = [f1_score(y, (p1 >= t).astype(int)) for t in ts]
    best_idx = int(np.argmax(f1s))
    plt.figure(figsize=(8, 5))
    plt.plot(ts, f1s)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("OOF F1 vs Threshold")
    plt.axvline(ts[best_idx])
    _savefig(out_path)

def plot_confusion_matrix(cm: np.ndarray, out_path: str):
    plt.figure(figsize=(5.5, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (OOF)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    _savefig(out_path)

def plot_prob_hist(y: np.ndarray, p1: np.ndarray, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.hist(p1[y == 0], bins=40, alpha=0.7, label="label=0")
    plt.hist(p1[y == 1], bins=40, alpha=0.7, label="label=1")
    plt.xlabel("OOF predicted probability p(y=1)")
    plt.ylabel("Count")
    plt.title("OOF Probability Distribution by Class")
    plt.legend()
    _savefig(out_path)

def plot_pr_curve(y: np.ndarray, p1: np.ndarray, out_path: str):
    precision, recall, _ = precision_recall_curve(y, p1)
    ap = average_precision_score(y, p1)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AP={ap:.4f})")
    _savefig(out_path)

def plot_roc_curve(y: np.ndarray, p1: np.ndarray, out_path: str):
    fpr, tpr, _ = roc_curve(y, p1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={roc_auc:.4f})")
    _savefig(out_path)

def plot_calibration_curve(y: np.ndarray, p1: np.ndarray, out_path: str):
    frac_pos, mean_pred = calibration_curve(y, p1, n_bins=10, strategy="uniform")
    plt.figure(figsize=(6.5, 5.5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (OOF)")
    _savefig(out_path)

def plot_fold_metrics(fold_f1s: List[float], out_path: str):
    xs = np.arange(1, len(fold_f1s) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.bar(xs, fold_f1s)
    plt.xticks(xs, [str(i) for i in xs])
    plt.xlabel("Fold")
    plt.ylabel("F1 @ threshold=0.5")
    plt.title("Per-fold F1@0.5")
    _savefig(out_path)

def plot_training_curves(log_histories: List[List[dict]], out_path: str):
    """
    Aggregated training/eval loss curves across folds.
    log_histories: list of trainer.state.log_history per fold
    """
    epochs = []
    train_loss = []
    eval_loss = []
    eval_f1 = []

    for hist in log_histories:
        for row in hist:
            if "epoch" not in row:
                continue
            ep = float(row["epoch"])
            if "loss" in row and "eval_loss" not in row:
                epochs.append(ep)
                train_loss.append(float(row["loss"]))
            if "eval_loss" in row:
                eval_loss.append((ep, float(row["eval_loss"])))
            if "eval_f1" in row:
                eval_f1.append((ep, float(row["eval_f1"])))

    eval_loss = sorted(eval_loss, key=lambda x: x[0])
    eval_f1 = sorted(eval_f1, key=lambda x: x[0])

    plt.figure(figsize=(8, 5))
    if len(epochs) > 0 and len(train_loss) > 0:
        pairs = sorted(zip(epochs, train_loss), key=lambda x: x[0])
        plt.plot([p[0] for p in pairs], [p[1] for p in pairs], label="train_loss")
    if len(eval_loss) > 0:
        plt.plot([p[0] for p in eval_loss], [p[1] for p in eval_loss], label="eval_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Eval Loss (aggregated across folds)")
    plt.legend()
    _savefig(out_path)

    if len(eval_f1) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot([p[0] for p in eval_f1], [p[1] for p in eval_f1])
        plt.xlabel("Epoch")
        plt.ylabel("F1")
        plt.title("Eval F1 (aggregated across folds)")
        _savefig(out_path.replace(".png", "_f1.png"))


# -----------------------------
# 7) Train one backbone with CV
# -----------------------------
def train_cv_for_model(cfg: CFG, model_name: str, texts: List[str], y: np.ndarray, test_texts: List[str]):
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    oof_p1 = np.zeros(len(y), dtype=np.float64)
    test_p1_folds = []

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[食品安全线索]", "[无明显安全线索]"]})
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    fold_f1s = []
    fold_log_histories = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold_dir = os.path.join(cfg.out_dir, f"{safe_name(model_name)}_fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]
        tr_y = y[tr_idx]
        va_y = y[va_idx]

        counts = np.bincount(tr_y, minlength=2).astype(np.float32)
        total = counts.sum()
        weights = total / (2.0 * np.maximum(counts, 1.0))
        class_weights = torch.tensor(weights, dtype=torch.float32)

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.resize_token_embeddings(len(tokenizer))

        tr_ds = SimpleDataset(tr_texts, tr_y.tolist(), tokenizer, cfg.max_len)
        va_ds = SimpleDataset(va_texts, va_y.tolist(), tokenizer, cfg.max_len)
        te_ds = SimpleDataset(test_texts, None, tokenizer, cfg.max_len)

        args = build_training_args(cfg, fold_dir)

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=args,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            tokenizer=tokenizer,
            data_collator=collator,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
        )

        trainer.train()

        # store logs for plots (does not affect training)
        fold_log_histories.append(list(trainer.state.log_history))

        va_logits = trainer.predict(va_ds).predictions
        va_p1 = softmax_probs(va_logits)[:, 1]
        oof_p1[va_idx] = va_p1

        te_logits = trainer.predict(te_ds).predictions
        te_p1 = softmax_probs(te_logits)[:, 1]
        test_p1_folds.append(te_p1)

        f1_05 = f1_score(va_y, (va_p1 >= 0.5).astype(int))
        fold_f1s.append(float(f1_05))
        print(f"[{model_name}] fold {fold}/{cfg.n_folds} F1@0.5 = {f1_05:.4f}")

        del trainer, model
        torch.cuda.empty_cache()

    test_p1 = np.mean(np.vstack(test_p1_folds), axis=0)
    return oof_p1, test_p1, fold_f1s, fold_log_histories


# -----------------------------
# 8) Main
# -----------------------------
def main():
    cfg = CFG()
    os.makedirs(cfg.out_dir, exist_ok=True)

    set_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_df = load_train(cfg.train_path)
    test_df = load_test(cfg.test_path)

    texts = [add_task_prompt_and_tags(t) for t in train_df["comment"].tolist()]
    y = train_df["label"].astype(int).to_numpy()
    test_texts = [add_task_prompt_and_tags(t) for t in test_df["comment"].tolist()]

    all_oof, all_test = [], []
    all_fold_f1s = {}          # model_name -> list
    all_log_histories = {}     # model_name -> list-of-fold-log_history

    for model_name in cfg.model_names:
        print("\n==============================")
        print("Training backbone:", model_name)
        print("==============================")
        oof_p1, test_p1, fold_f1s, fold_logs = train_cv_for_model(cfg, model_name, texts, y, test_texts)
        all_oof.append(oof_p1)
        all_test.append(test_p1)
        all_fold_f1s[model_name] = fold_f1s
        all_log_histories[model_name] = fold_logs

    oof_p1_ens = np.mean(np.vstack(all_oof), axis=0)
    test_p1_ens = np.mean(np.vstack(all_test), axis=0)

    # Threshold tuned on OOF for F1 (matches competition metric)
    best_t = 0.5
    best_f1 = f1_score(y, (oof_p1_ens >= 0.5).astype(int))
    if cfg.tune_threshold:
        best_t, best_f1 = find_best_threshold(y, oof_p1_ens, steps=cfg.threshold_grid_steps)

    oof_pred = (oof_p1_ens >= best_t).astype(int)
    cm = confusion_matrix(y, oof_pred)
    acc = accuracy_score(y, oof_pred)
    p, r, f1, _ = precision_recall_fscore_support(y, oof_pred, average="binary", zero_division=0)

    print("\n=== OOF (ensemble) metrics at best threshold ===")
    print("Best threshold:", best_t)
    print("OOF Accuracy :", acc)
    print("OOF Precision:", p)
    print("OOF Recall   :", r)
    print("OOF F1       :", f1)
    print("Confusion matrix:\n", cm)

    # Save OOF
    oof_df = train_df.copy()
    oof_df["p1"] = oof_p1_ens
    oof_df["pred"] = oof_pred
    oof_path = os.path.join(cfg.out_dir, "oof.csv")
    oof_df.to_csv(oof_path, index=False, encoding="utf-8")
    print("Saved:", oof_path)

    # Submission
    sub = pd.DataFrame({"id": test_df["id"], "label": (test_p1_ens >= best_t).astype(int)})
    sub_path = os.path.join(cfg.out_dir, "submission.csv")
    sub.to_csv(sub_path, index=False, encoding="utf-8")
    print("Saved:", sub_path)

    subp = pd.DataFrame({"id": test_df["id"], "p1": test_p1_ens, "label": (test_p1_ens >= best_t).astype(int)})
    subp_path = os.path.join(cfg.out_dir, "submission_with_probs.csv")
    subp.to_csv(subp_path, index=False, encoding="utf-8")
    print("Saved:", subp_path)

    # -----------------------------
    # 9) Plots (NEW) - no change to framework logic
    # -----------------------------
    if cfg.make_plots:
        plots_dir = _ensure_plots_dir(cfg.out_dir)

        plot_threshold_vs_f1(y, oof_p1_ens, cfg.threshold_grid_steps, os.path.join(plots_dir, "oof_threshold_vs_f1.png"))
        plot_confusion_matrix(cm, os.path.join(plots_dir, "oof_confusion_matrix.png"))
        plot_prob_hist(y, oof_p1_ens, os.path.join(plots_dir, "oof_prob_hist.png"))
        plot_pr_curve(y, oof_p1_ens, os.path.join(plots_dir, "oof_pr_curve.png"))
        plot_roc_curve(y, oof_p1_ens, os.path.join(plots_dir, "oof_roc_curve.png"))
        plot_calibration_curve(y, oof_p1_ens, os.path.join(plots_dir, "oof_calibration_curve.png"))

        for model_name, f1s in all_fold_f1s.items():
            plot_fold_metrics(f1s, os.path.join(plots_dir, f"{safe_name(model_name)}_fold_f1_at_0p5.png"))

        for model_name, logs in all_log_histories.items():
            plot_training_curves(logs, os.path.join(plots_dir, f"{safe_name(model_name)}_training_curves.png"))

        print("Saved plots to:", plots_dir)


if __name__ == "__main__":
    main()
