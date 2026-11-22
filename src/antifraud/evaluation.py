"""评估工具：指标、图表、误差分析、自助置信区间。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    classification_report,
)

from .config import FIGURES_DIR, ALL_CATEGORIES

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                   "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def bootstrap_ci(metric_fn, y_true, y_pred, n_boot: int = 1000,
                 seed: int = 42, ci: float = 0.95) -> Tuple[float, float]:
    """自助法估计指标 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo = (1 - ci) / 2
    hi = 1 - lo
    return float(np.quantile(vals, lo)), float(np.quantile(vals, hi))


def plot_model_comparison(results: Dict[str, Dict], metric: str,
                          path) -> None:
    """模型对比条形图。"""
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(names, vals, color="#4c72b0")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"模型对比（{metric.upper()}）")
    ax.set_xlim(0, min(1.05, max(vals) * 1.15))
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y_true, y_pred, labels: List[str], path,
                   normalize: bool = False) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_threshold_curve(y_true, y_prob, path, beta: float = 1.0) -> None:
    """阈值扫描：P/R/F1 随阈值变化，标注最优阈值。"""
    from .models import tune_threshold
    best_t, _ = tune_threshold(y_true, y_prob, beta)
    ts = np.arange(0.05, 0.96, 0.01)
    ps, rs, fs = [], [], []
    from sklearn.metrics import precision_score, recall_score, f1_score
    for t in ts:
        pred = (y_prob >= t).astype(int)
        ps.append(precision_score(y_true, pred, zero_division=0))
        rs.append(recall_score(y_true, pred, zero_division=0))
        fs.append(f1_score(y_true, pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts, ps, label="Precision", color="#4c72b0")
    ax.plot(ts, rs, label="Recall", color="#dd8452")
    ax.plot(ts, fs, label="F1", color="#55a868")
    ax.axvline(best_t, ls="--", c="gray", lw=1)
    ax.text(best_t + 0.01, 0.5, f"最优阈值={best_t}", rotation=90, va="center")
    ax.set_xlabel("阈值")
    ax.set_ylabel("分数")
    ax.set_title("阈值扫描（验证集）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_pr(y_true, y_prob, path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, color="#4c72b0", lw=2)
    axes[0].plot([0, 1], [0, 1], ls="--", c="gray")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC (AUC={roc_auc:.3f})")
    axes[1].plot(recall, precision, color="#dd8452", lw=2)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR (AUC={pr_auc:.3f})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ablation(results: Dict[str, float], metric_name: str, path) -> None:
    names = list(results.keys())
    vals = [results[n] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.barh(names, vals, color="#55a868")
    ax.set_xlabel(metric_name)
    ax.set_title(f"消融实验（{metric_name}）")
    ax.set_xlim(0, min(1.05, max(vals) * 1.15))
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_category_recall(cat_recall: Dict[str, float], path) -> None:
    cats = list(cat_recall.keys())
    vals = [cat_recall[c] for c in cats]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(cats, vals, color="#c44e52")
    ax.set_xlabel("Recall（外部真实垃圾短信集）")
    ax.set_title("外部 FBS 数据集各类别召回率")
    ax.set_xlim(0, 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.2f}", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_analysis_df(texts: List[str], y_true, y_pred, y_prob) -> pd.DataFrame:
    df = pd.DataFrame({
        "text": texts, "label": y_true, "pred": y_pred, "prob": y_prob,
    })
    df["error"] = df["label"] != df["pred"]
    return df


def multiclass_report_table(y_true, y_pred, classes) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=classes,
                                zero_division=0, output_dict=True)
    rows = []
    for c in classes:
        d = rep[c]
        rows.append({
            "类别": c,
            "precision": round(d["precision"], 3),
            "recall": round(d["recall"], 3),
            "f1": round(d["f1-score"], 3),
            "support": d["support"],
        })
    return pd.DataFrame(rows)
