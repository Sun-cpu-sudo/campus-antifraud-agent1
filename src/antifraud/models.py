"""模型训练、阈值调优、结果保存与加载。"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC

from .config import RANDOM_STATE, MODELS_DIR, ALL_CATEGORIES, NO_RISK_LABEL
from .features import build_features, make_char_features, make_word_features


def get_classifiers() -> Dict[str, Any]:
    """候选分类器（含基线）。"""
    return {
        "Dummy(多数类)": DummyClassifier(strategy="most_frequent"),
        "MultinomialNB": MultinomialNB(alpha=0.5, fit_prior=True),
        "ComplementNB": ComplementNB(alpha=0.5),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "LinearSVC": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight="balanced", max_iter=2000,
                      random_state=RANDOM_STATE),
            method="sigmoid", cv=3,
        ),
    }


FEATURE_BUILDERS = {
    "word": make_word_features,
    "char": make_char_features,
    "word+char": build_features,
}


def _binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def fit_predict_proba(clf, X_train, y_train, X_test) -> Tuple[Any, np.ndarray]:
    """训练并返回正类概率。LinearSVC 已在校准器中提供 predict_proba。"""
    clf.fit(X_train, y_train)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_test)[:, 1]
    else:
        proba = clf.decision_function(X_test)
    return clf, proba


def tune_threshold(y_true, y_prob, beta: float = 1.0) -> Tuple[float, float]:
    """在概率上扫描阈值，最大化 F-beta（默认 F1）。"""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (y_prob >= t).astype(int)
        f = (1 + beta ** 2) * precision_score(y_true, pred, zero_division=0) * \
            recall_score(y_true, pred, zero_division=0) / \
            (beta ** 2 * precision_score(y_true, pred, zero_division=0)
             + recall_score(y_true, pred, zero_division=0) + 1e-12)
        if f > best_f:
            best_f, best_t = f, float(round(t, 3))
    return best_t, float(best_f)


def tune_two_thresholds(y_true, y_prob):
    """搜索 (low, high) 双阈值，使中风险区间均衡兼顾 F1 与复核比例。

    目标：high 之上判高风险（precision 优先），low 之下判低风险（recall 优先），
    中间为"待人工复核"。返回 (t_low, t_high)。
    """
    best = None
    best_score = -1.0
    for t_high in np.arange(0.6, 0.96, 0.02):
        for t_low in np.arange(0.1, min(0.5, t_high - 0.2), 0.02):
            high = (y_prob >= t_high).astype(int)
            low = (y_prob < t_low).astype(int)
            mid = ~(high.astype(bool) | low.astype(bool))
            # 高风险区 precision、低风险区 recall、复核比例适中
            prec_h = precision_score(y_true, high, zero_division=0)
            # 低风险判为正常 -> 对正类的 recall 应尽量不漏（即漏检低）
            missed = ((y_prob < t_low) & (y_true == 1)).sum()
            rec_safe = 1.0 - missed / max(1, (y_true == 1).sum())
            review_ratio = mid.mean()
            score = 0.5 * prec_h + 0.4 * rec_safe - 0.4 * abs(review_ratio - 0.15)
            if score > best_score:
                best_score = score
                best = (float(round(t_low, 3)), float(round(t_high, 3)))
    return best


def train_binary(X_train, y_train, X_val, y_val, X_test, y_test,
                 clf, threshold_beta: float = 1.0) -> Dict[str, Any]:
    """训练二分类器并调阈值，返回结果字典。"""
    clf, val_prob = fit_predict_proba(clf, X_train, y_train, X_val)
    best_t, best_f = tune_threshold(y_val, val_prob, beta=threshold_beta)

    # 用测试集概率计算最终指标
    _, test_prob = fit_predict_proba(clone(clf), X_train, y_train, X_test)
    test_pred = (test_prob >= best_t).astype(int)
    metrics = _binary_metrics(y_test, test_pred, test_prob)
    metrics["threshold"] = best_t
    metrics["val_best_f1"] = best_f
    return {
        "clf": clf,
        "threshold": best_t,
        "metrics": metrics,
        "test_prob": test_prob,
        "test_pred": test_pred,
    }


def train_multiclass(X_train, y_train, X_test, y_test, classes) -> Dict[str, Any]:
    """训练多分类（含"无风险"共 10 类），用于欺诈类型解释。"""
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                             random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    pred = clf.predict(X_test)
    from sklearn.metrics import f1_score as fs, classification_report
    macro_f1 = float(fs(y_test, pred, average="macro", zero_division=0))
    per_class = classification_report(
        y_test, pred, labels=classes, zero_division=0, output_dict=True)
    return {
        "clf": clf,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "proba": proba,
        "pred": pred,
    }


def save_artifact(obj, path) -> None:
    joblib.dump(obj, path)


def load_artifact(path):
    return joblib.load(path)


def save_metrics_json(metrics: Dict[str, Any], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def load_metrics_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
