"""训练二分类与多分类模型、调优阈值与融合权重，保存产物。

用法： python scripts/train.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antifraud.config import (
    PROCESSED_DIR, MODELS_DIR, ensure_dirs,
    COL_TEXT, COL_LABEL, COL_CATEGORY, COL_CASE_ID,
    NO_RISK_LABEL, FRAUD_CATEGORIES, ALL_CATEGORIES, RANDOM_STATE,
)
from src.antifraud.features import build_features, preprocess_series
from src.antifraud.models import (
    get_classifiers, fit_predict_proba, tune_threshold, tune_two_thresholds,
    train_binary, save_artifact, save_metrics_json, _binary_metrics,
)
from src.antifraud.preprocessing import clean_text, extract_entities, preprocess_for_feature
from src.antifraud.rules import run_rules
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / f"{name}.csv", encoding="utf-8-sig",
                     dtype={COL_CASE_ID: str})
    return df


def batch_rule_scores(texts) -> np.ndarray:
    """对文本批量计算规则分（复用清洗与实体抽取）。"""
    scores = np.zeros(len(texts))
    for i, t in enumerate(texts):
        cleaned = clean_text(t)
        if not cleaned:
            continue
        ents = extract_entities(t)
        scores[i] = run_rules(t, cleaned, ents.urls).score
    return scores


def main() -> None:
    ensure_dirs()
    t0 = time.time()
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")
    print(f"训练 {len(train_df)} / 验证 {len(val_df)} / 测试 {len(test_df)}")

    y_train = train_df[COL_LABEL].to_numpy()
    y_val = val_df[COL_LABEL].to_numpy()
    y_test = test_df[COL_LABEL].to_numpy()

    # ------------------------------------------------------------------
    # 特征
    # ------------------------------------------------------------------
    print("构建特征（jieba 分词 + 词/字 TF-IDF）...")
    features = build_features()
    X_train = features.fit_transform(preprocess_series(train_df[COL_TEXT]))
    X_val = features.transform(preprocess_series(val_df[COL_TEXT]))
    X_test = features.transform(preprocess_series(test_df[COL_TEXT]))
    print(f"  特征维度 {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # 模型对比（验证集）
    # ------------------------------------------------------------------
    print("模型对比 ...")
    comparison = {}
    for name, clf in get_classifiers().items():
        t = time.time()
        try:
            clf, val_prob = fit_predict_proba(clf, X_train, y_train, X_val)
            m = _binary_metrics(y_val, (val_prob >= 0.5).astype(int), val_prob)
            comparison[name] = m
            print(f"  {name}: F1={m['f1']:.4f} PR-AUC={m['pr_auc']:.4f} "
                  f"AUC={m['roc_auc']:.4f} ({time.time()-t:.1f}s)")
        except Exception as e:
            print(f"  {name}: 失败 {e}")

    best_name = max(comparison, key=lambda k: comparison[k]["f1"])
    best_f1 = comparison[best_name]["f1"]
    # 与最优 F1 差距极小(<=0.002)时，优先选择可解释、概率校准的逻辑回归
    if "LogisticRegression" in comparison and \
            comparison["LogisticRegression"]["f1"] >= best_f1 - 0.002:
        best_name = "LogisticRegression"
    print(f"最优模型：{best_name}")

    # ------------------------------------------------------------------
    # 用最优模型拟合（在验证集上得到概率）
    # ------------------------------------------------------------------
    best_clf_factory = get_classifiers()[best_name]
    from sklearn.base import clone
    best_clf = clone(best_clf_factory)
    best_clf, val_prob = fit_predict_proba(best_clf, X_train, y_train, X_val)

    # 验证集规则分
    print("计算验证集规则分 ...")
    val_rule = batch_rule_scores(val_df[COL_TEXT])

    # ------------------------------------------------------------------
    # 融合权重：采用固定设计值 alpha=0.8（保留规则信号用于可解释与分布外鲁棒性）
    # 消融实验(scripts/evaluate.py)会量化 alpha 在域内/域外的敏感性。
    # ------------------------------------------------------------------
    from src.antifraud.config import DEFAULT_FUSION_ALPHA
    best_alpha = DEFAULT_FUSION_ALPHA
    fused_val = best_alpha * val_prob + (1 - best_alpha) * val_rule
    fused_val = np.maximum(fused_val, np.where(
        [run_rules(t, clean_text(t), extract_entities(t).urls).strong_hit
         for t in val_df[COL_TEXT]], 0.92, 0.0))
    best_t, best_f = tune_threshold(y_val, fused_val, beta=1.0)
    print(f"  alpha={best_alpha}，融合后最优阈值={best_t}（验证 F1={best_f:.4f}）")

    # ------------------------------------------------------------------
    # 风险等级双阈值调优
    # ------------------------------------------------------------------
    low_t, high_t = tune_two_thresholds(y_val, fused_val)
    print(f"  风险等级阈值：low={low_t}, high={high_t}")

    # ------------------------------------------------------------------
    # 测试集最终评估
    # ------------------------------------------------------------------
    print("测试集最终评估 ...")
    _, test_prob = fit_predict_proba(clone(best_clf), X_train, y_train, X_test)
    test_rule = batch_rule_scores(test_df[COL_TEXT])
    fused_test = best_alpha * test_prob + (1 - best_alpha) * test_rule
    # 强规则托底
    strong = np.array([run_rules(t, clean_text(t), extract_entities(t).urls).strong_hit
                       for t in test_df[COL_TEXT]])
    fused_test = np.maximum(fused_test, np.where(strong, 0.92, 0.0))
    test_pred = (fused_test >= best_t).astype(int)
    final_metrics = _binary_metrics(y_test, test_pred, fused_test)
    final_metrics.update({
        "threshold": best_t, "fusion_alpha": best_alpha,
        "threshold_low": low_t, "threshold_high": high_t,
        "model": best_name,
    })
    print(f"  测试 F1={final_metrics['f1']:.4f} "
          f"P={final_metrics['precision']:.4f} R={final_metrics['recall']:.4f} "
          f"PR-AUC={final_metrics['pr_auc']:.4f}")

    # ------------------------------------------------------------------
    # 多分类（10 类，LinearSVC OvR）
    # ------------------------------------------------------------------
    print("训练多分类模型（欺诈类型）...")
    y_train_cat = train_df[COL_CATEGORY].to_numpy()
    y_test_cat = test_df[COL_CATEGORY].to_numpy()
    mc = LinearSVC(C=1.0, class_weight="balanced", multi_class="ovr",
                   max_iter=3000, random_state=RANDOM_STATE)
    mc.fit(X_train, y_train_cat)
    mc_pred = mc.predict(X_test)
    macro_f1 = f1_score(y_test_cat, mc_pred, average="macro", zero_division=0)
    rep = classification_report(y_test_cat, mc_pred, labels=ALL_CATEGORIES,
                                zero_division=0, output_dict=True)
    print(f"  多分类 macro-F1={macro_f1:.4f}")

    # ------------------------------------------------------------------
    # 保存产物
    # ------------------------------------------------------------------
    print("保存模型与产物 ...")
    save_artifact(features, MODELS_DIR / "feature_union.joblib")
    save_artifact(best_clf, MODELS_DIR / "binary_clf.joblib")
    save_artifact(mc, MODELS_DIR / "multiclass_clf.joblib")
    np.save(MODELS_DIR / "feature_names.npy", features.get_feature_names_out())

    config = {
        "threshold": best_t,
        "threshold_high": high_t,
        "threshold_low": low_t,
        "fusion_alpha": best_alpha,
        "binary_threshold": best_t,
        "model": best_name,
        "categories": ALL_CATEGORIES,
        "no_risk_label": NO_RISK_LABEL,
        "random_state": RANDOM_STATE,
    }
    with open(MODELS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    metrics = {
        "final": final_metrics,
        "comparison_val": comparison,
        "best_model": best_name,
        "multiclass_macro_f1": float(macro_f1),
        "multiclass_report": rep,
    }
    save_metrics_json(metrics, MODELS_DIR / "metrics.json")

    # 保存测试集预测结果供评估脚本复用
    results = {
        "y_test": y_test.tolist(),
        "test_prob": np.asarray(test_prob).tolist(),
        "test_rule": test_rule.tolist(),
        "fused_test": fused_test.tolist(),
        "test_pred": test_pred.tolist(),
        "test_category": y_test_cat.tolist(),
        "mc_pred": mc_pred.tolist(),
        "test_text": test_df[COL_TEXT].tolist(),
    }
    save_metrics_json(results, MODELS_DIR / "test_predictions.json")

    print(f"全部完成，耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
