"""实验评估：模型对比图、混淆矩阵、阈值/ROC、消融、外部数据集、鲁棒性、误差分析。

用法： python scripts/evaluate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.antifraud.config import (
    PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR, ensure_dirs,
    COL_TEXT, COL_LABEL, COL_CATEGORY, NO_RISK_LABEL, ALL_CATEGORIES,
    RANDOM_STATE, DEFAULT_FUSION_ALPHA,
)
from src.antifraud.models import load_artifact, load_metrics_json, save_metrics_json, _binary_metrics
from src.antifraud.features import build_features, make_word_features, make_char_features, preprocess_series
from src.antifraud.preprocessing import clean_text, extract_entities, preprocess_for_feature
from src.antifraud.rules import run_rules
from src.antifraud import evaluation as ev


def load_agent():
    from src.antifraud.agent import AntifraudAgent
    return AntifraudAgent.from_dir(MODELS_DIR)


def batch_scores(agent, texts):
    """对文本批量计算融合分/模型分/规则分。"""
    feats = [preprocess_for_feature(t) for t in texts]
    X = agent.feature_union.transform(feats)
    prob = agent.binary_clf.predict_proba(X)[:, 1]
    rs = np.zeros(len(texts))
    strong = np.zeros(len(texts), dtype=bool)
    for i, t in enumerate(texts):
        c = clean_text(t)
        e = extract_entities(t)
        r = run_rules(t, c, e.urls)
        rs[i] = r.score
        strong[i] = r.strong_hit
    fused = agent.fusion_alpha * prob + (1 - agent.fusion_alpha) * rs
    fused = np.maximum(fused, np.where(strong, 0.92, 0.0))
    return fused, prob, rs


def main() -> None:
    ensure_dirs()
    agent = load_agent()
    alpha = agent.fusion_alpha
    thr = agent.binary_threshold
    metrics = load_metrics_json(MODELS_DIR / "metrics.json")
    preds = load_metrics_json(MODELS_DIR / "test_predictions.json")

    y_test = np.array(preds["y_test"])
    test_prob = np.array(preds["test_prob"])
    test_rule = np.array(preds["test_rule"])
    fused_test = np.array(preds["fused_test"])
    test_pred = np.array(preds["test_pred"])
    test_cat = np.array(preds["test_category"])
    mc_pred = np.array(preds["mc_pred"])
    test_text = preds["test_text"]

    print("== 测试集二元分类 ==")
    final = _binary_metrics(y_test, test_pred, fused_test)
    cm = confusion_matrix(y_test, test_pred)
    print(f"  acc={final['accuracy']:.4f} P={final['precision']:.4f} "
          f"R={final['recall']:.4f} F1={final['f1']:.4f} "
          f"AUC={final['roc_auc']:.4f} PR-AUC={final['pr_auc']:.4f}")
    print(f"  混淆矩阵 [[TN,FP],[FN,TP]] = {cm.tolist()}")

    # ---- 图：模型对比 / 混淆 / 阈值 / ROC ----
    ev.plot_model_comparison(metrics["comparison_val"], "f1",
                             FIGURES_DIR / "model_comparison.png")
    ev.plot_confusion(y_test, test_pred, ["无风险", "风险"],
                      FIGURES_DIR / "confusion_matrix.png")
    ev.plot_threshold_curve(y_test, fused_test, FIGURES_DIR / "threshold_curve.png")
    ev.plot_roc_pr(y_test, fused_test, FIGURES_DIR / "roc_pr.png")

    # ---- 消融：特征（词/字/词+字） ----
    print("== 特征消融（验证集）==")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv", encoding="utf-8-sig")
    y_train = train_df[COL_LABEL].to_numpy()
    y_val = val_df[COL_LABEL].to_numpy()
    feat_ablation = {}
    for name, builder in [("词级", make_word_features), ("字级", make_char_features),
                          ("词+字", build_features)]:
        fe = builder()
        Xtr = fe.fit_transform(preprocess_series(train_df[COL_TEXT]))
        Xv = fe.transform(preprocess_series(val_df[COL_TEXT]))
        clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_STATE)
        clf.fit(Xtr, y_train)
        p = clf.predict_proba(Xv)[:, 1]
        m = _binary_metrics(y_val, (p >= 0.5).astype(int), p)
        feat_ablation[name] = m["f1"]
        print(f"  {name}: F1={m['f1']:.4f} (P={m['precision']:.4f} R={m['recall']:.4f})")
    ev.plot_ablation(feat_ablation, "验证集 F1", FIGURES_DIR / "ablation_feature.png")

    # ---- 消融：融合（仅规则 / 仅模型 / 融合） + alpha 敏感性 ----
    print("== 融合消融（测试集）==")
    val_rule = batch_scores(agent, val_df[COL_TEXT])[1] if False else None
    # 重新计算验证集规则分用于规则阈值调优
    _, _, val_rs = batch_scores(agent, val_df[COL_TEXT])
    from src.antifraud.models import tune_threshold
    rule_t, _ = tune_threshold(y_val, val_rs, beta=1.0)

    fusion_ablation = {}
    # 仅规则
    rp = (test_rule >= rule_t).astype(int)
    fusion_ablation["仅规则"] = f1_score(y_test, rp, zero_division=0)
    # 仅模型（alpha=1.0）
    mp = (test_prob >= thr).astype(int)
    fusion_ablation["仅模型"] = f1_score(y_test, mp, zero_division=0)
    # 融合（alpha=0.8）
    fusion_ablation["融合(α=0.8)"] = f1_score(y_test, test_pred, zero_division=0)
    for k, v in fusion_ablation.items():
        print(f"  {k}: F1={v:.4f}")
    ev.plot_ablation(fusion_ablation, "测试集 F1", FIGURES_DIR / "ablation_fusion.png")

    # alpha 敏感性
    alpha_vals = {}
    for a in [0.0, 0.5, 0.7, 0.8, 0.9, 1.0]:
        f = a * test_prob + (1 - a) * test_rule
        pred = (f >= 0.5).astype(int)
        alpha_vals[f"α={a}"] = f1_score(y_test, pred, zero_division=0)
    pd.DataFrame([alpha_vals]).to_csv(TABLES_DIR / "alpha_sensitivity.csv",
                                      index=False, encoding="utf-8-sig")

    # ---- 多分类 ----
    print("== 多分类 ==")
    print(f"  macro-F1={metrics['multiclass_macro_f1']:.4f}")
    mc_df = ev.multiclass_report_table(test_cat, mc_pred, ALL_CATEGORIES)
    mc_df.to_csv(TABLES_DIR / "multiclass_report.csv", index=False,
                 encoding="utf-8-sig")
    # 各类别 F1 条形图
    cat_f1 = dict(zip(mc_df["类别"], mc_df["f1"]))
    ev.plot_category_recall(cat_f1, FIGURES_DIR / "multiclass_f1.png")

    # ---- 外部数据集 ----
    external = {}
    # mudou 外部留出（真实中文短信，未参与训练）
    print("== 外部：mudou 真实短信（留出集）==")
    mudou = pd.read_csv(PROCESSED_DIR / "mudou_holdout.csv", encoding="utf-8-sig")
    m_fused, m_prob, m_rule = batch_scores(agent, mudou[COL_TEXT].tolist())
    m_pred = (m_fused >= thr).astype(int)
    m_y = mudou[COL_LABEL].to_numpy()
    mm = _binary_metrics(m_y, m_pred, m_fused)
    print(f"  acc={mm['accuracy']:.4f} P={mm['precision']:.4f} R={mm['recall']:.4f} "
          f"F1={mm['f1']:.4f} AUC={mm['roc_auc']:.4f}")
    external["mudou"] = mm

    # FBS（真实伪基站短信，全为正类 -> 只看召回）
    print("== 外部：FBS 伪基站短信召回 ==")
    fbs = pd.read_csv(PROCESSED_DIR / "fbs.csv", encoding="utf-8-sig")
    f_fused, _, _ = batch_scores(agent, fbs[COL_TEXT].tolist())
    fbs["pred"] = (f_fused >= thr).astype(int)
    fbs_recall = {}
    for cat in ["广告营销", "诈骗类", "违法服务", "其他"]:
        sub = fbs[fbs[COL_CATEGORY] == cat]
        fbs_recall[cat] = float(sub["pred"].mean()) if len(sub) else 0.0
        print(f"  {cat}: recall={fbs_recall[cat]:.3f} (n={len(sub)})")
    external["fbs_recall"] = fbs_recall
    ev.plot_category_recall(fbs_recall, FIGURES_DIR / "fbs_recall.png")

    # 校园场景测试集
    print("== 外部：校园场景测试集 ==")
    campus = pd.read_csv(PROCESSED_DIR / "campus_test.csv", encoding="utf-8-sig")
    c_fused, _, _ = batch_scores(agent, campus[COL_TEXT].tolist())
    c_pred = (c_fused >= thr).astype(int)
    c_y = campus[COL_LABEL].to_numpy()
    cm_ = _binary_metrics(c_y, c_pred, c_fused)
    print(f"  acc={cm_['accuracy']:.4f} P={cm_['precision']:.4f} R={cm_['recall']:.4f} "
          f"F1={cm_['f1']:.4f}")
    external["campus"] = cm_
    # 校园样本明细
    campus_out = campus.copy()
    campus_out["pred"] = c_pred
    campus_out["score"] = np.round(c_fused, 3)
    campus_out.to_csv(TABLES_DIR / "campus_detail.csv", index=False,
                      encoding="utf-8-sig")

    # 领域差距消融：仅生成数据 vs 加入真实短信数据
    print("== 领域差距消融：仅生成数据 vs 混合数据 ==")
    fgrc_only = pd.read_csv(PROCESSED_DIR / "fgrc_only.csv", encoding="utf-8-sig")
    # 用 FGRC 仅数据训练一个对照模型（与最终模型同特征同算法）
    fe_g = build_features()
    Xg = fe_g.fit_transform(preprocess_series(fgrc_only[COL_TEXT]))
    clf_g = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                               random_state=RANDOM_STATE)
    clf_g.fit(Xg, fgrc_only[COL_LABEL].to_numpy())
    domain_gap = {}
    for dname, dset in [("mudou_holdout", mudou), ("campus", campus)]:
        Xd = fe_g.transform(preprocess_series(dset[COL_TEXT]))
        pg = clf_g.predict_proba(Xd)[:, 1]
        pred_g = (pg >= 0.5).astype(int)
        yd = dset[COL_LABEL].to_numpy()
        m = _binary_metrics(yd, pred_g, pg)
        domain_gap[dname] = m
        print(f"  FGRC-only 在 {dname}: P={m['precision']:.3f} "
              f"R={m['recall']:.3f} F1={m['f1']:.3f} AUC={m['roc_auc']:.3f}")
    print(f"  混合模型 在 mudou_holdout: P={mm['precision']:.3f} "
          f"R={mm['recall']:.3f} F1={mm['f1']:.3f}")
    print(f"  混合模型 在 campus: P={cm_['precision']:.3f} "
          f"R={cm_['recall']:.3f} F1={cm_['f1']:.3f}")
    external["domain_gap_fgrc_only"] = domain_gap

    # 混淆/鲁棒性
    print("== 鲁棒性：混淆短信 ==")
    obf = pd.read_csv(PROCESSED_DIR / "obfuscation_test.csv", encoding="utf-8-sig")
    o_fused, _, _ = batch_scores(agent, obf[COL_TEXT].tolist())
    obf["pred"] = (o_fused >= thr).astype(int)
    robust = {}
    for cat in ["原始", "空格打散", "全角数字", "零宽字符", "符号穿插"]:
        sub = obf[obf[COL_CATEGORY] == cat]
        robust[cat] = float(sub["pred"].mean())
        print(f"  {cat}: recall={robust[cat]:.3f}")
    external["obfuscation_recall"] = robust
    ev.plot_category_recall(robust, FIGURES_DIR / "robustness.png")

    # ---- 误差分析 ----
    print("== 误差分析 ==")
    fn_mask = (test_pred == 0) & (y_test == 1)
    fp_mask = (test_pred == 1) & (y_test == 0)
    err_rows = []
    for idx in np.where(fn_mask)[0][:30]:
        err_rows.append({"type": "漏报(假阴性)", "label": "风险", "pred": "无风险",
                         "score": float(fused_test[idx]),
                         "text": test_text[idx][:120]})
    for idx in np.where(fp_mask)[0][:30]:
        err_rows.append({"type": "误报(假阳性)", "label": "无风险", "pred": "风险",
                         "score": float(fused_test[idx]),
                         "text": test_text[idx][:120]})
    err_df = pd.DataFrame(err_rows)
    err_df.to_csv(TABLES_DIR / "error_cases.csv", index=False, encoding="utf-8-sig")
    print(f"  漏报 {int(fn_mask.sum())} 条，误报 {int(fp_mask.sum())} 条，"
          f"已导出前 30 条到 tables/error_cases.csv")

    # ---- 汇总 ----
    summary = {
        "test_binary": final,
        "confusion_matrix": cm.tolist(),
        "best_model": metrics["best_model"],
        "comparison_val": metrics["comparison_val"],
        "multiclass_macro_f1": metrics["multiclass_macro_f1"],
        "feature_ablation": feat_ablation,
        "fusion_ablation": fusion_ablation,
        "alpha_sensitivity": alpha_vals,
        "external": external,
        "threshold": thr,
        "fusion_alpha": alpha,
        "test_false_negative": int(fn_mask.sum()),
        "test_false_positive": int(fp_mask.sum()),
    }
    save_metrics_json(summary, TABLES_DIR / "summary.json")
    print("全部完成，汇总写入 tables/summary.json")


if __name__ == "__main__":
    main()
