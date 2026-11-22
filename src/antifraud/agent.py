"""Agent 工作流：输入校验 -> 预处理 -> 规则 -> 模型 -> 融合 -> 决策 -> 解释。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .config import (
    ALL_CATEGORIES,
    CATEGORY_RISK_DESC,
    DEFAULT_FUSION_ALPHA,
    DEFAULT_THRESH_HIGH,
    DEFAULT_THRESH_LOW,
    MAX_TEXT_LEN,
    NO_RISK_LABEL,
    RULE_OVERRIDE_THRESHOLD,
)
from .preprocessing import (
    clean_text,
    display_mask,
    extract_entities,
    preprocess_for_feature,
)
from .rules import run_rules


class ModelNotFoundError(RuntimeError):
    pass


@dataclass
class AnalysisResult:
    """一条短信的完整分析结果。"""
    ok: bool
    error: str
    text_masked: str
    text_hash: str
    entities: Dict[str, List[str]]
    rule_score: float
    rule_categories: List[str]
    rule_reasons: List[str]
    url_risk: Dict[str, Any]
    obfuscation: Dict[str, Any]
    strong_hit: bool
    ml_prob: float
    ml_type: str
    ml_type_probs: Dict[str, float]
    final_score: float
    risk_level: str
    confidence: float
    need_review: bool
    reasons: List[str]
    top_risk_features: List[str]
    advice: List[str]

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["text_hash"] = self.text_hash
        return d


class AntifraudAgent:
    """校园反诈 Agent。"""

    def __init__(
        self,
        feature_union,
        binary_clf,
        multiclass_clf=None,
        threshold_high: float = DEFAULT_THRESH_HIGH,
        threshold_low: float = DEFAULT_THRESH_LOW,
        fusion_alpha: float = DEFAULT_FUSION_ALPHA,
        binary_threshold: Optional[float] = None,
        feature_names: Optional[np.ndarray] = None,
    ):
        self.feature_union = feature_union
        self.binary_clf = binary_clf
        self.multiclass_clf = multiclass_clf
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.fusion_alpha = fusion_alpha
        self.binary_threshold = binary_threshold or 0.5
        self.feature_names = feature_names

    # ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, model_dir, **overrides):
        from .models import load_artifact
        import pathlib
        model_dir = pathlib.Path(model_dir)
        cfg_path = model_dir / "config.json"
        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        try:
            feature_union = load_artifact(model_dir / "feature_union.joblib")
            binary_clf = load_artifact(model_dir / "binary_clf.joblib")
        except FileNotFoundError as e:
            raise ModelNotFoundError(f"模型文件缺失: {e}") from e

        multiclass_clf = None
        mc_path = model_dir / "multiclass_clf.joblib"
        if mc_path.exists():
            multiclass_clf = load_artifact(mc_path)

        fn_path = model_dir / "feature_names.npy"
        feature_names = np.load(fn_path, allow_pickle=True) if fn_path.exists() else None

        return cls(
            feature_union=feature_union,
            binary_clf=binary_clf,
            multiclass_clf=multiclass_clf,
            threshold_high=overrides.get("threshold_high", cfg.get(
                "threshold_high", DEFAULT_THRESH_HIGH)),
            threshold_low=overrides.get("threshold_low", cfg.get(
                "threshold_low", DEFAULT_THRESH_LOW)),
            fusion_alpha=overrides.get("fusion_alpha", cfg.get(
                "fusion_alpha", DEFAULT_FUSION_ALPHA)),
            binary_threshold=cfg.get("binary_threshold", 0.5),
            feature_names=feature_names,
        )

    # ------------------------------------------------------------------
    def _ml_prob(self, cleaned: str) -> float:
        X = self.feature_union.transform([cleaned])
        proba = self.binary_clf.predict_proba(X)[0]
        # 索引 1 为"风险"类
        return float(proba[1])

    def _ml_type(self, cleaned: str) -> Dict[str, Any]:
        if self.multiclass_clf is None:
            return {"pred": NO_RISK_LABEL, "probs": {}}
        X = self.feature_union.transform([cleaned])
        classes = list(self.multiclass_clf.classes_)
        if hasattr(self.multiclass_clf, "predict_proba"):
            proba = self.multiclass_clf.predict_proba(X)[0]
        else:
            dec = self.multiclass_clf.decision_function(X)[0]
            e = np.exp(dec - dec.max())
            proba = e / e.sum()
        probs = {c: float(p) for c, p in zip(classes, proba)}
        pred = classes[int(np.argmax(proba))]
        return {"pred": pred, "probs": probs}

    def _coef(self):
        """获取二分类线性系数（兼容 CalibratedClassifierCV）。"""
        clf = self.binary_clf
        if hasattr(clf, "coef_") and clf.coef_ is not None:
            return np.asarray(clf.coef_)[0]
        # CalibratedClassifierCV 内部线性核估计器
        for est in getattr(clf, "calibrated_classifiers_", []):
            inner = getattr(est, "estimator", None)
            if inner is not None and hasattr(inner, "coef_"):
                return np.asarray(inner.coef_)[0]
        return None

    def _top_features(self, cleaned: str, k: int = 6) -> List[str]:
        """返回对当前样本风险判断贡献最大的特征词（线性模型）。"""
        if self.feature_names is None:
            return []
        coef = self._coef()
        if coef is None:
            return []
        X = self.feature_union.transform([cleaned])
        cx = X.tocoo()
        contributions = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            contributions.append((j, v * coef[j]))
        contributions.sort(key=lambda x: -x[1])
        top = [self.feature_names[j] for j, _ in contributions[:k]]
        # 去掉前缀 "word__"/"char__"
        return [t.split("__", 1)[-1] for t in top]

    # ------------------------------------------------------------------
    def analyze(self, text: str, sender: Optional[str] = None) -> AnalysisResult:
        if text is None or not str(text).strip():
            return AnalysisResult(
                ok=False, error="输入为空，请粘贴短信内容", text_masked="",
                text_hash="", entities={}, rule_score=0.0, rule_categories=[],
                rule_reasons=[], url_risk={}, obfuscation={}, strong_hit=False,
                ml_prob=0.0, ml_type=NO_RISK_LABEL, ml_type_probs={},
                final_score=0.0, risk_level="", confidence=0.0,
                need_review=False, reasons=[], top_risk_features=[], advice=[],
            )

        original = str(text).strip()
        if len(original) > MAX_TEXT_LEN:
            original = original[:MAX_TEXT_LEN]

        cleaned = clean_text(original)
        if not cleaned:
            return AnalysisResult(
                ok=False, error="内容清洗后为空，请检查是否仅为特殊字符",
                text_masked=display_mask(original), text_hash=self._hash(original),
                entities={}, rule_score=0.0, rule_categories=[],
                rule_reasons=[], url_risk={}, obfuscation={}, strong_hit=False,
                ml_prob=0.0, ml_type=NO_RISK_LABEL, ml_type_probs={},
                final_score=0.0, risk_level="", confidence=0.0,
                need_review=False, reasons=[], top_risk_features=[], advice=[],
            )

        entities = extract_entities(original)
        rule = run_rules(original, cleaned, entities.urls)

        feat_text = preprocess_for_feature(original)
        ml_prob = self._ml_prob(feat_text)
        ml_type = self._ml_type(feat_text)

        # 融合
        final = self.fusion_alpha * ml_prob + (1 - self.fusion_alpha) * rule.score
        if rule.strong_hit:
            final = max(final, RULE_OVERRIDE_THRESHOLD)

        risk_level, confidence, need_review = self._decide(final, ml_prob,
                                                           rule.score)
        reasons = list(rule.reasons)
        reasons.append(f"模型风险概率 {ml_prob:.2f}，规则分 {rule.score:.2f}")
        if ml_type["pred"] != NO_RISK_LABEL:
            reasons.append("疑似类型：" + ml_type["pred"] + " —— "
                           + CATEGORY_RISK_DESC.get(ml_type["pred"], ""))
        top = self._top_features(feat_text)
        if top:
            reasons.append("关键特征词：" + "、".join(top))

        advice = self._advise(risk_level, rule, entities, ml_prob)

        return AnalysisResult(
            ok=True, error="", text_masked=display_mask(original),
            text_hash=self._hash(original),
            entities=entities.as_dict(),
            rule_score=round(rule.score, 4),
            rule_categories=rule.categories,
            rule_reasons=rule.reasons,
            url_risk=rule.url_risk,
            obfuscation=rule.obfuscation,
            strong_hit=rule.strong_hit,
            ml_prob=round(ml_prob, 4),
            ml_type=ml_type["pred"],
            ml_type_probs={k: round(v, 4) for k, v in ml_type["probs"].items()},
            final_score=round(final, 4),
            risk_level=risk_level,
            confidence=round(confidence, 4),
            need_review=need_review,
            reasons=reasons,
            top_risk_features=top,
            advice=advice,
        )

    # ------------------------------------------------------------------
    def _decide(self, final, ml_prob, rule_score):
        if final >= self.threshold_high:
            return "高风险", min(1.0, final), False
        if final < self.threshold_low:
            conf = max(0.0, 1.0 - final)
            return "低风险", conf, False
        # 中风险：转人工复核
        conf = 1.0 - abs(final - (self.threshold_low + self.threshold_high) / 2)
        # 模型与规则分歧明显时，也提示复核
        disagree = abs(ml_prob - rule_score) > 0.5
        return "中风险", max(0.0, conf), True or disagree

    def _advise(self, risk_level, rule, entities, ml_prob) -> List[str]:
        adv: List[str] = []
        if risk_level == "高风险":
            adv.append("不要点击任何链接，不要向陌生账户转账")
            adv.append("不要透露验证码、银行卡号、密码或进行屏幕共享")
            if entities.urls:
                adv.append("短信中的链接疑似钓鱼，切勿访问")
            if rule.strong_hit:
                adv.append("命中高危诈骗话术，请立即停止互动")
            adv.append("通过官方 App/客服电话核实，必要时拨打 96110 或联系保卫处")
        elif risk_level == "中风险":
            adv.append("建议谨慎对待，先通过官方渠道核实真伪")
            if entities.urls:
                adv.append("不要点击链接，可手动输入官网地址核实")
            adv.append("已标记为待人工复核，请勿轻信")
        else:
            adv.append("初步判断为低风险，可正常处理")
            if entities.urls or entities.phones:
                adv.append("仍含链接/号码，请保持警惕，勿随意点击或回拨")
        return adv

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
