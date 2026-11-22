"""Agent 工作流单元测试（部分依赖训练产物，缺失则跳过）。"""
import pytest

from src.antifraud.agent import AntifraudAgent, ModelNotFoundError
from src.antifraud.config import MODELS_DIR


def test_empty_input():
    agent = _agent_or_none()
    if agent is None:
        pytest.skip("模型未训练")
    res = agent.analyze("   ")
    assert res.ok is False


def test_scam_high_risk():
    agent = _agent_or_none()
    if agent is None:
        pytest.skip("模型未训练")
    res = agent.analyze("我是公安局民警，你银行卡涉嫌洗钱，请转账到安全账户验证，否则冻结。")
    assert res.ok
    assert res.final_score > 0.5


def test_normal_low_or_review():
    agent = _agent_or_none()
    if agent is None:
        pytest.skip("模型未训练")
    res = agent.analyze("【图书馆】您预约的座位已到时间，请及时签到。")
    assert res.ok
    assert res.risk_level in ("低风险", "中风险")


def test_decide_thresholds():
    agent = AntifraudAgent(
        feature_union=None, binary_clf=None,
        threshold_high=0.8, threshold_low=0.3, fusion_alpha=0.8)
    assert agent._decide(0.9, 0.9, 0.9)[0] == "高风险"
    assert agent._decide(0.5, 0.5, 0.5)[0] == "中风险"
    assert agent._decide(0.1, 0.1, 0.1)[0] == "低风险"


def test_model_not_found():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ModelNotFoundError):
            AntifraudAgent.from_dir(Path(d))


def _agent_or_none():
    try:
        return AntifraudAgent.from_dir(MODELS_DIR)
    except ModelNotFoundError:
        return None
